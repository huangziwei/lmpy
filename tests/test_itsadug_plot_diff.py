"""
Numerical-equivalence tests for ``hea.gam.get_difference`` against R's
``itsadug::get_difference``.

For each case under ``tests/fixtures/itsadug_plot_diff/<case_id>/``, we re-fit
the same model in hea, replay the same ``(comp, cond, f, sim_ci, rm_ranef)``
arguments, and compare the per-row ``difference`` and ``CI`` against R's
output. For the ``sim_ci=True`` case we also check the deterministic
``se_fit`` (= ``sqrt(rowSums((X1-X2) Vc (X1-X2)^T))``) to high precision and
the empirical ``crit`` to a loose tolerance — Python and R don't share an
RNG, so the quantile of the max-abs-standardized-deviation envelope only
matches to the Monte-Carlo SE.

Fixtures are baked once via ``Rscript tests/scripts/itsadug_plot_diff_fixtures.R``;
re-run that script if ``itsadug`` or the model design changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from hea import gam

ROOT = Path(__file__).parent / "fixtures" / "itsadug_plot_diff"
MODEL_DIR = ROOT / "_model"
CASE_DIRS = sorted(p for p in ROOT.iterdir() if p.is_dir() and not p.name.startswith("_"))
CASE_IDS = [p.name for p in CASE_DIRS]


def _load_data() -> pl.DataFrame:
    """Load the synthetic dataset and re-attach factor levels — CSV
    round-trip drops R's factor type, but hea is happy with either pl.Utf8
    or pl.Enum at materialize time. We use pl.Enum with the explicit level
    order R wrote (A,B,C / Y,Z) for parity with mgcv's contrasts.
    """
    df = pl.read_csv(MODEL_DIR / "data.csv", null_values="NA")
    df = df.with_columns([
        df["group"].cast(pl.Enum(["A", "B", "C"])),
        df["cohort"].cast(pl.Enum(["Y", "Z"])),
    ])
    return df


@pytest.fixture(scope="module")
def fitted_model():
    data = _load_data()
    m = gam("y ~ group + cohort + s(x, by=group)", data=data, method="REML")
    return m, data


def _parse_args(path: Path) -> dict:
    """args.json round-trip: itsadug's R script wraps each value as a list,
    so a length-1 vector lands as ``["A"]`` not ``"A"``. We don't unwrap —
    hea's get_difference accepts list values uniformly.
    """
    raw = json.loads(path.read_text())
    comp = {k: tuple(v) for k, v in raw["comp"].items()}
    cond = {}
    for k, v in raw["cond"].items():
        # numeric arrays come through as list-of-numbers; string fixers
        # (cohort="Y") come through as list-of-strings.
        if isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
            cond[k] = np.asarray(v, dtype=float)
        else:
            cond[k] = list(v) if isinstance(v, list) else [v]
    return {
        "case_id": raw["case_id"],
        "comp": comp,
        "cond": cond,
        "f": float(raw["f"]),
        "sim_ci": bool(raw["sim.ci"]),
        "rm_ranef": raw["rm.ranef"],
        "n_grid": int(raw["n_grid"]),
        "has_sim_ci_col": bool(raw["has_sim_ci_col"]),
    }


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_get_difference_matches_itsadug(fitted_model, case_id: str):
    m, _ = fitted_model
    case_dir = ROOT / case_id
    args = _parse_args(case_dir / "args.json")
    ref = pl.read_csv(case_dir / "diff_table.csv", null_values="NA")

    # rm.ranef arrives as a Python bool here — get_difference accepts it
    # as-is. None of our cases exercise the substring-grep mode (we don't
    # have ranef smooths in this v1 fixture set).
    rm_ranef = args["rm_ranef"]
    if not isinstance(rm_ranef, bool):
        rm_ranef = list(rm_ranef) if isinstance(rm_ranef, list) else rm_ranef

    res = m.get_difference(
        comp=args["comp"],
        cond=args["cond"],
        f=args["f"],
        sim_ci=args["sim_ci"],
        rm_ranef=rm_ranef,
        rng=20260430,  # deterministic for the sim.ci path; loose tol on crit
        n_sim=10_000,
    )

    assert res.difference.shape[0] == args["n_grid"], (
        f"{case_id}: got {res.difference.shape[0]} grid rows, "
        f"want {args['n_grid']}"
    )

    # difference is a basis-invariant linear functional of β̂, so the
    # only source of disagreement with mgcv is REML convergence drift
    # in the smoothing parameters. For this dataset, hea and mgcv agree
    # on sp[1]/sp[2] to ~1e-6 relative but on sp[3] (group=C, flat
    # signal) only to ~1% — the REML loglik is very flat there, and
    # both solvers stop in slightly different places. The resulting
    # difference noise lands at ~5e-5 absolute (5e-3 relative near
    # zero crossings) — orders of magnitude below the CI half-width
    # itself (~0.27 here), so still a tight oracle in any practical sense.
    np.testing.assert_allclose(
        res.difference,
        ref["difference"].to_numpy(),
        rtol=1e-3,
        atol=2e-4,
        err_msg=f"{case_id}: difference diverges from itsadug",
    )

    # CI is f * sqrt(diag(p Vp p^T)) — same convergence-drift bound.
    np.testing.assert_allclose(
        res.ci,
        ref["CI"].to_numpy(),
        rtol=1e-3,
        atol=2e-4,
        err_msg=f"{case_id}: CI diverges from itsadug",
    )

    if args["sim_ci"]:
        assert args["has_sim_ci_col"], "fixture mislabel: sim.ci=TRUE but no sim.CI column"
        assert res.sim_ci is not None and res.crit is not None

        # se_fit = sim_ci / crit — deterministic given Vc and p. Same
        # convergence-drift bound as ``CI``: tight in absolute terms,
        # bounded by REML sp agreement.
        ref_se_fit = pl.read_csv(case_dir / "se_fit.csv")["se_fit"].to_numpy()
        ours_se_fit = res.sim_ci / res.crit
        np.testing.assert_allclose(
            ours_se_fit,
            ref_se_fit,
            rtol=1e-3,
            atol=2e-4,
            err_msg=f"{case_id}: simultaneous se_fit diverges",
        )

        # crit is an empirical 0.95 quantile over n_sim=10000 MVN draws.
        # Cross-RNG comparison: the two implementations sample
        # independently so the quantile differs by Monte-Carlo SE. The
        # standard error of the 0.95 quantile of the MASD with n=10000
        # draws is roughly 0.5–2% of the value; allow 5% for safety.
        ref_crit = float(pl.read_csv(case_dir / "crit.csv")["crit"][0])
        np.testing.assert_allclose(
            res.crit, ref_crit, rtol=0.05,
            err_msg=f"{case_id}: simultaneous crit diverges (ours={res.crit}, R={ref_crit})",
        )


def test_cohort_y_matches_basic(fitted_model):
    """Sanity: with the model ``y ~ group + cohort + s(x, by=group)`` and
    no group:cohort interaction, the (group=A) − (group=B) difference is
    identical regardless of the cohort the comparison is held at — both
    p1 and p2 carry the same cohort column, so it cancels. This case
    exists to exercise the cond-string-coerce path; the numerics should
    coincide with case_basic to machine precision.
    """
    m, _ = fitted_model
    args_b = _parse_args(ROOT / "case_basic" / "args.json")
    args_y = _parse_args(ROOT / "case_cohortY" / "args.json")
    res_b = m.get_difference(
        comp=args_b["comp"], cond=args_b["cond"], f=args_b["f"],
        sim_ci=False, rm_ranef=True,
    )
    res_y = m.get_difference(
        comp=args_y["comp"], cond=args_y["cond"], f=args_y["f"],
        sim_ci=False, rm_ranef=True,
    )
    np.testing.assert_allclose(res_y.difference, res_b.difference,
                                rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(res_y.ci, res_b.ci,
                                rtol=1e-12, atol=1e-12)
