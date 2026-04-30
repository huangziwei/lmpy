"""
mgcv-oracle regression tests for hea.gam.

Each test pins the printed numerical outputs of `mgcv::gam(..., method=...)`
on a fixed dataset so the hea port can be validated against the canonical
R/mgcv results. Coverage spans:

  - tp / cr / ps basis types
  - REML and GCV.Cp criteria
  - parametric + smooth combinations
  - tensor-product (te) smooths
  - by=factor multi-block smooths

Tolerances are set per-quantity. ρ=log(sp) is 4-decimal for tp and ps;
edf, σ², and the criterion typically agree to 4–5 decimal places. Smooth
basis coefficients themselves are not pinned (they're identifiable only
up to mgcv's reparametrization), but per-smooth edf totals are.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from conftest import load_dataset
from hea import gam
from hea.family import Tweedie, tw


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _allclose(actual, expected, *, atol, name=""):
    np.testing.assert_allclose(actual, expected, atol=atol,
                               err_msg=f"{name}: {actual} vs {expected}")


def _assert_param(m, col, est, *, atol=5e-3):
    if col not in m.bhat.columns:
        raise KeyError(f"{col!r} not in {list(m.bhat.columns)!r}")
    np.testing.assert_allclose(m.bhat[col][0], est, atol=atol,
                               err_msg=f"param[{col}]")


# ---------------------------------------------------------------------------
# 1) MASS::mcycle — single tp smooth, REML
# ---------------------------------------------------------------------------


def test_mcycle_tp_REML():
    """gam(accel ~ s(times), data=mcycle, method="REML")"""
    d = load_dataset("MASS", "mcycle")
    m = gam("accel ~ s(times)", d, method="REML")

    assert m.n == 133
    _allclose(m.sp[0], 7.758035879e-04, atol=5e-5, name="sp")
    _allclose(m.edf_total, 9.624691, atol=5e-4, name="edf_total")
    _allclose(m.sigma_squared, 506.3529, atol=5e-3, name="sigma2")
    _allclose(m.REML_criterion / 2, 616.1420, atol=5e-3, name="REML/2")
    _allclose(m.r_squared_adjusted, 0.7831484, atol=5e-4, name="r2adj")
    # mgcv's logLik.gam profiles σ² out at the MLE rss/n (not the unbiased
    # rss/(n-edf) reported as $sig2); pin both to lock that convention down.
    _allclose(m.loglike, -597.8345, atol=5e-3, name="loglike")
    _assert_param(m, "(Intercept)", -25.54586, atol=5e-3)
    _allclose(m.edf_by_smooth["s(times)"], 8.624691, atol=5e-4, name="edf[s(times)]")


# ---------------------------------------------------------------------------
# 2) MASS::mcycle — single tp smooth, GCV.Cp
# ---------------------------------------------------------------------------


def test_mcycle_tp_GCV():
    """gam(accel ~ s(times), data=mcycle, method="GCV.Cp")"""
    d = load_dataset("MASS", "mcycle")
    m = gam("accel ~ s(times)", d, method="GCV.Cp")

    assert m.n == 133
    _allclose(m.sp[0], 6.195886e-04, atol=5e-5, name="sp")
    _allclose(m.edf_total, 9.693314, atol=5e-3, name="edf_total")
    _allclose(m.sigma_squared, 506.0017, atol=5e-3, name="sigma2")
    _allclose(m.GCV_score, 545.7792, atol=5e-3, name="GCV")
    _assert_param(m, "(Intercept)", -25.54586, atol=5e-3)


# ---------------------------------------------------------------------------
# 3) gamSim eg1 — four tp smooths, REML
# ---------------------------------------------------------------------------


def test_gamSim_eg1_four_smooths_REML():
    """gam(y ~ s(x0)+s(x1)+s(x2)+s(x3), data=gamSim(eg=1), method="REML")"""
    d = load_dataset("mgcv", "gamSim_eg1")
    m = gam("y ~ s(x0) + s(x1) + s(x2) + s(x3)", d, method="REML")

    assert m.n == 400
    _allclose(m.edf_total, 15.88548, atol=5e-3, name="edf_total")
    _allclose(m.sigma_squared, 3.897969, atol=5e-3, name="sigma2")
    _allclose(m.REML_criterion / 2, 861.1296, atol=5e-3, name="REML/2")
    _allclose(m.r_squared_adjusted, 0.7156242, atol=5e-3, name="r2adj")
    _assert_param(m, "(Intercept)", 7.833279, atol=5e-3)

    # Per-smooth edf — sp[3] is on a flat ridge so the s(x3) edf pins to ~1.0
    # (the linear fallthrough). The other three are well-determined.
    _allclose(m.edf_by_smooth["s(x0)"], 3.020970, atol=5e-3, name="edf[s(x0)]")
    _allclose(m.edf_by_smooth["s(x1)"], 2.843246, atol=5e-3, name="edf[s(x1)]")
    _allclose(m.edf_by_smooth["s(x2)"], 8.019844, atol=5e-3, name="edf[s(x2)]")
    _allclose(m.edf_by_smooth["s(x3)"], 1.001421, atol=5e-2, name="edf[s(x3)]")


# ---------------------------------------------------------------------------
# 4) gamSim eg1 — tensor-product te(x1,x2), REML
# ---------------------------------------------------------------------------


def test_gamSim_eg1_tensor_REML():
    """gam(y ~ s(x0) + te(x1, x2), data=gamSim(eg=1), method="REML")"""
    d = load_dataset("mgcv", "gamSim_eg1")
    m = gam("y ~ s(x0) + te(x1, x2)", d, method="REML")

    assert m.n == 400
    _allclose(m.edf_total, 17.55095, atol=5e-3, name="edf_total")
    _allclose(m.sigma_squared, 4.386049, atol=5e-3, name="sigma2")
    _allclose(m.REML_criterion / 2, 881.5002, atol=5e-3, name="REML/2")
    _allclose(m.r_squared_adjusted, 0.6800164, atol=5e-3, name="r2adj")
    _assert_param(m, "(Intercept)", 7.833279, atol=5e-3)

    _allclose(m.edf_by_smooth["s(x0)"], 3.097122, atol=5e-3, name="edf[s(x0)]")
    _allclose(m.edf_by_smooth["te(x1,x2)"], 13.45382, atol=5e-3, name="edf[te]")

    # te has 2 marginal penalties → 2 smoothing parameters
    assert m.sp.shape == (3,)  # 1 for s(x0) + 2 for te
    _allclose(m.sp[0], 1.492971, atol=5e-3, name="sp[s(x0)]")
    _allclose(m.sp[1], 33.05461, atol=1e-1, name="sp[te-1]")
    _allclose(m.sp[2], 0.0882241, atol=5e-3, name="sp[te-2]")


# ---------------------------------------------------------------------------
# 5) by=factor — synthetic data, REML
# ---------------------------------------------------------------------------


def test_byfactor_smooth_REML():
    """gam(y ~ g + s(x, by=g), data=<synth>, method="REML")

    by=factor produces one smooth block per factor level (3 here →
    sp has length 3, edf rolls up per block, identifiability handled
    via mgcv's id="" + parametric main-effect g).
    """
    d = load_dataset("synthetic", "seed_synth_gam_by_factor")
    # Re-cast g as Enum since the schema sidecar may not exist for this synth file.
    if d.schema["g"] != pl.Enum(["A", "B", "C"]):
        d = d.with_columns(pl.col("g").cast(pl.Enum(["A", "B", "C"])))
    m = gam("y ~ g + s(x, by=g)", d, method="REML")

    assert m.n == 300
    assert m.sp.shape == (3,)
    _allclose(m.edf_total, 21.36070, atol=5e-3, name="edf_total")
    _allclose(m.sigma_squared, 0.04265686, atol=5e-4, name="sigma2")
    _allclose(m.REML_criterion / 2, -9.890208, atol=5e-3, name="REML/2")
    _allclose(m.r_squared_adjusted, 0.9164980, atol=5e-3, name="r2adj")
    _assert_param(m, "(Intercept)",  0.02332958, atol=5e-3)
    _assert_param(m, "gB",          -0.06749164, atol=5e-3)
    _assert_param(m, "gC",           0.63793878, atol=5e-3)

    _allclose(m.edf_by_smooth["s(x):gA"], 6.953522, atol=5e-3, name="edf[s(x):gA]")
    _allclose(m.edf_by_smooth["s(x):gB"], 6.745235, atol=5e-3, name="edf[s(x):gB]")
    _allclose(m.edf_by_smooth["s(x):gC"], 4.661939, atol=5e-3, name="edf[s(x):gC]")


# ---------------------------------------------------------------------------
# 6) MASS::mcycle — P-spline (bs="ps") smooth, REML
# ---------------------------------------------------------------------------


def test_mcycle_ps_REML():
    """gam(accel ~ s(times, bs="ps"), data=mcycle, method="REML")"""
    d = load_dataset("MASS", "mcycle")
    m = gam("accel ~ s(times, bs='ps')", d, method="REML")

    assert m.n == 133
    _allclose(m.sp[0], 0.09454488, atol=5e-3, name="sp")
    _allclose(m.edf_total, 8.801932, atol=5e-3, name="edf_total")
    _allclose(m.sigma_squared, 727.3234, atol=5e-2, name="sigma2")
    _allclose(m.REML_criterion / 2, 637.9549, atol=5e-3, name="REML/2")
    _allclose(m.r_squared_adjusted, 0.6885152, atol=5e-3, name="r2adj")
    _assert_param(m, "(Intercept)", -25.54586, atol=5e-3)
    _allclose(m.edf_by_smooth["s(times)"], 7.801932, atol=5e-3, name="edf[s(times)]")


# ---------------------------------------------------------------------------
# 7) gamSim eg1 — overlap case: s(x1)+s(x2)+te(x1,x2) requires gam.side
# ---------------------------------------------------------------------------


def test_gamSim_eg1_overlap_gamSide_REML():
    """gam(y ~ x0 + s(x1, bs='cr') + s(x2) + te(x1, x2), method='REML')

    The te(x1, x2) marginals overlap the main-effect smooths s(x1) and s(x2).
    Without identifiability constraints the joint design would be rank-deficient
    along the te marginals. mgcv handles this in `gam.side`: it builds X1 from
    the intercept + every strict-subset smooth (here s(x1) and s(x2)), then
    QR-with-pivoting picks the te columns that are linearly dependent on X1
    and deletes them (along with the matching rows/cols of each marginal S).

    For this dataset gam.side drops 2 te columns (24 → 22), so the full design
    has p = 42 columns. Pinning p exercises that path end-to-end.

    The REML surface has multiple near-equivalent optima differing in how
    they distribute penalty between the te marginals and the main-effect
    s(x1)/s(x2). hea and mgcv land at different optima, so the per-marginal
    sp's diverge. Overall fit quantities (σ², REML, r², intercept, x0)
    still agree closely, and edfs land within ~0.34.
    """
    d = load_dataset("mgcv", "gamSim_eg1")
    m = gam("y ~ x0 + s(x1, bs='cr') + s(x2) + te(x1, x2)", d, method="REML")

    assert m.n == 400
    # gam.side must drop 2 te columns: intercept + x0 + s(x1)[9] + s(x2)[9] + te[24-2]
    assert m.bhat.shape[1] == 42, f"gam.side drop failed: p={m.bhat.shape[1]} (expected 42)"

    # 4 sp's: s(x1), s(x2), te-marginal-1, te-marginal-2
    assert m.sp.shape == (4,)
    _allclose(m.sp[1], 7.998938e-03, atol=5e-4, name="sp[s(x2)]")

    # Tight: overall fit
    _allclose(m.sigma_squared, 4.149471, atol=5e-2, name="sigma2")
    _allclose(m.r_squared_adjusted, 0.697276, atol=5e-3, name="r2adj")
    _allclose(m.REML_criterion / 2, 866.7819, atol=5e-1, name="REML/2")
    _assert_param(m, "(Intercept)", 7.642771, atol=5e-3)
    _assert_param(m, "x0", 0.394401, atol=5e-3)

    # Looser: edfs (multi-modal sp surface — mgcv vs hea land at different optima)
    _allclose(m.edf_total, 13.836828, atol=5e-1, name="edf_total")
    _allclose(m.edf_by_smooth["s(x1)"], 2.790683, atol=2e-1, name="edf[s(x1)]")
    _allclose(m.edf_by_smooth["s(x2)"], 8.044964, atol=5e-2, name="edf[s(x2)]")
    _allclose(m.edf_by_smooth["te(x1,x2)"], 1.001181, atol=5e-1, name="edf[te]")


# ---------------------------------------------------------------------------
# 8) nlme::Machines — re smooths (Wood 2017 §6.5 example)
# ---------------------------------------------------------------------------


def test_machines_re_smooths_REML():
    """gam(score ~ Machine + s(Worker, bs='re') + s(Machine, Worker, bs='re'),
       data=Machines, method='REML') and the by=Machine variant.

    Two random-effect formulations from Wood 2017 §6.5. Exercises:

      - bs='re' on a single factor (one column per Worker level)
      - bs='re' on a Machine:Worker interaction (one column per cell)
      - bs='re' with by=factor (one block per Machine level)

    All three paths require Worker/Machine to be pl.Enum factors. With raw
    CSV dtypes (Int64/Utf8) they silently degrade to single-column random
    *slopes*, blowing edf to ~5 and AIC by ~170.

    Pins target mgcv's published values directly: gam.side is now skipped
    for `bs='re'` smooths (matching mgcv's `side.constrain=FALSE` on re
    smooths), so the design has all 27 cols, the REML optimum lands at
    mgcv's sp's, and edf/loglike/sp/coefficients agree to 4-5 digits.
    AIC uses df = sum(edf2)+1 (Wood 2017 §6.11.3) with edf2 including
    both Vc1 (∂β/∂ρ propagation) and Vc2 (Cholesky-derivative
    correction) — the full mgcv ``gam.fit3.post.proc`` decomposition.
    """
    d = load_dataset("nlme", "Machines")

    b1 = gam("score ~ Machine + s(Worker, bs='re') + s(Machine, Worker, bs='re')",
             data=d, method="REML")
    assert b1.n == 54
    assert b1.p == 27  # full mgcv design, no gam.side surgery on re smooths
    # mgcv reference (mgcv 1.9-4, REML, gam.vcomp). hea's outer Newton now
    # uses mgcv's exact stopping rule (gam.fit3.r:1646-1658,
    # ``max(|grad|) ≤ score.scale·conv.tol·5`` AND
    # ``|Δscore| ≤ score.scale·conv.tol``) at the same default ``conv.tol=1e-6``.
    # That puts hea and mgcv inside the same stopping band; residual drift
    # is the natural noise of where each implementation lands within the
    # band (≈ a few×1e-3 on the most leveraged CI, the small-std re-by-factor
    # smooth s(Worker):MachineA).
    _allclose(b1.edf_total,  17.76461, atol=5e-5, name="b1.edf")
    # edf < edf2 < edf1 — sp uncertainty inflates df, capped by tr(2F-F²).
    _allclose(b1.edf1_total, 17.99523, atol=5e-5, name="b1.edf1")
    _allclose(b1.edf2_total, 17.85995, atol=5e-5, name="b1.edf2")
    assert b1.edf_total < b1.edf2_total <= b1.edf1_total
    _allclose(b1.sigma_squared, 0.92463,  atol=5e-5, name="b1.sigma2")
    _allclose(b1.loglike,    -63.73532,   atol=5e-4, name="b1.loglike")
    _allclose(b1.AIC,        165.19055,   atol=5e-4, name="b1.AIC")
    _assert_param(b1, "(Intercept)", 52.3556, atol=5e-3)
    # both blocks should have meaningful edf — degraded path would give ~1
    assert b1.edf_by_smooth["s(Worker)"] > 3.0
    assert b1.edf_by_smooth["s(Machine,Worker)"] > 8.0

    # vcomp: matches mgcv to 4-5 decimals on points and CIs.
    vc = b1.vcomp
    assert vc.shape == (3, 4)
    assert vc["name"].to_list() == ["s(Worker)", "s(Machine,Worker)", "scale"]
    expected = {
        "s(Worker)":         (4.78106, 2.24987, 10.15997),
        "s(Machine,Worker)": (3.72952, 2.38281,  5.83737),
        "scale":             (0.96158, 0.76325,  1.21143),
    }
    for nm, (sd, lo, hi) in expected.items():
        row = vc.filter(pl.col("name") == nm).row(0, named=True)
        _allclose(row["std_dev"], sd, atol=5e-4, name=f"vcomp {nm}.std")
        _allclose(row["lower"],   lo, atol=5e-4, name=f"vcomp {nm}.lo")
        _allclose(row["upper"],   hi, atol=5e-4, name=f"vcomp {nm}.hi")

    b2 = gam("score ~ Machine + s(Worker, bs='re') + s(Worker, bs='re', by=Machine)",
             data=d, method="REML")
    assert b2.n == 54
    # by=Machine produces one block per level → 3 extra sp's, total 4
    assert b2.sp.shape == (4,)
    _allclose(b2.edf_total,  17.64453, atol=5e-5, name="b2.edf")
    _allclose(b2.edf2_total, 17.98557, atol=5e-5, name="b2.edf2")
    _allclose(b2.sigma_squared, 0.92463, atol=5e-5, name="b2.sigma2")
    _allclose(b2.loglike,    -63.82464,  atol=5e-4, name="b2.loglike")
    _allclose(b2.AIC,        165.62043,  atol=5e-4, name="b2.AIC")

    vc2 = b2.vcomp
    assert vc2.shape == (5, 4)
    assert vc2["name"].to_list() == [
        "s(Worker)", "s(Worker):MachineA", "s(Worker):MachineB",
        "s(Worker):MachineC", "scale",
    ]
    expected_b2 = {
        "s(Worker)":          (3.78595, 1.79873,  7.96861),
        "s(Worker):MachineA": (1.94032, 0.25319, 14.86973),
        "s(Worker):MachineB": (5.87402, 2.98833, 11.54628),
        "s(Worker):MachineC": (2.84547, 0.82993,  9.75584),
        "scale":              (0.96158, 0.76325,  1.21143),
    }
    for nm, (sd, lo, hi) in expected_b2.items():
        row = vc2.filter(pl.col("name") == nm).row(0, named=True)
        _allclose(row["std_dev"], sd, atol=5e-4, name=f"b2 vcomp {nm}.std")
        _allclose(row["lower"],   lo, atol=5e-4, name=f"b2 vcomp {nm}.lo")
        # MachineA has the smallest std (1.94) but largest upper (~14.87),
        # making the upper highly sensitive to small sp shifts inside
        # mgcv's stopping band — see the docstring above.
        upper_atol = 3e-3 if nm == "s(Worker):MachineA" else 5e-4
        _allclose(row["upper"],   hi, atol=upper_atol, name=f"b2 vcomp {nm}.hi")


def test_data_helper_applies_schema_sidecar():
    """`hea.data()` must restore R's factor type via the JSON schema sidecar.

    Without it, factor columns come back from CSV as Int64/Utf8 and bs='re'
    / by=factor / fs / sz smooths silently take the non-factor fallthrough
    path — which is the Machines b1/b2 footgun (AIC ~337 instead of ~165).
    """
    from hea import data
    d = data("Machines", "nlme")
    assert isinstance(d.schema["Worker"], pl.Enum), \
        f"Worker should be pl.Enum, got {d.schema['Worker']}"
    assert isinstance(d.schema["Machine"], pl.Enum), \
        f"Machine should be pl.Enum, got {d.schema['Machine']}"


def test_factor_helper():
    """`hea.factor()` is the polars equivalent of R's factor() — the
    user-side fix for wild-data Int64-stored factor columns.
    """
    from hea import factor
    from hea.formula import _ORDERED_COLS_CV, set_ordered_cols

    # Bypass `hea.data` (which applies our schema sidecar) to simulate the
    # wild-data scenario where factor info has been stripped — exactly what
    # rdatasets gives us out of the box.
    import rdatasets
    df = pl.from_pandas(rdatasets.data("nlme", "Machines")).drop("rownames")
    assert df.schema["Worker"] == pl.Int64  # the wild-data scenario

    # Auto-detect levels, alphanumeric sort
    out = factor(df["Worker"])
    assert isinstance(out.dtype, pl.Enum)
    assert out.dtype.categories.to_list() == ["1", "2", "3", "4", "5", "6"]
    assert out.name == "Worker"  # preserved → with_columns replaces

    # Explicit levels override sort order
    out2 = factor(df["Worker"], levels=["6", "2", "4", "1", "3", "5"])
    assert out2.dtype.categories.to_list() == ["6", "2", "4", "1", "3", "5"]

    # Casting fixes the s(...,bs='re') breakage end-to-end
    set_ordered_cols(frozenset())  # clean slate
    df_fixed = df.with_columns(factor(df["Worker"]))
    m = gam("score ~ Machine + s(Worker, bs='re')", data=df_fixed, method="REML")
    # degraded path would give edf ~ 2; correct path gives ~5
    assert m.edf_total > 4.0, f"factor() didn't fix the re basis: edf={m.edf_total}"

    # ordered=True adds to contextvar; ordered=False leaves it alone
    set_ordered_cols(frozenset())
    factor(df["Worker"], ordered=True)
    assert "Worker" in _ORDERED_COLS_CV.get()
    factor(df["Worker"], ordered=False)
    assert "Worker" in _ORDERED_COLS_CV.get(), "ordered=False shouldn't unregister"

    # labels= dict: reorder + rename in one pass (R's factor(x, levels=, labels=))
    test = pl.Series("test", [0, 1, 1, 0, 1])
    out_l = factor(test, labels={0: "negative", 1: "positive"})
    assert out_l.dtype.categories.to_list() == ["negative", "positive"]
    assert out_l.to_list() == ["negative", "positive", "positive", "negative", "positive"]
    assert out_l.name == "test"

    # dict insertion order controls level order (= reference level)
    out_rev = factor(test, labels={1: "positive", 0: "negative"})
    assert out_rev.dtype.categories.to_list() == ["positive", "negative"]

    # column value missing from labels keys → replace_strict errors
    with pytest.raises(pl.exceptions.InvalidOperationError):
        factor(pl.Series([0, 1, 2]), labels={0: "a", 1: "b"})

    # labels and levels together is a usage error
    with pytest.raises(ValueError, match="not both"):
        factor(test, levels=[0, 1], labels={0: "a", 1: "b"})

    # passing a dict to levels= is the easy typo — fail loudly, not silently
    with pytest.raises(TypeError, match="not a dict"):
        factor(test, levels={0: "negative", 1: "positive"})


# ---------------------------------------------------------------------------
# Cross-cutting: sp passthrough reproduces a fixed-sp fit
# ---------------------------------------------------------------------------


def test_sp_passthrough_matches_optimized():
    """Calling gam(..., sp=m_opt.sp) must give the same fit as the optimized one."""
    d = load_dataset("MASS", "mcycle")
    m_opt = gam("accel ~ s(times)", d, method="REML")
    m_fix = gam("accel ~ s(times)", d, method="REML", sp=m_opt.sp)

    np.testing.assert_allclose(m_fix.sigma_squared, m_opt.sigma_squared, atol=1e-6)
    np.testing.assert_allclose(m_fix.edf_total, m_opt.edf_total, atol=1e-6)
    np.testing.assert_allclose(m_fix.fitted, m_opt.fitted, atol=1e-6)


def test_predict_inSample_matches_fitted():
    d = load_dataset("MASS", "mcycle")
    m = gam("accel ~ s(times)", d, method="REML")
    np.testing.assert_array_equal(m.predict(), m.fitted)


# ---------------------------------------------------------------------------
# Regression: PIRLS init must produce a valid (β_null, η_null, μ_null) for
# canonical inverse-Gaussian (link = 1/μ²). The previous baseline β=0 ⇒ η=0
# is invalid for this link (valideta requires η>0 finite), and step-halving
# toward η_old=0 cannot escape the invalid region — the fit raised
# `FloatingPointError: PIRLS step halving failed (validity)`. The fix is
# mgcv's null.coef pattern: project a constant valid η onto colspan(X).
# ---------------------------------------------------------------------------


def test_pirls_init_canonical_inverse_gaussian():
    """IG canonical fit on Wald-distributed data must converge."""
    from hea import inverse_gaussian
    rng = np.random.default_rng(0)
    n = 200
    x = rng.uniform(0.0, 1.0, n)
    mu = 1.5 + 0.5 * np.sin(2 * np.pi * x)               # ∈ [1.0, 2.0], strictly positive
    y = rng.wald(mean=mu, scale=1.0)
    df = pl.DataFrame({"x": x, "y": y})
    m = gam("y ~ s(x)", df, family=inverse_gaussian(), method="REML")
    assert m.n == n
    assert np.all(np.isfinite(m._beta))
    assert np.all(np.isfinite(m.fitted))
    # m.fitted is μ = linkinv(η). For canonical IG link (1/μ²) μ>0 ⇔ η>0,
    # so this also serves as a valideta check on the converged fit.
    assert np.all(m.fitted > 0)
    assert np.all(m.linear_predictors > 0)
    # Phase 2.2 wiring: unknown-scale family ⇒ log φ enters the outer
    # vector and `m._log_phi_hat` is finite. ``m.scale = m.sigma_squared``
    # is the post-fit Pearson estimate (mgcv's ``m$sig2 = scale.est``,
    # gam.fit3.r:606), reported regardless of method. The optimizer's
    # converged scale ``exp(log φ̂)`` (mgcv's ``reml.scale``) lives on
    # ``m._log_phi_hat`` — for REML the two coincide at the optimum
    # (FOC); for ML they don't.
    assert m._log_phi_hat is not None
    assert np.isfinite(m._log_phi_hat)
    np.testing.assert_allclose(m.scale, m._pearson_scale, atol=0.0)
    assert np.isfinite(m._pearson_scale)
    assert m.sigma_squared > 0
    # Intercept ≈ link(mean(y)) = 1/mean(y)² for an intercept-only fit;
    # with a smooth that captures most of the signal it lands near
    # link(mean(mu_true)) = 1/1.5² ≈ 0.444.
    intercept = m.bhat["(Intercept)"][0]
    assert 0.30 < intercept < 0.60


# ---------------------------------------------------------------------------
# Phase 1.9 — non-Gaussian post-fit smoke. trees + Gamma(log) is the canonical
# small-n GLM example; mgcv's r.sq, deviance_explained, and null_deviance only
# depend on (y, μ, wt, family) and family.dev_resids, so those land at mgcv's
# values even before the REML score is family-aware (Phase 2). sp/edf/AIC
# still depend on the (Gaussian-only) REML score, so they're pinned at hea's
# current values with a TODO — Phase 4's mgcv-oracle battery tightens them.
# ---------------------------------------------------------------------------


def test_trees_gamma_log_smoke():
    """trees + Gamma(log), method='REML': pin family-agnostic post-fit values
    against mgcv (those that don't depend on sp), and hea's current
    sp-dependent values as a regression guard until Phase 2 lands."""
    from hea import Gamma
    d = load_dataset("R", "trees")
    m = gam("Volume ~ s(Height) + s(Girth)", d, family=Gamma(link="log"),
            method="REML")

    # Family / link plumbing.
    assert m.family.name == "Gamma"
    assert m.family.link.name == "log"
    assert m.family.scale_known is False

    # μ vs η: log-link ⇒ μ = exp(η), strictly positive.
    assert np.all(m.fitted_values > 0)
    np.testing.assert_allclose(
        m.fitted_values, np.exp(m.linear_predictors), atol=1e-12,
    )
    assert m.fitted is m.fitted_values or np.array_equal(m.fitted, m.fitted_values)

    # df: n=31, intercept-only null ⇒ df_null = n-1 = 30.
    assert m.n == 31
    np.testing.assert_allclose(m.df_null, 30.0, atol=0.0)

    # mgcv reference values (R 4.5.3, mgcv 1.9-3) at the converged fit:
    #   sp        = (15742.67387, 0.2112713142)
    #   edf_total = 4.738161, edf2_total = 5.270166
    #   scale     = m$reml.scale = 0.0068696749 (m$scale = 0.0068300304)
    #   deviance  = 0.1805645860, null_deviance = 8.3172012147
    #   r2_adj    = 0.9744391060, dev_expl = 0.9782902227
    #   AIC       = 144.3438870069, logLik = -65.9017771491
    #   intercept = 3.2756440543

    # Tight pins (independent of optimizer convergence trajectory).
    np.testing.assert_allclose(m.r_squared_adjusted, 0.9744391060, atol=5e-5)
    np.testing.assert_allclose(m.deviance_explained, 0.9782902227, atol=5e-5)
    np.testing.assert_allclose(m.null_deviance, 8.3172012147, atol=5e-7)
    np.testing.assert_allclose(m.deviance, 0.1805645860, atol=5e-4)
    np.testing.assert_allclose(m.bhat["(Intercept)"][0], 3.2756440543, atol=5e-3)

    # Looser pins on optimizer-dependent quantities. Phase 2.2 is using
    # L-BFGS-B with finite-difference gradients on the (ρ, log φ) outer
    # vector; the score has a long flat plateau in the Height-smooth
    # direction (its smooth saturates well before sp[0] hits the upper
    # rho bound), so sp[0] reproducibly lands ~50× larger than mgcv's
    # analytical-Newton answer while edf/scale/deviance agree to ~5e-3.
    # Phase 3 (analytical (ρ, log φ) gradients/Hessian) will tighten this.
    np.testing.assert_allclose(m.sp[1], 0.2112713142, rtol=2e-3)
    np.testing.assert_allclose(m.edf_total,  4.738161, atol=5e-2)
    np.testing.assert_allclose(m.edf2_total, 5.270166, atol=5e-2)
    np.testing.assert_allclose(m.scale,      0.0068696749, atol=5e-5)
    np.testing.assert_allclose(m.sigma_squared, m.scale, atol=0.0)
    np.testing.assert_allclose(m.logLik, -65.9017771491, atol=2e-2)
    np.testing.assert_allclose(m.AIC,    144.3438870069, atol=1e-1)

    # AIC.default identity: AIC = -2·logLik + 2·npar (by construction).
    np.testing.assert_allclose(m.AIC, -2.0 * m.logLik + 2.0 * m.npar, atol=1e-10)
    np.testing.assert_allclose(m.BIC, -2.0 * m.logLik + np.log(m.n) * m.npar,
                               atol=1e-10)

    # Intercept: log(weighted_mean(Volume)) ≈ log(30.17) ≈ 3.408 for an
    # intercept-only fit; with two smooths absorbing most of the signal
    # the fitted intercept lands near 3.276.
    np.testing.assert_allclose(m.bhat["(Intercept)"][0], 3.2756425861, atol=5e-5)

    # First-five fitted μ vs mgcv reference — Phase 2.2 lands within ~5e-4
    # of mgcv even with the FD optimizer plateau (the smooths matter for μ,
    # not the saturated Height direction).
    np.testing.assert_allclose(
        m.fitted_values[:5],
        [10.62414379, 10.36186212, 10.41212209, 16.42891707, 19.68356227],
        atol=5e-3,
    )

    # Gamma(log) residual identities:
    #   working = (y-μ)/(dμ/dη) = (y-μ)/μ
    #   pearson = (y-μ)·√(wt/V) = (y-μ)/μ        (V=μ², wt=1)
    # ⇒ pearson == working for log Gamma.
    pearson = m.residuals_of("pearson")
    working = m.residuals_of("working")
    np.testing.assert_allclose(pearson, working, atol=1e-12)
    response = m.residuals_of("response")
    np.testing.assert_allclose(response, m._y_arr - m.fitted_values, atol=0.0)
    # Default residuals = deviance residuals.
    np.testing.assert_array_equal(m.residuals, m.residuals_of("deviance"))


def test_gaussian_residual_identities_and_aic_self_consistency():
    """For Gaussian-identity all four residual types collapse to (y-μ).
    Independent of mgcv pins, so this catches future regressions in
    residuals_of without depending on a fixture."""
    d = load_dataset("MASS", "mcycle")
    m = gam("accel ~ s(times)", d, method="REML")
    y = m._y_arr
    mu = m.fitted_values
    target = y - mu
    # η == μ for identity link.
    np.testing.assert_allclose(m.linear_predictors, mu, atol=0.0)
    np.testing.assert_allclose(m.residuals_of("response"), target, atol=0.0)
    np.testing.assert_allclose(m.residuals_of("deviance"), target, atol=1e-12)
    np.testing.assert_allclose(m.residuals_of("pearson"),  target, atol=1e-12)
    np.testing.assert_allclose(m.residuals_of("working"),  target, atol=1e-12)
    # m.residuals defaults to deviance residuals.
    np.testing.assert_array_equal(m.residuals, m.residuals_of("deviance"))
    # Deviance residual identity: Σ d_i² = m.deviance for Gaussian (V=1).
    np.testing.assert_allclose(np.sum(m.residuals_of("deviance") ** 2),
                               m.deviance, atol=1e-9)
    # AIC.default self-consistency.
    np.testing.assert_allclose(m.AIC, -2.0 * m.logLik + 2.0 * m.npar, atol=1e-10)
    # Bad type raises.
    with pytest.raises(ValueError):
        m.residuals_of("partial")


def test_reml_finite_for_trees_gamma_log():
    """Sanity: for the converged Gamma(log) fit, `_reml` returns a
    finite value at the hea-current sp. (Phase 2.2 makes φ̂ a joint outer
    variable; this just ensures the formula is wired up correctly.)"""
    from hea import Gamma, gam
    d = load_dataset("R", "trees")
    m = gam("Volume ~ s(Height) + s(Girth)", d,
            family=Gamma(link="log"), method="REML")
    log_phi = float(np.log(m.scale))
    v = m._reml(m._rho_hat, log_phi)
    assert np.isfinite(v)


# ---------------------------------------------------------------------------
# gam.check() — port of mgcv::gam.check / mgcv::k.check
# ---------------------------------------------------------------------------


def test_kcheck_mcycle_matches_mgcv():
    """k.check on `accel ~ s(times)` (1D smooth, REML).

    mgcv pin (n.rep=10000, see development log):
        s(times)  k'=9   edf=8.62469100  k-index=1.14736165
    edf and k-index are deterministic in the residuals + covariate; we
    pin them tightly. The p-value is a permutation tail and depends on
    the RNG draw — pin it to a wide-enough band that the test stays
    robust across RNG seeds.
    """
    d = load_dataset("MASS", "mcycle")
    m = gam("accel ~ s(times)", d, method="REML")
    ktab = m._k_check(seed=0, n_rep=2000)
    assert ktab[""].to_list() == ["s(times)"]
    np.testing.assert_allclose(ktab["k'"].to_list(),     [9.0],          atol=0)
    np.testing.assert_allclose(ktab["edf"].to_list(),    [8.62469100],   atol=5e-5)
    np.testing.assert_allclose(ktab["k-index"].to_list(),[1.14736165],   atol=5e-5)
    # mgcv reports ~0.95 with 10k reps; permutation noise widens the band.
    assert 0.85 < ktab["p-value"][0] <= 1.0


def test_kcheck_handles_no_smooths_returns_none():
    """k.check is undefined when there are no smooth blocks. Mirrors
    mgcv: `k.check` returns NULL → `gam.check` skips the table."""
    d = load_dataset("R", "trees")
    m = gam("Volume ~ Height + Girth", d, method="REML")
    assert m._k_check() is None


def test_check_prints_convergence_block(capsys):
    """`gam.check()` runs end-to-end and emits the mgcv-style header.

    The exact gradient/eigenvalue numbers are not pinned (those are
    determined by the converged ρ̂ and would shift if the optimizer is
    re-tuned later); we only verify the structural lines are there.
    """
    d = load_dataset("MASS", "mcycle")
    m = gam("accel ~ s(times)", d, method="REML")
    m.check(seed=0, k_rep=200)
    out = capsys.readouterr().out
    assert "Method: REML" in out
    assert "Optimizer: outer newton" in out
    assert "iteration" in out
    assert "Gradient range" in out
    assert "score " in out and "scale " in out
    assert "Hessian" in out and "eigenvalue range" in out
    assert "Model rank = " in out
    assert "Basis dimension (k) checking" in out
    assert "s(times)" in out


def test_check_no_smooth_path(capsys):
    """When the model has no smooths, the convergence block reports
    `Model required no smoothing parameter selection` (mgcv text) and
    the k-check table is omitted."""
    d = load_dataset("R", "trees")
    m = gam("Volume ~ Height + Girth", d, method="REML")
    m.check()
    out = capsys.readouterr().out
    assert "Model required no smoothing parameter selection" in out
    assert "Basis dimension" not in out


# ---------------------------------------------------------------------------
# LHS expressions — `y^0.25 ~ ...`, `log(y) ~ ...`, `I(y/100) ~ ...`
# ---------------------------------------------------------------------------


def test_lhs_power_brain_matches_mgcv():
    """Wood §7.2: `gam(medFPQ^.25 ~ s(Y, X, k=100), data=brain)`.

    mgcv pins on the trimmed dataset (medFPQ > 5e-5, n=1565):
        edf_total ≈ 65.176, sigma2 ≈ 0.039541, GCV ≈ 0.041259
    """
    d = load_dataset("gamair", "brain").filter(pl.col("medFPQ") > 5e-5)
    m = gam("medFPQ^.25 ~ s(Y, X, k=100)", d)
    assert m.n == 1565
    assert m.y.name == "medFPQ^0.25"
    _allclose(m.edf_total,     65.1763,  atol=1e-3, name="edf_total")
    _allclose(m.sigma_squared, 0.039541, atol=5e-6, name="sigma2")
    _allclose(m.GCV_score,     0.041259, atol=5e-6, name="GCV")


def test_lhs_log_matches_manual_transform():
    """`log(y) ~ ...` should be identical to pre-computing log(y) in
    polars and fitting `log_y ~ ...` on the same RHS."""
    d = load_dataset("R", "trees")
    m_lhs = gam("log(Volume) ~ s(Height) + s(Girth)", d, method="REML")
    d2 = d.with_columns(pl.col("Volume").log().alias("log_v"))
    m_pre = gam("log_v ~ s(Height) + s(Girth)", d2, method="REML")
    np.testing.assert_allclose(m_lhs.fitted, m_pre.fitted, atol=1e-12)
    np.testing.assert_allclose(m_lhs.sp,     m_pre.sp,     atol=0)
    np.testing.assert_allclose(m_lhs._beta,  m_pre._beta,  atol=1e-12)
    assert m_lhs.y.name == "log(Volume)"


def test_lhs_I_div_matches_manual_transform():
    """`I(y/100) ~ ...` is just an unwrap; should equal pre-computing
    y/100. Also verifies the deparsed label survives I()."""
    d = load_dataset("R", "trees")
    m_lhs = gam("I(Volume / 100) ~ s(Height) + s(Girth)", d, method="REML")
    d2 = d.with_columns((pl.col("Volume") / 100.0).alias("v100"))
    m_pre = gam("v100 ~ s(Height) + s(Girth)", d2, method="REML")
    np.testing.assert_allclose(m_lhs.fitted, m_pre.fitted, atol=1e-12)
    # Deparser inserts spaces around `/`; mgcv shows `I(Volume/100)` instead,
    # but both reduce to the same column transform — the visible label is
    # the deparser's choice, which is acceptable.
    assert "Volume" in m_lhs.y.name and "100" in m_lhs.y.name


def test_lhs_unsupported_function_raises():
    """An unsupported function on the LHS should error with a helpful
    message naming the allowed transforms."""
    d = load_dataset("R", "trees")
    with pytest.raises(NotImplementedError, match="not supported"):
        gam("foo(Volume) ~ s(Height)", d, method="REML")


def test_lhs_cbind_raises():
    """cbind() multi-column response is not implemented yet — error clearly."""
    d = load_dataset("R", "trees")
    with pytest.raises(NotImplementedError, match="cbind"):
        gam("cbind(Volume, Height) ~ s(Girth)", d, method="REML")


def test_lhs_unknown_column_raises():
    """Reference to a non-existent column inside an LHS expression."""
    d = load_dataset("R", "trees")
    with pytest.raises(KeyError, match="nope"):
        gam("log(nope) ~ s(Height)", d, method="REML")


def test_lhs_na_omit_drops_lhs_referenced_columns():
    """If the LHS expression touches a column that has NAs, those rows
    must be dropped before evaluating the response — otherwise polars
    would surface NaN through the transform."""
    d = pl.DataFrame({
        "a":  [1.0, 4.0, None, 16.0, 25.0, 36.0,  49.0, 64.0,  81.0, 100.0],
        "x":  [1.0, 2.0, 3.0,   4.0,  5.0,  6.0,   7.0,  8.0,   9.0,  10.0],
    })
    m = gam("sqrt(a) ~ s(x, k=4)", d, method="REML")
    # The NA row in `a` was dropped; n is 9, not 10.
    assert m.n == 9
    np.testing.assert_allclose(np.asarray(m.y.to_list()),
                               np.sqrt([1, 4, 16, 25, 36, 49, 64, 81, 100]),
                               atol=1e-12)


def test_check_outer_info_is_populated_after_fit():
    """`_outer_info` should be filled with grad/hess/score/iter after
    a smooth fit, and remain None for the no-smooth path."""
    d = load_dataset("MASS", "mcycle")
    m = gam("accel ~ s(times)", d, method="REML")
    info = m._outer_info
    assert info is not None
    assert info["iter"] >= 1
    # Gaussian REML puts (ρ, log φ) on the outer vector — for one smooth
    # that's length 2; known-scale families would be length 1.
    g = info["grad"]
    H = info["hess"]
    assert g.size >= len(m.sp)
    assert H.shape == (g.size, g.size)
    assert np.isfinite(info["score"])
    # mcycle's REML surface is well-behaved → Hessian PD at optimum.
    ev = np.linalg.eigvalsh(0.5 * (info["hess"] + info["hess"].T))
    assert ev.min() > 0

    d2 = load_dataset("R", "trees")
    m2 = gam("Volume ~ Height + Girth", d2, method="REML")
    assert m2._outer_info is None


# ---------------------------------------------------------------------------
# offset(...) — both via formula and via offset= kwarg.
#
# Identity check: a parametric-only formula's gam fit must exactly match
# the equivalent glm fit (gam reduces to glm when there are no smooths).
# Offset-shift check: predict with newdata re-evaluates formula offsets.
# Parity check: mgcv pinned values for a small Poisson+offset GAM.
# ---------------------------------------------------------------------------


def test_gam_offset_in_formula_matches_glm():
    """No smooths → gam == glm. Offset(...) inside the formula must
    propagate identically through both."""
    from hea import glm, Quasi
    d = load_dataset("MASS", "quine")  # count data
    # Synthetic offset column to exercise the path.
    d = d.with_columns(off=pl.lit(0.3) * pl.col("Days").cast(pl.Float64).clip(lower_bound=1).log())
    formula = "Days ~ offset(off) + Sex + Age"
    fam = Quasi(link="log", variance="mu")
    b_glm = glm(formula, family=fam, data=d)
    b_gam = gam(formula, family=fam, data=d, method="REML")
    np.testing.assert_allclose(
        b_gam._beta, b_glm._bhat_arr, atol=1e-10,
    )
    np.testing.assert_allclose(b_gam.deviance, b_glm.deviance, atol=1e-10)
    np.testing.assert_allclose(
        b_gam.fitted_values, b_glm.fitted_values, atol=1e-10,
    )


def test_gam_offset_kwarg_equivalent_to_formula_offset():
    """offset(off) in formula should give the same fit as offset=off kwarg."""
    rng = np.random.default_rng(0)
    n = 100
    d = pl.DataFrame({
        "y": rng.poisson(3.0, n).astype(float),
        "x": rng.standard_normal(n),
        "off_col": rng.uniform(0.0, 1.0, n),
    })
    from hea import Poisson
    a = gam("y ~ offset(off_col) + x", family=Poisson(), data=d, method="REML")
    b = gam("y ~ x", family=Poisson(), data=d, method="REML",
            offset=d["off_col"].to_numpy())
    np.testing.assert_allclose(a._beta, b._beta, atol=1e-10)
    np.testing.assert_allclose(a.deviance, b.deviance, atol=1e-10)


def test_gam_gamma_kwarg_matches_mgcv_on_trees():
    """``gamma=`` (mgcv's smoothing-strength multiplier) — Wood §4.6 cites
    ``gamma=1.4`` as a reasonable default for over-fit protection.

    Pinned: trees + Gamma(log), GCV.Cp and REML, both γ=1 and γ=1.4.
    Criterion values come from mgcv 1.9.4 directly.
    """
    from hea import Gamma
    trees = load_dataset("mgcv", "trees")

    # GCV.Cp path
    m_gcv_1 = gam("Volume ~ s(Height) + s(Girth)",
                  family=Gamma(link="log"), data=trees,
                  method="GCV.Cp", gamma=1.0)
    np.testing.assert_allclose(m_gcv_1.GCV_score, 0.008082356, atol=1e-6)
    np.testing.assert_allclose(m_gcv_1.sp[1], 0.342711, atol=1e-4)

    m_gcv_14 = gam("Volume ~ s(Height) + s(Girth)",
                   family=Gamma(link="log"), data=trees,
                   method="GCV.Cp", gamma=1.4)
    np.testing.assert_allclose(m_gcv_14.GCV_score, 0.009228008, atol=1e-6)
    np.testing.assert_allclose(m_gcv_14.sp[1], 0.524542, atol=1e-4)
    # γ>1 produces smoother fits — sp[1] (Girth) increases.
    assert m_gcv_14.sp[1] > m_gcv_1.sp[1]

    # REML path — hea's REML_criterion is -2·V_R; mgcv's b$gcv.ubre is V_R.
    m_reml_1 = gam("Volume ~ s(Height) + s(Girth)",
                   family=Gamma(link="log"), data=trees,
                   method="REML", gamma=1.0)
    np.testing.assert_allclose(m_reml_1.REML_criterion / 2, 78.00469, atol=1e-3)

    m_reml_14 = gam("Volume ~ s(Height) + s(Girth)",
                    family=Gamma(link="log"), data=trees,
                    method="REML", gamma=1.4)
    np.testing.assert_allclose(m_reml_14.REML_criterion / 2, 59.35457, atol=1e-3)


def test_plot_smooth_dispatches_2d_to_contour():
    """``plot_smooth`` should auto-render contour for 2D smooths
    (Wood 2017 Fig. 4.14 — bold/dashed/dotted contours + data scatter)."""
    import matplotlib
    matplotlib.use("Agg")
    from hea import Gamma
    trees = load_dataset("mgcv", "trees")
    ct5 = gam("Volume ~ s(Height, Girth, k=25)",
              family=Gamma(link="log"), data=trees)
    fig = ct5.plot_smooth(too_far=0.1)
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    # Title should carry the smooth label + edf, mgcv-style.
    assert "s(Height,Girth," in ax.get_title()
    assert ax.get_xlabel() == "Height"
    assert ax.get_ylabel() == "Girth"

    # Mixed 1D + 2D: panel 0 is 1D (no title, ylabel carries the label),
    # panel 1 is 2D (title carries the label).
    m = gam("Volume ~ s(Height) + s(Height, Girth, k=20)",
            family=Gamma(link="log"), data=trees)
    fig2 = m.plot_smooth(too_far=0.1)
    assert len(fig2.axes) == 2
    assert fig2.axes[0].get_title() == ""        # 1D panel
    assert "s(Height," in fig2.axes[0].get_ylabel()
    assert "s(Height,Girth," in fig2.axes[1].get_title()  # 2D panel


def test_plot_smooth_all_terms_factor_termplot():
    """``plot_smooth(all_terms=True)`` should add a parametric panel for
    the factor — Wood 2017 Fig. 4.15."""
    import matplotlib
    matplotlib.use("Agg")
    from hea import Gamma
    trees = load_dataset("mgcv", "trees").with_columns(
        Hclass=((pl.col("Height") / 10).floor() - 5)
            .cast(pl.Int64)
            .replace_strict([1, 2, 3], ["small", "medium", "large"],
                            return_dtype=pl.Enum(["small", "medium", "large"])),
    )
    ct7 = gam("Volume ~ Hclass + s(Girth)",
              family=Gamma(link="log"), data=trees)
    fig = ct7.plot_smooth(all_terms=True)
    assert len(fig.axes) == 2

    # Panel 0: smooth s(Girth)
    assert fig.axes[0].get_xlabel() == "Girth"
    assert "s(Girth," in fig.axes[0].get_ylabel()

    # Panel 1: parametric Hclass (factor termplot)
    assert fig.axes[1].get_xlabel() == "Hclass"
    assert fig.axes[1].get_ylabel() == "Partial for Hclass"
    # x-tick labels are the level names in factor order.
    xticks = [t.get_text() for t in fig.axes[1].get_xticklabels()]
    assert xticks == ["small", "medium", "large"]

    # all_terms=False (default) → only the smooth panel.
    fig2 = ct7.plot_smooth()
    assert len(fig2.axes) == 1


def test_plot_smooth_select_by_name_and_list():
    """``select=`` accepts a smooth label, a list of labels, or a list of
    ints; ordering follows the list."""
    import matplotlib
    matplotlib.use("Agg")
    d = load_dataset("synthetic", "seed_synth_basic")
    m = gam("y ~ s(x1) + s(x2) + s(x3)", d, method="REML")

    # Single string → one panel.
    fig = m.plot_smooth(select="s(x2)")
    assert len(fig.axes) == 1
    assert "s(x2," in fig.axes[0].get_ylabel()

    # List of strings → panels in given order. Reverse formula order to
    # verify ordering is honored.
    fig = m.plot_smooth(select=["s(x3)", "s(x1)"])
    assert len(fig.axes) == 2
    assert "s(x3," in fig.axes[0].get_ylabel()
    assert "s(x1," in fig.axes[1].get_ylabel()

    # Mixed int + str works.
    fig = m.plot_smooth(select=[0, "s(x3)"])
    assert len(fig.axes) == 2
    assert "s(x1," in fig.axes[0].get_ylabel()
    assert "s(x3," in fig.axes[1].get_ylabel()

    # Unknown name lists the available labels.
    with pytest.raises(ValueError, match="doesn't match"):
        m.plot_smooth(select="s(missing)")
    with pytest.raises(IndexError, match="out of range"):
        m.plot_smooth(select=99)


def test_plot_smooth_scheme_persp_for_2d():
    """``scheme=1`` renders a 2D smooth as a 3D persp wireframe; the panel's
    axes must be a 3D Axes3D and carry the smooth label as zlabel."""
    import matplotlib
    matplotlib.use("Agg")
    from mpl_toolkits.mplot3d import Axes3D
    from hea import Gamma
    trees = load_dataset("mgcv", "trees")
    m = gam("Volume ~ s(Height, Girth, k=20)",
            family=Gamma(link="log"), data=trees)

    # scheme=1 → persp axes; zlabel carries the smooth label.
    fig = m.plot_smooth(scheme=1)
    assert len(fig.axes) == 1
    assert isinstance(fig.axes[0], Axes3D)
    assert "s(Height,Girth," in fig.axes[0].get_zlabel()

    # scheme=0 (default) keeps the contour rendering.
    fig = m.plot_smooth(scheme=0)
    assert not isinstance(fig.axes[0], Axes3D)
    assert "s(Height,Girth," in fig.axes[0].get_title()


def test_plot_smooth_scheme_per_panel_list():
    """``scheme=[...]`` aligns to selected panels — 1D smooths ignore it,
    2D panels get persp where requested. Mirrors Wood 2017 Fig. 7.9."""
    import matplotlib
    matplotlib.use("Agg")
    from mpl_toolkits.mplot3d import Axes3D
    from hea import Gamma
    trees = load_dataset("mgcv", "trees")
    m = gam("Volume ~ s(Height) + s(Height, Girth, k=20)",
            family=Gamma(link="log"), data=trees)

    # 1D, 2D-persp — last panel must be 3D, first 2D.
    fig = m.plot_smooth(scheme=[0, 1])
    assert len(fig.axes) == 2
    assert not isinstance(fig.axes[0], Axes3D)
    assert isinstance(fig.axes[1], Axes3D)

    # Length mismatch raises.
    with pytest.raises(ValueError, match="scheme list must have length 2"):
        m.plot_smooth(scheme=[0, 1, 0])


def test_plot_smooth_ax_3d_required_for_persp():
    """Passing ``ax=`` for a 2D scheme=1 panel demands a 3D Axes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hea import Gamma
    trees = load_dataset("mgcv", "trees")
    m = gam("Volume ~ s(Height, Girth, k=20)",
            family=Gamma(link="log"), data=trees)

    fig, ax2d = plt.subplots()
    with pytest.raises(TypeError, match="3D Axes"):
        m.plot_smooth(scheme=1, ax=ax2d)

    fig = plt.figure()
    ax3d = fig.add_subplot(111, projection="3d")
    out = m.plot_smooth(scheme=1, ax=ax3d)
    assert out is ax3d


def test_gam_gamma_validation():
    d = load_dataset("R", "iris")
    with pytest.raises(ValueError, match="gamma"):
        gam("Sepal.Length ~ s(Petal.Length)", d, gamma=0.0)
    with pytest.raises(ValueError, match="gamma"):
        gam("Sepal.Length ~ s(Petal.Length)", d, gamma=-0.5)


def test_gam_predict_reevaluates_offset_on_newdata():
    """predict.gam re-evaluates formula offset(...) atoms on newdata."""
    rng = np.random.default_rng(0)
    n = 80
    d = pl.DataFrame({
        "y": rng.poisson(4.0, n).astype(float),
        "x": rng.standard_normal(n),
        "off_col": rng.uniform(0.5, 1.5, n),
    })
    from hea import Poisson
    m = gam("y ~ offset(off_col) + x", family=Poisson(), data=d, method="REML")
    # Same X but a different offset column → η̂ should shift by exactly Δoffset.
    new = d.with_columns((pl.col("off_col") + 2.0).alias("off_col"))
    eta_orig = m.predict(type="link")
    eta_new = m.predict(new, type="link")
    np.testing.assert_allclose(eta_new - eta_orig, 2.0, atol=1e-10)


# ---------------------------------------------------------------------------
# select=TRUE — mgcv's null-space penalty for term selection
# ---------------------------------------------------------------------------


def test_select_true_doubles_n_sp():
    """select=TRUE adds one null-space penalty per smooth → n_sp doubles."""
    d = load_dataset("synthetic", "seed_synth_basic")
    m_off = gam("y ~ s(x1) + s(x2) + s(x3)", d, method="REML")
    m_on  = gam("y ~ s(x1) + s(x2) + s(x3)", d, method="REML", select=True)
    assert len(m_off.sp) == 3
    assert len(m_on.sp) == 6


def test_select_true_three_smooth_REML():
    """gam(y ~ s(x1)+s(x2)+s(x3), data=seed_synth_basic, method="REML", select=TRUE)

    Pinned to mgcv's converged values. s(x3) is signal-free in this data, so
    select=TRUE shrinks its edf to ~0 — the whole point of the null-space
    penalty.
    """
    d = load_dataset("synthetic", "seed_synth_basic")
    m = gam("y ~ s(x1) + s(x2) + s(x3)", d, method="REML", select=True)

    # mgcv-converged scalars (from gam(..., select=TRUE) on seed_synth_basic):
    _allclose(m.edf_total, 2.912088577, atol=5e-3, name="edf_total")
    _allclose(m.sigma_squared, 0.8940008109, atol=5e-3, name="sigma2")
    _allclose(m.REML_criterion / 2, 277.0814067, atol=5e-3, name="REML/2")
    _assert_param(m, "(Intercept)", 1.091137918, atol=5e-3)

    # Per-smooth edf — both implementations land in the flat plateau where
    # the heavily-shrunk null-space sps drift; pin only the well-determined
    # active edf and assert s(x3) is essentially shrunk out.
    _allclose(m.edf_by_smooth["s(x1)"], 0.9739738079, atol=5e-3, name="edf[s(x1)]")
    _allclose(m.edf_by_smooth["s(x2)"], 0.9379440321, atol=5e-3, name="edf[s(x2)]")
    assert m.edf_by_smooth["s(x3)"] < 1e-2, \
        f"s(x3) should be selected out, got edf={m.edf_by_smooth['s(x3)']}"


def test_select_true_single_smooth_sp_passthrough():
    """Single-smooth select=TRUE: gam(..., sp=m_free.sp) reproduces the
    free-optimization fit at the same sp — the augmented sp vector (now
    length 2) is correctly threaded through the slot machinery, and the
    sp= path's profile-out log φ̂ matches the optimizer's converged log φ̂
    to optimizer-tolerance precision.
    """
    d = load_dataset("MASS", "mcycle")
    m_free = gam("accel ~ s(times)", d, method="REML", select=True)
    assert len(m_free.sp) == 2
    m_fix  = gam("accel ~ s(times)", d, method="REML", select=True, sp=m_free.sp)
    # β at the same rho is independent of φ — fitted values match exactly.
    np.testing.assert_allclose(m_fix.fitted, m_free.fitted, atol=1e-10)
    np.testing.assert_allclose(m_fix.edf_total, m_free.edf_total, atol=1e-10)
    # σ² and REML are profile-out-identical only at the *exact* gradient zero;
    # the optimizer stops at gradient ≈ 0 → expect ~1e-6 relative agreement.
    np.testing.assert_allclose(m_fix.sigma_squared, m_free.sigma_squared, rtol=1e-5)
    np.testing.assert_allclose(m_fix.REML_criterion, m_free.REML_criterion, rtol=1e-7)


def test_select_true_at_mgcv_sp_matches_mgcv():
    """At a fixed sp vector, hea's select=TRUE fit must reproduce mgcv's
    post-fit numbers — checks the null-space penalty math directly,
    bypassing optimizer convergence differences.

    mgcv-converged sp on `gamSim_eg1` for `y ~ s(x0)+s(x1)+s(x2)+s(x3)` with
    `select=TRUE, method="REML"`:
    """
    d = load_dataset("mgcv", "gamSim_eg1")
    sp_mgcv = np.array([
        2.521010255, 423334.7801,    # s(x0): wig, null
        1.843214985, 1.820731653,    # s(x1): wig, null
        0.00569866453, 47639.04804,  # s(x2): wig, null
        84968.55542, 131.2834178,    # s(x3): wig, null (essentially zeroed)
    ])
    m = gam("y ~ s(x0) + s(x1) + s(x2) + s(x3)",
            d, method="REML", select=True, sp=sp_mgcv)

    # mgcv post-fit at this sp — bit-perfect targets:
    _allclose(m.edf_total, 14.45446565, atol=1e-3, name="edf_total")
    _allclose(m.sigma_squared, 3.933035582, atol=1e-3, name="sigma2")
    _allclose(m.REML_criterion / 2, 868.3979813, atol=1e-3, name="REML/2")
    _assert_param(m, "(Intercept)", 7.833279497, atol=1e-3)
    _allclose(m.edf_by_smooth["s(x0)"], 2.418051213, atol=1e-3, name="edf[s(x0)]")
    _allclose(m.edf_by_smooth["s(x1)"], 2.839713272, atol=1e-3, name="edf[s(x1)]")
    _allclose(m.edf_by_smooth["s(x2)"], 7.448219388, atol=1e-3, name="edf[s(x2)]")
    _allclose(m.edf_by_smooth["s(x3)"], 0.7484817774, atol=1e-3, name="edf[s(x3)]")


# ---------------------------------------------------------------------------
# Summary p-value dispatch — known-scale family (binomial), select=TRUE
# ---------------------------------------------------------------------------


def test_select_true_binomial_summary_matches_mgcv():
    """hea's summary() must dispatch on ``family.scale_known``: known-scale
    families use the Wald z-test for parametric coefficients and the Wood
    (2013) reTest with Davies' weighted-χ² CDF for smooth significance,
    not t/F. Pinned to mgcv on wesdr at mgcv's converged sp.
    """
    from scipy.stats import norm
    from hea import Binomial

    d = load_dataset("gamair", "wesdr")
    sp_mgcv = np.array([
        0.0164113465035,  4.59199813892,  # s(dur): wig, null
        1793.09515417,    0.953183305109, # s(gly): wig, null
        0.0458306723482,  5.7780644155,   # s(bmi): wig, null
    ])
    m = gam("ret ~ s(dur,k=5) + s(gly,k=5) + s(bmi,k=5)",
            d, family=Binomial(), method="REML", select=True, sp=sp_mgcv)

    # Family / scale-known dispatch flag.
    assert m.family.scale_known is True

    # mgcv post-fit at the same sp:
    _allclose(m.edf_total, 7.430392736, atol=1e-3, name="edf_total")
    # mgcv's `b$gcv.ubre` (printed in summary as `-REML`) — for binomial it's
    # the REML/2 we report, since hea's `REML_criterion` doubles mgcv's value.
    _allclose(m.REML_criterion / 2, 389.4888704, atol=1e-3, name="REML/2")

    # Parametric: z-test, not t-test (binomial → φ ≡ 1).
    j = list(m.bhat.columns).index("(Intercept)")
    est = float(m._beta[j])
    se = float(m._se[j])
    z = est / se
    p_z = 2.0 * norm.sf(abs(z))
    _allclose(est, -0.4150103, atol=1e-3, name="intercept")
    _allclose(se, 0.0887844, atol=1e-3, name="intercept SE")
    _allclose(z, -4.674361, atol=5e-3, name="z")
    _allclose(p_z, 2.948704e-06, atol=5e-7, name="Pr(>|z|)")

    # Smooth significance via reTest (mgcv summary.gam reTest path):
    # mgcv pins (edf, Ref.df, Chi.sq, p-value):
    targets = [
        ("s(dur)", 2.982517, 4, 15.58609, 0.0007177005),
        ("s(gly)", 0.989778, 4, 91.07272, 0.0),
        ("s(bmi)", 2.458097, 4, 13.64956, 0.0008958199),
    ]
    for m_idx, (label, edf_t, refdf_t, chisq_t, pv_t) in enumerate(targets):
        a, bcol = m._block_col_ranges[m_idx]
        beta_b = m._beta[a:bcol]
        Vp_b = m.Vp[a:bcol, a:bcol]
        stat, pval, ref_df = m._re_test(m_idx, beta_b, Vp_b)
        _allclose(float(m.edf[a:bcol].sum()), edf_t,
                  atol=1e-3, name=f"edf[{label}]")
        # Ref.df = effective rank from Davies' eigenvalue truncation.
        # Under select=TRUE this is the basis dimension (= ncol of the
        # smooth's design block).
        assert int(ref_df) == refdf_t, f"Ref.df[{label}]: {ref_df} vs {refdf_t}"
        _allclose(stat, chisq_t, atol=5e-3, name=f"Chi.sq[{label}]")
        if pv_t > 0:
            _allclose(pval, pv_t, atol=1e-5, name=f"p-value[{label}]")
        else:
            assert pval < 1e-15, f"p-value[{label}] not vanishing: {pval}"


# ---------------------------------------------------------------------------
# method="ML" — Laplace marginal likelihood (does not profile out fixed
# effects, so scores are comparable across different fixed-effect structures
# in `anova(m1, m2)` LRTs). Differs from REML by a `Mp·log(2π·φ)` constant
# in the score formula (gam.fit3.r:616, remlInd=0).
# ---------------------------------------------------------------------------


def test_mcycle_tp_ML():
    """gam(accel ~ s(times), data=mcycle, method="ML")"""
    d = load_dataset("MASS", "mcycle")
    m = gam("accel ~ s(times)", d, method="ML")

    assert m.n == 133
    _allclose(m.sp[0], 7.742109e-04, atol=5e-5, name="sp")
    _allclose(m.edf_total, 9.625375, atol=5e-4, name="edf_total")
    _allclose(m.sigma_squared, 506.3487, atol=5e-3, name="sigma2")
    _allclose(m.ML_criterion / 2, 622.2919, atol=5e-3, name="ML/2")
    _allclose(m.r_squared_adjusted, 0.7831502, atol=5e-4, name="r2adj")
    _assert_param(m, "(Intercept)", -25.54586, atol=5e-3)
    _allclose(m.edf_by_smooth["s(times)"], 8.625375, atol=5e-4, name="edf[s(times)]")


def test_gamSim_eg1_four_smooths_ML():
    """gam(y ~ s(x0)+s(x1)+s(x2)+s(x3), data=gamSim(eg=1), method="ML")"""
    d = load_dataset("mgcv", "gamSim_eg1")
    m = gam("y ~ s(x0) + s(x1) + s(x2) + s(x3)", d, method="ML")

    assert m.n == 400
    _allclose(m.edf_total, 15.44156, atol=5e-3, name="edf_total")
    _allclose(m.sigma_squared, 3.897865, atol=5e-3, name="sigma2")
    _allclose(m.ML_criterion / 2, 860.3114, atol=5e-3, name="ML/2")
    _allclose(m.r_squared_adjusted, 0.7156318, atol=5e-3, name="r2adj")
    _assert_param(m, "(Intercept)", 7.833279, atol=5e-3)

    _allclose(m.edf_by_smooth["s(x0)"], 2.816760, atol=5e-3, name="edf[s(x0)]")
    _allclose(m.edf_by_smooth["s(x1)"], 2.620159, atol=5e-3, name="edf[s(x1)]")
    _allclose(m.edf_by_smooth["s(x2)"], 8.004539, atol=5e-3, name="edf[s(x2)]")
    _allclose(m.edf_by_smooth["s(x3)"], 1.000098, atol=5e-2, name="edf[s(x3)]")


def test_wesdr_binomial_ML():
    """gam(ret ~ s(dur)+s(gly)+s(bmi), data=wesdr, family=binomial, method="ML")

    Scale-known family. Even with φ ≡ 1, REML and ML pick *different* sp
    because the Hessian log-det differs (REML uses log|H+S|; ML uses
    log|H_pp+S_pp|, the range-only block — see mgcv ``MLpenalty1`` in
    gdi.c:1532-1680). mgcv's pins:
        REML sp ≈ (0.0565, 4205, 0.1277), edf 9.117, score 386.350
        ML   sp ≈ (0.0787, 34055, 0.2153), edf 8.417, score 384.004
    """
    from hea import Binomial
    d = load_dataset("gamair", "wesdr")
    m_ml = gam("ret ~ s(dur) + s(gly) + s(bmi)",
               d, family=Binomial(), method="ML")

    _allclose(m_ml.ML_criterion / 2, 384.0036, atol=5e-3, name="ML/2")
    _allclose(m_ml.edf_total, 8.416686, atol=5e-3, name="edf_total")
    _assert_param(m_ml, "(Intercept)", -0.4176841, atol=5e-3)
    _allclose(m_ml.sp[0], 0.07866319, atol=5e-3, name="sp[s(dur)]")
    _allclose(m_ml.sp[2], 0.2152721,  atol=5e-3, name="sp[s(bmi)]")
    # sp[1] for s(gly) is on a ~flat ridge (mgcv 34055, hea > 1e7) — both
    # are effectively "fully smoothed", so don't pin its absolute value;
    # the resulting fit (edf, score) is what matches.


def test_method_validation():
    """gam() rejects bogus method strings before doing any work."""
    d = load_dataset("MASS", "mcycle")
    with pytest.raises(ValueError, match="REML.*ML.*GCV"):
        gam("accel ~ s(times)", d, method="UBRE")
    with pytest.raises(ValueError, match="REML.*ML.*GCV"):
        gam("accel ~ s(times)", d, method="GACV.Cp")
    with pytest.raises(ValueError, match="REML.*ML.*GCV"):
        gam("accel ~ s(times)", d, method="P-REML")


# ---------------------------------------------------------------------------
# Tweedie / tw — end-to-end fits on a synthetic compound Poisson-Gamma
# response. Mirrors the mgcv mack/egg-count workflow at small n; checks
# that p̂ recovers the truth and tw() never scores worse than Tweedie(p̂_init).
# ---------------------------------------------------------------------------


def _simulate_compound_poisson_gamma(rng, n, p_true=1.5, phi_true=1.0):
    """Compound Poisson-Gamma sample: N_i ~ Poisson(λ_i), N_i Gamma jumps."""
    x = rng.uniform(0.0, 1.0, n)
    mu_true = np.exp(0.5 + 1.5 * np.sin(2.0 * np.pi * x))
    lam = mu_true ** (2.0 - p_true) / (phi_true * (2.0 - p_true))
    N = rng.poisson(lam)
    shape = (2.0 - p_true) / (p_true - 1.0)
    scale = phi_true * (p_true - 1.0) * mu_true ** (p_true - 1.0)
    y = np.zeros(n)
    for i in range(n):
        if N[i] > 0:
            y[i] = rng.gamma(shape * N[i], scale[i])
    return x, y, mu_true


def test_gam_fit_with_tweedie_fixed_p():
    rng = np.random.default_rng(42)
    x, y, _ = _simulate_compound_poisson_gamma(rng, n=200)
    df = pl.DataFrame({"y": y, "x": x})
    m = gam("y ~ s(x, k=8)", df, family=Tweedie(p=1.5), method="REML")
    assert np.isfinite(m.REML_criterion)
    assert 1.0 < m.edf_total < 8.0
    assert 0.5 < float(np.exp(m._log_phi_hat)) < 2.0


def test_gam_fit_with_tw_recovers_p_near_truth():
    """tw() with default initialisation should converge near the true p."""
    rng = np.random.default_rng(123)
    x, y, _ = _simulate_compound_poisson_gamma(rng, n=400, p_true=1.5)
    df = pl.DataFrame({"y": y, "x": x})
    m = gam("y ~ s(x, k=10)", df, family=tw(), method="REML")
    info = m._tw_info
    assert info is not None
    assert 1.30 < info["p_hat"] < 1.70


def test_gam_tw_mack_mgcv_oracle():
    """Pin tw() joint outer-Newton output against mgcv 1.9-4 on gamair::mack.

    Generated with:
        library(gamair); data(mack)
        mack$log.net.area <- log(mack$net.area)
        keep <- complete.cases(mack[, c("egg.count", "lon", "lat",
                                        "b.depth", "c.dist", "salinity",
                                        "temp.surf", "temp.20m",
                                        "log.net.area")])
        m <- gam(egg.count ~ s(lon, lat, k=20) + s(temp.surf),
                 data=mack[keep,], family=tw(), method="REML",
                 offset=log.net.area)

    p̂ matches to ~6 digits, REML/2 to 7 digits, scale to ~5 digits.
    sp[1] (temp.surf) sits on the flat-ridge tail where mgcv and hea both
    effectively fully smooth; only the resulting REML/edf are pinned there,
    not the absolute sp value.
    """
    mack = load_dataset("gamair", "mack")
    keep_cols = ["egg.count", "lon", "lat", "b.depth", "c.dist",
                 "salinity", "temp.surf", "temp.20m", "net.area"]
    mack = mack.drop_nulls(subset=keep_cols)
    mack = mack.with_columns(log_net_area=pl.col("net.area").log())

    m = gam(
        "egg.count ~ s(lon, lat, k=20) + s(temp.surf)",
        mack, family=tw(), method="REML",
        offset=mack["log_net_area"].to_numpy().tolist(),
    )
    info = m._tw_info
    assert info is not None
    np.testing.assert_allclose(info["p_hat"], 1.39920632555438, atol=1e-4)
    np.testing.assert_allclose(m.REML_criterion / 2,
                               945.744274311548, atol=1e-4)
    np.testing.assert_allclose(np.exp(info["log_phi_hat"]),
                               4.00764107362287, rtol=5e-4)
    np.testing.assert_allclose(m.edf_total, 17.9986147698585, atol=5e-2)
    np.testing.assert_allclose(m.sp[0], 0.161829581092981, rtol=5e-3)
    # sp[1] for s(temp.surf) sits in a flat tail (mgcv: 5.62, hea: 5.72) —
    # both are effectively "fully smoothed"; pin the resulting fit (REML,
    # edf above) instead of the sp itself.


def test_gam_fit_tw_score_no_worse_than_fixed_p():
    """Joint outer Newton over (ρ, log φ, θ_tw) only accepts steps that
    improve the criterion, so tw()'s REML score should be ≤ Tweedie(1.5)'s."""
    rng = np.random.default_rng(7)
    x, y, _ = _simulate_compound_poisson_gamma(rng, n=300, p_true=1.4)
    df = pl.DataFrame({"y": y, "x": x})
    m_fixed = gam("y ~ s(x, k=8)", df, family=Tweedie(p=1.5), method="REML")
    m_tw = gam("y ~ s(x, k=8)", df, family=tw(), method="REML")
    assert m_tw.REML_criterion <= m_fixed.REML_criterion + 1e-6


def test_tw_rejects_gcv_method():
    rng = np.random.default_rng(0)
    x, y, _ = _simulate_compound_poisson_gamma(rng, n=100)
    df = pl.DataFrame({"y": y, "x": x})
    with pytest.raises(ValueError, match="REML"):
        gam("y ~ s(x, k=6)", df, family=tw(), method="GCV.Cp")


def test_tw_rejects_fixed_sp():
    rng = np.random.default_rng(0)
    x, y, _ = _simulate_compound_poisson_gamma(rng, n=100)
    df = pl.DataFrame({"y": y, "x": x})
    with pytest.raises(ValueError, match="incompatible"):
        gam("y ~ s(x, k=6)", df, family=tw(), method="REML", sp=np.array([0.1]))
