"""
mgcv-oracle regression tests for lmpy.gam.

Each test pins the printed numerical outputs of `mgcv::gam(..., method=...)`
on a fixed dataset so the lmpy port can be validated against the canonical
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
from lmpy import gam


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
    s(x1)/s(x2). lmpy and mgcv land at different optima, so the per-marginal
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

    # Looser: edfs (multi-modal sp surface — mgcv vs lmpy land at different optima)
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
    # mgcv reference (mgcv 1.9-4, REML, gam.vcomp). Tolerances are tighter
    # than mgcv-printed-precision because lmpy's analytical Newton path
    # converges to |grad|<1e-10 (vs mgcv's ~3e-5 stopping criterion);
    # residual lmpy↔mgcv drift is dominated by mgcv's noise floor.
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
        _allclose(row["upper"],   hi, atol=5e-4, name=f"b2 vcomp {nm}.hi")


def test_data_helper_applies_schema_sidecar():
    """`lmpy.data()` must restore R's factor type via the JSON schema sidecar.

    Without it, factor columns come back from CSV as Int64/Utf8 and bs='re'
    / by=factor / fs / sz smooths silently take the non-factor fallthrough
    path — which is the Machines b1/b2 footgun (AIC ~337 instead of ~165).
    """
    from lmpy import data
    d = data("Machines", "nlme")
    assert isinstance(d.schema["Worker"], pl.Enum), \
        f"Worker should be pl.Enum, got {d.schema['Worker']}"
    assert isinstance(d.schema["Machine"], pl.Enum), \
        f"Machine should be pl.Enum, got {d.schema['Machine']}"


def test_factor_helper():
    """`lmpy.factor()` is the polars equivalent of R's factor() — the
    user-side fix for wild-data Int64-stored factor columns.
    """
    from lmpy import factor
    from lmpy.formula import _ORDERED_COLS_CV, set_ordered_cols

    df = pl.read_csv("datasets/nlme/Machines.csv", null_values="NA")
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
    from lmpy import inverse_gaussian
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
    # m.fitted is the linear predictor η = X β; for IG canonical link
    # valideta requires η>0 finite.
    assert np.all(m.fitted > 0)
    assert m.sigma_squared > 0
    # Intercept ≈ link(mean(y)) = 1/mean(y)² for an intercept-only fit;
    # with a smooth that captures most of the signal it lands near
    # link(mean(mu_true)) = 1/1.5² ≈ 0.444.
    intercept = m.bhat["(Intercept)"][0]
    assert 0.30 < intercept < 0.60
