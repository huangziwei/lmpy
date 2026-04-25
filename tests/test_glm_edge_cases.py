"""Phase 6 — edge cases.

Each block here covers one ``glm()`` API surface that's commonly used in
R but easy to overlook in a port:

- 6.1 ``cbind(s, f) ~ ...`` LHS for binomial (vs the proportion+weights
  rewrite tests/test_glm_examples.py uses).
- 6.2 ``offset(...)`` inside the formula (vs the ``offset=`` kwarg).
- 6.3 frequency / prior ``weights=``.
- 6.4 intercept-only fit ``y ~ 1``.
- 6.5 rank-deficient X — duplicate column ⇒ NA-coefficient slot, R-style.
- 6.6 factor-response binomial — 2-level factor on LHS, level 2 = success.

Where the value is small and stable we inline an R-computed reference
(``Rscript -e ...`` produced the literals in the comment beside each
``np.testing.assert_allclose`` call). For larger pinned cases we reuse
the existing JSON oracles via :func:`load_glm_oracle`.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from conftest import load_dataset, load_glm_oracle
from lmpy import Binomial, Gaussian, Poisson, glm


# ---------------------------------------------------------------------------
# 6.1 — cbind(success, failure) on LHS for binomial.
# ---------------------------------------------------------------------------


def test_cbind_lhs_matches_proportion_weights_form():
    """``cbind(s, f) ~ x`` and ``p ~ x, weights=s+f`` must produce the
    same fit, since lmpy rewrites the former into the latter internally.
    """
    d = pl.DataFrame({
        "s": [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "f": [9.0, 8, 7, 6, 5, 4, 3, 2, 1, 0.1],
        "x": np.arange(10, dtype=float),
    })
    m_cb = glm("cbind(s, f) ~ x", d, family=Binomial())

    # Equivalent rewrite the user could do by hand.
    p = d["s"] / (d["s"] + d["f"])
    w = (d["s"] + d["f"]).to_numpy()
    d2 = d.with_columns(p.alias("p"))
    m_pw = glm("p ~ x", d2, family=Binomial(), weights=w)

    np.testing.assert_allclose(m_cb._bhat_arr, m_pw._bhat_arr, atol=1e-10)
    np.testing.assert_allclose(m_cb.deviance, m_pw.deviance, atol=1e-10)
    np.testing.assert_allclose(m_cb.fitted_values, m_pw.fitted_values, atol=1e-10)


def test_cbind_lhs_matches_menarche_oracle():
    """The R menarche oracle was generated from ``cbind(Menarche, Total -
    Menarche) ~ Age`` with logit. lmpy's cbind path must reproduce the
    same coefficients and deviance as R."""
    o = load_glm_oracle("binomial_logit_menarche")
    d = load_dataset("MASS", "menarche")
    # cbind(a, b) accepts any expression, including subtraction.
    m = glm("cbind(Menarche, Total - Menarche) ~ Age", d,
            family=Binomial(link="logit"))
    np.testing.assert_allclose(m._bhat_arr, np.asarray(o["coefficients"]),
                               atol=5e-5)
    np.testing.assert_allclose(m.deviance, o["deviance"], atol=5e-4)


def test_cbind_lhs_rejects_non_binomial():
    d = pl.DataFrame({"s": [1.0, 2], "f": [3.0, 4], "x": [1.0, 2]})
    with pytest.raises(ValueError, match="cbind.*Binomial"):
        glm("cbind(s, f) ~ x", d, family=Gaussian())


# ---------------------------------------------------------------------------
# 6.2 — offset(...) inside the formula.
# ---------------------------------------------------------------------------


def test_formula_offset_matches_kwarg_offset():
    """``y ~ x + offset(log(t))`` must produce the same fit as
    ``y ~ x, offset=log(t)``. Both are valid R syntax and ``glm.fit`` adds
    them together when both are present."""
    d = pl.DataFrame({
        "y": [1, 5, 12, 30, 50, 80],
        "x": [1.0, 2, 3, 4, 5, 6],
        "t": [1.0, 2, 5, 10, 20, 30],
    })
    log_t = np.log(d["t"].to_numpy().astype(float))

    m_form = glm("y ~ x + offset(log(t))", d, family=Poisson())
    m_kw   = glm("y ~ x", d, family=Poisson(), offset=log_t)
    np.testing.assert_allclose(m_form._bhat_arr, m_kw._bhat_arr, atol=1e-12)
    np.testing.assert_allclose(m_form.deviance, m_kw.deviance, atol=1e-12)


def test_formula_offset_sums_with_kwarg_offset():
    """When both are present, R sums them (η = Xβ + Σ formula_offsets +
    kwarg_offset). Verify lmpy does the same by checking that splitting an
    offset between formula and kwarg gives the same fit as putting it all
    on one side."""
    d = pl.DataFrame({
        "y": [1, 5, 12, 30, 50, 80],
        "x": [1.0, 2, 3, 4, 5, 6],
        "t": [1.0, 2, 5, 10, 20, 30],
    })
    log_t = np.log(d["t"].to_numpy().astype(float))
    half = log_t / 2

    m_split = glm("y ~ x + offset(log(t)/2)", d, family=Poisson(), offset=half)
    m_all   = glm("y ~ x", d, family=Poisson(), offset=log_t)
    np.testing.assert_allclose(m_split._bhat_arr, m_all._bhat_arr, atol=1e-10)


# ---------------------------------------------------------------------------
# 6.3 — frequency weights (Phase 1 already plumbed; just pin one oracle).
# ---------------------------------------------------------------------------


def test_weighted_poisson_matches_r():
    """``glm(y ~ x, weights=...)`` for Poisson. R-pinned literals from:
        d <- data.frame(y=c(1,2,3,4,5), x=c(1,2,3,4,5))
        coef(glm(y~x, data=d, family=poisson(), weights=c(1,2,1,3,2)))
    """
    d = pl.DataFrame({"y": [1.0, 2, 3, 4, 5], "x": [1.0, 2, 3, 4, 5]})
    w = np.array([1.0, 2, 1, 3, 2])
    m = glm("y ~ x", d, family=Poisson(), weights=w)
    np.testing.assert_allclose(
        m._bhat_arr, [-0.009707952, 0.3360234], atol=5e-7,
    )


# ---------------------------------------------------------------------------
# 6.4 — intercept-only.
# ---------------------------------------------------------------------------


def test_intercept_only_gaussian():
    """``y ~ 1`` Gaussian: β̂_0 = mean(y); deviance = Σ (y-ȳ)²; df_residual = n-1."""
    y = np.array([1.0, 2, 3, 4, 5])
    d = pl.DataFrame({"y": y})
    m = glm("y ~ 1", d, family=Gaussian())
    np.testing.assert_allclose(m._bhat_arr, [y.mean()], atol=1e-12)
    np.testing.assert_allclose(m.deviance, ((y - y.mean()) ** 2).sum(), atol=1e-12)
    assert m.df_residual == len(y) - 1
    assert m.df_null == len(y) - 1
    # null deviance == residual deviance for the intercept-only model.
    np.testing.assert_allclose(m.null_deviance, m.deviance, atol=1e-12)


def test_intercept_only_poisson():
    """``y ~ 1`` Poisson/log: β̂_0 = log(mean(y))."""
    y = np.array([1, 2, 3, 4, 5, 4, 6, 8])
    d = pl.DataFrame({"y": y})
    m = glm("y ~ 1", d, family=Poisson())
    np.testing.assert_allclose(m._bhat_arr, [np.log(y.mean())], atol=1e-7)


# ---------------------------------------------------------------------------
# 6.5 — rank-deficient X.
# ---------------------------------------------------------------------------


def test_rank_deficient_x_drops_to_NA_slot():
    """``y ~ x + z`` with z = 2x is rank-deficient. R sets the dropped
    coef to NA, lowers the model rank, and bumps df_residual accordingly.
    lmpy does the same via the QR pivot."""
    n = 8
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    z = 2 * x          # perfectly collinear
    y = 1 + 0.5 * x + rng.normal(scale=0.1, size=n)
    d = pl.DataFrame({"y": y, "x": x, "z": z})
    m = glm("y ~ x + z", d, family=Gaussian())

    coefs = m._bhat_arr
    # One slot — the second-encountered collinear column — must be NaN.
    assert np.sum(np.isnan(coefs)) == 1, f"expected exactly 1 NaN coef, got {coefs}"
    # The non-NaN coefs are the intercept and (x or z, whichever R kept).
    assert m.rank == 2
    assert m.df_residual == n - m.rank
    # Fit still hits the data: deviance ≈ Σ (y - ŷ)² with x's true slope.
    np.testing.assert_allclose(m.deviance,
                               ((y - m.fitted_values) ** 2).sum(),
                               atol=1e-12)


# ---------------------------------------------------------------------------
# 6.6 — factor-response binomial.
# ---------------------------------------------------------------------------


def test_factor_response_binomial_string():
    """``y ~ x`` with y a 2-level string column. R uses level-1 = failure
    (=0), level-2 = success (=1), with factor levels in alphabetical order
    when not explicitly declared. R-pinned literals from:
        d <- data.frame(y=factor(c("a","b","a","b","a","b","a","b")),
                        x=c(1.0,2,3,4,5,6,7,8))
        coef(glm(y ~ x, data=d, family=binomial()))
    """
    d = pl.DataFrame({
        "y": ["a", "b", "a", "b", "a", "b", "a", "b"],
        "x": [1.0, 2, 3, 4, 5, 6, 7, 8],
    })
    m = glm("y ~ x", d, family=Binomial())
    np.testing.assert_allclose(m._bhat_arr, [-0.8822461, 0.1960547], atol=5e-7)
    np.testing.assert_allclose(m.fitted_values[:3],
                               [0.3348809, 0.3798614, 0.4270048], atol=5e-6)


def test_factor_response_binomial_enum_respects_declared_order():
    """``pl.Enum`` declares level order explicitly; R's ``factor(..., levels=)``
    is equivalent. Reverse the levels and the encoding flips sign on β̂_x.
    """
    base = pl.DataFrame({
        "y": ["a", "b", "a", "b", "a", "b", "a", "b"],
        "x": [1.0, 2, 3, 4, 5, 6, 7, 8],
    })
    # default alphabetical ⇒ "a"=0, "b"=1
    d_ab = base.with_columns(pl.col("y").cast(pl.Enum(["a", "b"])))
    # reversed ⇒ "b"=0, "a"=1
    d_ba = base.with_columns(pl.col("y").cast(pl.Enum(["b", "a"])))
    m_ab = glm("y ~ x", d_ab, family=Binomial())
    m_ba = glm("y ~ x", d_ba, family=Binomial())
    # Reversing the success/failure flips the sign of every coefficient.
    np.testing.assert_allclose(m_ab._bhat_arr, -m_ba._bhat_arr, atol=1e-10)


def test_factor_response_binomial_rejects_three_level():
    d = pl.DataFrame({"y": ["a", "b", "c", "a", "b", "c"],
                      "x": [1.0, 2, 3, 4, 5, 6]})
    with pytest.raises(ValueError, match="2 levels"):
        glm("y ~ x", d, family=Binomial())


def test_factor_response_binomial_boolean():
    d = pl.DataFrame({
        "y": [False, True, False, True, False, True, False, True],
        "x": [1.0, 2, 3, 4, 5, 6, 7, 8],
    })
    m = glm("y ~ x", d, family=Binomial())
    # FALSE=0, TRUE=1 — same as factor("a","b") above (alphabetical).
    np.testing.assert_allclose(m._bhat_arr, [-0.8822461, 0.1960547], atol=5e-7)
