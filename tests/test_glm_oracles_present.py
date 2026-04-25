"""Smoke check: every (family, link, dataset) triple in the glm-port plan
(.claude/plans/glm-port.md, Phase 0.2) has an oracle JSON dumped by
tests/scripts/make_glm_oracles.R, and the oracle parses to the expected shape.

This test catches a missing fixture before the family-level parity tests
fail with a confusing FileNotFoundError. If you add a triple to Phase 0.2,
update both the R script and the EXPECTED list below.
"""

from __future__ import annotations

import pytest

from conftest import load_glm_oracle


EXPECTED = [
    ("gaussian_identity_iris",        "gaussian", "identity"),
    ("gaussian_log_insurance",        "gaussian", "log"),
    ("gamma_inverse_trees",           "Gamma",    "inverse"),
    ("gamma_log_trees",               "Gamma",    "log"),
    ("poisson_log_quine",             "poisson",  "log"),
    ("poisson_sqrt_quine",            "poisson",  "sqrt"),
    ("binomial_logit_menarche",       "binomial", "logit"),
    ("binomial_probit_menarche",      "binomial", "probit"),
    ("binomial_cauchit_menarche",     "binomial", "cauchit"),
    ("binomial_cloglog_menarche",     "binomial", "cloglog"),
    ("ig_canonical_insurance",        "inverse.gaussian", "1/mu^2"),
]

REQUIRED_KEYS = {
    "id", "formula", "family_name", "link_name", "n", "coef_names",
    "coefficients", "std_error", "test_stat", "p_value", "test_kind",
    "ci_lower", "ci_upper", "vcov",
    "deviance", "null_deviance", "df_residual", "df_null",
    "aic", "bic", "loglik", "loglik_df", "dispersion", "iter", "converged",
    "fitted_values", "linear_pred",
    "res_deviance", "res_pearson", "res_working", "res_response",
    "pred_link_fit", "pred_link_se", "pred_resp_fit", "pred_resp_se",
}


@pytest.mark.parametrize("oid,family,link", EXPECTED)
def test_oracle_present(oid: str, family: str, link: str):
    o = load_glm_oracle(oid)
    assert o["family_name"] == family, f"{oid}: family mismatch"
    assert o["link_name"] == link, f"{oid}: link mismatch"
    missing = REQUIRED_KEYS - set(o.keys())
    assert not missing, f"{oid}: missing keys {sorted(missing)}"
    p = len(o["coefficients"])
    assert len(o["std_error"]) == p
    assert len(o["test_stat"]) == p
    assert len(o["p_value"]) == p
    assert len(o["coef_names"]) == p
    n = o["n"]
    assert len(o["fitted_values"]) == n
    assert len(o["linear_pred"]) == n
    assert len(o["res_deviance"]) == n
    assert o["test_kind"] in ("z", "t")
    # binomial / poisson are scale-known → z-test; the rest → t-test.
    if family in ("poisson", "binomial"):
        assert o["test_kind"] == "z"
    else:
        assert o["test_kind"] == "t"
