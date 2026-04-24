"""
Notebook examples → regression tests for lmpy.lme.

Pins printed numerical outputs from Bates, "lme4: Mixed-effects Modeling
with R" (lMMwR.pdf, in example/data/) so the formulae→formula.py
migration can be validated against book-standard lme4 results. Models
covered: fm01, fm02 (Ch 1, Dyestuff/Dyestuff2); fm03, fm04, fm04a (Ch 2,
Penicillin/Pastes); fm06, fm07 (Ch 3, sleepstudy); fm10, fm16, fm17
(Ch 4, Machines/ergoStool).

The post-migration lme is expected to expose, at minimum:
    m.n, m.n_groups       — sample size, dict of group → #levels
    m.sigma                — residual SD
    m.sd_re[group]         — np.ndarray of component SDs (length 1 for
                             scalar bars; length 2+ for vector bars)
    m.corr_re[group]       — np.ndarray correlation matrix (only present
                             for vector bars; missing/None for scalar)
    m.bhat / m.se_bhat / m.t_values   — DataFrames keyed by fixed-effect
                                        column name (R-canonical, e.g.
                                        '(Intercept)', 'MachineB')
    m.REML_criterion       — only set when REML=True
    m.deviance, m.loglike, m.AIC, m.BIC, m.df_resid   — only set when
                                                        REML=False (ML)
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from conftest import load_dataset
from lmpy.lme import lme


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _assert_fixed(m, col, est, se=None, tval=None, *, atol=5e-3):
    if col not in m.bhat.columns:
        raise KeyError(f"{col!r} not in {list(m.bhat.columns)!r}")
    np.testing.assert_allclose(m.bhat[col][0], est, atol=atol)
    if se is not None:
        np.testing.assert_allclose(m.se_bhat[col][0], se, atol=atol)
    if tval is not None:
        np.testing.assert_allclose(m.t_values[col][0], tval, atol=5e-2)


def _assert_re_scalar(m, group, sd, *, atol=5e-3):
    sds = np.asarray(m.sd_re[group]).ravel()
    assert sds.shape == (1,), f"expected scalar bar at {group!r}, got shape {sds.shape}"
    np.testing.assert_allclose(sds[0], sd, atol=atol)


def _assert_re_vector(m, group, sds, corr=None, *, atol=5e-3, corr_atol=5e-2):
    got = np.asarray(m.sd_re[group]).ravel()
    np.testing.assert_allclose(got, np.asarray(sds), atol=atol)
    if corr is not None:
        C = np.asarray(m.corr_re[group])
        # Off-diagonal only — diagonal is trivially 1.
        i, j = np.triu_indices(C.shape[0], k=1)
        np.testing.assert_allclose(C[i, j], np.asarray(corr), atol=corr_atol)


def _assert_ml_summary(m, *, AIC, BIC, loglike, deviance, df_resid,
                       atol=5e-2):
    np.testing.assert_allclose(m.AIC, AIC, atol=atol)
    np.testing.assert_allclose(m.BIC, BIC, atol=atol)
    np.testing.assert_allclose(m.loglike, loglike, atol=atol)
    np.testing.assert_allclose(m.deviance, deviance, atol=atol)
    assert m.df_resid == df_resid


def _lrt(m_reduced, m_full):
    chisq = m_reduced.deviance - m_full.deviance
    df = m_full.npar - m_reduced.npar
    p = chi2.sf(chisq, df)
    return chisq, df, p


# ---------------------------------------------------------------------------
# Ch 1: A Simple, Linear, Mixed-effects Model
# ---------------------------------------------------------------------------


def test_bates_1_4_dyestuff_fm01_REML():
    """fm01 <- lmer(Yield ~ 1 + (1|Batch), Dyestuff)  -- REML (default)"""
    data = load_dataset("lme4", "Dyestuff")
    m = lme("Yield ~ 1 + (1|Batch)", data, REML=True)

    assert m.n == 30
    assert m.n_groups == {"Batch": 6}
    np.testing.assert_allclose(m.REML_criterion, 319.7, atol=0.1)
    np.testing.assert_allclose(m.sigma, 49.5101, atol=5e-3)
    _assert_re_scalar(m, "Batch", 42.0010)
    _assert_fixed(m, "(Intercept)", 1527.5, se=19.38, tval=78.80)


def test_bates_1_4_dyestuff_fm01_ML():
    """fm01ML <- lmer(Yield ~ 1 + (1|Batch), Dyestuff, REML=FALSE)"""
    data = load_dataset("lme4", "Dyestuff")
    m = lme("Yield ~ 1 + (1|Batch)", data, REML=False)

    _assert_ml_summary(
        m, AIC=333.3271, BIC=337.5307, loglike=-163.6635,
        deviance=327.3271, df_resid=27,
    )
    np.testing.assert_allclose(m.sigma, 49.5101, atol=5e-3)
    _assert_re_scalar(m, "Batch", 37.2602, atol=5e-3)
    _assert_fixed(m, "(Intercept)", 1527.5, se=17.6938, tval=86.33)


def test_bates_1_4_dyestuff2_fm02_REML():
    """fm02 <- lmer(Yield ~ 1 + (1|Batch), Dyestuff2)  -- singular fit, σ₁=0"""
    data = load_dataset("lme4", "Dyestuff2")
    m = lme("Yield ~ 1 + (1|Batch)", data, REML=True)

    np.testing.assert_allclose(m.REML_criterion, 161.8, atol=0.1)
    np.testing.assert_allclose(m.sigma, 3.7165, atol=5e-3)
    _assert_re_scalar(m, "Batch", 0.0, atol=1e-4)
    _assert_fixed(m, "(Intercept)", 5.6656, se=0.6784, tval=8.352)


# ---------------------------------------------------------------------------
# Ch 2: Models With Multiple Random-effects Terms
# ---------------------------------------------------------------------------


def test_bates_2_1_penicillin_fm03_REML():
    """fm03 <- lmer(diameter ~ 1 + (1|plate) + (1|sample), Penicillin)"""
    data = load_dataset("lme4", "Penicillin")
    m = lme("diameter ~ 1 + (1|plate) + (1|sample)", data, REML=True)

    assert m.n == 144
    assert m.n_groups == {"plate": 24, "sample": 6}
    np.testing.assert_allclose(m.REML_criterion, 330.9, atol=0.1)
    np.testing.assert_allclose(m.sigma, 0.5499, atol=5e-3)
    _assert_re_scalar(m, "plate", 0.8467)
    _assert_re_scalar(m, "sample", 1.9316)
    _assert_fixed(m, "(Intercept)", 22.9722, se=0.8086, tval=28.41)


def test_bates_2_2_pastes_fm04_ML():
    """fm04 <- lmer(strength ~ 1 + (1|sample) + (1|batch), Pastes, REML=FALSE)"""
    data = load_dataset("lme4", "Pastes")
    m = lme("strength ~ 1 + (1|sample) + (1|batch)", data, REML=False)

    assert m.n == 60
    assert m.n_groups == {"sample": 30, "batch": 10}
    _assert_ml_summary(
        m, AIC=255.9945, BIC=264.3724, loglike=-123.9972,
        deviance=247.9945, df_resid=56,
    )
    np.testing.assert_allclose(m.sigma, 0.8234, atol=5e-3)
    _assert_re_scalar(m, "sample", 2.9041)
    _assert_re_scalar(m, "batch", 1.0951)
    _assert_fixed(m, "(Intercept)", 60.0533, se=0.6421, tval=93.52)


def test_bates_2_2_pastes_fm04a_ML_and_LRT():
    """fm04a <- lmer(strength ~ 1 + (1|sample), Pastes, REML=FALSE);
       anova(fm04a, fm04) — LRT for σ_batch = 0."""
    data = load_dataset("lme4", "Pastes")
    full = lme("strength ~ 1 + (1|sample) + (1|batch)", data, REML=False)
    red = lme("strength ~ 1 + (1|sample)", data, REML=False)

    _assert_ml_summary(
        red, AIC=254.4020, BIC=260.6855, loglike=-124.2010,
        deviance=248.4020, df_resid=57,
    )
    np.testing.assert_allclose(red.sigma, 0.8234, atol=5e-3)
    _assert_re_scalar(red, "sample", 3.1037)
    _assert_fixed(red, "(Intercept)", 60.0533, se=0.5765, tval=104.2)

    chisq, df, p = _lrt(red, full)
    np.testing.assert_allclose(chisq, 0.4072, atol=5e-3)
    assert df == 1
    np.testing.assert_allclose(p, 0.5234, atol=5e-3)


# ---------------------------------------------------------------------------
# Ch 3: Models for Longitudinal Data (sleepstudy)
# ---------------------------------------------------------------------------


def test_bates_3_2_sleepstudy_fm07_uncorrelated_ML():
    """fm07 <- lmer(Reaction ~ 1+Days + (1|Subject) + (0+Days|Subject),
                    sleepstudy, REML=FALSE)"""
    data = load_dataset("lme4", "sleepstudy")
    m = lme(
        "Reaction ~ 1 + Days + (1|Subject) + (0+Days|Subject)",
        data, REML=False,
    )

    assert m.n == 180
    assert m.n_groups == {"Subject": 18}
    _assert_ml_summary(
        m, AIC=1762.0, BIC=1778.0, loglike=-876.00,
        deviance=1752.0, df_resid=175, atol=0.1,
    )
    np.testing.assert_allclose(m.sigma, 25.5556, atol=5e-3)
    # lme4 lists the two scalar bars on Subject as two rows; we expose the
    # second one under the disambiguated key "Subject.1".
    _assert_re_scalar(m, "Subject", 24.1717)        # (Intercept)
    _assert_re_scalar(m, "Subject.1", 5.7986)        # Days
    _assert_fixed(m, "(Intercept)", 251.405, se=6.708, tval=37.48)
    _assert_fixed(m, "Days",         10.467, se=1.519, tval=6.89)


def test_bates_3_2_sleepstudy_fm06_correlated_ML():
    """fm06 <- lmer(Reaction ~ 1+Days + (1+Days|Subject),
                    sleepstudy, REML=FALSE)  -- correlated REs"""
    data = load_dataset("lme4", "sleepstudy")
    m = lme(
        "Reaction ~ 1 + Days + (1+Days|Subject)",
        data, REML=False,
    )

    _assert_ml_summary(
        m, AIC=1763.9393, BIC=1783.0971, loglike=-875.9697,
        deviance=1751.9393, df_resid=174, atol=0.1,
    )
    np.testing.assert_allclose(m.sigma, 25.5918, atol=5e-3)
    _assert_re_vector(m, "Subject", sds=[23.7803, 5.7168], corr=[0.0813])
    _assert_fixed(m, "(Intercept)", 251.405, se=6.632, tval=37.907)
    _assert_fixed(m, "Days",         10.467, se=1.502, tval=6.968)


def test_bates_3_2_sleepstudy_LRT_fm07_vs_fm06():
    """anova(fm07, fm06): test whether the (Intercept,Days) correlation
       is non-zero. Book: χ²=0.0639 on 1 df, p=0.8004."""
    data = load_dataset("lme4", "sleepstudy")
    fm06 = lme("Reaction ~ 1 + Days + (1+Days|Subject)", data, REML=False)
    fm07 = lme(
        "Reaction ~ 1 + Days + (1|Subject) + (0+Days|Subject)",
        data, REML=False,
    )
    chisq, df, p = _lrt(fm07, fm06)
    np.testing.assert_allclose(chisq, 0.0639, atol=5e-3)
    assert df == 1
    np.testing.assert_allclose(p, 0.8004, atol=5e-3)


# ---------------------------------------------------------------------------
# Ch 4: Building Linear Mixed Models
# ---------------------------------------------------------------------------


def test_bates_4_1_machines_fm10_ML():
    """fm10 <- lmer(score ~ Machine + (1|Worker) + (1|Machine:Worker),
                    Machines, REML=FALSE)"""
    data = load_dataset("nlme", "Machines")
    m = lme(
        "score ~ Machine + (1|Worker) + (1|Machine:Worker)",
        data, REML=False,
    )

    assert m.n == 54
    assert m.n_groups == {"Worker": 6, "Machine:Worker": 18}
    _assert_ml_summary(
        m, AIC=237.2694, BIC=249.2034, loglike=-112.6347,
        deviance=225.2694, df_resid=48,
    )
    np.testing.assert_allclose(m.sigma, 0.9616, atol=5e-3)
    _assert_re_scalar(m, "Machine:Worker", 3.3970)
    _assert_re_scalar(m, "Worker",         4.3645)
    _assert_fixed(m, "(Intercept)", 52.3556, se=2.2692, tval=23.07)
    _assert_fixed(m, "MachineB",      7.9667, se=1.9873, tval=4.009)
    _assert_fixed(m, "MachineC",     13.9167, se=1.9873, tval=7.003)


def test_bates_4_2_ergostool_fm16_ML():
    """fm16 <- lmer(effort ~ 1 + (1|Subject) + (1|Type),
                    ergoStool, REML=FALSE)"""
    data = load_dataset("nlme", "ergoStool")
    m = lme("effort ~ 1 + (1|Subject) + (1|Type)", data, REML=False)

    assert m.n == 36
    assert m.n_groups == {"Subject": 9, "Type": 4}
    _assert_ml_summary(
        m, AIC=144.0224, BIC=150.3564, loglike=-68.0112,
        deviance=136.0224, df_resid=32,
    )
    np.testing.assert_allclose(m.sigma, 1.101, atol=5e-3)
    _assert_re_scalar(m, "Subject", 1.305)
    _assert_re_scalar(m, "Type",    1.505)
    _assert_fixed(m, "(Intercept)", 10.25)


def test_bates_4_2_ergostool_fm17_ML():
    """fm17 <- lmer(effort ~ 1 + Type + (1|Subject), ergoStool, REML=FALSE)"""
    data = load_dataset("nlme", "ergoStool")
    m = lme("effort ~ 1 + Type + (1|Subject)", data, REML=False)

    _assert_ml_summary(
        m, AIC=134.1444, BIC=143.6456, loglike=-61.0722,
        deviance=122.1444, df_resid=30,
    )
    np.testing.assert_allclose(m.sigma, 1.037, atol=5e-3)
    _assert_re_scalar(m, "Subject", 1.256)
    _assert_fixed(m, "(Intercept)", 8.5556)
    _assert_fixed(m, "TypeT2",      3.8889)
    _assert_fixed(m, "TypeT3",      2.2222)
    _assert_fixed(m, "TypeT4",      0.6667)
