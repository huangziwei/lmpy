"""Model-comparison helpers: ``anova``, ``AIC``, ``BIC``.

Lives above ``lm`` and ``lme`` in the import graph so both can be compared
here without creating a cycle. ``lm.py`` and ``lme.py`` stay unaware of
each other.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2, f

from .lm import lm
from .lme import lme
from .utils import significance_code

__all__ = ["anova", "AIC", "BIC"]


def AIC(*models) -> None:
    """Print an AIC comparison table for one or more fitted models.

    Each model must expose ``.AIC``, ``.npar``, and ``.formula``.
    """
    rows = pd.DataFrame.from_dict({
        "formula": [m.formula for m in models],
        "df":      [m.npar    for m in models],
        "AIC":     [m.AIC     for m in models],
    }).set_index("formula")
    print(rows.to_string(formatters={"AIC": "{:.2f}".format}))


def BIC(*models) -> None:
    """Print a BIC comparison table for one or more fitted models.

    Each model must expose ``.BIC``, ``.npar``, and ``.formula``.
    """
    rows = pd.DataFrame.from_dict({
        "formula": [m.formula for m in models],
        "df":      [m.npar    for m in models],
        "BIC":     [m.BIC     for m in models],
    }).set_index("formula")
    print(rows.to_string(formatters={"BIC": "{:.2f}".format}))


def anova(m0, m1):
    """Compare two nested fits.

    - Two ``lm`` fits → F-test ANOVA table.
    - Two ``lme`` fits → likelihood-ratio test (lme4-style). REML fits are
      internally refit by ML before the LRT (the LRT statistic requires ML).
    """
    if isinstance(m0, lme) and isinstance(m1, lme):
        return _anova_lme(m0, m1)
    if isinstance(m0, lm) and isinstance(m1, lm):
        return _anova_lm(m0, m1)
    raise TypeError(
        "anova(): both models must be the same type (two lm or two lme)"
    )


def _anova_lm(m0, m1):
    """F-test ANOVA table comparing two nested ``lm`` fits."""
    docstring = "Analysis of Variance Table\n\n"
    for i, model in enumerate([m0, m1]):
        docstring += f"model {i}: {model.formula}\n"

    df0, df1 = m0.df_residuals, m1.df_residuals
    rss0, rss1 = m0.rss, m1.rss

    fstat = ((rss0 - rss1) / (df0 - df1)) / (rss1 / df1)
    f_p_value = f.sf(fstat, df0 - df1, df1)
    sig = significance_code([f_p_value])[0]

    df_model = ["", f"{df0 - df1:.0f}"]
    SoS = ["", f"{np.sum((m1.yhat.values - m1.y.values.mean())**2):.3f}"]

    df = pd.DataFrame.from_dict(
        {
            "Res.Df": [df0, df1],
            "RSS": [rss0, rss1],
            "Df": df_model,
            "Sum of Sq": SoS,
            "F": ["", f"{fstat:.3f}"],
            "Pr(>F)": ["", f"{f_p_value:.3f}"],
            " ": ["", sig],
        }
    )

    print(docstring)
    print(df.to_string(formatters={"RSS": "{:.3f}".format}))
    print("---")
    print("Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")


def _anova_lme(m0, m1):
    """Likelihood-ratio test for two nested ``lme`` fits (lme4-style)."""
    # LRT requires ML; silently refit any REML inputs.
    if m0.REML:
        m0 = lme(m0.formula, m0.data, REML=False)
    if m1.REML:
        m1 = lme(m1.formula, m1.data, REML=False)
    # Match lme4: sort so smaller-npar model prints first.
    if m0.npar > m1.npar:
        m0, m1 = m1, m0

    chisq = m0.deviance - m1.deviance
    df_d = m1.npar - m0.npar
    p = float(chi2.sf(chisq, df_d)) if df_d > 0 else float("nan")
    sig = significance_code([p])[0]

    docstring = "Analysis of Variance Table (likelihood ratio test)\n\n"
    docstring += f"model 0: {m0.formula}\n"
    docstring += f"model 1: {m1.formula}\n"

    df = pd.DataFrame.from_dict(
        {
            "npar":     [m0.npar, m1.npar],
            "AIC":      [f"{m0.AIC:.4f}", f"{m1.AIC:.4f}"],
            "BIC":      [f"{m0.BIC:.4f}", f"{m1.BIC:.4f}"],
            "logLik":   [f"{m0.loglike:.4f}", f"{m1.loglike:.4f}"],
            "deviance": [f"{m0.deviance:.4f}", f"{m1.deviance:.4f}"],
            "Chisq":    ["", f"{chisq:.4f}"],
            "Df":       ["", f"{df_d}"],
            "Pr(>Chisq)": ["", f"{p:.4g}"],
            " ":        ["", sig],
        }
    )
    df.index = ["model 0", "model 1"]

    print(docstring)
    print(df.to_string())
    print("---")
    print("Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
