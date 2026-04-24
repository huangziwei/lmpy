"""Model-comparison helpers: ``anova``, ``AIC``, ``BIC``.

Lives above ``lm`` and ``lme`` in the import graph so both can be compared
here without creating a cycle. ``lm.py`` and ``lme.py`` stay unaware of
each other.
"""

from __future__ import annotations

import polars as pl
from scipy.stats import chi2, f

from .lm import lm
from .lme import lme
from .utils import significance_code

__all__ = ["anova", "AIC", "BIC"]


def AIC(*models) -> None:
    """Print an AIC comparison table for one or more fitted models.

    Each model must expose ``.AIC``, ``.npar``, and ``.formula``.
    """
    rows = pl.DataFrame({
        "formula": [m.formula for m in models],
        "df":      [m.npar    for m in models],
        "AIC":     [f"{m.AIC:.2f}" for m in models],
    })
    with pl.Config(tbl_rows=-1, tbl_cols=-1):
        print(rows)


def BIC(*models) -> None:
    """Print a BIC comparison table for one or more fitted models.

    Each model must expose ``.BIC``, ``.npar``, and ``.formula``.
    """
    rows = pl.DataFrame({
        "formula": [m.formula for m in models],
        "df":      [m.npar    for m in models],
        "BIC":     [f"{m.BIC:.2f}" for m in models],
    })
    with pl.Config(tbl_rows=-1, tbl_cols=-1):
        print(rows)


def anova(*models):
    """Compare two or more nested fits.

    - All ``lm`` fits → F-test ANOVA table (incremental for 3+).
    - All ``lme`` fits → likelihood-ratio test (lme4-style, incremental
      for 3+). REML fits are internally refit by ML before the LRT.

    Rows are sorted by parameter count (smaller model first), matching
    R's ``anova``. Row labels ``model 0..N`` preserve *input* order so
    they remain traceable to the passed arguments.
    """
    if len(models) < 2:
        raise TypeError("anova(): need at least two models")
    if all(isinstance(m, lme) for m in models):
        return _anova_lme(*models)
    if all(isinstance(m, lm) for m in models):
        return _anova_lm(*models)
    raise TypeError("anova(): all models must be the same type (lm or lme)")


def _anova_lm(*models):
    """F-test ANOVA table comparing nested ``lm`` fits."""
    # Sort ascending by npar (= descending by df_residuals, matching R).
    order = sorted(range(len(models)), key=lambda i: models[i].df_residuals,
                   reverse=True)
    labels = [f"model {i}" for i in range(len(models))]

    dfs  = [models[i].df_residuals for i in order]
    rss  = [models[i].rss           for i in order]
    # R uses the largest (least-constrained) model's MSE as the F denom.
    mse_full = rss[-1] / dfs[-1]

    df_col, sos_col, f_col, p_col, sig_col = [""], [""], [""], [""], [""]
    for k in range(1, len(order)):
        d_df = dfs[k - 1] - dfs[k]
        d_rss = rss[k - 1] - rss[k]
        if d_df <= 0:
            df_col.append(f"{d_df:.0f}"); sos_col.append(f"{d_rss:.3f}")
            f_col.append(""); p_col.append(""); sig_col.append("")
            continue
        fstat = (d_rss / d_df) / mse_full
        p = float(f.sf(fstat, d_df, dfs[-1]))
        df_col.append(f"{d_df:.0f}")
        sos_col.append(f"{d_rss:.3f}")
        f_col.append(f"{fstat:.3f}")
        p_col.append(f"{p:.4g}")
        sig_col.append(significance_code([p])[0])

    docstring = "Analysis of Variance Table\n\n"
    for i, m in enumerate(models):
        docstring += f"{labels[i]}: {m.formula}\n"

    df_ = pl.DataFrame({
        "model":     [labels[i] for i in order],
        "Res.Df":    dfs,
        "RSS":       [f"{r:.3f}" for r in rss],
        "Df":        df_col,
        "Sum of Sq": sos_col,
        "F":         f_col,
        "Pr(>F)":    p_col,
        " ":         sig_col,
    })

    print(docstring)
    with pl.Config(tbl_rows=-1, tbl_cols=-1):
        print(df_)
    print("---")
    print("Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")


def _anova_lme(*models):
    """Likelihood-ratio test for nested ``lme`` fits (lme4-style)."""
    # LRT requires ML; silently refit any REML inputs.
    models = tuple(
        (lme(m.formula, m.data, REML=False) if m.REML else m) for m in models
    )
    # Sort ascending by npar, preserving original indices for row labels.
    order = sorted(range(len(models)), key=lambda i: models[i].npar)
    labels = [f"model {i}" for i in range(len(models))]

    npar_col, aic_col, bic_col, ll_col, dev_col = [], [], [], [], []
    chi_col, dfc_col, p_col, sig_col = [], [], [], []
    for k, idx in enumerate(order):
        m = models[idx]
        npar_col.append(m.npar)
        aic_col.append(f"{m.AIC:.4f}")
        bic_col.append(f"{m.BIC:.4f}")
        ll_col.append(f"{m.loglike:.4f}")
        dev_col.append(f"{m.deviance:.4f}")
        if k == 0:
            chi_col.append(""); dfc_col.append(""); p_col.append(""); sig_col.append("")
            continue
        prev = models[order[k - 1]]
        chisq = prev.deviance - m.deviance
        d_df = m.npar - prev.npar
        p = float(chi2.sf(chisq, d_df)) if d_df > 0 else float("nan")
        chi_col.append(f"{chisq:.4f}")
        dfc_col.append(f"{d_df}")
        p_col.append(f"{p:.4g}")
        sig_col.append(significance_code([p])[0])

    docstring = "Analysis of Variance Table (likelihood ratio test)\n\n"
    for i, m in enumerate(models):
        docstring += f"{labels[i]}: {m.formula}\n"

    df_ = pl.DataFrame({
        "model":      [labels[i] for i in order],
        "npar":       npar_col,
        "AIC":        aic_col,
        "BIC":        bic_col,
        "logLik":     ll_col,
        "deviance":   dev_col,
        "Chisq":      chi_col,
        "Df":         dfc_col,
        "Pr(>Chisq)": p_col,
        " ":          sig_col,
    })

    print(docstring)
    with pl.Config(tbl_rows=-1, tbl_cols=-1):
        print(df_)
    print("---")
    print("Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
