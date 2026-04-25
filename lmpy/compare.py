"""Model-comparison helpers: ``anova``, ``AIC``, ``BIC``.

Lives above ``lm`` and ``lme`` in the import graph so both can be compared
here without creating a cycle. ``lm.py`` and ``lme.py`` stay unaware of
each other.
"""

from __future__ import annotations

import inspect

import polars as pl
from scipy.stats import chi2, f

from .lm import lm
from .lme import lme
from .utils import format_df, significance_code

__all__ = ["anova", "AIC", "BIC"]


def _caller_names(models, frame, fallback: str = "model") -> list[str]:
    """Recover caller-bound variable names for ``models``, like R's
    ``match.call``. Walks ``frame``'s locals + globals; falls back to
    ``f"{fallback} {i}"`` when a model has no unique binding (e.g.
    passed as an expression or aliased to multiple names).
    """
    if frame is None:
        return [f"{fallback} {i}" for i in range(len(models))]
    scope = {**frame.f_globals, **frame.f_locals}
    by_id: dict[int, list[str]] = {}
    for name, val in scope.items():
        if name.startswith("_"):
            continue
        by_id.setdefault(id(val), []).append(name)
    out = []
    for i, m in enumerate(models):
        names = by_id.get(id(m), [])
        out.append(names[0] if len(names) == 1 else f"{fallback} {i}")
    return out


def AIC(*models) -> None:
    """Print an AIC comparison table for one or more fitted models.

    Each model must expose ``.AIC`` and ``.npar``. Row labels are
    recovered from the caller's variable names (R-style); falls back
    to ``model i`` for unbound or aliased arguments.
    """
    names = _caller_names(models, inspect.currentframe().f_back)
    rows = pl.DataFrame({
        "":    names,
        "df":  [m.npar    for m in models],
        "AIC": [round(m.AIC, 2) for m in models],
    })
    print(format_df(rows))


def BIC(*models) -> None:
    """Print a BIC comparison table for one or more fitted models.

    Each model must expose ``.BIC`` and ``.npar``. Row labels are
    recovered from the caller's variable names (R-style); falls back
    to ``model i`` for unbound or aliased arguments.
    """
    names = _caller_names(models, inspect.currentframe().f_back)
    rows = pl.DataFrame({
        "":    names,
        "df":  [m.npar    for m in models],
        "BIC": [round(m.BIC, 2) for m in models],
    })
    print(format_df(rows))


def anova(*models):
    """Compare two or more nested fits.

    - All ``lm`` fits → F-test ANOVA table (incremental for 3+).
    - All ``lme`` fits → likelihood-ratio test (lme4-style, incremental
      for 3+). REML fits are internally refit by ML before the LRT.

    Rows are sorted by parameter count (smaller model first), matching
    R's ``anova``. Row labels are recovered from the caller's variable
    names (R-style); falls back to ``model i`` for unbound or aliased
    arguments, preserving *input* order.
    """
    if len(models) < 2:
        raise TypeError("anova(): need at least two models")
    labels = _caller_names(models, inspect.currentframe().f_back)
    if all(isinstance(m, lme) for m in models):
        return _anova_lme(*models, labels=labels)
    if all(isinstance(m, lm) for m in models):
        return _anova_lm(*models, labels=labels)
    raise TypeError("anova(): all models must be the same type (lm or lme)")


def _anova_lm(*models, labels: list[str]):
    """F-test ANOVA table comparing nested ``lm`` fits."""
    # Sort ascending by npar (= descending by df_residuals, matching R).
    order = sorted(range(len(models)), key=lambda i: models[i].df_residuals,
                   reverse=True)

    dfs  = [models[i].df_residuals for i in order]
    rss  = [models[i].rss           for i in order]
    # R uses the largest (least-constrained) model's MSE as the F denom.
    mse_full = rss[-1] / dfs[-1]

    df_col: list[int | None] = [None]
    sos_col: list[float | None] = [None]
    f_col: list[float | None] = [None]
    p_col: list[float | None] = [None]
    sig_col: list[str] = [""]
    for k in range(1, len(order)):
        d_df = dfs[k - 1] - dfs[k]
        d_rss = rss[k - 1] - rss[k]
        if d_df <= 0:
            df_col.append(d_df); sos_col.append(round(d_rss, 3))
            f_col.append(None); p_col.append(None); sig_col.append("")
            continue
        fstat = (d_rss / d_df) / mse_full
        p = float(f.sf(fstat, d_df, dfs[-1]))
        df_col.append(d_df)
        sos_col.append(round(d_rss, 3))
        f_col.append(round(fstat, 3))
        p_col.append(float(f"{p:.4g}"))
        sig_col.append(significance_code([p])[0])

    docstring = "Analysis of Variance Table\n\n"
    for i, m in enumerate(models):
        docstring += f"{labels[i]}: {m.formula}\n"

    df_ = pl.DataFrame({
        "":          [labels[i] for i in order],
        "Res.Df":    dfs,
        "RSS":       [round(r, 3) for r in rss],
        "Df":        df_col,
        "Sum of Sq": sos_col,
        "F":         f_col,
        "Pr(>F)":    p_col,
        " ":         sig_col,
    })

    print(docstring)
    print(format_df(df_))
    print("---")
    print("Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")


def _anova_lme(*models, labels: list[str]):
    """Likelihood-ratio test for nested ``lme`` fits (lme4-style)."""
    # LRT requires ML; silently refit any REML inputs.
    refit = any(m.REML for m in models)
    models = tuple(
        (lme(m.formula, m.data, REML=False) if m.REML else m) for m in models
    )
    if refit:
        print("refitting model(s) with ML (instead of REML)")
    # Sort ascending by npar, preserving original indices for row labels.
    order = sorted(range(len(models)), key=lambda i: models[i].npar)

    npar_col: list[int] = []
    aic_col: list[float] = []
    bic_col: list[float] = []
    ll_col: list[float] = []
    dev_col: list[float] = []
    chi_col: list[float | None] = []
    dfc_col: list[int | None] = []
    p_col: list[float | None] = []
    sig_col: list[str] = []
    for k, idx in enumerate(order):
        m = models[idx]
        npar_col.append(m.npar)
        aic_col.append(round(m.AIC, 4))
        bic_col.append(round(m.BIC, 4))
        ll_col.append(round(m.loglike, 4))
        dev_col.append(round(m.deviance, 4))
        if k == 0:
            chi_col.append(None); dfc_col.append(None); p_col.append(None); sig_col.append("")
            continue
        prev = models[order[k - 1]]
        chisq = prev.deviance - m.deviance
        d_df = m.npar - prev.npar
        p = float(chi2.sf(chisq, d_df)) if d_df > 0 else float("nan")
        chi_col.append(round(chisq, 4))
        dfc_col.append(d_df)
        p_col.append(float(f"{p:.4g}"))
        sig_col.append(significance_code([p])[0])

    docstring = "Analysis of Variance Table (likelihood ratio test)\n\n"
    for i, m in enumerate(models):
        docstring += f"{labels[i]}: {m.formula}\n"

    df_ = pl.DataFrame({
        "":           [labels[i] for i in order],
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
    print(format_df(df_))
    print("---")
    print("Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
