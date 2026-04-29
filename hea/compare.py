"""Model-comparison helpers: ``anova``, ``AIC``, ``BIC``.

Lives above ``lm`` and ``lme`` in the import graph so both can be compared
here without creating a cycle. ``lm.py`` and ``lme.py`` stay unaware of
each other.
"""

from __future__ import annotations

import inspect

import numpy as np
import polars as pl
from scipy.stats import chi2, f

from .gam import gam
from .glm import glm
from .lm import lm
from .lme import lme
from .utils import _dig_tst, format_df, format_pval, format_signif, significance_code

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


def anova(*models, test: str | None = None):
    """Compare nested fits, or decompose a single fit by Type-I SS.

    - One ``lm`` → sequential (Type I) ANOVA table, splitting the model's
      total SS into incremental contributions per RHS term in formula
      order. Mirrors R's ``anova(m)`` for a single ``lm``.
    - Multiple ``lm`` fits → F-test ANOVA table (incremental for 3+).
    - Multiple ``glm`` fits → analysis-of-deviance table (incremental for
      3+); ``test=`` selects the test statistic (see below).
    - Multiple ``lme`` fits → likelihood-ratio test (lme4-style, incremental
      for 3+). REML fits are internally refit by ML before the LRT.

    Parameters
    ----------
    test : {"Chisq", "LRT", "F", "Rao", None}, optional
        Only meaningful for ``glm`` comparisons. ``None`` (default) auto-
        picks ``"Chisq"`` for scale-known families (Poisson, Binomial) and
        ``"F"`` for unknown-scale (Gaussian, Gamma, IG), matching R's
        ``anova.glm`` recommendation. ``"LRT"`` is an alias for ``"Chisq"``.
        ``"Rao"`` (score test) is not implemented yet. For ``lm`` and ``lme``
        the test is fixed (always F / Chisq LRT respectively); passing
        ``test=`` for those raises.

    For multi-model calls rows are sorted by parameter count (smaller
    model first), matching R's ``anova``. Row labels are recovered from
    the caller's variable names (R-style); falls back to ``model i`` for
    unbound or aliased arguments, preserving *input* order.
    """
    if len(models) == 0:
        raise TypeError("anova(): need at least one model")
    if len(models) == 1:
        m = models[0]
        if isinstance(m, gam):
            if test is not None:
                raise TypeError("anova(gam): test= is not accepted")
            return _anova_gam_single(m)
        if isinstance(m, lm) and not isinstance(m, glm):
            if test is not None:
                raise TypeError("anova(lm): test= is not accepted (always F)")
            return _anova_lm_single(m)
        raise TypeError(
            "anova(m): single-model form supports lm and gam only "
            f"(got {type(m).__name__})"
        )
    labels = _caller_names(models, inspect.currentframe().f_back)
    if all(isinstance(m, lme) for m in models):
        if test is not None and test.upper() not in ("CHISQ", "LRT"):
            raise ValueError(
                f"anova(lme): only test='Chisq'/'LRT' (the default LRT) "
                f"is supported, got {test!r}"
            )
        return _anova_lme(*models, labels=labels)
    # glm before lm: glm is not an lm subclass, but the isinstance order
    # would still matter if it ever became one. Keep the explicit branch.
    if all(isinstance(m, glm) for m in models):
        return _anova_glm(*models, labels=labels, test=test)
    if all(isinstance(m, lm) for m in models):
        if test is not None:
            raise TypeError("anova(lm): test= is not accepted (always F)")
        return _anova_lm(*models, labels=labels)
    raise TypeError("anova(): all models must be the same type (lm, glm, or lme)")


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


def _anova_lm_single(m: lm):
    """Sequential (Type I) ANOVA — R's ``anova.lm(m)`` for a single fit.

    Refits the model with terms added one at a time in formula order,
    attributing each step's drop in RSS to that term. F = MS_term /
    MS_residual_full, p = upper-tail F. R uses QR-incremental SS, which
    is bit-equivalent for full-rank designs; refitting is conceptually
    simpler and reuses hea's existing rank-deficiency handling.
    """
    terms = m._expanded.terms
    if not terms:
        raise TypeError(
            "anova(m): single-model form needs at least one RHS term "
            "(got an intercept-only model)"
        )

    lhs = m.formula.split("~", 1)[0].strip()
    intercept_str = "1" if m._expanded.intercept else "0"

    def cumulative_formula(k: int) -> str:
        if k == 0:
            return f"{lhs} ~ {intercept_str}"
        rhs = " + ".join(t.label for t in terms[:k])
        return f"{lhs} ~ {intercept_str} + {rhs}"

    rss_chain: list[float] = []
    df_chain: list[int] = []
    for k in range(len(terms)):
        m_k = lm(cumulative_formula(k), m.data,
                 weights=m.weights, method=m.method)
        rss_chain.append(m_k.rss)
        df_chain.append(m_k.df_residuals)
    # Last entry = the original full model — reuse its values directly to
    # avoid a redundant refit and any floating-point drift from re-solving.
    rss_chain.append(m.rss)
    df_chain.append(m.df_residuals)

    mse_full = m.rss / m.df_residuals

    df_col: list[int] = []
    sos_col: list[float] = []
    ms_col: list[float] = []
    f_col: list[float | None] = []
    p_col: list[float | None] = []
    sig_col: list[str] = []
    for i, t in enumerate(terms):
        d_df = df_chain[i] - df_chain[i + 1]
        d_rss = rss_chain[i] - rss_chain[i + 1]
        if d_df <= 0:
            df_col.append(d_df); sos_col.append(round(d_rss, 4))
            ms_col.append(float("nan"))
            f_col.append(None); p_col.append(None); sig_col.append("")
            continue
        ms = d_rss / d_df
        fstat = ms / mse_full
        p = float(f.sf(fstat, d_df, m.df_residuals))
        df_col.append(d_df); sos_col.append(round(d_rss, 4))
        ms_col.append(round(ms, 4))
        f_col.append(round(fstat, 4))
        p_col.append(float(f"{p:.4g}"))
        sig_col.append(significance_code([p])[0])
    # Residuals row
    df_col.append(m.df_residuals); sos_col.append(round(m.rss, 4))
    ms_col.append(round(mse_full, 4))
    f_col.append(None); p_col.append(None); sig_col.append("")

    docstring = "Analysis of Variance Table\n\n"
    docstring += f"Response: {lhs}\n"

    df_ = pl.DataFrame({
        "":         [t.label for t in terms] + ["Residuals"],
        "Df":       df_col,
        "Sum Sq":   sos_col,
        "Mean Sq":  ms_col,
        "F value":  f_col,
        "Pr(>F)":   p_col,
        " ":        sig_col,
    })

    print(docstring)
    print(format_df(df_))
    print("---")
    print("Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")


def _anova_gam_single(m: gam):
    """``anova.gam``-style single-model output: parametric Terms F-table
    plus the smooth significance table. Mirrors mgcv's ``anova.gam`` for
    a single fit (which omits the lm-coefficient details that
    ``summary.gam`` prints).

    Per-term parametric F is the joint Wald test
    ``F = β_t' Vp_t⁻¹ β_t / k_t``, with ``k_t`` = number of model-matrix
    columns the term contributes (1 for a numeric, ``L−1`` for an
    ``L``-level factor). Term→column mapping is by name prefix — works
    for the common factor / numeric / simple-interaction cases.
    """
    digits = 4

    out = []
    out.append("")
    out.append(f"Family: {m.family.name}")
    out.append(f"Link function: {m.family.link.name}")
    out.append("")
    out.append(f"Formula: {m.formula}")
    out.append("")

    # ---- Parametric Terms (per-term joint Wald F) -----------------------
    # Collect non-intercept parametric columns. Intercept is excluded — mgcv
    # follows the same convention in anova.gam's pTerms.table.
    cols = m.parametric_columns
    col_idx = {c: i for i, c in enumerate(cols)}
    used = {"(Intercept)"} if "(Intercept)" in col_idx else set()

    rows: list[tuple[str, int, float, float]] = []
    if m._expanded.terms:
        for term in m._expanded.terms:
            label = term.label
            # Match: a column belongs to ``term`` if it equals the label
            # exactly (numeric term) or starts with the label (factor /
            # interaction). Pick the longest label match per column to
            # avoid e.g. ``Hclass`` claiming ``Hclassmedium:Girth`` when
            # the interaction term ``Hclass:Girth`` exists.
            term_cols = [
                c for c in cols
                if c not in used and (c == label or c.startswith(label))
            ]
            if not term_cols:
                continue
            used.update(term_cols)
            idx = np.array([col_idx[c] for c in term_cols], dtype=int)
            beta_t = m._beta[idx]
            Vp_t = m.Vp[np.ix_(idx, idx)]
            k = len(idx)
            try:
                solved = np.linalg.solve(Vp_t, beta_t)
                F_stat = float(beta_t @ solved) / k
            except np.linalg.LinAlgError:
                F_stat = float("nan")
            df_resid = float(m.df_residuals)
            if np.isfinite(F_stat) and df_resid > 0:
                p_val = float(f.sf(F_stat, k, df_resid))
            else:
                p_val = float("nan")
            rows.append((label, k, F_stat, p_val))

    if rows:
        sig = significance_code([r[3] for r in rows])
        tbl = pl.DataFrame({
            "":        [r[0] for r in rows],
            "df":      [r[1] for r in rows],
            "F":       format_signif([r[2] for r in rows], digits=digits),
            "p-value": format_pval([r[3] for r in rows],
                                   digits=_dig_tst(digits)),
            " ":       sig,
        })
        out.append("Parametric Terms:")
        out.append(format_df(
            tbl,
            align={c: "right" for c in ("df", "F", "p-value")},
        ))
        out.append("---")
        out.append(
            "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1"
        )
        out.append("")

    # ---- Smooth significance table (same logic as gam.summary) ----------
    if m._blocks:
        from scipy.stats import f as _f_dist
        rows_label: list[str] = []
        rows_edf: list[float] = []
        rows_refdf: list[float] = []
        rows_F: list[float] = []
        rows_p: list[float] = []
        for b, (a, bcol) in zip(m._blocks, m._block_col_ranges):
            beta_b = m._beta[a:bcol]
            Vp_b = m.Vp[a:bcol, a:bcol]
            X_b = m._X_full[:, a:bcol]
            edf_b = float(m.edf[a:bcol].sum())
            edf1_b = (
                float(m.edf1[a:bcol].sum())
                if hasattr(m, "edf1") else edf_b
            )
            p_b = bcol - a
            rank = float(min(p_b, edf1_b))
            Tr, ref_df = m._test_stat_type0(X_b, Vp_b, beta_b, rank)
            F_smooth = Tr / max(ref_df, 1e-8)
            p_smooth = (
                float(_f_dist.sf(F_smooth, ref_df, m.df_residuals))
                if m.df_residuals > 0 else float("nan")
            )
            rows_label.append(b.label)
            rows_edf.append(edf_b)
            rows_refdf.append(edf1_b)
            rows_F.append(F_smooth)
            rows_p.append(p_smooth)
        sig_smooth = significance_code(rows_p)
        sm_tbl = pl.DataFrame({
            "":        rows_label,
            "edf":     format_signif(rows_edf, digits=digits),
            "Ref.df":  format_signif(rows_refdf, digits=digits),
            "F":       format_signif(rows_F, digits=digits),
            "p-value": format_pval(rows_p, digits=_dig_tst(digits)),
            " ":       sig_smooth,
        })
        out.append("Approximate significance of smooth terms:")
        out.append(format_df(
            sm_tbl,
            align={c: "right" for c in ("edf", "Ref.df", "F", "p-value")},
        ))
        out.append("---")
        out.append(
            "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1"
        )

    print("\n".join(out))


def _anova_glm(*models, labels: list[str], test: str | None = None):
    """``anova.glm``-style deviance table for nested ``glm`` fits.

    With ``test=None`` we auto-pick (matches R's recommendation):
    - scale-known families (Poisson, Binomial) → ``Chisq`` LRT on Δdev.
    - unknown-scale families (Gaussian, Gamma, IG) → ``F``.

    Override via ``test=``:
    - ``"Chisq"`` / ``"LRT"`` (alias) → ``Δdev / dispersion_full ~ χ²(Δdf)``.
      For scale-known families ``dispersion_full = 1`` so this is just Δdev,
      matching the auto-pick. For unknown-scale, the division is the
      asymptotic chi-square test (R's ``anova.glm`` does the same).
    - ``"F"`` → ``F = (Δdev / Δdf) / dispersion_full`` against ``F(Δdf,
      df_residual_full)``. Allowed for scale-known families too (R does)
      though the chi-square version is preferred.
    - ``"Rao"`` → score test, not implemented yet.

    Three-or-more models are walked incrementally (row k vs row k-1 after
    sorting by ``df_residuals`` descending, matching ``_anova_lm``).
    """
    df_, docstring = _anova_glm_table(*models, labels=labels, test=test)
    print(docstring)
    print(format_df(df_))
    print("---")
    print("Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")


def _anova_glm_table(*models, labels: list[str], test: str | None = None):
    """Pure builder for the ``anova(glm,...)`` table.

    Returns ``(df, docstring)``. Used by ``_anova_glm`` (which prints) and
    by tests that need to inspect column values directly. See ``_anova_glm``
    for the semantics of ``test=``.
    """
    fam0 = models[0].family
    if not all(type(m.family) is type(fam0) and
               m.family.link.name == fam0.link.name for m in models):
        raise ValueError("anova(): all glm fits must share family and link")

    if test is None:
        test = "Chisq" if fam0.scale_known else "F"
    else:
        t_norm = test.upper()
        if t_norm == "LRT":
            test = "Chisq"
        elif t_norm == "RAO":
            raise NotImplementedError(
                "anova(glm, test='Rao'): score test not implemented yet"
            )
        elif t_norm == "CHISQ":
            test = "Chisq"
        elif t_norm == "F":
            test = "F"
        else:
            raise ValueError(
                f"anova(glm): test must be 'Chisq', 'LRT', 'F', 'Rao', or None; "
                f"got {test!r}"
            )

    # Sort ascending by npar (= descending by df_residuals), matching R.
    order = sorted(range(len(models)), key=lambda i: models[i].df_residuals,
                   reverse=True)
    dfs = [models[i].df_residual for i in order]
    devs = [models[i].deviance for i in order]
    full = models[order[-1]]
    disp_full = float(full.dispersion)
    df_full = int(full.df_residual)

    df_col: list[int | None] = [None]
    dev_col: list[float | None] = [None]
    stat_col: list[float | None] = [None]
    p_col: list[float | None] = [None]
    sig_col: list[str] = [""]
    for k in range(1, len(order)):
        d_df = dfs[k - 1] - dfs[k]
        d_dev = devs[k - 1] - devs[k]
        if d_df <= 0:
            df_col.append(d_df); dev_col.append(round(d_dev, 4))
            stat_col.append(None); p_col.append(None); sig_col.append("")
            continue
        if test == "Chisq":
            # disp_full == 1 for scale-known families (Poisson/Binomial),
            # so this matches the canonical LRT there. For unknown-scale
            # it's the asymptotic χ² test on the rescaled deviance — same
            # formula R uses when `test="Chisq"` is passed for Gaussian/
            # Gamma/IG fits.
            stat = d_dev / disp_full
            p = float(chi2.sf(stat, d_df))
        else:
            stat = (d_dev / d_df) / disp_full
            p = float(f.sf(stat, d_df, df_full))
        df_col.append(d_df)
        dev_col.append(round(d_dev, 4))
        stat_col.append(round(stat, 4))
        p_col.append(float(f"{p:.4g}"))
        sig_col.append(significance_code([p])[0])

    docstring = "Analysis of Deviance Table\n\n"
    for i, m in enumerate(models):
        docstring += f"{labels[i]}: {m.formula}\n"

    stat_lbl = "F" if test == "F" else "Deviance"
    p_lbl = "Pr(>F)" if test == "F" else "Pr(>Chi)"

    df_ = pl.DataFrame({
        "":           [labels[i] for i in order],
        "Resid. Df":  dfs,
        "Resid. Dev": [round(d, 4) for d in devs],
        "Df":         df_col,
        "Deviance":   dev_col,
        stat_lbl:     stat_col,
        p_lbl:        p_col,
        " ":          sig_col,
    })
    return df_, docstring


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
