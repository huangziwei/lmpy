"""Faraway-favorite single-purpose plotters: ``qqnorm``, ``halfnorm``,
``termplot``.

Unlike the ``plot()`` dispatch, these are direct entry points (R doesn't
S3-dispatch through them either). They return an ``Axes`` (or list, for
multi-term ``termplot``) so callers can chain ``qqline``/``points`` etc.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from ._util import draw_points


def qqnorm(
    x,
    *,
    ax=None,
    ylab: str = "Sample Quantiles",
    main: str = "Normal Q-Q",
    pch=None,
):
    """Standard-normal Q-Q scatter — pair with ``qqline(x, ax=ax)`` for the
    reference line. Mirrors R's ``stats::qqnorm``."""
    if ax is None:
        _, ax = plt.subplots()
    vals = np.asarray(x, dtype=float)
    vals = vals[~np.isnan(vals)]
    n = len(vals)
    if n < 2:
        return ax
    sorted_vals = np.sort(vals)
    probs = (np.arange(1, n + 1) - 0.5) / n
    q = norm.ppf(probs)
    draw_points(ax, q, sorted_vals, pch=pch)
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel(ylab)
    if main:
        ax.set_title(main)
    return ax


def halfnorm(
    x,
    nlab: int = 2,
    labs=None,
    *,
    ax=None,
    ylab: str = "Sorted Data",
):
    """Half-normal Q-Q — port of ``faraway::halfnorm``.

    Plots ``sort(|x|)`` against the half-normal theoretical quantiles
    ``qnorm((n + 1:n) / (2n + 1))``. The ``nlab`` largest values are
    labeled with strings from ``labs`` (defaults to 1-based integer index).
    Used in the book to flag high-leverage points, large Cook's distances,
    or top absolute coefficients.

    The ``nlab`` parameter is positional (matches Faraway: ``halfnorm(cook, 3, labs=...)``).
    """
    if ax is None:
        _, ax = plt.subplots()

    abs_x = np.abs(np.asarray(x, dtype=float))
    n = len(abs_x)
    if labs is None:
        labels = [str(i + 1) for i in range(n)]
    else:
        labels = [str(s) for s in labs]
    if len(labels) != n:
        raise ValueError(
            f"halfnorm(): labs has {len(labels)} entries but x has {n}"
        )

    sort_idx = np.argsort(abs_x)
    sorted_vals = abs_x[sort_idx]
    sorted_labs = [labels[i] for i in sort_idx]
    ui = norm.ppf((n + np.arange(1, n + 1)) / (2 * n + 1))

    nlab_eff = max(0, min(int(nlab), n))
    if nlab_eff < n:
        draw_points(ax, ui[: n - nlab_eff], sorted_vals[: n - nlab_eff])
    for i in range(n - nlab_eff, n):
        ax.annotate(
            sorted_labs[i],
            (ui[i], sorted_vals[i]),
            fontsize=8,
            color="black",
            xytext=(3, 3),
            textcoords="offset points",
        )
    ax.set_xlabel("Half-normal quantiles")
    ax.set_ylabel(ylab)
    if abs_x.size:
        ax.set_ylim(0, max(abs_x) * 1.05)
    return ax


def termplot(lmod, *, partial_resid: bool = False, terms=None, ax=None):
    """Per-term effect plot for a fitted ``lm``/``glm``.

    Each term is shown as ``b_j * (x_j - mean(x_j))`` — the centered
    contribution, matching R's ``predict(model, type="terms")`` convention.
    With ``partial_resid=True``, scatter overlays partial residuals
    ``e + b_j * (x_j - mean(x_j))``.

    ``terms`` selects a single term by 1-based int (R style) or by name,
    or a list of either. ``None`` plots every numeric main-effect term.

    v1: numeric main-effect terms only — factor dummies (0/1 columns)
    error out, since they collapse a categorical to a single level
    indicator and need a different presentation.
    """
    if not (hasattr(lmod, "_bhat_arr") and hasattr(lmod, "feature_names")):
        raise TypeError("termplot(): expected an lm/glm fit object")

    feats = list(lmod.feature_names)
    if terms is None:
        idx_list = list(range(len(feats)))
    else:
        as_list = terms if isinstance(terms, (list, tuple)) else [terms]
        idx_list = []
        for t in as_list:
            if isinstance(t, (int, np.integer)):
                idx_list.append(int(t) - 1)
            elif isinstance(t, str):
                if t not in feats:
                    raise ValueError(f"termplot(): term {t!r} not in {feats}")
                idx_list.append(feats.index(t))
            else:
                raise TypeError(f"termplot(terms=): unexpected entry {t!r}")

    n_terms = len(idx_list)
    if n_terms == 0:
        raise ValueError("termplot(): no terms selected")

    if ax is None:
        if n_terms == 1:
            _, only = plt.subplots(figsize=(5, 4))
            axes = [only]
        else:
            fig, ax_arr = plt.subplots(1, n_terms, figsize=(4 * n_terms, 3))
            axes = list(np.atleast_1d(ax_arr).flatten())
            fig.tight_layout()
    else:
        if n_terms > 1:
            raise ValueError(
                "termplot(ax=): pass a single ax only when selecting one term"
            )
        axes = [ax]

    bhat = np.asarray(lmod._bhat_arr).ravel()
    column_names = list(lmod.column_names)
    residuals = np.asarray(lmod._residuals_arr).ravel()

    for k, idx in enumerate(idx_list):
        feature = feats[idx]
        col_idx = column_names.index(feature)
        b = float(bhat[col_idx])
        x_vals = lmod.X[feature].to_numpy().astype(float)

        uniq = np.unique(x_vals)
        if set(uniq.tolist()).issubset({0.0, 1.0}):
            raise ValueError(
                f"termplot(): term {feature!r} looks like a factor dummy "
                f"(values {uniq.tolist()}); v1 supports continuous terms only"
            )

        x_centered = x_vals - x_vals.mean()
        line_y = b * x_centered
        order = np.argsort(x_vals)
        a = axes[k]
        if partial_resid:
            draw_points(a, x_vals, residuals + b * x_centered)
        a.plot(x_vals[order], line_y[order], color="black")
        a.set_xlabel(feature)
        a.set_ylabel(f"Partial for {feature}")

    return axes[0] if n_terms == 1 else axes
