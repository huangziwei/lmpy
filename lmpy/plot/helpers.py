"""Faraway-favorite single-purpose plotters: ``qqnorm``, ``halfnorm``,
``termplot``, ``pairs``.

Unlike the ``plot()`` dispatch, these are direct entry points (R doesn't
S3-dispatch through them either, except ``pairs`` which is reached as
``plot.data.frame``). They return an ``Axes`` (or array, for multi-term
``termplot`` and ``pairs``) so callers can chain ``qqline``/``points`` etc.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
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


def pairs(
    data,
    *,
    cols=None,
    diag: str = "label",
    labels=None,
    pch=None,
    cex: float | None = None,
    col=None,
    main: str | None = None,
    figsize: tuple[float, float] | None = None,
):
    """Scatterplot matrix — port of R's ``graphics::pairs.data.frame``.

    All pairs of selected columns are plotted in an n×n grid. Cell at
    row i / column j shows ``cols[i]`` on the y-axis vs ``cols[j]`` on
    the x-axis (R's convention). The diagonal shows column names by
    default; ``diag="hist"`` draws per-column histograms, ``diag="none"``
    leaves the diagonal blank.

    Tick labels alternate around the perimeter — top row at odd-index
    columns, bottom row at even-index columns, left column at odd-index
    rows, right column at even-index rows (0-indexed). This is R's
    ``pairs.default`` behavior: every variable's scale appears exactly
    once, on whichever side keeps the matrix readable, and inner cells
    stay clean.

    Parameters
    ----------
    data : pl.DataFrame
    cols : list[str] | None
        Columns to include. Defaults to every numeric column in ``data``.
    diag : {"label", "hist", "none"}
    labels : list[str] | None
        Override displayed names on the diagonal — must match ``len(cols)``.
        Useful when the raw column name is too long or contains LaTeX.
    pch, cex, col
        Per-cell point styling, forwarded to ``draw_points``.
    main : str | None
        Figure-level title.
    figsize : (w, h) | None
        Matplotlib figure size in inches. Defaults to ``(2*n, 2*n)`` —
        scales with the number of columns so cells stay roughly square.

    Returns
    -------
    np.ndarray
        The 2-D grid of matplotlib ``Axes``.
    """
    if not isinstance(data, pl.DataFrame):
        raise TypeError(
            f"pairs(): expected a polars DataFrame, got {type(data).__name__}"
        )
    if diag not in ("label", "hist", "none"):
        raise ValueError(
            f"pairs(): diag= must be 'label', 'hist', or 'none', got {diag!r}"
        )

    if cols is None:
        cols = [c for c in data.columns if data[c].dtype.is_numeric()]
    else:
        cols = list(cols)

    n = len(cols)
    if n < 2:
        raise ValueError(f"pairs(): need at least 2 columns, got {n}")

    if labels is None:
        labels = list(cols)
    elif len(labels) != n:
        raise ValueError(
            f"pairs(): labels has {len(labels)} entries but {n} columns selected"
        )

    fig, axes = plt.subplots(n, n, figsize=figsize or (2 * n, 2 * n), squeeze=False)
    arrs = {c: data[c].cast(pl.Float64).to_numpy() for c in cols}

    # Pin per-variable axis limits (with a small matplotlib-style pad) and
    # apply them to every cell — without this, the diagonal sets tight
    # limits while off-diagonals auto-scale with margin, so two cells in
    # the same column end up with mismatched tick intervals. Can't use
    # sharex='col'/sharey='row' instead because diag="hist" needs its own
    # count y-axis, which sharing would clobber.
    def _padded(a, frac=0.04):
        lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
            return (lo - 1.0, hi + 1.0) if np.isfinite(lo) else (-1.0, 1.0)
        pad = frac * (hi - lo)
        return lo - pad, hi + pad
    ranges = {c: _padded(arrs[c]) for c in cols}

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            x_arr = arrs[cols[j]]
            y_arr = arrs[cols[i]]
            if i == j:
                if diag == "hist":
                    ax.hist(x_arr, color="lightgray", edgecolor="black")
                elif diag == "label":
                    ax.text(0.5, 0.5, labels[i], ha="center", va="center",
                            transform=ax.transAxes, fontsize=12)
            else:
                draw_points(ax, x_arr, y_arr, pch=pch, cex=cex, col=col)

            ax.set_xlim(*ranges[cols[j]])
            if not (i == j and diag == "hist"):
                ax.set_ylim(*ranges[cols[i]])

            # R's pairs.default perimeter tick rule (0-indexed):
            top_lab = (i == 0 and j % 2 == 1)
            bot_lab = (i == n - 1 and j % 2 == 0)
            left_lab = (j == 0 and i % 2 == 1)
            right_lab = (j == n - 1 and i % 2 == 0)
            # diag="hist" puts counts on y, so a perimeter y-label here
            # would mislead — strip it for diagonal hist cells only.
            if i == j and diag == "hist":
                left_lab = right_lab = False

            ax.tick_params(
                top=top_lab, bottom=bot_lab,
                left=left_lab, right=right_lab,
                labeltop=top_lab, labelbottom=bot_lab,
                labelleft=left_lab, labelright=right_lab,
            )

    if main is not None:
        fig.suptitle(main)
    fig.tight_layout()
    return axes
