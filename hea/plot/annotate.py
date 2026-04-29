"""Annotation/overlay calls — layered onto an existing ``Axes`` via ``ax=``.

Phase 2 surface (from the Faraway inventory):
    abline    — straight lines: from a fit, (a, b), or h=/v=
    points    — scatter overlay
    lines     — line overlay; accepts (x, y) or formula + data
    legend    — corner-anchored labels
    segments  — arbitrary line segments
    qqline    — quartile-anchored Q-Q reference (paired with qqnorm)
"""

from __future__ import annotations

import inspect

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import norm

from ..formula import parse
from ._util import draw_points, r_lty, to_float
from .formula_eval import eval_side


def _require_ax(ax, fname: str):
    if ax is None:
        raise ValueError(f"{fname}(): `ax=` is required (hea.plot has no current-axes state)")


def _is_lm_like(obj) -> bool:
    return hasattr(obj, "_bhat_arr") and hasattr(obj, "column_names")


def _as_iter(v):
    """Wrap scalars in a 1-element list so we can loop uniformly."""
    if isinstance(v, (list, tuple, np.ndarray, pl.Series)):
        return list(v)
    return [v]


def abline(*args, h=None, v=None, lty=None, col="black", ax=None):
    """Add reference lines to ``ax``. R-flavored signatures all supported.

    Forms:
        abline(lmod)           — (intercept, slope) extracted from a fit
        abline(coef_array)     — length-2 array [intercept, slope]
        abline(a, b)           — intercept and slope as scalars
        abline(h=value)        — horizontal; value may be scalar or vector
        abline(v=value)        — vertical;   value may be scalar or vector

    ``h=`` and ``v=`` may be combined with each other but not with
    positional args. ``lty=`` accepts an R lty code or matplotlib string;
    when ``h``/``v`` is a vector, ``lty`` may also be a same-length vector
    for per-line styling (R's ``lty=1:2`` idiom).
    """
    _require_ax(ax, "abline")
    color = col

    if h is not None or v is not None:
        if args:
            raise TypeError("abline(): pass either positional args or h=/v=, not both")
        for which, vals in (("h", h), ("v", v)):
            if vals is None:
                continue
            vals_list = _as_iter(vals)
            ltys = _as_iter(lty) if lty is not None else [None] * len(vals_list)
            if len(ltys) != len(vals_list):
                ltys = (ltys * len(vals_list))[: len(vals_list)]
            for vv, ll in zip(vals_list, ltys):
                line_kw = {"linestyle": r_lty(ll), "color": color}
                (ax.axhline if which == "h" else ax.axvline)(float(vv), **line_kw)
        return ax

    if not args:
        raise TypeError("abline(): need either positional (a, b)/(lmod)/(coef) or h=/v=")

    if len(args) == 1:
        arg = args[0]
        if _is_lm_like(arg):
            bhat = np.asarray(arg._bhat_arr).ravel()
            cols = list(arg.column_names)
            if "(Intercept)" not in cols:
                raise ValueError("abline(lmod): fit has no (Intercept); pass coefs explicitly")
            if len(bhat) != 2:
                raise ValueError(
                    f"abline(lmod): expected 2 coefficients (intercept + 1 slope), got {len(bhat)} "
                    f"({cols}). For multivariate fits, plot one slice via ax explicitly."
                )
            a = float(bhat[cols.index("(Intercept)")])
            b = float(bhat[1 - cols.index("(Intercept)")])
        else:
            arr = np.asarray(arg, dtype=float).ravel()
            if arr.size != 2:
                raise ValueError(
                    f"abline(coef): expected length-2 [intercept, slope], got length {arr.size}"
                )
            a, b = float(arr[0]), float(arr[1])
    elif len(args) == 2:
        a, b = float(args[0]), float(args[1])
    else:
        raise TypeError(f"abline(): too many positional args ({len(args)})")

    xlim = ax.get_xlim()
    xs = np.array(xlim)
    ax.plot(xs, a + b * xs, linestyle=r_lty(lty), color=color)
    ax.set_xlim(xlim)  # don't auto-expand from the new line
    return ax


def points(x, y, *, ax=None, pch=None, cex=None, col=None):
    """Overlay scatter points on an existing axes."""
    _require_ax(ax, "points")
    draw_points(ax, x, y, pch=pch, cex=cex, col=col)
    return ax


def lines(*args, ax=None, data: pl.DataFrame | None = None, lty=None, col="black",
          type: str = "l", **_kwargs):
    """Overlay a line on an existing axes.

    Forms:
        lines(x, y)
        lines("y ~ x", data=df)        — formula form, mirrors R
    """
    _require_ax(ax, "lines")
    if not args:
        raise TypeError("lines(): need either (x, y) or (formula, data=)")

    a0 = args[0]
    if isinstance(a0, str):
        if data is None and len(args) >= 2 and isinstance(args[1], pl.DataFrame):
            data = args[1]
        if data is None:
            raise ValueError("lines(formula): `data=` is required")
        f = parse(a0)
        if f.lhs is None:
            raise ValueError("lines(formula): need LHS ~ RHS")
        caller = inspect.currentframe().f_back
        env = {**caller.f_globals, **caller.f_locals}
        ya, _ = eval_side(f.lhs, data, env)
        xa, _ = eval_side(f.rhs, data, env)
        x = to_float(xa)
        y = to_float(ya)
    elif len(args) >= 2:
        x = to_float(args[0])
        y = to_float(args[1])
    else:
        raise TypeError("lines(x, y): need both x and y")

    order = np.argsort(x)
    ax.plot(x[order], y[order], linestyle=r_lty(lty), color=col)
    return ax


_R_LOC = {
    "topright": "upper right", "topleft": "upper left",
    "bottomright": "lower right", "bottomleft": "lower left",
    "top": "upper center", "bottom": "lower center",
    "left": "center left", "right": "center right",
    "center": "center",
}


def legend(*args, ax=None, legend=None, pch=None, lty=None, col=None, **kwargs):
    """Add a legend to ``ax``.

    Forms:
        legend("topright", legend=["a", "b"], pch=[1, 2])
        legend(x, y, legend=["a", "b"], lty=[1, 2])

    Marker/line-style entries built from ``pch`` and ``lty``; matplotlib's
    location strings ("upper right" etc.) and R's compact form ("topright")
    both work.
    """
    _require_ax(ax, "legend")
    if legend is None:
        raise TypeError("legend(): `legend=` (the labels list) is required")
    labels = list(legend)

    handles = []
    pchs = list(pch) if pch is not None else [None] * len(labels)
    ltys = list(lty) if lty is not None else [None] * len(labels)
    cols = list(col) if isinstance(col, (list, tuple)) else [col] * len(labels)
    from ._util import _R_MARKERS
    from matplotlib.lines import Line2D
    for i, lab in enumerate(labels):
        marker = _R_MARKERS[int(pchs[i]) % len(_R_MARKERS)] if pchs[i] is not None else None
        ls = r_lty(ltys[i]) if ltys[i] is not None else "None"
        c = cols[i] if cols[i] is not None else "black"
        handles.append(Line2D([0], [0], marker=marker or "", linestyle=ls, color=c,
                              label=lab, markerfacecolor="none", markeredgecolor=c))

    loc = None
    if args:
        first = args[0]
        if isinstance(first, str):
            loc = _R_LOC.get(first, first)
        elif len(args) >= 2:
            loc = (float(args[0]), float(args[1]))
    ax.legend(handles=handles, labels=labels, loc=loc, **kwargs)
    return ax


def segments(x0, y0, x1, y1, *, ax=None, lty=None, col="black"):
    """Draw line segments from (x0, y0) to (x1, y1). All four can be scalars
    or matching-length vectors."""
    _require_ax(ax, "segments")
    x0a = np.atleast_1d(np.asarray(x0, dtype=float))
    y0a = np.atleast_1d(np.asarray(y0, dtype=float))
    x1a = np.atleast_1d(np.asarray(x1, dtype=float))
    y1a = np.atleast_1d(np.asarray(y1, dtype=float))
    n = max(x0a.size, y0a.size, x1a.size, y1a.size)
    x0a, y0a, x1a, y1a = (np.broadcast_to(a, (n,)) for a in (x0a, y0a, x1a, y1a))
    style = r_lty(lty)
    for i in range(n):
        ax.plot([x0a[i], x1a[i]], [y0a[i], y1a[i]], linestyle=style, color=col)
    return ax


def qqline(x, *, ax=None, col="black", lty=None):
    """Add a quartile-anchored reference line to a Q-Q plot already drawn on ``ax``.

    R's ``qqline`` fits a line through the (.25, .75) quantile pair against
    the corresponding standard-normal quantiles, then extends it across the
    plotted range. Mirrors the helper in ``lm._qq_plot``."""
    _require_ax(ax, "qqline")
    vals = np.asarray(x, dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size < 2:
        return ax
    ry1, ry3 = np.quantile(vals, [0.25, 0.75])
    qx1, qx3 = norm.ppf([0.25, 0.75])
    slope = (ry3 - ry1) / (qx3 - qx1)
    intercept = ry1 - slope * qx1
    xlim = ax.get_xlim()
    xs = np.array(xlim)
    ax.plot(xs, intercept + slope * xs, linestyle=r_lty(lty) if lty else "--", color=col)
    ax.set_xlim(xlim)
    return ax
