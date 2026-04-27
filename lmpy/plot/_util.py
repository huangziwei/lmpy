"""Shared low-level helpers for the lmpy.plot package."""

from __future__ import annotations

import numpy as np
import polars as pl

_R_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "+", "x"]


def to_codes(x):
    """Polars Enum/Categorical → integer codes; numpy/list → unchanged."""
    if isinstance(x, pl.Series) and x.dtype in (pl.Enum, pl.Categorical):
        return x.to_physical().to_numpy()
    if isinstance(x, pl.Series):
        return x.to_numpy()
    return np.asarray(x)


def to_float(x):
    if isinstance(x, pl.Series):
        return x.cast(pl.Float64).to_numpy()
    return np.asarray(x, dtype=float)


def r_lty(lty):
    """R lty (1 solid, 2 dashed, 3 dotted, 4 dotdash, 5 longdash, 6 twodash)
    or matplotlib string → matplotlib linestyle."""
    if lty is None:
        return "-"
    if isinstance(lty, str):
        return lty
    return {1: "-", 2: "--", 3: ":", 4: "-.", 5: (0, (10, 3)),
            6: (0, (5, 1, 1, 1))}.get(int(lty), "-")


def draw_points(ax, x, y, *, pch=None, cex=None, col=None):
    """Per-marker scatter draw — no axis labels touched.

    Used by both ``scatter()`` (the primary plotter) and ``points()`` (the
    overlay). When ``pch`` is a vector of integer codes, splits into one
    ``scatter`` call per unique code so matplotlib gets a scalar marker."""
    xa = to_float(x)
    ya = to_float(y)
    pch_codes = to_codes(pch) if pch is not None else None
    col_codes = to_codes(col) if col is not None else None

    base = {"facecolor": "none", "edgecolor": "black"}
    if cex is not None:
        base["s"] = (cex * 6) ** 2  # rough R cex≈1 default

    if pch_codes is not None and getattr(pch_codes, "ndim", 0) > 0:
        for code in np.unique(pch_codes):
            mask = pch_codes == code
            marker = _R_MARKERS[int(code) % len(_R_MARKERS)]
            kw = dict(base)
            if col_codes is not None and getattr(col_codes, "ndim", 0) > 0:
                kw["c"] = col_codes[mask]
                kw.pop("edgecolor", None)
            ax.scatter(xa[mask], ya[mask], marker=marker, **kw)
        return

    kw = dict(base)
    if col_codes is not None:
        if getattr(col_codes, "ndim", 0) > 0:
            kw["c"] = col_codes
            kw.pop("edgecolor", None)
        else:
            kw["edgecolor"] = col_codes
    if pch_codes is not None:
        kw["marker"] = _R_MARKERS[int(pch_codes) % len(_R_MARKERS)] \
            if isinstance(pch_codes, (int, np.integer)) else "o"
    ax.scatter(xa, ya, **kw)
