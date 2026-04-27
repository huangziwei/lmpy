"""Scatter renderer — covers ``num ~ num``, ``plot(x, y)``, ``plot(vec)``."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def _to_codes(x):
    """Polars Enum/Categorical → integer codes; numpy/list → unchanged."""
    if isinstance(x, pl.Series) and x.dtype in (pl.Enum, pl.Categorical):
        return x.to_physical().to_numpy()
    if isinstance(x, pl.Series):
        return x.to_numpy()
    return np.asarray(x)


def _to_float(x):
    if isinstance(x, pl.Series):
        return x.cast(pl.Float64).to_numpy()
    return np.asarray(x, dtype=float)


def scatter(
    x,
    y,
    *,
    ax=None,
    xlab: str | None = None,
    ylab: str | None = None,
    main: str | None = None,
    pch=None,
    cex: float | None = None,
    type: str = "p",
    xlim=None,
    ylim=None,
    log: str | None = None,
    col=None,
    lty=None,
):
    """Core scatter — used by formula ``num~num``, two-vector, and single-vec dispatch.

    ``type``: "p" (points), "l" (line), "b" (both), "h" (vertical stems),
    "o" (overplot points + line). ``log``: any of "", "x", "y", "xy".
    ``pch``/``col`` accept a polars Series/Enum (factor coloring/marker by level)
    or a scalar/array.
    """
    if ax is None:
        _, ax = plt.subplots()

    xa = _to_float(x)
    ya = _to_float(y)

    pch_codes = _to_codes(pch) if pch is not None else None
    col_codes = _to_codes(col) if col is not None else None

    if "p" in type or type in ("b", "o"):
        scatter_kw = {}
        if cex is not None:
            scatter_kw["s"] = (cex * 6) ** 2  # R cex=1 ≈ default; rough match
        if pch_codes is not None and pch_codes.ndim > 0:
            # one scatter call per unique marker code so matplotlib gets scalar markers
            _MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "+", "x"]
            for code in np.unique(pch_codes):
                mask = pch_codes == code
                marker = _MARKERS[int(code) % len(_MARKERS)]
                kw = dict(scatter_kw)
                if col_codes is not None and col_codes.ndim > 0:
                    kw["c"] = col_codes[mask]
                ax.scatter(xa[mask], ya[mask], marker=marker, facecolor="none",
                           edgecolor="black", **kw)
        else:
            kw = dict(scatter_kw)
            if col_codes is not None:
                kw["c"] = col_codes
            ax.scatter(xa, ya, facecolor="none", edgecolor="black", **kw)

    if "l" in type or type in ("b", "o"):
        order = np.argsort(xa)
        line_kw = {}
        if lty is not None:
            line_kw["linestyle"] = _r_lty(lty)
        ax.plot(xa[order], ya[order], color="black", **line_kw)

    if type == "h":
        ax.vlines(xa, 0, ya, color="black")

    if xlab is not None:
        ax.set_xlabel(xlab)
    if ylab is not None:
        ax.set_ylabel(ylab)
    if main is not None:
        ax.set_title(main)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if log:
        if "x" in log:
            ax.set_xscale("log")
        if "y" in log:
            ax.set_yscale("log")

    return ax


def _r_lty(lty):
    """R lty (1=solid, 2=dashed, 3=dotted, 4=dotdash, 5=longdash, 6=twodash)
    or matplotlib string → matplotlib linestyle."""
    if isinstance(lty, str):
        return lty
    return {1: "-", 2: "--", 3: ":", 4: "-.", 5: (0, (10, 3)),
            6: (0, (5, 1, 1, 1))}.get(int(lty), "-")
