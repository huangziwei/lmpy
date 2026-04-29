"""Scatter renderer — covers ``num ~ num``, ``plot(x, y)``, ``plot(vec)``."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ._util import draw_points, r_lty, to_float


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

    xa = to_float(x)
    ya = to_float(y)

    if "p" in type or type in ("b", "o"):
        draw_points(ax, xa, ya, pch=pch, cex=cex, col=col)

    if "l" in type or type in ("b", "o"):
        order = np.argsort(xa)
        ax.plot(xa[order], ya[order], color="black", linestyle=r_lty(lty))

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
