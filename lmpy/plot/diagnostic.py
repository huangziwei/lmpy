"""Diagnostic 4-panel for ``lm``/``glm`` results — ``plot(lmod)``."""

from __future__ import annotations

import matplotlib.pyplot as plt


def plot_lm(lmod, *, which=None, figsize=None, smooth: bool = True, label_n: int = 3):
    """4-panel (or subset) diagnostic for an ``lm``/``glm`` result.

    R parity: ``which=1:6`` selects from {1: resid-fitted, 2: QQ, 3:
    scale-location, 4: Cook's distance, 5: resid-leverage, 6: Cook vs
    leverage}. We currently support 1, 2, 3, 5 (the four panels that
    ``lm.plot()`` builds). Pass an int for a single panel, an iterable
    for a subset, or omit for the default 4-panel layout.
    """
    panels = (1, 2, 3, 5) if which is None else (
        (which,) if isinstance(which, int) else tuple(which)
    )

    n = len(panels)
    if figsize is None:
        figsize = (5 * min(n, 2), 4 * ((n + 1) // 2)) if n > 1 else (6, 4)

    if n == 1:
        fig, ax = plt.subplots(figsize=figsize)
        axes = [ax]
    else:
        rows = (n + 1) // 2
        cols = 2 if n > 1 else 1
        fig, axarr = plt.subplots(rows, cols, figsize=figsize)
        axes = list(axarr.flatten()) if hasattr(axarr, "flatten") else [axarr]
        for extra in axes[n:]:
            extra.set_visible(False)

    for i, p in enumerate(panels):
        ax = axes[i]
        if p == 1:
            lmod.plot_residuals(ax=ax, smooth=smooth, label_n=label_n)
        elif p == 2:
            lmod.plot_qq(ax=ax, label_n=label_n)
        elif p == 3:
            lmod.plot_scale_location(ax=ax, smooth=smooth, label_n=label_n)
        elif p == 5:
            lmod.plot_leverage(ax=ax, smooth=smooth, label_n=label_n)
        else:
            raise ValueError(
                f"plot(lmod, which={p}): only panels 1, 2, 3, 5 are supported "
                f"in Phase 1 (Cook's distance and Cook-vs-leverage are deferred)"
            )
    fig.tight_layout()
    return axes[0] if n == 1 else axes
