"""Boxplot renderer — covers formula ``num ~ factor``."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def boxplot_by(
    y,
    group,
    *,
    ax=None,
    xlab: str | None = None,
    ylab: str | None = None,
    main: str | None = None,
):
    """Vertical boxplots of ``y`` grouped by ``group`` (a polars Enum/Categorical
    or any string-like Series). Level order is taken from ``group`` if it's an
    Enum (R-faithful), otherwise from sorted unique values."""
    if ax is None:
        _, ax = plt.subplots()

    if isinstance(group, pl.Series):
        if group.dtype == pl.Enum:
            levels = group.cat.get_categories().to_list()
        elif group.dtype == pl.Categorical:
            levels = group.cat.get_categories().to_list()
        else:
            levels = sorted(group.drop_nulls().unique().to_list())
        g_arr = group.to_numpy()
    else:
        g_arr = np.asarray(group)
        levels = sorted(set(x for x in g_arr.tolist() if x is not None))

    if isinstance(y, pl.Series):
        y_arr = y.cast(pl.Float64).to_numpy()
    else:
        y_arr = np.asarray(y, dtype=float)

    groups = [y_arr[g_arr == lvl] for lvl in levels]

    ax.boxplot(groups, tick_labels=[str(lv) for lv in levels])

    if xlab is not None:
        ax.set_xlabel(xlab)
    if ylab is not None:
        ax.set_ylabel(ylab)
    if main is not None:
        ax.set_title(main)
    return ax
