"""``plot()`` — single dispatch entry, mirrors R's S3 ``plot.*`` family.

Routes positional argument types to the appropriate Phase 1 renderer.
"""

from __future__ import annotations

import inspect

import numpy as np
import polars as pl

from ..formula import parse
from .boxplot import boxplot_by
from .diagnostic import plot_lm
from .formula_eval import eval_side
from .scatter import scatter


def _is_lm_like(obj) -> bool:
    """True for an object that exposes the lm/glm diagnostic panel API."""
    return all(
        hasattr(obj, m)
        for m in ("plot_residuals", "plot_qq", "plot_scale_location", "plot_leverage")
    )


def _is_categorical(s) -> bool:
    if isinstance(s, pl.Series):
        return s.dtype in (pl.Enum, pl.Categorical) or s.dtype == pl.Utf8
    return False


def _frame_env(frame) -> dict:
    """Merge a frame's globals + locals (locals win, mirroring Python lookup)."""
    if frame is None:
        return {}
    return {**frame.f_globals, **frame.f_locals}


def plot(*args, data: pl.DataFrame | None = None, env: dict | None = None,
         ax=None, **kwargs):
    """Polars/matplotlib port of R's ``plot()`` dispatch.

    Forms:
        plot("y ~ x", data=df)             # formula
        plot("y ~ a + b", data=df)         # multi-RHS (one panel per term)
        plot(x, y)                         # two numeric vectors
        plot(vec)                          # vector vs index
        plot(lm_or_glm)                    # 4-panel diagnostic (uses `which=`)

    Returns a matplotlib ``Axes`` (or list/array of ``Axes`` for multi-panel).
    Annotations (``abline`` etc., Phase 2) take ``ax=`` explicitly.
    """
    if len(args) == 0:
        raise TypeError("plot() requires at least one positional argument")

    a0 = args[0]

    # Form: plot(lm_object) — 4-panel diagnostic
    if len(args) == 1 and _is_lm_like(a0):
        return plot_lm(a0, **kwargs)

    # Form: plot("formula", data=df)
    if isinstance(a0, str):
        if data is None and len(args) >= 2 and isinstance(args[1], pl.DataFrame):
            data = args[1]
        # Capture caller frame here while it's still on the stack at a known depth
        caller_env = _frame_env(inspect.currentframe().f_back)
        return _plot_formula(a0, data=data, caller_env=caller_env, env=env,
                             ax=ax, **kwargs)

    # Form: plot(x, y) — two vectors
    if len(args) == 2:
        return scatter(a0, args[1], ax=ax, **kwargs)

    # Form: plot(vec) — single vector vs index
    if len(args) == 1:
        v = a0.to_numpy() if isinstance(a0, pl.Series) else np.asarray(a0)
        return scatter(np.arange(len(v)), v, ax=ax, **kwargs)

    raise TypeError(
        f"plot(): no dispatch for {len(args)} positional args of types "
        f"{[type(a).__name__ for a in args]}"
    )


def _plot_formula(formula_str: str, *, data: pl.DataFrame | None,
                  caller_env: dict, env: dict | None, ax, **kwargs):
    """Parse + evaluate a formula string, dispatch on RHS dtype."""
    if data is None:
        raise ValueError("plot(formula): `data=` is required for formula dispatch")

    f = parse(formula_str)
    if f.lhs is None:
        raise ValueError("plot(formula): one-sided formula not supported (need LHS ~ RHS)")

    full_env = {**caller_env, **(env or {})}

    rhs_terms = _split_plus(f.rhs)
    n_panels = len(rhs_terms)

    if n_panels > 1:
        if ax is not None:
            raise ValueError(
                "plot(multi-RHS formula): pass each term separately when supplying ax="
            )
        import matplotlib.pyplot as plt
        fig, axarr = plt.subplots(1, n_panels, figsize=(4 * n_panels, 3))
        axes = list(axarr) if n_panels > 1 else [axarr]
        for i, term in enumerate(rhs_terms):
            _plot_one_panel(f.lhs, term, data, full_env, axes[i], **kwargs)
        fig.tight_layout()
        return axes

    return _plot_one_panel(f.lhs, rhs_terms[0], data, full_env, ax, **kwargs)


def _split_plus(node):
    """Top-level additive split — each ``+``-separated subexpression is one panel."""
    from ..formula import BinOp
    out = []
    def walk(n):
        if isinstance(n, BinOp) and n.op == "+":
            walk(n.left)
            walk(n.right)
        else:
            out.append(n)
    walk(node)
    return out


def _plot_one_panel(lhs_node, rhs_node, data, env, ax, **kwargs):
    y_vals, y_label = eval_side(lhs_node, data, env)
    x_vals, x_label = eval_side(rhs_node, data, env)

    rhs_is_factor = _is_categorical(x_vals)

    xlab = kwargs.pop("xlab", x_label)
    ylab = kwargs.pop("ylab", y_label)

    if rhs_is_factor:
        return boxplot_by(y_vals, x_vals, ax=ax, xlab=xlab, ylab=ylab, **kwargs)
    return scatter(x_vals, y_vals, ax=ax, xlab=xlab, ylab=ylab, **kwargs)
