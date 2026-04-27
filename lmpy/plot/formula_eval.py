"""Evaluate an ``lmpy.formula`` AST node against a polars frame + env.

Faraway's book uses formulas with arbitrary expression LHS/RHS:
``residuals(lmod) ~ year``, ``log(NOx) ~ E``, ``tail(r,n-1) ~ head(r,n-1)``.
``lmpy.formula.parse`` already produces the AST; this module turns each
side into a numpy array.

The evaluator looks up names against (in order):
1. polars columns of ``data``
2. the user-supplied ``env`` mapping (caller's locals/globals when
   ``plot()`` builds it via frame inspection)
3. ``DEFAULT_ENV`` — math functions and ``residuals``/``fitted``/``coef``
   that work on lmpy fit objects.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ..formula import (
    BinOp,
    Call,
    Literal,
    Name,
    Paren,
    Subscript,
    UnaryOp,
)


def _residuals(model):
    if hasattr(model, "_residuals_arr"):
        return model._residuals_arr
    if hasattr(model, "residuals"):
        r = model.residuals
        return r["residuals"].to_numpy() if isinstance(r, pl.DataFrame) else np.asarray(r)
    raise TypeError(f"residuals(): can't extract from {type(model).__name__}")


def _fitted(model):
    if hasattr(model, "yhat"):
        return model.yhat["Fitted"].to_numpy()
    if hasattr(model, "fitted_values"):
        return np.asarray(model.fitted_values)
    raise TypeError(f"fitted(): can't extract from {type(model).__name__}")


def _coef(model):
    if hasattr(model, "_bhat_arr"):
        return model._bhat_arr
    if hasattr(model, "bhat"):
        b = model.bhat
        return b.to_numpy().ravel() if isinstance(b, pl.DataFrame) else np.asarray(b)
    raise TypeError(f"coef(): can't extract from {type(model).__name__}")


DEFAULT_ENV: dict = {
    "residuals": _residuals,
    "fitted": _fitted,
    "coef": _coef,
    "predict": lambda m: _fitted(m),
    "log": np.log,
    "log2": np.log2,
    "log10": np.log10,
    "exp": np.exp,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "head": lambda x, n: np.asarray(x)[: int(n)],
    "tail": lambda x, n: np.asarray(x)[-int(n):],
    "sort": np.sort,
    "unique": np.unique,
    "rev": lambda x: np.asarray(x)[::-1],
}


def _arr(x):
    """Coerce a polars Series, list, or scalar to a numpy array (or scalar)."""
    if isinstance(x, pl.Series):
        return x.to_numpy()
    if isinstance(x, (int, float, str, bool, np.generic)):
        return x
    return np.asarray(x)


def eval_node(node, data: pl.DataFrame | None, env: dict):
    """Walk an ``lmpy.formula`` AST node and return a numpy array (or scalar)."""
    if isinstance(node, Name):
        if data is not None and node.ident in data.columns:
            return data[node.ident]  # keep as polars Series so dtype check works
        if node.ident in env:
            return env[node.ident]
        raise NameError(
            f"name {node.ident!r} not found in data columns or env "
            f"(available cols: {list(data.columns) if data is not None else []})"
        )

    if isinstance(node, Literal):
        if node.kind == "num":
            v = node.value
            return float(v) if "." in str(v) or "e" in str(v).lower() else int(v)
        return node.value  # string literal

    if isinstance(node, Paren):
        return eval_node(node.expr, data, env)

    if isinstance(node, BinOp):
        L = _arr(eval_node(node.left, data, env))
        R = _arr(eval_node(node.right, data, env))
        op = node.op
        if op == "+": return L + R
        if op == "-": return L - R
        if op == "*": return L * R
        if op == "/": return L / R
        if op == "^": return L ** R
        raise ValueError(f"unsupported binary op {op!r} in expression")

    if isinstance(node, UnaryOp):
        v = _arr(eval_node(node.operand, data, env))
        if node.op == "-": return -v
        if node.op == "+": return v
        raise ValueError(f"unsupported unary op {node.op!r}")

    if isinstance(node, Call):
        f = env.get(node.fn)
        if f is None:
            raise NameError(f"function {node.fn!r} not in env (Phase 1 default env)")
        args = [eval_node(a, data, env) for a in node.args]
        kwargs = {k: eval_node(v, data, env) for k, v in (node.kwargs or {}).items()}
        return f(*args, **kwargs)

    if isinstance(node, Subscript):
        target = _arr(eval_node(node.obj, data, env))
        idx = [eval_node(a, data, env) for a in node.idx]
        if len(idx) == 1:
            return target[_arr(idx[0])]
        return target[tuple(_arr(i) for i in idx)]

    raise TypeError(f"unhandled formula node type: {type(node).__name__}")


def eval_side(node, data: pl.DataFrame | None, env: dict | None = None):
    """Evaluate one side of a formula. Returns ``(values, label)``.

    ``values`` is a polars Series (when the result is a raw column — preserves
    dtype for downstream dispatch) or a numpy array (for any computed expression).
    ``label`` is a string suitable for an axis label, derived via
    ``lmpy.formula.deparse``.
    """
    from ..formula import deparse

    full_env = {**DEFAULT_ENV, **(env or {})}
    out = eval_node(node, data, full_env)
    label = deparse(node)
    if isinstance(out, pl.Series):
        return out, label
    return np.asarray(out), label
