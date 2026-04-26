"""Formula → fitting-ready design bundle.

The orchestration layer that sits between ``lmpy.formula`` (parse trees,
term algebra, basis construction) and the model classes. Given a formula
string and a polars DataFrame, ``prepare_design``:

1. Parses the formula, expands the RHS into terms, materializes the
   fixed-effect design matrix.
2. Evaluates the LHS — which may be a bare name or a small expression
   (``log(y)``, ``y^0.25``, ``I(y/100)``, etc.).
3. NA-omits rows referenced by either side, mirroring R's
   ``na.action = na.omit``.
4. Returns a ``Design`` bundle that downstream models specialize as
   they see fit.

User-facing data prep — ``data()`` and ``factor()`` — lives in
``lmpy.data``.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from .formula import (
    BinOp,
    Call,
    ExpandedFormula,
    Literal,
    Name,
    Paren,
    UnaryOp,
    deparse,
    expand,
    materialize,
    parse,
    referenced_columns,
)

__all__ = ["Design", "prepare_design"]


@dataclass(slots=True)
class Design:
    """Bundle returned by ``prepare_design``.

    Attributes
    ----------
    expanded : ExpandedFormula
        Output of ``formula.expand`` for the parsed formula. Pass this
        to downstream materializers (``materialize_bars`` for lme,
        ``materialize_smooths`` for gam) so they share the same parse.
    data : polars.DataFrame
        Input data with rows dropped where the response or any
        RHS-referenced column is NA. Row positions align with ``X``
        and ``y``.
    X : polars.DataFrame
        Materialized fixed-effect design with R-canonical column names.
    y : polars.Series
        Response column, with NA rows dropped. For non-trivial LHS
        expressions, holds the *evaluated* response (e.g. log(y));
        the Series name is the deparsed LHS source.
    response : str
        Response label — bare column name for ``y ~ ...`` formulas,
        deparsed LHS source (e.g. ``"medFPQ^0.25"``) otherwise.
    """
    expanded: ExpandedFormula
    data: pl.DataFrame
    X: pl.DataFrame
    y: pl.Series
    response: str


# LHS function table — maps R-side function names to a polars-expr builder.
# Mirrors what mgcv/base R accept on a formula LHS: arithmetic via
# `_eval_lhs_expr` (UnaryOp/BinOp), plus these elementary transforms.
_LHS_FUNCS: dict[str, "callable"] = {
    "log":   lambda e: e.log(),
    "log2":  lambda e: e.log(2.0),
    "log10": lambda e: e.log10(),
    "exp":   lambda e: e.exp(),
    "sqrt":  lambda e: e.sqrt(),
    "abs":   lambda e: e.abs(),
}


def _lhs_referenced_cols(node, columns: set[str]) -> set[str]:
    """Walk an LHS AST and collect ``Name`` idents that exist in ``data``.

    Used by ``prepare_design`` to decide which columns to NA-drop on
    before evaluating the response. Names that don't match a column name
    are silently skipped — they'll error later in ``_eval_lhs_expr``.
    """
    out: set[str] = set()
    def visit(n):
        if isinstance(n, Name):
            if n.ident in columns:
                out.add(n.ident)
            return
        if isinstance(n, Literal):
            return
        if isinstance(n, Paren):
            visit(n.expr); return
        if isinstance(n, UnaryOp):
            visit(n.operand); return
        if isinstance(n, BinOp):
            visit(n.left); visit(n.right); return
        if isinstance(n, Call):
            for a in n.args:
                visit(a)
            for v in n.kwargs.values():
                visit(v)
            return
        # Anything else (Dot, Empty, Subscript, …) shouldn't appear on a
        # response LHS — let _eval_lhs_expr raise the clearer error.
    visit(node)
    return out


def _eval_lhs_expr(node, columns: set[str]) -> pl.Expr:
    """Recursively evaluate an LHS AST as a polars expression.

    Supported:
      * ``Name``    → ``pl.col(name)``
      * numeric ``Literal``
      * ``+``/``-`` (unary), ``+``/``-``/``*``/``/``/``^``
      * ``I(expr)`` (R's "as is" — just unwraps)
      * one-arg numeric calls listed in ``_LHS_FUNCS``
      * parens

    Multi-column responses (``cbind(succ, fail)``) and arbitrary user
    functions are not yet supported.
    """
    if isinstance(node, Name):
        if node.ident not in columns:
            raise KeyError(f"LHS references unknown column {node.ident!r}")
        return pl.col(node.ident)
    if isinstance(node, Literal):
        if node.kind != "num":
            raise NotImplementedError(
                f"LHS literal kind {node.kind!r} not supported"
            )
        return pl.lit(float(node.value))
    if isinstance(node, Paren):
        return _eval_lhs_expr(node.expr, columns)
    if isinstance(node, UnaryOp):
        e = _eval_lhs_expr(node.operand, columns)
        if node.op == "-":
            return -e
        if node.op == "+":
            return e
        raise NotImplementedError(f"LHS unary op {node.op!r} not supported")
    if isinstance(node, BinOp):
        L = _eval_lhs_expr(node.left, columns)
        R = _eval_lhs_expr(node.right, columns)
        if node.op == "+":
            return L + R
        if node.op == "-":
            return L - R
        if node.op == "*":
            return L * R
        if node.op == "/":
            return L / R
        if node.op == "^":
            return L ** R
        raise NotImplementedError(f"LHS binary op {node.op!r} not supported")
    if isinstance(node, Call):
        if node.kwargs:
            raise NotImplementedError(
                f"LHS call {node.fn}() with kwargs is not supported"
            )
        if node.fn == "I":
            if len(node.args) != 1:
                raise ValueError("I() takes exactly one argument")
            return _eval_lhs_expr(node.args[0], columns)
        if node.fn == "cbind":
            raise NotImplementedError(
                "multi-column response cbind() (e.g. for binomial trials) "
                "is not yet supported on the LHS"
            )
        if node.fn in _LHS_FUNCS:
            if len(node.args) != 1:
                raise ValueError(f"{node.fn}() takes exactly one argument on LHS")
            return _LHS_FUNCS[node.fn](_eval_lhs_expr(node.args[0], columns))
        raise NotImplementedError(
            f"LHS function {node.fn}() not supported "
            f"(allowed: I, {', '.join(sorted(_LHS_FUNCS))})"
        )
    raise NotImplementedError(
        f"LHS contains unsupported node {type(node).__name__}"
    )


def prepare_design(formula: str, data: pl.DataFrame) -> Design:
    """Parse a formula, expand, and materialize the fixed-effect design.

    NA-omit policy matches R's ``na.action = na.omit``: rows with NA in
    the response or in any RHS-referenced column are dropped before the
    design matrix is built. All three outputs (``Design.data``,
    ``Design.X``, ``Design.y``) share the same row ordering.

    The LHS may be either a bare column name or a small expression
    R/mgcv accepts: arithmetic (``y/100``, ``y^0.25``), unary minus,
    one-arg transforms (``log``, ``log2``, ``log10``, ``exp``,
    ``sqrt``, ``abs``), and ``I(expr)``. The deparsed LHS becomes
    the response label (``Design.response``) so downstream printers
    can show e.g. ``medFPQ^0.25`` rather than a placeholder name.
    """
    f_parsed = parse(formula)
    if f_parsed.lhs is None:
        raise NotImplementedError("formula must have a response (LHS)")

    columns = set(data.columns)
    if isinstance(f_parsed.lhs, Name):
        response_label = f_parsed.lhs.ident
        lhs_cols = {f_parsed.lhs.ident} & columns
    else:
        response_label = deparse(f_parsed.lhs)
        lhs_cols = _lhs_referenced_cols(f_parsed.lhs, columns)

    expanded = expand(f_parsed, data_columns=list(data.columns))

    na_cols = (referenced_columns(expanded) | lhs_cols) & columns
    if na_cols:
        data_clean = data.drop_nulls(subset=list(na_cols))
    else:
        data_clean = data

    if isinstance(f_parsed.lhs, Name):
        y = data_clean[f_parsed.lhs.ident]
    else:
        # Evaluate the LHS expression over the cleaned frame and tag
        # the resulting Series with the deparsed label so consumers
        # (stats printers, residual formatters) see the original text.
        y = data_clean.select(
            _eval_lhs_expr(f_parsed.lhs, columns).alias(response_label)
        )[response_label]

    X = materialize(expanded, data_clean)
    return Design(expanded=expanded, data=data_clean, X=X, y=y, response=response_label)
