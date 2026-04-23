"""
R-style formula parser + design-matrix generator.

Consumes a formula string in R syntax — WR fixed-effects ops (+, -, *, :, /,
^, %in%, .), lme4 random-effect bars ((... | g), (... || g)), and mgcv smooth
constructors (s, te, ti, t2) — and will eventually emit whatever design
matrices the formula implies: X always, Z/Lambdat/theta if bars appear,
per-smooth blocks if smooth constructors appear.

This module is the library's core. It replaces the old formulaic dependency.

Current state: tokenizer + parser only. AST is rich enough to drive later
materialization passes; no materialization yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

__all__ = [
    "tokenize",
    "parse",
    "ParseError",
    "Token",
    "Formula",
    "Name",
    "Literal",
    "Dot",
    "Empty",
    "UnaryOp",
    "BinOp",
    "Call",
    "Paren",
    "Subscript",
]


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------

# Token kinds:
#   TILDE PLUS MINUS STAR SLASH CARET COLON BAR DOUBLE_BAR PERCENT_OP
#   DOLLAR LBRACKET RBRACKET
#   GT LT GE LE EQEQ NEQ BANG
#   LPAREN RPAREN COMMA EQUALS
#   IDENT NUMBER STRING DOT EOF


@dataclass(frozen=True, slots=True)
class Token:
    kind: str
    value: str
    pos: int


class ParseError(ValueError):
    """Raised when a formula can't be tokenized or parsed."""


_SINGLE_OPS = {
    "~": "TILDE", "+": "PLUS", "-": "MINUS", "*": "STAR", "/": "SLASH",
    "^": "CARET", ":": "COLON", "(": "LPAREN", ")": "RPAREN", ",": "COMMA",
    "=": "EQUALS", "$": "DOLLAR", "[": "LBRACKET", "]": "RBRACKET",
    "<": "LT", ">": "GT", "!": "BANG",
}


def tokenize(src: str) -> list[Token]:
    """Split an R-flavoured formula string into tokens.

    Handles: WR operators, lme4 |/||, mgcv call syntax with kwargs, dotted
    identifiers (`stack.loss`), standalone `.` (all-vars sentinel), numbers,
    quoted strings, backticked identifiers, R keywords (TRUE, FALSE, NA), and
    %op% (custom infix like %in%).
    """
    i, n = 0, len(src)
    out: list[Token] = []

    while i < n:
        c = src[i]

        if c.isspace():
            i += 1
            continue

        # Double-bar (lme4)
        if c == "|" and i + 1 < n and src[i + 1] == "|":
            out.append(Token("DOUBLE_BAR", "||", i))
            i += 2
            continue

        # Single-char bar (lme4 RE)
        if c == "|":
            out.append(Token("BAR", "|", i))
            i += 1
            continue

        # Two-char comparisons: ==, !=, <=, >=. Check before single-char ops.
        if i + 1 < n and src[i:i + 2] in ("==", "!=", "<=", ">="):
            kind = {"==": "EQEQ", "!=": "NEQ", "<=": "LE", ">=": "GE"}[src[i:i + 2]]
            out.append(Token(kind, src[i:i + 2], i))
            i += 2
            continue

        # %op% infix (e.g. %in%)
        if c == "%":
            j = src.find("%", i + 1)
            if j == -1:
                raise ParseError(f"unterminated %op% at {i}")
            out.append(Token("PERCENT_OP", src[i:j + 1], i))
            i = j + 1
            continue

        # Single-char operators / punct
        if c in _SINGLE_OPS:
            out.append(Token(_SINGLE_OPS[c], c, i))
            i += 1
            continue

        # Numbers — int, float, with optional exponent. `.5` counts; bare `.`
        # is handled later as a sentinel.
        if c.isdigit() or (c == "." and i + 1 < n and src[i + 1].isdigit()):
            j = i
            while j < n and src[j].isdigit():
                j += 1
            if j < n and src[j] == "." and (j + 1 >= n or src[j + 1].isdigit()):
                j += 1
                while j < n and src[j].isdigit():
                    j += 1
            if j < n and src[j] in "eE":
                j += 1
                if j < n and src[j] in "+-":
                    j += 1
                while j < n and src[j].isdigit():
                    j += 1
            out.append(Token("NUMBER", src[i:j], i))
            i = j
            continue

        # Strings (double or single quoted, with simple backslash escape)
        if c == '"' or c == "'":
            quote = c
            j = i + 1
            while j < n and src[j] != quote:
                j += 2 if src[j] == "\\" and j + 1 < n else 1
            if j >= n:
                raise ParseError(f"unterminated string at {i}")
            out.append(Token("STRING", src[i + 1:j], i))
            i = j + 1
            continue

        # Backticked identifier: `some name`
        if c == "`":
            j = i + 1
            while j < n and src[j] != "`":
                j += 1
            if j >= n:
                raise ParseError(f"unterminated backtick at {i}")
            out.append(Token("IDENT", src[i + 1:j], i))
            i = j + 1
            continue

        # Identifier — [A-Za-z_.][A-Za-z0-9_.]*
        #   A greedy consume; if the resulting string is exactly ".", emit as
        #   the DOT sentinel (WR all-vars). Dotted names like `stack.loss` are
        #   single IDENT tokens.
        if c.isalpha() or c == "_" or c == ".":
            j = i
            while j < n and (src[j].isalnum() or src[j] in "_."):
                j += 1
            ident = src[i:j]
            out.append(
                Token("DOT", ident, i) if ident == "."
                else Token("IDENT", ident, i)
            )
            i = j
            continue

        raise ParseError(f"unexpected character {c!r} at {i}")

    out.append(Token("EOF", "", n))
    return out


# ---------------------------------------------------------------------------
# AST nodes
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Name:
    ident: str


@dataclass(slots=True)
class Literal:
    value: Union[int, float, str, bool, None]
    kind: str  # "num" | "str" | "bool" | "NA"


@dataclass(slots=True)
class Dot:
    """The `.` sentinel — WR all-vars placeholder."""


@dataclass(slots=True)
class Empty:
    """A skipped positional arg, e.g. the middle slot in `C(f, , 1)`."""


@dataclass(slots=True)
class UnaryOp:
    op: str
    operand: "Node"


@dataclass(slots=True)
class BinOp:
    op: str
    left: "Node"
    right: "Node"


@dataclass(slots=True)
class Call:
    fn: str
    args: list["Node"]
    kwargs: dict[str, "Node"] = field(default_factory=dict)


@dataclass(slots=True)
class Paren:
    expr: "Node"


@dataclass(slots=True)
class Subscript:
    """`x[i]` — single-bracket indexing. Multi-index (`x[i, j]`) stored in `idx` as a list."""
    obj: "Node"
    idx: list["Node"]


@dataclass(slots=True)
class Formula:
    lhs: Optional["Node"]
    rhs: "Node"


Node = Union[Name, Literal, Dot, Empty, UnaryOp, BinOp, Call, Paren, Subscript]


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

# (precedence, right-associative?). Higher binds tighter.
#
# `$` and `[...]` are handled postfix in `_parse_postfix`, not here: they bind
# tighter than anything listed and need special shape (RHS is IDENT for `$`,
# bracketed arg list for `[`), so keeping them out of the generic BinOp loop
# also gives us the correct left-to-right chaining for `a$b[1]`.
_BINOPS: dict[str, tuple[int, bool]] = {
    "TILDE":      (1, False),
    "BAR":        (2, False),
    "DOUBLE_BAR": (2, False),
    "EQEQ":       (3, False),
    "NEQ":        (3, False),
    "LT":         (3, False),
    "GT":         (3, False),
    "LE":         (3, False),
    "GE":         (3, False),
    "PLUS":       (4, False),
    "MINUS":      (4, False),
    "STAR":       (5, False),
    "SLASH":      (5, False),
    "COLON":      (7, False),
    "PERCENT_OP": (7, False),
    "CARET":      (8, True),
}


class _Parser:
    __slots__ = ("tokens", "i")

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.i = 0

    def peek(self, k: int = 0) -> Token:
        return self.tokens[self.i + k]

    def advance(self) -> Token:
        t = self.tokens[self.i]
        self.i += 1
        return t

    def expect(self, kind: str) -> Token:
        t = self.peek()
        if t.kind != kind:
            raise ParseError(f"expected {kind}, got {t.kind} ({t.value!r}) at {t.pos}")
        return self.advance()

    def parse_expr(self, min_prec: int) -> Node:
        left = self._parse_prefix()
        while True:
            t = self.peek()
            info = _BINOPS.get(t.kind)
            if info is None or info[0] < min_prec:
                break
            prec, right_assoc = info
            self.advance()
            next_min = prec if right_assoc else prec + 1
            right = self.parse_expr(next_min)
            left = BinOp(t.value, left, right)
        return left

    def _parse_prefix(self) -> Node:
        t = self.peek()
        if t.kind == "MINUS":
            self.advance()
            return UnaryOp("-", self._parse_prefix())
        if t.kind == "PLUS":
            self.advance()
            return self._parse_prefix()
        if t.kind == "BANG":
            self.advance()
            return UnaryOp("!", self._parse_prefix())
        return self._parse_postfix()

    def _parse_postfix(self) -> Node:
        left = self._parse_atom()
        while True:
            t = self.peek()
            if t.kind == "LBRACKET":
                self.advance()
                idx: list[Node] = []
                while self.peek().kind != "RBRACKET":
                    if self.peek().kind == "COMMA":
                        idx.append(Empty())
                        self.advance()
                        continue
                    idx.append(self.parse_expr(0))
                    if self.peek().kind == "COMMA":
                        self.advance()
                        continue
                    break
                self.expect("RBRACKET")
                left = Subscript(left, idx)
                continue
            if t.kind == "DOLLAR":
                self.advance()
                name_tok = self.expect("IDENT")
                left = BinOp("$", left, Name(name_tok.value))
                continue
            break
        return left

    def _parse_atom(self) -> Node:
        t = self.peek()
        if t.kind == "LPAREN":
            self.advance()
            e = self.parse_expr(0)
            self.expect("RPAREN")
            return Paren(e)
        if t.kind == "NUMBER":
            self.advance()
            v = t.value
            return Literal(float(v) if ("." in v or "e" in v or "E" in v) else int(v), "num")
        if t.kind == "STRING":
            self.advance()
            return Literal(t.value, "str")
        if t.kind == "DOT":
            self.advance()
            return Dot()
        if t.kind == "IDENT":
            self.advance()
            # Function call?
            if self.peek().kind == "LPAREN":
                return self._parse_call_tail(t.value)
            # R keywords that become literals
            if t.value in ("TRUE", "T"):
                return Literal(True, "bool")
            if t.value in ("FALSE", "F"):
                return Literal(False, "bool")
            if t.value == "NA":
                return Literal(None, "NA")
            return Name(t.value)
        raise ParseError(f"unexpected token {t.kind} ({t.value!r}) at {t.pos}")

    def _parse_call_tail(self, fn: str) -> Call:
        self.expect("LPAREN")
        args: list[Node] = []
        kwargs: dict[str, Node] = {}

        while self.peek().kind != "RPAREN":
            # empty positional (e.g. middle of `C(f, , 1)`)
            if self.peek().kind == "COMMA":
                args.append(Empty())
                self.advance()
                continue
            # named arg?
            if self.peek().kind == "IDENT" and self.peek(1).kind == "EQUALS":
                name = self.advance().value
                self.advance()  # consume '='
                kwargs[name] = self.parse_expr(0)
            else:
                args.append(self.parse_expr(0))
            if self.peek().kind == "COMMA":
                self.advance()
                continue
            if self.peek().kind == "RPAREN":
                break
            t = self.peek()
            raise ParseError(f"expected , or ) in call to {fn!r}, got {t.kind} at {t.pos}")

        self.expect("RPAREN")
        return Call(fn, args, kwargs)


def parse(src: str) -> Formula:
    """Parse an R-style formula string into an AST.

    Accepts one-sided (`~ x + y`) and two-sided (`y ~ x + y`) formulas. For a
    bare expression (no tilde), returns `Formula(lhs=None, rhs=expr)`.
    """
    tokens = tokenize(src)
    p = _Parser(tokens)

    if p.peek().kind == "TILDE":
        p.advance()
        rhs = p.parse_expr(0)
        p.expect("EOF")
        return Formula(lhs=None, rhs=rhs)

    expr = p.parse_expr(0)
    p.expect("EOF")
    if isinstance(expr, BinOp) and expr.op == "~":
        return Formula(lhs=expr.left, rhs=expr.right)
    return Formula(lhs=None, rhs=expr)
