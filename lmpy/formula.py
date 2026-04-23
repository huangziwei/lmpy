"""
R-style formula parser + design-matrix generator.

Consumes a formula string in R syntax — WR fixed-effects ops (+, -, *, :, /,
^, %in%, .), lme4 random-effect bars ((... | g), (... || g)), and mgcv smooth
constructors (s, te, ti, t2) — and will eventually emit whatever design
matrices the formula implies: X always, Z/Lambdat/theta if bars appear,
per-smooth blocks if smooth constructors appear.

This module is the library's core. It replaces the old formulaic dependency.

Current state: tokenizer + parser + term algebra. The term-expansion layer
produces R-compatible term labels (used as ground truth in fixture X_meta.json)
and separates fixed-effect terms from lme4 RE bars. No X/Z materialization yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
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
    "Term",
    "ExpandedFormula",
    "expand",
    "deparse",
    "materialize",
    "materialize_bars",
    "ReTerms",
    "materialize_smooths",
    "SmoothBlock",
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


# ---------------------------------------------------------------------------
# Deparse — canonical R-style string for an AST node.
# ---------------------------------------------------------------------------
#
# Matches R's deparse() for the operators that show up in formula term labels.
# The rules came from inspecting fixture X_meta.json term_labels:
#   tight (no spaces):  :  ^  $  /  [
#   spaced (padded):    +  -  *  ==  !=  <  >  <=  >=  |  ||  ~  %op%
#   kwargs '=':         spaced
#
# Parens are emitted as typed, unary ops as prefix.

_TIGHT_BINOPS = frozenset({":", "^", "$", "/"})


def _deparse(node) -> str:
    if isinstance(node, Name):
        return node.ident
    if isinstance(node, Literal):
        if node.kind == "num":
            return str(node.value)
        if node.kind == "str":
            return f'"{node.value}"'
        if node.kind == "bool":
            return "TRUE" if node.value else "FALSE"
        if node.kind == "NA":
            return "NA"
        return str(node.value)
    if isinstance(node, Dot):
        return "."
    if isinstance(node, Empty):
        return ""
    if isinstance(node, UnaryOp):
        return f"{node.op}{_deparse(node.operand)}"
    if isinstance(node, BinOp):
        sep = node.op if node.op in _TIGHT_BINOPS else f" {node.op} "
        return f"{_deparse(node.left)}{sep}{_deparse(node.right)}"
    if isinstance(node, Paren):
        return f"({_deparse(node.expr)})"
    if isinstance(node, Subscript):
        return f"{_deparse(node.obj)}[{', '.join(_deparse(i) for i in node.idx)}]"
    if isinstance(node, Call):
        parts = [_deparse(a) for a in node.args]
        parts.extend(f"{k} = {_deparse(v)}" for k, v in node.kwargs.items())
        return f"{node.fn}({', '.join(parts)})"
    raise TypeError(f"cannot deparse {type(node).__name__}")


def deparse(node) -> str:
    """Public deparser: canonical R-style source for any AST node."""
    return _deparse(node)


# ---------------------------------------------------------------------------
# Term algebra — WR formula-operator expansion.
# ---------------------------------------------------------------------------
#
# A Term is an interaction: an unordered set of atoms (Nodes that survive
# expansion — Names, Calls, Subscripts, I(…) wrappers, and other leaves). The
# empty term = intercept.
#
# Two terms are equal iff they contain the same atoms (by canonical deparse
# key); ordering within a term is preserved as first seen, because R's
# `:`-joined term label retains source order (`Insul:Temp`, not `Temp:Insul`).


@dataclass(frozen=True, slots=True)
class Term:
    atoms: tuple  # tuple[Node, ...] — empty tuple = intercept

    @property
    def _key(self) -> frozenset:
        return frozenset(_deparse(a) for a in self.atoms)

    def __hash__(self) -> int:
        return hash(self._key)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Term):
            return NotImplemented
        return self._key == other._key

    @property
    def label(self) -> str:
        """R-style label: atom-key joined by ':'. Intercept → '(Intercept)'."""
        if not self.atoms:
            return "(Intercept)"
        return ":".join(_deparse(a) for a in self.atoms)

    def union(self, other: "Term") -> "Term":
        """Merge two terms. Preserve first-seen order; dedupe by atom key."""
        seen = {_deparse(a) for a in self.atoms}
        merged = list(self.atoms)
        for a in other.atoms:
            k = _deparse(a)
            if k not in seen:
                seen.add(k)
                merged.append(a)
        return Term(tuple(merged))


_EMPTY_TERM = Term(atoms=())


@dataclass(slots=True)
class ExpandedFormula:
    """Result of WR term expansion.

    Fixed-effect side is a list of Terms plus an intercept flag. RE bars from
    lme4 are separated out (each an unextracted BinOp with op '|' or '||');
    mgcv smooth calls (`s`/`te`/`ti`/`t2`) are extracted into `smooths`.
    `offsets` holds the inner arg of every `offset(...)` call — those never
    contribute to the design matrix but are added directly to the linear
    predictor at fit time.
    """
    intercept: bool
    terms: list[Term] = field(default_factory=list)
    bars: list[BinOp] = field(default_factory=list)
    offsets: list = field(default_factory=list)
    smooths: list[Call] = field(default_factory=list)

    @property
    def term_labels(self) -> list[str]:
        return [t.label for t in self.terms]


def _is_bar(node) -> bool:
    if isinstance(node, BinOp) and node.op in ("|", "||"):
        return True
    return isinstance(node, Paren) and isinstance(node.expr, BinOp) and node.expr.op in ("|", "||")


def _bar_node(node) -> BinOp:
    """Return the underlying BinOp from a bar (stripping a possible Paren)."""
    if isinstance(node, Paren):
        return node.expr
    return node


def _split_additive(node) -> list[tuple[int, object]]:
    """Flatten top-level + / - into a list of signed subexpressions.

    Walks only through `+`, `-`, and unary `+`/`-` at the surface — does not
    descend into `*`, `:`, parens, or calls. Paren(bar) stops there too.
    """
    out: list[tuple[int, object]] = []

    def walk(n, sign):
        if isinstance(n, BinOp) and n.op == "+":
            walk(n.left, sign)
            walk(n.right, sign)
        elif isinstance(n, BinOp) and n.op == "-":
            walk(n.left, sign)
            walk(n.right, -sign)
        elif isinstance(n, UnaryOp) and n.op == "-":
            walk(n.operand, -sign)
        elif isinstance(n, UnaryOp) and n.op == "+":
            walk(n.operand, sign)
        else:
            out.append((sign, n))

    walk(node, +1)
    return out


def _expand_non_additive(node) -> list[Term]:
    """Expand a subexpression that has no top-level + or -.

    Returns an unsigned list of Terms. Handles `*`, `:`, `%in%`, `/`, `^`,
    parens (transparent unless bar), and leaf atoms.
    """
    if _is_bar(node):
        # Bars never contribute to fixed-effect terms.
        return []
    if isinstance(node, Paren):
        # Paren is transparent to term expansion.
        return _expand_toplevel(node.expr)
    if isinstance(node, BinOp):
        if node.op == "*":
            L = _expand_toplevel(node.left)
            R = _expand_toplevel(node.right)
            return _dedup(L + R + _interact(L, R))
        if node.op in (":", "%in%"):
            return _interact(_expand_toplevel(node.left), _expand_toplevel(node.right))
        if node.op == "/":
            L = _expand_toplevel(node.left)
            R = _expand_toplevel(node.right)
            # R semantics (per `terms.formula`): a/b = a + a:b; compound LHS
            # collapses to one "parent" interaction before nesting RHS:
            #   (a+b)/c   -> a + b + a:b:c
            #   a/b/c     -> a + a:b + a:b:c
            parent = _EMPTY_TERM
            for t in L:
                parent = parent.union(t)
            nested = [parent.union(t) for t in R]
            return _dedup(L + nested)
        if node.op == "^":
            if not (isinstance(node.right, Literal) and node.right.kind == "num"):
                # `^` with non-literal exponent — treat whole node as atom.
                return [Term((node,))]
            L = _expand_toplevel(node.left)
            return _power_expand(L, int(node.right.value))
    # Leaf: Name, Call, Subscript, UnaryOp, BinOp in non-formula ops, Literal.
    return [Term((node,))]


def _expand_toplevel(node) -> list[Term]:
    """Expand at a level where + and - still count as formula operators.

    Splits top-level additive structure, then expands each piece via the
    non-additive pass. Signed list collapses via `_finalize` (here we just
    aggregate into a deduped positive-only list, matching R semantics).
    """
    pieces = _split_additive(node)
    # Aggregate by sign. In R, terms are a set — repeated + is idempotent; -
    # removes from the running set regardless of multiplicity.
    added: list[Term] = []
    removed: list[Term] = []
    for sign, sub in pieces:
        if _is_bar(sub):
            continue
        # Special literals for intercept toggling.
        if isinstance(sub, Literal) and sub.kind == "num":
            if sub.value == 1:
                (added if sign > 0 else removed).append(_EMPTY_TERM)
                continue
            if sub.value == 0:
                # `+0` removes intercept, `-0` adds it back.
                (removed if sign > 0 else added).append(_EMPTY_TERM)
                continue
        sub_terms = _expand_non_additive(sub)
        for t in sub_terms:
            (added if sign > 0 else removed).append(t)

    # Dedupe additions preserving insertion order; then strip removals.
    result: list[Term] = []
    seen: set[Term] = set()
    for t in added:
        if t not in seen:
            seen.add(t)
            result.append(t)
    removed_set = set(removed)
    return [t for t in result if t not in removed_set]


def _interact(L: list[Term], R: list[Term]) -> list[Term]:
    return [l.union(r) for l in L for r in R]


def _power_expand(L: list[Term], n: int) -> list[Term]:
    out: list[Term] = []
    for k in range(1, n + 1):
        for combo in combinations(L, k):
            term = _EMPTY_TERM
            for t in combo:
                term = term.union(t)
            out.append(term)
    return _dedup(out)


def _dedup(ts: list[Term]) -> list[Term]:
    seen: set[Term] = set()
    out: list[Term] = []
    for t in ts:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _collect_bars(node) -> list[BinOp]:
    """Find every `Paren(BinOp('|'|'||', ...))` reachable through + / - / unary."""
    bars: list[BinOp] = []

    def walk(n):
        if _is_bar(n):
            bars.append(_bar_node(n))
            return
        if isinstance(n, BinOp) and n.op in ("+", "-"):
            walk(n.left)
            walk(n.right)
        elif isinstance(n, UnaryOp):
            walk(n.operand)

    walk(node)
    return bars


def _expand_dot(node, data_columns: list[str], response_names: set[str]):
    """Substitute every `Dot()` in the tree with `col1 + col2 + ...` over the
    non-response data columns. Returns a new node.
    """
    cols = [c for c in data_columns if c not in response_names]
    if not cols:
        raise ValueError("dot expansion: no non-response columns available")
    # Build `c1 + c2 + ... + cn` as a left-folded BinOp('+').
    expansion: object = Name(cols[0])
    for c in cols[1:]:
        expansion = BinOp("+", expansion, Name(c))

    def rewrite(n):
        if isinstance(n, Dot):
            return expansion
        if isinstance(n, BinOp):
            return BinOp(n.op, rewrite(n.left), rewrite(n.right))
        if isinstance(n, UnaryOp):
            return UnaryOp(n.op, rewrite(n.operand))
        if isinstance(n, Paren):
            return Paren(rewrite(n.expr))
        # Calls, Subscripts: Don't expand `.` inside function args — R doesn't.
        return n

    return rewrite(node)


def _response_names(lhs) -> set[str]:
    """Collect bare names on the LHS so `.` expansion can skip them.

    For `y ~ .` the response is `y`; for `cbind(s, f) ~ .` it's {s, f}; for
    `100/mpg ~ .` it's {mpg} (all Names reachable through arithmetic).
    """
    out: set[str] = set()

    def walk(n):
        if n is None:
            return
        if isinstance(n, Name):
            out.add(n.ident)
        elif isinstance(n, BinOp):
            walk(n.left); walk(n.right)
        elif isinstance(n, UnaryOp):
            walk(n.operand)
        elif isinstance(n, Paren):
            walk(n.expr)
        elif isinstance(n, Call):
            for a in n.args:
                walk(a)
            for v in n.kwargs.values():
                walk(v)

    walk(lhs)
    return out


def expand(
    formula: Formula,
    data_columns: Optional[list[str]] = None,
) -> ExpandedFormula:
    """Expand a parsed formula's RHS into intercept + ordered term list + bars.

    If the RHS contains `.` (all-vars sentinel), `data_columns` is required;
    LHS-response names are automatically excluded from the expansion.
    """
    rhs = formula.rhs
    # Dot expansion happens before term algebra.
    if _contains_dot(rhs):
        if data_columns is None:
            raise ValueError("formula contains '.' but no data_columns supplied")
        rhs = _expand_dot(rhs, data_columns, _response_names(formula.lhs))

    bars = _collect_bars(rhs)
    # Implicit intercept: prepend `1 +`.
    augmented = BinOp("+", Literal(1, "num"), rhs)
    all_terms = _expand_toplevel(augmented)
    intercept = _EMPTY_TERM in all_terms
    fixed_terms = [t for t in all_terms if t != _EMPTY_TERM]

    # Extract offsets: a term whose single atom is `offset(...)`. R's terms()
    # filters these out of term.labels and stores them under the "offset"
    # attribute. If an offset shows up mixed into an interaction (e.g.
    # `offset(x):y`), that's not meaningful R; we leave it in terms.
    #
    # Same treatment for mgcv smooth constructors s/te/ti/t2: they don't
    # contribute to the parametric X; they become per-smooth (X_block, S_blocks)
    # pairs in `materialize_smooths`.
    offsets: list = []
    smooths: list[Call] = []
    kept: list[Term] = []
    for t in fixed_terms:
        if len(t.atoms) == 1 and isinstance(t.atoms[0], Call):
            c = t.atoms[0]
            if c.fn == "offset":
                if c.args:
                    offsets.append(c.args[0])
                continue
            if c.fn in ("s", "te", "ti", "t2"):
                smooths.append(c)
                continue
        kept.append(t)

    # R's terms() sorts by interaction order (main effects → pairwise → …),
    # with ties broken by first-appearance. Python's sort is stable.
    kept.sort(key=lambda t: len(t.atoms))

    return ExpandedFormula(
        intercept=intercept, terms=kept, bars=bars, offsets=offsets,
        smooths=smooths,
    )


def _contains_dot(node) -> bool:
    if isinstance(node, Dot):
        return True
    if isinstance(node, BinOp):
        return _contains_dot(node.left) or _contains_dot(node.right)
    if isinstance(node, UnaryOp):
        return _contains_dot(node.operand)
    if isinstance(node, Paren):
        return _contains_dot(node.expr)
    # Intentionally don't recurse into Call args — R doesn't expand `.` there.
    return False


# ---------------------------------------------------------------------------
# Materialization — turn a parsed + expanded formula into a design matrix X.
# ---------------------------------------------------------------------------
#
# Layered pipeline:
#   1. Per-atom evaluation (`_eval_atom`) returns either a numeric column block
#      or a factor record (codes + levels + optional forced contrast).
#   2. Per-term encoding (`_encode_term`) applies contrast matrices to factors
#      using R's promote1 rule (walk atoms; the first factor whose "hole" isn't
#      already covered by an earlier term gets FULL coding, others REDUCED).
#   3. Row-wise Khatri-Rao product across atom blocks produces the term's
#      columns; column names follow R's convention (atom labels joined by `:`).
#
# Current scope: identity/log/exp/sqrt/abs/scale, I(…), arithmetic,
# factor()/as.factor()/ordered()/C() with treatment/sum/helmert/SAS/poly,
# raw-mode poly() (for matching R's `poly(x, n, raw = TRUE)` columns).
# Not yet: bs(), ns(), orthogonal poly(), cut(), pmin/pmax.


import locale as _locale  # noqa: E402
import math  # noqa: E402
import numpy as np  # noqa: E402  — kept near usage to localize heavy import
import pandas as pd  # noqa: E402


# R's factor() sorts levels via locale-aware `sort(unique(x))`. On macOS the
# default is en_US.UTF-8, under which e.g. "<1l" sorts before "1-1.5l" and
# "-" sorts before "+". Pure ASCII sort diverges on punctuation. We try to
# match by using the same locale for our level-ordering key.
def _factor_sort_key(x):
    return _locale.strxfrm(str(x))


try:
    _locale.setlocale(_locale.LC_COLLATE, "en_US.UTF-8")
except _locale.Error:
    pass  # fall back to whatever collation was already set


# R constants that are not data columns. When a Name matches one of these, it
# resolves to the value here rather than a df lookup.
_R_CONSTANTS = {
    "pi": math.pi,
}


@dataclass(slots=True)
class _NumBlock:
    """Numeric atom: 2-D values (n, k), one suffix per col, single atom label."""
    values: np.ndarray
    suffixes: list[str]  # what gets appended to `label` per column
    label: str


@dataclass(slots=True)
class _FactorBlock:
    """Categorical atom: integer codes into `levels`, with optional forced contrast."""
    codes: np.ndarray         # (n,) int, each in range(len(levels))
    levels: list              # ordered list of level values
    ordered: bool
    label: str                # e.g. "Species" or "C(Species, contr.sum)"
    forced_contrast: Optional[str] = None  # override for default treatment/poly
    how_many: Optional[int] = None  # C(f, _, how.many): truncate contrast cols


def _as_float(x) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype.kind in "ui":
        return arr.astype(float)
    if arr.dtype.kind == "b":
        return arr.astype(float)
    if arr.dtype.kind == "f":
        return arr
    return arr.astype(float)


def _series(data: pd.DataFrame, name: str) -> pd.Series:
    if name not in data.columns:
        raise KeyError(f"column {name!r} not in data")
    return data[name]


def _is_categorical(series: pd.Series) -> bool:
    if pd.api.types.is_categorical_dtype(series):
        return True
    if series.dtype == object:
        return True
    return False


def _factor_from_series(series: pd.Series, label: str, ordered_hint: bool = False) -> _FactorBlock:
    if pd.api.types.is_categorical_dtype(series):
        cat = series.cat
        return _FactorBlock(
            codes=np.asarray(cat.codes),
            levels=list(cat.categories),
            ordered=bool(cat.ordered) or ordered_hint,
            label=label,
        )
    # R's factor() uses locale-aware sort(unique(x)) on strings, but numeric
    # columns sort numerically (factor() first coerces to character and R's
    # sort on numerics is numeric when the input was numeric).
    uniq = series.dropna().unique().tolist()
    if pd.api.types.is_numeric_dtype(series):
        levels = sorted(uniq)
    else:
        levels = sorted(uniq, key=_factor_sort_key)
    code_map = {lv: i for i, lv in enumerate(levels)}
    codes = np.array([code_map.get(v, -1) for v in series], dtype=int)
    return _FactorBlock(codes=codes, levels=levels, ordered=ordered_hint, label=label)


def _eval_maybe_string(node, data: pd.DataFrame) -> np.ndarray:
    """Like `_eval_numeric` but preserves string dtype for comparison branches."""
    if isinstance(node, Literal) and node.kind == "str":
        return np.full(len(data), node.value, dtype=object)
    if isinstance(node, Name):
        if node.ident in _R_CONSTANTS:
            return np.full(len(data), float(_R_CONSTANTS[node.ident]))
        s = _series(data, node.ident)
        if s.dtype == object or pd.api.types.is_categorical_dtype(s):
            return s.to_numpy()
        return _as_float(s.to_numpy())
    # Fallback to numeric; comparison ops on numeric are fine.
    return _eval_numeric(node, data)


def _eval_numeric(node, data: pd.DataFrame) -> np.ndarray:
    """Evaluate a node to a 1-D float array, assuming it's strictly numeric.

    Used inside `I(...)` and as argument evaluation for numeric-only builtins.
    Does NOT handle factor atoms — those go through `_eval_atom` instead.
    """
    if isinstance(node, Paren):
        return _eval_numeric(node.expr, data)
    if isinstance(node, Name):
        if node.ident in _R_CONSTANTS:
            return np.full(len(data), float(_R_CONSTANTS[node.ident]))
        s = _series(data, node.ident)
        return _as_float(s.to_numpy())
    if isinstance(node, Literal):
        if node.kind == "num":
            return np.full(len(data), float(node.value))
        if node.kind == "bool":
            return np.full(len(data), 1.0 if node.value else 0.0)
        if node.kind == "str":
            # Strings don't have a numeric value, but in comparison contexts
            # (`I(f == "Ctl")`) they need to flow through BinOp. Carry the
            # value via an object array so BinOp's comparison branch can
            # compare elementwise with the other side.
            return np.full(len(data), node.value, dtype=object)
        raise TypeError(f"non-numeric literal in numeric context: {node!r}")
    if isinstance(node, UnaryOp):
        v = _eval_numeric(node.operand, data)
        if node.op == "-":
            return -v
        if node.op == "+":
            return v
        if node.op == "!":
            return (v == 0).astype(float)
        raise TypeError(f"unsupported unary op {node.op!r}")
    if isinstance(node, BinOp):
        op = node.op
        if op == "$":
            # pandas column accessor: data$col  => data[col]
            if isinstance(node.left, Name) and isinstance(node.right, Name):
                return _as_float(_series(data, node.right.ident).to_numpy())
            raise TypeError("`$` only supported as `data$col`")
        # Comparisons may operate on strings (e.g. `I(f == "Ctl")`). Evaluate
        # sides with type preserved, then do the comparison, then convert to
        # float.
        if op in ("==", "!=", "<", ">", "<=", ">="):
            l = _eval_maybe_string(node.left, data)
            r = _eval_maybe_string(node.right, data)
            if op == "==":  return (l == r).astype(float)
            if op == "!=":  return (l != r).astype(float)
            if op == "<":   return (l < r).astype(float)
            if op == ">":   return (l > r).astype(float)
            if op == "<=":  return (l <= r).astype(float)
            if op == ">=":  return (l >= r).astype(float)
        l = _eval_numeric(node.left, data)
        r = _eval_numeric(node.right, data)
        if op == "+":   return l + r
        if op == "-":   return l - r
        if op == "*":   return l * r
        if op == "/":   return l / r
        if op == "^":   return l ** r
        raise TypeError(f"unsupported binop {op!r} in numeric context")
    if isinstance(node, Call):
        block = _eval_call(node, data)
        if isinstance(block, _FactorBlock):
            raise TypeError(f"factor-producing call {node.fn!r} used in numeric context")
        if block.values.shape[1] != 1:
            raise TypeError(f"multi-column call {node.fn!r} used in numeric context")
        return block.values[:, 0]
    if isinstance(node, Subscript):
        base = _eval_numeric(node.obj, data)
        # Only single integer-literal index supported for now (e.g. `b.d[1]`).
        if len(node.idx) == 1 and isinstance(node.idx[0], Literal) and node.idx[0].kind == "num":
            i = int(node.idx[0].value) - 1  # R is 1-indexed
            return np.full(len(data), float(base[i]) if base.ndim == 1 else float(base[i, 0]))
        raise TypeError("complex subscripts not yet supported")
    raise TypeError(f"cannot numerically evaluate {type(node).__name__}")


# Function-call atom evaluator: returns _NumBlock or _FactorBlock.
def _eval_call(call: Call, data: pd.DataFrame):
    fn = call.fn
    label = _deparse(call)

    if fn == "I":
        # `I(e)` protects e from formula algebra; evaluate as pure numeric.
        v = _eval_numeric(call.args[0], data)
        return _NumBlock(values=v.reshape(-1, 1), suffixes=[""], label=label)

    if fn in ("log", "exp", "sqrt", "abs", "cos", "sin", "tan", "expm1", "log1p", "log2", "log10"):
        v = _eval_numeric(call.args[0], data)
        f = {
            "log": lambda x: np.log(x) if "base" not in call.kwargs
                            else np.log(x) / np.log(_eval_numeric(call.kwargs["base"], data)),
            "exp": np.exp, "sqrt": np.sqrt, "abs": np.abs,
            "cos": np.cos, "sin": np.sin, "tan": np.tan,
            "expm1": np.expm1, "log1p": np.log1p,
            "log2": np.log2, "log10": np.log10,
        }[fn]
        return _NumBlock(values=f(v).reshape(-1, 1), suffixes=[""], label=label)

    if fn == "scale":
        v = _eval_numeric(call.args[0], data)
        center = call.kwargs.get("center")
        scale_ = call.kwargs.get("scale")
        c = True if center is None else (isinstance(center, Literal) and center.value is True)
        s = True if scale_ is None else (isinstance(scale_, Literal) and scale_.value is True)
        out = v.copy()
        if c:
            out = out - out.mean()
        if s:
            sd = out.std(ddof=1)
            if sd != 0:
                out = out / sd
        return _NumBlock(values=out.reshape(-1, 1), suffixes=[""], label=label)

    if fn in ("factor", "as.factor"):
        # First arg = variable; kwargs may include levels, labels, ordered.
        src = call.args[0]
        if isinstance(src, Name):
            s = _series(data, src.ident)
        else:
            s = pd.Series(_eval_numeric(src, data))
        ordered = False
        if "ordered" in call.kwargs:
            ok = call.kwargs["ordered"]
            ordered = isinstance(ok, Literal) and ok.value is True
        blk = _factor_from_series(s, label=label, ordered_hint=ordered or (fn == "ordered"))
        if "levels" in call.kwargs:
            lvl_node = call.kwargs["levels"]
            lvls = _eval_level_list(lvl_node)
            # Recode to the explicit level order.
            code_map = {lv: i for i, lv in enumerate(lvls)}
            new_codes = np.array([code_map.get(v, -1) for v in s], dtype=int)
            blk = _FactorBlock(codes=new_codes, levels=lvls, ordered=blk.ordered, label=label)
        return blk

    if fn == "ordered":
        # Same as factor() but ordered=TRUE, and default contrast becomes poly.
        src = call.args[0]
        s = _series(data, src.ident) if isinstance(src, Name) else pd.Series(_eval_numeric(src, data))
        blk = _factor_from_series(s, label=label, ordered_hint=True)
        if "levels" in call.kwargs:
            lvls = _eval_level_list(call.kwargs["levels"])
            code_map = {lv: i for i, lv in enumerate(lvls)}
            new_codes = np.array([code_map.get(v, -1) for v in s], dtype=int)
            blk = _FactorBlock(codes=new_codes, levels=lvls, ordered=True, label=label)
        return blk

    if fn == "dummy":
        # lme4's dummy(f, level) → 0/1 indicator for `f == level`.
        src = call.args[0]
        s = _series(data, src.ident) if isinstance(src, Name) else pd.Series(_eval_numeric(src, data))
        level_node = call.args[1] if len(call.args) >= 2 else call.kwargs.get("level")
        if isinstance(level_node, Literal):
            level = level_node.value
        elif isinstance(level_node, Name):
            level = level_node.ident
        else:
            raise TypeError("dummy(): second arg must be a literal level")
        values = (s.to_numpy() == level).astype(float).reshape(-1, 1)
        return _NumBlock(values=values, suffixes=[""], label=label)

    if fn == "relevel":
        # relevel(f, ref) — move `ref` to position 0.
        inner = _eval_atom(call.args[0], data)
        if not isinstance(inner, _FactorBlock):
            if isinstance(call.args[0], Name):
                inner = _factor_from_series(_series(data, call.args[0].ident), label=label)
            else:
                raise TypeError("relevel() requires a factor-like first argument")
        ref_node = call.kwargs.get("ref")
        if ref_node is None and len(call.args) >= 2:
            ref_node = call.args[1]
        if isinstance(ref_node, Literal):
            ref = ref_node.value
        elif isinstance(ref_node, Name):
            ref = ref_node.ident
        else:
            raise TypeError("relevel(ref=) must be a literal")
        if ref not in inner.levels:
            raise ValueError(f"relevel: ref {ref!r} not in levels {inner.levels}")
        new_levels = [ref] + [lv for lv in inner.levels if lv != ref]
        old_to_new = {lv: i for i, lv in enumerate(new_levels)}
        new_codes = np.array(
            [old_to_new[inner.levels[c]] if c >= 0 else -1 for c in inner.codes],
            dtype=int,
        )
        return _FactorBlock(
            codes=new_codes, levels=new_levels, ordered=inner.ordered, label=label,
        )

    if fn == "cut":
        # cut(x, breaks, labels=NULL, right=TRUE) — bin numeric into factor.
        x = _eval_numeric(call.args[0], data)
        breaks_node = call.args[1] if len(call.args) >= 2 else call.kwargs.get("breaks")
        if isinstance(breaks_node, Call) and breaks_node.fn == "c":
            breaks = np.array([
                float(a.value) if isinstance(a, Literal)
                else float(_eval_numeric(a, data)[0])
                for a in breaks_node.args
            ], dtype=float)
        elif isinstance(breaks_node, Literal) and breaks_node.kind == "num":
            # cut(x, 3) → 3 equal-width breaks between min and max
            n_breaks = int(breaks_node.value)
            lo, hi = x.min(), x.max()
            # R widens endpoints by 0.1% like cut.default does
            span = hi - lo
            lo2, hi2 = lo - 0.001 * span, hi + 0.001 * span
            breaks = np.linspace(lo2, hi2, n_breaks + 1)
        else:
            raise TypeError(f"cut(): can't parse breaks from {breaks_node!r}")
        right_kw = call.kwargs.get("right")
        right = True if right_kw is None else bool(right_kw.value)
        # Build R-style level labels: "(a,b]" or "[a,b)"
        def _fmt(v):
            return f"{v:g}"
        if right:
            level_labels = [f"({_fmt(breaks[i])},{_fmt(breaks[i+1])}]" for i in range(len(breaks) - 1)]
            # np.digitize(x, bins, right=True): returns i such that bins[i-1] < x <= bins[i]
            idx = np.digitize(x, breaks, right=True) - 1
        else:
            level_labels = [f"[{_fmt(breaks[i])},{_fmt(breaks[i+1])})" for i in range(len(breaks) - 1)]
            idx = np.digitize(x, breaks, right=False) - 1
        # Values outside breaks become NA (code -1)
        mask = (idx < 0) | (idx >= len(level_labels))
        idx = np.where(mask, -1, idx)
        return _FactorBlock(codes=idx.astype(int), levels=level_labels, ordered=False, label=label)

    if fn == "C":
        # C(f, contrast, how.many) — wrap factor with explicit contrast choice.
        inner = _eval_atom(call.args[0], data)
        if not isinstance(inner, _FactorBlock):
            if isinstance(call.args[0], Name):
                inner = _factor_from_series(_series(data, call.args[0].ident), label=_deparse(call.args[0]))
            else:
                raise TypeError("C() requires a factor-like first argument")
        forced = None
        base_kw = call.kwargs.get("base")
        if len(call.args) >= 2 and not isinstance(call.args[1], Empty):
            c = call.args[1]
            if isinstance(c, Name):
                forced = c.ident  # contr.treatment → "contr.treatment", etc.
            elif isinstance(c, Literal) and c.kind == "str":
                forced = str(c.value)
        if base_kw is not None and isinstance(base_kw, Literal) and base_kw.kind == "num":
            # C(f, base=2) chooses the 2nd level as reference (1-indexed in R).
            forced = f"contr.treatment:base={int(base_kw.value)}"
        how_many = None
        hm_kw = call.kwargs.get("how.many")
        if hm_kw is not None and isinstance(hm_kw, Literal) and hm_kw.kind == "num":
            how_many = int(hm_kw.value)
        elif len(call.args) >= 3 and isinstance(call.args[2], Literal) and call.args[2].kind == "num":
            how_many = int(call.args[2].value)
        return _FactorBlock(
            codes=inner.codes, levels=inner.levels, ordered=inner.ordered,
            label=label, forced_contrast=forced, how_many=how_many,
        )

    if fn == "poly":
        # Raw polynomials only for now — matches `poly(x, n, raw = TRUE)`.
        v = _eval_numeric(call.args[0], data)
        degree = int(call.args[1].value) if len(call.args) >= 2 and isinstance(call.args[1], Literal) \
            else int(call.kwargs["degree"].value) if "degree" in call.kwargs \
            else 1
        raw_k = call.kwargs.get("raw")
        is_raw = isinstance(raw_k, Literal) and raw_k.value is True
        if not is_raw:
            cols = _poly_orthogonal(v, degree)
        else:
            cols = np.stack([v ** d for d in range(1, degree + 1)], axis=1)
        suffixes = [str(d) for d in range(1, degree + 1)]
        return _NumBlock(values=cols, suffixes=suffixes, label=label)

    if fn == "bs":
        v = _eval_numeric(call.args[0], data)
        degree = _int_kw_or_arg(call, "degree", default=3)
        df = _int_kw_or_arg(call, "df", default=None)
        knots_node = call.kwargs.get("knots")
        if knots_node is None and len(call.args) >= 3:
            knots_node = call.args[2]
        interior = _parse_knots(knots_node, data)
        bnd = _parse_boundary(call.kwargs.get("Boundary.knots"), data, v)
        intercept_kw = call.kwargs.get("intercept")
        intercept = isinstance(intercept_kw, Literal) and intercept_kw.value is True
        cols = _bs_basis(v, degree, bnd, interior, df, intercept)
        suffixes = [str(i + 1) for i in range(cols.shape[1])]
        return _NumBlock(values=cols, suffixes=suffixes, label=label)

    if fn == "ns":
        v = _eval_numeric(call.args[0], data)
        df = _int_kw_or_arg(call, "df", default=None)
        knots_node = call.kwargs.get("knots")
        if knots_node is None and len(call.args) >= 2 and not isinstance(call.args[1], Literal):
            # ns(x, knots=...) via positional — but ns's 2nd positional is df.
            pass
        # ns(x, df=k): df = len(interior_knots) + 1 + intercept. Interior knots
        # at evenly-spaced quantiles of x, like R's ns default.
        interior = _parse_knots(knots_node, data)
        if df is None and len(call.args) >= 2 and isinstance(call.args[1], Literal):
            df = int(call.args[1].value)
        bnd = _parse_boundary(call.kwargs.get("Boundary.knots"), data, v)
        intercept_kw = call.kwargs.get("intercept")
        intercept = isinstance(intercept_kw, Literal) and intercept_kw.value is True
        cols = _ns_basis(v, bnd, interior, df, intercept)
        suffixes = [str(i + 1) for i in range(cols.shape[1])]
        return _NumBlock(values=cols, suffixes=suffixes, label=label)

    # Fallback: try to treat as numeric single-argument elementwise function.
    raise NotImplementedError(f"unsupported call: {fn}(…)")


def _int_kw_or_arg(call, kw, default):
    node = call.kwargs.get(kw)
    if node is None:
        return default
    if isinstance(node, Literal) and node.kind == "num":
        return int(node.value)
    return default


def _parse_knots(node, data):
    if node is None:
        return np.array([], dtype=float)
    if isinstance(node, Call) and node.fn == "c":
        return np.array([
            float(a.value) if isinstance(a, Literal) else float(_eval_numeric(a, data)[0])
            for a in node.args
        ], dtype=float)
    if isinstance(node, Literal) and node.kind == "num":
        return np.array([float(node.value)], dtype=float)
    return np.array([], dtype=float)


def _parse_boundary(node, data, x):
    if node is None:
        return (float(x.min()), float(x.max()))
    if isinstance(node, Call) and node.fn == "c":
        vals = [float(a.value) if isinstance(a, Literal) else float(_eval_numeric(a, data)[0])
                for a in node.args]
        return (vals[0], vals[1])
    return (float(x.min()), float(x.max()))


def _bs_basis(x, degree, boundary, interior_knots, df, intercept):
    """B-spline basis matching R's bs(). Returns (n, df) matrix."""
    from scipy.interpolate import BSpline as _BSpline
    ord = degree + 1
    # If df given and no explicit knots, place interior knots at quantiles.
    if df is not None and len(interior_knots) == 0:
        n_interior = df - degree - (1 if intercept else 0)
        if n_interior > 0:
            probs = np.linspace(0, 1, n_interior + 2)[1:-1]
            interior_knots = np.quantile(x, probs)
    Aknots = np.concatenate([
        np.repeat(boundary[0], ord),
        np.sort(np.asarray(interior_knots, dtype=float)),
        np.repeat(boundary[1], ord),
    ])
    n_basis = len(Aknots) - ord
    out = np.zeros((len(x), n_basis))
    # Evaluate each basis function. scipy's BSpline returns NaN outside [t[k], t[n]]
    # but we want the right endpoint included, so clamp.
    xe = np.clip(x, boundary[0], boundary[1])
    for i in range(n_basis):
        c = np.zeros(n_basis)
        c[i] = 1.0
        spl = _BSpline(Aknots, c, degree, extrapolate=False)
        y = spl(xe)
        # scipy gives 0 (not NaN) for points outside the half-open interval at
        # the right boundary for all but the last basis. Force the last basis
        # to 1 at the right endpoint (R's behavior).
        out[:, i] = np.nan_to_num(y, nan=0.0)
    # At exact right boundary, scipy's half-open interval gives 0 for all
    # basis functions; R gives 1 for the last. Patch points == boundary[1].
    right_mask = x == boundary[1]
    if right_mask.any():
        out[right_mask, :] = 0
        out[right_mask, -1] = 1.0
    if not intercept:
        out = out[:, 1:]
    return out


def _ns_basis(x, boundary, interior_knots, df, intercept):
    """Natural cubic spline basis matching R's ns().

    ns() starts from bs(x, degree=3, knots=..., Boundary.knots=...,
    intercept=TRUE) and postmultiplies by a matrix H that enforces zero
    second derivative at each boundary knot. Returns (n, df) matrix with
    first column dropped if intercept=FALSE.
    """
    from scipy.interpolate import BSpline as _BSpline
    if df is not None and len(interior_knots) == 0:
        n_interior = df - 1 - (1 if intercept else 0)
        if n_interior > 0:
            probs = np.linspace(0, 1, n_interior + 2)[1:-1]
            interior_knots = np.quantile(x, probs)
    degree = 3
    ord = 4
    interior_knots = np.sort(np.asarray(interior_knots, dtype=float))
    Aknots = np.concatenate([
        np.repeat(boundary[0], ord),
        interior_knots,
        np.repeat(boundary[1], ord),
    ])
    n_basis = len(Aknots) - ord
    # Evaluate basis at x and at boundaries for 2nd derivative constraint.
    def _B(xe):
        xc = np.clip(xe, boundary[0], boundary[1])
        out = np.zeros((len(xe), n_basis))
        for i in range(n_basis):
            c = np.zeros(n_basis); c[i] = 1.0
            out[:, i] = np.nan_to_num(
                _BSpline(Aknots, c, degree, extrapolate=False)(xc), nan=0.0
            )
        right = np.asarray(xe) == boundary[1]
        if right.any():
            out[right, :] = 0; out[right, -1] = 1.0
        return out

    def _Bdd(xe):
        out = np.zeros((len(xe), n_basis))
        for i in range(n_basis):
            c = np.zeros(n_basis); c[i] = 1.0
            out[:, i] = _BSpline(Aknots, c, degree, extrapolate=False).derivative(2)(xe)
        return np.nan_to_num(out, nan=0.0)

    B = _B(x)
    const = _Bdd(np.array([boundary[0], boundary[1]], dtype=float))  # 2 × n_basis
    # R's ns drops the first (intercept) column of basis & const BEFORE the
    # QR-based null-space projection; the order matters because it changes
    # which Q we compute.
    if not intercept:
        B = B[:, 1:]
        const = const[:, 1:]
    Q, _ = np.linalg.qr(const.T, mode="complete")
    H = Q[:, 2:]
    return B @ H


def _eval_level_list(node) -> list:
    """Extract a list of level labels from an AST node like c("a","b","c") or c(1,2,3)."""
    if isinstance(node, Call) and node.fn == "c":
        out = []
        for a in node.args:
            if isinstance(a, Literal):
                out.append(a.value)
            elif isinstance(a, Name):
                out.append(a.ident)
            else:
                raise TypeError(f"can't extract level from {a!r}")
        return out
    if isinstance(node, Literal):
        return [node.value]
    raise TypeError(f"level list expected, got {type(node).__name__}")


def _poly_orthogonal(x: np.ndarray, degree: int) -> np.ndarray:
    """Orthogonal polynomials matching R's `poly(x, degree)` (non-raw).

    R's algorithm: QR on outer(x - mean(x), 0:degree, "^"). The returned
    columns are Q with signs flipped to match R's diagonal, drop constant.
    """
    x = np.asarray(x, dtype=float)
    xc = x - x.mean()
    X = np.column_stack([xc ** d for d in range(degree + 1)])
    Q, R_mat = np.linalg.qr(X)
    signs = np.sign(np.diag(R_mat))
    return (Q * signs)[:, 1:]


def _eval_atom(node, data: pd.DataFrame):
    """Per-atom entry point: returns _NumBlock or _FactorBlock."""
    if isinstance(node, Name):
        s = _series(data, node.ident)
        if _is_categorical(s):
            return _factor_from_series(s, label=node.ident)
        return _NumBlock(values=_as_float(s.to_numpy()).reshape(-1, 1),
                         suffixes=[""], label=node.ident)
    if isinstance(node, Literal):
        if node.kind == "num":
            return _NumBlock(values=np.full((len(data), 1), float(node.value)),
                             suffixes=[""], label=_deparse(node))
        raise TypeError(f"literal atom not supported: {node!r}")
    if isinstance(node, Paren):
        return _eval_atom(node.expr, data)
    if isinstance(node, Call):
        return _eval_call(node, data)
    if isinstance(node, BinOp) and node.op == "$":
        if isinstance(node.left, Name) and isinstance(node.right, Name):
            s = _series(data, node.right.ident)
            if _is_categorical(s):
                return _factor_from_series(s, label=_deparse(node))
            return _NumBlock(values=_as_float(s.to_numpy()).reshape(-1, 1),
                             suffixes=[""], label=_deparse(node))
        raise TypeError("`$` only supported as `df$col`")
    if isinstance(node, (UnaryOp, BinOp, Subscript)):
        v = _eval_numeric(node, data)
        return _NumBlock(values=v.reshape(-1, 1), suffixes=[""], label=_deparse(node))
    raise TypeError(f"cannot evaluate atom {type(node).__name__}")


# ---------------------------------------------------------------------------
# Contrast matrices — map k levels onto either k columns (full) or k-1 (reduced).
# ---------------------------------------------------------------------------

def _contrast_full(k: int) -> np.ndarray:
    return np.eye(k)


def _contrast_treatment(k: int, base: int = 0) -> np.ndarray:
    """Reduced treatment coding: k-by-(k-1). Row `base` is all zeros."""
    M = np.eye(k)
    return np.delete(M, base, axis=1)


def _contrast_SAS(k: int) -> np.ndarray:
    # R's contr.SAS drops the LAST level.
    return _contrast_treatment(k, base=k - 1)


def _contrast_sum(k: int) -> np.ndarray:
    """Reduced sum-to-zero coding: last level is -1 across all columns."""
    M = np.eye(k)[:, : k - 1].astype(float)
    M[-1, :] = -1
    return M


def _contrast_helmert(k: int) -> np.ndarray:
    """R's contr.helmert: column j compares level j+1 to mean of 1..j."""
    M = np.zeros((k, k - 1))
    for j in range(k - 1):
        M[: j + 1, j] = -1
        M[j + 1, j] = j + 1
    return M


def _contrast_poly(k: int) -> np.ndarray:
    """Orthogonal polynomial contrasts on equally-spaced levels 1..k."""
    x = np.arange(1, k + 1, dtype=float)
    return _poly_orthogonal(x, k - 1)


_CONTRAST_FNS = {
    "contr.treatment": _contrast_treatment,
    "contr.SAS":       _contrast_SAS,
    "contr.sum":       _contrast_sum,
    "contr.helmert":   _contrast_helmert,
    "contr.poly":      _contrast_poly,
}


def _contrast_matrix(fb: _FactorBlock, reduced: bool) -> tuple[np.ndarray, list[str]]:
    """Return (contrast_matrix, column_suffixes) for a factor.

    When `reduced=True` we give k-1 columns using the chosen (or default)
    contrast function. When `reduced=False` we give the k-column identity
    coding (full level membership) — used for the first factor in an
    intercept-less model, or when the "hole" left by this atom isn't covered
    by any earlier term.
    """
    k = len(fb.levels)
    if reduced:
        name = fb.forced_contrast
        if name is None:
            name = "contr.poly" if fb.ordered else "contr.treatment"
        # C(f, base=N) is encoded as a pseudo-name.
        if isinstance(name, str) and name.startswith("contr.treatment:base="):
            base = int(name.split("=")[1]) - 1  # R is 1-indexed
            M = _contrast_treatment(k, base=base)
            kept = [str(fb.levels[i]) for i in range(k) if i != base]
            return _truncate_contrast(M, kept, fb.how_many)
        fn = _CONTRAST_FNS.get(name)
        if fn is None:
            raise ValueError(f"unknown contrast {name!r}")
        M = fn(k)
        if name == "contr.treatment":
            suffs = [str(lv) for lv in fb.levels[1:]]
        elif name == "contr.SAS":
            suffs = [str(lv) for lv in fb.levels[:-1]]
        elif name == "contr.sum":
            suffs = [str(i + 1) for i in range(k - 1)]
        elif name == "contr.helmert":
            suffs = [str(i + 1) for i in range(k - 1)]
        elif name == "contr.poly":
            suff = [".L", ".Q", ".C"] + [f"^{i}" for i in range(4, k)]
            suffs = suff[: k - 1]
        else:
            suffs = [str(lv) for lv in fb.levels[1:]]
        return _truncate_contrast(M, suffs, fb.how_many)
    # Full coding: k identity columns named by level.
    return _contrast_full(k), [str(lv) for lv in fb.levels]


def _truncate_contrast(M: np.ndarray, suffs: list[str], how_many):
    if how_many is None or how_many >= M.shape[1]:
        return M, suffs
    return M[:, :how_many], suffs[:how_many]


# ---------------------------------------------------------------------------
# Term encoding + interaction product
# ---------------------------------------------------------------------------


def _encode_factor(fb: _FactorBlock, reduced: bool) -> _NumBlock:
    M, suffs = _contrast_matrix(fb, reduced=reduced)
    # One-hot expand then right-multiply by contrast matrix.
    onehot = np.eye(len(fb.levels))[fb.codes]
    values = onehot @ M
    return _NumBlock(values=values, suffixes=suffs, label=fb.label)


def _khatri_rao(blocks: list[_NumBlock]) -> _NumBlock:
    """Row-wise tensor product across atom blocks.

    R convention: within an interaction, the FIRST atom's levels vary fastest
    and the last atom's vary slowest. Column names are atom labels joined by
    `:`, with each atom's own column suffix glued to its label.
    """
    if len(blocks) == 1:
        blk = blocks[0]
        names = [blk.label + s for s in blk.suffixes]
        return _NumBlock(values=blk.values, suffixes=names, label=blk.label)

    n = blocks[0].values.shape[0]
    cur_values = blocks[0].values
    cur_names = [blocks[0].label + s for s in blocks[0].suffixes]
    for b in blocks[1:]:
        bn = [b.label + s for s in b.suffixes]
        new_v = np.empty((n, cur_values.shape[1] * b.values.shape[1]))
        new_names: list[str] = []
        col = 0
        # Outer loop over the new (rightmost-so-far) block makes it the
        # "slowest varying" axis; inner loop keeps existing accumulator's
        # fast-slow order intact.
        for j, rn in enumerate(bn):
            for i, ln in enumerate(cur_names):
                new_v[:, col] = cur_values[:, i] * b.values[:, j]
                new_names.append(f"{ln}:{rn}")
                col += 1
        cur_values = new_v
        cur_names = new_names
    return _NumBlock(values=cur_values, suffixes=cur_names, label="")


def _term_needs_full_first_factor(term: Term, earlier_terms: list[Term], intercept: bool) -> bool:
    """R's promote1: the first factor atom in a term gets FULL coding iff the
    term formed by removing that atom isn't already present in the model
    (including the empty/intercept term).
    """
    if not term.atoms:
        return False
    # Find first factor atom — we only "check the hole" for it.
    # Build residual = Term(atoms minus first factor index). For the
    # promote1 check we use atom keys.
    # But here we don't know yet which atom is a factor — caller decides.
    # This helper is invoked per-factor candidate; for the outer logic we
    # check the singleton "hole" using Term equality.
    raise NotImplementedError  # placeholder — see _encode_term


def _encode_term(
    term: Term,
    data: pd.DataFrame,
    earlier_terms: list[Term],
    intercept: bool,
) -> _NumBlock:
    """Encode a single term to its numeric column block.

    Applies R's promote1 for factor coding: walk atoms, and the FIRST factor
    whose "hole" (term minus this atom) isn't already covered gets FULL
    coding; other factors stay REDUCED. If intercept is on, the empty term
    is in-model, so all factors are REDUCED when their hole is {}.
    """
    if not term.atoms:
        # Intercept
        return _NumBlock(values=np.ones((len(data), 1)), suffixes=["(Intercept)"], label="")

    atom_blocks: list[_NumBlock | _FactorBlock] = [_eval_atom(a, data) for a in term.atoms]

    # R's promote1: for each factor atom, FULL coding iff its "hole" (this
    # term minus that atom) is NOT in the model (including the intercept/empty
    # term). Multiple factors can get promoted in the same term — e.g.
    # `~ wool:tension` with intercept on produces the full 2×3 = 6 interaction
    # cells because neither {wool} nor {tension} is in the model.
    encoded_blocks: list[_NumBlock] = []
    for i, blk in enumerate(atom_blocks):
        if isinstance(blk, _FactorBlock):
            hole_atoms = tuple(a for j, a in enumerate(term.atoms) if j != i)
            hole = Term(hole_atoms)
            hole_in_model = (hole == _EMPTY_TERM and intercept) or (hole in earlier_terms)
            encoded_blocks.append(_encode_factor(blk, reduced=hole_in_model))
        else:
            encoded_blocks.append(blk)

    return _khatri_rao(encoded_blocks)


def _collect_names(node, names: set[str]) -> None:
    """Gather every data-column Name referenced in a term AST.

    Used to figure out which columns to na.omit-filter before materialization,
    matching R's default `na.action = na.omit` on the model.frame.
    """
    if node is None:
        return
    if isinstance(node, Name):
        if node.ident not in _R_CONSTANTS:
            names.add(node.ident)
    elif isinstance(node, BinOp):
        if node.op == "$":
            # `data$col` — only the right-hand side is a column.
            if isinstance(node.right, Name):
                names.add(node.right.ident)
            return
        _collect_names(node.left, names)
        _collect_names(node.right, names)
    elif isinstance(node, UnaryOp):
        _collect_names(node.operand, names)
    elif isinstance(node, Paren):
        _collect_names(node.expr, names)
    elif isinstance(node, Call):
        for a in node.args:
            _collect_names(a, names)
        for v in node.kwargs.values():
            _collect_names(v, names)
    elif isinstance(node, Subscript):
        _collect_names(node.obj, names)
        for i in node.idx:
            _collect_names(i, names)


def materialize(expanded: ExpandedFormula, data: pd.DataFrame) -> pd.DataFrame:
    """Turn an expanded formula + data frame into a design matrix X.

    NA-omit: rows with missing values in any referenced column are dropped
    first (R's `na.action = na.omit` default). Column order: intercept (if
    on), then each term in `expanded.terms` order, with term-internal column
    order matching R's interaction convention (first atom's levels vary
    fastest). Column names follow R: `(Intercept)`, `x`, `fb`, `x:fb`,
    `I(x^2)`, etc.
    """
    referenced: set[str] = set()
    for t in expanded.terms:
        for a in t.atoms:
            _collect_names(a, referenced)
    for b in expanded.bars:
        _collect_names(b, referenced)
    for o in expanded.offsets:
        _collect_names(o, referenced)
    referenced &= set(data.columns)
    if referenced:
        keep = ~data[list(referenced)].isna().any(axis=1)
        data = data.loc[keep]

    blocks: list[_NumBlock] = []
    running_terms: list[Term] = []

    if expanded.intercept:
        blocks.append(_encode_term(_EMPTY_TERM, data, running_terms, expanded.intercept))
        running_terms.append(_EMPTY_TERM)

    for t in expanded.terms:
        blocks.append(_encode_term(t, data, running_terms, expanded.intercept))
        running_terms.append(t)

    all_values = np.hstack([b.values for b in blocks]) if blocks else np.zeros((len(data), 0))
    all_names: list[str] = []
    for b in blocks:
        all_names.extend(b.suffixes)
    return pd.DataFrame(all_values, columns=all_names, index=data.index)


# ---------------------------------------------------------------------------
# lme4 random-effect bars → Z, Λᵀ template, θ
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ReTerms:
    """Materialized random-effect side of a mixed-effects formula.

    Matches the layout lme4 produces internally:
      * ``Z``        — ``(n, q)`` dense RE design matrix.
      * ``Lambdat``  — ``(q, q)`` integer template; each nonzero's value is a
        1-indexed θ position so callers can fill in real parameters later.
      * ``theta``    — ``(n_theta,)`` initial θ (identity: 1 on the diagonal
        of each per-level Cholesky factor, 0 off-diagonal).
      * ``flist_names`` / ``flist_levels`` / ``cnms`` / ``Gp`` — bookkeeping
        that mirrors lme4's ``reTrms`` fields for downstream consumers.
    """
    Z: np.ndarray
    Lambdat: np.ndarray
    theta: np.ndarray
    flist_names: list[str]
    flist_levels: dict[str, list]
    cnms: dict[str, object]
    Gp: list[int]


def _bar_lhs_to_ef(lhs_node) -> ExpandedFormula:
    """Treat a bar's LHS as the RHS of a fake formula and expand it."""
    return expand(Formula(lhs=None, rhs=lhs_node))


def _materialize_re_lhs(lhs_ef: ExpandedFormula, data: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Materialize a bar's LHS as a dense (n, c) matrix + component names.

    Uses the same code path as the fixed-effect materializer: contrast-coded
    factors, Khatri–Rao for interactions, promote1 for hole-filling.
    """
    blocks: list[_NumBlock] = []
    running_terms: list[Term] = []
    if lhs_ef.intercept:
        blocks.append(_encode_term(_EMPTY_TERM, data, running_terms, lhs_ef.intercept))
        running_terms.append(_EMPTY_TERM)
    for t in lhs_ef.terms:
        blocks.append(_encode_term(t, data, running_terms, lhs_ef.intercept))
        running_terms.append(t)
    if not blocks:
        return np.zeros((len(data), 0)), []
    values = np.hstack([b.values for b in blocks])
    names: list[str] = []
    for b in blocks:
        names.extend(b.suffixes)
    return values, names


def _flatten_nested_group(node) -> list:
    """Expand ``a/b`` in a bar's RHS. ``a/b`` → ``[a, b:a]`` (lme4 names the
    interaction child:parent). Deeper chains ``a/b/c`` recurse the same way.
    """
    if isinstance(node, Paren):
        return _flatten_nested_group(node.expr)
    if isinstance(node, BinOp) and node.op == "/":
        left = _flatten_nested_group(node.left)
        right = _flatten_nested_group(node.right)
        out = list(left)
        # Every level on the right nests under the deepest existing level.
        for r in right:
            out.append(BinOp(op=":", left=r, right=out[-1]))
        return out
    return [node]


def _eval_group(node, data: pd.DataFrame) -> tuple[np.ndarray, list, str]:
    """Resolve a bar's grouping expression → (codes, levels, label)."""
    if isinstance(node, Paren):
        return _eval_group(node.expr, data)
    if isinstance(node, Name):
        s = _series(data, node.ident)
        fb = _factor_from_series(s, label=node.ident)
        return fb.codes, fb.levels, node.ident
    if isinstance(node, BinOp) and node.op == ":":
        lc, lv, ln = _eval_group(node.left, data)
        rc, rv, rn = _eval_group(node.right, data)
        label = f"{ln}:{rn}"
        n = len(lc)
        # Sort pairs by (l_code, r_code) — since lv and rv are already in
        # canonical order, this gives lex order on level identities.
        seen: set[tuple[int, int]] = set()
        for i in range(n):
            if lc[i] >= 0 and rc[i] >= 0:
                seen.add((int(lc[i]), int(rc[i])))
        ordered = sorted(seen)
        pair_to_idx = {p: i for i, p in enumerate(ordered)}
        codes = np.array([
            pair_to_idx.get((int(lc[i]), int(rc[i])), -1) if lc[i] >= 0 and rc[i] >= 0 else -1
            for i in range(n)
        ], dtype=int)
        levels = [f"{lv[a]}:{rv[b]}" for a, b in ordered]
        return codes, levels, label
    if isinstance(node, Call):
        blk = _eval_atom(node, data)
        if isinstance(blk, _FactorBlock):
            return blk.codes, blk.levels, _deparse(node)
    # Fallback: evaluate as numeric then factor-ize.
    try:
        v = _eval_numeric(node, data)
        s = pd.Series(v)
        fb = _factor_from_series(s, label=_deparse(node))
        return fb.codes, fb.levels, _deparse(node)
    except Exception as e:
        raise TypeError(f"can't resolve grouping {_deparse(node)!r}: {e}") from e


def materialize_bars(expanded: ExpandedFormula, data: pd.DataFrame) -> ReTerms:
    """Build lme4's Z / Λᵀ template / θ from the bars of an expanded formula.

    Matches lme4's conventions: bars with nested groupings ``a/b`` split into
    ``(lhs|a) + (lhs|b:a)``; ``(lhs||g)`` splits into independent scalar bars
    per LHS column; final bar order is stable-sorted by descending number of
    grouping-factor levels. θ is initialized to the identity Cholesky factor.
    """
    referenced: set[str] = set()
    for t in expanded.terms:
        for a in t.atoms:
            _collect_names(a, referenced)
    for b in expanded.bars:
        _collect_names(b, referenced)
    for o in expanded.offsets:
        _collect_names(o, referenced)
    referenced &= set(data.columns)
    if referenced:
        keep = ~data[list(referenced)].isna().any(axis=1)
        data = data.loc[keep]
    n = len(data)

    # Normalize each parsed bar into (lhs_matrix, cnames, group_codes,
    # group_levels, group_label). For `||` split LHS into scalar bars; for
    # nested `a/b` split group into [a, b:a].
    @dataclass
    class _SimpleBar:
        Z_lhs: np.ndarray         # (n, c)
        cnames: list[str]         # component names (length c)
        g_codes: np.ndarray       # (n,) int codes into g_levels
        g_levels: list
        g_label: str

    simple: list[_SimpleBar] = []
    for bar in expanded.bars:
        if not (isinstance(bar, BinOp) and bar.op in ("|", "||")):
            continue
        lhs_node = bar.left
        group_nodes = _flatten_nested_group(bar.right)
        is_double = bar.op == "||"
        lhs_ef = _bar_lhs_to_ef(lhs_node)
        if is_double:
            # Split LHS: intercept (if any) as one scalar bar, each term as
            # another (with intercept=False so it stays a single component).
            lhs_parts: list[ExpandedFormula] = []
            if lhs_ef.intercept:
                lhs_parts.append(ExpandedFormula(
                    intercept=True, terms=[], bars=[], offsets=[],
                ))
            for t in lhs_ef.terms:
                lhs_parts.append(ExpandedFormula(
                    intercept=False, terms=[t], bars=[], offsets=[],
                ))
        else:
            lhs_parts = [lhs_ef]
        for g_node in group_nodes:
            g_codes, g_levels, g_label = _eval_group(g_node, data)
            for lef in lhs_parts:
                Z_lhs, cnames = _materialize_re_lhs(lef, data)
                if Z_lhs.shape[1] == 0:
                    continue
                simple.append(_SimpleBar(
                    Z_lhs=Z_lhs, cnames=cnames,
                    g_codes=g_codes, g_levels=g_levels, g_label=g_label,
                ))

    # Sort by descending #levels of the grouping factor (stable).
    simple.sort(key=lambda sb: -len(sb.g_levels))

    # Build Z, Lambdat, theta per-bar.
    Z_blocks: list[np.ndarray] = []
    Lt_sizes: list[int] = []        # per-bar q contribution
    Lt_templates: list[np.ndarray] = []  # per-bar full (k*c, k*c) template
    theta_parts: list[np.ndarray] = []
    theta_offset = 0

    Gp = [0]
    flist_names: list[str] = []
    flist_levels: dict[str, list] = {}
    cnms: dict[str, object] = {}

    for sb in simple:
        c = sb.Z_lhs.shape[1]
        k = len(sb.g_levels)
        n_theta_block = c * (c + 1) // 2

        # Z: column = level * c + component
        Zb = np.zeros((n, k * c))
        valid = sb.g_codes >= 0
        lvl = sb.g_codes[valid]
        rows = np.where(valid)[0]
        for comp in range(c):
            Zb[rows, lvl * c + comp] = sb.Z_lhs[rows, comp]
        Z_blocks.append(Zb)

        # Per-level c×c upper-triangular template. lme4 stores Λ (lower) in
        # column-major, so Λᵀ's upper triangle is filled row-by-row: θ[0] at
        # (0,0), θ[1] at (0,1), θ[2] at (0,2), θ[3] at (1,1), θ[4] at (1,2),
        # θ[5] at (2,2), ...
        tmpl = np.zeros((c, c), dtype=int)
        idx = 0
        for i in range(c):
            for j in range(i, c):
                idx += 1
                tmpl[i, j] = theta_offset + idx
        Ltb = np.zeros((k * c, k * c), dtype=int)
        for l in range(k):
            Ltb[l*c:(l+1)*c, l*c:(l+1)*c] = tmpl
        Lt_templates.append(Ltb)
        Lt_sizes.append(k * c)

        # Initial theta: identity Cholesky (1 on diag of Λ, 0 elsewhere).
        theta_block = np.zeros(n_theta_block)
        idx = 0
        for i in range(c):
            for j in range(i, c):
                idx += 1
                if i == j:
                    theta_block[idx - 1] = 1.0
        theta_parts.append(theta_block)

        theta_offset += n_theta_block
        Gp.append(Gp[-1] + k * c)

        # flist / cnms bookkeeping.
        gname = sb.g_label
        if gname not in flist_names:
            flist_names.append(gname)
            flist_levels[gname] = list(sb.g_levels)
        cnms_key = gname
        # If this group appears in multiple bars, lme4 suffixes .1, .2, ...
        suffix = 0
        while cnms_key in cnms:
            suffix += 1
            cnms_key = f"{gname}.{suffix}"
        cnms[cnms_key] = sb.cnames if c > 1 else sb.cnames[0]

    q_total = sum(Lt_sizes)
    if q_total == 0:
        Z = np.zeros((n, 0))
        Lambdat = np.zeros((0, 0), dtype=int)
        theta = np.zeros(0)
    else:
        Z = np.hstack(Z_blocks)
        Lambdat = np.zeros((q_total, q_total), dtype=int)
        off = 0
        for blk in Lt_templates:
            s = blk.shape[0]
            Lambdat[off:off+s, off:off+s] = blk
            off += s
        theta = np.concatenate(theta_parts)

    return ReTerms(
        Z=Z, Lambdat=Lambdat, theta=theta,
        flist_names=flist_names, flist_levels=flist_levels,
        cnms=cnms, Gp=Gp,
    )


# ---------------------------------------------------------------------------
# mgcv smooth constructors → per-smooth (X, S_list)
# ---------------------------------------------------------------------------
#
# mgcv's `smoothCon(sp, data, absorb.cons=TRUE, scale.penalty=TRUE)` does:
#   1. Dispatch on class(sp) → smooth.construct.<bs>.smooth.spec
#      - re:  X = model.matrix(~ term1:term2:...-1, data);  S = [I]
#      - cr:  cubic regression spline basis with 2nd-derivative penalty
#      - tp:  thin-plate regression spline (eigen-reduced from n basis)
#      - tensor (te/ti/t2): tensor product of marginal bases
#      - ...
#   2. Absorb sum-to-zero constraint (drops 1 col from X, reparameterizes S).
#   3. Rescale each S[i] so `norm(S[i],"O")/ncol(S[i])` matches
#      `norm(X,"I")^2/ncol(X)` — makes penalty magnitudes comparable to X'X
#      for numerical conditioning.


@dataclass(slots=True)
class SmoothBlock:
    """One per-smooth basis + penalty set.

    Produced by `materialize_smooths`. Matches what R's
    `smoothCon(..., absorb.cons=TRUE, scale.penalty=TRUE)` returns for each
    block under a given smooth.spec.
    """
    label: str                      # e.g. "s(x)", "s(Machine,Worker)"
    term: list[str]                 # variable names referenced
    cls: str                        # class name (e.g. "re.smooth.spec")
    X: np.ndarray                   # basis matrix, (n, k)
    S: list[np.ndarray]             # penalty matrices, each (k, k)


def _smooth_bs(call: Call) -> str:
    """Pick the bs string for an s()/te()/ti()/t2() call."""
    if call.fn in ("te", "ti", "t2"):
        # Tensor constructors default to cr marginals but the class is
        # `tensor.smooth.spec` / `t2.smooth.spec` regardless. The bs kwarg
        # there controls marginal bs.
        return "tensor"
    # s(): default bs is "tp"
    bs = call.kwargs.get("bs")
    if bs is None:
        return "tp"
    if isinstance(bs, Literal) and bs.kind == "str":
        return str(bs.value)
    return "tp"


def _smooth_term_vars(call: Call) -> list[str]:
    """Pluck variable names from an s(...)'s positional args.

    mgcv treats the non-keyword args of s() as the term variables. e.g.
    `s(Machine, Worker, bs="re")` → ["Machine", "Worker"].
    """
    names: list[str] = []
    for a in call.args:
        if isinstance(a, Name):
            names.append(a.ident)
        else:
            # Could be an expression — not supported yet.
            names.append(_deparse(a))
    return names


def _smooth_label(call: Call) -> str:
    """Reproduce mgcv's smooth label (e.g. `s(x)`, `s(Machine,Worker)`)."""
    return f"{call.fn}({','.join(_smooth_term_vars(call))})"


def _scale_penalty(X: np.ndarray, S_list: list[np.ndarray]) -> list[np.ndarray]:
    """Match mgcv's `scale.penalty=TRUE` rescaling.

    For each penalty S in S_list:
        scale = (norm(X,"I")^2 / ncol(X)) / (norm(S,"O") / ncol(S))
        S := S * scale
    where norm("I") = max abs row sum, norm("O") = max abs col sum.
    """
    if X.size == 0 or not S_list:
        return [np.asarray(s, dtype=float) for s in S_list]
    maXX = float(np.abs(X).sum(axis=1).max()) ** 2 / X.shape[1]
    out = []
    for S in S_list:
        S = np.asarray(S, dtype=float)
        if S.size == 0:
            out.append(S)
            continue
        maS = float(np.abs(S).sum(axis=0).max()) / S.shape[1]
        if maS == 0:
            out.append(S)
        else:
            out.append(S * (maXX / maS))
    return out


def _build_re_smooth(call: Call, data: pd.DataFrame) -> list[SmoothBlock]:
    """`bs="re"` → `model.matrix(~ term1:term2:...-1, data)`.

    Penalty is `diag(ncol(X))`, then scale.penalty rescales.
    """
    term_vars = _smooth_term_vars(call)
    # Build the formula `~ term1:term2:...-1` using our existing machinery.
    rhs = None
    for v in term_vars:
        node = Name(v)
        rhs = node if rhs is None else BinOp(":", rhs, node)
    # `-1`: no intercept.
    fake = Formula(lhs=None, rhs=rhs)
    ef = expand(fake)
    ef.intercept = False  # override implicit intercept
    X_df = materialize(ef, data)
    X = X_df.values.astype(float)
    S_list = [np.eye(X.shape[1])]
    S_list = _scale_penalty(X, S_list)
    return [SmoothBlock(
        label=_smooth_label(call),
        term=term_vars,
        cls="re.smooth.spec",
        X=X,
        S=S_list,
    )]


def materialize_smooths(
    expanded: ExpandedFormula, data: pd.DataFrame,
) -> list[list[SmoothBlock]]:
    """Materialize each smooth in `expanded.smooths` to one or more blocks.

    Returns a list parallel to `expanded.smooths`; each entry is the list of
    SmoothBlock that mgcv's smoothCon returns for that spec. Most smooths
    produce exactly one block; `by = <factor>` with `id` yields n_levels.
    """
    # NA-drop using every referenced variable, to match R's `na.omit` over the
    # union of all formula variables.
    referenced: set[str] = set()
    for c in expanded.smooths:
        for v in _smooth_term_vars(c):
            if v in data.columns:
                referenced.add(v)
    if referenced:
        data = data.dropna(subset=list(referenced)).reset_index(drop=True)

    out: list[list[SmoothBlock]] = []
    for call in expanded.smooths:
        bs = _smooth_bs(call)
        if bs == "re":
            out.append(_build_re_smooth(call, data))
        else:
            raise NotImplementedError(
                f"smooth bs={bs!r} (class dispatch for {_smooth_label(call)}) "
                "not yet implemented"
            )
    return out
