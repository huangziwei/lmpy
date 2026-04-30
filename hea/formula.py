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
    # Cached identity key: deparsed-atom frozenset. Excluded from init/repr/
    # compare/hash so the dataclass machinery treats it as a pure cache. We
    # still override __hash__/__eq__ below to use it explicitly.
    _key: frozenset = field(
        init=False, repr=False, compare=False, hash=False,
        default_factory=frozenset,
    )

    def __post_init__(self) -> None:
        # Frozen + slots: bypass __setattr__ to populate the cache once.
        object.__setattr__(
            self, "_key", frozenset(_deparse(a) for a in self.atoms)
        )

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


import contextlib  # noqa: E402
import contextvars  # noqa: E402
import locale as _locale  # noqa: E402
import math  # noqa: E402
import numpy as np  # noqa: E402  — kept near usage to localize heavy import
import polars as pl  # noqa: E402
from scipy.linalg import eigh_tridiagonal as _eigh_tridiagonal  # noqa: E402


# Polars 1.40+ made pl.Categorical process-global (shared string cache across
# DataFrames), so hea can no longer use pl.Enum vs pl.Categorical as the
# ordered-vs-unordered factor signal — only pl.Enum preserves per-column level
# order. Callers instead declare ordered columns via `with_ordered_cols(...)`
# (or the helper `set_ordered_cols`) before materializing. `_factor_from_series`
# consults this context to decide whether a factor should use poly contrasts.
_ORDERED_COLS_CV: contextvars.ContextVar[frozenset[str]] = contextvars.ContextVar(
    "_hea_ordered_cols", default=frozenset()
)


@contextlib.contextmanager
def with_ordered_cols(cols):
    """Context manager: inside the `with` block, the given column names are
    treated as R-style ordered factors (poly contrasts, drop-first in `by=`)."""
    token = _ORDERED_COLS_CV.set(frozenset(cols))
    try:
        yield
    finally:
        _ORDERED_COLS_CV.reset(token)


def set_ordered_cols(cols):
    """Set the ordered-columns context without restoring on exit. Intended for
    test harnesses that reset per-test via a fixture rather than a `with`."""
    _ORDERED_COLS_CV.set(frozenset(cols))


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


def _series(data: pl.DataFrame, name: str) -> pl.Series:
    if name not in data.columns:
        raise KeyError(f"column {name!r} not in data")
    return data[name]


def _is_categorical(series: pl.Series) -> bool:
    dt = series.dtype
    if dt in (pl.Categorical, pl.Enum):
        return True
    if dt in (pl.String, pl.Utf8, pl.Object):
        return True
    return False


def _factor_from_series(series: pl.Series, label: str, ordered_hint: bool = False) -> _FactorBlock:
    dt = series.dtype
    if dt in (pl.Categorical, pl.Enum):
        # polars 1.40+ gives all pl.Categorical columns in one DataFrame a merged
        # string pool, so cat.get_categories() returns every category seen across
        # sibling columns and to_physical() indexes into that merged pool. Drop
        # levels absent from this column (matches R's droplevels semantics lme4
        # and mgcv use when building Z / model matrices) and remap codes densely.
        #
        # Enum declares its own per-column level pool so we can read levels via
        # the dtype (~10× faster than .cat.get_categories()) and skip the remap
        # when every declared level appears (the common case for schema-cast
        # fixtures).
        if dt == pl.Enum:
            full = dt.categories.to_list()
        else:
            full = series.cat.get_categories().to_list()
        codes_raw = series.to_physical().to_numpy()
        null_mask = series.is_null().to_numpy() if series.null_count() > 0 else None
        valid = codes_raw if null_mask is None else codes_raw[~null_mask]
        k_full = len(full)
        if valid.size == 0:
            levels = []
            codes = np.empty(0, dtype=np.int64)
        else:
            present_max = int(valid.max())
            # Fast path: all declared levels are present. np.bincount with
            # minlength=k_full is O(n) and avoids the np.unique sort.
            if present_max < k_full and valid.size >= k_full and \
                    np.all(np.bincount(valid.astype(np.intp, copy=False), minlength=k_full) > 0):
                levels = full
                codes = codes_raw.astype(np.int64, copy=False)
            else:
                present = np.unique(valid)
                levels = [full[int(c)] for c in present]
                remap = np.full(k_full + 1, -1, dtype=np.int64)
                for new, old in enumerate(present):
                    remap[int(old)] = new
                codes = remap[codes_raw]
        if null_mask is not None:
            codes = np.where(null_mask, -1, codes)
        codes = codes.astype(int, copy=False)
        # Ordered-factor signal: polars has no native ordered-factor dtype, and
        # polars 1.40+ made pl.Categorical process-global (so we can't use the
        # Enum/Categorical split anymore either). Callers declare ordered cols
        # via `with_ordered_cols(...)`; the explicit `ordered_hint` wins when
        # the call site already knows (e.g. `ordered(x)` in a formula).
        ordered = ordered_hint or (label in _ORDERED_COLS_CV.get())
        return _FactorBlock(codes=codes, levels=levels, ordered=ordered, label=label)
    # R's factor() uses locale-aware sort(unique(x)) on strings, but numeric
    # columns sort numerically (factor() first coerces to character and R's
    # sort on numerics is numeric when the input was numeric).
    values = series.to_numpy()
    null_mask = series.is_null().to_numpy() if series.null_count() > 0 else None
    if null_mask is not None:
        uniq = series.drop_nulls().unique().to_list()
    else:
        uniq = series.unique().to_list()
    if dt.is_numeric():
        levels = sorted(uniq)
    else:
        levels = sorted(uniq, key=_factor_sort_key)
    code_map = {lv: i for i, lv in enumerate(levels)}
    if null_mask is not None:
        codes = np.array(
            [-1 if null_mask[i] else code_map.get(values[i], -1) for i in range(len(values))],
            dtype=int,
        )
    else:
        codes = np.array([code_map.get(v, -1) for v in values], dtype=int)
    ordered = ordered_hint or (label in _ORDERED_COLS_CV.get())
    return _FactorBlock(codes=codes, levels=levels, ordered=ordered, label=label)


def _eval_maybe_string(node, data: pl.DataFrame) -> np.ndarray:
    """Like `_eval_numeric` but preserves string dtype for comparison branches."""
    if isinstance(node, Literal) and node.kind == "str":
        return np.full(len(data), node.value, dtype=object)
    if isinstance(node, Name):
        if node.ident in _R_CONSTANTS:
            return np.full(len(data), float(_R_CONSTANTS[node.ident]))
        s = _series(data, node.ident)
        if s.dtype in (pl.Categorical, pl.Enum, pl.String, pl.Utf8, pl.Object):
            return s.to_numpy()
        return _as_float(s.to_numpy())
    # Fallback to numeric; comparison ops on numeric are fine.
    return _eval_numeric(node, data)


def _eval_numeric(node, data: pl.DataFrame) -> np.ndarray:
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
            # DataFrame column accessor: data$col  => data[col]
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
def _eval_call(call: Call, data: pl.DataFrame):
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
            s = pl.Series(_eval_numeric(src, data))
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
            new_codes = np.array(
                [code_map.get(v, -1) for v in s.to_list()], dtype=int
            )
            blk = _FactorBlock(codes=new_codes, levels=lvls, ordered=blk.ordered, label=label)
        return blk

    if fn == "ordered":
        # Same as factor() but ordered=TRUE, and default contrast becomes poly.
        src = call.args[0]
        s = _series(data, src.ident) if isinstance(src, Name) else pl.Series(_eval_numeric(src, data))
        blk = _factor_from_series(s, label=label, ordered_hint=True)
        if "levels" in call.kwargs:
            lvls = _eval_level_list(call.kwargs["levels"])
            code_map = {lv: i for i, lv in enumerate(lvls)}
            new_codes = np.array(
                [code_map.get(v, -1) for v in s.to_list()], dtype=int
            )
            blk = _FactorBlock(codes=new_codes, levels=lvls, ordered=True, label=label)
        return blk

    if fn == "dummy":
        # lme4's dummy(f, level) → 0/1 indicator for `f == level`.
        src = call.args[0]
        s = _series(data, src.ident) if isinstance(src, Name) else pl.Series(_eval_numeric(src, data))
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
                inner = _factor_from_series(
                    _series(data, call.args[0].ident), label=label,
                )
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
    """B-spline basis matching R's `splines::bs()`. Returns (n, df) matrix.
    Parametric-term helper used from `_eval_call` (e.g. `y ~ bs(x, df=10)`).
    NOT mgcv's `s(x, bs="bs")` smoother — that's `_build_bs_smooth` below.
    """
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


def _eval_atom(node, data: pl.DataFrame, cache: dict | None = None):
    """Per-atom entry point: returns _NumBlock or _FactorBlock.

    `cache`, if provided, memoizes Name/`df$col` atoms within a single
    materialize call so interaction-heavy formulas (e.g. `A*B*C*D`) don't
    re-encode the same factor column once per term.
    """
    if isinstance(node, Name):
        if cache is not None:
            hit = cache.get(node.ident)
            if hit is not None:
                return hit
        s = _series(data, node.ident)
        if _is_categorical(s):
            blk = _factor_from_series(s, label=node.ident)
        else:
            blk = _NumBlock(values=_as_float(s.to_numpy()).reshape(-1, 1),
                            suffixes=[""], label=node.ident)
        if cache is not None:
            cache[node.ident] = blk
        return blk
    if isinstance(node, Literal):
        if node.kind == "num":
            return _NumBlock(values=np.full((len(data), 1), float(node.value)),
                             suffixes=[""], label=_deparse(node))
        raise TypeError(f"literal atom not supported: {node!r}")
    if isinstance(node, Paren):
        return _eval_atom(node.expr, data, cache)
    if isinstance(node, Call):
        return _eval_call(node, data)
    if isinstance(node, BinOp) and node.op == "$":
        if isinstance(node.left, Name) and isinstance(node.right, Name):
            key = ("$", node.left.ident, node.right.ident) if cache is not None else None
            if cache is not None:
                hit = cache.get(key)
                if hit is not None:
                    return hit
            s = _series(data, node.right.ident)
            if _is_categorical(s):
                blk = _factor_from_series(s, label=_deparse(node))
            else:
                blk = _NumBlock(values=_as_float(s.to_numpy()).reshape(-1, 1),
                                suffixes=[""], label=_deparse(node))
            if cache is not None:
                cache[key] = blk
            return blk
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
    # `eye(k)[codes] @ M` is a dressed-up row gather; `M[codes]` is the same
    # result without allocating the k×k identity or running a GEMM.
    values = np.ascontiguousarray(M[fb.codes])
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
    data: pl.DataFrame,
    earlier_terms: list[Term],
    intercept: bool,
    cache: dict | None = None,
) -> _NumBlock:
    """Encode a single term to its numeric column block.

    Applies R's promote1 for factor coding: walk atoms, and the FIRST factor
    whose "hole" (term minus this atom) isn't already covered gets FULL
    coding; other factors stay REDUCED. If intercept is on, the empty term
    is in-model, so all factors are REDUCED when their hole is {}.

    `cache` is forwarded to `_eval_atom` so an outer loop (e.g. `materialize`)
    can memoize per-atom encoding across sibling terms in interaction-heavy
    formulas like `A*B*C*D`.
    """
    if not term.atoms:
        # Intercept
        return _NumBlock(values=np.ones((len(data), 1)), suffixes=["(Intercept)"], label="")

    atom_blocks: list[_NumBlock | _FactorBlock] = [_eval_atom(a, data, cache) for a in term.atoms]

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
            # Cache encoded factor blocks: in a full crossing like `A*B*C*D`
            # each factor is always REDUCED in every term it appears in, so
            # one _encode_factor call suffices per (factor, reduced) pair
            # instead of once per term. Safe to share the same _NumBlock
            # reference — _khatri_rao and downstream only read `values`.
            # Key by label rather than id(blk): Call-node atoms aren't in
            # atom_cache so their _FactorBlock refs can be GC'd between
            # terms, causing id() reuse to collide across different factors.
            if cache is not None:
                enc_key = ("enc", blk.label, hole_in_model)
                enc = cache.get(enc_key)
                if enc is None:
                    enc = _encode_factor(blk, reduced=hole_in_model)
                    cache[enc_key] = enc
                encoded_blocks.append(enc)
            else:
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


def referenced_columns(expanded: ExpandedFormula) -> set[str]:
    """Every data-column Name referenced by any term, bar, offset, or smooth.

    Public helper used by the NA-omit paths in ``materialize`` /
    ``materialize_bars`` and by ``hea.design.prepare_design`` (so prepare
    can align the response to the NA-cleaned X without relying on a
    shared index — polars has none). Smooths must be included so that
    ``prepare_design`` drops NAs on smooth-only variables (e.g.
    ``y ~ s(x)``) — otherwise the parametric design and the smooth basis
    produced by ``materialize_smooths`` end up with different row counts.
    """
    referenced: set[str] = set()
    for t in expanded.terms:
        for a in t.atoms:
            _collect_names(a, referenced)
    for b in expanded.bars:
        _collect_names(b, referenced)
    for o in expanded.offsets:
        _collect_names(o, referenced)
    for s in expanded.smooths:
        _collect_names(s, referenced)
    return referenced


def materialize(expanded: ExpandedFormula, data: pl.DataFrame) -> pl.DataFrame:
    """Turn an expanded formula + data frame into a design matrix X.

    NA-omit: rows with missing values in any referenced column are dropped
    first (R's `na.action = na.omit` default). Column order: intercept (if
    on), then each term in `expanded.terms` order, with term-internal column
    order matching R's interaction convention (first atom's levels vary
    fastest). Column names follow R: `(Intercept)`, `x`, `fb`, `x:fb`,
    `I(x^2)`, etc.
    """
    referenced = referenced_columns(expanded) & set(data.columns)
    ref_list = list(referenced)
    if ref_list and any(data[c].null_count() > 0 for c in ref_list):
        data = data.drop_nulls(subset=ref_list)

    blocks: list[_NumBlock] = []
    running_terms: list[Term] = []
    atom_cache: dict = {}

    if expanded.intercept:
        blocks.append(_encode_term(_EMPTY_TERM, data, running_terms, expanded.intercept, atom_cache))
        running_terms.append(_EMPTY_TERM)

    for t in expanded.terms:
        blocks.append(_encode_term(t, data, running_terms, expanded.intercept, atom_cache))
        running_terms.append(t)

    all_names: list[str] = []
    for b in blocks:
        all_names.extend(b.suffixes)
    if not blocks or sum(b.values.shape[1] for b in blocks) == 0:
        # Polars can't represent (n, 0); return an empty frame and let
        # callers use ``len(input_data)`` if they need the row count.
        return pl.DataFrame()
    all_values = np.hstack([b.values for b in blocks])
    return pl.from_numpy(all_values, schema=all_names)


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


def _materialize_re_lhs(lhs_ef: ExpandedFormula, data: pl.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Materialize a bar's LHS as a dense (n, c) matrix + component names.

    Uses the same code path as the fixed-effect materializer: contrast-coded
    factors, Khatri–Rao for interactions, promote1 for hole-filling.
    """
    blocks: list[_NumBlock] = []
    running_terms: list[Term] = []
    atom_cache: dict = {}
    if lhs_ef.intercept:
        blocks.append(_encode_term(_EMPTY_TERM, data, running_terms, lhs_ef.intercept, atom_cache))
        running_terms.append(_EMPTY_TERM)
    for t in lhs_ef.terms:
        blocks.append(_encode_term(t, data, running_terms, lhs_ef.intercept, atom_cache))
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


def _eval_group(node, data: pl.DataFrame) -> tuple[np.ndarray, list, str]:
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
        s = pl.Series(v)
        fb = _factor_from_series(s, label=_deparse(node))
        return fb.codes, fb.levels, _deparse(node)
    except Exception as e:
        raise TypeError(f"can't resolve grouping {_deparse(node)!r}: {e}") from e


def materialize_bars(expanded: ExpandedFormula, data: pl.DataFrame) -> ReTerms:
    """Build lme4's Z / Λᵀ template / θ from the bars of an expanded formula.

    Matches lme4's conventions: bars with nested groupings ``a/b`` split into
    ``(lhs|a) + (lhs|b:a)``; ``(lhs||g)`` splits into independent scalar bars
    per LHS column; final bar order is stable-sorted by descending number of
    grouping-factor levels. θ is initialized to the identity Cholesky factor.
    """
    referenced = referenced_columns(expanded) & set(data.columns)
    ref_list = list(referenced)
    if ref_list and any(data[c].null_count() > 0 for c in ref_list):
        data = data.drop_nulls(subset=ref_list)
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

    `spec` carries the predict-time state (raw-basis evaluator + optional
    `by=` and absorb-constraint replays). It is the hea port of mgcv's
    `Predict.matrix.<class>` dispatch — calling `spec.predict_mat(new_data)`
    rebuilds the design rows for new x in the same parameterization the fit
    used.
    """
    label: str                      # e.g. "s(x)", "s(Machine,Worker)"
    term: list[str]                 # variable names referenced
    cls: str                        # class name (e.g. "re.smooth.spec")
    X: np.ndarray                   # basis matrix, (n, k)
    S: list[np.ndarray]             # penalty matrices, each (k, k)
    spec: Optional["BasisSpec"] = None   # predict-time replay state


# ---------------------------------------------------------------------------
# Predict.matrix machinery
#
# Each smooth block carries a `BasisSpec` that fully captures the state needed
# to re-evaluate its design rows on new data. mgcv's `Predict.matrix.<class>`
# dispatch is replicated here as a small class hierarchy: a `_RawBasis`
# subclass per bs evaluates the raw (pre-by, pre-absorb) basis at new x; a
# `_ByMask` (optional) replays factor / numeric `by=` masking; an
# `_AbsorbTransform` (optional) replays the sum-to-zero rotation that
# `_absorb_sumzero` applied during fitting.
#
# Each `_build_*_smooth` constructs the appropriate `_RawBasis`, hands it
# to `_apply_by_and_absorb`, and the resulting block's `spec` field carries
# the complete chain. `block.spec.predict_mat(self.data) ≈ block.X` is a
# strong sanity invariant — checked by `tests/test_smooths_predict.py`.
# ---------------------------------------------------------------------------


class _RawBasis:
    """Per-bs raw basis evaluator. ``eval(data) → (n, k_pre)`` matrix.

    Concrete subclasses store whatever state mgcv's smooth.construct stashes
    in ``object$xt`` (knots, eigenvectors, scale factors, …) so the same
    basis can be evaluated on new data.
    """
    def eval(self, data: pl.DataFrame) -> np.ndarray:
        raise NotImplementedError


@dataclass(slots=True)
class _AbsorbTransform:
    """Replay ``_absorb_sumzero`` on new-data rows.

    Two paths matching the in-sample code:
      * full path (``full_Z`` set): ``X_new = X @ full_Z``.
      * sparse path (``indi``/``Z_sub``/``keep_mask`` set): ``X[:, indi]`` is
        rotated by ``Z_sub``, the result placed back at ``indi[:nz]``, then
        the dropped column is removed via ``keep_mask``.
    """
    full_Z: Optional[np.ndarray] = None
    indi: Optional[np.ndarray] = None
    Z_sub: Optional[np.ndarray] = None
    keep_mask: Optional[np.ndarray] = None

    def apply(self, X: np.ndarray) -> np.ndarray:
        if self.full_Z is not None:
            return X @ self.full_Z
        if self.indi is None:
            return X
        nz = len(self.indi) - 1
        X_new = X.copy()
        if nz > 0:
            X_new[:, self.indi[:nz]] = X[:, self.indi] @ self.Z_sub
        return X_new[:, self.keep_mask]


@dataclass(slots=True)
class _ByMask:
    """Replay ``by=`` masking. Factor: indicator ``by_col == level``. Numeric:
    multiply by ``by_col`` value.
    """
    expr: str
    kind: str        # "factor" | "numeric"
    level: object = None  # for factor; None for numeric

    def apply(self, X: np.ndarray, data: pl.DataFrame) -> np.ndarray:
        col = _eval_by_col(self.expr, data)
        if self.kind == "factor":
            arr = col.to_numpy() if isinstance(col, pl.Series) else col
            mask = (arr == self.level).astype(float)
            return X * mask[:, None]
        # numeric
        if isinstance(col, pl.Series):
            arr = col.to_numpy().astype(float)
        else:
            arr = np.asarray(col, dtype=float)
        return X * arr[:, None]


@dataclass(slots=True)
class BasisSpec:
    """Predict-time state for one SmoothBlock — chains raw → by → absorb → keep_cols.

    ``keep_cols`` is set by ``gam.side`` when overlapping smooths force the
    drop of linearly-dependent columns from this block's design (matches
    mgcv's ``fixDependence``). All other steps run unchanged.

    ``predict_raw`` overrides ``raw`` at predict time when mgcv's predict
    basis differs from the fit basis. The case is ``t2``: ``smoothCon``
    returns the partial absorb (``sm$X``) for fit, but ``PredictMat`` applies
    the full absorb (via ``sm$qrc`` from ``sm$Cp``). The two span different
    24-d subspaces of the same 25-d raw column space, so no remap from one
    to the other exists; we instead carry a separate raw evaluator for
    predict that re-applies the full-absorb ``Z_p`` from raw.

    ``coef_remap`` is set when fit and predict bases differ (also t2 only):
    after fit, β must be transformed so ``predict_mat(new) @ β`` matches
    ``X_fit @ β_partial`` from the partial-absorb fit. Stored as ``(M, X̄)``
    such that ``X_fit = 1·X̄ + X_predict @ M`` exactly (both in- and
    out-of-sample). Mirrors mgcv's ``G$P`` post-fit transform in
    ``estimate.gam`` (smooth.r:264-267).
    """
    raw: _RawBasis
    by: Optional[_ByMask] = None
    absorb: Optional[_AbsorbTransform] = None
    keep_cols: Optional[np.ndarray] = None
    predict_raw: Optional[_RawBasis] = None
    coef_remap: Optional[tuple[np.ndarray, np.ndarray]] = None

    def predict_mat(self, data: pl.DataFrame) -> np.ndarray:
        raw = self.predict_raw if self.predict_raw is not None else self.raw
        X = raw.eval(data)
        if self.by is not None:
            X = self.by.apply(X, data)
        if self.absorb is not None:
            X = self.absorb.apply(X)
        if self.keep_cols is not None:
            X = X[:, self.keep_cols]
        return X


# ---- Concrete _RawBasis subclasses (one per bs) -----------------------------
#
# Each class stores exactly the state mgcv's smooth.construct.<bs>.smooth.spec
# stashes in `object` for predict-time evaluation. Method `eval(data)` mirrors
# `Predict.matrix.<class>(object, data)` — returns the (n, k_pre) raw basis on
# new rows, before by-masking and absorb.cons replays (which BasisSpec layers
# on top).


@dataclass(slots=True)
class _CRRawBasis(_RawBasis):
    """`smooth.construct.cr.smooth.spec` — natural cubic regression spline."""
    term: str
    knots: np.ndarray

    def eval(self, data: pl.DataFrame) -> np.ndarray:
        x = data[self.term].to_numpy().astype(float)
        return _cr_basis(x, self.knots)


@dataclass(slots=True)
class _CCRawBasis(_RawBasis):
    """`smooth.construct.cc.smooth.spec` — cyclic cubic regression spline."""
    term: str
    knots: np.ndarray
    BD: np.ndarray

    def eval(self, data: pl.DataFrame) -> np.ndarray:
        x = data[self.term].to_numpy().astype(float)
        return _cc_basis(x, self.knots, self.BD)


@dataclass(slots=True)
class _PSRawBasis(_RawBasis):
    """`smooth.construct.ps.smooth.spec` — Eilers & Marx P-spline."""
    term: str
    knots: np.ndarray
    m0: int

    def eval(self, data: pl.DataFrame) -> np.ndarray:
        x = data[self.term].to_numpy().astype(float)
        return _ps_basis(x, self.knots, self.m0)


@dataclass(slots=True)
class _BSRawBasis(_RawBasis):
    """`smooth.construct.bs.smooth.spec` — mgcv's B-spline (NOT splines::bs)."""
    term: str
    knots: np.ndarray
    m0: int

    def eval(self, data: pl.DataFrame) -> np.ndarray:
        x = data[self.term].to_numpy().astype(float)
        return _bs_design(x, self.knots, self.m0, deriv=0)


@dataclass(slots=True)
class _CPRawBasis(_RawBasis):
    """`smooth.construct.cp.smooth.spec` — cyclic P-spline."""
    term: str
    knots: np.ndarray
    ord_: int

    def eval(self, data: pl.DataFrame) -> np.ndarray:
        x = data[self.term].to_numpy().astype(float)
        return _cp_basis(x, self.knots, self.ord_)


@dataclass(slots=True)
class _GPRawBasis(_RawBasis):
    """`smooth.construct.gp.smooth.spec` — Gaussian-process / Kammann–Wand.

    Stores the resolved kernel definition (`defn`), the centered knot grid
    (`xu_c`), the kept eigenvectors (`UZ`), and the data shift; predict
    rebuilds [E(x_c, xu_c) @ UZ | T(x_c)] using the same defn so rho is
    fixed at fit time.
    """
    term: list[str]
    shift: np.ndarray
    xu_c: np.ndarray
    defn: tuple[float, float, float]
    UZ: np.ndarray
    stationary: bool

    def eval(self, data: pl.DataFrame) -> np.ndarray:
        x_full = np.column_stack(
            [data[v].to_numpy().astype(float) for v in self.term]
        )
        x_c = x_full - self.shift
        E_x = _gp_E_with_defn(x_c, self.xu_c, self.defn)
        X_radial = E_x @ self.UZ
        T_mat = _gp_T(x_c, self.stationary)
        return np.hstack([X_radial, T_mat])


@dataclass(slots=True)
class _TPRawBasis(_RawBasis):
    """`smooth.construct.tp.smooth.spec` — thin-plate regression spline.

    Predict is `[η(||x_c - Xu_c||) | T(x_c)] @ UZ`, with column-norm
    rescaling `w` reapplied after — same chain as `_tp_raw` builds at fit.
    """
    term: list[str]
    shift: np.ndarray
    Xu: np.ndarray   # unique, centered knot grid (nu, d)
    m: int
    d: int
    M: int
    k: int
    UZ: np.ndarray   # (nu + M, k)
    w: np.ndarray    # column-norm rescaling, length k

    def eval(self, data: pl.DataFrame) -> np.ndarray:
        x_full = np.column_stack(
            [data[v].to_numpy().astype(float) for v in self.term]
        )
        x_c = x_full - self.shift
        n = x_c.shape[0]
        nu = self.Xu.shape[0]
        eta0 = _tp_eta_const(self.m, self.d)
        # Pairwise η(||x_i - Xu_j||): (n, nu).
        diff = x_c[:, None, :] - self.Xu[None, :, :]
        rsq = (diff * diff).sum(axis=-1)
        E = _tp_fast_eta_vec(self.m, self.d, rsq.ravel(), eta0).reshape(n, nu)
        T_mat = _tp_T(x_c, self.m, self.d)
        # b = [E | T] is (n, nu + M); X_raw = b @ UZ then rescale by w.
        b = np.hstack([E, T_mat])
        X_raw = b @ self.UZ
        return X_raw / self.w


@dataclass(slots=True)
class _TPDropNullRawBasis(_RawBasis):
    """tp with `m = c(m, 0)` (null space dropped). Wraps a full tp basis,
    keeps the first ``keep`` columns, and subtracts the fit-time column
    means — the centering mgcv applies in this branch (mgcv: smooth.r,
    `if (object$m[2]==0)`).
    """
    inner: _RawBasis
    keep: int
    col_means: np.ndarray  # fit-time column means, shape (keep,)

    def eval(self, data: pl.DataFrame) -> np.ndarray:
        X = self.inner.eval(data)
        return X[:, : self.keep] - self.col_means[None, :]


@dataclass(slots=True)
class _ADRawBasis(_RawBasis):
    """`smooth.construct.ad.smooth.spec` — adaptive P-spline (1D or 2D).

    Underlying basis is ps (1D) or row-Kron of two ps margins (2D). The
    adaptive part lives entirely in S, so predict re-evaluates the ps basis
    at new x.
    """
    term: list[str]
    knots_per_term: list[np.ndarray]
    m0: int
    k_per_term: list[int]

    def eval(self, data: pl.DataFrame) -> np.ndarray:
        cols = [
            data[t].to_numpy().astype(float) for t in self.term
        ]
        bases = [
            _ps_basis(cols[i], self.knots_per_term[i], self.m0)
            for i in range(len(self.term))
        ]
        if len(bases) == 1:
            return bases[0]
        # 2D: row-wise Kronecker, term[0] inner (matches _build_ad_smooth).
        Xi, Xj = bases
        n = Xi.shape[0]
        ki, kj = self.k_per_term
        return (Xi[:, :, None] * Xj[:, None, :]).reshape(n, ki * kj)


@dataclass(slots=True)
class _RERawBasis(_RawBasis):
    """`smooth.construct.re.smooth.spec` — random-effect indicator basis.

    Predict reproduces mgcv's PredictMat output: one column per fit-time
    column even if new data exercises fewer levels (those columns are
    all-zero on rows that don't match any fit-time combo). ``combos[j]`` is
    the tuple of factor values defining column j (single-element tuple for
    1D re; multi-element for `s(f1, f2, bs="re")`-style interactions).
    """
    term: list[str]
    combos: list[tuple]

    def eval(self, data: pl.DataFrame) -> np.ndarray:
        n = data.shape[0]
        cols = [data[t].to_numpy() for t in self.term]
        out = np.zeros((n, len(self.combos)))
        for j, combo in enumerate(self.combos):
            mask = np.ones(n, dtype=bool)
            for ci, lev in zip(cols, combo):
                mask &= (ci == lev)
            out[:, j] = mask.astype(float)
        return out


@dataclass(slots=True)
class _FSRawBasis(_RawBasis):
    """`smooth.construct.fs.smooth.spec` — factor-smooth interaction.

    Predict replays:
      1. base tp (or other) basis evaluated at new x via stored ``base_raw``
      2. nat.param(type=1) reparameterization via ``P`` (so X_r = Xb @ P);
         we store ``Xr_T = P`` directly — the post-canonicalization rotation
         is captured by ``null_rot`` (eigenvectors of Xn'Xn) and ``null_signs``
      3. block-wise duplicate across factor levels, masking by `data[fterm]`
    The full result has shape ``(n, p * nf)`` matching the fit-time block.
    """
    fterm: str
    flev: list
    p: int
    rank: int
    null_d: int
    base_raw: _RawBasis  # the inner tp/etc. raw basis
    P: np.ndarray   # nat.param transform: Xr = Xb @ P
    null_rot: Optional[np.ndarray]  # (null_d, null_d) — None if null_d == 0
    null_signs: Optional[np.ndarray]  # length null_d

    def eval(self, data: pl.DataFrame) -> np.ndarray:
        Xb = self.base_raw.eval(data)
        Xr = Xb @ self.P
        if self.null_d > 0:
            # Re-rotate the null block: same as _canonicalize_fs_null_basis
            # but using the *fixed* rotation from training (not recomputed
            # from new data).
            Xn = Xr[:, self.rank:] @ self.null_rot
            Xn *= self.null_signs[None, :]
            Xr = np.concatenate([Xr[:, :self.rank], Xn], axis=1)
        n = Xr.shape[0]
        nf = len(self.flev)
        p = self.p
        out = np.zeros((n, p * nf))
        fac_arr = data[self.fterm].to_numpy()
        for j, lev in enumerate(self.flev):
            mask = (fac_arr == lev).astype(float)
            out[:, j * p : (j + 1) * p] = Xr * mask[:, None]
        return out


@dataclass(slots=True)
class _LinearTransformRawBasis(_RawBasis):
    """Wrap a raw basis with a fixed post-multiplication: ``inner.eval(d) @ M``.

    Used by ti's centered margins (M = sum-to-zero Z) and te/ti's np=TRUE
    SVD reparameterization (M = XP).
    """
    inner: _RawBasis
    M: np.ndarray

    def eval(self, data: pl.DataFrame) -> np.ndarray:
        return self.inner.eval(data) @ self.M


@dataclass(slots=True)
class _TensorRawBasis(_RawBasis):
    """Row-wise Kronecker of margin raw bases — mgcv's
    ``tensor.prod.model.matrix`` for ``Predict.matrix.tensor.smooth``.

    Each margin produces an (n, p_i) matrix; the tensor row is their
    Khatri-Rao product (margin 0 outermost). Predict re-runs each margin
    on the new data and tensor-multiplies, identical to
    `_tensor_prod_X([m.eval(data) for m in margins])`.
    """
    margins: list[_RawBasis]

    def eval(self, data: pl.DataFrame) -> np.ndarray:
        Xm = [m.eval(data) for m in self.margins]
        return _tensor_prod_X(Xm)


@dataclass(slots=True)
class _T2RawBasis(_RawBasis):
    """Wood's t2 tensor — like ``_TensorRawBasis`` but each margin's columns
    are pre-rotated by the stored ``nat.param`` ``P_i`` (so the margin design
    splits into [range | null]), and the final output is the
    ``t2.model.matrix`` block-Kronecker structure with optional
    null-space-block sum-to-zero ``Zn``.

    Stored state mirrors ``_build_t2_smooth``:
      * ``margins``     — raw margin bases (no nat.param applied)
      * ``P_per_margin`` — per-margin nat.param transform; ``Xi_np = Xi_raw @ P``
      * ``ranks``       — per-margin range size (range first, null after)
      * ``null_dim``    — overall null dimension (controls drop-last-col / Zn)
      * ``Zn``          — partial absorb.cons rotation for the null block when
        ``null_dim >= 2`` (else None — null_dim 0 needs no constraint, null_dim 1
        drops the last column).
    """
    margins: list[_RawBasis]
    P_per_margin: list[np.ndarray]
    ranks: list[int]
    null_dim: int
    Zn: Optional[np.ndarray]

    def eval(self, data: pl.DataFrame) -> np.ndarray:
        Xm_np = [
            m.eval(data) @ P
            for m, P in zip(self.margins, self.P_per_margin)
        ]
        X, sub_cols = _t2_model_matrix(Xm_np, self.ranks)
        nup = sum(sub_cols)
        if self.null_dim == 0:
            return X
        if self.null_dim == 1:
            keep = np.ones(X.shape[1], dtype=bool)
            keep[-1] = False
            return X[:, keep]
        X_R = X[:, :nup]
        X_N = X[:, nup:]
        return np.concatenate([X_R, X_N @ self.Zn], axis=1)


@dataclass(slots=True)
class _T2PredictRawBasis(_RawBasis):
    """`Predict.matrix.t2.smooth` — full absorb.cons via ``sm$qrc``.

    Where ``_T2RawBasis`` mirrors ``smoothCon``'s partial-absorb output
    (``sm$X`` keeps a constant component in the range cols), this mirrors
    ``PredictMat``: re-evaluate the raw t2 design at new data and apply
    ``Z_p`` from ``qr.qy(qrc, [0; I_q])`` — i.e. drop the 1-d constant
    direction from the *full* tensor column span. ``Z_p`` is computed at
    fit time from ``colSums(X_raw_full)`` so it is independent of new data.
    """
    margins: list[_RawBasis]
    P_per_margin: list[np.ndarray]
    ranks: list[int]
    Z_p: np.ndarray

    def eval(self, data: pl.DataFrame) -> np.ndarray:
        Xm_np = [
            m.eval(data) @ P
            for m, P in zip(self.margins, self.P_per_margin)
        ]
        X, _ = _t2_model_matrix(Xm_np, self.ranks)
        return X @ self.Z_p


@dataclass(slots=True)
class _SZRawBasis(_RawBasis):
    """`smooth.construct.sz.smooth.spec` — zero-center nested smooth.

    Predict replays:
      1. base smooth (typically tp) at new non-factor terms
      2. block-wise expansion across all combinations of factor levels
      3. Kronecker sum-to-zero contrast (XZKr) — drops last-level block per
         factor and subtracts it from each non-last block.
    Stored state matches `_build_sz_smooth`: factor terms + their levels in
    schema order, the base raw evaluator, and the per-factor sizes for XZKr.
    """
    term_full: list[str]
    ftermlist: list[str]
    flev: list[list]
    nf: list[int]
    base_raw: _RawBasis
    p0: int

    def eval(self, data: pl.DataFrame) -> np.ndarray:
        Xb = self.base_raw.eval(data)
        n = Xb.shape[0]
        total_levels = int(np.prod(self.nf))
        p_full = self.p0 * total_levels
        fac_arrs = [data[ft].to_numpy() for ft in self.ftermlist]

        def _iter_combos():
            if not self.nf:
                yield ()
                return
            idx = [0] * len(self.nf)
            while True:
                yield tuple(idx)
                for d in range(len(self.nf) - 1, -1, -1):
                    idx[d] += 1
                    if idx[d] < self.nf[d]:
                        break
                    idx[d] = 0
                else:
                    return

        X = np.zeros((n, p_full))
        for blk_pos, combo in enumerate(_iter_combos()):
            mask = np.ones(n, dtype=float)
            for j, a in enumerate(combo):
                mask *= (fac_arrs[j] == self.flev[j][a]).astype(float)
            X[:, blk_pos * self.p0 : (blk_pos + 1) * self.p0] = Xb * mask[:, None]
        return _xz_kr_contrast(X, self.nf, self.p0)


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
    """Pluck variable names (or deparsed expressions) from an s(...)'s
    positional args.

    mgcv treats the non-keyword args of s() as the term variables. e.g.
    ``s(Machine, Worker, bs="re")`` → ``["Machine", "Worker"]``. Expression
    args (``s(I(b.depth^.5))``, ``s(log(x))``, ``s(x*2)``) are deparsed to
    their formula text — the same string mgcv prints in summaries — and
    are materialised into a synthesised column of that name by
    :func:`_smooth_arg_expr_map` / :func:`_apply_smooth_arg_exprs` before
    each smooth basis is built. Predict-time replay re-evaluates the AST
    against the new data using the same machinery.
    """
    names: list[str] = []
    for a in call.args:
        if isinstance(a, Name):
            names.append(a.ident)
        else:
            names.append(_deparse(a))
    return names


def _collect_name_idents(node, out: set[str]) -> None:
    """Walk an AST collecting every ``Name.ident`` reference. Used to map
    a smooth-arg expression like ``I(b.depth^.5)`` back to the underlying
    columns (``{"b.depth"}``) so NA-drop covers them and predict-time
    re-evaluation can request them."""
    if node is None:
        return
    if isinstance(node, Name):
        if node.ident not in _R_CONSTANTS:
            out.add(node.ident)
        return
    if isinstance(node, Literal):
        return
    if isinstance(node, Paren):
        _collect_name_idents(node.expr, out)
        return
    if isinstance(node, UnaryOp):
        _collect_name_idents(node.operand, out)
        return
    if isinstance(node, BinOp):
        _collect_name_idents(node.left, out)
        _collect_name_idents(node.right, out)
        return
    if isinstance(node, Call):
        for a in node.args:
            _collect_name_idents(a, out)
        for v in node.kwargs.values():
            _collect_name_idents(v, out)
        return
    # Subscript, Dot, Empty: no Name references we care about for this path.


def _smooth_arg_expr_map(expanded: "ExpandedFormula") -> dict[str, "Node"]:
    """Walk every smooth Call (``s``/``te``/``ti``/``t2``) in ``expanded``
    and collect a ``{deparsed_text: ast_node}`` map for every non-``Name``
    positional argument. Same deparse text appears in
    :func:`_smooth_term_vars`, so the resulting map's keys line up with the
    column names the smooth builders ask for at fit/predict time.
    """
    out: dict[str, "Node"] = {}
    for c in expanded.smooths:
        for a in c.args:
            if isinstance(a, Name):
                continue
            key = _deparse(a)
            # First sighting wins — multiple smooths writing identical
            # expressions deparse identically and reuse the same column.
            out.setdefault(key, a)
    return out


def _apply_smooth_arg_exprs(
    data: pl.DataFrame, expr_map: dict[str, "Node"],
) -> pl.DataFrame:
    """Materialise smooth-arg expressions into columns of ``data``.

    For each ``(synth_name, ast_node)`` in ``expr_map``, evaluate the AST
    via :func:`_eval_numeric` and append a column under ``synth_name``.
    Idempotent: if ``synth_name`` already exists in ``data``, it's left
    alone. Used both at fit time (in :func:`materialize_smooths`) and at
    predict time (from :meth:`hea.gam.gam.predict`)."""
    if not expr_map:
        return data
    additions: dict[str, np.ndarray] = {}
    for synth_name, node in expr_map.items():
        if synth_name in data.columns:
            continue
        try:
            v = _eval_numeric(node, data)
        except Exception as e:
            raise ValueError(
                f"smooth-arg expression {synth_name!r} failed to evaluate "
                f"against data: {e}"
            ) from e
        additions[synth_name] = np.asarray(v, dtype=float)
    if additions:
        data = data.with_columns([
            pl.Series(name, vals) for name, vals in additions.items()
        ])
    return data


def _smooth_label(call: Call) -> str:
    """Reproduce mgcv's smooth label (e.g. `s(x)`, `s(Machine,Worker)`)."""
    return f"{call.fn}({','.join(_smooth_term_vars(call))})"


def _smooth_by_expr(call: Call) -> str | None:
    """Return the `by=` expression as a string, or None if unset/NA."""
    by = call.kwargs.get("by")
    if by is None:
        return None
    if isinstance(by, Name) and by.ident == "NA":
        return None
    return _deparse(by)


def _eval_by_col(by_expr: str, data: pl.DataFrame) -> pl.Series | np.ndarray:
    """Evaluate a smooth's ``by=`` expression against ``data``.

    We support the four forms that R/mgcv users actually write — anything
    more exotic would need full Python-expression evaluation machinery that
    polars does not ship. Supported:
      * plain column name → the column (as a ``pl.Series``)
      * ``as.numeric(<name>)`` → float ndarray
      * ``<name> == <lit>`` / ``<name> != <lit>`` → bool ndarray
      * ``as.numeric(<name> == <lit>)`` → float ndarray (0/1 indicator)
    """
    expr = by_expr.strip()
    if expr in data.columns:
        return data[expr]

    import re as _re

    m = _re.fullmatch(r"as\.numeric\((.*)\)", expr)
    if m:
        inner = _eval_by_col(m.group(1).strip(), data)
        if isinstance(inner, pl.Series):
            inner = inner.to_numpy()
        return np.asarray(inner).astype(float)

    # Binary comparisons: <col> (==|!=) <literal>
    m = _re.fullmatch(
        r'(\w+(?:\.\w+)*)\s*(==|!=)\s*(?:"([^"]*)"|\'([^\']*)\'|([-+]?\d+(?:\.\d+)?))',
        expr,
    )
    if m:
        col_name, op, s_dq, s_sq, num = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        if col_name not in data.columns:
            raise KeyError(f"by=: column {col_name!r} not in data")
        col = data[col_name]
        if num is not None:
            # R's `factor == 1` coerces the factor levels to match the integer
            # literal (numeric levels round-trip as strings on our polars side,
            # so compare as strings; Polars categoricals carry string levels).
            if col.dtype in (pl.Categorical, pl.Enum, pl.String, pl.Utf8):
                n_num = float(num)
                n_int = int(n_num)
                lit = str(n_int) if n_num == n_int else str(n_num)
            else:
                lit = float(num)
        else:
            lit = s_dq if s_dq is not None else s_sq
        arr = col.to_numpy()
        mask = (arr == lit) if op == "==" else (arr != lit)
        return mask.astype(bool)

    raise ValueError(
        f"by={by_expr!r} is not supported; expected a column name, "
        "`col == lit`, `col != lit`, or `as.numeric(...)` wrapping one of those"
    )


def _is_factor_like(col: pl.Series | np.ndarray) -> bool:
    """Treat string / object / categorical columns as factor-ish."""
    if isinstance(col, pl.Series):
        return col.dtype in (pl.Categorical, pl.Enum, pl.String, pl.Utf8, pl.Object)
    return col.dtype.kind in ("O", "U", "S")


def _is_ordered_by(col: pl.Series) -> bool:
    """Whether a smooth's `by=` column is an R-style ordered factor.

    mgcv's `smooth.construct` drops the first level for ordered `by` factors
    (so the baseline level is absorbed into the main effect). Detection here
    matches `_factor_from_series`'s ordered signal: look up the column name
    in the `_ORDERED_COLS_CV` context declared by the caller.
    """
    name = getattr(col, "name", None)
    return bool(name) and name in _ORDERED_COLS_CV.get()


def _factor_levels(col: pl.Series) -> list:
    """R-style factor levels: sorted unique values (alphabetic for strings,
    numeric for numerics)."""
    if col.dtype in (pl.Categorical, pl.Enum):
        # See `_factor_from_series`: polars 1.40+ merges Categorical category
        # pools across sibling columns in a DataFrame, so cat.get_categories()
        # may include levels from other columns. Restrict to levels actually
        # present in this column, keeping their schema order.
        full = col.cat.get_categories().to_list()
        codes = col.drop_nulls().to_physical().to_numpy().astype(np.int64)
        present = np.unique(codes) if codes.size else np.empty(0, dtype=np.int64)
        return [full[int(c)] for c in present]
    uniq = col.drop_nulls().unique().to_list()
    if col.dtype.is_numeric():
        return sorted(uniq)
    return sorted(uniq, key=_factor_sort_key)


def _apply_by_and_absorb(
    call: Call,
    data: pl.DataFrame,
    X: np.ndarray,
    S_list: list[np.ndarray],
    cls: str,
    term: list[str],
    raw_basis: _RawBasis | None = None,
) -> list[SmoothBlock]:
    """Apply mgcv's smoothCon post-processing:
    scale.penalty → by-handling → absorb.cons → SmoothBlock(s).

    For numeric `by` with variance: multiply X by by; skip absorb.cons.
    For factor `by`: produce one block per level with (by==lev)*X; each
    gets absorb.cons applied.
    For no `by`: one block with absorb.cons applied.

    ``raw_basis`` (when provided) gets attached to each block as the predict
    half of mgcv's ``Predict.matrix.<class>``. Per-bs constructors thread their
    `_RawBasis` subclass here so `block.spec.predict_mat(new_data)` reproduces
    the same scale.penalty-free design rows the fit used (the by-mask and
    absorb-rotation steps are layered on automatically).
    """
    S_list = [(S + S.T) / 2.0 for S in S_list]
    S_list = _scale_penalty(X, S_list)

    by_expr = _smooth_by_expr(call)
    base_label = _smooth_label(call)
    if by_expr is None:
        X2, S2, T = _absorb_sumzero(X, S_list)
        spec = (
            BasisSpec(raw=raw_basis, by=None, absorb=T)
            if raw_basis is not None else None
        )
        return [SmoothBlock(label=base_label, term=term, cls=cls,
                            X=X2, S=S2, spec=spec)]

    by_col = _eval_by_col(by_expr, data)
    if _is_factor_like(by_col):
        # by_col is a pl.Series (the _is_factor_like(np.ndarray) branch only
        # fires for object/unicode arrays, which don't appear on the current
        # _eval_by_col paths).
        levels = _factor_levels(by_col)
        if _is_ordered_by(by_col) and len(levels) > 1:
            levels = levels[1:]
        by_arr = by_col.to_numpy()
        # mgcv sets `sm$C = colSums(sm$X)` once on the full (pre-by) X, then
        # applies the same Householder Q to every per-level X_lev. Without
        # this each level's absorb uses colSums(X_lev) → a different Z, and
        # the resulting subspaces diverge from mgcv's.
        blocks: list[SmoothBlock] = []
        for lev in levels:
            mask = (by_arr == lev).astype(float)
            X_lev = X * mask[:, None]
            X2, S2, T = _absorb_sumzero(X_lev, S_list, C_source=X)
            label = f"{base_label}:{by_expr}{lev}"
            spec = (
                BasisSpec(
                    raw=raw_basis,
                    by=_ByMask(expr=by_expr, kind="factor", level=lev),
                    absorb=T,
                )
                if raw_basis is not None else None
            )
            blocks.append(SmoothBlock(label=label, term=term, cls=cls,
                                      X=X2, S=S2, spec=spec))
        return blocks

    # Numeric by: multiply X by by-column, skip absorb.cons.
    if isinstance(by_col, pl.Series):
        by_arr = by_col.to_numpy().astype(float)
    else:
        by_arr = np.asarray(by_col, dtype=float)
    X2 = X * by_arr[:, None]
    spec = (
        BasisSpec(
            raw=raw_basis,
            by=_ByMask(expr=by_expr, kind="numeric"),
            absorb=None,
        )
        if raw_basis is not None else None
    )
    return [SmoothBlock(
        label=f"{base_label}:{by_expr}",
        term=term, cls=cls, X=X2, S=S_list, spec=spec,
    )]


def _scale_penalty(X: np.ndarray, S_list: list[np.ndarray]) -> list[np.ndarray]:
    """Match mgcv's `scale.penalty=TRUE` rescaling.

    mgcv applies this on the raw (pre-absorb.cons) X and S:
        maXX = norm(X, "I")^2      # max abs row sum, squared
        maS  = norm(S, "O") / maXX # default R norm() = one-norm
        S   := S / maS = S * maXX / norm(S, "O")
    """
    if X.size == 0 or not S_list:
        return [np.asarray(s, dtype=float) for s in S_list]
    maXX = float(np.abs(X).sum(axis=1).max()) ** 2
    out = []
    for S in S_list:
        S = np.asarray(S, dtype=float)
        if S.size == 0:
            out.append(S)
            continue
        normS = float(np.abs(S).sum(axis=0).max())
        if normS == 0 or maXX == 0:
            out.append(S)
        else:
            out.append(S * (maXX / normS))
    return out


def _build_re_smooth(call: Call, data: pl.DataFrame) -> list[SmoothBlock]:
    """`bs="re"` → `model.matrix(~ term1:term2:...-1, data)`.

    Penalty is `diag(ncol(X))`. re sets `no.rescale=TRUE` in mgcv, so
    scale.penalty is skipped — but then smoothCon's default sum-to-zero
    constraint is also typically disabled for re... actually, empirically
    the fixtures show S has been rescaled. mgcv's re.smooth.spec produces
    X = model.matrix(...), S = [I], and scale.penalty runs normally.
    """
    term_vars = _smooth_term_vars(call)
    # Build the formula `~ term1:term2:...-1` using our existing machinery.
    rhs = None
    for v in term_vars:
        node = Name(v)
        rhs = node if rhs is None else BinOp(":", rhs, node)
    fake = Formula(lhs=None, rhs=rhs)
    ef = expand(fake)
    ef.intercept = False
    X_df = materialize(ef, data)
    X = X_df.to_numpy().astype(float)
    S_list = [np.eye(X.shape[1])]
    S_list = _scale_penalty(X, S_list)
    # re.smooth.spec sets C = empty (no absorb.cons). `by` is still honored:
    # factor by → one block per level; numeric by → multiply X by by.
    base_label = _smooth_label(call)
    by_expr = _smooth_by_expr(call)

    # Derive the (term-value tuple) per X column so predict reconstructs the
    # same column set on new data — first row where X[:,j]==1 names the combo.
    fac_arrs = [data[t].to_numpy() for t in term_vars]
    combos: list[tuple] = []
    for j in range(X.shape[1]):
        nz = np.where(X[:, j] > 0)[0]
        if len(nz) == 0:
            # No row matches this column — happens only for pathological
            # designs; record an unmatchable sentinel so predict yields zeros.
            combos.append((object(),) * len(term_vars))
        else:
            r = int(nz[0])
            combos.append(tuple(arr[r] for arr in fac_arrs))
    raw = _RERawBasis(term=list(term_vars), combos=combos)
    if by_expr is None:
        return [SmoothBlock(label=base_label, term=term_vars,
                            cls="re.smooth.spec", X=X, S=S_list,
                            spec=BasisSpec(raw=raw, by=None, absorb=None))]
    by_col = _eval_by_col(by_expr, data)
    if _is_factor_like(by_col):
        levels = _factor_levels(by_col)
        if _is_ordered_by(by_col) and len(levels) > 1:
            levels = levels[1:]
        by_arr = by_col.to_numpy()
        return [
            SmoothBlock(
                label=f"{base_label}:{by_expr}{lev}", term=term_vars,
                cls="re.smooth.spec",
                X=X * (by_arr == lev).astype(float)[:, None],
                S=S_list,
                spec=BasisSpec(
                    raw=raw,
                    by=_ByMask(expr=by_expr, kind="factor", level=lev),
                    absorb=None,
                ),
            )
            for lev in levels
        ]
    if isinstance(by_col, pl.Series):
        by_arr = by_col.to_numpy().astype(float)
    else:
        by_arr = np.asarray(by_col, dtype=float)
    return [SmoothBlock(
        label=f"{base_label}:{by_expr}", term=term_vars,
        cls="re.smooth.spec",
        X=X * by_arr[:, None], S=S_list,
        spec=BasisSpec(
            raw=raw, by=_ByMask(expr=by_expr, kind="numeric"), absorb=None,
        ),
    )]


def _absorb_sumzero(
    X: np.ndarray,
    S_list: list[np.ndarray],
    C_source: np.ndarray | None = None,
) -> tuple[np.ndarray, list[np.ndarray], _AbsorbTransform]:
    """Apply mgcv's default `absorb.cons=TRUE` sum-to-zero constraint.

    Mirrors mgcv `smoothCon`: C = colMeans(X) (1×k). If every entry of C is
    nonzero (the usual case), apply a single Householder reflector Q from
    qr(t(C)) and return X Q[:,1:], Z' S Z. If any entry of C is exactly zero
    (happens when a covariate is mean-centered integers → exact-zero column
    sums), mgcv takes a different branch: it QRs only the nonzero subset and
    drops the *last* nonzero column, leaving the exact-zero columns in place.
    Without this branch, floating-point noise on the near-zero columns causes
    a rotation that swaps columns in the output (fixtures mgcv_0066, _0182).

    `C_source`: optional matrix whose column means define the constraint.
    For by=factor, mgcv sets `sm$C = colSums(sm$X)` once from the pre-by X,
    then applies the same Householder Q to every `by.dum * X` block. Callers
    that want that behaviour pass the pre-by X as C_source so each per-level
    absorb uses the shared constraint instead of each block's own colSums.

    Returns ``(X_new, S_new, transform)`` where ``transform.apply(X_raw_new)``
    replays the same rotation on new-data rows — the predict-time half of
    mgcv's ``Predict.matrix`` dispatch.
    """
    n, k = X.shape
    if k == 0:
        return X, list(S_list), _AbsorbTransform()
    C = (C_source if C_source is not None else X).mean(axis=0)
    indi = np.flatnonzero(C != 0)  # exact inequality, matching mgcv
    nx = len(indi)
    if nx == k:
        # Normal path: single Householder reflector on full C.
        Q, _ = np.linalg.qr(C.reshape(k, 1), mode="complete")
        Z = Q[:, 1:]
        X_new = X @ Z
        S_new = [Z.T @ S @ Z for S in S_list]
        return X_new, S_new, _AbsorbTransform(full_Z=Z)

    # Sparse-like path: some cols of C are exactly 0; QR only the nonzero
    # subset, place the (nx-1) null-space cols back at indi[:nx-1], and drop
    # the col at indi[-1]. Cols not in indi stay put, unrotated.
    nc = 1  # single constraint (one row)
    nz = nx - nc
    Q_sub, _ = np.linalg.qr(C[indi].reshape(nx, 1), mode="complete")  # nx × nx
    Z_sub = Q_sub[:, 1:]  # nx × (nx-1)

    X_new = X.copy()
    if nz > 0:
        X_new[:, indi[:nz]] = X[:, indi] @ Z_sub
    drop_idx = indi[-1]
    keep_mask = np.ones(k, dtype=bool)
    keep_mask[drop_idx] = False
    X_new = X_new[:, keep_mask]

    S_new = []
    for S in S_list:
        ZSZ = S.copy()
        if nz > 0:
            ZSZ[indi[:nz], :] = Z_sub.T @ S[indi, :]
        ZSZ = ZSZ[keep_mask, :]
        if nz > 0:
            ZSZ[:, indi[:nz]] = ZSZ[:, indi] @ Z_sub
        ZSZ = ZSZ[:, keep_mask]
        S_new.append(ZSZ)
    return X_new, S_new, _AbsorbTransform(
        indi=indi, Z_sub=Z_sub, keep_mask=keep_mask,
    )


def _cr_F_matrix(knots: np.ndarray) -> np.ndarray:
    """Natural-cubic-spline 'F' matrix: y'' at knots = F @ y.

    F[0,:] = F[-1,:] = 0 (natural boundary). Interior rows solve the
    standard tridiagonal y'' system.
    """
    nk = len(knots)
    h = np.diff(knots)
    B = np.zeros((nk - 2, nk - 2))
    D = np.zeros((nk - 2, nk))
    for i in range(nk - 2):
        B[i, i] = (h[i] + h[i + 1]) / 3
        if i > 0:
            B[i, i - 1] = h[i] / 6
        if i < nk - 3:
            B[i, i + 1] = h[i + 1] / 6
        D[i, i] = 1.0 / h[i]
        D[i, i + 1] = -1.0 / h[i] - 1.0 / h[i + 1]
        D[i, i + 2] = 1.0 / h[i + 1]
    F = np.zeros((nk, nk))
    F[1:nk - 1, :] = np.linalg.solve(B, D)
    return F


def _cr_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Evaluate the natural cubic regression spline basis at points `x`.

    Returns (n, nk) matrix. Based on Wood (2017) §5.3.2. Each column acts
    like a Lagrange indicator (column j is 1 at knot j, 0 at other knots).
    """
    nk = len(knots)
    h = np.diff(knots)
    F = _cr_F_matrix(knots)
    # Bracket each x: j s.t. knots[j] <= x < knots[j+1]; clamp at both ends.
    j = np.searchsorted(knots, x, side="right") - 1
    j = np.clip(j, 0, nk - 2)
    hj = h[j]
    a_r = (x - knots[j]) / hj
    a_l = 1.0 - a_r
    c_l = (a_l ** 3 - a_l) * hj ** 2 / 6.0
    c_r = (a_r ** 3 - a_r) * hj ** 2 / 6.0
    nx = len(x)
    X = np.zeros((nx, nk))
    idx_n = np.arange(nx)
    X[idx_n, j] += a_l
    X[idx_n, j + 1] += a_r
    # Add contributions via F for the second-derivative part.
    X += c_l[:, None] * F[j, :] + c_r[:, None] * F[j + 1, :]
    return X


def _cr_penalty(knots: np.ndarray) -> np.ndarray:
    """Natural-cubic-spline integrated-squared-second-derivative penalty.

    S = F' T F where T is banded: T[j,j] picks up h[j-1]/3 + h[j]/3 at
    interior, T[j,j+1] = h[j]/6.
    """
    nk = len(knots)
    h = np.diff(knots)
    F = _cr_F_matrix(knots)
    T = np.zeros((nk, nk))
    for j in range(nk - 1):
        T[j, j] += h[j] / 3.0
        T[j + 1, j + 1] += h[j] / 3.0
        T[j, j + 1] += h[j] / 6.0
        T[j + 1, j] += h[j] / 6.0
    return F.T @ T @ F


def _cr_default_k(call: Call) -> int:
    """Pick basis dim k for a cr smooth: kwarg `k=` if given, else 10."""
    k = call.kwargs.get("k")
    if isinstance(k, Literal) and k.kind == "num":
        return int(k.value)
    return 10


def _cr_is_fixed(call: Call) -> bool:
    """Honor `fx = TRUE` ⇒ fixed / unpenalized."""
    fx = call.kwargs.get("fx")
    if isinstance(fx, Name) and fx.ident in ("TRUE", "T"):
        return True
    if isinstance(fx, Literal):
        if fx.kind == "bool" and fx.value:
            return True
        if fx.kind == "num" and fx.value:
            return True
    return False


def _cr_raw(
    call: Call, data: pl.DataFrame, term: list[str] | None = None,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """Bare cr output (X, S_list, knots) — no scale.penalty, no absorb.cons.

    Used by te/ti/t2 which build their own reparameterization/absorb on top
    of the marginal bases.
    """
    if term is None:
        term = _smooth_term_vars(call)
    if len(term) != 1:
        raise ValueError("cr smooth must be 1D")
    x = data[term[0]].to_numpy().astype(float)
    nk = _cr_default_k(call)
    xu = np.unique(x[~np.isnan(x)])
    if len(xu) < nk:
        raise ValueError(f"cr smooth: fewer unique x than knots ({len(xu)} < {nk})")
    # R's default knots: quantile(unique(x), seq(0, 1, length=nk)), type 7.
    # numpy.quantile default matches R's type 7.
    knots = np.quantile(xu, np.linspace(0.0, 1.0, nk))
    X = _cr_basis(x, knots)
    S_list = [] if _cr_is_fixed(call) else [_cr_penalty(knots)]
    return X, S_list, knots


def _build_cr_smooth(call: Call, data: pl.DataFrame) -> list[SmoothBlock]:
    """Build cubic regression spline (`bs="cr"`) smooth block.

    Follows mgcv's smooth.construct.cr.smooth.spec: knots at quantiles of
    unique(x), natural cubic basis, 2nd-deriv penalty, then absorb.cons
    (sum-to-zero) and scale.penalty.
    """
    term = _smooth_term_vars(call)
    X, S_list, knots = _cr_raw(call, data, term)
    raw = _CRRawBasis(term=term[0], knots=knots)
    return _apply_by_and_absorb(
        call, data, X, S_list, "cr.smooth.spec", term, raw_basis=raw,
    )


# ---- cc (cyclic cubic regression spline) -----------------------------------
#
# Periodic variant of `cr`: knot 1 and knot nk are identified, so at the seam
# the value and the 2nd derivative match. Ported from mgcv's R code:
# smooth.construct.cc.smooth.spec + Predict.matrix.cyclic.smooth + place.knots.
# Basis has nk-1 columns pre-absorb.cons (cyclic identification removes one).


def _cc_place_knots(x: np.ndarray, nk: int) -> np.ndarray:
    """mgcv's `place.knots` for cc: evenly-spaced knots over range of unique x."""
    xs = np.sort(np.unique(x))
    n = len(xs)
    if nk > n:
        raise ValueError("more knots than unique data values is not allowed")
    if nk < 2:
        raise ValueError("too few knots")
    if nk == 2:
        return np.array([xs[0], xs[-1]], dtype=float)
    delta = (n - 1) / (nk - 1)
    i = np.arange(1, nk - 1)
    lbi = np.floor(delta * i).astype(int) + 1  # 1-based into xs
    frac = delta * i + 1 - lbi
    # R uses xs[lbi] and x.shift[lbi] = xs[lbi+1] (0-based: xs[lbi-1] and xs[lbi]).
    interior = xs[lbi - 1] * (1 - frac) + xs[lbi] * frac
    knots = np.empty(nk, dtype=float)
    knots[0] = xs[0]
    knots[-1] = xs[-1]
    knots[1:-1] = interior
    return knots


def _cc_getBD(knots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cyclic tridiagonal B and D, (nk-1) × (nk-1)."""
    nk = len(knots)
    h = np.diff(knots)  # length nk-1
    n = nk - 1
    B = np.zeros((n, n))
    D = np.zeros((n, n))
    # Row 1 (0-indexed 0): wraps to row n-1 on the left.
    B[0, 0] = (h[n - 1] + h[0]) / 3.0
    B[0, 1] = h[0] / 6.0
    B[0, n - 1] = h[n - 1] / 6.0
    D[0, 0] = -(1.0 / h[0] + 1.0 / h[n - 1])
    D[0, 1] = 1.0 / h[0]
    D[0, n - 1] = 1.0 / h[n - 1]
    for i in range(1, n - 1):
        B[i, i - 1] = h[i - 1] / 6.0
        B[i, i] = (h[i - 1] + h[i]) / 3.0
        B[i, i + 1] = h[i] / 6.0
        D[i, i - 1] = 1.0 / h[i - 1]
        D[i, i] = -(1.0 / h[i - 1] + 1.0 / h[i])
        D[i, i + 1] = 1.0 / h[i]
    # Row n (0-indexed n-1): wraps to row 0 on the right.
    B[n - 1, n - 2] = h[n - 2] / 6.0
    B[n - 1, n - 1] = (h[n - 2] + h[n - 1]) / 3.0
    B[n - 1, 0] = h[n - 1] / 6.0
    D[n - 1, n - 2] = 1.0 / h[n - 2]
    D[n - 1, n - 1] = -(1.0 / h[n - 2] + 1.0 / h[n - 1])
    D[n - 1, 0] = 1.0 / h[n - 1]
    return B, D


def _cc_cwrap(x0: float, x1: float, x: np.ndarray) -> np.ndarray:
    """Fold x into [x0, x1] via periodic wrap (mgcv's `cwrap`)."""
    h = x1 - x0
    out = x.copy()
    over = out > x1
    if np.any(over):
        out[over] = x0 + np.mod(out[over] - x1, h)
    under = out < x0
    if np.any(under):
        out[under] = x1 - np.mod(x0 - out[under], h)
    return out


def _cc_basis(x: np.ndarray, knots: np.ndarray, BD: np.ndarray) -> np.ndarray:
    """Evaluate cyclic cubic basis at x; returns (n_obs, nk-1)."""
    nk = len(knots)
    h = np.diff(knots)
    x = _cc_cwrap(float(knots[0]), float(knots[-1]), x)
    # Find j such that knot[j] is the smallest knot ≥ x (1-based; MIN j is 2).
    # For x == knots[0], the loop leaves j=2 (since knots[2-1]==knots[1] >= x only if...).
    # Use the same semantics as R's loop: start j=x (numeric copy), then overwrite.
    j = np.full(x.shape, nk, dtype=int)  # 1-based
    for i in range(nk, 1, -1):
        mask = x <= knots[i - 1]
        j[mask] = i
    # For x strictly below knots[0] (shouldn't happen after cwrap), j stays nk
    # but then j1 = nk-1 and wrap j=nk→1 handles it.
    j1 = j - 1  # left bracket index, 1-based in [1, nk-1]
    hj = j1 - 1  # 0-based index into h (length nk-1)
    # Wrap j: j == nk → j = 1 (cyclic).
    j_wrap = j.copy()
    j_wrap[j_wrap == nk] = 1
    # 0-based indices
    j1_0 = j1 - 1
    j_0 = j_wrap - 1
    I = np.eye(nk - 1)
    xk_right = knots[j1]  # knots[j1+1] 1-based → knots[j1] 0-based
    xk_left = knots[j1 - 1]  # knots[j1] 1-based → knots[j1-1] 0-based
    h_local = h[hj]
    a_r = (xk_right - x) / h_local  # (knots[j1+1] - x)/h[hj]
    a_l = (x - xk_left) / h_local  # (x - knots[j1])/h[hj]
    c_r = (xk_right - x) ** 3 / (6.0 * h_local) - h_local * (xk_right - x) / 6.0
    c_l = (x - xk_left) ** 3 / (6.0 * h_local) - h_local * (x - xk_left) / 6.0
    X = (
        BD[j1_0, :] * c_r[:, None]
        + BD[j_0, :] * c_l[:, None]
        + I[j1_0, :] * a_r[:, None]
        + I[j_0, :] * a_l[:, None]
    )
    return X


def _cc_default_k(call: Call) -> int:
    k = call.kwargs.get("k")
    if isinstance(k, Literal) and k.kind == "num":
        return int(k.value)
    return 10


def _cc_is_fixed(call: Call) -> bool:
    fx = call.kwargs.get("fx")
    if isinstance(fx, Name) and fx.ident in ("TRUE", "T"):
        return True
    if isinstance(fx, Literal):
        if fx.kind == "bool" and fx.value:
            return True
        if fx.kind == "num" and fx.value:
            return True
    return False


def _build_cc_smooth(call: Call, data: pl.DataFrame) -> list[SmoothBlock]:
    term = _smooth_term_vars(call)
    if len(term) != 1:
        raise ValueError("cc smooth must be 1D")
    x = data[term[0]].to_numpy().astype(float)
    nk = _cc_default_k(call)
    if nk < 4:
        nk = 4
    knots = _cc_place_knots(x, nk)
    B, D = _cc_getBD(knots)
    BD = np.linalg.solve(B, D)
    X = _cc_basis(x, knots, BD)
    if _cc_is_fixed(call):
        S_list: list[np.ndarray] = []
    else:
        S = D.T @ BD
        S_list = [S]
    raw = _CCRawBasis(term=term[0], knots=knots, BD=BD)
    return _apply_by_and_absorb(
        call, data, X, S_list, "cc.smooth.spec", term, raw_basis=raw,
    )


# ---- ps (P-spline, Eilers & Marx) ------------------------------------------
#
# Port of mgcv's smooth.construct.ps.smooth.spec:
#   m = (basis_order, penalty_order), default (2, 2). Basis is B-spline of
#   order m[0]+2 (= degree m[0]+1) on `k` evenly-spaced knots covering the
#   data range, with m[0]+1 extension knots on each side. Penalty is
#   D_{m[1]}^T D_{m[1]} where D is the m[1]-th-order finite-difference matrix.


def _eval_c_vec_ints(node) -> list[int] | None:
    """Parse `c(a, b, ...)` or a bare numeric literal into a list of ints.
    Returns None if the node isn't a recognizable numeric literal/vector.
    """
    if isinstance(node, Literal) and node.kind == "num":
        return [int(node.value)]
    if isinstance(node, Call) and node.fn == "c":
        out: list[int] = []
        for a in node.args:
            if isinstance(a, Literal) and a.kind == "num":
                out.append(int(a.value))
            else:
                return None
        return out
    return None


def _eval_c_vec_floats(node) -> list[float] | None:
    """Like `_eval_c_vec_ints` but preserves float precision.
    Also accepts a unary minus wrapping a numeric literal (e.g. `c(-1, 0.5)`)."""
    def _lit(n):
        if isinstance(n, Literal) and n.kind == "num":
            return float(n.value)
        if isinstance(n, UnaryOp) and n.op == "-":
            inner = _lit(n.operand)
            return None if inner is None else -inner
        return None
    if isinstance(node, Call) and node.fn == "c":
        out: list[float] = []
        for a in node.args:
            v = _lit(a)
            if v is None:
                return None
            out.append(v)
        return out
    v = _lit(node)
    return None if v is None else [v]


def _ps_order_m(call: Call) -> tuple[int, int]:
    """Resolve p.order: m=c(basis_order, penalty_order). Single scalar → both.
    Default (2, 2)."""
    m_src = call.kwargs.get("m")
    vals = _eval_c_vec_ints(m_src) if m_src is not None else None
    if not vals:
        return (2, 2)
    if len(vals) == 1:
        return (vals[0], vals[0])
    return (vals[0], vals[1])


def _ps_default_k(call: Call, m0: int) -> int:
    k = call.kwargs.get("k")
    if isinstance(k, Literal) and k.kind == "num":
        return int(k.value)
    return max(10, m0 + 1)


def _ps_knots(x: np.ndarray, m0: int, k: int) -> np.ndarray:
    """mgcv's evenly-spaced P-spline knot vector.
    nk interior knots + m0+1 extension on each side → nk + 2*m0 + 2 knots total.
    """
    nk = k - m0
    if nk <= 0:
        raise ValueError(f"basis dimension {k} too small for B-spline order {m0}")
    xl = float(np.min(x))
    xu = float(np.max(x))
    xr = xu - xl
    xl -= xr * 0.001
    xu += xr * 0.001
    dx = (xu - xl) / (nk - 1)
    return np.linspace(xl - dx * (m0 + 1), xu + dx * (m0 + 1), nk + 2 * m0 + 2)


def _ps_basis(x: np.ndarray, knots: np.ndarray, m0: int) -> np.ndarray:
    """B-spline basis of degree m0+1 evaluated at x (matches splines::spline.des)."""
    from scipy.interpolate import BSpline
    return BSpline.design_matrix(x, knots, m0 + 1).toarray()


def _ps_penalty(k: int, m1: int) -> np.ndarray:
    """D^T D where D is the m1-th-order finite-difference matrix on R^k."""
    D = np.eye(k)
    if m1 > 0:
        D = np.diff(D, n=m1, axis=0)
    return D.T @ D


def _build_ps_smooth(call: Call, data: pl.DataFrame) -> list[SmoothBlock]:
    term = _smooth_term_vars(call)
    if len(term) != 1:
        raise ValueError('bs="ps" only handles 1D smooths')
    x = data[term[0]].to_numpy().astype(float)
    m = _ps_order_m(call)
    k = _ps_default_k(call, m[0])
    knots = _ps_knots(x, m[0], k)
    X = _ps_basis(x, knots, m[0])
    S = _ps_penalty(k, m[1])
    raw = _PSRawBasis(term=term[0], knots=knots, m0=m[0])
    return _apply_by_and_absorb(
        call, data, X, [S], "pspline.smooth", term, raw_basis=raw,
    )


# ---- cp (cyclic P-spline) --------------------------------------------------
#
# Port of mgcv's smooth.construct.cp.smooth.spec. Same (basis order, penalty
# order) parsing as ps, but:
#   - knots are evenly spaced on [min x, max x] with nk = bs.dim + 1 points,
#     no boundary padding — the left-extension for evaluation is built inside
#     `_cp_basis` (cSplineDes equivalent).
#   - basis has `bs.dim` columns (nk - 1) after the cyclic wrap.
#   - penalty is the mth-order difference of diag(np + m) with the first m
#     columns added back onto the last m (closing the loop).


def _cp_default_k(call: Call, m0: int) -> int:
    k = call.kwargs.get("k")
    if isinstance(k, Literal) and k.kind == "num":
        return int(k.value)
    return max(10, m0)


def _cp_basis(x: np.ndarray, knots: np.ndarray, ord_: int) -> np.ndarray:
    """`cSplineDes(x, knots, ord=ord_)` — cyclic B-spline basis.
    Evaluates each basis function via `BSpline.basis_element` so that we can
    sample points where the left-padded basis is only partially defined;
    points above `xc` get an additive contribution from the wrapped x.
    """
    from scipy.interpolate import BSpline
    nk = len(knots)
    if ord_ < 2:
        raise ValueError("order too low")
    if nk < ord_:
        raise ValueError("too few knots")
    k1 = float(knots[0])
    kn = float(knots[-1])
    # Left-pad by (ord-1) knots that mirror the last (ord-1) interior gaps.
    pad_src = knots[nk - ord_ : nk - 1]
    t = np.concatenate([k1 - (kn - pad_src), knots])
    nb = len(t) - ord_

    def _eval(xv: np.ndarray) -> np.ndarray:
        out = np.zeros((len(xv), nb))
        for i in range(nb):
            loc = t[i : i + ord_ + 1]
            if loc[0] == loc[-1]:
                continue
            be = BSpline.basis_element(loc, extrapolate=False)
            v = be(xv)
            out[:, i] = np.nan_to_num(v, nan=0.0)
        return out

    X1 = _eval(x)
    xc = float(knots[nk - ord_])
    ind = x > xc
    if np.any(ind):
        x_w = x[ind] - kn + k1
        X1[ind] += _eval(x_w)
    return X1


def _cp_penalty(np_cols: int, p_ord: int) -> np.ndarray:
    """Cyclic mth-order difference penalty: diff the identity m times, then
    fold the first m columns onto the last m to close the loop."""
    De = np.eye(np_cols + p_ord)
    if p_ord > 0:
        for _ in range(p_ord):
            De = np.diff(De, axis=0)
        D = De[:, p_ord:].copy()
        D[:, np_cols - p_ord : np_cols] += De[:, :p_ord]
    else:
        D = De
    return D.T @ D


# ---- bs (mgcv's B-spline wrapper, derivative-based penalty) ----------------
#
# Port of mgcv's smooth.construct.bs.smooth.spec. Cubic-by-default B-spline
# basis with an integrated-squared-mth-derivative penalty ∫ (f^(m_j))² dx,
# where f is the spline and m can specify MULTIPLE penalty orders
# (m = c(basis_order, m2_1, m2_2, ...)). Each m2 produces one penalty.
#
# Note on naming: `s(x, bs="bs")` in mgcv is UNRELATED to R's top-level
# `splines::bs()` (the parametric basis used in `y ~ bs(x, df=...)`, handled
# elsewhere by `_bs_basis`). mgcv just happens to use "bs" as its spec key.
# The two paths share nothing — different knots, different purpose, different
# class. Helpers in this section (`_bs_design`, `_bs_penalty`, etc.) are all
# the mgcv-smoother variant.
#
# Penalty assembly (per m2): f^(m2) is a piecewise polynomial of degree
# pord = m[0] - m2 on each interval of the interior knots k0. Evaluate the
# basis derivative at pord+1 evenly-spaced points per interval (sharing
# endpoints across neighbors) and use the Vandermonde/monomial-integral
# trick to get a local quadrature weight matrix W1; sum over intervals,
# scaled by h[i]/2. Then S = D^T W D.


def _bs_order_m(call: Call) -> list[int]:
    """Resolve p.order for bs: returns [basis_order, m2_1, m2_2, ...].
    Defaults per mgcv: scalar m → (m, max(0, m-1)); NA-handling → (3, 2)."""
    m_src = call.kwargs.get("m")
    vals = _eval_c_vec_ints(m_src) if m_src is not None else None
    if not vals:
        return [3, 2]
    if len(vals) == 1:
        return [vals[0], max(0, vals[0] - 1)]
    return list(vals)


def _bs_default_k(call: Call, m0: int) -> int:
    k = call.kwargs.get("k")
    if isinstance(k, Literal) and k.kind == "num":
        return int(k.value)
    return max(10, m0)


def _bs_knots_eval(x: np.ndarray, m0: int, k: int) -> np.ndarray:
    """mgcv's default bs knot vector: nk = k - m0 + 1 interior + m0 extension
    on each side. Evenly spaced."""
    nk = k - m0 + 1
    if nk <= 0:
        raise ValueError(f"basis dimension {k} too small for b-spline order {m0}")
    xl = float(np.min(x))
    xu = float(np.max(x))
    xr = xu - xl
    xl -= xr * 0.001
    xu += xr * 0.001
    dx = (xu - xl) / (nk - 1)
    return np.linspace(xl - dx * m0, xu + dx * m0, nk + 2 * m0)


def _bs_design(x: np.ndarray, knots: np.ndarray, m0: int, deriv: int = 0) -> np.ndarray:
    """B-spline of order m0+1 (degree m0) evaluated at x (or its `deriv`-th
    derivative). Matches `splines::spline.des(knots, x, m0+1, derivs=deriv)`."""
    from scipy.interpolate import BSpline
    if deriv == 0:
        return BSpline.design_matrix(x, knots, m0).toarray()
    nb = len(knots) - (m0 + 1)
    out = np.zeros((len(x), nb))
    for i in range(nb):
        c = np.zeros(nb)
        c[i] = 1.0
        spl = BSpline(knots, c, m0, extrapolate=False)
        out[:, i] = np.nan_to_num(spl(x, nu=deriv), nan=0.0)
    return out


def _bs_penalty_W1(pord: int) -> np.ndarray:
    """Local quadrature weight matrix for ∫_{-1}^{1} p(t)^2 dt, where p is a
    polynomial of degree pord represented by its values at pord+1 evenly-
    spaced points in [-1,1]. Returns a (pord+1)×(pord+1) SPD matrix."""
    pts = np.linspace(-1, 1, pord + 1)
    V = np.stack([pts ** j for j in range(pord + 1)], axis=1)  # V[i,j] = pts[i]^j
    P = np.linalg.inv(V)
    H = np.zeros((pord + 1, pord + 1))
    for i in range(pord + 1):
        for j in range(pord + 1):
            s = i + j
            if s % 2 == 0:
                H[i, j] = 2.0 / (s + 1)
    return P.T @ H @ P


def _bs_penalty(knots: np.ndarray, m0: int, m2: int) -> np.ndarray:
    """Integrated-squared-m2-derivative penalty for the bs basis with order m0+1."""
    nk = len(knots) - 2 * m0
    # Interior knots: k0 = knots[m0 : m0 + nk]
    k0 = knots[m0 : m0 + nk]
    h = np.diff(k0)  # length nk-1
    pord = m0 - m2
    if pord < 0:
        raise ValueError("requested non-existent derivative in B-spline penalty")
    n_basis = len(knots) - (m0 + 1)

    if pord == 0:
        # integrand is a step function; midpoint quadrature
        k1 = 0.5 * (k0[:-1] + k0[1:])
        D = _bs_design(k1, knots, m0, deriv=m2)
        D_scaled = np.sqrt(h)[:, None] * D
        return D_scaled.T @ D_scaled

    # pord > 0: pord+1 evenly spaced points per interval, shared endpoints
    # Build k1 by cumulative step over each interval.
    h1 = np.repeat(h / pord, pord)
    k1 = np.cumsum(np.concatenate([[k0[0]], h1]))
    # Evaluate the m2-th derivative of each basis at k1 → (len(k1), n_basis)
    D = _bs_design(k1, knots, m0, deriv=m2)

    W1 = _bs_penalty_W1(pord)  # (pord+1, pord+1)
    n_quad = len(k1)
    W = np.zeros((n_quad, n_quad))
    # Each interval i contributes (h[i]/2) * W1 to rows/cols [i*pord .. i*pord+pord]
    for i in range(nk - 1):
        a = i * pord
        b = a + pord + 1
        W[a:b, a:b] += (h[i] / 2.0) * W1

    return D.T @ W @ D


def _build_bs_smooth(call: Call, data: pl.DataFrame) -> list[SmoothBlock]:
    term = _smooth_term_vars(call)
    if len(term) != 1:
        raise ValueError('bs="bs" only handles 1D smooths')
    x = data[term[0]].to_numpy().astype(float)
    m = _bs_order_m(call)
    m0 = m[0]
    bs_dim = _bs_default_k(call, m0)
    knots = _bs_knots_eval(x, m0, bs_dim)
    X = _bs_design(x, knots, m0, deriv=0)
    S_list = [_bs_penalty(knots, m0, m2) for m2 in m[1:]]
    raw = _BSRawBasis(term=term[0], knots=knots, m0=m0)
    return _apply_by_and_absorb(
        call, data, X, S_list, "Bspline.smooth", term, raw_basis=raw,
    )


def _build_cp_smooth(call: Call, data: pl.DataFrame) -> list[SmoothBlock]:
    term = _smooth_term_vars(call)
    if len(term) != 1:
        raise ValueError('bs="cp" only handles 1D smooths')
    x = data[term[0]].to_numpy().astype(float)
    m = _ps_order_m(call)
    bs_dim = _cp_default_k(call, m[0])
    nk = bs_dim + 1
    if nk <= m[0]:
        raise ValueError(f"basis dim {bs_dim} too small for b-spline order {m[0]}")
    knots = np.linspace(float(np.min(x)), float(np.max(x)), nk)
    ord_ = m[0] + 2
    X = _cp_basis(x, knots, ord_)
    if m[1] > X.shape[1] - 1:
        raise ValueError("penalty order too high for basis dimension")
    S = _cp_penalty(X.shape[1], m[1])
    raw = _CPRawBasis(term=term[0], knots=knots, ord_=ord_)
    return _apply_by_and_absorb(
        call, data, X, [S], "cpspline.smooth", term, raw_basis=raw,
    )


# ---- gp (Gaussian process / Kammann–Wand) ----------------------------------
#
# Port of mgcv's smooth.construct.gp.smooth.spec. Covariance kernel over
# (centered) covariates, truncated via eigendecomposition of the kernel
# matrix on unique data points. Penalty S has the top-k eigenvalues of the
# kernel on the diagonal (zeros on the null-space block). Design matrix X
# = [E(x, knt) @ UZ | T(x)], where UZ are the kept kernel eigenvectors and
# T is the polynomial null space (constant, or constant+linear when not
# stationary). `m=c(sign*type, rho, k)` controls the kernel family.


def _gp_parse_m(call: Call) -> tuple[bool, int, float, float]:
    """Resolve (stationary, type, rho_init, power_k).
    Defaults: stationary=False, type=3 (Matern ν=1.5), rho=auto, k=1."""
    m_src = call.kwargs.get("m")
    vals = _eval_c_vec_floats(m_src) if m_src is not None else None
    if not vals:
        return (False, 3, -1.0, 1.0)
    stationary = vals[0] < 0
    t = abs(int(round(vals[0])))
    if t == 0:  # shouldn't happen, but guard
        t = 3
    rho = vals[1] if len(vals) > 1 else -1.0
    pk = vals[2] if len(vals) > 2 else 1.0
    return (stationary, t, rho, pk)


def _gp_apply_kernel(E: np.ndarray, type_: int, power_k: float) -> np.ndarray:
    if type_ == 1:  # spherical (compact support on [0,1])
        return (1.0 - 1.5 * E + 0.5 * E ** 3) * (E <= 1.0)
    if type_ == 2:  # power exponential
        return np.exp(-(E ** power_k))
    eE = np.exp(-E)
    if type_ == 3:  # Matern ν=1.5
        return (1.0 + E) * eE
    if type_ == 4:  # Matern ν=2.5
        return eE + (E * eE) * (1.0 + E / 3.0)
    if type_ == 5:  # Matern ν=3.5
        return eE + (E * eE) * (1.0 + 0.4 * E + E ** 2 / 15.0)
    raise ValueError(f"unknown GP kernel type {type_}")


def _gp_E_defn(
    x: np.ndarray, xk: np.ndarray, type_: int, rho_init: float, power_k: float,
    sign_type: int,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """`gpE(x, xk, defn)`: distance-based kernel matrix + resolved defn.
    If rho_init <= 0, rho is set to max of the raw distance matrix."""
    diff = x[:, None, :] - xk[None, :, :]
    E_raw = np.sqrt((diff ** 2).sum(axis=2))
    rho = rho_init if rho_init > 0 else float(E_raw.max())
    E = E_raw / rho
    K = _gp_apply_kernel(E, type_, power_k)
    return K, (sign_type * type_, rho, power_k)


def _gp_E_with_defn(x: np.ndarray, xk: np.ndarray, defn: tuple[float, float, float]) -> np.ndarray:
    """Recompute kernel using a pre-resolved defn (rho already known)."""
    st_t, rho, power_k = defn
    type_ = abs(int(round(st_t)))
    diff = x[:, None, :] - xk[None, :, :]
    E = np.sqrt((diff ** 2).sum(axis=2)) / rho
    return _gp_apply_kernel(E, type_, power_k)


def _gp_T(x_c: np.ndarray, stationary: bool) -> np.ndarray:
    """Polynomial null-space basis: [1] when stationary, else [1, x_1, ..., x_d]."""
    n = x_c.shape[0]
    if stationary:
        return np.ones((n, 1))
    return np.column_stack([np.ones(n), x_c])


def _gp_default_k(call: Call, d: int, null_space_dim: int) -> int:
    k = call.kwargs.get("k")
    if isinstance(k, Literal) and k.kind == "num":
        return int(k.value)
    # mgcv default: d + 1 + [10, 30, 100][d-1]
    table = {1: 10, 2: 30, 3: 100}
    add = table.get(d, 100)
    bs_dim = d + 1 + add
    if bs_dim < d + 2:
        bs_dim = d + 2
    return bs_dim


def _build_gp_smooth(call: Call, data: pl.DataFrame) -> list[SmoothBlock]:
    term = _smooth_term_vars(call)
    d = len(term)
    x_full = np.column_stack([data[v].to_numpy().astype(float) for v in term])
    stationary, type_, rho_init, power_k = _gp_parse_m(call)
    null_space_dim = 1 if stationary else d + 1

    bs_dim = _gp_default_k(call, d, null_space_dim)
    if bs_dim < d + 2:
        bs_dim = d + 2

    # Knots: unique covariate combinations (mgcv's `uniquecombs`).
    xu = np.unique(x_full, axis=0)
    nk = xu.shape[0]
    if nk < bs_dim:
        raise ValueError(
            f"gp: fewer unique covariate combinations ({nk}) than bs.dim ({bs_dim})"
        )

    # Center both data and knots by the DATA column means.
    shift = x_full.mean(axis=0)
    x_c = x_full - shift
    xu_c = xu - shift

    # Kernel matrix on knots — this is also what resolves `rho` if default.
    sign_type = -1 if stationary else 1
    E, defn = _gp_E_defn(xu_c, xu_c, type_, rho_init, power_k, sign_type)

    k_radial = bs_dim - null_space_dim

    if k_radial < nk:
        # Top-k eigendecomposition of E by magnitude. For positive-definite
        # kernels this is just top-k by value; spherical/negative-sign types
        # may produce indefinite E, so sort by |λ|.
        eigs, vecs = np.linalg.eigh(E)
        order = np.argsort(-np.abs(eigs))[:k_radial]
        lam = eigs[order]
        UZ = vecs[:, order]
        D = np.zeros((bs_dim, bs_dim))
        D[:k_radial, :k_radial] = np.diag(lam)
    else:
        UZ = np.eye(nk)
        D = np.zeros((bs_dim, bs_dim))
        D[:nk, :nk] = E

    # Design matrix on original (possibly duplicate) x using resolved defn.
    E_x = _gp_E_with_defn(x_c, xu_c, defn)
    X_radial = E_x @ UZ
    T = _gp_T(x_c, stationary)
    X = np.hstack([X_radial, T])

    raw = _GPRawBasis(
        term=list(term), shift=shift, xu_c=xu_c, defn=defn,
        UZ=UZ, stationary=stationary,
    )
    return _apply_by_and_absorb(
        call, data, X, [D], "gp.smooth.spec", term, raw_basis=raw,
    )


# ---- tp (thin-plate regression spline, Wood 2003) --------------------------
#
# Translated from mgcv/src/tprs.c (tprs_setup + eta_const + fast_eta + QT +
# HQmult). Outline:
#   1. Shift each covariate by its mean (`shift` = mean(x)).
#   2. Collapse duplicate rows: Xu = unique rows of (n × d) input.
#   3. Build E: E_ij = η(||Xu_i - Xu_j||) where η has closed form from mgcv.
#   4. Build T: polynomial basis of null space (1, x, x^2, ..., x^(m-1) in 1D;
#      generalized multivariate in higher d).
#   5. Top-k eigendecomposition of E (by |eigenvalue|), then reorder DESCENDING
#      by signed value — this matches mgcv's Lanczos output order.
#   6. TU = T' U (M × k). QT-factorize TU via Householder reflections.
#   7. Build X1 = U @ diag(v) @ Q, pad polynomial part with T.
#   8. Build UZ: first nu rows = U @ Q for radial cols, last M rows = identity
#      for polynomial cols.
#   9. Build S = Q' @ diag(v) @ Q (zero out polynomial part).
#   10. Rescale each X column so its RMS = 1 (equiv. to col-norm² = n),
#       apply the same scaling to UZ and S.
# Sign convention for U eigenvectors is not unique; we match R's eigen by
# using np.linalg.eigh and then sign-normalizing each column (largest-abs
# element positive). The per-column result is the same up to a sign flip.


def _tp_eta_const(m: int, d: int) -> float:
    """`eta_const` from mgcv/src/tprs.c — the irrelevant constant for TPS basis.

    For d=1, m=2: returns 1/12 (verified empirically vs mgcv).
    """
    pi = math.pi
    Ghalf = math.sqrt(pi)
    d2 = d // 2
    m2 = 2 * m
    if m2 <= d:
        raise ValueError("need 2m > d for thin-plate spline")
    if d % 2 == 0:
        f = 1.0 if (m + 1 + d2) % 2 == 0 else -1.0
        for _ in range(m2 - 1):
            f /= 2.0
        for _ in range(d2):
            f /= pi
        for i in range(2, m):
            f /= i
        for i in range(2, m - d2 + 1):
            f /= i
    else:
        f = Ghalf
        kk = m - (d - 1) // 2
        for i in range(kk):
            f /= -0.5 - i
        for _ in range(m):
            f /= 4.0
        for _ in range(d2):
            f /= pi
        f /= Ghalf
        for i in range(2, m):
            f /= i
    return f


def _tp_fast_eta(m: int, d: int, rsq: float, f0: float) -> float:
    """`fast_eta` from mgcv/src/tprs.c. `rsq` is distance SQUARED."""
    if rsq <= 0.0:
        return 0.0
    d2 = d // 2
    f = f0
    if d % 2 == 0:
        f *= math.log(rsq) * 0.5
        for _ in range(m - d2):
            f *= rsq
    else:
        for _ in range(m - d2 - 1):
            f *= rsq
        f *= math.sqrt(rsq)
    return f


def _tp_fast_eta_vec(m: int, d: int, rsq: np.ndarray, f0: float) -> np.ndarray:
    """Broadcasted `_tp_fast_eta`. Replicates the scalar form's iterated-
    multiplication order so the output matches bit-for-bit on shared ops
    (sign of Ritz vectors inside _tp_rlanczos is sensitive to any drift)."""
    out = np.zeros_like(rsq, dtype=float)
    mask = rsq > 0.0
    if not np.any(mask):
        return out
    r = rsq[mask]
    d2 = d // 2
    if d % 2 == 0:
        f = np.log(r) * 0.5
        f *= f0
        for _ in range(m - d2):
            f *= r
    else:
        f = np.full_like(r, f0)
        for _ in range(m - d2 - 1):
            f *= r
        f *= np.sqrt(r)
    out[mask] = f
    return out


def _tp_null_space_dim(d: int, m: int) -> int:
    """Dim of penalty null space = C(m+d-1, d).

    If 2m ≤ d, mgcv bumps m up until 2m > d+1 (visual smoothness).
    """
    if 2 * m <= d:
        m = 1
        while 2 * m < d + 2:
            m += 1
    M = 1
    for i in range(d):
        M *= d + m - 1 - i
    for i in range(2, d + 1):
        M //= i
    return M


def _tp_default_m(d: int) -> int:
    """Default wiggliness penalty order m for d-dim tp. mgcv uses m s.t.
    2m > d+1 — for d=1 that's m=2, d=2 it's m=2, d=3 it's m=3, etc."""
    m = 1
    while 2 * m < d + 2:
        m += 1
    return m


def _tp_order_m(call: Call, d: int) -> int:
    """Resolve `m=` kwarg for s(): integer → used directly (if 2m>d); else default.
    mgcv's rule: if 2m ≤ d, bump m up to smallest value with 2m > d+1.
    Also accepts `m=c(m, ...)` and takes the first entry.
    """
    m_src = call.kwargs.get("m")
    if isinstance(m_src, Literal) and m_src.kind == "num":
        m = int(m_src.value)
        if 2 * m <= d:
            return _tp_default_m(d)
        return m
    if isinstance(m_src, Call) and m_src.fn == "c" and m_src.args:
        first = m_src.args[0]
        if isinstance(first, Literal) and first.kind == "num":
            m = int(first.value)
            if 2 * m <= d:
                return _tp_default_m(d)
            return m
    return _tp_default_m(d)


def _tp_drop_null(call: Call) -> bool:
    """Detect `m=c(m, 0)` shrinkage — drop.null = M flag."""
    m_src = call.kwargs.get("m")
    if isinstance(m_src, Call) and m_src.fn == "c" and len(m_src.args) >= 2:
        second = m_src.args[1]
        if isinstance(second, Literal) and second.kind == "num" and int(second.value) == 0:
            return True
    return False


def _tp_gen_poly_powers(M: int, m: int, d: int) -> np.ndarray:
    """`gen_tps_poly_powers` — sequence of M polynomial exponent tuples
    (each of length d, sum < m) spanning the null space."""
    out = np.zeros((M, d), dtype=int)
    idx = [0] * d
    for i in range(M):
        out[i, :] = idx
        s = sum(idx)
        if s < m - 1:
            idx[0] += 1
        else:
            s -= idx[0]
            idx[0] = 0
            for j in range(1, d):
                idx[j] += 1
                s += 1
                if s == m:
                    s -= idx[j]
                    idx[j] = 0
                else:
                    break
    return out


def _tp_T(X: np.ndarray, m: int, d: int) -> np.ndarray:
    """`tpsT` — the polynomial null-space basis evaluated row-wise on X (n × d).

    Returns (n, M) where M = null_space_dim(d, m). Col j corresponds to the
    polynomial ∏_k x_k^{pi[j,k]}.
    """
    M = _tp_null_space_dim(d, m)
    pi_pow = _tp_gen_poly_powers(M, m, d)
    n = X.shape[0]
    T = np.ones((n, M), dtype=float)
    for j in range(M):
        for k in range(d):
            pk = pi_pow[j, k]
            if pk > 0:
                T[:, j] *= X[:, k] ** pk
    return T


def _tp_E(Xu: np.ndarray, m: int, d: int) -> np.ndarray:
    """`tpsE` — full η matrix on unique rows Xu (nu × d).

    E_ij = η(||Xu_i - Xu_j||) with mgcv's fast_eta convention (r passed as r²).
    """
    nu = Xu.shape[0]
    eta0 = _tp_eta_const(m, d)
    if nu == 0:
        return np.zeros((0, 0))
    # Pairwise squared distances. `(diff*diff).sum(axis=-1)` matches the scalar
    # `np.dot(diff, diff)` summation order for d ≤ 2 (trivially); for d ≥ 3 the
    # reduction order could differ by a ULP from BLAS ddot, which would rotate
    # Ritz vectors inside near-degenerate eigenspaces — tests catch that.
    diff = Xu[:, None, :] - Xu[None, :, :]
    rsq = (diff * diff).sum(axis=-1)
    # Diagonal is exactly 0 from the subtraction; the vec helper also returns
    # 0 there because its `rsq > 0` mask excludes it, so no fill is needed.
    return _tp_fast_eta_vec(m, d, rsq, eta0)


def _tp_qt_factor(A_in: np.ndarray) -> np.ndarray:
    """`QT(Q, A, fullQ=0)` — produces HH vector storage Z (n × m).

    A is n × m with n ≤ m. Z stores, in each row, the scaled Householder vector
    u_i (I - u_i u_i') such that A Q = [0, T] where Q = H_0 H_1 … H_{n-1}.
    """
    A = A_in.astype(float, copy=True)
    n, m = A.shape
    Z = np.zeros((n, m))
    for i in range(n):
        row = A[i]
        cu = m - i
        mx = float(np.max(np.abs(row[:cu]))) if cu > 0 else 0.0
        if mx > 0.0:
            row[:cu] /= mx
        lsq = float(np.sqrt(np.sum(row[:cu] ** 2)))
        if row[cu - 1] < 0:
            lsq = -lsq
        row[cu - 1] += lsq
        if lsq != 0:
            g = 1.0 / (lsq * row[cu - 1])
        else:
            g = 0.0
        lsq *= mx
        for j in range(i + 1, n):
            x = float(np.dot(row[:cu], A[j, :cu])) * g
            A[j, :cu] -= x * row[:cu]
        g_sqrt = math.sqrt(g) if g > 0 else 0.0
        Z[i, :cu] = row[:cu] * g_sqrt
        Z[i, cu:] = 0.0
        # A[i] becomes [0,…,0, -lsq, untouched trailing]; only used implicitly.
    return Z


def _tp_hqmult_right(C_in: np.ndarray, Z: np.ndarray, transposed: bool = False) -> np.ndarray:
    """`HQmult(C, Z, p=0, t=transposed)`.

    p=0 (post-mult), t=0: C := C @ H_0 @ H_1 @ … @ H_{r-1}  (ascending order).
    p=0, t=1:             C := C @ H_{r-1} @ … @ H_0         (descending).
    Each H_k = I - u_k u_k' with u_k = Z[k, :].
    """
    C = C_in.astype(float, copy=True)
    nhh = Z.shape[0]
    order = range(nhh - 1, -1, -1) if transposed else range(nhh)
    for k in order:
        u = Z[k]
        Cu = C @ u
        C -= np.outer(Cu, u)
    return C


def _tp_hqmult_left_transposed(C_in: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """`HQmult(C, Z, p=1, t=1)` — C := Q' @ C where Q = H_0 … H_{r-1}.

    Since each H_k is symmetric, Q' = H_{r-1}' … H_0' = H_{r-1} … H_0, so
    the loop applies H_0 first from the left, then H_1, etc. (ascending).
    """
    C = C_in.astype(float, copy=True)
    nhh = Z.shape[0]
    for k in range(nhh):
        u = Z[k]
        uC = u @ C
        C -= np.outer(u, uC)
    return C


def _tp_default_k(call: Call, d: int, m: int) -> int:
    """k = bs.dim: user-supplied `k=` kwarg, else mgcv's default (M + 8 for
    d=1, M + 27 for d=2, M + 100 for d=3; only d=1,2,3 relevant here).
    M here is computed from the resolved m (not the default m).
    """
    ksrc = call.kwargs.get("k")
    if isinstance(ksrc, Literal) and ksrc.kind == "num":
        return int(ksrc.value)
    M = _tp_null_space_dim(d, m)
    default = (8, 27, 100)[min(d, 3) - 1]
    return M + default


def _tp_is_fixed(call: Call) -> bool:
    fx = call.kwargs.get("fx")
    if isinstance(fx, Name) and fx.ident in ("TRUE", "T"):
        return True
    if isinstance(fx, Literal):
        if fx.kind == "bool" and fx.value:
            return True
        if fx.kind == "num" and fx.value:
            return True
    return False


def _tp_rlanczos(
    A: np.ndarray, m: int, lm: int, tol: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Port of mgcv/src/mat.c::Rlanczos (symmetric Lanczos w/ full reorth).

    A: n×n symmetric. Returns (D, U) where D has length m+lm and U is n×(m+lm).
    If lm<0 on entry ("biggest" mode), returns the m largest-magnitude
    eigenpairs, with the positive eigenvalues filling the leading slots in
    descending order and the negative ones in the trailing slots (most
    negative last) — matching mgcv tprs.c's `minus = -1` call at tprs.c:408.

    The key reason for porting (vs. using np.linalg.eigh) is matching mgcv's
    basis choice inside degenerate eigenspaces of A. Lanczos with a fixed
    start vector picks one specific orthonormal basis for any such subspace;
    a dense eigendecomp picks a different one. For the tp smooth the Ritz
    vectors feed into U @ diag(v) @ Z and the final X matrix, so matching
    mgcv bit-for-bit requires the same start vector (same LCG seed), same
    iteration order, same reorth strategy, and same tridiag eigendecomp.
    """
    n = A.shape[0]
    if tol is None:
        tol = float(np.finfo(float).eps) ** 0.7

    biggest = False
    if lm < 0:
        biggest = True
        lm = 0

    # How often to do the tridiag eigendecomp / convergence test. Direct
    # port of mgcv's heuristic.
    f_check = (m + lm) // 2
    if f_check < 10:
        f_check = 10
    kk_fc = n // 10
    if kk_fc < 1:
        kk_fc = 1
    if kk_fc < f_check:
        f_check = kk_fc

    # Reorthogonalization uses classical Gram-Schmidt twice (CGS2) via a
    # batched BLAS matmul — mgcv's C code uses sequential MGS2 in a tight
    # per-column loop. CGS2 has a different floating-point trajectory, so
    # eigenvectors of T_j may rotate within degenerate subspaces, but the
    # final X matrix stays within the 1e-5 relative tolerance of the R
    # oracle on the full tp/te/fs fixture suite.
    #
    # a_j / b_j and the start-vector norm still use `np.sum(u*v)` (pairwise,
    # BLAS-free, deterministic). Those feed the tridiagonal eigendecomp, so
    # run-to-run consistency there avoids spurious basis drift beyond what
    # CGS2 already introduces.

    # mgcv's LCG-seeded start vector. The specific constants (106, 1283, 6075)
    # and the `jran=1` seed are load-bearing — changing them would pick a
    # different basis within degenerate subspaces.
    ia, ic, im_mod = 106, 1283, 6075
    jran = 1
    q0 = np.empty(n, dtype=float)
    for i in range(n):
        jran = (jran * ia + ic) % im_mod
        q0[i] = jran / im_mod - 0.5
    q0 /= float(np.sqrt(np.sum(q0 * q0)))

    # Q stores Lanczos vectors as rows of a dense (n+1, n) array. Row-wise
    # layout makes Q[i] a C-contiguous view, so np.multiply(Q[i], z, out=buf)
    # has the same pairwise-reduce semantics as np.sum(q[i] * z) for 1D q.
    Q = np.zeros((n + 1, n))
    Q[0] = q0

    a = np.zeros(n)
    b = np.zeros(n)
    err = np.full(n, 1e300)

    # Scratch buffer for aj / bj reductions. Keeps the np.sum(u*v) pairwise
    # reduction (not BLAS dot) so the tridiagonal T_j entries are deterministic
    # between runs — the wine dataset (n=47, mgcv_0020) has near-degenerate
    # eigenvalues where a 1 ULP shift rotates eigenvectors within a degenerate
    # subspace and breaks fixture comparison. CGS2 above is tolerant of that
    # because reorth is already O(eps) off anyway; a[j] / b[j] directly feed
    # the eigendecomp, so stability matters more there.
    buf = np.empty(n)

    d_sorted: np.ndarray | None = None
    v_eig: np.ndarray | None = None

    j = 0
    while j < n:
        qj = Q[j]
        # z = A q[j]  (full matvec; symmetry is exploited by dsymv in mgcv
        # but the numerical difference vs a dense matmul is within the
        # 1e-5 basis-equivalence tolerance we test against).
        z = A @ qj
        np.multiply(qj, z, out=buf)
        aj = float(buf.sum())
        a[j] = aj
        if j == 0:
            z -= aj * qj
        else:
            z -= aj * qj
            z -= b[j - 1] * Q[j - 1]
            # Reorthogonalize via classical Gram-Schmidt, twice (CGS2).
            # mgcv's C code uses sequential MGS-twice; CGS2 has a different
            # arithmetic trajectory but achieves machine-precision orthogonality
            # in two passes and collapses the inner O(j) dot-product loop into
            # a single BLAS matmul. The resulting basis still lies in the same
            # Krylov subspace; eigenvectors of T_j may rotate within degenerate
            # subspaces, but stay within the 1e-5 relative tolerance of the
            # R-oracle fixtures. Verified empirically on the full tp/te/fs
            # suite (tests/test_smooths.py).
            Qact = Q[: j + 1]  # (j+1, n)
            for _ in range(2):
                # c = Qact @ z  -> (j+1,), then z -= Qact.T @ c
                c = Qact @ z
                z -= c @ Qact
        np.multiply(z, z, out=buf)
        bj = float(np.sqrt(buf.sum()))
        b[j] = bj
        if j < n - 1:
            if bj > 0.0:
                np.divide(z, bj, out=Q[j + 1])
            # else: Q[j+1] already zero-initialized.

        if ((j >= m + lm) and (j % f_check == 0)) or (j == n - 1):
            d_copy = a[: j + 1].copy()
            g_copy = b[:j].copy()
            # Pin the LAPACK driver to stemr (MRRR). scipy 1.17 changed the
            # `lapack_driver='auto'` default from stemr to stevd (D&C), and
            # for small near-zero off-diagonals the two drivers pick different
            # eigenvector signs. That shows up downstream as a ~0.1 rotation in
            # the absorb.cons'd X matrix — passed at scipy 1.15 / fails at 1.17
            # (see mgcv_0020 wine tp). stemr is what matches mgcv's ground-truth
            # basis empirically on every tp/te/tp-by fixture we have.
            w, V = _eigh_tridiagonal(d_copy, g_copy, lapack_driver="stemr")
            # scipy returns ascending; mgcv_trisymeig(descending=1) returns
            # descending. Reverse.
            w = w[::-1].copy()
            V = V[:, ::-1].copy()
            d_sorted = w
            v_eig = V

            normTj = max(abs(w[0]), abs(w[-1]))
            err[: j + 1] = np.abs(bj * V[j, :])

            if j >= m + lm:
                max_err = normTj * tol
                if biggest:
                    pi = 0
                    ni = 0
                    converged = True
                    while pi + ni < m:
                        if abs(w[pi]) >= abs(w[j - ni]):
                            if err[pi] > max_err:
                                converged = False
                                break
                            pi += 1
                        else:
                            # mgcv checks err[ni], not err[j-ni]. Replicating
                            # the exact C code for behavioral parity — the
                            # convergence check is loose but the eigenvectors
                            # produced are still correct once the iteration
                            # stops.
                            if err[ni] > max_err:
                                converged = False
                                break
                            ni += 1
                    if converged:
                        m = pi
                        lm = ni
                        j += 1
                        break
                else:
                    ok = True
                    for i_chk in range(m):
                        if err[i_chk] > max_err:
                            ok = False
                    for i_chk in range(j, j - lm, -1):
                        if err[i_chk] > max_err:
                            ok = False
                    if ok:
                        j += 1
                        break
        j += 1

    assert d_sorted is not None and v_eig is not None

    # Ritz vectors: U[:,k] = sum_{l<j} q[l] * V[l, idx(k)].
    # Q stored row-wise (Q[l] = q[l]); transpose to match the (n, j) @ (j,)
    # matmul shape the original code used with np.column_stack(q[:j]).
    Qj = Q[:j].T
    D_out = np.empty(m + lm)
    U_out = np.zeros((n, m + lm))
    for k_idx in range(m):
        D_out[k_idx] = d_sorted[k_idx]
        U_out[:, k_idx] = Qj @ v_eig[:j, k_idx]
    for k_idx in range(m, m + lm):
        kk_idx = j - (m + lm - k_idx)
        D_out[k_idx] = d_sorted[kk_idx]
        U_out[:, k_idx] = Qj @ v_eig[:j, kk_idx]
    return D_out, U_out


def _tp_raw(
    call: Call, data: pl.DataFrame, term: list[str],
) -> tuple[np.ndarray, list[np.ndarray], int, int, int, dict]:
    """Bare `smooth.construct.tp.smooth.spec` output: no scale.penalty,
    no absorb.cons, no drop_null. Returns `(X_raw, S_list, M, k, rank, state)`
    where `rank = k - M` and ``state`` carries the predict-time replay
    fields (``shift``, ``Xu``, ``m``, ``d``, ``UZ``, ``w``).

    Separated from `_build_tp_smooth` so that `fs` smooths (which
    reparameterize the bare tp output before duplicating across factor
    levels) can reuse the same code with a caller-provided `term` list.
    """
    d = len(term)
    # Build (n × d) matrix of covariates, shifted by column mean.
    x_full = np.column_stack([data[v].to_numpy().astype(float) for v in term])
    shift = x_full.mean(axis=0)
    x_c = x_full - shift
    n = x_c.shape[0]

    m = _tp_order_m(call, d)
    M = _tp_null_space_dim(d, m)
    k = _tp_default_k(call, d, m)
    if k < M + 1:
        k = M + 1

    # Collapse to unique rows (Xu). Preserve an index mapping from data rows
    # to their position in Xu.
    Xu, yxindex = np.unique(x_c, axis=0, return_inverse=True)
    nu = Xu.shape[0]
    if nu < k:
        raise ValueError(f"tp smooth: fewer unique covariate rows than k ({nu} < {k})")

    E = _tp_E(Xu, m, d)
    T_mat = _tp_T(Xu, m, d)

    pure_knot = (nu == k)

    if pure_knot:
        # When nu == k mgcv skips the eigendecomposition entirely (no
        # truncation needed) and builds UZ from QT(T', fullQ=1). X is then
        # computed by evaluating the TPS kernel + polynomial basis at each
        # data point directly, rather than by indexing an X1-on-Xu table.
        # The resulting basis differs from the Lanczos-based one by a rotation
        # within the null-space block, so the two paths are not interchangeable.
        Z_hh = _tp_qt_factor(T_mat.T.copy())
        # Full Q (nu × nu) via applying HH reflectors to identity from the right.
        Q_full = _tp_hqmult_right(np.eye(nu), Z_hh, transposed=False)

        UZ = np.zeros((nu + M, k))
        UZ[:nu, :k - M] = Q_full[:, :k - M]
        for i in range(M):
            UZ[(nu + M) - i - 1, k - i - 1] = 1.0

        eta0 = _tp_eta_const(m, d)
        pi_pow = _tp_gen_poly_powers(M, m, d)
        X_raw = np.zeros((n, k))
        b = np.empty(nu + M)
        for i in range(n):
            x_i = x_c[i]
            diff = Xu - x_i  # (nu, d)
            rsq = np.sum(diff * diff, axis=1)
            for j in range(nu):
                b[j] = _tp_fast_eta(m, d, float(rsq[j]), eta0)
            for j in range(M):
                val = 1.0
                for dim_idx in range(d):
                    pk = pi_pow[j, dim_idx]
                    if pk > 0:
                        val *= x_i[dim_idx] ** pk
                b[nu + j] = val
            X_raw[i, :] = UZ.T @ b

        if _tp_is_fixed(call):
            S_list: list[np.ndarray] = []
        else:
            # S starts from E itself (not diag(v)), embedded in a k×k frame.
            S = np.zeros((k, k))
            S[:nu, :nu] = E
            S = _tp_hqmult_right(S, Z_hh, transposed=False)
            S = _tp_hqmult_left_transposed(S, Z_hh)
            S[-M:, :] = 0.0
            S[:, -M:] = 0.0
            S = (S + S.T) / 2.0
            S_list = [S]
    else:
        # Top-k eigendecomposition of E via mgcv's Rlanczos. Lanczos (with
        # a fixed LCG start vector) picks a specific orthonormal basis for
        # any degenerate eigenspace; np.linalg.eigh picks a different one,
        # which causes several tp fixtures with clustered/repeated eigenvalues
        # to diverge from mgcv's output by a basis rotation.
        v_k, U = _tp_rlanczos(E, k, -1)

        # TU = T' U, QT-factorize, apply to U, T, S.
        TU = T_mat.T @ U
        Z_hh = _tp_qt_factor(TU)

        # X1 on unique rows: first (k-M) cols = U diag(v) applied with Q,
        # last M cols = polynomial T.
        X1 = U * v_k  # col-wise scaling
        X1 = _tp_hqmult_right(X1, Z_hh, transposed=False)
        X1[:, k - M:] = 0.0
        X1[:, k - M:k - M + M] = T_mat
        X_raw = X1[yxindex, :]  # map unique → full data rows

        # UZ: (nu + M) × k. Radial block = U @ Q on rows 0..nu-1. Poly block
        # on last M rows is the identity (diagonal on the last M cols).
        UZ = np.zeros((nu + M, k))
        UZ[:nu, :] = U
        UZ[:nu, :] = _tp_hqmult_right(UZ[:nu, :], Z_hh, transposed=False)
        UZ[:nu, k - M:] = 0.0
        for i in range(M):
            UZ[(nu + M) - i - 1, k - i - 1] = 1.0

        # Penalty S: Q' diag(v) Q, zero-pad last M rows/cols (polynomial
        # part is unpenalized).
        if _tp_is_fixed(call):
            S_list: list[np.ndarray] = []
        else:
            S = np.diag(v_k).astype(float)
            S = _tp_hqmult_right(S, Z_hh, transposed=False)
            S = _tp_hqmult_left_transposed(S, Z_hh)
            S[-M:, :] = 0.0
            S[:, -M:] = 0.0
            S = (S + S.T) / 2.0
            S_list = [S]

    # Rescale each X column so its sum-of-squares = n (= mgcv's "rms=1").
    # Apply same factor to UZ cols and to S rows+cols. After this, we have
    # the pre-absorb.cons smooth; still need scale.penalty + absorb.cons.
    w = np.sqrt(np.sum(X_raw ** 2, axis=0) / n)
    w = np.where(w == 0, 1.0, w)
    X_raw = X_raw / w
    S_list = [S / w[:, None] / w[None, :] for S in S_list]

    state = dict(shift=shift, Xu=Xu, m=m, d=d, UZ=UZ, w=w, M=M, k=k)
    return X_raw, S_list, M, k, k - M, state


def _build_tp_smooth(call: Call, data: pl.DataFrame) -> list[SmoothBlock]:
    """Build thin-plate regression spline (`bs="tp"`, the default) block.

    Exact port of mgcv/src/tprs.c::tprs_setup (case `n_knots < k`, which is
    the default when user passes no `knots=`). Matches mgcv's output for
    `absorb.cons=TRUE, scale.penalty=TRUE` after our standard post-processing.
    """
    term = _smooth_term_vars(call)
    X_raw, S_list, M, k, _rank, state = _tp_raw(call, data, term)

    if _tp_drop_null(call):
        # `m=c(m, 0)`: drop the last M (null-space) columns, center remaining,
        # and skip absorb.cons entirely.
        keep = k - M
        full_raw = _TPRawBasis(
            term=list(term), shift=state["shift"], Xu=state["Xu"],
            m=state["m"], d=state["d"], M=M, k=k,
            UZ=state["UZ"], w=state["w"],
        )
        X_raw = X_raw[:, :keep]
        col_means = X_raw.mean(axis=0)
        X_raw = X_raw - col_means[None, :]
        S_list = [S[:keep, :keep] for S in S_list]
        S_list = [(S + S.T) / 2.0 for S in S_list]
        S_list = _scale_penalty(X_raw, S_list)
        raw = _TPDropNullRawBasis(inner=full_raw, keep=keep, col_means=col_means)
        return [SmoothBlock(
            label=_smooth_label(call), term=term,
            cls="tprs.smooth", X=X_raw, S=S_list,
            spec=BasisSpec(raw=raw, by=None, absorb=None),
        )]

    raw = _TPRawBasis(
        term=list(term), shift=state["shift"], Xu=state["Xu"],
        m=state["m"], d=state["d"], M=M, k=k, UZ=state["UZ"], w=state["w"],
    )
    return _apply_by_and_absorb(
        call, data, X_raw, S_list, "tprs.smooth", term, raw_basis=raw,
    )


# ---- fs: factor-smooth interaction ------------------------------------------
#
# mgcv's `smooth.construct.fs.smooth.spec` builds one base smooth on the
# non-factor terms, reparameterizes it via nat.param, then duplicates that
# basis block-wise across factor levels. Penalties: one block-diagonal range
# penalty plus one single-entry penalty per null-space dimension. Sets
# `side.constrain = FALSE` and `C = matrix(0, 0, ncol(X))` so smoothCon
# skips absorb.cons entirely. scale.penalty still runs on the final block.


def _nat_param(
    X: np.ndarray, S: np.ndarray, rank: int, type_: int = 1, unit_fnorm: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Port of mgcv's `nat.param(X, S, rank, type, unit.fnorm)`.

    type=1: QR on X, then eigendecompose R^-T S R^-1; rescale so the
    penalty is identity on its range.
    type=3: eigendecompose S directly; rescale columns by sqrt(eigenvalue)
    (range) or by a col-norm match (null). Null-space eigenvectors are
    post-rotated so the final column is closest to a constant vector.

    Returns `(X_new, D, P)`: X_new = X @ P; D is the range diagonal of the
    transformed penalty (length `rank`); P is the parameter-transform."""
    p = X.shape[1]

    if type_ == 3:
        S_sym = 0.5 * (S + S.T)
        # Use LAPACK's `syevr` (MRRR) driver — for degenerate null-spaces
        # the eigenvector basis is LAPACK-choice-dependent, and `evr` is
        # what R's default eigen() resolves to for symmetric matrices.
        # numpy's linalg.eigh uses `evd` (D&C), which gives a different
        # rotation of the same null space.
        from scipy.linalg import eigh as _sla_eigh
        w, V = _sla_eigh(S_sym, driver="evr")
        idx = np.argsort(-w)  # descending to match R's eigen()
        w = w[idx]
        V = V[:, idx]
        null_exists = rank < p
        E = np.ones(p)
        if rank > 0:
            E[:rank] = np.sqrt(np.clip(w[:rank], 0.0, None))
        X_rot = X @ V
        col_norm = np.sum(X_rot ** 2, axis=0) / (E ** 2)
        av_norm = float(np.mean(col_norm[:rank])) if rank > 0 else 1.0
        if null_exists:
            E[rank:] = np.sqrt(col_norm[rank:] / av_norm)
        E_safe = np.where(E > 1e-14, E, 1.0)
        X_new = X_rot / E_safe
        P = V / E_safe
        # Re-rotate null space so the constant vector is the final column.
        if null_exists and rank < p - 1:
            ind = np.arange(rank, p)
            rind = np.arange(p - 1, rank - 1, -1)  # reversed
            Xn = X_new[:, ind]
            n_rows = Xn.shape[0]
            Xn_c = Xn - Xn.mean(axis=0, keepdims=True)
            w_n, V_n = np.linalg.eigh(Xn_c.T @ Xn_c)
            o = np.argsort(-w_n)
            V_n = V_n[:, o]
            X_new[:, rind] = X_new[:, ind] @ V_n
            P[:, rind] = P[:, ind] @ V_n
        D = np.zeros(rank)
        if unit_fnorm:
            if rank > 0:
                scale = 1.0 / np.sqrt(np.mean(X_new[:, :rank] ** 2))
                X_new[:, :rank] *= scale
                P[:, :rank] *= scale
                D = np.full(rank, scale ** 2)
            if null_exists:
                scalef = 1.0 / np.sqrt(np.mean(X_new[:, rank:] ** 2))
                X_new[:, rank:] *= scalef
                P[:, rank:] *= scalef
        else:
            if rank > 0:
                D = np.ones(rank)
        return X_new, D, P

    Q, R = np.linalg.qr(X, mode="reduced")
    # RSR = R^-T @ S @ R^-1.
    Y = np.linalg.solve(R.T, S)
    RSR = np.linalg.solve(R.T, Y.T).T
    RSR = 0.5 * (RSR + RSR.T)
    # Match R's eigen() eigenvector basis inside degenerate eigenspaces by
    # calling the same LAPACK driver (MRRR, via evr). numpy's eigh uses evd
    # (D&C), which rotates the null-space differently and leaks into the
    # final X/P columns for fs-style smooths that don't re-rotate the null.
    from scipy.linalg import eigh as _sla_eigh
    w, V = _sla_eigh(RSR, driver="evr")
    order = np.argsort(-w)  # descending
    w = w[order]
    V = V[:, order]

    D = np.clip(w[:rank].copy(), 0.0, None)
    X_new = Q @ V
    P = np.linalg.solve(R, V)

    if type_ == 1:
        E = np.concatenate([np.sqrt(D), np.ones(p - rank)])
        E_safe = np.where(E > 1e-14, E, 1.0)
        X_new = X_new / E_safe
        P = P / E_safe
        D = np.ones(rank)

    if unit_fnorm:
        if rank > 0:
            scale = 1.0 / np.sqrt(np.mean(X_new[:, :rank] ** 2))
            X_new[:, :rank] *= scale
            P[:, :rank] *= scale
            D = D * scale ** 2
        if rank < p:
            scalef = 1.0 / np.sqrt(np.mean(X_new[:, rank:] ** 2))
            X_new[:, rank:] *= scalef
            P[:, rank:] *= scalef

    return X_new, D, P


def _fs_find_factor(term: list[str], data: pl.DataFrame) -> tuple[str | None, list[str]]:
    """Split a term list into (factor_var | None, non_factor_vars).

    Returns `(None, term)` when no term is factor-like — mgcv falls through
    to the base smooth in that case (smooth.r line 2025-2028)."""
    fterm: str | None = None
    others: list[str] = []
    for v in term:
        col = data[v]
        if _is_factor_like(col):
            if fterm is not None:
                raise ValueError("fs smooths can only have one factor argument")
            fterm = v
        else:
            others.append(v)
    return fterm, others


def _canonicalize_fs_null_basis(
    Xr: np.ndarray, rank: int,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Rotate the null-space columns of Xr to a LAPACK-independent basis.

    mgcv's `nat.param(type=1)` leaves the null eigenspace of RSR spanned by
    whatever orthonormal basis its LAPACK returns. For degenerate eigenspaces
    that choice varies between LAPACK builds (R's netlib vs scipy's
    Accelerate), so Xr's last `null_d` columns are implementation-dependent.
    Rotate them by the principal components of the centered null columns —
    a deterministic, data-driven basis computable from any LAPACK's nat.param
    output (the 2×2 / small eigendecomp of `Xn^T Xn` is stable across libs).
    Sign convention: largest-magnitude entry of each column is positive.

    Returns ``(out, V_n, signs)`` where ``V_n`` is the (null_d, null_d)
    rotation and ``signs`` are the per-column ±1 flips. Both are ``None``
    when ``null_d == 0``.
    """
    p = Xr.shape[1]
    null_d = p - rank
    if null_d == 0:
        return Xr, None, None
    Xn = Xr[:, rank:] - Xr[:, rank:].mean(axis=0, keepdims=True)
    _w, V_n = np.linalg.eigh(Xn.T @ Xn)  # ascending; largest eig last
    Xn_rot = Xr[:, rank:] @ V_n
    signs = np.ones(null_d)
    for c in range(null_d):
        m = int(np.argmax(np.abs(Xn_rot[:, c])))
        if Xn_rot[m, c] < 0:
            Xn_rot[:, c] = -Xn_rot[:, c]
            signs[c] = -1.0
    out = Xr.copy()
    out[:, rank:] = Xn_rot
    return out, V_n, signs


def _build_fs_smooth(call: Call, data: pl.DataFrame) -> list[SmoothBlock]:
    """`bs="fs"` — factor-smooth interaction.

    Default base smooth is `tp` (`xt$bs="tp"` in mgcv; alternative bases
    require `xt=list(bs=...)` which is not yet parsed here).

    Fallthrough: mgcv's fs.smooth.spec checks for a factor among the terms;
    if none is found it reclasses the object as the base smooth and
    returns that constructor's output (smooth.r line 2025-2028). We mirror
    that by dispatching to `_build_tp_smooth` on the full term list.
    """
    term = _smooth_term_vars(call)
    fterm, others = _fs_find_factor(term, data)
    if fterm is None:
        return _build_tp_smooth(call, data)

    # Build base tp smooth on the non-factor terms — bare output, no
    # scale.penalty, no absorb.cons.
    Xb, Sb_list, M, k, rank, state = _tp_raw(call, data, others)
    null_d = k - rank
    Sb = Sb_list[0]

    # nat.param(type=1) — make the base penalty an identity on its range.
    Xr, D, P = _nat_param(Xb, Sb, rank=rank, type_=1, unit_fnorm=True)
    # mgcv inherits its LAPACK's rotation of the degenerate null eigenspace;
    # re-rotate to a canonical basis so hea's output is deterministic.
    Xr, null_rot, null_signs = _canonicalize_fs_null_basis(Xr, rank)
    p = Xr.shape[1]

    # Factor levels in alphabetic order (R's factor() default).
    fac_col = data[fterm]
    flev = _factor_levels(fac_col)
    nf = len(flev)
    n = Xr.shape[0]

    # Duplicate block-wise across levels.
    X = np.zeros((n, p * nf))
    fac_arr = fac_col.to_numpy()
    for j, lev in enumerate(flev):
        mask = (fac_arr == lev).astype(float)
        X[:, j * p : (j + 1) * p] = Xr * mask[:, None]

    # Penalty 1: range — diag of [D, 0...0] replicated nf times.
    range_block = np.concatenate([D, np.zeros(null_d)])
    S_range = np.diag(np.tile(range_block, nf))

    # Penalties 2..null_d+1: one per null-space dimension. Each is
    # diag(replicated e_vec) where e_vec has a single 1 at position rank+i.
    S_list: list[np.ndarray] = [S_range]
    for i in range(null_d):
        um = np.zeros(p)
        um[rank + i] = 1.0
        S_list.append(np.diag(np.tile(um, nf)))

    # scale.penalty runs on the final (duplicated) X and each S.
    S_list = _scale_penalty(X, S_list)

    base_raw = _TPRawBasis(
        term=list(others), shift=state["shift"], Xu=state["Xu"],
        m=state["m"], d=state["d"], M=M, k=k, UZ=state["UZ"], w=state["w"],
    )
    raw = _FSRawBasis(
        fterm=fterm, flev=list(flev), p=p, rank=rank, null_d=null_d,
        base_raw=base_raw, P=P, null_rot=null_rot, null_signs=null_signs,
    )
    return [SmoothBlock(
        label=_smooth_label(call), term=term,
        cls="fs.interaction", X=X, S=S_list,
        spec=BasisSpec(raw=raw, by=None, absorb=None),
    )]


def _xz_kr_contrast(X: np.ndarray, m: list[int], inner_p: int) -> np.ndarray:
    """Port of mgcv's `XZKr(X, m)` (smooth.r:3747) — Kronecker sum-to-zero.

    Postmultiplies X by a Kronecker product of sum-to-zero contrasts
    `rbind(diag(m[i]-1), -1)` per factor, with a trailing inner identity of
    size `inner_p`. In mgcv, `inner_p = ncol(X) / prod(m)` so this is the
    base-smooth dimension after all factor blocks have been accounted for.
    Column layout is [level ⊗ inner]: the outer factor indices cycle slowest.
    """
    X = np.asarray(X, dtype=float).copy()
    n = X.shape[0]
    # Replicate mgcv's in-place reshape-then-contract loop.
    for mi in m:
        L = X.size // mi
        X = X.reshape(L, mi, order="F")
        # For each factor: contrast = last block is subtracted from every
        # non-last block, then the last block is dropped. After this the
        # factor dimension is mi-1 and we transpose so the next factor
        # populates the trailing axis.
        X = (X[:, :mi - 1] - X[:, mi - 1:mi]).T
    p = inner_p
    X = X.reshape(X.size // p, p, order="F")
    X = X.T  # matches mgcv's trailing `X <- t(X); dim(X) <- c(length(X)/n, n)`
    return X.reshape(X.size // n, n, order="F").T


def _build_sz_smooth(call: Call, data: pl.DataFrame) -> list[SmoothBlock]:
    """`bs="sz"` — zero-center nested smooth.

    Default base smooth is `tp` (`xt$bs="tp"` in mgcv; alternative bases
    require `xt=list(bs=...)` which is not yet parsed here).

    Fallthrough: mgcv's sz.smooth.spec (smooth.r:2211-2214) checks for any
    factor among the terms; if none is found it reclasses the object as the
    base smooth and returns that constructor's output. We mirror that by
    dispatching to `_build_tp_smooth` on the full term list.

    Factor-present path: build the base tp smooth on the non-factor terms
    (`_tp_raw`, *before* scale.penalty and absorb.cons — mgcv calls the
    constructor directly, not via smoothCon). Duplicate the base X block-wise
    across factor levels, build one block-diagonal penalty per level (one S
    per `prod(nf)` block unless `id=` is set), then apply the Kronecker
    sum-to-zero contrast (`XZKr` — drops the last-level block and subtracts
    it from each other level). scale.penalty runs on the pre-contrast X.
    """
    term = _smooth_term_vars(call)
    fterm, others = _fs_find_factor(term, data)
    if fterm is None:
        return _build_tp_smooth(call, data)

    # Collect all factor terms in the order they appear in the call (mgcv
    # preserves `object$term` order). `_fs_find_factor` returns the first
    # factor; gather the full list for the multi-factor tensor case.
    ftermlist: list[str] = []
    otherlist: list[str] = []
    for t in term:
        if data[t].dtype in (pl.Categorical, pl.Enum):
            ftermlist.append(t)
        else:
            otherlist.append(t)
    # Factor-only case: mgcv reclasses to `re.smooth.spec`. Not exercised
    # by current fixtures — defer until we see one.
    if not otherlist:
        raise NotImplementedError(
            "sz smooth with only factor terms (→ re fallback) not supported"
        )

    # Base smooth: tp on non-factor terms, raw constructor output.
    Xb, Sb_list, M, k, rank, state = _tp_raw(call, data, otherlist)
    if len(Sb_list) != 1:
        raise NotImplementedError("sz with multiply-penalized base basis not supported")
    S_base = Sb_list[0]
    p0 = Xb.shape[1]
    n = Xb.shape[0]

    # Factor level lists and sizes (R's factor levels = Categorical categories).
    flev: list[list] = []
    nf: list[int] = []
    fac_arrs: list[np.ndarray] = []
    for ft in ftermlist:
        fcol = data[ft]
        lv = _factor_levels(fcol)
        flev.append(lv)
        nf.append(len(lv))
        fac_arrs.append(fcol.to_numpy())

    total_levels = int(np.prod(nf))
    p_full = p0 * total_levels

    # Build the expanded X via tensor.prod.model.matrix with factor indicator
    # matrices: X[i, (a1, a2, ..., b)] = prod_j 1{fac_j[i]==lev_j[a_j]} * Xb[i, b].
    # Column layout matches mgcv's: outermost factor index cycles slowest.
    X = np.zeros((n, p_full))
    # Enumerate all factor-index combinations in row-major over factors.
    def _iter_factor_indices():
        if not nf:
            yield ()
            return
        idx = [0] * len(nf)
        while True:
            yield tuple(idx)
            # Increment — last dim fastest.
            for d in range(len(nf) - 1, -1, -1):
                idx[d] += 1
                if idx[d] < nf[d]:
                    break
                idx[d] = 0
            else:
                return

    for blk_pos, combo in enumerate(_iter_factor_indices()):
        mask = np.ones(n, dtype=float)
        for j, a in enumerate(combo):
            mask *= (fac_arrs[j] == flev[j][a]).astype(float)
        X[:, blk_pos * p0 : (blk_pos + 1) * p0] = Xb * mask[:, None]

    # Build penalties. mgcv's sz:
    #   if id is NULL: one penalty per prod(nf) block (prod(nf) penalties).
    #   else: single penalty = sum of block-diagonal S_base across all blocks.
    has_id = call.kwargs.get("id") is not None
    S_list: list[np.ndarray] = []
    if has_id:
        S_combined = np.zeros((p_full, p_full))
        for b in range(total_levels):
            S_combined[b * p0 : (b + 1) * p0, b * p0 : (b + 1) * p0] += S_base
        S_list.append(S_combined)
    else:
        for b in range(total_levels):
            S_b = np.zeros((p_full, p_full))
            S_b[b * p0 : (b + 1) * p0, b * p0 : (b + 1) * p0] = S_base
            S_list.append(S_b)

    # scale.penalty runs on the duplicated X and each S (mgcv applies it
    # before the Kronecker constraint — XZKr rescales column norms but
    # `scale.penalty=TRUE` matches on the pre-contrast `sm$X`).
    S_list = [(S + S.T) / 2.0 for S in S_list]
    S_list = _scale_penalty(X, S_list)

    # Kronecker sum-to-zero: absorb.cons with C = c(0, nf). XZKr drops the
    # last-level block per factor and subtracts it from each non-last block.
    X_out = _xz_kr_contrast(X, nf, p0)
    S_out: list[np.ndarray] = []
    for S in S_list:
        S_k = _xz_kr_contrast(_xz_kr_contrast(S, nf, p0).T, nf, p0).T
        S_out.append(S_k)

    label = _smooth_label(call)
    base_raw = _TPRawBasis(
        term=list(otherlist), shift=state["shift"], Xu=state["Xu"],
        m=state["m"], d=state["d"], M=M, k=k, UZ=state["UZ"], w=state["w"],
    )
    raw = _SZRawBasis(
        term_full=list(term), ftermlist=list(ftermlist),
        flev=[list(lv) for lv in flev], nf=list(nf),
        base_raw=base_raw, p0=p0,
    )
    return [SmoothBlock(
        label=label, term=term, cls="sz.interaction", X=X_out, S=S_out,
        spec=BasisSpec(raw=raw, by=None, absorb=None),
    )]


# ---- ad (adaptive P-spline) -------------------------------------------------
#
# Port of mgcv's smooth.construct.ad.smooth.spec (smooth.r:2435).
#
# The construction of the spline basis is the same as `ps`, but the single
# penalty is replaced with multiple adaptive penalties. In 1D, the k-th
# adaptive penalty is `Db^T diag(V[:, k]) Db` where Db is the 2nd-order
# finite difference on a k_pen-length coefficient grid and V is a smaller
# ps basis of penalty-coefficient weights. In 2D, a tensor of two ps
# bases is built (matching `te(...,bs="ps",...,np=FALSE)` in mgcv), then
# a discretized thin-plate penalty (`Drr^T Drr + Dcc^T Dcc + Dcr^T Dcr`
# from `D2`) is optionally weighted element-wise by V columns to produce
# kp[0]*kp[1] adaptive penalties. The unweighted (m=1 kp.tot=1) case is a
# single fixed penalty.


def _ad_order_m(call: Call, d: int) -> tuple[int, int]:
    """Parse `m` for ad; default 5 in 1D, 3 in 2D. Single value reused per dim."""
    m_src = call.kwargs.get("m")
    vals = _eval_c_vec_ints(m_src) if m_src is not None else None
    if d == 1:
        if not vals:
            return (5, 0)
        return (vals[0], 0)
    # 2D
    if not vals:
        return (3, 3)
    if len(vals) == 1:
        return (vals[0], vals[0])
    return (vals[0], vals[1])


def _ad_default_k(call: Call, d: int) -> tuple[int, ...]:
    """Default k: 40 in 1D, 15 per-margin in 2D."""
    k_src = call.kwargs.get("k")
    default = 40 if d == 1 else 15
    if k_src is None:
        return tuple([default] * d)
    if isinstance(k_src, Literal) and k_src.kind == "num":
        return tuple([int(k_src.value)] * d)
    vals = _eval_c_vec_ints(k_src)
    if not vals:
        return tuple([default] * d)
    if len(vals) == 1:
        return tuple([vals[0]] * d)
    return tuple(vals[:d])


def _ad_Db_1d(nk: int) -> np.ndarray:
    """2nd-order finite difference matrix on R^nk — mgcv's `diff(diff(diag(nk)))`."""
    D = np.eye(nk)
    return np.diff(np.diff(D, axis=0), axis=0)


def _ad_penalty_basis_1d(nk: int, k_pen: int) -> np.ndarray:
    """The inner ps basis V (shape (nk-2, k_pen)) used to weight the
    rows of the outer 2nd-difference matrix Db. Matches mgcv:

        x <- 1:(nk-2)/nk
        s(x, k=k_pen, bs="ps", m=2, fx=TRUE)
    """
    x_v = np.arange(1, nk - 1, dtype=float) / nk
    knots = _ps_knots(x_v, m0=2, k=k_pen)
    return _ps_basis(x_v, knots, m0=2)


def _ad_D2(ni: int, nj: int) -> dict:
    """Port of mgcv's `D2(ni, nj)` (smooth.r:2377).

    Returns second-difference matrices (`Drr`, `Dcc`, `Dcr`) on a ni-by-nj
    coefficient grid, plus the row/col indices of each D's central
    stencil (used to evaluate the penalty-weighting basis at those
    locations). The mixed-derivative factor `sqrt(0.125)` bakes the `2`
    from the thin-plate penalty into Dcr, so
    `Drr^T Drr + Dcc^T Dcc + Dcr^T Dcr` is the discrete TPS penalty.
    """
    Ind = np.arange(ni * nj).reshape(nj, ni).T  # column-major like R
    rmt = np.tile(np.arange(1, ni + 1), nj)
    cmt = np.repeat(np.arange(1, nj + 1), ni)

    def _mfil(shape, rows, flat_cols, vals):
        M = np.zeros(shape, dtype=float)
        M[rows, flat_cols] = vals
        return M

    # Drr: 2nd diff along rows (i direction), fixed j.
    ci0 = Ind[1 : ni - 1, 0:nj].ravel(order="F")
    n_ci = len(ci0)
    rows = np.arange(n_ci)
    Drr = _mfil((n_ci, ni * nj), rows, ci0, -2.0)
    ci_back = Ind[0 : ni - 2, 0:nj].ravel(order="F")
    Drr[rows, ci_back] = 1.0
    ci_fwd = Ind[2:ni, 0:nj].ravel(order="F")
    Drr[rows, ci_fwd] = 1.0
    rr_ri = rmt[ci0]
    rr_ci = cmt[ci0]

    # Dcc: 2nd diff along cols.
    ci0 = Ind[0:ni, 1 : nj - 1].ravel(order="F")
    n_ci = len(ci0)
    rows = np.arange(n_ci)
    Dcc = _mfil((n_ci, ni * nj), rows, ci0, -2.0)
    ci_back = Ind[0:ni, 0 : nj - 2].ravel(order="F")
    Dcc[rows, ci_back] = 1.0
    ci_fwd = Ind[0:ni, 2:nj].ravel(order="F")
    Dcc[rows, ci_fwd] = 1.0
    cc_ri = rmt[ci0]
    cc_ci = cmt[ci0]

    # Dcr: cross derivative.
    ci0 = Ind[1 : ni - 1, 1 : nj - 1].ravel(order="F")
    n_ci = len(ci0)
    rows = np.arange(n_ci)
    cr_ri = rmt[ci0]
    cr_ci = cmt[ci0]
    Dcr = np.zeros((n_ci, ni * nj))
    s = np.sqrt(0.125)
    ci_mm = Ind[0 : ni - 2, 0 : nj - 2].ravel(order="F")
    Dcr[rows, ci_mm] = s
    ci_pp = Ind[2:ni, 2:nj].ravel(order="F")
    Dcr[rows, ci_pp] = s
    ci_mp = Ind[0 : ni - 2, 2:nj].ravel(order="F")
    Dcr[rows, ci_mp] = -s
    ci_pm = Ind[2:ni, 0 : nj - 2].ravel(order="F")
    Dcr[rows, ci_pm] = -s

    return dict(
        Drr=Drr, Dcc=Dcc, Dcr=Dcr,
        rr_ri=rr_ri, rr_ci=rr_ci,
        cc_ri=cc_ri, cc_ci=cc_ci,
        cr_ri=cr_ri, cr_ci=cr_ci,
        rmt=rmt, cmt=cmt,
    )


def _ad_inner_2d_basis(kp: tuple[int, int], Db: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the inner penalty-weight matrices Vrr, Vcc, Vcr (one column per
    adaptive penalty) for the 2D adaptive case with `kp.tot > 1`.

    Mirrors mgcv:
        m <- min(min(kp)-2, 1); m <- c(m, m)
        ps2 <- smooth.construct(te(i, j, bs="ps", k=kp, fx=TRUE, m=m, np=FALSE),
                                data=data.frame(i=rmt, j=cmt))
        Vrr/Vcc/Vcr <- Predict.matrix(ps2, <rr/cc/cr indices>)

    With `np=FALSE`, te's basis is the row-wise Kronecker product of
    two ps bases, so Predict.matrix evaluates each margin at the given
    (i, j) grid coordinates and tensor-multiplies.
    """
    kp_tot = kp[0] * kp[1]
    if kp_tot == 3:
        # "Planar adaptiveness": V = [1, Drr_flat, Dcc_flat] — but Drr/Dcc
        # here are the indices, not the matrices. Use the rr/cc/cr indices
        # directly as the two planar coordinates.
        raise NotImplementedError("ad 2D kp.tot=3 planar adaptiveness not implemented")
    # General adaptive: build an inner ps basis per margin, then Kronecker.
    m_inner = min(min(kp) - 2, 1)
    # Inner basis on each margin — the grid is the full (rmt, cmt) grid.
    rmt = Db["rmt"].astype(float)
    cmt = Db["cmt"].astype(float)
    # Each ps basis needs knots over the relevant range.
    ki, kj = kp
    knots_i = _ps_knots(rmt, m0=m_inner, k=ki)
    knots_j = _ps_knots(cmt, m0=m_inner, k=kj)

    def _eval_V(ri: np.ndarray, ci: np.ndarray) -> np.ndarray:
        Bi = _ps_basis(ri.astype(float), knots_i, m0=m_inner)
        Bj = _ps_basis(ci.astype(float), knots_j, m0=m_inner)
        n = Bi.shape[0]
        # te(i, j) has margin-i as outermost in tensor.prod.model.matrix,
        # so V[r, a*kj + b] = Bi[r, a] * Bj[r, b].
        return (Bi[:, :, None] * Bj[:, None, :]).reshape(n, ki * kj)

    Vrr = _eval_V(Db["rr_ri"], Db["rr_ci"])
    Vcc = _eval_V(Db["cc_ri"], Db["cc_ci"])
    Vcr = _eval_V(Db["cr_ri"], Db["cr_ci"])
    return Vrr, Vcc, Vcr


def _build_ad_smooth(call: Call, data: pl.DataFrame) -> list[SmoothBlock]:
    """`bs="ad"` — adaptive P-spline (1D or 2D).

    1D: builds a standard ps basis, then replaces its single penalty with
    `k_pen = m` adaptive penalties of the form `Db^T diag(V[:,i]) Db`.
    2D: builds a tensor of two ps bases (equivalent to the X matrix of
    `te(..., bs="ps", np=FALSE)`), then replaces penalties with either a
    single discrete-TPS penalty (kp.tot=1) or `kp[0]*kp[1]` adaptive
    versions weighted by an inner ps basis.
    """
    term = _smooth_term_vars(call)
    d = len(term)
    if d > 2:
        raise ValueError("ad smooth supports only 1 or 2 covariates")

    if d == 1:
        x = data[term[0]].to_numpy().astype(float)
        m = _ad_order_m(call, 1)
        k = _ad_default_k(call, 1)[0]
        knots = _ps_knots(x, m0=2, k=k)
        X = _ps_basis(x, knots, m0=2)
        nk = X.shape[1]
        k_pen = m[0]
        if k_pen >= nk - 2:
            raise ValueError("ad penalty basis too large for smoothing basis")
        Db = _ad_Db_1d(nk)
        V = _ad_penalty_basis_1d(nk, k_pen)
        S_list = [Db.T @ (V[:, i : i + 1] * Db) for i in range(k_pen)]
        raw = _ADRawBasis(
            term=list(term), knots_per_term=[knots],
            m0=2, k_per_term=[X.shape[1]],
        )
        return _apply_by_and_absorb(
            call, data, X, S_list, "pspline.smooth", term, raw_basis=raw,
        )

    # d == 2
    k_vec = _ad_default_k(call, 2)
    ki, kj = int(k_vec[0]), int(k_vec[1])
    kp = _ad_order_m(call, 2)
    xi = data[term[0]].to_numpy().astype(float)
    xj = data[term[1]].to_numpy().astype(float)
    knots_i = _ps_knots(xi, m0=2, k=ki)
    knots_j = _ps_knots(xj, m0=2, k=kj)
    Xi = _ps_basis(xi, knots_i, m0=2)
    Xj = _ps_basis(xj, knots_j, m0=2)
    # Row-wise Kronecker (tensor.prod.model.matrix on two matrices). mgcv
    # iterates column of Xj outermost: X[r, a + ki*b] = Xi[r, a] * Xj[r, b].
    n = Xi.shape[0]
    X = (Xi[:, :, None] * Xj[:, None, :]).reshape(n, ki * kj)
    Db = _ad_D2(ki, kj)
    kp_tot = kp[0] * kp[1]
    if kp_tot == 1:
        Drr, Dcc, Dcr = Db["Drr"], Db["Dcc"], Db["Dcr"]
        S = Drr.T @ Drr + Dcc.T @ Dcc + Dcr.T @ Dcr
        S_list = [S]
    else:
        Vrr, Vcc, Vcr = _ad_inner_2d_basis(kp, Db)
        Drr, Dcc, Dcr = Db["Drr"], Db["Dcc"], Db["Dcr"]
        S_list = []
        for i in range(kp_tot):
            S = (Drr.T @ (Vrr[:, i : i + 1] * Drr)
                 + Dcc.T @ (Vcc[:, i : i + 1] * Dcc)
                 + Dcr.T @ (Vcr[:, i : i + 1] * Dcr))
            S_list.append(S)
    raw = _ADRawBasis(
        term=list(term), knots_per_term=[knots_i, knots_j],
        m0=2, k_per_term=[Xi.shape[1], Xj.shape[1]],
    )
    return _apply_by_and_absorb(
        call, data, X, S_list, "pspline.smooth", term, raw_basis=raw,
    )


# ---- te / ti / t2 (tensor-product smooths) ---------------------------------
#
# Port of mgcv's te()/ti()/t2() specifications and their constructors at
# smooth.r:359-711 (wrappers) and smooth.r:741-887 (tensor.smooth.spec),
# plus t2 pieces at smooth.r:911+.
#
# te:  margins built raw (no marginal absorb.cons); Khatri-Rao X;
#      Kronecker-lifted penalties; outer absorb.cons.
# ti:  like te but inter=TRUE — each margin is centered (absorb.cons) first
#      (subject to `mc` kwarg), and the outer absorb.cons is skipped.
# t2:  Wood's alternative: split each margin into null + range, form
#      block-structured X and list of block-ridge penalties.


def _te_parse_vec(val, n: int, default, cast) -> list:
    """Parse a te-style kwarg that may be scalar or c(...). Returns a
    length-n list. `default` is the scalar fallback; `cast` converts each
    value."""
    if val is None:
        return [default] * n
    if isinstance(val, Call) and val.fn == "c":
        vals = [cast(a) for a in val.args]
        if len(vals) == 1:
            return vals * n
        if len(vals) == n:
            return vals
        raise ValueError(f"c(...) length {len(vals)} doesn't match margin count {n}")
    return [cast(val)] * n


def _te_cast_int(node) -> int:
    if isinstance(node, Literal) and node.kind == "num":
        return int(node.value)
    if isinstance(node, UnaryOp) and node.op == "-":
        return -_te_cast_int(node.operand)
    raise ValueError(f"expected integer literal, got {node!r}")


def _te_cast_str(node) -> str:
    if isinstance(node, Literal) and node.kind == "str":
        return str(node.value)
    raise ValueError(f"expected string literal, got {node!r}")


def _te_cast_bool(node) -> bool:
    if isinstance(node, Name) and node.ident in ("TRUE", "T", "FALSE", "F"):
        return node.ident in ("TRUE", "T")
    if isinstance(node, Literal):
        if node.kind == "bool":
            return bool(node.value)
        if node.kind == "num":
            return bool(node.value)
    raise ValueError(f"expected boolean, got {node!r}")


def _te_parse_margins(call: Call, data: pl.DataFrame) -> list[dict]:
    """Parse a te/ti/t2 call into a list of margin specs.

    Each margin spec is a dict `{term, bs, k, m, fx}` describing what to
    pass to the underlying marginal smooth constructor.
    """
    term = _smooth_term_vars(call)
    dim = len(term)
    # d: number of covariates per margin. Default is c(1, ..., 1).
    d_src = call.kwargs.get("d")
    if d_src is None:
        d_list = [1] * dim
    else:
        d_list = _te_parse_vec(d_src, dim, 1, _te_cast_int)
        if sum(d_list) != dim:
            raise ValueError(f"te d= must sum to dim ({sum(d_list)} != {dim})")
    n_bases = len(d_list)

    k_list = _te_parse_vec(call.kwargs.get("k"), n_bases, 5, _te_cast_int)
    bs_list = _te_parse_vec(call.kwargs.get("bs"), n_bases, "cr", _te_cast_str)
    fx_list = _te_parse_vec(call.kwargs.get("fx"), n_bases, False, _te_cast_bool)
    # m is parsed as-is per margin (default NA → None here); each margin's
    # constructor handles its own default.
    m_src = call.kwargs.get("m")
    if m_src is None:
        m_list: list = [None] * n_bases
    elif isinstance(m_src, Call) and m_src.fn == "c":
        if len(m_src.args) == 1:
            m_list = [m_src.args[0]] * n_bases
        elif len(m_src.args) == n_bases:
            m_list = list(m_src.args)
        else:
            raise ValueError(f"m= length {len(m_src.args)} doesn't match margin count {n_bases}")
    else:
        m_list = [m_src] * n_bases

    # Promote bs=cr/cs/ps/cp to tp for multi-d margins.
    for i in range(n_bases):
        if d_list[i] > 1 and bs_list[i] in ("cr", "cs", "ps", "cp"):
            bs_list[i] = "tp"

    specs = []
    j = 0
    for i in range(n_bases):
        j1 = j + d_list[i]
        specs.append(dict(
            term=term[j:j1], bs=bs_list[i], k=k_list[i],
            m=m_list[i], fx=fx_list[i],
        ))
        j = j1
    return specs


def _te_make_margin_call(spec: dict) -> Call:
    """Build a synthetic `s(term..., k=..., bs=..., m=..., fx=...)` Call
    for a single margin so we can reuse the existing bs-specific raw
    helpers."""
    args: list = [Name(ident=t) for t in spec["term"]]
    kwargs: dict = {}
    kwargs["k"] = Literal(value=spec["k"], kind="num")
    kwargs["bs"] = Literal(value=spec["bs"], kind="str")
    if spec["m"] is not None:
        kwargs["m"] = spec["m"]
    if spec["fx"]:
        kwargs["fx"] = Name(ident="TRUE")
    return Call(fn="s", args=args, kwargs=kwargs)


def _te_build_margin_raw(
    spec: dict, data: pl.DataFrame,
) -> tuple[np.ndarray, list[np.ndarray], object, bool, _RawBasis]:
    """Build the bare margin (no absorb.cons) — returns
    `(X, S_list, predict, noterp, raw)`. `noterp=True` signals that this
    basis is already "nicely parameterised" and the tensor constructor
    should skip the np=TRUE SVD reparameterization (matches mgcv's
    `noterp` attribute on cr / cc / cs). ``raw`` is the corresponding
    `_RawBasis` for predict-time replay.
    """
    mcall = _te_make_margin_call(spec)
    bs = spec["bs"]
    term = spec["term"]
    if bs == "cr":
        X, S_list, knots = _cr_raw(mcall, data, term)
        raw = _CRRawBasis(term=term[0], knots=knots)
        def _predict(x_new: np.ndarray) -> np.ndarray:
            return _cr_basis(np.asarray(x_new, dtype=float), knots)
        return X, S_list, _predict, True, raw
    if bs == "tp":
        X, S_list, M, k, _rank, state = _tp_raw(mcall, data, term)
        raw = _TPRawBasis(
            term=list(term), shift=state["shift"], Xu=state["Xu"],
            m=state["m"], d=state["d"], M=M, k=k,
            UZ=state["UZ"], w=state["w"],
        )
        # np-reparam predict callable: only invoked when len(term) == 1.
        def _predict(x_new: np.ndarray) -> np.ndarray:
            df = pl.DataFrame({term[0]: np.asarray(x_new, dtype=float)})
            return raw.eval(df)
        return X, S_list, _predict, False, raw
    raise NotImplementedError(f"te/ti margin with bs={bs!r} not yet supported")


def _te_build_margin_centered(
    spec: dict, data: pl.DataFrame,
) -> tuple[np.ndarray, list[np.ndarray], object, bool, _RawBasis]:
    """Build a centered margin (absorb.cons applied), returning
    `(X, S_list, predict, noterp, raw)`. Equivalent to
    `smoothCon(..., absorb.cons=TRUE)[[1]]` but keeps the Z matrix so
    `predict(x_new)` evaluates the raw basis at new points and applies
    the same sum-to-zero rotation. ``raw`` chains the bare basis with
    the post-multiplication ``Z`` for predict-time replay.
    """
    mcall = _te_make_margin_call(spec)
    bs = spec["bs"]
    term = spec["term"]
    if bs == "cr":
        X_raw, S_raw, knots = _cr_raw(mcall, data, term)
        S_sym = [(S + S.T) / 2.0 for S in S_raw]
        S_scaled = _scale_penalty(X_raw, S_sym)
        C = X_raw.mean(axis=0)
        Q, _ = np.linalg.qr(C.reshape(-1, 1), mode="complete")
        Z = Q[:, 1:]
        X = X_raw @ Z
        S_list = [Z.T @ S @ Z for S in S_scaled]
        bare = _CRRawBasis(term=term[0], knots=knots)
        raw = _LinearTransformRawBasis(inner=bare, M=Z)
        def _predict(x_new: np.ndarray) -> np.ndarray:
            return _cr_basis(np.asarray(x_new, dtype=float), knots) @ Z
        return X, S_list, _predict, True, raw
    if bs == "tp":
        X_raw, S_raw, M, k, _rank, state = _tp_raw(mcall, data, term)
        S_sym = [(S + S.T) / 2.0 for S in S_raw]
        S_scaled = _scale_penalty(X_raw, S_sym)
        C = X_raw.mean(axis=0)
        Q, _ = np.linalg.qr(C.reshape(-1, 1), mode="complete")
        Z = Q[:, 1:]
        X = X_raw @ Z
        S_list = [Z.T @ S @ Z for S in S_scaled]
        bare = _TPRawBasis(
            term=list(term), shift=state["shift"], Xu=state["Xu"],
            m=state["m"], d=state["d"], M=M, k=k,
            UZ=state["UZ"], w=state["w"],
        )
        raw = _LinearTransformRawBasis(inner=bare, M=Z)
        def _predict(x_new: np.ndarray) -> np.ndarray:
            df = pl.DataFrame({term[0]: np.asarray(x_new, dtype=float)})
            return bare.eval(df) @ Z
        return X, S_list, _predict, False, raw
    raise NotImplementedError(f"ti/t2 centered margin with bs={bs!r} not yet supported")


def _te_reparam_margin(X: np.ndarray, S_list: list[np.ndarray], x_vals: np.ndarray,
                       predict_basis) -> tuple[np.ndarray, list[np.ndarray], np.ndarray | None]:
    """Reparameterize a margin to spread basis evenly in x (matches mgcv's
    `np=TRUE` path at smooth.r:796-822).

    Evaluates `predict_basis(knt)` at `knt = seq(min(x), max(x), length=np)`
    where `np = ncol(X)`, SVDs that matrix, and applies `XP = V D^-1 U^T` so
    that `X_new = X @ XP`, `S_new = XP^T @ S @ XP`. Returns None for XP if
    the matrix is too ill-conditioned.
    """
    np_cols = X.shape[1]
    knt = np.linspace(float(np.min(x_vals)), float(np.max(x_vals)), np_cols)
    P = predict_basis(knt)
    U, d, Vt = np.linalg.svd(P, full_matrices=False)
    if d[-1] / d[0] < float(np.finfo(float).eps) ** 0.66:
        return X, S_list, None
    XP = Vt.T @ (U.T / d[:, None])  # V @ diag(1/d) @ U^T
    X_new = X @ XP
    S_new = [XP.T @ S @ XP for S in S_list]
    return X_new, S_new, XP


def _tensor_prod_X(Xm: list[np.ndarray]) -> np.ndarray:
    """Row-wise Kronecker / Khatri-Rao product of margin X matrices.
    Matches mgcv's `tensor.prod.model.matrix`: ith row = X[0][i,] %x% X[1][i,] %x% ...
    (margin 0 outermost)."""
    n = Xm[0].shape[0]
    out = Xm[0]
    for Xk in Xm[1:]:
        p_out = out.shape[1]
        p_k = Xk.shape[1]
        out = (out[:, :, None] * Xk[:, None, :]).reshape(n, p_out * p_k)
    return out


def _tensor_prod_S(Sm: list[np.ndarray]) -> list[np.ndarray]:
    """Kronecker-lift each marginal penalty over the tensor basis.
    Matches mgcv's `tensor.prod.penalties`: S_i lifts to `I_1 ⊗ ... ⊗ S_i ⊗ ... ⊗ I_m`.
    """
    m = len(Sm)
    dims = [S.shape[0] for S in Sm]
    out: list[np.ndarray] = []
    for i in range(m):
        M = Sm[i] if i == 0 else np.eye(dims[0])
        for j in range(1, m):
            M = np.kron(M, Sm[i] if i == j else np.eye(dims[j]))
        # Symmetrize (mgcv does this too).
        if M.shape[0] == M.shape[1]:
            M = 0.5 * (M + M.T)
        out.append(M)
    return out


def _build_te_smooth(call: Call, data: pl.DataFrame, *, inter: bool = False,
                     mc: list[bool] | None = None) -> list[SmoothBlock]:
    """`te(...)` / `ti(...)` constructor.

    Shared code path, differentiated by:
      - `inter=False` (te): raw margins, outer absorb.cons applied.
      - `inter=True` (ti): centered margins (per `mc`), no outer absorb.cons.
    """
    specs = _te_parse_margins(call, data)
    n_bases = len(specs)

    if inter:
        if mc is None:
            mc_list = [True] * n_bases
        else:
            # mgcv accepts 0/1 or FALSE/TRUE; length 1 recycles.
            if len(mc) == 1:
                mc_list = [bool(mc[0])] * n_bases
            elif len(mc) == n_bases:
                mc_list = [bool(v) for v in mc]
            else:
                raise ValueError(f"mc= length {len(mc)} doesn't match margin count {n_bases}")
    else:
        mc_list = [False] * n_bases

    np_flag = call.kwargs.get("np")
    do_np = True
    if np_flag is not None:
        do_np = _te_cast_bool(np_flag)

    Xm: list[np.ndarray] = []
    Sm: list[np.ndarray] = []
    margin_raws: list[_RawBasis] = []
    for i, spec in enumerate(specs):
        if mc_list[i]:
            Xi, Si_list, predict_i, noterp_i, raw_i = _te_build_margin_centered(spec, data)
        else:
            Xi, Si_list, predict_i, noterp_i, raw_i = _te_build_margin_raw(spec, data)
        if len(Si_list) != 1:
            raise ValueError(f"te margin {i} has {len(Si_list)} penalties; only one allowed")
        S_i = Si_list[0]

        # np=TRUE SVD reparam, skipped for margins that opt out (cr/cc/cs set
        # noterp=TRUE — their basis is already "nicely parameterised").
        if do_np and len(spec["term"]) == 1 and not noterp_i:
            x_vals = data[spec["term"][0]].to_numpy().astype(float)
            Xi, S_list_new, XP = _te_reparam_margin(Xi, [S_i], x_vals, predict_i)
            S_i = S_list_new[0]
            if XP is not None:
                raw_i = _LinearTransformRawBasis(inner=raw_i, M=XP)

        # Scale each marginal penalty by its largest eigenvalue.
        eigs = np.linalg.eigvalsh(0.5 * (S_i + S_i.T))
        top = float(eigs[-1])
        if top > 0:
            S_i = S_i / top

        Xm.append(Xi)
        Sm.append(S_i)
        margin_raws.append(raw_i)

    X = _tensor_prod_X(Xm)
    S_list = _tensor_prod_S(Sm)

    # fx: drop margins whose fx=TRUE.
    for i in reversed(range(n_bases)):
        if specs[i]["fx"]:
            del S_list[i]

    cls = "tensor.smooth"
    label = _smooth_label(call)
    term_all = _smooth_term_vars(call)
    tensor_raw = _TensorRawBasis(margins=margin_raws)

    if inter:
        # Skip outer absorb.cons (C = matrix(0,0,0)).
        S_list = _scale_penalty(X, S_list)
        return [SmoothBlock(
            label=label, term=term_all, cls=cls, X=X, S=S_list,
            spec=BasisSpec(raw=tensor_raw, by=None, absorb=None),
        )]

    # te: outer absorb.cons.
    return _apply_by_and_absorb(
        call, data, X, S_list, cls, term_all, raw_basis=tensor_raw,
    )


def _build_ti_smooth(call: Call, data: pl.DataFrame) -> list[SmoothBlock]:
    """`ti(...)` — like te but each margin is centered (absorb.cons
    applied) before the tensor, and the outer absorb.cons is skipped.

    The `mc` kwarg selects which margins get centered (default: all)."""
    mc_src = call.kwargs.get("mc")
    if mc_src is None:
        mc_vals = None
    elif isinstance(mc_src, Call) and mc_src.fn == "c":
        mc_vals = [_te_cast_int(a) != 0 for a in mc_src.args]
    else:
        mc_vals = [_te_cast_int(mc_src) != 0]
    return _build_te_smooth(call, data, inter=True, mc=mc_vals)


def _t2_margin_raw_and_rank(
    spec: dict, data: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray, int, _RawBasis]:
    """Build a t2 margin (no absorb.cons) and return `(X, S, rank, raw)`.
    The marginal penalty rank determines the range/null split used by
    `t2.model.matrix`. ``raw`` is the corresponding `_RawBasis` for the
    bare margin (predict-time replay)."""
    mcall = _te_make_margin_call(spec)
    bs = spec["bs"]
    term = spec["term"]
    if bs == "cr":
        X, S_list, knots = _cr_raw(mcall, data, term)
        S = 0.5 * (S_list[0] + S_list[0].T)
        # cr null.space.dim = 2 for un-shrunk cr.
        rank = X.shape[1] - 2
        raw = _CRRawBasis(term=term[0], knots=knots)
        return X, S, rank, raw
    if bs == "tp":
        X, S_list, M, k, rank, state = _tp_raw(mcall, data, term)
        S = 0.5 * (S_list[0] + S_list[0].T)
        raw = _TPRawBasis(
            term=list(term), shift=state["shift"], Xu=state["Xu"],
            m=state["m"], d=state["d"], M=M, k=k,
            UZ=state["UZ"], w=state["w"],
        )
        return X, S, rank, raw
    raise NotImplementedError(f"t2 margin with bs={bs!r} not yet supported")


def _t2_model_matrix(
    Xm: list[np.ndarray], ranks: list[int],
) -> tuple[np.ndarray, list[int]]:
    """Port of mgcv's `t2.model.matrix` with `full=FALSE, ord=NULL`.

    Each margin's X is split into range (first `rank[i]` cols) and null
    (remaining cols). Builds all Kronecker combinations and returns
    `(X, sub_cols)` where `sub_cols` is the column count per sub-block
    with the trailing all-null block dropped (that block is unpenalized).
    """
    n = Xm[0].shape[0]

    def _row_kron(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        pa, pb = A.shape[1], B.shape[1]
        return (A[:, :, None] * B[:, None, :]).reshape(n, pa * pb)

    r0 = ranks[0]
    Z0 = Xm[0][:, :r0]
    null0_exists = r0 < Xm[0].shape[1]
    blocks: list[np.ndarray] = [Z0]
    if null0_exists:
        blocks.append(Xm[0][:, r0:])
    no_null = not null0_exists

    for i in range(1, len(Xm)):
        ri = ranks[i]
        Zi = Xm[i][:, :ri]
        null_i_exists = ri < Xm[i].shape[1]
        if not null_i_exists:
            no_null = True
        Ni = Xm[i][:, ri:] if null_i_exists else None
        new_blocks: list[np.ndarray] = []
        # Range products first: X1[ii] * Zi for all ii.
        for Xii in blocks:
            new_blocks.append(_row_kron(Xii, Zi))
        # Then null products: X1[ii] * Ni for all ii (if null exists).
        if null_i_exists:
            for Xii in blocks:
                new_blocks.append(_row_kron(Xii, Ni))
        blocks = new_blocks

    sub_cols = [B.shape[1] for B in blocks]
    X = np.concatenate(blocks, axis=1)
    if not no_null:
        # Trailing block is the pure null×null×... tail — unpenalized.
        sub_cols = sub_cols[:-1]
    return X, sub_cols


def _build_t2_smooth(call: Call, data: pl.DataFrame) -> list[SmoothBlock]:
    """`t2(...)` — Wood's alternative tensor product.

    Each margin is reparameterized by `nat.param(type=3, unit.fnorm=TRUE)`
    so its penalty becomes diag([1..1, 0..0]) with range (rank) first, then
    null. `t2.model.matrix(full=FALSE)` builds sub-blocks via all Kronecker
    combinations; each penalized sub-block gets a simple identity ridge.
    The tensor's null space is then constrained by a single row C
    (smooth.r:1117-1120) before scaling and constraint absorption.
    """
    specs = _te_parse_margins(call, data)
    n_bases = len(specs)

    Xm: list[np.ndarray] = []
    ranks: list[int] = []
    margin_raws: list[_RawBasis] = []
    P_per_margin: list[np.ndarray] = []
    for spec in specs:
        Xi_raw, Si, ri, raw_i = _t2_margin_raw_and_rank(spec, data)
        Xi_np, _D, P_i = _nat_param(Xi_raw, Si, rank=ri, type_=3, unit_fnorm=True)
        Xm.append(Xi_np)
        ranks.append(ri)
        margin_raws.append(raw_i)
        P_per_margin.append(P_i)

    X_raw_full, sub_cols = _t2_model_matrix(Xm, ranks)
    nsc = len(sub_cols)
    p = X_raw_full.shape[1]

    # Penalties: simple ridge on each sub-block.
    cx = [0]
    for s in sub_cols:
        cx.append(cx[-1] + s)
    S_list: list[np.ndarray] = []
    for j in range(nsc):
        D = np.zeros(p)
        D[cx[j]:cx[j + 1]] = 1.0
        S_list.append(np.diag(D))

    # fx handling: drop penalties for margins marked fx (mgcv only has per-margin fx,
    # but applies it per-sub-block by index — no fixture exercises it here).
    for i in reversed(range(n_bases)):
        if specs[i]["fx"]:
            raise NotImplementedError("t2 with fx=TRUE not yet supported")

    # Tensor null-space constraint (smooth.r:1117-1120). Rank of the null is
    # p - sum(sub_cols). Build C = [0_{nup}, colSums(X[:, nup:])] as 1×p.
    nup = sum(sub_cols)
    null_dim = p - nup
    cls = "t2.smooth"
    label = _smooth_label(call)
    term_all = _smooth_term_vars(call)

    if null_dim == 0:
        # No null space, no identifiability constraint needed; sm$Cp is NULL,
        # so fit basis == predict basis.
        S_list = _scale_penalty(X_raw_full, S_list)
        t2_raw = _T2RawBasis(
            margins=margin_raws, P_per_margin=P_per_margin,
            ranks=ranks, null_dim=0, Zn=None,
        )
        return [SmoothBlock(
            label=label, term=term_all, cls=cls, X=X_raw_full, S=S_list,
            spec=BasisSpec(raw=t2_raw, by=None, absorb=None),
        )]

    # Predict-time basis (mgcv's full absorb.cons via sm$Cp = colSums(X_raw_full)):
    # `Predict.matrix.t2.smooth` applies Z_p = qr.qy(qrc, [0; I_q]) — equivalent
    # to Q_p[:, 1:] from the complete QR of colSums(X_raw_full).T — to the raw
    # t2 design. Z_p only depends on fit-time X_raw_full (not on new data), so
    # we cache it here and replay it via _T2PredictRawBasis.
    cP = X_raw_full.sum(axis=0).reshape(-1, 1)
    Q_p, _ = np.linalg.qr(cP, mode="complete")
    Z_p = Q_p[:, 1:]
    t2_predict = _T2PredictRawBasis(
        margins=margin_raws, P_per_margin=P_per_margin,
        ranks=ranks, Z_p=Z_p,
    )

    if null_dim == 1:
        # Fix the single null-space parameter to zero — drop its column (smooth.r:1118).
        keep = np.ones(p, dtype=bool)
        keep[-1] = False
        X = X_raw_full[:, keep]
        S_list = [S[np.ix_(keep, keep)] for S in S_list]
        S_list = _scale_penalty(X, S_list)
        t2_raw = _T2RawBasis(
            margins=margin_raws, P_per_margin=P_per_margin,
            ranks=ranks, null_dim=1, Zn=None,
        )
        Xp_in = X_raw_full @ Z_p
        X_bar = X.mean(axis=0)
        M = np.linalg.lstsq(Xp_in, X - X_bar, rcond=None)[0]
        return [SmoothBlock(
            label=label, term=term_all, cls=cls, X=X, S=S_list,
            spec=BasisSpec(raw=t2_raw, by=None, absorb=None,
                           predict_raw=t2_predict, coef_remap=(M, X_bar)),
        )]

    # Partial absorb.cons (smooth.r:4076-4100): C has zero cols on the range
    # space, so only the null-space cols of X / S get rotated. QR is on the
    # nx×1 slice cN = colSums(X_N); Z' = Q[:, 1:] spans the null of cN within
    # the null-space coordinates. Range-space cols pass through unchanged.
    S_list = _scale_penalty(X_raw_full, S_list)
    X_R = X_raw_full[:, :nup]
    X_N = X_raw_full[:, nup:]
    cN = X_N.sum(axis=0).reshape(-1, 1)
    Qn, _ = np.linalg.qr(cN, mode="complete")
    Zn = Qn[:, 1:]
    X = np.concatenate([X_R, X_N @ Zn], axis=1)
    S_list_new: list[np.ndarray] = []
    for S in S_list:
        S_RR = S[:nup, :nup]
        S_RN = S[:nup, nup:] @ Zn
        S_NR = Zn.T @ S[nup:, :nup]
        S_NN = Zn.T @ S[nup:, nup:] @ Zn
        top = np.concatenate([S_RR, S_RN], axis=1)
        bot = np.concatenate([S_NR, S_NN], axis=1)
        S_list_new.append(np.concatenate([top, bot], axis=0))
    t2_raw = _T2RawBasis(
        margins=margin_raws, P_per_margin=P_per_margin,
        ranks=ranks, null_dim=null_dim, Zn=Zn,
    )
    Xp_in = X_raw_full @ Z_p
    X_bar = X.mean(axis=0)
    M = np.linalg.lstsq(Xp_in, X - X_bar, rcond=None)[0]
    return [SmoothBlock(
        label=label, term=term_all, cls=cls, X=X, S=S_list_new,
        spec=BasisSpec(raw=t2_raw, by=None, absorb=None,
                       predict_raw=t2_predict, coef_remap=(M, X_bar)),
    )]


def materialize_smooths(
    expanded: ExpandedFormula, data: pl.DataFrame,
) -> list[list[SmoothBlock]]:
    """Materialize each smooth in `expanded.smooths` to one or more blocks.

    Returns a list parallel to `expanded.smooths`; each entry is the list of
    SmoothBlock that mgcv's smoothCon returns for that spec. Most smooths
    produce exactly one block; `by = <factor>` with `id` yields n_levels.

    Smooth args may be plain column names (``s(x)``) or expressions
    (``s(I(b.depth^.5))``, ``s(log(x))``). Expressions are deparsed to a
    synthesised column name, NA-dropped on their underlying source
    columns, then materialised into ``data`` once before the per-smooth
    builders run. Predict-time replay uses the same machinery via
    :func:`_apply_smooth_arg_exprs` against the new dataframe.
    """
    # NA-drop on every column the smooths reference. For plain ``Name``
    # args this is just the column name; for expressions we union in every
    # ``Name.ident`` mentioned inside the AST so e.g. ``s(I(b.depth^.5))``
    # NA-drops on ``b.depth`` (not on the not-yet-materialised
    # ``"I(b.depth^0.5)"`` synth column).
    referenced: set[str] = set()
    for c in expanded.smooths:
        for a in c.args:
            if isinstance(a, Name):
                if a.ident in data.columns:
                    referenced.add(a.ident)
            else:
                _collect_name_idents(a, referenced)
        # ``by=`` may also be a non-Name expression in mgcv; the existing
        # ``_eval_by_col`` only supports a small set of forms and pulls
        # source columns out itself, so we skip walking it here.
    referenced &= set(data.columns)
    if referenced:
        data = data.drop_nulls(subset=list(referenced))

    # Materialise smooth-arg expressions into synthesised columns. After
    # this, every term name returned by ``_smooth_term_vars`` resolves
    # against ``data.columns`` directly.
    expr_map = _smooth_arg_expr_map(expanded)
    if expr_map:
        data = _apply_smooth_arg_exprs(data, expr_map)

    out: list[list[SmoothBlock]] = []
    for call in expanded.smooths:
        # Tensor-product smooths dispatch by the top-level fn (te/ti/t2),
        # not by `bs=` (which in their case describes each margin's basis).
        if call.fn in ("te", "ti", "t2"):
            if call.fn == "te":
                out.append(_build_te_smooth(call, data))
            elif call.fn == "ti":
                out.append(_build_ti_smooth(call, data))
            else:
                out.append(_build_t2_smooth(call, data))
            continue
        bs = _smooth_bs(call)
        if bs == "re":
            out.append(_build_re_smooth(call, data))
        elif bs == "cr":
            out.append(_build_cr_smooth(call, data))
        elif bs == "cc":
            out.append(_build_cc_smooth(call, data))
        elif bs == "tp":
            out.append(_build_tp_smooth(call, data))
        elif bs == "ps":
            out.append(_build_ps_smooth(call, data))
        elif bs == "cp":
            out.append(_build_cp_smooth(call, data))
        elif bs == "bs":
            out.append(_build_bs_smooth(call, data))
        elif bs == "gp":
            out.append(_build_gp_smooth(call, data))
        elif bs == "fs":
            out.append(_build_fs_smooth(call, data))
        elif bs == "sz":
            out.append(_build_sz_smooth(call, data))
        elif bs == "ad":
            out.append(_build_ad_smooth(call, data))
        else:
            raise NotImplementedError(
                f"smooth bs={bs!r} (class dispatch for {_smooth_label(call)}) "
                "not yet implemented"
            )
    return out
