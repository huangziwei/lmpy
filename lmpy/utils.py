"""Cross-model helpers shared by lmpy.lm, lmpy.lme, and (later) lmpy.gam.

Includes:

* ``prepare_design`` — common formula intake (parse → expand → materialize
  + response NA-omit), returning a ``Design`` bundle that downstream
  models specialize as they see fit.
* ``data`` — fetches a CSV from this repo's ``datasets/`` tree
  (downloading on first access).
* ``significance_code`` — R-style ``***``/``**``/``*``/``.`` formatter
  for p-values.

Model-comparison helpers (``anova``, ``AIC``, ``BIC``) live in
``lmpy.compare``.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from .formula import (
    ExpandedFormula,
    Name,
    expand,
    materialize,
    parse,
    referenced_columns,
    set_ordered_cols,
)

__all__ = ["Design", "prepare_design", "data", "significance_code", "format_df"]


_MAX_DECIMALS = 6


def _format_numeric_column(vals: list) -> list[str]:
    """Format a numeric column with a single column-wide decimal count.

    Each value is converted to the shortest round-trip Python ``repr``; the
    column-wide decimal count is the max observed there (capped at 6, matching
    pandas' default ``display.precision``), and every value is then padded to
    that count. Scientific-repr values (e.g. ``6.8e-07``) stay in scientific
    form so they don't balloon to long fixed strings.
    """
    out: list[str] = []
    reprs: list[tuple[str, float] | None] = []
    max_dec = 0
    for v in vals:
        if v is None:
            reprs.append(None)
            continue
        fv = float(v)
        if not np.isfinite(fv):
            reprs.append(("nan", fv))
            continue
        r = repr(fv)
        if "e" in r or "E" in r:
            reprs.append(("sci", fv))
            continue
        if "." in r:
            max_dec = max(max_dec, len(r.split(".")[1]))
        reprs.append(("fix", fv))
    max_dec = min(max_dec, _MAX_DECIMALS)
    for item in reprs:
        if item is None:
            out.append("")
        elif item[0] == "nan":
            out.append("NaN")
        elif item[0] == "sci":
            out.append(f"{item[1]:.{_MAX_DECIMALS}g}")
        else:
            out.append(f"{item[1]:.{max_dec}f}")
    return out


def format_df(df: pl.DataFrame, align: dict[str, str] | None = None) -> str:
    """Render a polars DataFrame pandas-style for human-readable output.

    - Numeric columns right-aligned with column-wide fixed decimals.
    - String columns left-aligned.
    - A column named ``""`` prints its header row blank (treated as a row-label column).
    - ``align`` overrides the per-column alignment (``"left"``/``"right"``).
    """
    align = align or {}
    headers = list(df.columns)
    n_rows = df.shape[0]

    formatted: list[list[str]] = []
    aligns: list[str] = []
    for c in headers:
        s = df[c]
        if s.dtype.is_integer():
            cells = [
                "" if v is None else str(int(v))
                for v in s.to_list()
            ]
            default_align = "right"
        elif s.dtype.is_numeric():
            cells = _format_numeric_column(s.to_list())
            default_align = "right"
        else:
            cells = ["" if v is None else str(v) for v in s.to_list()]
            default_align = "left"
        formatted.append(cells)
        aligns.append(align.get(c, default_align))

    widths = [
        max([len(c)] + [len(x) for x in formatted[j]])
        for j, c in enumerate(headers)
    ]

    def pad(s: str, w: int, a: str) -> str:
        return s.ljust(w) if a == "left" else s.rjust(w)

    sep = "  "
    header_cells = [
        (" " * w if c == "" else pad(c, w, a))
        for c, w, a in zip(headers, widths, aligns)
    ]
    lines = [sep.join(header_cells).rstrip()]
    for i in range(n_rows):
        row = sep.join(pad(formatted[j][i], widths[j], aligns[j]) for j in range(len(headers)))
        lines.append(row.rstrip())
    return "\n".join(lines)


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
        Response column, with NA rows dropped.
    response : str
        Bare name of the response column (LHS of the formula).
    """
    expanded: ExpandedFormula
    data: pl.DataFrame
    X: pl.DataFrame
    y: pl.Series
    response: str


def prepare_design(formula: str, data: pl.DataFrame) -> Design:
    """Parse a formula, expand, and materialize the fixed-effect design.

    NA-omit policy matches R's ``na.action = na.omit``: rows with NA in
    the response or in any RHS-referenced column are dropped before the
    design matrix is built. All three outputs (``Design.data``,
    ``Design.X``, ``Design.y``) share the same row ordering.
    """
    f_parsed = parse(formula)
    if not isinstance(f_parsed.lhs, Name):
        raise NotImplementedError(
            f"only single-name response (y ~ ...) is supported; got LHS={f_parsed.lhs!r}"
        )
    response = f_parsed.lhs.ident
    expanded = expand(f_parsed, data_columns=list(data.columns))

    na_cols = (referenced_columns(expanded) | {response}) & set(data.columns)
    if na_cols:
        data_clean = data.drop_nulls(subset=list(na_cols))
    else:
        data_clean = data

    X = materialize(expanded, data_clean)
    y = data_clean[response]
    return Design(expanded=expanded, data=data_clean, X=X, y=y, response=response)


def _find_bundled_dataset(package: str, name: str) -> Path | None:
    """Walk up from CWD looking for a bundled ``datasets/{package}/{name}.csv``.

    Returns the first match in CWD or any ancestor, or ``None`` if no
    bundled copy exists anywhere up the tree (e.g. when ``lmpy`` is
    installed as a package and the caller is outside the source repo).
    """
    rel = Path("datasets") / package / f"{name}.csv"
    cwd = Path.cwd()
    for ancestor in (cwd, *cwd.parents):
        candidate = ancestor / rel
        if candidate.is_file():
            return candidate
    return None


# Accumulator for ordered-factor columns across data() calls within a session,
# mirroring tests/conftest.py. Polars has no per-column "ordered" flag, so
# ordered factors are tracked via a contextvar that lmpy.formula consults when
# building contrasts. This set lets multiple data() calls coexist without one
# clobbering another's ordered registrations.
_data_ordered_cols: set[str] = set()


def _apply_dataset_schema(df: pl.DataFrame, csv_path: Path) -> pl.DataFrame:
    """Apply the JSON schema sidecar next to ``csv_path``, if present.

    Cast factor columns to ``pl.Enum`` and register ordered factors with the
    formula machinery. Without this, R's factor type erased by CSV round-trip
    silently degrades ``s(...,bs='re')``, ``by=factor``, ``fs``, ``sz``, and
    ordered-contrast paths. Sidecar format mirrors tests/conftest.py:
    ``{"factors": {col: {"levels": [...], "ordered": bool}}}``.
    """
    schema_path = csv_path.with_suffix(".schema.json")
    if not schema_path.is_file():
        return df
    try:
        sch = json.loads(schema_path.read_text())
    except (OSError, json.JSONDecodeError):
        return df
    factors = sch.get("factors") or {}
    if not factors:
        return df
    exprs = []
    new_ordered = set()
    for col, spec in factors.items():
        if col not in df.columns:
            continue
        levels = [str(v) for v in spec.get("levels", [])]
        if not levels:
            continue
        exprs.append(pl.col(col).cast(pl.Utf8).cast(pl.Enum(levels)))
        if spec.get("ordered"):
            new_ordered.add(col)
    if exprs:
        df = df.with_columns(exprs)
    if new_ordered:
        _data_ordered_cols.update(new_ordered)
        set_ordered_cols(frozenset(_data_ordered_cols))
    return df


def data(name: str, package: str = "R", save_to: str = "./data",
         overwrite: bool = False) -> pl.DataFrame:
    """Load a named dataset from this repo's published ``datasets/`` tree.

    Looks first for a bundled ``datasets/{package}/{name}.csv`` by
    walking up from the current working directory — so in-repo callers
    (notebooks under ``example/``, dev scripts) read the checked-in copy
    directly and never download. If no bundled copy is found, caches the
    CSV under ``save_to/{package}/{name}.csv`` on first access. Pass
    ``overwrite=True`` to bypass both lookups and re-download.

    A JSON schema sidecar (``{name}.schema.json``) is loaded alongside the CSV
    and used to restore R's factor type — columns listed under ``factors``
    are cast to ``pl.Enum`` with their declared levels, and ones marked
    ``ordered: true`` are registered for poly contrasts. This is essential
    for ``bs='re'`` / ``by=factor`` / ``fs`` / ``sz`` smooths, which silently
    take a non-factor fallthrough path when factor columns come back as
    Int64/Utf8 from raw CSV.
    """
    if not overwrite:
        bundled = _find_bundled_dataset(package, name)
        if bundled is not None:
            df = pl.read_csv(bundled, null_values="NA")
            return _apply_dataset_schema(df, bundled)

    datapath = os.path.join(save_to, package)
    os.makedirs(datapath, exist_ok=True)
    csv_path = Path(datapath) / f"{name}.csv"
    if not csv_path.exists() or overwrite:
        print(f"Downloading {name} (from {package})...")
        base = f"https://raw.githubusercontent.com/huangziwei/lmpy/main/datasets/{package}/{name}"
        urllib.request.urlretrieve(f"{base}.csv", csv_path)
        # Best-effort: pull the schema sidecar too. Not all bundled datasets
        # ship one, so a 404 is silently OK — _apply_dataset_schema then
        # no-ops on the missing file.
        try:
            urllib.request.urlretrieve(
                f"{base}.schema.json", csv_path.with_suffix(".schema.json")
            )
        except urllib.error.HTTPError:
            pass
    df = pl.read_csv(csv_path, null_values="NA")
    return _apply_dataset_schema(df, csv_path)


def significance_code(p_values) -> list[str]:
    """R-style significance codes for an iterable of p-values."""
    out: list[str] = []
    for p in p_values:
        if p < 0.001:
            out.append("***")
        elif p < 0.01:
            out.append("**")
        elif p < 0.05:
            out.append("*")
        elif p < 0.1:
            out.append(".")
        else:
            out.append(" ")
    return out
