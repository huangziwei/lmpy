"""Cross-model formatting helpers and dataset loader.

Includes:

* ``data`` — fetches a CSV from this repo's ``datasets/`` tree
  (downloading on first access).
* ``significance_code`` — R-style ``***``/``**``/``*``/``.`` formatter
  for p-values.
* ``format_df`` / ``format_signif`` / ``format_pval`` — pandas-style
  printers used by every model's ``summary()``.

The formula → fitting-ready design pipeline lives in ``lmpy.design``;
model-comparison helpers (``anova``, ``AIC``, ``BIC``) in
``lmpy.compare``.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import polars as pl

from .formula import set_ordered_cols

__all__ = ["data",
           "significance_code", "format_df",
           "format_signif", "format_signif_jointly", "format_pval"]


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


_MACHINE_EPS = float(np.finfo(float).eps)  # ≈ 2.22e-16, matches R's `.Machine$double.eps`

# Magnitudes |x| < 10^_FIXED_LOWER print as scientific; otherwise fixed.
# Matches R's ``format()`` rule of switching to scientific when ``expo < -4``.
_FIXED_LOWER = -4


def _is_fixed(fv: float) -> bool:
    """Whether ``fv`` formats as fixed (vs. scientific) in R's ``format()``."""
    if fv == 0 or not np.isfinite(fv):
        return True
    return int(np.floor(np.log10(abs(fv)))) >= _FIXED_LOWER


def _signif_column_decimals(values, digits: int) -> int:
    """Joint decimal count over entries that print as fixed: chosen so the
    smallest-magnitude fixed entry gets ``digits`` significant figures
    (R's ``format(x, digits=)`` per-vector decimals).
    """
    fixed_nonzero = []
    for v in values:
        if v is None:
            continue
        fv = float(v)
        if np.isfinite(fv) and fv != 0 and _is_fixed(fv):
            fixed_nonzero.append(abs(fv))
    if not fixed_nonzero:
        return 0
    smallest = min(fixed_nonzero)
    return max(0, digits - 1 - int(np.floor(np.log10(smallest))))


def _format_with_decimals(values, decimals: int, digits: int) -> list[str]:
    """Per-element formatter: fixed at ``decimals`` for in-range entries,
    scientific (``digits-1`` after the leading digit) for entries below
    the fixed threshold or that would round to all-zeros at this decimal
    count.
    """
    sci_fmt = f".{max(digits - 1, 0)}e"
    out: list[str] = []
    for v in values:
        if v is None:
            out.append("")
            continue
        fv = float(v)
        if np.isnan(fv):
            out.append("NaN")
            continue
        if not np.isfinite(fv):
            out.append("Inf" if fv > 0 else "-Inf")
            continue
        rounds_to_zero = fv != 0 and abs(fv) < 0.5 * 10 ** (-decimals)
        if not _is_fixed(fv) or rounds_to_zero:
            out.append(format(fv, sci_fmt))
        else:
            out.append(f"{fv:.{decimals}f}")
    return out


def format_signif(values, digits: int = 4, *, min_decimals: int = 0) -> list[str]:
    """R-style ``format(x, digits=digits)`` for a numeric vector.

    Picks a single column-wide decimal count so the smallest-magnitude
    fixed-form entry gets ``digits`` significant figures, then applies
    it uniformly. Entries with ``|x| < 1e-4`` (or that would otherwise
    round to ``0.000…``) fall back to scientific independently — matches
    R's ``format()`` per-element fixed-vs-scientific switch.
    ``None`` → ``""``; non-finite → ``"NaN"``/``"Inf"``/``"-Inf"``.
    """
    decimals = max(min_decimals,
                   min(_signif_column_decimals(values, digits), 15))
    return _format_with_decimals(values, decimals, digits)


def format_signif_jointly(columns, digits: int = 4) -> list[list[str]]:
    """R's ``printCoefmat`` joint formatting for the coefficient + SE
    group: pool magnitudes across all input columns, pick one shared
    decimal count so the smallest pooled magnitude gets ``digits`` sig
    figs (with at least 1 decimal, matching ``max(1L, digits - digmin)``
    in R), and format each input column at that count.

    Returns a list of formatted-string columns, parallel to ``columns``.
    Use for Estimate / SE / CI together so a row like ``470.4444 4.0817``
    prints as ``470.444 4.082`` (decimals driven by the smaller value).
    """
    pool: list[float] = []
    for col in columns:
        for v in col:
            if v is None:
                continue
            fv = float(v)
            if np.isfinite(fv) and fv != 0 and _is_fixed(fv):
                pool.append(abs(fv))
    if not pool:
        decimals = 1
    else:
        decimals = max(1, digits - 1 - int(np.floor(np.log10(min(pool)))))
    decimals = min(decimals, 15)
    return [_format_with_decimals(col, decimals, digits) for col in columns]


def format_pval(values, digits: int = 4, eps: float | None = None) -> list[str]:
    """R-style ``format.pval`` for an iterable of p-values.

    Mirrors R's ``stats::format.pval(x, digits=digits, eps=eps)``:

    * Entries ``< eps`` (default ``.Machine$double.eps``) print as
      ``"<{eps_str}"``, where ``eps_str`` uses ``max(1, digits-2)`` sig
      figs (R further-reduces it if it would crowd the column, and picks
      a leading space iff the reduced digits ≠ 1 or the column is wide).
    * Other entries format with ``digits`` sig figs via ``format_signif``
      (per-element fixed-vs-scientific based on magnitude, joint decimals
      across fixed entries).

    Note: R's ``printCoefmat`` calls ``format.pval`` with a *reduced*
    ``digits = max(1, min(5, digits-1))`` (its ``dig.tst``); summary
    methods that want printCoefmat-faithful output should pass that
    reduced value here, not the raw user-facing ``digits``.
    """
    if eps is None:
        eps = _MACHINE_EPS
    big = [
        float(v) for v in values
        if v is not None and np.isfinite(float(v)) and float(v) >= eps
    ]
    big_strs = format_signif(big, digits=digits) if big else []

    eps_digits = max(1, digits - 2)
    if big_strs:
        nc = max(len(s) for s in big_strs)
        if eps_digits > 1 and eps_digits + 6 > nc:
            eps_digits = max(1, nc - 7)
        sep = "" if (eps_digits == 1 and nc <= 6) else " "
    else:
        sep = "" if eps_digits == 1 else " "
    eps_str = f"{eps:.{eps_digits}g}"
    eps_display = f"<{sep}{eps_str}"

    big_iter = iter(big_strs)
    out: list[str] = []
    for v in values:
        if v is None:
            out.append("")
            continue
        fv = float(v)
        if np.isnan(fv):
            out.append("NaN")
            continue
        if fv < eps:
            out.append(eps_display)
        else:
            out.append(next(big_iter))
    return out


def _dig_tst(digits: int) -> int:
    """R's ``printCoefmat`` reduction of ``digits`` for test-statistic /
    p-value columns: ``max(1, min(5, digits - 1))``.
    """
    return max(1, min(5, digits - 1))
