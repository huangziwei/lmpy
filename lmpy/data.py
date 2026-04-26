"""User-facing data prep: dataset loader and ``factor()``.

* ``data`` — fetch a CSV from this repo's ``datasets/`` tree (downloading
  on first access), restoring R's factor type from a JSON schema sidecar
  when one is present.
* ``factor`` — polars equivalent of R's ``factor()``: cast a Series to
  ``pl.Enum`` and (optionally) register it as an ordered factor for poly
  contrasts.

Both run *before* a model is fit; the formula → design pipeline lives in
``lmpy.design`` and consumes whatever data() / factor() produce.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import polars as pl

from .formula import set_ordered_cols

__all__ = ["data", "factor"]


def factor(series: pl.Series, levels=None, ordered: bool = False) -> pl.Series:
    """Polars equivalent of R's ``factor()`` — cast a Series to ``pl.Enum``.

    Use with ``df.with_columns(factor(df["col"]))``. The returned Series
    keeps its input name, so ``with_columns`` replaces the original column.

    Parameters
    ----------
    series : pl.Series
        Column to convert. Int64 inputs route through Utf8 (``pl.Enum``
        can't accept integers directly).
    levels : list | None, optional
        Level order. If None, auto-detected via ``unique().sort()`` on the
        string-cast values — that's Unicode collation, which can diverge
        from R's locale-aware ``factor()`` default for non-ASCII or
        punctuation-heavy levels. For poly contrasts on ordered factors,
        pass levels explicitly to control the order.
    ordered : bool, optional
        If True, also register the series's name in lmpy's ordered-cols
        contextvar so subsequent ``gam``/``lm``/``lme`` calls in this
        session apply poly contrasts. Process-global; pair with
        ``lmpy.formula.with_ordered_cols`` if you need scoped use.
        ``ordered=False`` does NOT remove an already-registered name —
        call ``set_ordered_cols(frozenset())`` to clear.
    """
    s = series.cast(pl.Utf8)
    if levels is None:
        levels_list = s.drop_nulls().unique().sort().to_list()
    else:
        levels_list = [str(v) for v in levels]
    out = s.cast(pl.Enum(levels_list))
    if ordered and series.name:
        from .formula import _ORDERED_COLS_CV
        set_ordered_cols(_ORDERED_COLS_CV.get() | frozenset({series.name}))
    return out


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
