"""User-facing data prep: dataset loader and ``factor()``.

* ``data`` — fetch a named dataset. Pulls from the bundled ``rdatasets``
  Python package when the ``(package, name)`` pair is covered there
  (``R``/``datasets``, ``MASS``, ``lme4``, ``nlme``); otherwise reads a
  CSV from this repo's ``datasets/`` tree (downloading on first access).
  In both cases, R's factor type is restored from a JSON schema sidecar
  next to the corresponding CSV path.
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


# Our only label rewrite for rdatasets: ``"R"`` (lmpy's name for R's
# built-in ``datasets`` package — mirrors our ``datasets/R/`` folder, which
# avoids the ``datasets/datasets/`` path duplication). Every other package
# label is passed straight through to rdatasets.
_RDATASETS_PKG_ALIAS = {"R": "datasets"}


def factor(
    series: pl.Series,
    levels=None,
    labels: dict | None = None,
    ordered: bool = False,
) -> pl.Series:
    """Polars equivalent of R's ``factor()`` — cast a Series to ``pl.Enum``.

    Use with ``df.with_columns(factor(df["col"]))``. The returned Series
    keeps its input name, so ``with_columns`` replaces the original column.

    Parameters
    ----------
    series : pl.Series
        Column to convert. Int64 inputs route through Utf8 (``pl.Enum``
        can't accept integers directly).
    levels : list | None, optional
        Level order, no relabel. If None, auto-detected via
        ``unique().sort()`` on the string-cast values — that's Unicode
        collation, which can diverge from R's locale-aware ``factor()``
        default for non-ASCII or punctuation-heavy levels. For poly
        contrasts on ordered factors, pass levels explicitly to control
        the order. Mutually exclusive with ``labels``.
    labels : dict | None, optional
        ``{level: label}`` mapping that combines R's ``factor(x, levels=,
        labels=)`` into one argument: keys are the expected raw values
        (insertion order = level order), values are the displayed labels.
        Errors if the column contains a value not in ``labels.keys()``
        (via ``replace_strict``). Use this for coded integer columns —
        e.g. ``factor(s, labels={0: "no", 1: "yes"})`` collapses cast +
        rename into one pass. Mutually exclusive with ``levels``.
    ordered : bool, optional
        If True, also register the series's name in lmpy's ordered-cols
        contextvar so subsequent ``gam``/``lm``/``lme`` calls in this
        session apply poly contrasts. Process-global; pair with
        ``lmpy.formula.with_ordered_cols`` if you need scoped use.
        ``ordered=False`` does NOT remove an already-registered name —
        call ``set_ordered_cols(frozenset())`` to clear.
    """
    if levels is not None and labels is not None:
        raise ValueError(
            "factor(): pass either `levels=` (list, reorder only) or "
            "`labels=` (dict {level: label}, reorder + rename), not both."
        )
    if isinstance(levels, dict):
        raise TypeError(
            "factor(): `levels=` expects a list/sequence, not a dict. "
            "For {level: label} mapping, pass it as `labels=` instead."
        )

    s = series.cast(pl.Utf8)

    if labels is not None:
        old = [str(k) for k in labels.keys()]
        new = [str(v) for v in labels.values()]
        out = s.replace_strict(old, new, return_dtype=pl.Enum(new))
    else:
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


def _find_schema(package: str, name: str) -> Path | None:
    """Walk up from CWD looking for ``datasets/{package}/{name}.schema.json``.

    Schema sidecars carry R factor info (levels + ordered flag) that CSV
    round-trip and ``rdatasets`` both erase. They are kept locally even when
    the data itself is sourced from ``rdatasets``.
    """
    rel = Path("datasets") / package / f"{name}.schema.json"
    cwd = Path.cwd()
    for ancestor in (cwd, *cwd.parents):
        candidate = ancestor / rel
        if candidate.is_file():
            return candidate
    return None


def _normalize_rownames(df: pl.DataFrame) -> pl.DataFrame:
    """Standardize the row-id column rdatasets injects.

    rdatasets always adds a ``rownames`` column; we rename it to
    ``rowname`` (singular — tibble convention, matches what
    ``export_data.R`` writes for bundled CSVs) and drop it entirely when
    the values are just sequential ``1..n``, which carries no information.
    """
    if "rownames" not in df.columns:
        return df
    rn = df["rownames"]
    if rn.dtype.is_integer() and rn.to_list() == list(range(1, df.height + 1)):
        return df.drop("rownames")
    return df.rename({"rownames": "rowname"})


def _try_load_rdatasets(package: str, name: str) -> pl.DataFrame | None:
    """Load ``(package, name)`` from the ``rdatasets`` package, or None if missing.

    Tries ``package`` (after the ``R`` → ``datasets`` alias) against the
    rdatasets package list, then against its item list. Returns None if
    either lookup fails — caller falls back to bundled CSV / download.
    The injected ``rownames`` column is normalized via
    ``_normalize_rownames``.
    """
    try:
        import rdatasets
    except ImportError:
        return None
    rd_pkg = _RDATASETS_PKG_ALIAS.get(package, package)
    if rd_pkg not in rdatasets.packages():
        return None
    items = {it.removesuffix(".pkl") for it in rdatasets.items(rd_pkg)}
    if name not in items:
        return None
    df = pl.from_pandas(rdatasets.data(rd_pkg, name))
    return _normalize_rownames(df)


# Accumulator for ordered-factor columns across data() calls within a session,
# mirroring tests/conftest.py. Polars has no per-column "ordered" flag, so
# ordered factors are tracked via a contextvar that lmpy.formula consults when
# building contrasts. This set lets multiple data() calls coexist without one
# clobbering another's ordered registrations.
_data_ordered_cols: set[str] = set()


def _apply_dataset_schema(df: pl.DataFrame, schema_path: Path | None) -> pl.DataFrame:
    """Apply the JSON schema sidecar at ``schema_path``, if present.

    Cast factor columns to ``pl.Enum`` and register ordered factors with the
    formula machinery. Without this, R's factor type erased by CSV round-trip
    (or stripped by ``rdatasets``) silently degrades ``s(...,bs='re')``,
    ``by=factor``, ``fs``, ``sz``, and ordered-contrast paths. Sidecar format
    mirrors tests/conftest.py: ``{"factors": {col: {"levels": [...], "ordered": bool}}}``.
    """
    if schema_path is None or not schema_path.is_file():
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
    """Load a named dataset.

    Resolution order:

    1. ``rdatasets`` — tried for any package it carries (``MASS``,
       ``lme4``, ``nlme``, ``HistData``, ``ggplot2``, ``palmerpenguins``,
       …75 packages — see ``rdatasets.packages()``). The label ``"R"`` is
       aliased to rdatasets's ``"datasets"`` (R's built-in data). Offline,
       deterministic, ships with the package. The ``rownames`` column
       rdatasets injects is dropped.
    2. Bundled ``datasets/{package}/{name}.csv`` walked up from CWD —
       used for ``faraway``/``gamair``/``mgcv``/``rstanarm``/``synthetic``
       (not in rdatasets) and the few items rdatasets doesn't carry
       (e.g. ``lme4::ergoStool``).
    3. CSV download into ``save_to/{package}/{name}.csv`` — last resort,
       used when ``lmpy`` is installed outside the source repo and no
       bundled CSV exists. Pass ``overwrite=True`` to force a re-fetch.

    A JSON schema sidecar (``datasets/{package}/{name}.schema.json``) is
    loaded next and used to restore R's factor type — columns listed
    under ``factors`` are cast to ``pl.Enum``, and ones with
    ``ordered: true`` are registered for poly contrasts. The sidecar is
    looked up via the same CWD-walk as the bundled CSV, so it applies
    even when the data itself came from rdatasets (which strips factor
    info on the way out of pandas).
    """
    df: pl.DataFrame | None = None

    if not overwrite:
        df = _try_load_rdatasets(package, name)
        if df is None:
            bundled = _find_bundled_dataset(package, name)
            if bundled is not None:
                df = pl.read_csv(bundled, null_values="NA")

    if df is None:
        datapath = os.path.join(save_to, package)
        csv_path = Path(datapath) / f"{name}.csv"
        if not csv_path.exists() or overwrite:
            # Snapshot which dirs don't exist yet so a failed download can
            # roll them back — otherwise a network error leaves an empty
            # data/<package>/ (and possibly data/) behind in the CWD.
            created_dirs = [Path(p) for p in (save_to, datapath) if not os.path.exists(p)]
            os.makedirs(datapath, exist_ok=True)
            print(f"Downloading {name} (from {package})...")
            base = f"https://raw.githubusercontent.com/huangziwei/lmpy/main/datasets/{package}/{name}"
            try:
                urllib.request.urlretrieve(f"{base}.csv", csv_path)
            except Exception:
                csv_path.unlink(missing_ok=True)
                for p in reversed(created_dirs):
                    try:
                        p.rmdir()
                    except OSError:
                        pass
                raise
            try:
                urllib.request.urlretrieve(
                    f"{base}.schema.json", csv_path.with_suffix(".schema.json")
                )
            except urllib.error.HTTPError:
                pass
        df = pl.read_csv(csv_path, null_values="NA")

    return _apply_dataset_schema(df, _find_schema(package, name))
