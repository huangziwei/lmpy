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

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from .formula import (
    ExpandedFormula,
    Name,
    expand,
    materialize,
    parse,
    referenced_columns,
)

__all__ = ["Design", "prepare_design", "data", "significance_code"]


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


def data(name: str, package: str = "R", save_to: str = "./data",
         overwrite: bool = False) -> pl.DataFrame:
    """Load a named dataset from this repo's published ``datasets/`` tree.

    Looks first for a bundled ``datasets/{package}/{name}.csv`` by
    walking up from the current working directory — so in-repo callers
    (notebooks under ``example/``, dev scripts) read the checked-in copy
    directly and never download. If no bundled copy is found, caches the
    CSV under ``save_to/{package}/{name}.csv`` on first access. Pass
    ``overwrite=True`` to bypass both lookups and re-download.
    """
    if not overwrite:
        bundled = _find_bundled_dataset(package, name)
        if bundled is not None:
            return pl.read_csv(bundled, null_values="NA")

    datapath = os.path.join(save_to, package)
    os.makedirs(datapath, exist_ok=True)
    csv_path = os.path.join(datapath, f"{name}.csv")
    if not os.path.exists(csv_path) or overwrite:
        print(f"Downloading {name} (from {package})...")
        url = f"https://raw.githubusercontent.com/huangziwei/lmpy/main/datasets/{package}/{name}.csv"
        urllib.request.urlretrieve(url, csv_path)
    return pl.read_csv(csv_path, null_values="NA")


def AIC(*models) -> None:
    """Print an AIC comparison table for one or more fitted models.

    Each model must expose ``.AIC``, ``.npar``, and ``.formula``.
    """
    rows = pl.DataFrame({
        "formula": [m.formula for m in models],
        "df":      [m.npar    for m in models],
        "AIC":     [m.AIC     for m in models],
    })
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=2):
        print(rows)


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
