"""Cross-model helpers shared by lmpy.lm, lmpy.lme, and (later) lmpy.gam.

Includes:

* ``prepare_design`` — common formula intake (parse → expand → materialize
  + response NA-omit), returning a ``Design`` bundle that downstream
  models specialize as they see fit.
* ``data`` — fetches a CSV from this repo's ``datasets/`` tree
  (downloading on first access).
* ``AIC`` — model-comparison table that prints AIC + parameter count for
  each model, keyed by formula.
* ``significance_code`` — R-style ``***``/``**``/``*``/``.`` formatter
  for p-values.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass

import pandas as pd

from .formula import ExpandedFormula, Name, expand, materialize, parse

__all__ = ["Design", "prepare_design", "data", "AIC", "significance_code"]


@dataclass(slots=True)
class Design:
    """Bundle returned by ``prepare_design``.

    Attributes
    ----------
    expanded : ExpandedFormula
        Output of ``formula.expand`` for the parsed formula. Pass this
        to downstream materializers (``materialize_bars`` for lme,
        ``materialize_smooths`` for gam) so they share the same parse.
    data : pandas.DataFrame
        Input data with rows dropped where the response is NA. Pass
        this (not the original ``data``) to downstream materializers so
        any further row-drops on RHS NAs stay aligned with ``X``.
    X : pandas.DataFrame
        Materialized fixed-effect design with R-canonical column names.
        Its ``index`` may be a strict subset of ``data.index`` if any
        RHS column also had NAs.
    y : pandas.Series
        Response, indexed by ``X.index``.
    response : str
        Bare name of the response column (LHS of the formula).
    """
    expanded: ExpandedFormula
    data: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    response: str


def prepare_design(formula: str, data: pd.DataFrame) -> Design:
    """Parse a formula, expand, and materialize the fixed-effect design.

    NA-omit policy matches R's ``na.action = na.omit``: rows with NA in
    the response or in any RHS-referenced column are dropped before the
    design matrix is built. ``Design.data`` is the response-cleaned
    frame; ``Design.X.index`` reflects additional drops materialize()
    does for RHS NAs.
    """
    f_parsed = parse(formula)
    if not isinstance(f_parsed.lhs, Name):
        raise NotImplementedError(
            f"only single-name response (y ~ ...) is supported; got LHS={f_parsed.lhs!r}"
        )
    response = f_parsed.lhs.ident
    expanded = expand(f_parsed, data_columns=list(data.columns))
    data_clean = data.dropna(subset=[response])
    X = materialize(expanded, data_clean)
    y = data_clean.loc[X.index, response]
    return Design(expanded=expanded, data=data_clean, X=X, y=y, response=response)


def data(name: str, package: str = "R", save_to: str = "./data",
         overwrite: bool = False) -> pd.DataFrame:
    """Load a named dataset from this repo's published ``datasets/`` tree.

    Caches the CSV under ``save_to/{package}/{name}.csv`` on first
    access; pass ``overwrite=True`` to re-download.
    """
    import polars as pl

    datapath = os.path.join(save_to, package)
    os.makedirs(datapath, exist_ok=True)
    csv_path = os.path.join(datapath, f"{name}.csv")
    if not os.path.exists(csv_path) or overwrite:
        print(f"Downloading {name} (from {package})...")
        url = f"https://raw.githubusercontent.com/huangziwei/lmpy/main/datasets/{package}/{name}.csv"
        urllib.request.urlretrieve(url, csv_path)
    return pl.read_csv(csv_path, null_values="NA").to_pandas()


def AIC(*models) -> None:
    """Print an AIC comparison table for one or more fitted models.

    Each model must expose ``.AIC``, ``.npar``, and ``.formula``.
    """
    rows = pd.DataFrame.from_dict({
        "formula": [m.formula for m in models],
        "df":      [m.npar    for m in models],
        "AIC":     [m.AIC     for m in models],
    }).set_index("formula")
    print(rows.to_string(formatters={"AIC": "{:.2f}".format}))


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
