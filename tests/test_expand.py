"""
Compare lmpy.formula.expand() against R's X_meta.json ground truth.

Per WR fixture: parse + expand, then assert `intercept` and `term_labels`
match R exactly. One test per fixture.
"""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import fixture_meta, fixtures_by_kind, load_dataset
from lmpy.formula import expand, parse

WR_FIXTURES = fixtures_by_kind("wr")
WR_IDS = [e["id"] for e in WR_FIXTURES]


def _normalize_labels(tl) -> list[str]:
    if tl is None:
        return []
    if isinstance(tl, str):
        return [tl]
    return list(tl)


@pytest.mark.parametrize("fx_id", WR_IDS)
def test_wr_expand_matches_R(fx_id: str):
    meta, xmeta = fixture_meta(fx_id)
    formula_src = meta["formula"]
    want_intercept = xmeta.get("intercept", True)
    want_labels = _normalize_labels(xmeta.get("term_labels"))

    f = parse(formula_src)
    data_cols = None
    if "." in formula_src:
        ds = meta["dataset"]
        data_cols = list(load_dataset(ds["pkg"], ds["name"]).columns)
    ef = expand(f, data_columns=data_cols)

    assert ef.intercept == want_intercept, (
        f"intercept: got {ef.intercept}, want {want_intercept}"
    )
    assert ef.term_labels == want_labels, (
        f"term_labels:\n  got:  {ef.term_labels}\n  want: {want_labels}"
    )
