"""
Compare lmpy.formula.materialize() against R's X.csv fixture by fixture.

Per WR fixture: parse + expand + materialize, then assert shape and values
match R's stats::model.matrix output. Replaces the legacy
test_wr_fixtures.py, which compared formulaic against R.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import FIXTURE_ROOT, fixture_meta, fixtures_by_kind, load_dataset
from lmpy.formula import expand, materialize, parse

WR_FIXTURES = fixtures_by_kind("wr")
WR_IDS = [e["id"] for e in WR_FIXTURES]


def _load_X_ref(fx_id: str, n_rows: int) -> pd.DataFrame:
    try:
        return pd.read_csv(FIXTURE_ROOT / fx_id / "X.csv")
    except pd.errors.EmptyDataError:
        # R emitted a zero-column X (e.g. `y ~ 0`). Build an empty frame
        # with the right row count.
        return pd.DataFrame(index=range(n_rows))


@pytest.mark.parametrize("fx_id", WR_IDS)
def test_wr_materialize_matches_R(fx_id: str):
    meta, _ = fixture_meta(fx_id)
    pkg, name = meta["dataset"]["pkg"], meta["dataset"]["name"]
    data = load_dataset(pkg, name)

    X_ref = _load_X_ref(fx_id, len(data))

    f = parse(meta["formula"])
    data_cols = list(data.columns) if "." in meta["formula"] else None
    ef = expand(f, data_columns=data_cols)
    X_got = materialize(ef, data)

    assert X_got.shape == X_ref.shape, (
        f"shape: got {X_got.shape} want {X_ref.shape}  formula={meta['formula']!r}"
    )

    np.testing.assert_allclose(
        X_got.values.astype(float),
        X_ref.values.astype(float),
        rtol=1e-6,
        atol=1e-8,
        err_msg=(
            f"formula={meta['formula']!r}\n"
            f"  got cols: {list(X_got.columns)}\n"
            f"  ref cols: {list(X_ref.columns)}"
        ),
    )
