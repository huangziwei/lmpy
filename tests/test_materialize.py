"""
Compare lmpy.formula.materialize() against R's X.csv fixture by fixture.

Per WR fixture: parse + expand + materialize, then assert shape and values
match R's stats::model.matrix output. Replaces the legacy
test_wr_fixtures.py, which compared formulaic against R.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from conftest import FIXTURE_ROOT, fixture_meta, fixtures_by_kind, load_dataset
from lmpy.formula import expand, materialize, parse, referenced_columns
from lmpy.design import prepare_design

WR_FIXTURES = fixtures_by_kind("wr")
WR_IDS = [e["id"] for e in WR_FIXTURES]


def _load_X_ref(fx_id: str, n_rows: int) -> pl.DataFrame:
    path = FIXTURE_ROOT / fx_id / "X.csv"
    head = path.read_text().splitlines()[:1]
    # Zero-column X (e.g. `y ~ 0`) lands as either an empty file or a
    # file of bare newlines with no header.
    if not head or not head[0].strip() or "," not in head[0] and '"' not in head[0]:
        return pl.DataFrame()
    # infer_schema_length=0 forces all columns to Float64 — X fixtures are
    # always numeric, and the default-100 inference picks i64 for columns
    # whose first rows happen to be integral and then chokes on later floats.
    return pl.read_csv(path, null_values="NA", infer_schema_length=0).cast(pl.Float64)


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

    # polars collapses a 0-column frame to (0, 0) — treat that as the
    # R-equivalent zero-column case for this assertion.
    got_shape = X_got.shape if X_got.width > 0 else (len(data), 0)
    ref_shape = X_ref.shape if X_ref.width > 0 else (len(data), 0)
    assert got_shape == ref_shape, (
        f"shape: got {got_shape} want {ref_shape}  formula={meta['formula']!r}"
    )

    if X_got.width > 0:
        np.testing.assert_allclose(
            X_got.to_numpy().astype(float),
            X_ref.to_numpy().astype(float),
            rtol=1e-6,
            atol=1e-8,
            err_msg=(
                f"formula={meta['formula']!r}\n"
                f"  got cols: {list(X_got.columns)}\n"
                f"  ref cols: {list(X_ref.columns)}"
            ),
        )


def test_referenced_columns_includes_smooth_vars():
    # NA-omit must see smooth-only variables; otherwise prepare_design and
    # materialize_smooths disagree on row count and gam() raises a concat
    # ValueError. Regression: pisa's `s(Income)` had 3 rows with non-NA
    # Overall but NA Income, breaking n=57 vs 54.
    ef = expand(parse("y ~ x + s(z, bs='cr') + te(u, v, by=g)"))
    assert referenced_columns(ef) >= {"x", "z", "u", "v", "g"}


def test_prepare_design_drops_na_on_smooth_only_var():
    df = pl.DataFrame({
        "y": [1.0, 2.0, 3.0, 4.0, 5.0],
        "z": [0.1, None, 0.3, 0.4, None],
    })
    d = prepare_design("y ~ s(z)", df)
    assert d.data.height == 3
    assert d.X.height == 3
    assert d.y.len() == 3
