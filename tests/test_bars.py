"""
Compare hea.formula.materialize_bars() against R/lme4's Z, Lambdat, theta.

Per lme4 fixture: parse + expand + materialize_bars, then assert Z (dense),
Lambdat template (integer-indexed), and theta match R's output.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scipy.io import mmread

from conftest import FIXTURE_ROOT, fixture_meta, fixtures_by_kind, load_dataset
from hea.formula import expand, materialize_bars, parse

LME4_FIXTURES = fixtures_by_kind("lme4")
LME4_IDS = [e["id"] for e in LME4_FIXTURES]


def _load_theta(path) -> np.ndarray:
    return pl.read_csv(path)[:, 0].to_numpy().astype(float)


@pytest.mark.parametrize("fx_id", LME4_IDS)
def test_lme4_bars_matches_R(fx_id: str):
    meta, _ = fixture_meta(fx_id)
    fx = FIXTURE_ROOT / fx_id

    pkg, name = meta["dataset"]["pkg"], meta["dataset"]["name"]
    data = load_dataset(pkg, name)

    f = parse(meta["formula"])
    data_cols = list(data.columns) if "." in meta["formula"] else None
    ef = expand(f, data_columns=data_cols)
    got = materialize_bars(ef, data)

    Z_ref = np.asarray(mmread(fx / "Z.mtx").todense())
    Lt_ref = np.asarray(mmread(fx / "Lambdat.mtx").todense())
    theta_ref = _load_theta(fx / "theta.csv")

    assert got.Z.shape == Z_ref.shape, (
        f"Z shape: got {got.Z.shape} want {Z_ref.shape}"
    )
    assert got.Lambdat.shape == Lt_ref.shape, (
        f"Lambdat shape: got {got.Lambdat.shape} want {Lt_ref.shape}"
    )
    assert got.theta.shape == theta_ref.shape, (
        f"theta shape: got {got.theta.shape} want {theta_ref.shape}"
    )

    np.testing.assert_allclose(got.Z, Z_ref, rtol=1e-6, atol=1e-8,
                               err_msg=f"Z values diverge [{fx_id}]")
    assert np.array_equal(got.Lambdat.astype(int), Lt_ref.astype(int)), (
        f"Lambdat template (int indices) differs [{fx_id}]"
    )
    np.testing.assert_allclose(got.theta, theta_ref, rtol=1e-6, atol=1e-8,
                               err_msg=f"theta values diverge [{fx_id}]")
