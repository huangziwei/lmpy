"""
Compare formulaic's model_matrix against R's stats::model.matrix, fixture by fixture.

Each fixture pins a formula + dataset + R's ground-truth X.csv + X_meta.json.
We parametrize over every 'ok' WR fixture and assert shape + per-column values
match. Column names differ stylistically between formulaic and R — we compare
by position after a shape check, and record the name mapping for diagnosis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from formulaic import model_matrix

from conftest import fixture_X_ref, fixture_meta, fixtures_by_kind, load_dataset

WR_FIXTURES = fixtures_by_kind("wr")
WR_IDS = [e["id"] for e in WR_FIXTURES]


def _build_X(formula: str, data: pd.DataFrame) -> pd.DataFrame:
    """Run formulaic and extract the RHS design matrix as a DataFrame."""
    result = model_matrix(formula, data)
    # Two-sided formula → ModelMatrices(lhs=..., rhs=...); RHS-only → ModelMatrix directly.
    if hasattr(result, "rhs"):
        return result.rhs
    return result


@pytest.mark.parametrize("fx_id", WR_IDS)
def test_wr_X_matches_R(fx_id: str):
    meta, xmeta = fixture_meta(fx_id)
    X_ref = fixture_X_ref(fx_id)

    pkg, name = meta["dataset"]["pkg"], meta["dataset"]["name"]
    data = load_dataset(pkg, name)

    X_py = _build_X(meta["formula"], data)

    # Shape must match before anything else makes sense
    assert X_py.shape == X_ref.shape, (
        f"shape: py={X_py.shape} r={X_ref.shape}  "
        f"formula={meta['formula']}"
    )

    # Compare values column-by-column (order fixed by position).
    # We don't enforce column-name equality yet — formulaic and R use different
    # stylistic conventions (Intercept vs (Intercept), f[T.b] vs fb).
    py_vals = X_py.values.astype(float)
    r_vals = X_ref.values.astype(float)
    np.testing.assert_allclose(
        py_vals, r_vals, rtol=1e-8, atol=1e-10,
        err_msg=(
            f"X values diverge. formula={meta['formula']}\n"
            f"  py cols: {list(X_py.columns)}\n"
            f"  r  cols: {list(X_ref.columns)}"
        ),
    )
