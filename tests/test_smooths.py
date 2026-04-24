"""
Compare lmpy.formula.materialize_smooths() against R/mgcv per-smooth fixtures.

Per mgcv fixture: parse + expand + materialize_smooths, then for each
smooth/block assert X and S (penalty) match R. Fixtures where R itself
reported `smoothCon failed` on any smooth are skipped (no ground truth).

Sign conventions for basis columns are arbitrary between np.linalg.eigh
and mgcv's Rlanczos, so we match each column up to sign and apply the
same flip to S. Tolerances are normalized against max|ref| because S
(and X, for high-k tp) spans several orders of magnitude.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.io import mmread

from conftest import FIXTURE_ROOT, fixture_meta, fixtures_by_kind, load_dataset
from lmpy.formula import expand, materialize_smooths, parse

MGCV_FIXTURES = fixtures_by_kind("mgcv")


def _has_error(fx_id: str) -> bool:
    sm_meta = json.loads((FIXTURE_ROOT / fx_id / "smooth_meta.json").read_text())
    return any("error" in s for s in sm_meta.get("smooths", []))


MGCV_OK = [e["id"] for e in MGCV_FIXTURES if not _has_error(e["id"])]


@pytest.mark.parametrize("fx_id", MGCV_OK)
def test_mgcv_smooths_match_R(fx_id: str):
    meta, _ = fixture_meta(fx_id)
    fx = FIXTURE_ROOT / fx_id
    sm_meta = json.loads((fx / "smooth_meta.json").read_text())

    pkg, name = meta["dataset"]["pkg"], meta["dataset"]["name"]
    data = load_dataset(pkg, name)

    f = parse(meta["formula"])
    data_cols = list(data.columns) if "." in meta["formula"] else None
    ef = expand(f, data_columns=data_cols)

    # R's gam drops rows with NA in ANY formula variable — match that.
    need = set(meta.get("need_vars", [])) & set(data.columns)
    if need:
        data = data.drop_nulls(subset=list(need))

    ours = materialize_smooths(ef, data)

    assert len(ours) == len(sm_meta["smooths"]), (
        f"n smooths: got {len(ours)} want {len(sm_meta['smooths'])}"
    )

    for i, (ours_blocks, r_meta) in enumerate(zip(ours, sm_meta["smooths"]), start=1):
        assert len(ours_blocks) == r_meta["n_blocks"], (
            f"smooth #{i}: got {len(ours_blocks)} blocks want {r_meta['n_blocks']}"
        )

        for k, blk in enumerate(ours_blocks, start=1):
            X_ref = np.asarray(
                mmread(fx / f"smooth_{i}_{k}_X.mtx").todense(), dtype=float
            )
            assert blk.X.shape == X_ref.shape, (
                f"smooth #{i} block {k}: X shape got {blk.X.shape} want {X_ref.shape}"
            )

            # mgcv's Lanczos uses an arbitrary per-eigenvector sign convention;
            # lmpy's np.linalg.eigh uses its own. Match each column up to sign,
            # then apply the same flip to S.
            signs = np.ones(blk.X.shape[1])
            X_got = blk.X.copy()
            for c in range(blk.X.shape[1]):
                plus = float(np.max(np.abs(blk.X[:, c] - X_ref[:, c])))
                minus = float(np.max(np.abs(blk.X[:, c] + X_ref[:, c])))
                if minus < plus:
                    signs[c] = -1.0
                    X_got[:, c] = -blk.X[:, c]

            tol_X = max(1e-6, 1e-5 * float(np.max(np.abs(X_ref))))
            assert np.allclose(X_got, X_ref, atol=tol_X, rtol=0), (
                f"smooth #{i} block {k} ({r_meta['class']}): X values diverge"
            )

            assert len(blk.S) == r_meta["n_penalties"], (
                f"smooth #{i} block {k}: got {len(blk.S)} penalties want {r_meta['n_penalties']}"
            )
            for j, S_got in enumerate(blk.S, start=1):
                S_ref = np.asarray(
                    mmread(fx / f"smooth_{i}_{k}_S_{j}.mtx").todense(), dtype=float
                )
                assert S_got.shape == S_ref.shape, (
                    f"smooth #{i} block {k} S_{j}: got {S_got.shape} want {S_ref.shape}"
                )
                S_got_flipped = S_got * signs[:, None] * signs[None, :]
                tol_S = max(1e-6, 1e-5 * float(np.max(np.abs(S_ref))))
                assert np.allclose(S_got_flipped, S_ref, atol=tol_S, rtol=0), (
                    f"smooth #{i} block {k} S_{j} ({r_meta['class']}): penalty values diverge"
                )
