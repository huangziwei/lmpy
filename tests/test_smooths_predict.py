"""
Compare BasisSpec.predict_mat() against mgcv's Predict.matrix output.

For each mgcv fixture, the R generator now also dumps PredictMat(s, predict_data)
per smooth block (`smooth_<i>_<k>_Xpred.mtx`) and the predict_data subset
(`predict_data.csv`). We rebuild the smooth at fit-time on `data`, then ask
each block's BasisSpec to produce the design at `predict_data` and compare
against R's output.

Sign / null-rotation handling mirrors test_smooths.py: tp's Lanczos and fs's
null-eigenvector signs differ between LAPACKs, so we match each column up to
sign (same flip the fit-time test uses). Fixtures whose Xpred file is missing
(R's PredictMat raised — usually for niche bs combos) are skipped.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest
from scipy.io import mmread

from conftest import (
    FIXTURE_ROOT,
    _apply_schema,
    fixture_meta,
    fixtures_by_kind,
    load_dataset,
)
from hea.formula import (
    _canonicalize_fs_null_basis,
    _factor_levels,
    _fs_find_factor,
    expand,
    materialize_smooths,
    parse,
)


def _load_predict_data(fx_id: str, pkg: str, name: str) -> pl.DataFrame:
    """Load `predict_data.csv` and re-apply factor schema (CSV round-trip
    erases R factor types — without this, fs/sz/by=factor smooths fail to
    match levels in our predict closure)."""
    df = pl.read_csv(FIXTURE_ROOT / fx_id / "predict_data.csv", null_values="NA")
    return _apply_schema(df, pkg, name)


def _canonicalize_fs_reference(X_ref, r_meta, data):
    """Per-level fs canonical-rotation rebuild — same shape as in test_smooths."""
    term = r_meta["term"] if isinstance(r_meta["term"], list) else [r_meta["term"]]
    fterm, _others = _fs_find_factor(term, data)
    assert fterm is not None, f"fs.interaction needs a factor term; got {term}"
    flev = _factor_levels(data[fterm])
    p = r_meta["bs_dim"]
    null_d = r_meta["n_penalties"] - 1
    rank = p - null_d

    fac_arr = data[fterm].to_numpy()
    Xr = np.zeros((X_ref.shape[0], p))
    for j, lev in enumerate(flev):
        mask = fac_arr == lev
        Xr[mask, :] = X_ref[mask, j * p : (j + 1) * p]

    Xr_canonical, _rot, _signs = _canonicalize_fs_null_basis(Xr, rank)

    X_new = np.zeros_like(X_ref)
    for j, lev in enumerate(flev):
        mask = (fac_arr == lev).astype(float)
        X_new[:, j * p : (j + 1) * p] = Xr_canonical * mask[:, None]
    return X_new


MGCV_FIXTURES = fixtures_by_kind("mgcv")


def _has_error(fx_id: str) -> bool:
    sm_meta = json.loads((FIXTURE_ROOT / fx_id / "smooth_meta.json").read_text())
    return any("error" in s for s in sm_meta.get("smooths", []))


def _has_predict_data(fx_id: str) -> bool:
    return (FIXTURE_ROOT / fx_id / "predict_data.csv").exists()


MGCV_OK = [
    e["id"] for e in MGCV_FIXTURES
    if not _has_error(e["id"]) and _has_predict_data(e["id"])
]


@pytest.mark.parametrize("fx_id", MGCV_OK)
def test_mgcv_predict_mat_matches_R(fx_id: str):
    meta, _ = fixture_meta(fx_id)
    fx = FIXTURE_ROOT / fx_id
    sm_meta = json.loads((fx / "smooth_meta.json").read_text())

    pkg, name = meta["dataset"]["pkg"], meta["dataset"]["name"]
    data = load_dataset(pkg, name)

    f = parse(meta["formula"])
    data_cols = list(data.columns) if "." in meta["formula"] else None
    ef = expand(f, data_columns=data_cols)

    # NA-omit on every formula variable — same rule R's gam applies and
    # what `make_mgcv_fixture` did when constructing predict_data.
    need = set(meta.get("need_vars", [])) & set(data.columns)
    if need:
        data = data.drop_nulls(subset=list(need))

    ours = materialize_smooths(ef, data)
    new = _load_predict_data(fx_id, pkg, name)

    assert len(ours) == len(sm_meta["smooths"]), (
        f"n smooths: got {len(ours)} want {len(sm_meta['smooths'])}"
    )

    for i, (ours_blocks, r_meta) in enumerate(zip(ours, sm_meta["smooths"]), start=1):
        for k, blk in enumerate(ours_blocks, start=1):
            xpred_path = fx / f"smooth_{i}_{k}_Xpred.mtx"
            if not xpred_path.exists():
                # mgcv's PredictMat refused for this block (rare niche cases) —
                # nothing to compare against.
                continue

            assert blk.spec is not None, (
                f"smooth #{i} block {k} ({r_meta['class']}): missing BasisSpec"
            )

            X_pred_ref = np.asarray(mmread(xpred_path).todense(), dtype=float)
            X_pred_ours = blk.spec.predict_mat(new)

            assert X_pred_ours.shape == X_pred_ref.shape, (
                f"smooth #{i} block {k}: predict shape "
                f"got {X_pred_ours.shape} want {X_pred_ref.shape}"
            )

            # fs.interaction: same null-eigenvector rotation we apply at fit;
            # apply the canonical rotation to mgcv's predict reference using
            # the predict_data factor column.
            if r_meta["class"] == "fs.smooth.spec":
                X_pred_ref = _canonicalize_fs_reference(X_pred_ref, r_meta, new)

            # Match column signs against an in-sample anchor that lives in the
            # same column space as the predict basis. For most bases this is
            # `sm$X` (smooth_*_X.mtx) — fit and predict bases coincide. For
            # `t2` they don't: sm$X is the partial absorb (sm$Cp ignored), but
            # PredictMat applies the full absorb via sm$qrc, giving a basis
            # whose per-column signs do not match sm$X's. For those we use
            # `Xpredfit.mtx` (PredictMat at fit data), which carries the
            # predict-side sign convention and is paired with our
            # in-sample `predict_mat(data)` output. We gate on coef_remap —
            # only t2 needs this; for other bases sm$X already matches the
            # predict basis (sometimes Xpredfit doesn't, e.g. random.effect).
            xpredfit_path = fx / f"smooth_{i}_{k}_Xpredfit.mtx"
            use_predfit_anchor = (
                xpredfit_path.exists()
                and blk.spec is not None
                and blk.spec.coef_remap is not None
            )
            if use_predfit_anchor:
                anchor_ref = np.asarray(
                    mmread(xpredfit_path).todense(), dtype=float
                )
                anchor_ours = np.asarray(blk.spec.predict_mat(data), dtype=float)
            else:
                anchor_ref = np.asarray(
                    mmread(fx / f"smooth_{i}_{k}_X.mtx").todense(), dtype=float
                )
                if r_meta["class"] == "fs.smooth.spec":
                    anchor_ref = _canonicalize_fs_reference(anchor_ref, r_meta, data)
                anchor_ours = blk.X

            signs = np.ones(blk.X.shape[1])
            for c in range(blk.X.shape[1]):
                plus = float(np.max(np.abs(anchor_ours[:, c] - anchor_ref[:, c])))
                minus = float(np.max(np.abs(anchor_ours[:, c] + anchor_ref[:, c])))
                if minus < plus:
                    signs[c] = -1.0
            X_pred_aligned = X_pred_ours * signs[None, :]

            tol = max(1e-6, 1e-5 * float(np.max(np.abs(X_pred_ref))))
            assert np.allclose(X_pred_aligned, X_pred_ref, atol=tol, rtol=0), (
                f"smooth #{i} block {k} ({r_meta['class']}): predict_mat diverges "
                f"(max abs diff = {float(np.max(np.abs(X_pred_aligned - X_pred_ref))):.2e}, "
                f"tol = {tol:.2e})"
            )
