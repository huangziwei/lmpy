"""Tests for ``gam.vis()`` — the hea port of mgcv's ``vis.gam``.

The correctness invariant for ``vis()`` is ``vis(view) == predict(grid)``: the
method just calls ``predict`` on a regular grid over two view variables, with
all other variables held at their typical (median / modal) value. The
``predict`` end-to-end vs mgcv comparison lives in ``test_smooths_predict.py``;
here we check that the grid construction, dtype handling, SE pipeline,
``too_far`` masking, and factor-axis support all behave as expected.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl
import pytest

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from conftest import load_dataset

from hea import gam
from hea.gam import VisResult


@pytest.fixture(scope="module")
def trees_te():
    """trees with a 2D tensor smooth — the canonical vis.gam example."""
    data = (
        load_dataset("mgcv", "trees")
        .rename({"Volume": "vol", "Girth": "g", "Height": "h"})
    )
    m = gam("vol ~ te(g, h)", data=data, method="REML")
    return m, data


@pytest.fixture(scope="module")
def factor_model():
    """A model with one numeric and one factor RHS variable."""
    rng = np.random.RandomState(0)
    df = pl.DataFrame({
        "y": rng.randn(120),
        "x": rng.rand(120),
        "g": (["a", "b", "c"] * 40),
    })
    m = gam("y ~ s(x) + g", data=df)
    return m, df


def test_vis_matches_predict_on_same_grid(trees_te):
    """vis(view, n_grid) must equal predict(grid) — no extra computation."""
    m, data = trees_te
    v = m.vis(view=("g", "h"), n_grid=20, type="link")

    G, H = np.meshgrid(v.m1, v.m2, indexing="ij")
    new = pl.DataFrame({"g": G.ravel(), "h": H.ravel()}).with_columns(
        pl.col("g").cast(data["g"].dtype),
        pl.col("h").cast(data["h"].dtype),
    )
    fit_pred = m.predict(new, type="link").reshape(20, 20)
    assert np.allclose(v.fit, fit_pred, atol=1e-12, rtol=0)


def test_vis_se_matches_predict_se(trees_te):
    """SE on the grid must match predict(se_fit=True) on the same grid."""
    m, data = trees_te
    v = m.vis(view=("g", "h"), n_grid=15, type="link", se=True)

    G, H = np.meshgrid(v.m1, v.m2, indexing="ij")
    new = pl.DataFrame({"g": G.ravel(), "h": H.ravel()}).with_columns(
        pl.col("g").cast(data["g"].dtype),
        pl.col("h").cast(data["h"].dtype),
    )
    fit_pred, se_pred = m.predict(new, type="link", se_fit=True)
    assert np.allclose(v.fit, fit_pred.reshape(15, 15), atol=1e-12)
    assert np.allclose(v.se, se_pred.reshape(15, 15), atol=1e-12)


def test_vis_response_scale_matches_link_via_inverse(trees_te):
    """type='response' = linkinv(η̂); SE scaled by |dμ/dη| (delta method)."""
    m, _ = trees_te
    v_link = m.vis(view=("g", "h"), n_grid=10, type="link", se=True)
    v_resp = m.vis(view=("g", "h"), n_grid=10, type="response", se=True)
    # Identity link → response == link, identical SEs
    assert np.allclose(v_link.fit, v_resp.fit)
    assert np.allclose(v_link.se, v_resp.se)


def test_auto_pick_view(trees_te):
    """No `view`: pick the first two RHS vars with variation."""
    m, _ = trees_te
    v = m.vis()
    assert v.view == ("g", "h")
    assert v.fit.shape == (30, 30)


def test_grid_endpoints(trees_te):
    """Numeric grids span [min(x), max(x)] of the fit data."""
    m, data = trees_te
    v = m.vis(view=("g", "h"), n_grid=8)
    assert v.m1[0] == float(data["g"].min())
    assert v.m1[-1] == float(data["g"].max())
    assert v.m2[0] == float(data["h"].min())
    assert v.m2[-1] == float(data["h"].max())


def test_too_far_masks_distant_points(trees_te):
    """``too_far > 0`` replaces distant grid cells with NaN."""
    m, _ = trees_te
    v0 = m.vis(view=("g", "h"), n_grid=20, too_far=0.0)
    v1 = m.vis(view=("g", "h"), n_grid=20, too_far=0.1)
    assert np.all(np.isfinite(v0.fit))
    assert np.any(np.isnan(v1.fit))
    # No false-positives: every kept cell in v1 == v0 (NaN-only diff)
    keep = ~np.isnan(v1.fit)
    assert np.allclose(v0.fit[keep], v1.fit[keep])


def test_cond_overrides_typical_value():
    """`cond={var: val}` shifts the held-fixed value, changing the surface."""
    rng = np.random.RandomState(1)
    df = pl.DataFrame({
        "y": rng.randn(80),
        "x1": rng.rand(80),
        "x2": rng.rand(80),
        "x3": rng.rand(80),
    })
    m = gam("y ~ s(x1) + s(x2) + s(x3)", data=df, method="REML")
    # x3 is held at median by default; override and the surface changes.
    v_default = m.vis(view=("x1", "x2"), n_grid=8)
    v_override = m.vis(view=("x1", "x2"), n_grid=8, cond={"x3": 0.9})
    # With purely-additive smooths the *shape* of the surface over (x1, x2)
    # only differs by an offset (the s(x3) at x3=median vs x3=0.9). So check
    # that fit_default - fit_override is a constant.
    diff = v_default.fit - v_override.fit
    assert np.std(diff) < 1e-10
    assert abs(np.mean(diff)) > 1e-6  # but the offset is non-zero


def test_factor_view_axis(factor_model):
    """Factor view: m2 contains the level names; surface is well-defined."""
    m, _ = factor_model
    v = m.vis(view=("x", "g"), n_grid=10)
    assert v.fit.shape == (10, 10)
    assert set(np.unique(v.m2)) == {"a", "b", "c"}
    assert np.all(np.isfinite(v.fit))


def test_factor_view_too_far_returns_no_mask(factor_model):
    """too_far is undefined when an axis is a factor; mgcv would crash, we
    quietly return all-False."""
    m, _ = factor_model
    v = m.vis(view=("x", "g"), n_grid=10, too_far=0.5)
    assert np.all(np.isfinite(v.fit))


def test_invalid_view():
    """View must be 2 names from the formula's RHS variables."""
    df = pl.DataFrame({"y": np.arange(10.0), "x": np.arange(10.0)})
    m = gam("y ~ s(x)", data=df, method="REML")
    with pytest.raises(ValueError):
        # only one RHS var with variation — auto-pick fails
        m.vis()
    with pytest.raises(ValueError):
        m.vis(view=("x",))
    with pytest.raises(ValueError):
        m.vis(view=("x", "nope"))


def test_invalid_type(trees_te):
    m, _ = trees_te
    with pytest.raises(ValueError):
        m.vis(view=("g", "h"), type="bogus")


def test_vis_result_repr(trees_te):
    m, _ = trees_te
    v = m.vis(view=("g", "h"), n_grid=5)
    s = repr(v)
    assert "VisResult" in s and "view=('g', 'h')" in s


def test_plot_contour_smoke(trees_te):
    """``.plot(kind='contour')`` returns an Axes without raising."""
    m, _ = trees_te
    v = m.vis(view=("g", "h"), n_grid=10, se=True)
    ax = v.plot(kind="contour")
    assert ax is not None
    plt.close("all")


def test_plot_persp_smoke(trees_te):
    """``.plot(kind='persp')`` with se_mult draws ± envelopes."""
    m, _ = trees_te
    v = m.vis(view=("g", "h"), n_grid=10, se=True)
    ax = v.plot(kind="persp", se_mult=2.0)
    assert ax is not None
    plt.close("all")


def test_plot_factor_axis_ticks(factor_model):
    """Factor axis on a contour plot: ticks rendered as level names."""
    m, _ = factor_model
    v = m.vis(view=("x", "g"), n_grid=10)
    ax = v.plot(kind="contour")
    yticks = [t.get_text() for t in ax.get_yticklabels()]
    # levels appear repeated in the grid; the unique non-empty labels must
    # be a subset of {"a", "b", "c"}.
    assert {t for t in yticks if t} <= {"a", "b", "c"}
    plt.close("all")


def test_invalid_plot_kind(trees_te):
    m, _ = trees_te
    v = m.vis(view=("g", "h"), n_grid=5)
    with pytest.raises(ValueError):
        v.plot(kind="surface")  # only contour/persp supported
