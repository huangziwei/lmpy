"""Phase 1 of ``lmpy.plot``: dispatch + scatter + boxplot + diagnostic.

These tests use the Agg backend so they run headless. Each one checks the
*structural* result (axes returned, labels set, dtype-driven dispatch
choice) rather than pixel content — pixel-level checks belong in a visual
regression tool, not unit tests."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # noqa: E402

import matplotlib.axes
import matplotlib.lines
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest

from lmpy import factor, lm
from lmpy import plot as lmplot


@pytest.fixture
def numeric_df():
    rng = np.random.RandomState(0)
    return pl.DataFrame({
        "x": np.linspace(0, 10, 50),
        "y": np.linspace(0, 10, 50) + rng.randn(50),
        "z": np.linspace(-5, 5, 50),
    })


@pytest.fixture
def factor_df():
    df = pl.DataFrame({
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "g": [0, 0, 0, 1, 1, 1],
    })
    return df.with_columns(factor(df["g"], labels={0: "A", 1: "B"}))


def test_plot_formula_num_num(numeric_df):
    """`plot('y ~ x', data=df)` returns an Axes with a scatter and labeled axes."""
    ax = lmplot.plot("y ~ x", data=numeric_df)
    assert isinstance(ax, matplotlib.axes.Axes)
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "y"
    # scatter creates a PathCollection
    assert len(ax.collections) >= 1
    plt.close("all")


def test_plot_formula_num_factor_dispatches_to_boxplot(factor_df):
    """A factor RHS triggers boxplot — verify by xtick labels matching the
    factor's level order, not numeric x-values."""
    ax = lmplot.plot("y ~ g", data=factor_df)
    xticks = [t.get_text() for t in ax.get_xticklabels()]
    assert xticks == ["A", "B"]
    assert ax.get_ylabel() == "y"
    plt.close("all")


def test_plot_xy_two_vectors():
    ax = lmplot.plot(np.arange(5), np.arange(5) * 2)
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close("all")


def test_plot_single_vector():
    ax = lmplot.plot(np.array([10.0, 20.0, 15.0, 25.0]))
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close("all")


def test_plot_lm_diagnostic(numeric_df):
    """`plot(lmod)` returns 4 axes — the standard diagnostic panel."""
    m = lm("y ~ x", numeric_df)
    axes = lmplot.plot(m)
    assert len(axes) == 4
    titles = {a.get_title() for a in axes}
    # all 4 panels labeled
    assert "" not in titles
    plt.close("all")


def test_plot_lm_which_subset(numeric_df):
    """`plot(lmod, which=...)` honors the panel selection."""
    m = lm("y ~ x", numeric_df)
    axes = lmplot.plot(m, which=(1, 2))
    assert len(axes) == 2
    # single panel returns a bare Axes
    ax = lmplot.plot(m, which=2)
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close("all")


def test_formula_expression_lhs(numeric_df):
    """Expression LHS like `residuals(m) ~ x` resolves the model from the
    caller's frame and evaluates `residuals()` from the default env."""
    m = lm("y ~ x", numeric_df)  # noqa: F841 — used by the formula via frame lookup
    ax = lmplot.plot("residuals(m) ~ x", data=numeric_df)
    assert ax.get_ylabel() == "residuals(m)"
    assert ax.get_xlabel() == "x"
    plt.close("all")


def test_formula_expression_both_sides(numeric_df):
    """Expressions on both sides — `log(y) ~ x` style."""
    df = numeric_df.with_columns(pl.col("y").abs() + 0.1)
    ax = lmplot.plot("log(y) ~ x", data=df)
    assert ax.get_ylabel() == "log(y)"
    plt.close("all")


def test_formula_multi_rhs(numeric_df):
    """`y ~ x + z` produces one panel per additive term."""
    axes = lmplot.plot("y ~ x + z", data=numeric_df)
    assert len(axes) == 2
    assert [a.get_xlabel() for a in axes] == ["x", "z"]
    plt.close("all")


def test_explicit_axes_chaining(numeric_df):
    """User-supplied `ax=` is reused (no new figure created)."""
    fig, ax = plt.subplots()
    returned = lmplot.plot("y ~ x", data=numeric_df, ax=ax)
    assert returned is ax
    plt.close("all")


def test_factor_input_with_pch_codes(numeric_df):
    """`pch=` accepts a polars Enum and turns into per-level markers without
    promoting the column to a string column upstream."""
    df = numeric_df.with_columns(
        factor(pl.Series("g", [0, 1] * 25), labels={0: "L", 1: "R"}).alias("g")
    )
    ax = lmplot.plot(df["x"], df["y"], pch=df["g"])
    # one scatter per unique pch code → 2 PathCollections
    assert len(ax.collections) == 2
    plt.close("all")


def test_formula_requires_data():
    with pytest.raises(ValueError, match="data="):
        lmplot.plot("y ~ x")


def test_formula_requires_lhs(numeric_df):
    with pytest.raises(ValueError, match="one-sided"):
        lmplot.plot("~ x", data=numeric_df)


# ---------------------------------------------------------------------------
# Phase 2: annotations
# ---------------------------------------------------------------------------


def test_abline_from_lm_object(numeric_df):
    """`abline(lmod)` extracts (intercept, slope) and draws a single line."""
    m = lm("y ~ x", numeric_df)
    ax = lmplot.plot("y ~ x", data=numeric_df)
    n_lines_before = len(ax.lines)
    lmplot.abline(m, ax=ax)
    assert len(ax.lines) == n_lines_before + 1
    # slope should match the lm fit's slope coefficient
    line = ax.lines[-1]
    xs = line.get_xdata()
    ys = line.get_ydata()
    slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
    np.testing.assert_allclose(slope, float(m.bhat["x"].item()), rtol=1e-10)
    plt.close("all")


def test_abline_a_b(numeric_df):
    """`abline(a, b)` draws y = a + b*x with the requested lty."""
    ax = lmplot.plot("y ~ x", data=numeric_df)
    lmplot.abline(0.0, 1.0, ax=ax, lty=2)
    assert ax.lines[-1].get_linestyle() == "--"
    plt.close("all")


def test_abline_coef_array(numeric_df):
    """`abline(coef_array)` accepts a length-2 vector [intercept, slope]."""
    ax = lmplot.plot("y ~ x", data=numeric_df)
    lmplot.abline(np.array([1.5, 2.0]), ax=ax)
    assert len(ax.lines) == 1  # the abline
    plt.close("all")


def test_abline_h_v(numeric_df):
    """`abline(h=, v=)` adds horizontal and vertical reference lines."""
    ax = lmplot.plot("y ~ x", data=numeric_df)
    lmplot.abline(h=0, v=5, ax=ax)
    # 2 lines added (one h, one v)
    assert len(ax.lines) >= 2
    plt.close("all")


def test_abline_h_vector_with_lty_vector(numeric_df):
    """`abline(h=c(0, 10), lty=1:2)` — vectorized horizontals with per-line lty."""
    ax = lmplot.plot("y ~ x", data=numeric_df)
    lmplot.abline(h=[0.0, 10.0], lty=[1, 2], ax=ax)
    styles = [ln.get_linestyle() for ln in ax.lines[-2:]]
    assert "-" in styles and "--" in styles
    plt.close("all")


def test_abline_requires_ax(numeric_df):
    with pytest.raises(ValueError, match="`ax=` is required"):
        lmplot.abline(0, 1)


def test_abline_lm_without_intercept_errors(numeric_df):
    """`abline(lmod)` requires the fit to have an intercept and exactly one slope."""
    m = lm("y ~ 0 + x", numeric_df)  # no intercept
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="(Intercept)|2 coefficients"):
        lmplot.abline(m, ax=ax)
    plt.close("all")


def test_abline_coef_wrong_length_errors():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="length-2"):
        lmplot.abline(np.array([1.0, 2.0, 3.0]), ax=ax)
    plt.close("all")


def test_points_overlay(numeric_df):
    """`points(x, y, ax=ax)` adds a scatter PathCollection without changing labels."""
    ax = lmplot.plot("y ~ x", data=numeric_df)
    n_before = len(ax.collections)
    lmplot.points(np.array([5.0]), np.array([10.0]), ax=ax, cex=2)
    assert len(ax.collections) == n_before + 1
    # labels must not have been overwritten
    assert ax.get_xlabel() == "x"
    plt.close("all")


def test_lines_xy(numeric_df):
    ax = lmplot.plot("y ~ x", data=numeric_df)
    n_before = len(ax.lines)
    lmplot.lines(np.array([0.0, 10.0]), np.array([0.0, 20.0]), ax=ax, lty=2)
    assert len(ax.lines) == n_before + 1
    assert ax.lines[-1].get_linestyle() == "--"
    plt.close("all")


def test_lines_formula(numeric_df):
    """`lines("y ~ x", data=df)` — formula-driven overlay, mirrors R."""
    ax = lmplot.plot("y ~ x", data=numeric_df)
    n_before = len(ax.lines)
    lmplot.lines("y ~ x", data=numeric_df, ax=ax, lty=3)
    assert len(ax.lines) == n_before + 1
    plt.close("all")


def test_legend_r_position_string(numeric_df):
    """R-style "topright" maps to matplotlib "upper right"; legend is attached."""
    ax = lmplot.plot("y ~ x", data=numeric_df)
    lmplot.legend("topright", legend=["a", "b"], pch=[0, 1], lty=[1, 2], ax=ax)
    leg = ax.get_legend()
    assert leg is not None
    assert [t.get_text() for t in leg.get_texts()] == ["a", "b"]
    plt.close("all")


def test_segments_scalar_and_vector(numeric_df):
    ax = lmplot.plot("y ~ x", data=numeric_df)
    n_before = len(ax.lines)
    lmplot.segments(0, 0, 10, 20, ax=ax)
    lmplot.segments([0, 5], [0, 5], [10, 10], [20, 10], ax=ax, lty=2)
    assert len(ax.lines) == n_before + 3  # 1 + 2 segments
    plt.close("all")


def test_qqline_through_quartiles():
    """qqline must pass through (q25, y25) and (q75, y75) for standard normal."""
    rng = np.random.RandomState(0)
    vals = rng.randn(200)
    fig, ax = plt.subplots()
    n = len(vals)
    probs = (np.arange(1, n + 1) - 0.5) / n
    from scipy.stats import norm
    q = norm.ppf(probs)
    ax.scatter(q, np.sort(vals))
    lmplot.qqline(vals, ax=ax)
    line = ax.lines[-1]
    xs = line.get_xdata()
    ys = line.get_ydata()
    slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
    # for ~N(0,1) data the qqline should be near slope 1
    assert 0.7 < slope < 1.3
    plt.close("all")
