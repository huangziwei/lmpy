"""Phase 1 of ``hea.plot``: dispatch + scatter + boxplot + diagnostic.

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

from hea import factor, lm
from hea import plot as lmplot


@pytest.fixture
def numeric_df():
    rng = np.random.RandomState(0)
    # z carries its own noise so it isn't perfectly collinear with x
    # (linspace(-5, 5) = linspace(0, 10) - 5 → would alias to (Intercept) + x).
    return pl.DataFrame({
        "x": np.linspace(0, 10, 50),
        "y": np.linspace(0, 10, 50) + rng.randn(50),
        "z": np.linspace(-5, 5, 50) + rng.randn(50),
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


# ---------------------------------------------------------------------------
# Phase 3: Faraway helpers — qqnorm, halfnorm, termplot
# ---------------------------------------------------------------------------


def test_qqnorm_pairs_with_qqline():
    """qqnorm draws a Q-Q scatter; chaining qqline adds the reference line."""
    rng = np.random.RandomState(0)
    vals = rng.randn(100)
    ax = lmplot.qqnorm(vals)
    assert ax.get_xlabel() == "Theoretical Quantiles"
    assert ax.get_ylabel() == "Sample Quantiles"
    assert ax.get_title() == "Normal Q-Q"
    n_lines_before = len(ax.lines)
    lmplot.qqline(vals, ax=ax)
    assert len(ax.lines) == n_lines_before + 1
    plt.close("all")


def test_qqnorm_handles_nans():
    """NaNs are dropped before computing quantiles — no crash, no warning."""
    vals = np.array([1.0, np.nan, 2.0, 3.0, np.nan, -1.0])
    ax = lmplot.qqnorm(vals)
    # 4 finite values plotted as a single PathCollection
    assert len(ax.collections) == 1
    assert ax.collections[0].get_offsets().shape[0] == 4
    plt.close("all")


def test_halfnorm_labels_top_nlab():
    """halfnorm with nlab=3 labels the 3 largest |x|; smaller points scattered."""
    vals = np.array([0.1, -0.2, 0.3, 5.0, -6.0, 7.0])  # last 3 are biggest
    ax = lmplot.halfnorm(vals, 3)
    # 3 annotations placed
    texts = [t.get_text() for t in ax.texts]
    assert len(texts) == 3
    # default labels are 1-based indices of the input order; biggest |x| at indices 4, 5, 6
    assert set(texts) == {"4", "5", "6"}
    # remaining 3 points scattered
    assert ax.collections[0].get_offsets().shape[0] == 3
    plt.close("all")


def test_halfnorm_custom_labs():
    """`labs=` overrides default integer indices."""
    vals = np.array([0.1, 5.0, 0.2, 6.0])
    ax = lmplot.halfnorm(vals, 2, labs=["a", "b", "c", "d"])
    texts = sorted(t.get_text() for t in ax.texts)
    # biggest |x| at indices 1, 3 → labels "b", "d"
    assert texts == ["b", "d"]
    plt.close("all")


def test_halfnorm_labs_length_mismatch():
    with pytest.raises(ValueError, match="halfnorm"):
        lmplot.halfnorm(np.array([1.0, 2.0, 3.0]), labs=["a", "b"])


def test_termplot_single_term(numeric_df):
    """`termplot(lmod)` on a one-term fit returns a bare Axes with the term name."""
    m = lm("y ~ x", numeric_df)
    ax = lmplot.termplot(m)
    assert isinstance(ax, matplotlib.axes.Axes)
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "Partial for x"
    # one line drawn (the centered contribution)
    assert len(ax.lines) == 1
    plt.close("all")


def test_termplot_multi_term(numeric_df):
    """Multi-term fit returns a list of axes, one per term."""
    m = lm("y ~ x + z", numeric_df)
    axes = lmplot.termplot(m)
    assert len(axes) == 2
    assert [a.get_xlabel() for a in axes] == ["x", "z"]
    plt.close("all")


def test_termplot_terms_selection_by_name(numeric_df):
    """`terms='z'` picks just that term."""
    m = lm("y ~ x + z", numeric_df)
    ax = lmplot.termplot(m, terms="z")
    assert isinstance(ax, matplotlib.axes.Axes)
    assert ax.get_xlabel() == "z"
    plt.close("all")


def test_termplot_terms_selection_by_index(numeric_df):
    """1-based int (R style) picks term by position."""
    m = lm("y ~ x + z", numeric_df)
    ax = lmplot.termplot(m, terms=2)
    assert ax.get_xlabel() == "z"
    plt.close("all")


def test_termplot_partial_residuals(numeric_df):
    """`partial_resid=True` overlays a scatter of residuals + b*(x - mean(x))."""
    m = lm("y ~ x", numeric_df)
    ax = lmplot.termplot(m, partial_resid=True)
    # one line + one scatter PathCollection
    assert len(ax.lines) == 1
    assert len(ax.collections) == 1
    plt.close("all")


def test_termplot_factor_dummy_errors(factor_df):
    """v1 rejects factor dummies (binary 0/1 columns)."""
    m = lm("y ~ g", factor_df)
    with pytest.raises(ValueError, match="factor dummy|continuous"):
        lmplot.termplot(m)
    plt.close("all")


def test_termplot_unknown_term_errors(numeric_df):
    m = lm("y ~ x", numeric_df)
    with pytest.raises(ValueError, match="not in"):
        lmplot.termplot(m, terms="nope")
    plt.close("all")


def test_termplot_non_lm_input():
    with pytest.raises(TypeError, match="lm/glm"):
        lmplot.termplot("not a model")


def test_pairs_default_uses_all_numeric_columns(numeric_df):
    """pairs() defaults to every numeric column → 3×3 grid for the fixture."""
    axes = lmplot.pairs(numeric_df)
    assert axes.shape == (3, 3)
    # diagonal in "label" mode prints column name as text, no scatter points
    diag_text = [t.get_text() for t in axes[0, 0].texts]
    assert diag_text == ["x"]
    assert len(axes[0, 0].collections) == 0
    # off-diagonal cell has a scatter PathCollection
    assert len(axes[0, 1].collections) == 1
    plt.close("all")


def test_pairs_cols_subset_and_labels(numeric_df):
    """cols= picks a subset and labels= renames the diagonal text."""
    axes = lmplot.pairs(numeric_df, cols=["x", "y"], labels=["X", "Y"])
    assert axes.shape == (2, 2)
    assert axes[0, 0].texts[0].get_text() == "X"
    assert axes[1, 1].texts[0].get_text() == "Y"
    plt.close("all")


def test_pairs_diag_hist(numeric_df):
    """diag='hist' draws a histogram (Patch artists) on each diagonal cell."""
    axes = lmplot.pairs(numeric_df, cols=["x", "y"], diag="hist")
    # hist creates Rectangle patches; label/none modes add none
    assert len(axes[0, 0].patches) > 0
    assert len(axes[1, 1].patches) > 0
    plt.close("all")


def test_pairs_rejects_non_dataframe():
    with pytest.raises(TypeError, match="DataFrame"):
        lmplot.pairs(np.arange(10))


def test_pairs_rejects_single_column(numeric_df):
    with pytest.raises(ValueError, match="at least 2"):
        lmplot.pairs(numeric_df, cols=["x"])


def test_pairs_invalid_diag(numeric_df):
    with pytest.raises(ValueError, match="diag="):
        lmplot.pairs(numeric_df, diag="bogus")


def test_pairs_labels_length_mismatch(numeric_df):
    with pytest.raises(ValueError, match="labels has"):
        lmplot.pairs(numeric_df, cols=["x", "y"], labels=["only-one"])


def test_plot_dataframe_dispatches_to_pairs(numeric_df):
    """plot(df) for a DataFrame routes to pairs() (R's plot.data.frame)."""
    axes = lmplot.plot(numeric_df)
    assert axes.shape == (3, 3)
    plt.close("all")


def test_plot_leverage_constant_swaps_to_factor_levels():
    """One-way ANOVA on a balanced design (PlantGrowth: 30 obs / 3 groups)
    has h_ii = 1/n_g for every row → constant leverage. R's plot.lm swaps
    panel 5 from 'Residuals vs Leverage' to 'Constant Leverage: Residuals
    vs Factor Levels'. Match that."""
    from hea import data
    pg = data("PlantGrowth")
    m = lm("weight ~ group", data=pg)
    fig, ax = plt.subplots()
    m.plot_leverage(ax=ax)
    assert "Constant Leverage" in ax.get_title()
    assert ax.get_xlabel() == "Factor Level Combinations"
    # 3 groups → 3 distinct x-tick positions
    assert len(ax.get_xticks()) == 3
    tick_labels = [t.get_text() for t in ax.get_xticklabels()]
    assert set(tick_labels) == {"ctrl", "trt1", "trt2"}
    plt.close("all")


def test_plot_leverage_keeps_standard_view_for_continuous_predictor(numeric_df):
    """A model with a non-constant hat matrix (any continuous predictor)
    must still draw the standard Residuals-vs-Leverage view."""
    m = lm("y ~ x + z", data=numeric_df)
    fig, ax = plt.subplots()
    m.plot_leverage(ax=ax)
    assert ax.get_title() == "Residuals vs. Leverage"
    assert ax.get_xlabel() == "Leverage"
    plt.close("all")


def test_pairs_perimeter_tick_labels_alternate(numeric_df):
    """R's pairs.default places tick labels on alternating perimeter sides
    so each variable's scale appears once around the matrix edge:
    top row ↔ odd-index cols, bottom row ↔ even-index cols,
    left col ↔ odd-index rows, right col ↔ even-index rows (0-indexed)."""
    axes = lmplot.pairs(numeric_df)  # 3 numeric cols → 3×3 grid
    n = axes.shape[0]

    def visible_sides(ax):
        # Read back the labelXXX flags via a major tick's label visibility
        # — matplotlib stores them on each tick after tick_params runs.
        xt = ax.xaxis.get_major_ticks()[0] if ax.xaxis.get_major_ticks() else None
        yt = ax.yaxis.get_major_ticks()[0] if ax.yaxis.get_major_ticks() else None
        return {
            "top":    bool(xt and xt.label2.get_visible()),
            "bottom": bool(xt and xt.label1.get_visible()),
            "left":   bool(yt and yt.label1.get_visible()),
            "right":  bool(yt and yt.label2.get_visible()),
        }

    for i in range(n):
        for j in range(n):
            sides = visible_sides(axes[i, j])
            assert sides["top"] == (i == 0 and j % 2 == 1), (i, j, sides)
            assert sides["bottom"] == (i == n - 1 and j % 2 == 0), (i, j, sides)
            assert sides["left"] == (j == 0 and i % 2 == 1), (i, j, sides)
            assert sides["right"] == (j == n - 1 and i % 2 == 0), (i, j, sides)
    plt.close("all")


def test_pairs_columns_share_x_limits(numeric_df):
    """Every cell in column j is pinned to variable_j's range, so tick
    marks line up vertically (matching R's pairs)."""
    axes = lmplot.pairs(numeric_df)
    n = axes.shape[0]
    for j in range(n):
        col_xlims = [axes[i, j].get_xlim() for i in range(n)]
        assert all(xl == col_xlims[0] for xl in col_xlims), \
            f"col {j} xlims diverge: {col_xlims}"
    plt.close("all")


def test_pairs_rows_share_y_limits_off_diag_under_hist(numeric_df):
    """Under diag='hist' the diagonal cell uses a count y-axis, but every
    OFF-diagonal cell in row i is still pinned to variable_i's range so
    horizontal tick marks line up across the row."""
    axes = lmplot.pairs(numeric_df, diag="hist")
    n = axes.shape[0]
    for i in range(n):
        off = [axes[i, j].get_ylim() for j in range(n) if j != i]
        assert all(yl == off[0] for yl in off), \
            f"row {i} off-diag ylims diverge: {off}"
    plt.close("all")


def test_interaction_plot_matches_r_cell_means():
    """interaction_plot cell means match R's tapply on Pinheiro & Bates'
    Machines data — Worker on trace, Machine on x, score on y."""
    from hea import data
    m = data("Machines", "nlme")
    ax = lmplot.interaction_plot("Machine", "Worker", "score", data=m)
    assert ax.get_xlabel() == "Machine"
    assert ax.get_ylabel() == "mean of score"
    assert [t.get_text() for t in ax.get_xticklabels()] == ["A", "B", "C"]
    assert len(ax.lines) == 6  # 6 workers
    by_worker = {line.get_label(): line.get_ydata().tolist() for line in ax.lines}
    # Reference values from R tapply(score, list(Worker, Machine), mean)
    expected = {
        "6": [46.8000, 43.6333, 61.3000],
        "2": [52.5667, 59.5667, 61.8333],
        "4": [51.2333, 62.7333, 64.7667],
        "1": [52.6333, 62.9000, 67.2000],
        "3": [59.5333, 68.0333, 70.8000],
        "5": [51.3667, 65.0667, 71.7333],
    }
    for w, ref in expected.items():
        np.testing.assert_allclose(by_worker[w], ref, atol=5e-3)
    plt.close("all")


def test_interaction_plot_series_input_form():
    """Series form: pass each Series directly without data=."""
    from hea import data
    m = data("Machines", "nlme")
    ax = lmplot.interaction_plot(m["Machine"], m["Worker"], m["score"])
    assert len(ax.lines) == 6
    plt.close("all")


def test_interaction_plot_string_without_data_errors():
    df = pl.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})  # noqa: F841
    with pytest.raises(ValueError, match="data="):
        lmplot.interaction_plot("x", "y", "z")


def test_pairs_diag_hist_suppresses_y_perimeter_labels(numeric_df):
    """Diagonal cells with diag='hist' show counts on y, so the perimeter
    y-label rule must not fire there (would print misleading count ticks)."""
    # n=3 (odd) → diagonal (2,2) would normally get right labels via the
    # alternating rule (j=n-1=2, i=2 even). Suppressed for hist diag.
    axes = lmplot.pairs(numeric_df, diag="hist")
    yt = axes[2, 2].yaxis.get_major_ticks()
    if yt:
        assert not yt[0].label2.get_visible()
    plt.close("all")
