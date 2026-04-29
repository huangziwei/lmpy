"""Tests for ``hea.dataframe`` — the chapter-3 tidyverse verbs.

Examples mirror ``dev/r4ds/data-transform.qmd`` so the test names tie
back to the source they implement.
"""

from __future__ import annotations

import polars as pl
import pytest

import hea
from hea import DataFrame, GroupBy, desc, tbl


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def df():
    """Small frame with two groups, used in most tests."""
    return DataFrame(
        {
            "g": ["a", "a", "a", "b", "b", "b"],
            "x": [1, 2, 3, 4, 5, 6],
            "y": [10, 20, 30, 40, 50, 60],
        }
    )


@pytest.fixture
def tied():
    """Frame with ties in ``x`` for slice_min/max with_ties tests."""
    return DataFrame({"g": ["a", "a", "a", "b"], "x": [1, 1, 2, 5]})


# ---------------------------------------------------------------------------
# Class identity / IS-A
# ---------------------------------------------------------------------------


def test_is_pl_dataframe_subclass(df):
    """The whole point of subclassing: hea functions accepting ``pl.DataFrame``
    must accept our subclass without conversion."""
    assert isinstance(df, pl.DataFrame)
    assert isinstance(df, DataFrame)


def test_helpers_exposed_on_polars_namespace():
    """`import hea` patches free-function helpers onto `pl` so chains
    can stay in one namespace."""
    assert pl.desc is desc
    assert pl.tbl is tbl
    assert pl.factor is hea.factor


def test_data_returns_subclass():
    gala = hea.data("gala", package="faraway")
    assert isinstance(gala, DataFrame)
    assert isinstance(gala, pl.DataFrame)


def test_tbl_rewraps_plain_dataframe(df):
    plain = pl.DataFrame.with_columns(df, q=pl.col("x") + 100)
    # Native polars dropped the subclass; tbl() restores it.
    rewrapped = tbl(plain)
    assert isinstance(rewrapped, DataFrame)
    # tbl() on an already-wrapped frame is a no-op.
    assert tbl(rewrapped) is rewrapped


def test_methods_preserve_subclass(df):
    """Every tidyverse method must return our subclass so chains stay typed."""
    out = (
        df.filter(pl.col("x") > 1)
        .arrange("x")
        .distinct()
        .mutate(z=pl.col("x") * 2)
        .select("g", "z")
        .rename(group="g")
        .relocate("z")
    )
    assert isinstance(out, DataFrame)


# ---------------------------------------------------------------------------
# Row verbs
# ---------------------------------------------------------------------------


def test_filter_passthrough(df):
    out = df.filter(pl.col("x") > 3)
    assert out["x"].to_list() == [4, 5, 6]


def test_arrange_ascending(df):
    out = df.arrange("y")
    assert out["y"].to_list() == [10, 20, 30, 40, 50, 60]


def test_arrange_desc(df):
    out = df.arrange(desc("x"))
    assert out["x"].to_list() == [6, 5, 4, 3, 2, 1]


def test_arrange_multi_with_desc(df):
    """Mix ascending and descending within one call."""
    out = df.arrange("g", desc("x"))
    assert out["g"].to_list() == ["a", "a", "a", "b", "b", "b"]
    assert out["x"].to_list() == [3, 2, 1, 6, 5, 4]


def test_arrange_puts_nulls_last():
    """dplyr semantics: NAs sort to the end regardless of direction.

    Polars' default puts nulls first; ``arrange`` overrides to match
    dplyr so the rows you actually want to look at land at the head.
    """
    df = DataFrame({"x": [3, None, 1, 2, None]})
    asc = df.arrange("x")
    assert asc["x"].to_list() == [1, 2, 3, None, None]
    dsc = df.arrange(desc("x"))
    assert dsc["x"].to_list() == [3, 2, 1, None, None]


def test_distinct_no_args_dedupes_full_row():
    df = DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
    out = df.distinct()
    assert out.height == 2


def test_distinct_subset_drops_other_columns_by_default(df):
    """dplyr default: ``distinct(cols)`` returns just those columns."""
    out = df.distinct("g")
    assert out.columns == ["g"]
    assert out.height == 2


def test_distinct_subset_keep_all_true(df):
    """``keep_all=True`` mirrors dplyr's ``.keep_all = TRUE``."""
    out = df.distinct("g", keep_all=True)
    assert out.height == 2
    assert set(out.columns) == {"g", "x", "y"}


# ---------------------------------------------------------------------------
# Column verbs
# ---------------------------------------------------------------------------


def test_mutate_kwargs_auto_alias(df):
    """The motivating ergonomics fix: kwarg name becomes the column name."""
    out = df.mutate(z=pl.col("x") + pl.col("y"))
    assert "z" in out.columns
    assert out["z"].to_list() == [11, 22, 33, 44, 55, 66]


def test_mutate_positional_passthrough(df):
    out = df.mutate(pl.col("x").alias("xx"))
    assert "xx" in out.columns


def test_mutate_before_after_mutually_exclusive(df):
    with pytest.raises(ValueError, match="_before= or _after="):
        df.mutate(z=pl.col("x"), _before="g", _after="x")


def test_mutate_before_places_new_column(df):
    out = df.mutate(z=pl.col("x") + 1, _before="g")
    assert out.columns == ["z", "g", "x", "y"]


def test_mutate_after_places_new_column(df):
    out = df.mutate(z=pl.col("x") + 1, _after="g")
    assert out.columns == ["g", "z", "x", "y"]


def test_mutate_before_position_int(df):
    """``_before=1`` matches dplyr's ``.before = 1`` (before the first column)."""
    out = df.mutate(z=pl.col("x") + 1, _before=1)
    assert out.columns == ["z", "g", "x", "y"]


def test_mutate_after_position_int(df):
    out = df.mutate(z=pl.col("x") + 1, _after=2)
    # _after=2 → after the 2nd column (x), position 2.
    assert out.columns == ["g", "x", "z", "y"]


def test_mutate_position_out_of_range(df):
    with pytest.raises(ValueError, match="out of range"):
        df.mutate(z=pl.col("x"), _before=99)
    with pytest.raises(ValueError, match="out of range"):
        df.mutate(z=pl.col("x"), _before=0)  # dplyr is 1-indexed; 0 is invalid


def test_mutate_keep_none_drops_existing(df):
    out = df.mutate(z=pl.col("x") + 1, _keep="none")
    assert out.columns == ["z"]


def test_mutate_keep_used_keeps_referenced_only():
    """The r4ds example: keep originals referenced by new expressions
    plus the new columns. Self-references (gain_per_hour → gain) don't
    cause `gain` to be doubled — it's already a new column."""
    df = DataFrame({
        "dep_delay": [1.0], "arr_delay": [2.0], "air_time": [60.0],
        "extra": [99],  # NOT referenced — should be dropped
    })
    out = df.mutate(
        gain=pl.col("dep_delay") - pl.col("arr_delay"),
        hours=pl.col("air_time") / 60,
        gain_per_hour=pl.col("gain") / pl.col("hours"),
        _keep="used",
    )
    assert out.columns == [
        "dep_delay", "arr_delay", "air_time",  # referenced originals
        "gain", "hours", "gain_per_hour",      # new
    ]


def test_mutate_is_sequential():
    """Later expressions can refer to columns created earlier in the same call.
    Polars' ``with_columns`` is parallel; we chain so dplyr semantics hold."""
    df = DataFrame({"x": [1, 2, 3]})
    out = df.mutate(
        y=pl.col("x") * 2,
        z=pl.col("y") + 1,
    )
    assert out["z"].to_list() == [3, 5, 7]


def test_mutate_keep_unused_drops_referenced():
    """Inverse of `used`: drop the originals referenced; keep the rest."""
    df = DataFrame({
        "dep_delay": [1.0], "arr_delay": [2.0], "air_time": [60.0],
        "extra": [99],
    })
    out = df.mutate(
        gain=pl.col("dep_delay") - pl.col("arr_delay"),
        _keep="unused",
    )
    # dep_delay, arr_delay are dropped (referenced); air_time and extra survive.
    assert out.columns == ["air_time", "extra", "gain"]


def test_mutate_by_is_windowed(df):
    """``_by`` makes each expression compute within groups; row count preserved."""
    out = df.mutate(x_mean=pl.col("x").mean(), _by="g")
    assert out.height == df.height
    assert out.filter(pl.col("g") == "a")["x_mean"].unique().to_list() == [2.0]
    assert out.filter(pl.col("g") == "b")["x_mean"].unique().to_list() == [5.0]


def test_cols_between():
    """dplyr's ``year:day`` slice syntax via list helper."""
    d = DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]})
    assert d.cols_between("a", "c") == ["a", "b", "c"]
    # Order-insensitive: end before start still yields the same range.
    assert d.cols_between("c", "a") == ["a", "b", "c"]
    # Splat into select.
    assert d.select(d.cols_between("a", "c")).columns == ["a", "b", "c"]
    # Negate via pl.exclude.
    assert d.select(pl.exclude(d.cols_between("a", "c"))).columns == ["d"]


def test_cols_between_missing_column():
    d = DataFrame({"a": [1], "b": [2]})
    with pytest.raises(ValueError, match="not in frame"):
        d.cols_between("a", "z")


def test_select_with_kwargs_renames(df):
    out = df.select("g", x_plus_one=pl.col("x") + 1)
    assert out.columns == ["g", "x_plus_one"]
    assert out["x_plus_one"].to_list() == [2, 3, 4, 5, 6, 7]


def test_select_kwarg_string_is_column_ref(df):
    """dplyr's ``select(tail_num = tailnum)``: bare-string RHS is a column ref."""
    out = df.select(group="g", val="x")
    assert out.columns == ["group", "val"]
    assert out["group"].to_list() == ["a", "a", "a", "b", "b", "b"]
    assert out["val"].to_list() == [1, 2, 3, 4, 5, 6]


def test_rename_kwargs_new_equals_old(df):
    out = df.rename(group="g")
    assert "group" in out.columns and "g" not in out.columns


def test_rename_dict_polars_style(df):
    out = df.rename({"g": "group"})
    assert "group" in out.columns and "g" not in out.columns


def test_rename_rejects_both_forms(df):
    with pytest.raises(ValueError, match="dict or kwargs"):
        df.rename({"g": "group"}, x="xx")


def test_relocate_default_moves_to_front(df):
    out = df.relocate("y")
    assert out.columns == ["y", "g", "x"]


def test_relocate_before(df):
    out = df.relocate("y", _before="x")
    assert out.columns == ["g", "y", "x"]


def test_relocate_after(df):
    out = df.relocate("y", _after="g")
    assert out.columns == ["g", "y", "x"]


def test_relocate_position_int(df):
    """Positional anchors apply to the columns *after* the moves are removed.

    ``_before=1`` of the remaining columns (g, x) → at the very front.
    """
    out = df.relocate("y", _before=1)
    assert out.columns == ["y", "g", "x"]
    out2 = df.relocate("y", _after=2)
    assert out2.columns == ["g", "x", "y"]


def test_relocate_accepts_list_from_cols_between():
    """``cols_between`` output is a list — relocate flattens it."""
    d = DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]})
    out = d.relocate(d.cols_between("a", "b"), _after="d")
    assert out.columns == ["c", "d", "a", "b"]


def test_relocate_accepts_selector():
    """Polars selectors expand against the frame schema."""
    import polars.selectors as cs
    d = DataFrame({"arr_a": [1], "arr_b": [2], "dep_a": [3], "x": [4]})
    out = d.relocate(cs.starts_with("arr"), _before="x")
    # arr columns moved to right before x; dep_a stays where it was.
    assert out.columns == ["dep_a", "arr_a", "arr_b", "x"]


def test_relocate_preserves_frame_order_of_movers():
    """dplyr behavior: movers retain their original relative order."""
    d = DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]})
    # Specified b,a but original order is a,b — result should be a,b,c,d.
    out = d.relocate("b", "a")
    assert out.columns == ["a", "b", "c", "d"]


def test_relocate_anchor_must_exist(df):
    with pytest.raises(ValueError):
        df.relocate("y", _before="nope")


def test_relocate_anchor_cannot_be_moving(df):
    with pytest.raises(ValueError):
        df.relocate("y", _before="y")


# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------


def test_group_by_returns_groupby(df):
    g = df.group_by("g")
    assert isinstance(g, GroupBy)
    assert g.groups == ["g"]


def test_group_by_summarize_named(df):
    """Reproduces the user's motivating example."""
    out = df.group_by("g").summarize(travel=pl.col("x").mean())
    assert out["g"].to_list() == ["a", "b"]
    assert out["travel"].to_list() == [2.0, 5.0]


def test_group_by_maintain_order_default():
    """tibble convention: group_by preserves first-seen order, not lex sort."""
    df = DataFrame({"g": ["b", "a", "b", "a"], "x": [1, 2, 3, 4]})
    out = df.group_by("g").summarize(n=pl.len())
    assert out["g"].to_list() == ["b", "a"]


def test_summarize_no_group_collapses_to_one_row(df):
    out = df.summarize(total=pl.col("x").sum())
    assert out.height == 1
    assert out["total"].item() == 21


def test_summarize_by_kwarg(df):
    """The dplyr 1.1 ``.by =`` per-call grouping form."""
    out = df.summarize(mean_x=pl.col("x").mean(), _by="g")
    assert sorted(out["g"].to_list()) == ["a", "b"]


def test_summarise_british_alias(df):
    out = df.summarise(s=pl.col("x").sum())
    assert out["s"].item() == 21


def test_count_with_columns(df):
    out = df.count("g")
    assert out["g"].to_list() == ["a", "b"]
    assert out["n"].to_list() == [3, 3]


def test_count_no_columns_returns_total(df):
    out = df.count()
    assert out.height == 1 and out["n"].item() == 6


def test_count_sort_descending():
    df = DataFrame({"g": ["a", "b", "b", "b", "c", "c"]})
    out = df.count("g", sort=True)
    assert out["g"].to_list() == ["b", "c", "a"]


def test_count_custom_name(df):
    out = df.count("g", name="freq")
    assert "freq" in out.columns


def test_groupby_count(df):
    out = df.group_by("g").count()
    assert out["n"].to_list() == [3, 3]


def test_groupby_mutate_is_windowed(df):
    """Exercise 6f: ``group_by(g) |> mutate(...)`` is windowed, not collapsing."""
    out = df.group_by("g").mutate(mean_x=pl.col("x").mean())
    assert out.height == df.height
    assert out.filter(pl.col("g") == "a")["mean_x"].unique().to_list() == [2.0]


def test_ungroup_returns_underlying(df):
    g = df.group_by("g")
    assert g.ungroup() is df


def test_ungroup_on_dataframe_is_noop(df):
    """Symmetric API: ungroup on a flat frame is a no-op."""
    assert df.ungroup() is df


# ---------------------------------------------------------------------------
# Slice family
# ---------------------------------------------------------------------------


def test_slice_head(df):
    out = df.slice_head(2)
    assert out.height == 2


def test_slice_tail(df):
    out = df.slice_tail(1)
    assert out["x"].to_list() == [6]


def test_slice_min_with_ties_keeps_all(tied):
    out = tied.slice_min("x", n=1)
    # x=1 appears twice; with_ties=True keeps both.
    assert out.height == 2


def test_slice_min_no_ties(tied):
    out = tied.slice_min("x", n=1, with_ties=False)
    assert out.height == 1


def test_slice_max_with_ties(df):
    out = df.slice_max("y", n=1)
    assert out["y"].to_list() == [60]


def test_groupby_slice_min(df):
    out = df.group_by("g").slice_min("x", n=1)
    # Smallest x per group: a→1, b→4.
    assert sorted(zip(out["g"].to_list(), out["x"].to_list())) == [("a", 1), ("b", 4)]


def test_groupby_slice_max(df):
    out = df.group_by("g").slice_max("y", n=1)
    assert sorted(zip(out["g"].to_list(), out["y"].to_list())) == [("a", 30), ("b", 60)]


def test_groupby_slice_max_keeps_all_null_group():
    """dplyr parity: a group with only null values still yields ``n`` rows.

    Critical for r4ds chapter-3 example: ``flights |> group_by(dest)
    |> slice_max(arr_delay, n=1)`` → 108 rows because LGA has a single
    null-arr_delay row that must survive.
    """
    df = DataFrame({
        "g": ["a", "a", "a", "b"],
        "x": [3, 1, 2, None],  # b has only a null
    })
    out = df.group_by("g").slice_max("x", n=1)
    # a's max=3 (1 row); b's only row is null → kept.
    assert out.height == 2
    b_row = out.filter(pl.col("g") == "b")
    assert b_row.height == 1
    assert b_row["x"].item() is None


def test_slice_max_keeps_all_null_when_only_rows():
    """Same fix applies to the ungrouped slice_max."""
    df = DataFrame({"x": [None, None]})
    assert df.slice_max("x", n=1).height == 2  # both NAs tied at the cutoff
    assert df.slice_max("x", n=1, with_ties=False).height == 1


def test_groupby_slice_max_n_gt_1(df):
    out = df.group_by("g").slice_max("y", n=2)
    assert out.height == 4


def test_groupby_slice_head_per_group(df):
    out = df.group_by("g").slice_head(1)
    assert out["g"].to_list() == ["a", "b"]


def test_slice_sample_n(df):
    out = df.slice_sample(n=3, seed=0)
    assert out.height == 3


def test_slice_sample_prop(df):
    out = df.slice_sample(prop=0.5, seed=0)
    assert out.height == 3


def test_slice_sample_requires_one_of_n_prop(df):
    with pytest.raises(ValueError):
        df.slice_sample()
    with pytest.raises(ValueError):
        df.slice_sample(n=1, prop=0.1)


def test_groupby_slice_sample(df):
    out = df.group_by("g").slice_sample(n=2, seed=0)
    assert out.height == 4
    assert sorted(out["g"].unique().to_list()) == ["a", "b"]


# ---------------------------------------------------------------------------
# End-to-end integration with hea.lm
# ---------------------------------------------------------------------------


def test_chain_then_lm():
    """A full tidyverse chain still produces something hea.lm accepts."""
    gala = hea.data("gala", package="faraway")
    sub = (
        gala.filter(pl.col("Area") > 0)
        .mutate(log_area=pl.col("Area").log())
        .select("Species", "log_area", "Elevation")
    )
    m = hea.lm("Species ~ log_area + Elevation", sub)
    assert m is not None
