"""Tests for the R-style numeric/p-value formatters in lmpy.utils.

Each case is anchored to what R's ``format()`` / ``format.pval()`` produce
(the targets of these helpers).
"""

from __future__ import annotations

import math

from lmpy.utils import format_pval, format_signif, format_signif_jointly


def test_signif_jointly_smaller_drives_decimals():
    # format(c(470.4444, 4.0817), digits=4) → c("470.444", "4.082")
    est, se = format_signif_jointly([[470.4444], [4.0817]], digits=4)
    assert est == ["470.444"]
    assert se == ["4.082"]


def test_signif_single_element():
    # format(115.2562, digits=4) → "115.3"
    assert format_signif([115.2562], digits=4) == ["115.3"]


def test_signif_switches_to_scientific_below_1e_minus_4():
    # format(c(1e-7, 5.0), digits=4) — R picks scientific for 1e-7,
    # fixed for 5.0; column shares the fixed-form decimals.
    out = format_signif([1e-7, 5.0], digits=4)
    assert out[0].endswith("e-07")
    assert out[1] == "5.000"


def test_signif_handles_none_nan_inf():
    out = format_signif([1.5, None, math.nan, math.inf, -math.inf, 0.0], digits=4)
    assert out == ["1.500", "", "NaN", "Inf", "-Inf", "0.000"]


def test_pval_caps_below_machine_eps():
    # format.pval(0, digits=4) prints "<2.22e-16" (eps_digits = 4-2 = 2;
    # no big values so sep is " " when eps_digits > 1, "" when 1).
    out = format_pval([0.0], digits=4)
    assert out[0].startswith("<")
    # eps_digits = max(1, 4-2) = 2 → "2.2e-16"
    assert out[0].lstrip("<").lstrip() == "2.2e-16"


def test_pval_printcoefmat_style_eps_display():
    # When called as printCoefmat does — digits = max(1, min(5, d-1)) = 3 for d=4
    # → eps_digits = max(1, 3-2) = 1 → "<2e-16" (no space, since digits==1).
    out = format_pval([0.0], digits=3)
    assert out[0] == "<2e-16"


def test_pval_mixed_with_eps_and_big():
    # R: format.pval(c(1e-300, 0.000213, 0.5), digits=3)
    #     → c("< 2e-16", "0.000213", "0.500000")
    out = format_pval([1e-300, 0.000213, 0.5], digits=3)
    assert out[0] == "< 2e-16"
    assert out[1] == "0.000213"
    assert out[2] == "0.500000"


def test_pval_scientific_below_1e_minus_4():
    # format.pval(3.8e-11, digits=4) → "3.800e-11"
    out = format_pval([3.8e-11], digits=4)
    assert out[0] == "3.800e-11"
