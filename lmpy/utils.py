"""Cross-model formatting helpers.

* ``significance_code`` — R-style ``***``/``**``/``*``/``.`` formatter
  for p-values.
* ``format_df`` / ``format_signif`` / ``format_pval`` — pandas-style
  printers used by every model's ``summary()``.

Dataset loading and ``factor()`` live in ``lmpy.data``; the formula →
fitting-ready design pipeline lives in ``lmpy.design``; model-comparison
helpers (``anova``, ``AIC``, ``BIC``) in ``lmpy.compare``.
"""

from __future__ import annotations

import numpy as np
import polars as pl

__all__ = ["significance_code", "format_df",
           "format_signif", "format_signif_jointly", "format_pval"]


_MAX_DECIMALS = 6


def _format_numeric_column(vals: list) -> list[str]:
    """Format a numeric column with a single column-wide decimal count.

    Each value is converted to the shortest round-trip Python ``repr``; the
    column-wide decimal count is the max observed there (capped at 6, matching
    pandas' default ``display.precision``), and every value is then padded to
    that count. Scientific-repr values (e.g. ``6.8e-07``) stay in scientific
    form so they don't balloon to long fixed strings.
    """
    out: list[str] = []
    reprs: list[tuple[str, float] | None] = []
    max_dec = 0
    for v in vals:
        if v is None:
            reprs.append(None)
            continue
        fv = float(v)
        if not np.isfinite(fv):
            reprs.append(("nan", fv))
            continue
        r = repr(fv)
        if "e" in r or "E" in r:
            reprs.append(("sci", fv))
            continue
        if "." in r:
            max_dec = max(max_dec, len(r.split(".")[1]))
        reprs.append(("fix", fv))
    max_dec = min(max_dec, _MAX_DECIMALS)
    for item in reprs:
        if item is None:
            out.append("")
        elif item[0] == "nan":
            out.append("NaN")
        elif item[0] == "sci":
            out.append(f"{item[1]:.{_MAX_DECIMALS}g}")
        else:
            out.append(f"{item[1]:.{max_dec}f}")
    return out


def format_df(df: pl.DataFrame, align: dict[str, str] | None = None) -> str:
    """Render a polars DataFrame pandas-style for human-readable output.

    - Numeric columns right-aligned with column-wide fixed decimals.
    - String columns left-aligned.
    - A column named ``""`` prints its header row blank (treated as a row-label column).
    - ``align`` overrides the per-column alignment (``"left"``/``"right"``).
    """
    align = align or {}
    headers = list(df.columns)
    n_rows = df.shape[0]

    formatted: list[list[str]] = []
    aligns: list[str] = []
    for c in headers:
        s = df[c]
        if s.dtype.is_integer():
            cells = [
                "" if v is None else str(int(v))
                for v in s.to_list()
            ]
            default_align = "right"
        elif s.dtype.is_numeric():
            cells = _format_numeric_column(s.to_list())
            default_align = "right"
        else:
            cells = ["" if v is None else str(v) for v in s.to_list()]
            default_align = "left"
        formatted.append(cells)
        aligns.append(align.get(c, default_align))

    widths = [
        max([len(c)] + [len(x) for x in formatted[j]])
        for j, c in enumerate(headers)
    ]

    def pad(s: str, w: int, a: str) -> str:
        return s.ljust(w) if a == "left" else s.rjust(w)

    sep = "  "
    header_cells = [
        (" " * w if c == "" else pad(c, w, a))
        for c, w, a in zip(headers, widths, aligns)
    ]
    lines = [sep.join(header_cells).rstrip()]
    for i in range(n_rows):
        row = sep.join(pad(formatted[j][i], widths[j], aligns[j]) for j in range(len(headers)))
        lines.append(row.rstrip())
    return "\n".join(lines)


def significance_code(p_values) -> list[str]:
    """R-style significance codes for an iterable of p-values."""
    out: list[str] = []
    for p in p_values:
        if p < 0.001:
            out.append("***")
        elif p < 0.01:
            out.append("**")
        elif p < 0.05:
            out.append("*")
        elif p < 0.1:
            out.append(".")
        else:
            out.append(" ")
    return out


_MACHINE_EPS = float(np.finfo(float).eps)  # ≈ 2.22e-16, matches R's `.Machine$double.eps`

# Magnitudes |x| < 10^_FIXED_LOWER print as scientific; otherwise fixed.
# Matches R's ``format()`` rule of switching to scientific when ``expo < -4``.
_FIXED_LOWER = -4


def _is_fixed(fv: float) -> bool:
    """Whether ``fv`` formats as fixed (vs. scientific) in R's ``format()``."""
    if fv == 0 or not np.isfinite(fv):
        return True
    return int(np.floor(np.log10(abs(fv)))) >= _FIXED_LOWER


def _signif_column_decimals(values, digits: int) -> int:
    """Joint decimal count over entries that print as fixed: chosen so the
    smallest-magnitude fixed entry gets ``digits`` significant figures
    (R's ``format(x, digits=)`` per-vector decimals).
    """
    fixed_nonzero = []
    for v in values:
        if v is None:
            continue
        fv = float(v)
        if np.isfinite(fv) and fv != 0 and _is_fixed(fv):
            fixed_nonzero.append(abs(fv))
    if not fixed_nonzero:
        return 0
    smallest = min(fixed_nonzero)
    return max(0, digits - 1 - int(np.floor(np.log10(smallest))))


def _format_with_decimals(values, decimals: int, digits: int) -> list[str]:
    """Per-element formatter: fixed at ``decimals`` for in-range entries,
    scientific (``digits-1`` after the leading digit) for entries below
    the fixed threshold or that would round to all-zeros at this decimal
    count.
    """
    sci_fmt = f".{max(digits - 1, 0)}e"
    out: list[str] = []
    for v in values:
        if v is None:
            out.append("")
            continue
        fv = float(v)
        if np.isnan(fv):
            out.append("NaN")
            continue
        if not np.isfinite(fv):
            out.append("Inf" if fv > 0 else "-Inf")
            continue
        rounds_to_zero = fv != 0 and abs(fv) < 0.5 * 10 ** (-decimals)
        if not _is_fixed(fv) or rounds_to_zero:
            out.append(format(fv, sci_fmt))
        else:
            out.append(f"{fv:.{decimals}f}")
    return out


def format_signif(values, digits: int = 4, *, min_decimals: int = 0) -> list[str]:
    """R-style ``format(x, digits=digits)`` for a numeric vector.

    Picks a single column-wide decimal count so the smallest-magnitude
    fixed-form entry gets ``digits`` significant figures, then applies
    it uniformly. Entries with ``|x| < 1e-4`` (or that would otherwise
    round to ``0.000…``) fall back to scientific independently — matches
    R's ``format()`` per-element fixed-vs-scientific switch.
    ``None`` → ``""``; non-finite → ``"NaN"``/``"Inf"``/``"-Inf"``.
    """
    decimals = max(min_decimals,
                   min(_signif_column_decimals(values, digits), 15))
    return _format_with_decimals(values, decimals, digits)


def format_signif_jointly(columns, digits: int = 4) -> list[list[str]]:
    """R's ``printCoefmat`` joint formatting for the coefficient + SE
    group: pool magnitudes across all input columns, pick one shared
    decimal count so the smallest pooled magnitude gets ``digits`` sig
    figs (with at least 1 decimal, matching ``max(1L, digits - digmin)``
    in R), and format each input column at that count.

    Returns a list of formatted-string columns, parallel to ``columns``.
    Use for Estimate / SE / CI together so a row like ``470.4444 4.0817``
    prints as ``470.444 4.082`` (decimals driven by the smaller value).
    """
    pool: list[float] = []
    for col in columns:
        for v in col:
            if v is None:
                continue
            fv = float(v)
            if np.isfinite(fv) and fv != 0 and _is_fixed(fv):
                pool.append(abs(fv))
    if not pool:
        decimals = 1
    else:
        decimals = max(1, digits - 1 - int(np.floor(np.log10(min(pool)))))
    decimals = min(decimals, 15)
    return [_format_with_decimals(col, decimals, digits) for col in columns]


def format_pval(values, digits: int = 4, eps: float | None = None) -> list[str]:
    """R-style ``format.pval`` for an iterable of p-values.

    Mirrors R's ``stats::format.pval(x, digits=digits, eps=eps)``:

    * Entries ``< eps`` (default ``.Machine$double.eps``) print as
      ``"<{eps_str}"``, where ``eps_str`` uses ``max(1, digits-2)`` sig
      figs (R further-reduces it if it would crowd the column, and picks
      a leading space iff the reduced digits ≠ 1 or the column is wide).
    * Other entries format with ``digits`` sig figs via ``format_signif``
      (per-element fixed-vs-scientific based on magnitude, joint decimals
      across fixed entries).

    Note: R's ``printCoefmat`` calls ``format.pval`` with a *reduced*
    ``digits = max(1, min(5, digits-1))`` (its ``dig.tst``); summary
    methods that want printCoefmat-faithful output should pass that
    reduced value here, not the raw user-facing ``digits``.
    """
    if eps is None:
        eps = _MACHINE_EPS
    big = [
        float(v) for v in values
        if v is not None and np.isfinite(float(v)) and float(v) >= eps
    ]
    big_strs = format_signif(big, digits=digits) if big else []

    eps_digits = max(1, digits - 2)
    if big_strs:
        nc = max(len(s) for s in big_strs)
        if eps_digits > 1 and eps_digits + 6 > nc:
            eps_digits = max(1, nc - 7)
        sep = "" if (eps_digits == 1 and nc <= 6) else " "
    else:
        sep = "" if eps_digits == 1 else " "
    eps_str = f"{eps:.{eps_digits}g}"
    eps_display = f"<{sep}{eps_str}"

    big_iter = iter(big_strs)
    out: list[str] = []
    for v in values:
        if v is None:
            out.append("")
            continue
        fv = float(v)
        if np.isnan(fv):
            out.append("NaN")
            continue
        if fv < eps:
            out.append(eps_display)
        else:
            out.append(next(big_iter))
    return out


def _dig_tst(digits: int) -> int:
    """R's ``printCoefmat`` reduction of ``digits`` for test-statistic /
    p-value columns: ``max(1, min(5, digits - 1))``.
    """
    return max(1, min(5, digits - 1))
