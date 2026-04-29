"""R-style ``stats`` package wrappers for use alongside :class:`hea.lm`.

The point isn't a parallel inference engine — it's so notebooks comparing
"common tests as linear models" (Lindeløv 2019) can write the same call
twice: once via the named test (``t_test``, ``cor_test``, …) and once via
``lm`` / ``glm``. Each function returns an :class:`HTest` whose ``__repr__``
mimics R's ``print.htest`` output, so the two paths look comparable in
the notebook.

Coverage matches the cheat-sheet rows: t.test, wilcox.test, cor.test, aov,
kruskal.test, chisq.test, plus ``signed_rank`` and ``rank`` helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import polars as pl
from scipy import stats as _sps

__all__ = [
    "HTest",
    "AnovaTable",
    "rank",
    "signed_rank",
    "t_test",
    "wilcox_test",
    "cor_test",
    "kruskal_test",
    "chisq_test",
    "aov",
]


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class HTest:
    """R's ``htest`` class as a Python dataclass.

    Mirrors ``stats:::print.htest``: ``method`` is the title, ``statistic``
    the named scalar, ``parameter`` the df line, plus optional p-value,
    CI, point ``estimate``, and ``alternative``. ``data_name`` is the
    "data:" label R prints before the stats.
    """

    method: str
    statistic: dict = field(default_factory=dict)
    parameter: dict = field(default_factory=dict)
    p_value: Optional[float] = None
    conf_int: Optional[tuple] = None
    estimate: dict = field(default_factory=dict)
    null_value: Optional[Union[float, dict]] = None
    alternative: str = "two.sided"
    data_name: str = ""
    conf_level: float = 0.95

    def __repr__(self) -> str:
        out = ["", f"\t{self.method}", ""]
        if self.data_name:
            out.append(f"data:  {self.data_name}")
        bits = []
        for k, v in self.statistic.items():
            bits.append(f"{k} = {_fmt(v)}")
        for k, v in self.parameter.items():
            bits.append(f"{k} = {_fmt(v)}")
        if self.p_value is not None:
            bits.append(f"p-value = {_fmt_pval(self.p_value)}")
        if bits:
            out.append(", ".join(bits))
        if self.alternative:
            null = self.null_value
            tail = "not equal to"
            if self.alternative == "greater":
                tail = "greater than"
            elif self.alternative == "less":
                tail = "less than"
            if isinstance(null, dict):
                null_str = ", ".join(f"{k} = {_fmt(v)}" for k, v in null.items())
                out.append(f"alternative hypothesis: true {null_str.split(' = ')[0]} is {tail} {null_str.split(' = ')[1]}")
            elif null is not None:
                # name from estimate keys when possible
                nm = next(iter(self.estimate.keys()), "value")
                out.append(f"alternative hypothesis: true {nm} is {tail} {_fmt(null)}")
        if self.conf_int is not None:
            out.append(f"{int(self.conf_level * 100)} percent confidence interval:")
            out.append(f" {_fmt(self.conf_int[0])} {_fmt(self.conf_int[1])}")
        if self.estimate:
            out.append("sample estimates:")
            keys = "  ".join(f"{k}" for k in self.estimate)
            vals = "  ".join(f"{_fmt(v)}" for v in self.estimate.values())
            out.append(keys)
            out.append(vals)
        return "\n".join(out) + "\n"


@dataclass
class AnovaTable:
    """R-style ``Anova`` / ``anova`` table (Type-II by default for ``aov``).

    Stored as a list of rows (term, df, sum_sq, mean_sq, F, p) plus a
    Residuals row. ``__repr__`` formats it close to R's printout.
    """

    response: str
    rows: list  # list of dicts: term, df, sum_sq, mean_sq, F, p
    residual_df: int
    residual_ss: float
    type: str = "II"

    def __repr__(self) -> str:
        out = [f"Anova Table (Type {self.type} tests)", "",
               f"Response: {self.response}",
               f"{'':<12}{'Sum Sq':>10}{'Df':>4}{'F value':>10}{'Pr(>F)':>12}"]
        for r in self.rows:
            out.append(
                f"{r['term']:<12}{_fmt(r['sum_sq']):>10}{r['df']:>4}"
                f"{_fmt(r['F']):>10}{_fmt_pval(r['p']):>12}"
            )
        out.append(
            f"{'Residuals':<12}{_fmt(self.residual_ss):>10}{self.residual_df:>4}"
        )
        return "\n".join(out)


# ---------------------------------------------------------------------------
# Formatters (light — model summaries handle the heavy lifting)
# ---------------------------------------------------------------------------


def _fmt(x) -> str:
    if x is None:
        return ""
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    fx = float(x)
    if not np.isfinite(fx):
        return str(fx)
    ax = abs(fx)
    if ax != 0 and (ax < 1e-4 or ax >= 1e5):
        return f"{fx:.5g}"
    return f"{fx:.5g}"


def _fmt_pval(p: float) -> str:
    if p is None:
        return ""
    if p < 2.2e-16:
        return "< 2.2e-16"
    return _fmt(p)


def _as_array(x) -> np.ndarray:
    if isinstance(x, pl.Series):
        return x.to_numpy().astype(float)
    return np.asarray(x, dtype=float)


# ---------------------------------------------------------------------------
# Rank helpers (R's ``rank`` and Lindeløv's ``signed_rank``)
# ---------------------------------------------------------------------------


def rank(x) -> np.ndarray:
    """R's ``rank()`` with ``ties.method = "average"`` (R's default).

    Returns a float array so downstream lm() formulas treat it as numeric.
    """
    return _sps.rankdata(_as_array(x), method="average")


def signed_rank(x) -> np.ndarray:
    """Lindeløv's ``signed_rank = function(x) sign(x) * rank(abs(x))``.

    Used to turn Wilcoxon signed-rank into an intercept-only ``lm``.
    """
    arr = _as_array(x)
    return np.sign(arr) * _sps.rankdata(np.abs(arr), method="average")


# ---------------------------------------------------------------------------
# t.test
# ---------------------------------------------------------------------------


def t_test(
    x,
    y=None,
    *,
    paired: bool = False,
    var_equal: bool = True,
    mu: float = 0.0,
    alternative: str = "two.sided",
    conf_level: float = 0.95,
) -> HTest:
    """R's ``t.test``.

    - ``y=None``                     → one-sample t-test on ``x``.
    - ``y`` given, ``paired=True``   → paired t-test on ``x - y``.
    - ``y`` given, ``var_equal=True``→ Student's two-sample (pooled var).
    - ``y`` given, ``var_equal=False``→ Welch's two-sample.

    Always uses Student-t CI on the appropriate df.
    """
    alt = {"two.sided": "two-sided", "greater": "greater", "less": "less"}[alternative]
    x = _as_array(x)
    if y is None:
        res = _sps.ttest_1samp(x, mu, alternative=alt)
        ci = res.confidence_interval(conf_level)
        return HTest(
            method="One Sample t-test",
            statistic={"t": float(res.statistic)},
            parameter={"df": float(res.df)},
            p_value=float(res.pvalue),
            conf_int=(float(ci.low), float(ci.high)),
            estimate={"mean of x": float(np.mean(x))},
            null_value=mu,
            alternative=alternative,
            conf_level=conf_level,
            data_name="x",
        )
    y = _as_array(y)
    if paired:
        d = x - y
        res = _sps.ttest_1samp(d, mu, alternative=alt)
        ci = res.confidence_interval(conf_level)
        return HTest(
            method="Paired t-test",
            statistic={"t": float(res.statistic)},
            parameter={"df": float(res.df)},
            p_value=float(res.pvalue),
            conf_int=(float(ci.low), float(ci.high)),
            estimate={"mean of the differences": float(np.mean(d))},
            null_value=mu,
            alternative=alternative,
            conf_level=conf_level,
            data_name="x and y",
        )
    res = _sps.ttest_ind(x, y, equal_var=var_equal, alternative=alt)
    ci = res.confidence_interval(conf_level)
    method = "Two Sample t-test" if var_equal else "Welch Two Sample t-test"
    return HTest(
        method=method,
        statistic={"t": float(res.statistic)},
        parameter={"df": float(res.df)},
        p_value=float(res.pvalue),
        conf_int=(float(ci.low), float(ci.high)),
        estimate={"mean of x": float(np.mean(x)), "mean of y": float(np.mean(y))},
        null_value=mu,
        alternative=alternative,
        conf_level=conf_level,
        data_name="x and y",
    )


# ---------------------------------------------------------------------------
# wilcox.test
# ---------------------------------------------------------------------------


def wilcox_test(
    x,
    y=None,
    *,
    paired: bool = False,
    alternative: str = "two.sided",
    correct: bool = True,
) -> HTest:
    """R's ``wilcox.test``.

    Defaults to continuity correction (``correct=True``). One-sample and
    paired branches use ``scipy.stats.wilcoxon``; the two-sample branch
    uses ``mannwhitneyu`` (R's "Wilcoxon rank-sum" with W statistic).
    """
    alt = {"two.sided": "two-sided", "greater": "greater", "less": "less"}[alternative]
    x = _as_array(x)
    if y is None:
        res = _sps.wilcoxon(x, alternative=alt, correction=correct, zero_method="wilcox")
        return HTest(
            method="Wilcoxon signed rank test"
            + (" with continuity correction" if correct else ""),
            statistic={"V": float(res.statistic)},
            p_value=float(res.pvalue),
            null_value=0.0,
            alternative=alternative,
            data_name="x",
        )
    y = _as_array(y)
    if paired:
        res = _sps.wilcoxon(x, y, alternative=alt, correction=correct, zero_method="wilcox")
        return HTest(
            method="Wilcoxon signed rank test"
            + (" with continuity correction" if correct else ""),
            statistic={"V": float(res.statistic)},
            p_value=float(res.pvalue),
            null_value=0.0,
            alternative=alternative,
            data_name="x and y",
        )
    res = _sps.mannwhitneyu(
        x, y, alternative=alt, use_continuity=correct, method="asymptotic"
    )
    return HTest(
        method="Wilcoxon rank sum test"
        + (" with continuity correction" if correct else ""),
        statistic={"W": float(res.statistic)},
        p_value=float(res.pvalue),
        null_value=0.0,
        alternative=alternative,
        data_name="x and y",
    )


# ---------------------------------------------------------------------------
# cor.test
# ---------------------------------------------------------------------------


def cor_test(
    x,
    y,
    *,
    method: str = "pearson",
    alternative: str = "two.sided",
    conf_level: float = 0.95,
) -> HTest:
    """R's ``cor.test`` with ``method`` in {pearson, spearman, kendall}.

    For Pearson, we report ``t``, df = n-2, and Fisher-z CI. Spearman
    reports ``S`` (rank-sum statistic R's ``cor.test`` shows). Kendall
    reports ``z``.
    """
    alt = {"two.sided": "two-sided", "greater": "greater", "less": "less"}[alternative]
    x = _as_array(x)
    y = _as_array(y)
    if len(x) != len(y):
        raise ValueError("'x' and 'y' must have the same length")
    n = len(x)
    if method == "pearson":
        res = _sps.pearsonr(x, y, alternative=alt)
        r = float(res.statistic)
        df = n - 2
        t = r * np.sqrt(df / max(1 - r * r, 1e-300))
        ci = res.confidence_interval(conf_level)
        return HTest(
            method="Pearson's product-moment correlation",
            statistic={"t": t},
            parameter={"df": df},
            p_value=float(res.pvalue),
            conf_int=(float(ci.low), float(ci.high)),
            estimate={"cor": r},
            null_value=0.0,
            alternative=alternative,
            conf_level=conf_level,
            data_name="x and y",
        )
    if method == "spearman":
        res = _sps.spearmanr(x, y, alternative=alt)
        rho = float(res.statistic)
        # R reports S = sum((rank(x) - rank(y))^2) for the Spearman test
        S = float(np.sum((_sps.rankdata(x) - _sps.rankdata(y)) ** 2))
        return HTest(
            method="Spearman's rank correlation rho",
            statistic={"S": S},
            p_value=float(res.pvalue),
            estimate={"rho": rho},
            null_value=0.0,
            alternative=alternative,
            data_name="x and y",
        )
    if method == "kendall":
        res = _sps.kendalltau(x, y, alternative=alt)
        return HTest(
            method="Kendall's rank correlation tau",
            statistic={"z": float(res.statistic)},
            p_value=float(res.pvalue),
            estimate={"tau": float(res.statistic)},
            null_value=0.0,
            alternative=alternative,
            data_name="x and y",
        )
    raise ValueError(f"unknown method: {method}")


# ---------------------------------------------------------------------------
# kruskal.test
# ---------------------------------------------------------------------------


def kruskal_test(formula: str, data: pl.DataFrame) -> HTest:
    """R's ``kruskal.test(y ~ group, data)``.

    Only the formula form is supported here — that's what the notebook
    uses. The numeric LHS is grouped by the RHS factor and passed to
    ``scipy.stats.kruskal``.
    """
    if "~" not in formula:
        raise ValueError("formula must look like 'y ~ group'")
    lhs, rhs = [s.strip() for s in formula.split("~", 1)]
    groups = [
        data.filter(pl.col(rhs) == g)[lhs].to_numpy().astype(float)
        for g in data[rhs].unique().to_list()
    ]
    res = _sps.kruskal(*groups)
    return HTest(
        method="Kruskal-Wallis rank sum test",
        statistic={"Kruskal-Wallis chi-squared": float(res.statistic)},
        parameter={"df": int(len(groups) - 1)},
        p_value=float(res.pvalue),
        alternative="",
        data_name=f"{lhs} by {rhs}",
    )


# ---------------------------------------------------------------------------
# chisq.test
# ---------------------------------------------------------------------------


def chisq_test(
    x,
    y=None,
    *,
    p=None,
    correct: bool = True,
) -> HTest:
    """R's ``chisq.test``.

    - 1-D ``x`` (and no ``y``)         → goodness-of-fit against ``p`` (uniform if None).
    - 2-D ``x`` (matrix or 2-D array)  → contingency-table test.
    - 1-D ``x`` and 1-D ``y``          → contingency on ``crosstab(x, y)``.
    """
    arr = np.asarray(x)
    if y is not None:
        # tabulate
        x_ser = pl.Series("x", x).cast(pl.Utf8)
        y_ser = pl.Series("y", y).cast(pl.Utf8)
        tbl = (
            pl.DataFrame({"x": x_ser, "y": y_ser})
            .group_by(["x", "y"]).len()
            .pivot(values="len", index="x", on="y")
            .fill_null(0)
            .drop("x")
            .to_numpy()
        )
        return _chisq_table(tbl, correct=correct, name="x and y")
    if arr.ndim == 2:
        return _chisq_table(arr, correct=correct, name="x")
    # goodness of fit
    counts = arr.astype(float)
    if p is None:
        p = np.full_like(counts, 1.0 / len(counts))
    p = np.asarray(p, dtype=float)
    expected = counts.sum() * p
    stat = float(np.sum((counts - expected) ** 2 / expected))
    df = len(counts) - 1
    pval = float(_sps.chi2.sf(stat, df))
    return HTest(
        method="Chi-squared test for given probabilities",
        statistic={"X-squared": stat},
        parameter={"df": df},
        p_value=pval,
        alternative="",
        data_name="x",
    )


def _chisq_table(tbl: np.ndarray, *, correct: bool, name: str) -> HTest:
    res = _sps.chi2_contingency(tbl, correction=(correct and tbl.shape == (2, 2)))
    return HTest(
        method="Pearson's Chi-squared test"
        + (" with Yates' continuity correction" if (correct and tbl.shape == (2, 2)) else ""),
        statistic={"X-squared": float(res.statistic)},
        parameter={"df": int(res.dof)},
        p_value=float(res.pvalue),
        alternative="",
        data_name=name,
    )


# ---------------------------------------------------------------------------
# aov — a thin wrapper over hea.lm that produces an Anova table
# ---------------------------------------------------------------------------


def aov(formula: str, data: pl.DataFrame, *, type: str = "II") -> AnovaTable:
    """R's ``aov`` followed by ``car::Anova(..., type='II')``.

    Computes Type-II sums of squares by dropping one top-level term at a
    time and comparing ``ΔRSS``. Works for either form the notebook uses:
    factor formulas (``value ~ group``) or explicit-dummy formulas
    (``value ~ 1 + group_b + group_c``) — both go through ``hea.lm``,
    so the term grouping comes from the formula's own ``term_labels``.
    """
    from .lm import lm  # local import to avoid circular at package load

    fit = lm(formula, data)
    term_labels = list(fit._expanded.term_labels)
    rss_full = float(fit.rss)
    df_full = int(fit.df_residuals)

    lhs = formula.split("~", 1)[0]
    rows = []
    for term in term_labels:
        kept = [t for t in term_labels if t != term]
        reduced_rhs = " + ".join(["1"] + kept) if kept else "1"
        sub_formula = f"{lhs} ~ {reduced_rhs}"
        sub = lm(sub_formula, data)
        ss = float(sub.rss - rss_full)
        df_term = int(sub.df_residuals - df_full)
        F = (ss / df_term) / (rss_full / df_full) if df_term > 0 else None
        p = float(_sps.f.sf(F, df_term, df_full)) if F is not None else None
        rows.append(
            {
                "term": term,
                "df": df_term,
                "sum_sq": ss,
                "mean_sq": ss / df_term if df_term else None,
                "F": F,
                "p": p,
            }
        )
    return AnovaTable(
        response=fit.y.name,
        rows=rows,
        residual_df=df_full,
        residual_ss=rss_full,
        type=type,
    )
