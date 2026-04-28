from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.linalg import cholesky, lu, qr, solve_triangular
from scipy.optimize import minimize
from scipy.stats import f, norm, t

from .formula import _eval_atom, materialize
from .design import prepare_design
from .utils import (
    _dig_tst,
    format_df,
    format_pval,
    format_signif,
    format_signif_jointly,
    significance_code,
)

__all__ = ["lm"]


def _row_frame(values: np.ndarray, columns: list[str]) -> pl.DataFrame:
    """Build a 1-row pl.DataFrame from a flat numpy array + column names."""
    flat = np.asarray(values).reshape(-1)
    return pl.DataFrame({c: [float(flat[i])] for i, c in enumerate(columns)})


def _lowess(x, y, frac=2 / 3, it=3):
    """Cleveland's LOWESS — local linear smoother with robustness reweighting.

    Matches the smoother R uses in `panel.smooth` for `plot.lm`.
    Returns (x_sorted, y_smooth). O(n^2) memory; pass `smooth=False` if n is huge.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 4:
        return x, y
    idx = np.argsort(x)
    xs, ys = x[idx], y[idx]
    r = max(2, int(np.ceil(frac * n)))

    dists = np.abs(xs[:, None] - xs[None, :])
    bw = np.partition(dists, r - 1, axis=1)[:, r - 1]
    bw = np.where(bw == 0, 1.0, bw)
    W = np.clip(dists / bw[:, None], 0, 1)
    W = (1 - W ** 3) ** 3

    yhat = ys.copy()
    delta = np.ones(n)
    for _ in range(it + 1):
        Wd = W * delta[None, :]
        S = Wd.sum(axis=1)
        Sx = Wd @ xs
        Sxx = Wd @ (xs * xs)
        Sy = Wd @ ys
        Sxy = Wd @ (xs * ys)
        det = S * Sxx - Sx * Sx
        ok = det > 0
        safe_det = np.where(ok, det, 1.0)
        safe_S = np.where(S > 0, S, 1.0)
        a_local = (Sxx * Sy - Sx * Sxy) / safe_det
        b_local = (S * Sxy - Sx * Sy) / safe_det
        wmean = Sy / safe_S
        yhat = np.where(ok, a_local + b_local * xs, wmean)
        resid = ys - yhat
        s = np.median(np.abs(resid))
        if s == 0:
            break
        u = np.clip(resid / (6 * s), -1, 1)
        delta = (1 - u * u) ** 2
    return xs, yhat


def _label_top_n(ax, xs, ys, scores, n=3, indices=None):
    """Annotate the n points with the largest |scores|."""
    if not n:
        return
    n = min(int(n), len(scores))
    if n == 0:
        return
    top = np.argsort(-np.abs(np.asarray(scores)))[:n]
    for i in top:
        label = str(int(indices[i]) if indices is not None else int(i))
        ax.annotate(
            label,
            (xs[i], ys[i]),
            fontsize=8,
            color="black",
            xytext=(3, 3),
            textcoords="offset points",
        )


def _qq_plot(
    ax, vals, labels=None, label_n=3,
    xlabel="Theoretical Quantiles",
    ylabel="Standardized Residuals",
    title="Normal Q-Q",
):
    """Normal Q-Q on ax with quartile-based reference line, label top |vals|.

    `labels` optionally maps an index to a custom annotation string;
    otherwise the integer index is used.
    """
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    if n < 2:
        return
    sort_idx = np.argsort(vals)
    v = vals[sort_idx]
    a = 3.0 / 8.0 if n <= 10 else 0.5
    probs = (np.arange(1, n + 1) - a) / (n + 1 - 2 * a)
    q = norm.ppf(probs)
    ax.scatter(q, v, facecolor="none", edgecolor="black")
    ry1, ry3 = np.quantile(v, [0.25, 0.75])
    qx1, qx3 = norm.ppf([0.25, 0.75])
    slope = (ry3 - ry1) / (qx3 - qx1)
    intercept = ry1 - slope * qx1
    xs = np.array([q.min(), q.max()])
    ax.plot(xs, slope * xs + intercept, color="black", linestyle="--")
    if label_n:
        n_lab = min(int(label_n), n)
        top = np.argsort(-np.abs(vals))[:n_lab]
        rank = np.empty(n, dtype=int)
        rank[sort_idx] = np.arange(n)
        for orig_i in top:
            pos = rank[orig_i]
            text = str(labels[orig_i]) if labels is not None else str(int(orig_i))
            ax.annotate(
                text, (q[pos], v[pos]),
                fontsize=8, color="black",
                xytext=(3, 3), textcoords="offset points",
            )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


class lm:
    def __init__(
        self,
        formula: str,
        data: pl.DataFrame,
        weights: Union[None, np.array] = None,
        method: str = "qr",
    ):

        # meta
        self.formula = formula
        self.data = data
        self.weights = weights
        self.method = method

        d = prepare_design(formula, data)
        self._expanded = d.expanded
        self._design_data = d.data
        self.X = d.X
        self.y = d.y  # pl.Series

        self.column_names = list(self.X.columns)
        self.feature_names = (
            self.column_names[1:]
            if "(Intercept)" in self.column_names
            else self.column_names
        )

        X = self.X.to_numpy().astype(float)
        y = self.y.to_numpy().astype(float).flatten()

        self.n, self.p = (
            n,
            p,
        ) = X.shape  # n_samples x n_features (intercept included if available)
        if weights is not None and len(weights) != n:
            raise ValueError(
                "Length of weights should be the same as the number of rows in the dataframe"
            )
        self.W = W = np.eye(n) if weights is None else np.diag(weights)

        # model degree of freedom
        self.df_model = self.p - 1 if "(Intercept)" in self.column_names else self.p

        # residual degrees of freedom (n - p)
        self.df_residuals = (
            self.n - self.df_model - 1
            if "(Intercept)" in self.column_names
            else self.n - self.df_model
        )

        # total parameter count (p fixed + 1 residual variance), for the
        # generic AIC() comparison table and AIC/BIC formulas below.
        self.npar = self.p + 1

        # Sum any `offset(...)` atoms from the formula. R's lm() solves
        # (y - offset) ~ X, then adds the offset back to ŷ — so β̂ has the
        # same df as without offset. expanded.offsets holds the inner ASTs.
        off = np.zeros(n)
        for off_node in d.expanded.offsets:
            off = off + _eval_atom(off_node, d.data).values.flatten().astype(float)
        self._offset = off
        y_solve = y - off

        ##############
        # Estimation #
        ##############

        if method == "nll":
            bhat, self.sigma, self.XtX, self.Xty, self.loss = self.compute_bhat(
                X, y_solve, W, "nll"
            )
        elif method == "sse":
            bhat, self.XtX, self.Xty, self.loss = self.compute_bhat(X, y_solve, W, "sse")
        else:
            bhat, self.XtX, self.Xty = self.compute_bhat(X, y_solve, W, method)

        self._bhat_arr = np.asarray(bhat).reshape(-1)
        self.bhat = _row_frame(self._bhat_arr, self.column_names)

        # compute predicted (fitted values ŷ = Xβ̂)
        self.yhat = self.compute_yhat()
        yhat = self.yhat["Fitted"].to_numpy().astype(float)

        # compute residuals ϵ̂
        residuals = y - yhat
        self._residuals_arr = residuals
        self.residuals = pl.DataFrame({"residuals": residuals})

        # compute residual sum of squares (RSS)
        self.rss = float(residuals @ residuals)

        # compute standard deviation of model coefficients
        # aka Residual SE: σ^2 = RSS / df_residuals
        if method == "nll":
            self.sigma_squared = self.sigma**2
        else:
            self.sigma_squared = self.rss / self.df_residuals
            self.sigma = np.sqrt(self.sigma_squared)

        # compute standard error for β̂
        self.XtXinv = self.compute_XtXinv()

        se_bhat, V_bhat = self.compute_se_bhat()
        self.V_bhat = V_bhat
        self._se_bhat_arr = np.asarray(se_bhat).reshape(-1)
        self.se_bhat = _row_frame(self._se_bhat_arr, self.column_names)

        # hat-matrix diagonal h_ii (leverages) and internally studentized
        # residuals — cached once so plot_qq / plot_scale_location /
        # plot_leverage don't each recompute them.
        HX = X @ self.XtXinv
        h = (HX * X).sum(axis=1)
        w = np.diag(W)
        self.leverage = h * w
        denom = self.sigma * np.sqrt(np.clip(1.0 - self.leverage, 1e-12, None))
        self.std_residuals = residuals * np.sqrt(w) / denom

        # compute confidence interval for β̂
        self.ci_bhat = self.compute_ci_bhat()

        # compute t values of model coefficients
        self.t_values = self.compute_t_values()

        # p values
        self.p_values = self.compute_p_values()

        # compute r2 and r2adjusted, aka coefficient of determination
        # aka percentage of variance explained. Noted that the formulae
        # are different for cases with and without intercept
        (
            self.tss,
            self.r_squared,
            self.r_squared_adjusted,
        ) = self.compute_goodness_of_fit()

        # compute F-statistics with scipy.stats.f.sf
        # H0: all coefficients == 0
        # H1: at least one coefficient != 0
        self.fstats, self.f_p_value = self.compute_fstats()

        # compute log-likelihood
        self.loglike = self.compute_loglikelihood()

        # compute AIC (Akaike Information criterion): -2logL + 2p, p is the total number of parameters
        self.AIC = self.compute_AIC()

        # compute BIC (Bayes Information criterian): -2logL + p * log(n)
        self.BIC = self.compute_BIC()

    def __repr__(self):

        docstring = f"""Formula: {self.formula}\n\n"""
        docstring += "Coefficients:\n"
        docstring += format_df(self.bhat)

        return docstring

    def __str__(self):

        return self.__repr__()

    def compute_XtXinv(self):
        U, S, Vt = np.linalg.svd(self.XtX, full_matrices=False)
        XtXinv = Vt.T @ np.diag(1 / S) @ U.T
        return XtXinv

    def compute_bhat(self, X, y, W, method="qr", return_ss=True):

        match method:
            case "qr":
                bhat, XtX, Xty = _qr(X, y, W)
            case "lu":
                bhat, XtX, Xty = _lu(X, y, W)
            case "chol":
                bhat, XtX, Xty = _chol(X, y)
            case "svd":
                bhat, XtX, Xty = _svd(X, y)
            case "nq":
                bhat, XtX, Xty = _nq(X, y)
            case "sse":
                bhat, loss = _sse(X, y)
                if return_ss:
                    XtX = X.T @ X
                    Xty = X.T @ y
                    return bhat, XtX, Xty, loss
                else:
                    return bhat
            case "nll":
                bhat, sigma, loss = _nll(X, y)

                if return_ss:
                    XtX = X.T @ X
                    Xty = X.T @ y
                    return bhat, sigma, XtX, Xty, loss
                else:
                    return bhat
            case _:
                raise ValueError("Please enter a valid method.")

        if return_ss:
            return bhat, XtX, Xty
        else:
            return bhat

    def compute_se_bhat(self):
        V_bhat = self.sigma_squared * self.XtXinv
        se_bhat = np.sqrt(np.diag(V_bhat))[:, None]
        return se_bhat, V_bhat

    def compute_ci_bhat(self, alpha=0.05):

        se_bhat = self._se_bhat_arr[:, None]
        bhat = self._bhat_arr[:, None]
        ci = (
            t.ppf(1 - alpha / 2, self.df_residuals) * se_bhat * np.array([-1, 1]) + bhat
        )
        return pl.DataFrame(
            {
                "coef": self.column_names,
                f"CI[{alpha/2*100}%]": ci[:, 0],
                f"CI[{100-alpha/2*100}]%": ci[:, 1],
            }
        )

    def compute_ci_bhat_bootstrap(self, num_bootstrap=4000, alpha=0.05):

        X = self.X.to_numpy().astype(float)
        W = self.W
        bhat = self._bhat_arr[:, None]
        residuals = self._residuals_arr
        bhat_stars = np.zeros([num_bootstrap, self.p])
        for i in range(num_bootstrap):
            residuals_star = np.random.choice(
                residuals, size=len(residuals), replace=True
            )
            y_star = X @ bhat + residuals_star[:, None]
            bhat_star = self.compute_bhat(X, y_star.flatten(), W, return_ss=False)
            bhat_stars[i] = bhat_star

        quantiles = np.quantile(
            bhat_stars, q=[alpha / 2, 1 - alpha / 2], axis=0
        ).T
        ci_bhat_bootstrap = pl.DataFrame(
            {
                "coef": self.column_names,
                f"CI[{alpha/2*100}%]": quantiles[:, 0],
                f"CI[{100-alpha/2*100}]%": quantiles[:, 1],
            }
        )
        self.ci_bhat_bootstrap = ci_bhat_bootstrap
        self.bhat_bootstrap = pl.DataFrame(
            {c: bhat_stars[:, i] for i, c in enumerate(self.column_names)}
        )

        return ci_bhat_bootstrap

    def compute_yhat(self, Xnew=None, interval=None, alpha=0.05):
        if Xnew is None:
            X = self.X.to_numpy().astype(float)
            off = self._offset
        else:
            X = materialize(self._expanded, Xnew).to_numpy().astype(float)
            # Re-evaluate formula offsets against newdata, mirroring R's
            # predict.lm. Offsets are zero unless the formula uses offset(...).
            off = np.zeros(X.shape[0])
            for off_node in self._expanded.offsets:
                off = off + _eval_atom(off_node, Xnew).values.flatten().astype(float)
        # compute predicted or fitted values ŷ = Xβ̂ + offset
        bhat = self._bhat_arr[:, None]
        yhat_vals = (X @ bhat).flatten() + off
        yhat = pl.DataFrame({"Fitted": yhat_vals})

        match interval:
            case None:
                return yhat
            case True:
                ci_yhat = self.compute_ci_yhat(yhat, Xnew, alpha)
                pi_yhat = self.compute_pi_yhat(yhat, Xnew, alpha)
                return pl.concat([yhat, ci_yhat, pi_yhat], how="horizontal")
            case "prediction":
                pi_yhat = self.compute_pi_yhat(yhat, Xnew, alpha)
                return pl.concat([yhat, pi_yhat], how="horizontal")
            case "confidence":
                ci_yhat = self.compute_ci_yhat(yhat, Xnew, alpha)
                return pl.concat([yhat, ci_yhat], how="horizontal")
            case _:
                raise ValueError(
                    "Please enter a valid value: [None, True, 'prediction', 'confidence']"
                )

    def compute_ci_yhat(self, yhat, Xnew=None, alpha=0.05):

        if Xnew is None:
            X = self.X.to_numpy().astype(float)
        else:
            X = materialize(self._expanded, Xnew).to_numpy().astype(float)

        # Var(ŷ) = σ² · x'(X'X)⁻¹x  ⇒  se = √diag(σ² · X(X'X)⁻¹Xᵀ).
        var_mean = np.einsum("ij,jk,ik->i", X, self.XtXinv, X) * self.sigma_squared
        se_yhat_mean = np.sqrt(np.maximum(var_mean, 0.0))
        yhat_vals = yhat["Fitted"].to_numpy().astype(float)[:, None]
        ci = (
            t.ppf(1 - alpha / 2, self.df_residuals)
            * se_yhat_mean[:, None]
            * np.array([-1, 1])
            + yhat_vals
        )
        return pl.DataFrame(
            {
                f"CI[{alpha/2*100}%]": ci[:, 0],
                f"CI[{100-alpha/2*100}]%": ci[:, 1],
            }
        )

    def compute_pi_yhat(self, yhat, Xnew=None, alpha=0.05):

        if Xnew is None:
            X = self.X.to_numpy().astype(float)
        else:
            X = materialize(self._expanded, Xnew).to_numpy().astype(float)

        # Var(y_new − ŷ) = σ²·(1 + x'(X'X)⁻¹x).
        var_mean = np.einsum("ij,jk,ik->i", X, self.XtXinv, X) * self.sigma_squared
        se_yhat = np.sqrt(self.sigma_squared + np.maximum(var_mean, 0.0))
        yhat_vals = yhat["Fitted"].to_numpy().astype(float)[:, None]
        pi = (
            t.ppf(1 - alpha / 2, self.df_residuals)
            * se_yhat[:, None]
            * np.array([-1, 1])
            + yhat_vals
        )
        return pl.DataFrame(
            {
                f"PI[{alpha/2*100}%]": pi[:, 0],
                f"PI[{100-alpha/2*100}]%": pi[:, 1],
            }
        )

    def compute_t_values(self):

        t_values = self._bhat_arr / self._se_bhat_arr

        return _row_frame(t_values, self.column_names)

    def compute_p_values(self):
        # compute p values of model coefficients with scipy.stats.t.sf
        # H0: βi==0
        # H1: βi!=0
        t_arr = self._bhat_arr / self._se_bhat_arr
        p_values = 2 * t.sf(np.abs(t_arr), self.df_residuals)
        return _row_frame(p_values, self.column_names)

    def compute_goodness_of_fit(self):

        y = self.y.to_numpy().astype(float)

        if "(Intercept)" in self.column_names:
            tss = np.sum((y - y.mean()) ** 2)
            # Eq: r2 = 1 - RSS / TSS = 1 -  sum((ŷ - yi)**2) / sum((y - ȳ)**2)
            r_squared = float(1 - self.rss / tss)
            # Eq: r2adj = 1 - (1 - r2) * (n - 1) / df_residuals
            r_squared_adjusted = 1 - (1 - r_squared) * (self.n - 1) / (
                self.df_residuals
            )
        else:
            tss = np.sum(y**2)
            # Eq: r2 = 1 - RSS / TSS = 1 -  sum((ŷ - yi)**2) / sum((y)**2)
            r_squared = float(1 - self.rss / tss)
            # Eq: r2adj = 1 - (1 - r2) * n / df_residuals
            r_squared_adjusted = 1 - (1 - r_squared) * self.n / (self.df_residuals)

        return tss, r_squared, r_squared_adjusted

    def compute_fstats(self):
        if self.df_model != 0:
            fstats = float(
                ((self.tss - self.rss) / self.df_model) / (self.rss / self.df_residuals)
            )
            f_p_value = float(f.sf(fstats, self.df_model, self.df_residuals))
        else:
            fstats, f_p_value = None, None
        return fstats, f_p_value

    def compute_loglikelihood(self):
        return float(
            -0.5 * self.n * (np.log(self.rss / self.n) + np.log(2 * np.pi) + 1)
        )

    def compute_AIC(self):
        # npar = p + 1 (residual variance) — matches R, see
        # https://stackoverflow.com/q/37917437
        return -2 * self.loglike + 2 * self.npar

    def compute_BIC(self):
        return -2 * self.loglike + np.log(self.n) * self.npar

    def predict(self, new=None, interval=None, alpha=0.05):
        return self.compute_yhat(Xnew=new, interval=interval, alpha=alpha)

    def _residuals_lines(self, digits: int = 4) -> list[str]:
        qs = np.quantile(self._residuals_arr, [0.0, 0.25, 0.5, 0.75, 1.0])
        labels = ["Min", "1Q", "Median", "3Q", "Max"]
        vals = format_signif(qs, digits=digits)
        widths = [max(len(l), len(v)) for l, v in zip(labels, vals)]
        hdr = " ".join(l.rjust(w) for l, w in zip(labels, widths))
        row = " ".join(v.rjust(w) for v, w in zip(vals, widths))
        return ["Residuals:", hdr, row]

    def _correlation_block(self) -> str:
        # cov2cor(vcov(m)) — correlation between coefficient estimates,
        # including the intercept. Layout mirrors R's print.summary.lm:
        # strict lower triangle, first row and last column dropped.
        sd = np.sqrt(np.diag(self.V_bhat))
        corr = self.V_bhat / np.outer(sd, sd)
        names = self.column_names
        rows, cols = names[1:], names[:-1]
        # R's format() pads non-negatives with a leading space iff the
        # displayed values contain any negative (so signs line up).
        tri_vals = [corr[i + 1, j] for i in range(len(rows)) for j in range(i + 1)]
        pad_pos = any(v < 0 for v in tri_vals)
        def fmt(v: float) -> str:
            s = f"{v:.2f}"
            return s if s.startswith("-") or not pad_pos else " " + s
        cells = [
            [fmt(corr[i + 1, j]) if j <= i else "" for j in range(len(cols))]
            for i in range(len(rows))
        ]
        col_w = [
            max(len(cols[j]), max(len(cells[i][j]) for i in range(len(rows))))
            for j in range(len(cols))
        ]
        row_w = max(len(r) for r in rows)
        hdr = " " * row_w + " " + " ".join(c.ljust(w) for c, w in zip(cols, col_w))
        lines = [hdr.rstrip()]
        for i, r in enumerate(rows):
            line = r.ljust(row_w) + " " + " ".join(
                cells[i][j].ljust(col_w[j]) for j in range(len(cols))
            )
            lines.append(line.rstrip())
        return "\n".join(lines)

    def summary(self, digits=4, cor=False):

        docstring = f"""Formula: {self.formula}\n\n"""
        docstring += "\n".join(self._residuals_lines(digits=digits))
        docstring += "\n\nCoefficients:\n"

        t_arr = self._bhat_arr / self._se_bhat_arr
        p_arr = np.asarray(self.p_values.row(0), dtype=float)
        sig = significance_code(p_arr)
        ci_low_col, ci_hi_col = self.ci_bhat.columns[1], self.ci_bhat.columns[2]
        ci_low_arr = self.ci_bhat[ci_low_col].to_numpy()
        ci_hi_arr = self.ci_bhat[ci_hi_col].to_numpy()
        # Joint format Estimate+SE so the smaller-magnitude column drives
        # decimals (mirrors R's `printCoefmat` cs.ind block). CI columns
        # join their own group — they often have smaller magnitudes than
        # Estimate/SE and would otherwise force extra decimals on those.
        est_s, se_s = format_signif_jointly(
            [self._bhat_arr, self._se_bhat_arr], digits=digits,
        )
        cilo_s, cihi_s = format_signif_jointly(
            [ci_low_arr, ci_hi_arr], digits=digits,
        )
        res = pl.DataFrame(
            {
                "": self.column_names,
                "Estimate": est_s,
                "Std. Error": se_s,
                ci_low_col: cilo_s,
                ci_hi_col: cihi_s,
                "t value": format_signif(t_arr, digits=digits),
                "Pr(>|t|)": format_pval(p_arr, digits=_dig_tst(digits)),
                " ": sig,
            }
        )
        self.results = res

        num_align = {c: "right" for c in
                     ("Estimate", "Std. Error", ci_low_col, ci_hi_col,
                      "t value", "Pr(>|t|)")}
        docstring += format_df(res, align=num_align)
        docstring += "\n---"
        docstring += "\nSignif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1"

        docstring += f"\n\nn = {self.n}, p = {self.p}, Residual SE = {np.sqrt(self.sigma_squared):.3f} on {self.df_residuals} DF\n"
        docstring += f"R-Squared = {self.r_squared:.4f}, adjusted R-Squared = {self.r_squared_adjusted:.4f}\n"

        if self.fstats is not None:
            docstring += f'F-statistics = {self.fstats:.4f} on {self.df_model} and {self.df_residuals} DF, p-value: {self.f_p_value:{".2f" if self.f_p_value > 1e-5 else "e"}}\n\n'

        docstring += f"Log Likelihood = {self.loglike:.4f}, AIC = {self.AIC:.4f}, BIC = {self.BIC:.4f}"

        if cor is True and self.V_bhat.shape[0] >= 2:
            docstring += "\n\nCorrelation of Coefficients:\n"
            docstring += self._correlation_block()

        print(docstring)

    def plot_observed_fitted(
        self,
        ax=None,
        figsize=None,
        facecolor="none",
        edgecolor="black",
        label_n=3,
    ):

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        y = self.y.to_numpy().astype(float)
        yhat = self.yhat["Fitted"].to_numpy().astype(float)
        ax.scatter(yhat, y, facecolor=facecolor, edgecolor=edgecolor)
        lo = float(min(y.min(), yhat.min()))
        hi = float(max(y.max(), yhat.max()))
        ax.plot([lo, hi], [lo, hi], color="black", linestyle="--")
        _label_top_n(ax, yhat, y, scores=self._residuals_arr, n=label_n)
        ax.set_xlabel("Fitted")
        ax.set_ylabel("Observed")
        ax.set_title("Observed vs. Fitted")

    def plot_residuals(
        self,
        ax=None,
        figsize=None,
        facecolor="none",
        edgecolor="black",
        smooth=True,
        label_n=3,
    ):

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        yhat = self.yhat["Fitted"].to_numpy().astype(float)
        r = self._residuals_arr
        ax.scatter(yhat, r, facecolor=facecolor, edgecolor=edgecolor)
        ax.axhline(0, color="black", linestyle="--")
        if smooth:
            xs, ys = _lowess(yhat, r)
            ax.plot(xs, ys, color="red", linewidth=1.0)
        _label_top_n(ax, yhat, r, scores=r, n=label_n)
        ax.set_xlabel("Fitted")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals vs. Fitted Plot")

    def plot_qq(
        self,
        ax=None,
        figsize=None,
        facecolor="none",
        edgecolor="black",
        label_n=3,
    ):

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        _qq_plot(ax, self.std_residuals, label_n=label_n)

    def plot_scale_location(
        self,
        ax=None,
        figsize=None,
        facecolor="none",
        edgecolor="black",
        smooth=True,
        label_n=3,
    ):

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        yhat = self.yhat["Fitted"].to_numpy().astype(float)
        s = np.sqrt(np.abs(self.std_residuals))
        ax.scatter(yhat, s, facecolor=facecolor, edgecolor=edgecolor)
        if smooth:
            xs, ys = _lowess(yhat, s)
            ax.plot(xs, ys, color="red", linewidth=1.0)
        _label_top_n(ax, yhat, s, scores=self.std_residuals, n=label_n)
        ax.set_xlabel("Fitted")
        ax.set_ylabel(r"$\sqrt{|\mathrm{Std.\ Residuals}|}$")
        ax.set_title("Scale-Location")

    def plot_leverage(
        self,
        ax=None,
        figsize=None,
        facecolor="none",
        edgecolor="black",
        cook_levels=(0.5, 1.0),
        smooth=True,
        label_n=3,
    ):

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        h = self.leverage
        r = self.std_residuals

        # R's plot.lm swaps panel 5 to a factor-level stripchart when hat
        # values are essentially constant (pure-ANOVA, balanced one-way
        # design, etc.). The standard Residuals-vs-Leverage view is
        # degenerate then — every point stacks at the same h, and the
        # Cook's contours sweep over leverages that don't exist in the
        # data, looking informative but meaning nothing.
        h_mean = float(np.mean(h))
        if h_mean > 0 and bool(np.all(np.abs(h - h_mean) < 1e-10 * h_mean)):
            return self._plot_constant_leverage(
                ax=ax, r=r, facecolor=facecolor, edgecolor=edgecolor,
                label_n=label_n,
            )

        ax.scatter(h, r, facecolor=facecolor, edgecolor=edgecolor)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        if smooth:
            xs, ys = _lowess(h, r)
            ax.plot(xs, ys, color="red", linewidth=1.0)
        # Cook's contours: D_i = (r_i^2 / p) · h_i / (1 - h_i)
        # ⇒ r = ±sqrt(c · p · (1 - h) / h)
        ymin, ymax = ax.get_ylim()
        h_max = float(np.clip(h.max() * 1.1, 1e-3, 0.999))
        h_grid = np.linspace(1e-3, h_max, 200)
        for c in cook_levels:
            rline = np.sqrt(c * self.p * (1 - h_grid) / h_grid)
            ax.plot(h_grid, rline, color="red", linestyle="--", linewidth=0.8)
            ax.plot(h_grid, -rline, color="red", linestyle="--", linewidth=0.8)
        ax.set_ylim(ymin, ymax)
        # label by Cook's distance
        cook = (r ** 2 / self.p) * h / np.clip(1 - h, 1e-12, None)
        _label_top_n(ax, h, r, scores=cook, n=label_n)
        ax.set_xlabel("Leverage")
        ax.set_ylabel("Standardized Residuals")
        ax.set_title("Residuals vs. Leverage")

    def _plot_constant_leverage(self, *, ax, r, facecolor, edgecolor, label_n):
        """Stripchart of standardized residuals vs factor-level combinations
        — R's fallback for plot.lm panel 5 when leverage is constant.
        Levels of every categorical RHS predictor are concatenated with
        ``:`` (matching R's ``apply(..., paste, collapse=":")``); models
        with no categorical predictor fall back to observation index."""
        from .formula import referenced_columns

        referenced = referenced_columns(self._expanded)
        factor_cols = [
            c for c in self._design_data.columns
            if c in referenced
            and self._design_data[c].dtype in (pl.Enum, pl.Categorical, pl.Utf8)
        ]

        if factor_cols:
            keys = self._design_data.select(factor_cols).to_numpy().astype(str)
            level_labels = [":".join(row) for row in keys]
            unique_levels = list(dict.fromkeys(level_labels))
            level_to_x = {lab: i for i, lab in enumerate(unique_levels)}
            x_pos = np.array([level_to_x[lab] for lab in level_labels],
                             dtype=float)
            ax.scatter(x_pos, r, facecolor=facecolor, edgecolor=edgecolor)
            ax.set_xticks(range(len(unique_levels)))
            ax.set_xticklabels(unique_levels)
            ax.set_xlim(-0.5, len(unique_levels) - 0.5)
            ax.set_xlabel("Factor Level Combinations")
        else:
            x_pos = np.arange(len(r), dtype=float)
            ax.scatter(x_pos, r, facecolor=facecolor, edgecolor=edgecolor)
            ax.set_xlabel("Obs. number")

        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        if label_n:
            _label_top_n(ax, x_pos, r, scores=np.abs(r), n=label_n)
        ax.set_ylabel("Standardized Residuals")
        ax.set_title("Constant Leverage:\nResiduals vs Factor Levels")

    def plot(self, figsize=None, smooth=True, label_n=3):
        """4-panel diagnostic display: Residuals, Q-Q, Scale-Location, Leverage."""
        if figsize is None:
            figsize = (10, 8)
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        self.plot_residuals(ax=axes[0, 0], smooth=smooth, label_n=label_n)
        self.plot_qq(ax=axes[0, 1], label_n=label_n)
        self.plot_scale_location(ax=axes[1, 0], smooth=smooth, label_n=label_n)
        self.plot_leverage(ax=axes[1, 1], smooth=smooth, label_n=label_n)
        fig.tight_layout()

    def plot_contrast(
        self, features=None, figsize=None, subplots=None, away_from="median"
    ):

        """Visreg style contrast plot.

        "It showed the effect of changing Xj away from an arbitrary point xj*;
        the choice of xj* thereby determines the intercept, as the line by definition passes through (xj*, 0.)
        The equation of this line is y = (x - xj*)bj.

        Ref:
            Breheny & Burchett (2017)
        Note:
            https://stats.stackexchange.com/questions/520774/questions-concerning-visualizing-model-results-with-the-r-package-visreg
        """

        if features is None:
            num_subplots = self.df_model
            features = self.feature_names
        else:
            if type(features) is str:
                num_subplots = 1
                features = [features]
            else:
                num_subplots = len(features)

        if figsize is None:
            figsize = np.array([4 * num_subplots, 3])

        if subplots is None:
            subplots = (1, num_subplots)

        if num_subplots > 1:
            fig, ax = plt.subplots(subplots[0], subplots[1], figsize=figsize)
        else:
            fig, ax = plt.subplots(figsize=figsize)
        ax = np.array([ax]).flatten()

        for i, name in enumerate(features):

            xx = self.X[name].to_numpy().astype(float)

            if away_from == "median":
                xxbar = float(np.median(xx))
            elif away_from == "mean":
                xxbar = float(np.mean(xx))
            elif away_from == "0":
                xxbar = 0.0
            else:
                raise ValueError(f'The Input value for "{away_from}" is not supported.')

            X_arr = self.X.with_columns(pl.lit(xxbar).alias(name)).to_numpy().astype(float)
            rj = self.y.to_numpy().astype(float) - (X_arr @ self._bhat_arr)
            ax[i].scatter(xx, rj, color="gray", facecolor="none", edgecolor="black")
            ax[i].set_xlabel(name)
            ax[i].set_ylabel("Δ" + self.y.name)
            ax[i].spines["top"].set_visible(False)
            ax[i].spines["right"].set_visible(False)

            se_scalar = float(self.se_bhat[name].item())
            Vx = (xx - xxbar) ** 2 * se_scalar ** 2
            se = np.sqrt(Vx)

            tt = t.ppf(1 - 0.05 / 2, self.df_residuals)
            yy = (xx - xxbar) * float(self.bhat[name].item())
            idx_sorted = np.argsort(xx)
            ax[i].plot(xx[idx_sorted], yy[idx_sorted], color="black")
            ax[i].fill_between(
                xx[idx_sorted],
                yy[idx_sorted] + tt * se[idx_sorted],
                yy[idx_sorted] - tt * se[idx_sorted],
                alpha=0.5,
            )

        fig.tight_layout()

    def plot_conditional(
        self, features=None, figsize=None, subplots=None, away_from="median"
    ):

        """Visreg style conditional plot.

        It showed the relationship between E(Y) and Xj while holding other variables constant (mean or median).

        Ref:
            Breheny & Burchett (2017)
        Note:
            https://stats.stackexchange.com/questions/520774/questions-concerning-visualizing-model-results-with-the-r-package-visreg
        """

        if features is None:
            num_subplots = self.df_model
            features = self.feature_names
        else:
            if type(features) is str:
                num_subplots = 1
                features = [features]
            else:
                num_subplots = len(features)

        if figsize is None:
            figsize = np.array([4 * num_subplots, 3])

        if subplots is None:
            subplots = (1, num_subplots)

        if num_subplots > 1:
            fig, ax = plt.subplots(subplots[0], subplots[1], figsize=figsize)
        else:
            fig, ax = plt.subplots(figsize=figsize)
        ax = np.array([ax]).flatten()

        for i, name in enumerate(features):

            xx = self.X[name].to_numpy().astype(float)

            if away_from == "median":
                repl = {
                    name1: float(self.X[name1].median())
                    for name1 in self.column_names
                    if name1 != name
                }
            elif away_from == "mean":
                repl = {
                    name1: float(self.X[name1].mean())
                    for name1 in self.column_names
                    if name1 != name
                }
            elif away_from == "0":
                repl = {name1: 0.0 for name1 in self.column_names if name1 != name}
            else:
                raise ValueError('The Input value for "away_from" is not supported.')

            Xnew = self.X.with_columns(
                [pl.lit(v).alias(k) for k, v in repl.items()]
            ).to_numpy().astype(float)

            rj = self._residuals_arr + (Xnew @ self._bhat_arr)
            ax[i].scatter(xx, rj, color="gray", facecolor="none", edgecolor="black")
            ax[i].set_xlabel(name)
            ax[i].set_ylabel(self.y.name)
            ax[i].spines["top"].set_visible(False)
            ax[i].spines["right"].set_visible(False)

            Vx = Xnew @ self.V_bhat @ Xnew.T
            se = np.sqrt(np.diag(Vx))

            tt = t.ppf(1 - 0.05 / 2, self.df_residuals)
            yy = (Xnew @ self._bhat_arr).flatten()

            ax[i].plot(xx[np.argsort(xx)], yy[np.argsort(xx)], color="black")
            ax[i].fill_between(
                xx[np.argsort(xx)],
                yy[np.argsort(xx)] + tt * se[np.argsort(xx)],
                yy[np.argsort(xx)] - tt * se[np.argsort(xx)],
                alpha=0.5,
            )

        fig.tight_layout()


#################
# Estimate bhat #
#################


def _nq(X: np.array, y: np.array, W: np.array, return_ss: bool = True):
    """
    Solving the normal equations by directly inverting
    the gram matrix.

    return_ss: return sufficient statistics

    """
    XtX = X.T @ W @ X
    Xty = X.T @ W @ y
    b = np.linalg.inv(XtX) @ (Xty)
    if return_ss:
        return b, XtX, Xty
    else:
        return b


def _lu(X: np.array, y: np.array, W: np.array, return_ss: bool = True):
    """
    LU decomposition.
    The same as using numpy.linalg.solve()
    """
    XtX = X.T @ W @ X
    Xty = X.T @ W @ y
    P, L, U = lu(XtX, permute_l=False)
    z = solve_triangular(L, P @ Xty, lower=True)
    b = solve_triangular(U, z)

    if return_ss:
        return b, XtX, Xty
    else:
        return b


def _chol(X: np.array, y: np.array, W: np.array, return_ss: bool = True):
    """
    Cholesky decomposition.
    """
    XtX = X.T @ W @ X
    Xty = X.T @ W @ y
    L = cholesky(XtX, lower=True)
    b = solve_triangular(L.T, solve_triangular(L, Xty, lower=True))
    if return_ss:
        return b, XtX, Xty
    else:
        return b


def _svd(X: np.array, y: np.array, return_ss: bool = True):
    """
    Single value decomposition.
    """
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    Sinv = np.diag(1 / S)
    b = Vt.T @ Sinv @ U.T @ y
    if return_ss:
        XtX = X.T @ X
        Xty = X.T @ y
        return b, XtX, Xty
    else:
        return b


def _qr(X: np.array, y: np.array, W: np.array, return_ss: bool = True):
    """
    QR decomposition.
    """

    L = cholesky(W, lower=True)
    Xhat = L.T @ X
    yhat = L.T @ y

    Q, R = qr(Xhat, mode="economic")
    f = Q.T @ yhat
    b = solve_triangular(R, f)

    if return_ss:
        XtX = X.T @ W @ X
        Xty = X.T @ W @ y
        return b, XtX, Xty
    else:
        return b


def _nll(X, y):

    """
    Negative log-likelihood.
    """

    y = y.flatten()
    n, m = X.shape
    b = np.zeros(m)
    sigma = 1e-5
    p = np.hstack([sigma, b])

    def cost(p, X, y):
        mu = X @ p[1:]
        L = -np.sum(norm.logpdf(y, loc=mu, scale=p[0]))
        return L

    res = minimize(cost, p, args=(X, y), method="L-BFGS-B")
    popt = res.x

    return popt[1:], popt[0], res.fun


def _sse(X, y):

    """
    Sum squared error.
    """

    y = y.flatten()
    n, m = X.shape
    p = np.zeros(m)

    def cost(p, X, y):
        mu = X @ p
        L = np.sum((y - mu) ** 2)
        return L

    res = minimize(cost, p, args=(X, y), method="L-BFGS-B")
    return res.x, res.fun
