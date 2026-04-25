"""Parametric generalized linear models — port of R's ``stats::glm``.

Fisher-scored IRLS using the family/link primitives in :mod:`lmpy.family`.
Output API mirrors :class:`lmpy.lm` (so ``bhat``, ``se_bhat``, ``ci_bhat``,
``yhat`` exist as 1-row :class:`polars.DataFrame`s) and adds the GLM-specific
fields that ``summary.glm`` reports: ``deviance``, ``null_deviance``,
``df_residual``, ``df_null``, ``dispersion``, ``aic``, ``iter``, ``converged``.

The module is family-agnostic: it never branches on ``family.name`` /
``link.name``. The only allowed dispatch is on ``family.scale_known``
(controls dispersion estimation and the t-vs-z choice for Wald inference).
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import polars as pl
from scipy.linalg import qr, solve_triangular
from scipy.stats import norm, t as student_t

from .family import Family, Gaussian, Link
from .formula import materialize
from .utils import format_df, prepare_design, significance_code

__all__ = ["glm"]


# ---------------------------------------------------------------------------
# IRLS engine
# ---------------------------------------------------------------------------


class _IRLSResult:
    """Converged Fisher-IRLS state. Same shape downstream code consumes for
    the main fit and the null-deviance fit."""
    __slots__ = ("beta", "eta", "mu", "w", "deviance", "iter", "converged",
                 "R", "rank", "X_used")

    def __init__(self, *, beta, eta, mu, w, deviance, iter, converged,
                 R, rank, X_used):
        self.beta = beta
        self.eta = eta              # η = Xβ + offset
        self.mu = mu
        self.w = w                  # IRLS weights at convergence
        self.deviance = deviance
        self.iter = iter
        self.converged = converged
        self.R = R                  # upper-triangular R of QR(√w · X)
        self.rank = rank
        self.X_used = X_used        # X with rank-deficient columns dropped


def _fit_irls(
    X: np.ndarray,
    y: np.ndarray,
    *,
    family: Family,
    prior_w: np.ndarray,
    offset: np.ndarray,
    epsilon: float = 1e-8,
    maxit: int = 25,
    qr_tol: float = 1e-7,
) -> _IRLSResult:
    """Fisher-scored IRLS — drop-in replacement for ``stats::glm.fit``.

    Each step solves weighted least squares on ``√w · X`` (matches R's
    ``Cdqrls``) by economy QR. Step-halves the β-update if μ leaves the
    family's valid region or if the deviance worsens (mgcv "inner loop 2/3"
    rule, also what ``glm.fit`` does via ``valideta`` / ``validmu``).

    Convergence rule mirrors R: ``|Δdev| / (|dev| + 0.1) < epsilon``.
    """
    link: Link = family.link
    n, p = X.shape

    mu = family.initialize(y, prior_w)
    eta = link.link(mu) - offset           # η excludes offset; offset added on use
    beta = np.zeros(p)

    # If mustart pushed μ to the boundary, do mgcv's startup nudge so the
    # first IRLS sweep has finite Fisher weights.
    ii = 0
    while not (link.valideta(eta + offset) and family.validmu(mu)):
        ii += 1
        if ii > 20:
            raise FloatingPointError("glm IRLS init: cannot find valid starting μ̂")
        eta = 0.9 * eta
        mu = link.linkinv(eta + offset)

    dev = float(np.sum(family.dev_resids(y, mu, prior_w)))

    R = np.empty((p, p))
    rank = p
    converged = False
    last_iter = 0

    for it in range(1, maxit + 1):
        last_iter = it
        eta_full = eta + offset
        mu_eta_v = link.mu_eta(eta_full)
        V = family.variance(mu)
        if np.any(V == 0) or np.any(~np.isfinite(V)):
            raise FloatingPointError("V(μ) is 0 or non-finite in glm IRLS")
        # Fisher weights w_i = prior_w_i · (dμ/dη)² / V(μ)  — `glm.fit`'s
        # `w` uses ``sqrt(prior.weights * (mu.eta.val)^2 / variance(mu))``
        # in the QR step; we square that here and drop the sqrt back in
        # before QR.
        w = prior_w * mu_eta_v ** 2 / V
        # working response z_i: η_i + (y_i − μ_i)/(dμ/dη)_i, in the
        # offset-stripped η so the LS solve recovers β directly.
        z = eta + (y - mu) / mu_eta_v

        good = w > 0
        if not np.all(good):
            # R drops zero-weight rows from the QR; we mirror by row-masking.
            sw = np.sqrt(w[good])
            X_w = X[good] * sw[:, None]
            z_w = z[good] * sw
        else:
            sw = np.sqrt(w)
            X_w = X * sw[:, None]
            z_w = z * sw

        Q, R = qr(X_w, mode="economic")
        # Rank diagnosis from the |R[i,i]|. R::Cdqrls drops columns whose
        # |R[i,i]| < tol · |R[0,0]|; we replicate. The dropped β-entries are
        # set to NaN downstream so the printed table matches R's NA rows.
        diag_R = np.abs(np.diag(R))
        if diag_R.size and diag_R[0] > 0:
            keep = diag_R >= qr_tol * diag_R[0]
        else:
            keep = np.zeros(p, dtype=bool)
        rank = int(keep.sum())
        if rank < p:
            # Re-do QR on the kept columns; the dropped β's are NaN.
            X_w_keep = X_w[:, keep]
            Q, R_keep = qr(X_w_keep, mode="economic")
            f = Q.T @ z_w
            beta_keep = solve_triangular(R_keep, f)
            beta_new = np.full(p, np.nan)
            beta_new[keep] = beta_keep
            R_full = np.full((p, p), np.nan)
            R_full[np.ix_(keep, keep)] = R_keep
            R = R_full
            X_used = X[:, keep]
        else:
            f = Q.T @ z_w
            beta_new = solve_triangular(R, f)
            X_used = X

        eta_new = X @ np.nan_to_num(beta_new)
        eta_full_new = eta_new + offset
        mu_new = link.linkinv(eta_full_new)

        # mgcv "inner loop 2": step-halve until μ_new is family-valid.
        ii = 0
        while not (link.valideta(eta_full_new) and family.validmu(mu_new)):
            ii += 1
            if ii > maxit:
                raise FloatingPointError("glm IRLS: validity step-halving failed")
            beta_new = 0.5 * (beta_new + beta)
            eta_new = X @ np.nan_to_num(beta_new)
            eta_full_new = eta_new + offset
            mu_new = link.linkinv(eta_full_new)

        dev_new = float(np.sum(family.dev_resids(y, mu_new, prior_w)))

        # mgcv "inner loop 3" / glm.fit step-halving on dev increase. R's
        # criterion: dev_new is non-finite OR (it > 1 and dev_new > dev).
        ii = 0
        while (not np.isfinite(dev_new)) or (it > 1 and dev_new > dev):
            ii += 1
            if ii > maxit:
                break
            beta_new = 0.5 * (beta_new + beta)
            eta_new = X @ np.nan_to_num(beta_new)
            eta_full_new = eta_new + offset
            mu_new = link.linkinv(eta_full_new)
            if not (link.valideta(eta_full_new) and family.validmu(mu_new)):
                continue
            dev_new = float(np.sum(family.dev_resids(y, mu_new, prior_w)))

        # convergence check: R uses ``|dev_new - dev| / (0.1 + |dev_new|)``.
        if abs(dev_new - dev) / (0.1 + abs(dev_new)) < epsilon:
            beta = beta_new
            eta = eta_new
            mu = mu_new
            dev = dev_new
            converged = True
            break

        beta = beta_new
        eta = eta_new
        mu = mu_new
        dev = dev_new

    # Final consistent w at the converged (β, μ, η).
    eta_full = eta + offset
    mu_eta_v = link.mu_eta(eta_full)
    V = family.variance(mu)
    w_final = prior_w * mu_eta_v ** 2 / V

    return _IRLSResult(
        beta=beta, eta=eta_full, mu=mu, w=w_final, deviance=dev,
        iter=last_iter, converged=converged, R=R, rank=rank, X_used=X_used,
    )


def _row_frame(values: np.ndarray, columns: list[str]) -> pl.DataFrame:
    """Build a 1-row pl.DataFrame from a flat numpy array + column names."""
    flat = np.asarray(values).reshape(-1)
    return pl.DataFrame({c: [float(flat[i])] for i, c in enumerate(columns)})


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class glm:
    """Generalized linear model — parity target is R's ``stats::glm``.

    Parameters
    ----------
    formula : str
        R-style formula, e.g. ``"y ~ x1 + x2"``. Single-name LHS only;
        the ``cbind(success, failure) ~ ...`` binomial form is not yet
        supported (use ``weights=`` to pass the binomial size ``m``).
    data : polars.DataFrame
        Input data; rows with NA in any referenced column are dropped
        (R's ``na.action = na.omit``).
    family : :class:`lmpy.family.Family`, optional
        Defaults to :class:`Gaussian` (= identity link). Pass e.g.
        ``Poisson()``, ``Gamma(link="log")``, ``Binomial(link="probit")``.
    weights : array-like or None
        Prior weights. For binomial, this is the binomial size ``m``
        (= 1 for Bernoulli). Defaults to ones.
    offset : array-like or None
        Length-``n`` offset added directly to the linear predictor
        (η = Xβ + offset). Defaults to zeros.
    control : dict
        IRLS control: ``epsilon`` (default 1e-8), ``maxit`` (25). Match R.

    Attributes
    ----------
    bhat, se_bhat, ci_bhat, t_values, p_values : 1-row pl.DataFrame
        Coefficient estimates and Wald inference. The third column header
        in the printed summary is ``Pr(>|t|)`` for unknown-scale families
        (Gaussian, Gamma, IG) and ``Pr(>|z|)`` for known-scale (Poisson,
        binomial), matching ``summary.glm``.
    yhat : pl.DataFrame
        Single-column ``Fitted`` (= μ̂). Use ``predict`` for new data.
    fitted_values, linear_predictors : numpy.ndarray
        μ̂ and η̂ on the training rows.
    deviance, null_deviance, df_residual, df_null : float / int
        From ``family.dev_resids``; null fit refits an intercept-only model.
    dispersion : float
        Pearson estimator ``Σ w·(y−μ)²/V / df_residual``; ``1.0`` for
        scale-known families. Aliased as ``scale`` and ``sigma_squared``.
    aic, bic, loglike : float
        ``aic = family.aic(...) + 2·k`` with ``k = rank(X) + (1 if not
        scale_known else 0)``; ``BIC = aic - 2k + log(n)·k``;
        ``loglike = -(aic - 2k)/2``.
    iter, converged : int / bool
        Fisher-scoring iteration count and convergence flag.
    """

    def __init__(
        self,
        formula: str,
        data: pl.DataFrame,
        family: Optional[Family] = None,
        weights: Union[None, np.ndarray, list] = None,
        offset: Union[None, np.ndarray, list] = None,
        control: Optional[dict] = None,
    ):
        self.formula = formula
        self.data = data
        self.family = Gaussian() if family is None else family
        ctl = {"epsilon": 1e-8, "maxit": 25}
        if control:
            ctl.update(control)

        d = prepare_design(formula, data)
        self._expanded = d.expanded
        self._design_data = d.data
        self.X = d.X                                  # pl.DataFrame
        self.y = d.y                                  # pl.Series

        X = self.X.to_numpy().astype(float)
        y = self.y.to_numpy().astype(float).flatten()
        n, p = X.shape

        prior_w = (np.ones(n) if weights is None
                   else np.asarray(weights, dtype=float).flatten())
        if prior_w.shape != (n,):
            raise ValueError(
                f"weights must have length {n}, got {prior_w.shape}"
            )
        if np.any(prior_w < 0):
            raise ValueError("negative weights not allowed")
        self._prior_w = prior_w

        off = (np.zeros(n) if offset is None
               else np.asarray(offset, dtype=float).flatten())
        if off.shape != (n,):
            raise ValueError(f"offset must have length {n}, got {off.shape}")
        self._offset = off

        self.column_names = list(self.X.columns)
        self.feature_names = (
            self.column_names[1:]
            if "(Intercept)" in self.column_names
            else self.column_names
        )
        self.has_intercept = "(Intercept)" in self.column_names

        self.n = n
        self.p = p

        # ---- main IRLS fit ------------------------------------------------
        fit = _fit_irls(
            X, y, family=self.family, prior_w=prior_w, offset=off,
            epsilon=ctl["epsilon"], maxit=ctl["maxit"],
        )
        self._fit = fit
        self.iter = fit.iter
        self.converged = fit.converged
        self.rank = fit.rank
        self.df_residual = n - fit.rank
        self.df_residuals = self.df_residual            # lm-style alias

        self._bhat_arr = fit.beta.copy()
        self.bhat = _row_frame(self._bhat_arr, self.column_names)
        self.coefficients = self.bhat                   # R-canonical alias

        # μ̂, η̂ (η̂ includes offset).
        self.fitted_values = fit.mu
        self.linear_predictors = fit.eta
        self.yhat = pl.DataFrame({"Fitted": fit.mu})    # lm-style
        self.fitted = self.fitted_values

        # Deviance.
        self.deviance = fit.deviance

        # ---- dispersion + vcov + SE --------------------------------------
        self.dispersion = self._compute_dispersion(fit, prior_w, y)
        self.scale = self.dispersion                    # gam-style alias
        self.sigma_squared = self.dispersion            # lm-style alias
        self.sigma = float(np.sqrt(self.dispersion))

        self.vcov, self._XtWXinv = self._compute_vcov(fit)
        self.V_bhat = self.vcov                         # lm-style alias
        se = np.sqrt(np.diag(self.vcov))
        self._se_bhat_arr = se
        self.se_bhat = _row_frame(se, self.column_names)

        # ---- inference: t-or-z, p, CI ------------------------------------
        self._test_kind = "z" if self.family.scale_known else "t"
        with np.errstate(invalid="ignore", divide="ignore"):
            stat = self._bhat_arr / self._se_bhat_arr
        self._stat_arr = stat
        self.t_values = _row_frame(stat, self.column_names)
        self.z_values = self.t_values                   # alias for known-scale
        self.p_values = _row_frame(self._wald_p(stat), self.column_names)
        self.ci_bhat = self._compute_ci(0.05)

        # ---- null deviance, AIC, BIC, logLik -----------------------------
        self.null_deviance, self.df_null = self._compute_null_deviance(
            y, prior_w, off
        )
        # Mirror R's stats:::logLik.glm exactly:
        #   p <- rank + (family %in% c("gaussian","Gamma","inverse.gaussian"))
        #   loglik <- p - aic/2;   df <- p
        # Then stats:::BIC.default uses df from logLik:  -2·loglik + log(n)·df.
        # And stats:::glm.fit's `aic` field is `family$aic + 2·rank` — where
        # family$aic already includes a "+2" for the dispersion df on
        # Gaussian/Gamma/IG (see lmpy/family.py). So the dispersion df enters
        # once via family$aic (not a second time via rank), but the *npar*
        # used by logLik/BIC explicitly counts it.
        self.npar = fit.rank + (0 if self.family.scale_known else 1)
        self.aic = self._compute_aic(y, fit.mu, prior_w, k_for_aic=fit.rank)
        self.AIC = self.aic
        self.loglike = float(self.npar) - 0.5 * self.aic
        # R's nobs(glm) returns n (length of fitted), not Σwt — Σwt only
        # enters via the family's weighted log-likelihood inside aic.
        self.bic = -2.0 * self.loglike + float(np.log(self.n)) * self.npar
        self.BIC = self.bic

    # ----- helpers ---------------------------------------------------------

    def _compute_dispersion(self, fit: _IRLSResult,
                            prior_w: np.ndarray, y: np.ndarray) -> float:
        if self.family.scale_known:
            return 1.0
        if self.df_residual <= 0:
            return float("nan")
        V = self.family.variance(fit.mu)
        # Pearson: Σ prior_w · (y - μ)² / V(μ)  (matches summary.glm$dispersion).
        chi2 = float(np.sum(prior_w * (y - fit.mu) ** 2 / V))
        return chi2 / self.df_residual

    def _compute_vcov(self, fit: _IRLSResult):
        # vcov = dispersion · (XᵀWX)⁻¹ at converged W. We have R from
        # QR(√w·X_used); (X_usedᵀWX_used)⁻¹ = R⁻¹ (R⁻¹)ᵀ. For dropped
        # columns (NaN β), insert NaN rows/cols so the printed table aligns.
        p = self.p
        rank = fit.rank
        V = np.full((p, p), np.nan)
        if rank == 0:
            return V, V.copy()
        # Build `keep` mask from the diagonal of fit.R (NaN in dropped slots).
        diag = np.diag(fit.R)
        keep = ~np.isnan(diag)
        R_keep = fit.R[np.ix_(keep, keep)]
        Rinv = solve_triangular(R_keep, np.eye(rank))
        XtWXinv_keep = Rinv @ Rinv.T
        XtWXinv = np.full((p, p), np.nan)
        XtWXinv[np.ix_(keep, keep)] = XtWXinv_keep
        V[np.ix_(keep, keep)] = self.dispersion * XtWXinv_keep
        return V, XtWXinv

    def _wald_p(self, stat: np.ndarray) -> np.ndarray:
        with np.errstate(invalid="ignore"):
            if self._test_kind == "z":
                return 2.0 * norm.sf(np.abs(stat))
            return 2.0 * student_t.sf(np.abs(stat), self.df_residual)

    def _compute_ci(self, alpha: float) -> pl.DataFrame:
        # R's confint.default — used as the default `confint` for glm
        # objects — applies the qnorm quantile to ALL families, even
        # unknown-scale ones (Gaussian/Gamma/IG). The t-quantile is
        # only used by confint.lm. (confint.glm itself returns profile-
        # likelihood CIs, which are out of scope here.)
        q = float(norm.ppf(1.0 - alpha / 2.0))
        bhat = self._bhat_arr
        se = self._se_bhat_arr
        lo = bhat - q * se
        hi = bhat + q * se
        return pl.DataFrame({
            "coef": self.column_names,
            f"CI[{alpha/2*100}%]": lo,
            f"CI[{100-alpha/2*100}]%": hi,
        })

    def _compute_null_deviance(self, y, prior_w, offset):
        """Intercept-only fit deviance.

        Score equation for an intercept-only model gives the link-applied
        weighted mean of y as the unique solution, *for any monotone link*
        (μ is constant ⇒ ∂L/∂β₀ = 0 ⇒ Σ wᵢ(yᵢ - μ̂) = 0). This matches
        ``glm.fit``'s ``wtdmu`` shortcut. With offset, μ varies but the
        same closed form needs an IRLS step — fall back to that case.
        """
        df_null = self.n - (1 if self.has_intercept else 0)
        if not self.has_intercept:
            # Null model is the empty model: μ = link.linkinv(offset).
            mu0 = self.family.link.linkinv(offset)
            null_dev = float(np.sum(self.family.dev_resids(y, mu0, prior_w)))
            return null_dev, df_null

        if np.allclose(offset, 0.0):
            # Closed form: μ̂ = wtdmu = Σ wy / Σ w (constant).
            mu0_const = float(np.sum(prior_w * y) / np.sum(prior_w))
            mu0 = np.full(self.n, mu0_const)
            null_dev = float(np.sum(self.family.dev_resids(y, mu0, prior_w)))
            return null_dev, df_null

        # Offset present → run IRLS on the intercept-only design.
        X1 = np.ones((self.n, 1))
        try:
            null_fit = _fit_irls(
                X1, y, family=self.family, prior_w=prior_w, offset=offset,
            )
            null_dev = null_fit.deviance
        except FloatingPointError:
            null_dev = float("nan")
        return null_dev, df_null

    def _compute_aic(self, y, mu, prior_w, *, k_for_aic: int) -> float:
        # R glm.fit: aic.model = family$aic(y, n, μ, w, dev) + 2 · rank,
        # where `dev` is the model deviance — *not* `scale · Σwt`. That
        # distinction matters for Gamma/IG: their family.aic recovers the
        # dispersion as `dev/Σwt` (deviance/n moment estimator), which
        # differs from the Pearson estimator `summary()` reports.
        # mgcv's gam.fit3 swaps in `scale · Σwt` (see family._aic_dev1) so
        # the AIC tracks the REML/Pearson scale; that's a GAM-only choice
        # and matching R glm requires bypassing it here.
        family_aic = float(self.family.aic(
            y, mu, self.deviance, prior_w, self.n,
        ))
        return family_aic + 2.0 * k_for_aic

    # ----- residuals_of ---------------------------------------------------

    def residuals_of(self, type: str = "deviance") -> np.ndarray:
        """Residuals of the requested type.

        ``type`` ∈ ``{"deviance", "pearson", "working", "response"}``;
        defaults to deviance, matching ``residuals.glm``.
        """
        y = self.y.to_numpy().astype(float).flatten()
        mu = self.fitted_values
        prior_w = self._prior_w
        link = self.family.link
        if type == "response":
            return y - mu
        if type == "working":
            mu_eta_v = link.mu_eta(self.linear_predictors)
            return (y - mu) / mu_eta_v
        if type == "pearson":
            V = self.family.variance(mu)
            return np.sqrt(prior_w) * (y - mu) / np.sqrt(V)
        if type == "deviance":
            per_obs = self.family.dev_resids(y, mu, prior_w)
            return np.sign(y - mu) * np.sqrt(np.maximum(per_obs, 0.0))
        raise ValueError(
            f"unknown residual type {type!r}; "
            "expected one of: deviance, pearson, working, response"
        )

    @property
    def residuals(self) -> pl.DataFrame:
        # Default = deviance residuals (R glm() default). Returned as a
        # 1-col DataFrame for consistency with lm.residuals.
        return pl.DataFrame({"residuals": self.residuals_of("deviance")})

    # ----- predict --------------------------------------------------------

    def predict(
        self,
        new: Optional[pl.DataFrame] = None,
        type: str = "response",
        se_fit: bool = False,
        offset: Union[None, np.ndarray, list] = None,
        alpha: float = 0.05,
    ):
        """Generate predictions on new data — :func:`predict.glm` parity.

        ``type='response'`` returns μ̂ = linkinv(η̂); ``type='link'`` returns
        η̂. If ``se_fit=True``, also returns the standard error: on the link
        scale ``√diag(X·vcov·Xᵀ)``; on the response scale this is multiplied
        by ``|dμ/dη|`` (delta method, matches R).
        """
        if new is None:
            X_new = self.X.to_numpy().astype(float)
            n_new = self.n
        else:
            X_new = materialize(self._expanded, new).to_numpy().astype(float)
            n_new = X_new.shape[0]
        off_new = (np.zeros(n_new) if offset is None
                   else np.asarray(offset, dtype=float).flatten())
        if off_new.shape != (n_new,):
            raise ValueError(f"offset must have length {n_new}")

        # Replace NaN coef slots with 0 — they correspond to dropped rank-
        # deficient columns, which R also reports as NA-coefficient and
        # excludes from the prediction. The matching X column is multiplied
        # by 0 here (same as R's behaviour for `singular.ok`).
        beta = np.nan_to_num(self._bhat_arr)
        eta = X_new @ beta + off_new
        if type == "link":
            fit = eta
        elif type == "response":
            fit = self.family.link.linkinv(eta)
        else:
            raise ValueError(f"unknown predict type {type!r}; expected 'response' or 'link'")

        if not se_fit:
            return fit

        # SE on the link scale: √diag(X · vcov · Xᵀ); use the kept-column
        # subspace so NaN-vcov entries don't propagate.
        keep = ~np.isnan(np.diag(self.vcov))
        Xk = X_new[:, keep]
        Vk = self.vcov[np.ix_(keep, keep)]
        var_link = np.einsum("ij,jk,ik->i", Xk, Vk, Xk)
        se_link = np.sqrt(np.maximum(var_link, 0.0))
        if type == "link":
            return fit, se_link
        # response-scale SE = |dμ/dη(η̂)| · se_link  (delta method).
        mu_eta_v = self.family.link.mu_eta(eta)
        return fit, np.abs(mu_eta_v) * se_link

    # ----- printing -------------------------------------------------------

    def __repr__(self) -> str:
        d = f"glm({self.formula!r}, family={self.family!r})\n\n"
        d += "Coefficients:\n"
        d += format_df(self.bhat)
        return d

    def __str__(self) -> str:
        return self.__repr__()

    def summary(self, digits: int = 3) -> None:
        """Print a ``summary.glm``-styled report."""
        stat_lbl = "z value" if self._test_kind == "z" else "t value"
        p_lbl = "Pr(>|z|)" if self._test_kind == "z" else "Pr(>|t|)"

        p_arr = np.asarray(self.p_values.row(0), dtype=float)
        sig = significance_code(p_arr)
        ci_low_col, ci_hi_col = self.ci_bhat.columns[1], self.ci_bhat.columns[2]

        coef_df = pl.DataFrame({
            "":            self.column_names,
            "Estimate":    np.round(self._bhat_arr, digits),
            "Std. Error":  np.round(self._se_bhat_arr, digits),
            ci_low_col:    np.round(self.ci_bhat[ci_low_col].to_numpy(), digits),
            ci_hi_col:     np.round(self.ci_bhat[ci_hi_col].to_numpy(), digits),
            stat_lbl:      np.round(self._stat_arr, digits),
            p_lbl:         np.round(p_arr, digits),
            " ":           sig,
        })

        out = f"Call:\nglm(formula = {self.formula}, family = {self.family})\n\n"
        out += "Coefficients:\n"
        out += format_df(coef_df)
        out += "\n---"
        out += "\nSignif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1\n"

        disp_word = "fixed" if self.family.scale_known else "estimated"
        out += (
            f"\n(Dispersion parameter for {self.family.name} family "
            f"taken to be {self.dispersion:.{digits}g}; {disp_word})\n"
        )
        out += (
            f"\n    Null deviance: {self.null_deviance:.4f}  on {self.df_null} degrees of freedom"
            f"\nResidual deviance: {self.deviance:.4f}  on {self.df_residual} degrees of freedom"
        )
        out += f"\nAIC: {self.aic:.4f}    BIC: {self.bic:.4f}    logLik: {self.loglike:.4f}"
        out += f"\n\nNumber of Fisher Scoring iterations: {self.iter}"
        out += "" if self.converged else "  (did NOT converge!)"
        print(out)
