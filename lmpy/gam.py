"""Generalized additive model — mgcv-style penalized regression with
REML/GCV smoothing-parameter selection.

Built on lmpy.formula's ``parse → expand → materialize / materialize_smooths``
pipeline: the parametric side comes from ``materialize`` (R-canonical
column names); each smooth call (``s``/``te``/``ti``/``t2``) is passed to
``materialize_smooths`` which mirrors mgcv's ``smoothCon(..., absorb.cons=
TRUE, scale.penalty=TRUE)``.

The penalized design is assembled once as
``X = [X_param | X_block_1 | X_block_2 | …]`` with a parallel list of
penalty matrices ``S_k`` (one per (block, penalty) slot) embedded in
``p × p`` templates. Smoothing parameters ``λ = exp(ρ)`` are selected by
minimizing REML (default) or GCV over ``ρ`` with L-BFGS-B; at each
evaluation ``β̂(λ) = (XᵀX + Sλ)⁻¹ Xᵀy`` is solved by Cholesky.

Identifiability across nested smooths (``s(x1) + te(x1, x2)``) is
handled by an in-Python port of mgcv's ``gam.side`` / ``fixDependence``:
te columns that are linearly dependent on the marginal smooths are
deleted before fitting, dropping te from 24 → 22 cols (matching
``ncol(model.matrix(m))``).

Gaussian identity link only in this first port. Non-Gaussian families,
penalized null-space shrinkage, prediction intervals, and out-of-sample
prediction for smooth terms (needs a mgcv-style ``PredictMat`` shim)
are out of scope here.

References
----------
Wood (2011), "Fast stable REML and ML estimation of semiparametric GLMs",
JRSS B 73(1), §3-4.
Wood (2017), *Generalized Additive Models* (2nd ed.), §6.2, §6.6.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.stats import f as f_dist, norm, t as t_dist

from .family import Family, Gaussian
from .formula import SmoothBlock, materialize_smooths
from .utils import format_df, prepare_design, significance_code

__all__ = ["gam"]


class gam:
    """Generalized additive model (Gaussian identity), fit by REML or GCV.

    Parameters
    ----------
    formula : str
        mgcv-style formula, e.g. ``"y ~ x1 + s(x2) + s(x3, bs='cr') +
        te(u, v)"``.
    data : polars.DataFrame
        Data table; rows with NA in any referenced column are dropped
        before fitting.
    method : {"REML", "GCV.Cp"}, default "REML"
        Smoothing-parameter selection criterion.
    sp : None or array-like, optional
        If given, fix smoothing parameters at these (non-negative)
        values and skip optimization. Length must match the total number
        of penalty slots across all smooth blocks.

    Attributes (always set)
    -----------------------
    n, p : int
        Sample size, total # of model coefficients (parametric + smooth).
    p_param : int
        Number of parametric coefficients.
    bhat, se_bhat : polars.DataFrame
        Coefficient estimates / Bayesian SEs (one row each, keyed by
        R-canonical coefficient names ``(Intercept)``, ``MachineB``,
        ``s(x).1``, ``s(x).2``, …).
    t_values, p_values : polars.DataFrame
        Per-coefficient Wald t-stat and p-value — only meaningful for
        *parametric* rows; smooth-basis rows are reported but users
        should interpret via the smooth-level table (``smooth_table``).
    linear_predictors : np.ndarray
        Length-n linear predictor ``η = Xβ̂``.
    fitted_values : np.ndarray
        Length-n fitted mean ``μ̂ = g⁻¹(η)``. For Gaussian-identity, μ = η.
    fitted : np.ndarray
        Alias for ``fitted_values`` (was ``η``; equivalent for Gaussian).
    residuals : np.ndarray
        Length-n response residuals ``y − μ̂``. Use ``residuals_of(type=…)``
        to request deviance/Pearson/working/response variants.
    sigma, sigma_squared : float
        Residual SD and variance (``scale`` in mgcv).
    sp : np.ndarray
        Optimized (or fixed) smoothing parameters, length
        ``n_sp = Σ_blocks |S_block|``.
    edf : np.ndarray
        Per-coefficient effective degrees of freedom, diagonal of the
        influence matrix in coefficient space
        ``F = (XᵀX + Sλ)⁻¹ XᵀX``. Parametric entries are 1.
    edf_by_smooth : dict[str, float]
        Summed edf per smooth label (``"s(x)"``, ``"te(u,v)"``, …).
    edf_total : float
        ``sum(edf)`` — total model degrees of freedom (β + 1 for σ
        is *not* added; use ``npar`` for the MLE parameter count).
    Vp : np.ndarray
        Bayesian posterior covariance ``σ² (XᵀX + Sλ)⁻¹``. Matches
        mgcv's ``$Vp``.
    Ve : np.ndarray
        Frequentist covariance ``σ² (XᵀX + Sλ)⁻¹ XᵀX (XᵀX + Sλ)⁻¹``.
        Matches mgcv's ``$Ve``.
    r_squared, r_squared_adjusted : float
        As mgcv: 1 − rss/tss and the df-adjusted variant.
    deviance : float
        ``rss`` for Gaussian.
    loglike : float
        Unpenalized Gaussian log-likelihood at the fitted β̂.
    AIC, BIC : float
        ``-2·loglike + 2·npar`` (and ``log(n)·npar`` for BIC), where
        ``npar = edf_total + 1`` for the residual variance — matches R's
        ``AIC(gam_fit)``.
    npar : float
        ``edf_total + 1``. Not an integer because edf isn't.
    formula : str
    data : polars.DataFrame

    Attributes (method="REML" only)
    -------------------------------
    REML_criterion : float
        Optimized Laplace-approximate REML criterion, ``-2·V_R(ρ̂)``.

    Attributes (method="GCV.Cp" only)
    ---------------------------------
    GCV_score : float
        Optimized GCV score, ``n · rss / (n − edf_total)²``.
    """

    def __init__(
        self,
        formula: str,
        data: pl.DataFrame,
        *,
        method: str = "GCV.Cp",
        sp: np.ndarray | None = None,
        family: Family | None = None,
    ):
        if method not in ("REML", "GCV.Cp"):
            raise ValueError(f"method must be 'REML' or 'GCV.Cp', got {method!r}")

        self.formula = formula
        self.method = method
        self.family = Gaussian() if family is None else family
        # GCV.Cp dispatches by family.scale_known: scale-unknown (Gaussian,
        # Gamma, IG) → GCV `n·D/(n−τ)²`; scale-known (Poisson, Binomial) →
        # UBRE `D/n + 2·τ/n − 1`. mgcv's `gam.outer` does the same dispatch
        # under method="GCV.Cp".
        d = prepare_design(formula, data)
        self._expanded = d.expanded
        self.data = d.data
        X_param_df = d.X
        y = d.y.to_numpy().astype(float)
        X_param = X_param_df.to_numpy().astype(float)
        n, p_param = X_param.shape

        sb_lists = materialize_smooths(d.expanded, d.data) if d.expanded.smooths else []
        blocks: list[SmoothBlock] = [b for group in sb_lists for b in group]
        # mgcv's gam.side: when one smooth's variable set is a strict subset
        # of another's (e.g. `s(x1) + te(x1, x2)`), the wider smooth's basis
        # contains a copy of the narrower's main effect, which makes the
        # combined design rank-deficient and the REML/GCV optimum drift away
        # from mgcv's. Apply orthogonality constraints (column-rotate the
        # wider smooth so its columns are orthogonal in the data space to
        # the narrower's). This typically drops one column per overlap, so
        # `te(x1, x2)` next to `s(x1) + s(x2)` shrinks 24 → 22 cols, matching
        # mgcv's `model.matrix` exactly.
        blocks = _apply_gam_side(blocks)

        # Build full design X = [X_param | X_block_1 | X_block_2 | …] and the
        # parallel list of penalty "slots" (one per (block, S_j) pair). Each
        # slot carries its column range in the full design so we can embed the
        # k×k penalty in the p×p full-design template without allocating a
        # zero-padded copy per evaluation.
        Xs = [X_param]
        slots: list[_PenaltySlot] = []
        block_col_ranges: list[tuple[int, int]] = []
        col_cursor = p_param
        for b in blocks:
            Xb = np.asarray(b.X, dtype=float)
            Xs.append(Xb)
            k = Xb.shape[1]
            a, bcol = col_cursor, col_cursor + k
            block_col_ranges.append((a, bcol))
            for S_j in b.S:
                slots.append(_PenaltySlot(block=b, col_start=a, col_end=bcol,
                                          S=np.asarray(S_j, dtype=float)))
            col_cursor = bcol
        X = np.concatenate(Xs, axis=1) if len(Xs) > 1 else X_param
        p = X.shape[1]

        # Column names: parametric (R-canonical) + "s(x).1", "s(x).2", … per
        # block. Matches mgcv's `coef(gam_fit)` labels. For multi-block
        # smooths (by = factor), the block label already includes the level
        # suffix (see formula._apply_by_and_absorb).
        column_names = list(X_param_df.columns)
        for b, (a, bcol) in zip(blocks, block_col_ranges):
            for i in range(1, bcol - a + 1):
                column_names.append(f"{b.label}.{i}")
        assert len(column_names) == p

        # ------------- sufficient statistics -------------------------------
        XtX = X.T @ X
        Xty = X.T @ y
        yty = float(y @ y)
        y_mean = float(y.mean())
        has_intercept = "(Intercept)" in X_param_df.columns
        tss = float(np.sum((y - y_mean) ** 2)) if has_intercept else yty

        self.X = X_param_df              # parametric design (user-facing)
        self._X_full = X                 # penalized full design
        self.y = d.y
        self._y_arr = y
        self.n = n
        self.p = p
        self.p_param = p_param
        self._blocks = blocks
        self._slots = slots
        self._block_col_ranges = block_col_ranges
        self.column_names = column_names
        self._XtX = XtX
        self._Xty = Xty
        self._yty = yty
        self._has_intercept = has_intercept
        self._tss = tss
        self.parametric_columns = list(X_param_df.columns)

        # Null-space dimension Mp for REML. Sum of per-block null dimensions
        # from the combined (over penalties in that block) penalty rank,
        # plus all parametric columns.
        Mp = p_param
        for b, (a, bcol) in zip(blocks, block_col_ranges):
            k = bcol - a
            if not b.S:
                Mp += k
                continue
            S_sum = np.sum([np.asarray(s, dtype=float) for s in b.S], axis=0)
            rank = _sym_rank(S_sum)
            Mp += k - rank
        self._Mp = Mp
        # Total penalty rank across all blocks — dimension of the penalized
        # subspace. Used below to take the log-determinant of Sλ over its
        # range space (rather than over eigenvalues > some tolerance, which
        # silently drops directions whose own λ_j shrinks to zero and breaks
        # the log|Sλ|_+ → -∞ behaviour that penalizes λ_j → 0 in REML).
        self._penalty_rank = p - Mp

        # ------------- smoothing-param optimization ------------------------
        n_sp = len(slots)
        # Set by the optimizer branch below when log φ enters the outer
        # vector (PIRLS path, unknown-scale family). None means φ is
        # profiled (Gaussian-identity strict-additive) or fixed at 1
        # (scale-known families) — i.e., off the outer-vec.
        self._log_phi_hat: float | None = None
        if n_sp == 0:
            # No smooths — degenerate to unpenalized least squares. This is
            # the lm path; we still go through it so all the mgcv-style
            # post-fit attributes are populated.
            self.sp = np.zeros(0)
            rho_hat = np.zeros(0)
            fit = self._fit_given_rho(rho_hat)
        elif sp is not None:
            sp_arr = np.asarray(sp, dtype=float)
            if sp_arr.shape != (n_sp,):
                raise ValueError(
                    f"sp must have length {n_sp} (one per penalty slot), got {sp_arr.shape}"
                )
            if np.any(sp_arr < 0):
                raise ValueError("sp entries must be non-negative")
            # guard log(0) — a hard zero sp means "no penalty," which we
            # represent as exp(-large) instead, matching mgcv's handling.
            rho_hat = np.log(np.maximum(sp_arr, 1e-10))
            self.sp = sp_arr
            fit = self._fit_given_rho(rho_hat)
        else:
            # Unified outer optimization. PIRLS inner solve + general
            # `_reml(ρ, log φ)` + analytical Newton, family-agnostic.
            # ``include_log_phi`` is True for unknown-scale (Gaussian, Gamma,
            # IG): θ = (ρ, log φ). False for known-scale (Poisson, Binomial):
            # θ = ρ with log φ ≡ 0. mgcv's gam.outer behaves the same way.
            family = self.family
            include_log_phi = (not family.scale_known) and method == "REML"
            n_lp = 1 if include_log_phi else 0
            theta_dim = n_sp + n_lp

            # Initial seed.
            #
            # REML and GCV both run analytical Newton on the criterion's
            # exact Hessian (mgcv's gam.outer). REML starts at ρ=0 (Newton's
            # eigen-clamped quadratic model handles the global descent).
            # GCV uses a coordinate grid-scan first, then Newton: the
            # criterion has flat saturation tails on some smooths (e.g.
            # mcycle's tp) where Newton from ρ=0 can drift toward the
            # boundary; the grid scan finds the right basin.
            def _pearson_log_phi(rho_eval) -> float:
                if not include_log_phi:
                    return 0.0
                try:
                    fit_seed = self._fit_given_rho(rho_eval)
                except Exception:
                    return 0.0
                df_resid_seed = max(self.n - self._Mp, 1.0)
                V_seed = family.variance(fit_seed.mu)
                pearson = float(np.sum(
                    (self._y_arr - fit_seed.mu) ** 2
                    / np.maximum(V_seed, 1e-300)
                ))
                return float(np.log(max(pearson / df_resid_seed, 1e-12)))

            if method == "REML":
                cur_rho = np.zeros(n_sp)
                cur_logphi = _pearson_log_phi(cur_rho)
            else:
                # Mirror mgcv's initial.sp (mgcv/R/gam.fit3.r): for each smooth
                # k, def.sp[k] = mean(diag(X_k'X_k)) / mean(diag(S_k)) over the
                # penalised column-rows of S_k. The threshold filter matches
                # mgcv's ``thresh = .Machine$double.eps^0.8 * max(|S_k|)``.
                cur_rho = self._initial_sp_rho()
                cur_logphi = 0.0  # GCV does not put log φ in θ

            theta0 = np.r_[cur_rho, cur_logphi] if include_log_phi else cur_rho

            theta_hat = self._outer_newton(
                theta0,
                criterion=method if method == "REML" else "GCV",
                include_log_phi=include_log_phi,
            )

            if include_log_phi:
                rho_hat = theta_hat[:n_sp]
                self._log_phi_hat = float(theta_hat[n_sp])
            else:
                rho_hat = theta_hat
                self._log_phi_hat = None
            self.sp = np.exp(rho_hat)
            fit = self._fit_given_rho(rho_hat)

        # Unpack fit results. ``fit.A_chol`` is the Newton-W factorization
        # used by REML's log|H+S| term and the IFT for ∂β̂/∂ρ. mgcv's
        # post-fit reporting (m$edf, m$Vp, m$Ve) instead plugs in the
        # Fisher weight W_F = μ_η²/V (gam.fit3.r:644). Build a Fisher view
        # for those; for canonical links Newton ≡ Fisher and the view
        # reuses fit's chol — cheap.
        beta = fit.beta
        rss = fit.rss
        pen = fit.pen
        Sλ = fit.S_full

        self._rho_hat = rho_hat

        fit_F = self._fisher_view(fit)
        A_chol = fit_F.A_chol
        A_chol_lower = fit_F.A_chol_lower
        log_det_A = fit_F.log_det_A
        # Posterior β covariance Vp = σ²·A_F⁻¹. We get A_F⁻¹ once via
        # cho_solve(I) rather than via diag-tricks, since we need the full
        # matrix for Ve, per-coef SEs, and predict().
        A_inv = cho_solve((A_chol, A_chol_lower), np.eye(p))
        if fit_F.w is None or np.allclose(fit_F.w, 1.0):
            XtWX = XtX
        else:
            Xw = X * np.sqrt(fit_F.w)[:, None]
            XtWX = Xw.T @ Xw
        A_inv_XtWX = A_inv @ XtWX
        # Per-coefficient edf = diag(F) where F = A⁻¹ X'WX. F is not
        # symmetric, so individual diag entries can be negative — mgcv
        # reports them verbatim (matches m$edf), and the per-smooth sum
        # remains non-negative and interpretable.
        edf = np.diag(A_inv_XtWX).copy()
        edf_total = float(edf.sum())
        # Prior weights (PIRLS uses ones today; binomial size / offset / prior-w
        # land later). Stored so residuals_of and Pearson-scale share the same
        # weights PIRLS fit with.
        self._wt = np.ones(n)
        wt = self._wt
        # df.residual used in mgcv = n - edf_total. For unknown-scale
        # families fit by REML through the (ρ, log φ) outer optimizer, mgcv
        # reports `m$scale = reml.scale = exp(log φ̂)` (gam.fit3.r:639). The
        # Pearson estimator Σwt·(y-μ)²/V(μ)/df_resid is also kept around
        # under `m._pearson_scale` since it's mgcv's `scale.est` and is
        # what the GCV path returns. For Gaussian-identity (φ profiled out
        # of the outer vector, _log_phi_hat=None) this falls through to the
        # Pearson formula, which for V=1/wt=1 collapses to rss/df_resid —
        # bit-identical to the pre-Phase-2 Gaussian flow.
        df_resid = float(n - edf_total)
        if df_resid > 0 and not self.family.scale_known:
            V = self.family.variance(fit.mu)
            pearson_scale = float(np.sum(wt * (y - fit.mu) ** 2 / V)) / df_resid
        else:
            pearson_scale = 1.0 if self.family.scale_known else float("nan")
        self._pearson_scale = pearson_scale
        if self.family.scale_known:
            scale = 1.0
        elif self._log_phi_hat is not None:
            scale = float(np.exp(self._log_phi_hat))
        else:
            scale = pearson_scale
        sigma_squared = scale                 # alias kept for back-compat
        sigma = float(np.sqrt(sigma_squared)) if np.isfinite(sigma_squared) and sigma_squared >= 0 else float("nan")

        Vp = sigma_squared * A_inv
        Ve = sigma_squared * A_inv_XtWX @ A_inv

        # ------------- attribute assembly ----------------------------------
        self.bhat = _row_frame(beta, column_names)
        self._beta = beta
        se = np.sqrt(np.diag(Vp))
        self.se_bhat = _row_frame(se, column_names)
        self._se = se
        # Wald stats — useful for the parametric-row summary table; smooth
        # rows use the chi-squared-style test built on F per smooth, not per
        # basis column.
        t_stats = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
        self.t_values = _row_frame(t_stats, column_names)
        # Use Student-t on df.residual (parametric Wald in mgcv summary).
        if df_resid > 0 and np.isfinite(df_resid):
            pv = 2 * t_dist.sf(np.abs(t_stats), df_resid)
        else:
            pv = np.full_like(t_stats, np.nan)
        self.p_values = _row_frame(pv, column_names)

        eta = fit.eta
        mu = fit.mu
        self.linear_predictors = eta
        self.fitted_values = mu
        self.fitted = mu                      # alias; for Gaussian μ = η
        # Default residuals = deviance residuals (mgcv default). For Gaussian
        # with prior weights = 1, sign(y-μ)·√((y-μ)²) = (y-μ), so the existing
        # Gaussian RSS-based summaries stay bit-identical.
        self.residuals = self._deviance_residuals(y, mu, self._wt)
        self.sigma = sigma
        self.sigma_squared = sigma_squared
        self.scale = sigma_squared            # mgcv's `$scale`
        self.df_residuals = df_resid
        # Family deviance: `_FitState.dev` already holds Σ family.dev_resids
        # (Gaussian path: same as RSS). Keep `m.rss` as an alias for the
        # Gaussian-era name; new code should read `m.deviance`.
        self.deviance = float(fit.dev)
        self.rss = self.deviance              # alias (Gaussian: dev = rss)

        # Null deviance: deviance of the intercept-only model. For an intercept-
        # only GLM the score equation gives μ̂ = weighted mean of y for any
        # link (μ is constant and the weighted mean is the unique solution).
        # Without an intercept fall back to η ≡ 0 ⇒ μ ≡ linkinv(0). Mirrors
        # `glm.fit`'s `wtdmu`. For Gaussian (V=1, wt=1) with intercept this
        # reduces to Σ(y - mean(y))² = tss; without intercept to Σy² = yty.
        if has_intercept:
            mu_null_const = float(np.sum(wt * y) / np.sum(wt))
            mu_null = np.full(n, mu_null_const)
        else:
            mu_null = self.family.link.linkinv(np.zeros(n))
        self.null_deviance = float(np.sum(self.family.dev_resids(y, mu_null, wt)))
        self.df_null = float(n - 1) if has_intercept else float(n)

        self.Vp = Vp
        self.Ve = Ve
        self._A_inv = A_inv
        self.edf = edf
        self.edf_total = edf_total
        # Per-smooth edf: sum over the block's column range. Multi-block
        # smooths (by=factor) still roll up to a per-label dict — mgcv prints
        # one line per block.
        edf_by_smooth: dict[str, float] = {}
        for b, (a, bcol) in zip(blocks, block_col_ranges):
            edf_by_smooth[b.label] = float(edf[a:bcol].sum())
        self.edf_by_smooth = edf_by_smooth

        # Response-scale residual SS is what mgcv's r.sq is built on (uses
        # `object$y - object$fitted.values`, not deviance residuals — see
        # `summary.gam` line ~4055 in mgcv 1.9). For Gaussian-identity with
        # an intercept, sum(y - μ) = 0 from the unpenalized intercept's score
        # equation, so the variance-based formula reduces algebraically to
        # `1 - rss·(n-1)/(tss·df_resid)`, matching the legacy
        # `1 - (1 - rss/tss)(n-1)/df_resid` exactly.
        ss_resid_response = float(np.sum(wt * (y - mu) ** 2))
        if has_intercept and tss > 0:
            r_squared = 1.0 - ss_resid_response / tss
        elif yty > 0:
            r_squared = 1.0 - ss_resid_response / yty
        else:
            r_squared = float("nan")
        # mgcv's r.sq formula: 1 - var(√w·(y-μ))·(n-1) / (var(√w·(y-mean.y))·df_resid)
        # with var() = unbiased sample variance (denom n-1), matching R's var().
        if df_resid > 0 and n > 1:
            sqrt_wt = np.sqrt(wt)
            mean_y_w = float(np.sum(wt * y) / np.sum(wt))
            v_resid = float(np.var(sqrt_wt * (y - mu), ddof=1))
            v_total = float(np.var(sqrt_wt * (y - mean_y_w), ddof=1))
            if v_total > 0:
                r_squared_adjusted = 1.0 - v_resid * (n - 1) / (v_total * df_resid)
            else:
                r_squared_adjusted = float("nan")
        else:
            r_squared_adjusted = float("nan")
        self.r_squared = float(r_squared)
        self.r_squared_adjusted = float(r_squared_adjusted)
        # Deviance explained — mgcv: (null.deviance - deviance) / null.deviance.
        if self.null_deviance > 0:
            self.deviance_explained = float(
                (self.null_deviance - self.deviance) / self.null_deviance
            )
        else:
            self.deviance_explained = float("nan")

        # Augmented REML Hessian wrt (ρ, log σ²) — both edf12 (Vr in Vc1
        # and Vc2) and vcomp (CIs on log σ_k) need it. Computed once and
        # cached. For GCV / no-smooth / non-finite σ², leave as None and
        # the consumers fall back to whatever they can do.
        if (
            method == "REML"
            and n_sp > 0
            and np.isfinite(sigma_squared)
            and sigma_squared > 0
        ):
            log_phi_hat_for_aug = (
                self._log_phi_hat
                if self._log_phi_hat is not None
                else float(np.log(sigma_squared))
            )
            H_aug = 0.5 * self._reml_hessian(
                rho_hat, log_phi_hat_for_aug, fit=fit, include_log_phi=True,
            )
            H_aug = 0.5 * (H_aug + H_aug.T)
        else:
            H_aug = None
        self._H_aug = H_aug
        # mgcv's df rule (`logLik.gam`): use sum(edf2) when available, where
        # edf2 is the sp-uncertainty-corrected df from Wood 2017 §6.11.3.
        # edf alone systematically under-counts because it conditions on the
        # estimated λ; edf2 = diag((σ²A⁻¹ + Vc1 + Vc2) X'X)/σ² absorbs the
        # extra variance from λ̂. Vc1 = (∂β/∂ρ) Vr (∂β/∂ρ)ᵀ is the obvious
        # bit; Vc2 = σ² Σ_{i,j} Vr[i,j] M_i M_j^T accounts for the
        # ρ-dependence of L^{-T} in the Bayesian draw β̃ = β̂ + σ L^{-T} z.
        # edf1 = tr(2F-F²) is the upper bound; cap edf2 at edf1 in total
        # only. sc.p = 1 if scale is estimated, 0 if known (mgcv convention).
        if n_sp > 0:
            edf2_per_coef, edf1_per_coef = self._compute_edf12(
                rho_hat, fit, sigma_squared, A_inv, A_inv_XtWX, edf, H_aug,
            )
            self.edf1 = edf1_per_coef
            self.edf2 = edf2_per_coef
            self.edf1_total = float(edf1_per_coef.sum())
            self.edf2_total = float(edf2_per_coef.sum())
        else:
            self.edf1 = edf.copy()
            self.edf2 = edf.copy()
            self.edf1_total = edf_total
            self.edf2_total = edf_total

        # AIC / logLik via mgcv's logLik.gam machinery (mgcv.r:4420):
        #   m$aic = family.aic(y, μ, dev1, wt, n) + 2·sum(edf)         (mgcv.r:1843)
        #   logLik(m) = sum(edf) + sc.p − m$aic/2                       (mgcv.r:4428)
        #   df_for_AIC = min(sum(edf2) + sc.p,  p_coef + sc.p)          (mgcv.r:4431-33)
        #   AIC(m) = -2·logLik(m) + 2·df_for_AIC                        (R's AIC.default)
        # `dev1` is family-specific (Gaussian uses dev directly, the Pearson
        # σ̂² is moment-based for the rest); see Family._aic_dev1.
        sc_p = 0.0 if self.family.scale_known else 1.0
        dev1 = self.family._aic_dev1(self.deviance, scale, wt)
        family_aic = float(self.family.aic(y, fit.mu, dev1, wt, n))
        mgcv_aic = family_aic + 2.0 * edf_total                    # mgcv's m$aic
        logLik = sc_p + edf_total - 0.5 * mgcv_aic                 # mgcv's logLik value
        df_for_aic = min(self.edf2_total + sc_p, float(p) + sc_p)  # capped at np
        self.loglike = float(logLik)
        self.logLik = self.loglike                                 # alias (mgcv-style name)
        self.npar = float(df_for_aic)
        self.AIC = -2.0 * logLik + 2.0 * df_for_aic
        self.BIC = -2.0 * logLik + float(np.log(n)) * df_for_aic
        self._mgcv_aic = float(mgcv_aic)                           # mgcv's m$aic (different from AIC!)

        if method == "REML":
            if n_sp > 0:
                # `_reml` returns -2·V_R; `summary()`'s `/2` recovers
                # mgcv's `-REML` display value. Scale-known families (Poisson,
                # Binomial) substitute log φ = 0; scale-unknown read the
                # outer-optimizer's converged log φ̂.
                log_phi_hat = (
                    self._log_phi_hat if self._log_phi_hat is not None else 0.0
                )
                self.REML_criterion = float(
                    self._reml(rho_hat, log_phi_hat, fit=fit)
                )
            else:
                self.REML_criterion = float("nan")
        else:
            if n_sp > 0:
                self.GCV_score = float(self._gcv(rho_hat))
            else:
                self.GCV_score = float("nan")

        # Variance components: σ² and the implied per-slot std.dev's
        # σ_k = σ/√sp_k, with delta-method CIs (REML only). Mirrors mgcv's
        # gam.vcomp(rescale=FALSE). Cheap to compute eagerly for typical
        # n_sp; users can ignore the attribute if they don't need it.
        self.vcomp = self._compute_vcomp()

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------

    def _build_S_lambda(self, rho: np.ndarray) -> np.ndarray:
        """Assemble the full p×p penalty matrix Sλ at log-smoothing-params ρ.

        Each slot's k×k S_j is placed at its block's column range and
        multiplied by λ = exp(ρᵢ). Slots within the same block overlap
        (same col range) and are summed there — that's how tensor smooths
        get multiple penalties per block."""
        p = self.p
        Sλ = np.zeros((p, p))
        for rho_i, slot in zip(rho, self._slots):
            lam = float(np.exp(rho_i))
            a, b = slot.col_start, slot.col_end
            Sλ[a:b, a:b] += lam * slot.S
        return Sλ

    def _initial_sp_rho(self) -> np.ndarray:
        """mgcv's ``initial.sp`` seed for log-smoothing-params (gam.fit3.r).

        For each smooth k:

            def.sp[k] = mean(diag(X_k'X_k)[ind]) / mean(diag(S_k)[ind])

        where ``ind`` filters S_k to its penalised rows/cols using the
        ``thresh = .Machine$double.eps^0.8 * max(|S_k|)`` test on row-mean,
        col-mean, and diagonal magnitudes simultaneously. ``X_k`` is the
        block of the design matrix for the smooth's columns. Returns
        log(def.sp) — i.e. the ρ-space seed.
        """
        X = self._X_full
        ldxx = np.einsum("ij,ij->j", X, X)  # column sums of squares
        n_sp = len(self._slots)
        rho0 = np.zeros(n_sp)
        for k, slot in enumerate(self._slots):
            S_k = slot.S
            absS = np.abs(S_k)
            maS = float(absS.max()) if absS.size else 0.0
            if maS <= 0.0:
                rho0[k] = 0.0
                continue
            thresh = float(np.finfo(float).eps ** 0.8) * maS
            rsS = absS.mean(axis=1)
            csS = absS.mean(axis=0)
            dS = np.abs(np.diag(S_k))
            ind = (rsS > thresh) & (csS > thresh) & (dS > thresh)
            if not np.any(ind):
                rho0[k] = 0.0
                continue
            ss = np.diag(S_k)[ind]
            xx = ldxx[slot.col_start:slot.col_end][ind]
            sizeXX = float(np.mean(xx))
            sizeS = float(np.mean(ss))
            if sizeS <= 0.0 or sizeXX <= 0.0:
                rho0[k] = 0.0
                continue
            rho0[k] = float(np.log(sizeXX / sizeS))
        return rho0

    def _fit_given_rho(self, rho: np.ndarray) -> "_FitState":
        """Penalized IRLS at log-smoothing-params ρ.

        Iterate Newton-form working weights/responses

            αᵢ = 1 + (yᵢ − μᵢ)·(V'(μᵢ)/V(μᵢ) + g''(μᵢ)·dμᵢ/dηᵢ)
            wᵢ = αᵢ · (dμᵢ/dηᵢ)² / V(μᵢ)
            zᵢ = ηᵢ + (yᵢ − μᵢ) / ((dμᵢ/dηᵢ)·αᵢ)

        and solve ``(X'WX + Sλ)β = X'Wz`` by Cholesky each step. The Newton
        form (vs. plain Fisher PIRLS, which uses ``α=1``) makes the converged
        ``H = X'WX + Sλ`` the *observed* penalized Hessian, which is what
        the implicit-function ``∂β̂/∂ρ = -exp(ρ_k) H⁻¹ S_k β̂`` derivation
        assumes — and matches mgcv's gam.fit3 default for non-canonical
        links. For canonical links (incl. Gaussian-identity, Poisson-log,
        Gamma-inverse) ``α ≡ 1`` so Newton == Fisher.

        Step-halving (mgcv's "inner loop 3") is applied if the penalized
        deviance increases beyond a small threshold; convergence is on
        |Δpdev|/(0.1+|pdev|) < ε.
        """
        family = self.family
        link = family.link
        X = self._X_full
        y = self._y_arr
        n, p = self.n, self.p
        Sλ = self._build_S_lambda(rho)
        Sλ = 0.5 * (Sλ + Sλ.T)
        wt = np.ones(n)                 # prior weights = 1 (no offset/prior-w yet)

        # Start μ̂ from the family's mustart (= y for Gamma/IG). The
        # *baseline* for step-halving and divergence is mgcv's ``null.coef``
        # pattern: project a constant valid η onto colspan(X) so that the
        # triple (β_null, η_null, μ_null) lives inside the family's valid
        # region for every canonical link. The plain β=0 ⇒ η=0 baseline
        # fails for canonical IG (1/μ² requires η>0 finite) — halving an
        # invalid η_new toward η_old=0 never escapes — and using the
        # saturated η as baseline gives old_pdev=0, so any positive iter-1
        # pdev would look like divergence.
        mu = family.initialize(y, wt)
        eta = link.link(mu)
        beta = np.zeros(p)

        mu_null_const = float(np.average(mu, weights=wt))
        eta_null_const = link.link(np.full(n, mu_null_const))
        null_coef, *_ = np.linalg.lstsq(X, eta_null_const, rcond=None)
        eta_null = X @ null_coef
        mu_null = link.linkinv(eta_null)
        if not (link.valideta(eta_null) and family.validmu(mu_null)):
            # Constant-η projection drifted out of valid region — only
            # plausible for an X with no near-constant column. Fall back
            # to zeros; if the canonical link rejects η=0 the user will
            # still get a clear error from the validity step-halver below
            # rather than silent divergence.
            null_coef = np.zeros(p)
            eta_null = np.zeros(n)
            mu_null = link.linkinv(eta_null)
        beta_old = null_coef.copy()
        eta_old = eta_null.copy()
        dev = float(np.sum(family.dev_resids(y, mu, wt)))
        # mgcv: old.pdev = sum(dev.resids at null) + null.coef' St null.coef.
        old_pdev = (float(np.sum(family.dev_resids(y, mu_null, wt)))
                    + float(null_coef @ Sλ @ null_coef))

        # mgcv startup loop: if family.initialize returns a boundary value
        # (rare; e.g., Bernoulli at y=0/1 with linkinv-clamped initialize),
        # nudge η toward the null baseline until valid. Typically a no-op.
        ii = 0
        while not (link.valideta(eta) and family.validmu(mu)):
            ii += 1
            if ii > 20:
                raise FloatingPointError(
                    "PIRLS init: cannot find valid starting μ̂"
                )
            eta = 0.9 * eta + 0.1 * eta_old
            mu = link.linkinv(eta)

        eps = 1e-8
        max_it = 50
        for it in range(max_it):
            mu_eta_v = link.mu_eta(eta)
            V = family.variance(mu)
            if np.any(V == 0) or np.any(np.isnan(V)):
                raise FloatingPointError("V(μ)=0 or NaN in PIRLS")
            # mgcv's gam.fit3 IRLS uses Fisher weights w = μ_η²/V (gam.fit3.r
            # line ~270). For canonical links the Newton-form full-Hessian
            # weight α·μ_η²/V coincides (α≡1 by canonical identity); for
            # non-canonical (Gamma+log, Gaussian+log, ...) Fisher and Newton
            # give different β̂ — and mgcv ships Fisher. Wood 2011 derives
            # exact ∂/∂ρ derivatives starting from the Fisher-converged β̂,
            # which is what we replicate.
            alpha = np.ones(n)
            z = eta + (y - mu) / mu_eta_v
            w = mu_eta_v ** 2 / V

            XtWX = (X.T * w) @ X
            XtWz = X.T @ (w * z)
            A = XtWX + Sλ
            A = 0.5 * (A + A.T)
            try:
                A_chol, lower = cho_factor(A, lower=True, overwrite_a=False)
            except np.linalg.LinAlgError:
                ridge = 1e-8 * np.trace(A) / p
                A_chol, lower = cho_factor(
                    A + ridge * np.eye(p), lower=True, overwrite_a=False,
                )
            start = cho_solve((A_chol, lower), XtWz)
            eta_new = X @ start
            if np.any(~np.isfinite(start)):
                raise FloatingPointError("non-finite β in PIRLS")

            mu_new = link.linkinv(eta_new)
            # If μ leaves the family's valid region, halve the step toward
            # the previous iterate (mgcv "inner loop 2").
            ii = 0
            while not (link.valideta(eta_new) and family.validmu(mu_new)):
                ii += 1
                if ii > max_it:
                    raise FloatingPointError("PIRLS step halving failed (validity)")
                start = 0.5 * (start + beta_old)
                eta_new = 0.5 * (eta_new + eta_old)
                mu_new = link.linkinv(eta_new)

            dev_new = float(np.sum(family.dev_resids(y, mu_new, wt)))
            pen_new = float(start @ Sλ @ start)
            pdev_new = dev_new + pen_new

            # mgcv "inner loop 3": step-halve toward old iterate while the
            # penalized deviance is increasing meaningfully.
            div_thresh = 10.0 * (0.1 + abs(old_pdev)) * (np.finfo(float).eps ** 0.5)
            ii = 0
            while pdev_new - old_pdev > div_thresh:
                ii += 1
                if ii > 100:
                    break
                start = 0.5 * (start + beta_old)
                eta_new = 0.5 * (eta_new + eta_old)
                mu_new = link.linkinv(eta_new)
                if not (link.valideta(eta_new) and family.validmu(mu_new)):
                    continue
                dev_new = float(np.sum(family.dev_resids(y, mu_new, wt)))
                pen_new = float(start @ Sλ @ start)
                pdev_new = dev_new + pen_new

            beta = start
            eta = eta_new
            mu = mu_new
            dev = dev_new
            pen = pen_new

            # mgcv convergence: |Δpdev| < ε·(|scale|+|pdev|). Without scale
            # available here (it's profiled outside or known), use 1 as the
            # scale floor — the criterion is ratio-based and works on the
            # trees example.
            if abs(pdev_new - old_pdev) < eps * (1.0 + abs(pdev_new)):
                break
            old_pdev = pdev_new
            beta_old = beta.copy()
            eta_old = eta.copy()

        # Final consistent state (recompute w, z, alpha at converged β̂ for
        # downstream derivative routines — they expect these exact values).
        # PIRLS inner loop above used Fisher W (matches mgcv gam.fit3.r:270).
        # For the analytical score (REML / GCV) and its ρ-derivatives we use
        # the Newton-form "exact" W = α · μ_η² / V (Wood 2011). At the
        # PIRLS-converged β̂ both Fisher and Newton solve the same penalized-
        # score equation (so β̂ is invariant), but the log|X'WX + Sλ| term
        # and the chain-rule ingredients (dw/dη, d²w/dη²) depend on which
        # W enters. mgcv's score computation uses Newton W; we evaluate α
        # at the Fisher-converged β̂ here so downstream code sees Newton W.
        mu_eta_v = link.mu_eta(eta)
        V = family.variance(mu)
        d2g = link.d2link(mu)
        alpha = 1.0 + (y - mu) * (family.dvar(mu) / V + d2g * mu_eta_v)
        alpha = np.where(alpha == 0.0, np.finfo(float).eps, alpha)
        z = eta + (y - mu) / (mu_eta_v * alpha)
        w = alpha * mu_eta_v ** 2 / V
        is_fisher_fallback = False
        if np.any(w < 0):
            # Newton W has negative entries → fall back to Fisher in the
            # score too (drop α'/α terms accordingly).
            alpha = np.ones(n)
            z = eta + (y - mu) / mu_eta_v
            w = mu_eta_v ** 2 / V
            is_fisher_fallback = True

        XtWX = (X.T * w) @ X
        A = XtWX + Sλ
        A = 0.5 * (A + A.T)
        try:
            A_chol, lower = cho_factor(A, lower=True, overwrite_a=False)
        except np.linalg.LinAlgError:
            ridge = 1e-8 * np.trace(A) / p
            A_chol, lower = cho_factor(
                A + ridge * np.eye(p), lower=True, overwrite_a=False,
            )
        log_det_A = 2.0 * float(np.log(np.abs(np.diag(A_chol))).sum())

        return _FitState(
            beta=beta, dev=dev, pen=pen,
            A_chol=A_chol, A_chol_lower=lower,
            S_full=Sλ, log_det_A=log_det_A,
            eta=eta, mu=mu, w=w, z=z, alpha=alpha,
            is_fisher_fallback=is_fisher_fallback,
        )

    def _log_det_S_pos(self, rho: np.ndarray) -> float:
        """log|Sλ|_+ — log-determinant of Sλ on its fixed range space.

        The range space is fixed (dimension p − Mp, set at init from the
        *structural* penalty), and we take the top ``penalty_rank``
        eigenvalues by magnitude. This is what makes the REML criterion
        push back against λ_j → 0: those directions still count, and their
        vanishing eigenvalues drive ``log(λ_small) → −∞``. A pure
        ``eigenvalue > tol`` filter would silently drop them and remove
        the penalty — exactly the failure mode for tensor / by-factor
        smooths with multiple λ's.
        """
        r = self._penalty_rank
        if r <= 0:
            return 0.0
        Sλ = self._build_S_lambda(rho)
        Sλ = 0.5 * (Sλ + Sλ.T)
        w = np.linalg.eigvalsh(Sλ)
        # Take the top-r eigenvalues (descending). Clip to a tiny positive
        # floor so we don't take log of an FP-noise negative; exact-zero
        # null-space directions are excluded by the rank cap.
        w_sorted = np.sort(w)[::-1]
        top = w_sorted[:r]
        top = np.clip(top, 1e-300, None)
        return float(np.sum(np.log(top)))

    def _reml(self, rho: np.ndarray, log_phi: float = 0.0,
                      fit: "_FitState | None" = None) -> float:
        """Laplace-approximate REML in 2·V_R units, family/link-agnostic.

        Direct port of mgcv's gam.fit3.r:616 (γ=1, remlInd=1):

            2·V_R = Dp/φ − 2·ls0 + log|X'WX + Sλ| − log|Sλ|_+ − Mp·log(2π·φ)

        with Dp = fit.dev + β̂'Sλβ̂ at PIRLS-converged β̂ and
        ls0 = family.ls(y, wt, φ)[0]. ``fit.log_det_A`` is the un-φ-scaled
        log|X'WX + Sλ|; the φ-coefficients of the prior-normalisation term
        and the Hessian/penalty Jacobi cancel everywhere except the
        −Mp·log(2π·φ) prior-rank term — see the Laplace derivation in
        Wood 2017 §6.6.

        Reduction-to-Gaussian: profile out φ̂ = Dp/(n−Mp) and substitute.
        With Gaussian ls0 = −n·log(2πφ)/2 (wt=1 ⇒ Σlog wt = 0),

            2·V_R(φ̂) = (n−Mp)·(1 + log(2π·Dp/(n−Mp)))
                       + log|A| − log|S|_+

        which equals ``_reml(rho)`` exactly. Verified numerically by
        ``test_reml_reduces_to_profiled_gaussian``.

        For scale-known families (Poisson, Binomial) φ ≡ 1 ⇒ log_phi=0
        ⇒ ``Mp·log(2π·φ)`` = Mp·log(2π); ls0 then carries the entire
        likelihood contribution, which is exactly mgcv's behaviour.
        """
        if fit is None:
            fit = self._fit_given_rho(rho)
        Dp = fit.dev + fit.pen
        if Dp <= 0 or not np.isfinite(Dp):
            return 1e15
        Mp = float(self._Mp)
        phi = float(np.exp(log_phi))
        if not (np.isfinite(phi) and phi > 0):
            return 1e15
        # Prior weights placeholder. PIRLS uses the same `np.ones(n)` today;
        # when the user-facing ``weights=`` arg lands, both paths read from
        # ``self._wt_prior``. ``family.ls`` returns (ls0, d_ls/d_log_φ,
        # d²_ls/d_log_φ²) — Phase 2.1 only needs ls0; the derivatives feed
        # the (rho, log φ) Hessian in Phase 3.
        wt = np.ones(self.n)
        ls0 = float(self.family.ls(self._y_arr, wt, phi)[0])
        log_det_S = self._log_det_S_pos(rho)
        return (
            Dp / phi
            - 2.0 * ls0
            + fit.log_det_A
            - log_det_S
            - Mp * float(np.log(2.0 * np.pi * phi))
        )

    def _reml_grad(self, rho: np.ndarray, log_phi: float = 0.0,
                           fit: "_FitState | None" = None,
                           include_log_phi: bool = False) -> np.ndarray:
        """Analytical gradient of `_reml` (2·V_R units).

        Length n_sp if `include_log_phi=False`, else n_sp+1 with log_phi
        appended. Wood 2011 §4 + mgcv gam.fit3.r:622, 630:

            ∂(2·V_R)/∂ρ_k    = (∂Dp/∂ρ_k)/φ + ∂log|H|/∂ρ_k − ∂log|S|+/∂ρ_k
            ∂(2·V_R)/∂log φ  = −Dp/φ − 2·ls'_lmpy − Mp

        ls'_lmpy is the d/d(log φ) chain-rule output from `family.ls(y, wt, φ)[1]`
        (lmpy convention, see family.py:338 docstring).
        """
        if fit is None:
            fit = self._fit_given_rho(rho)
        n_sp = len(self._slots)
        phi = float(np.exp(log_phi))
        if not (np.isfinite(phi) and phi > 0):
            size = n_sp + (1 if include_log_phi else 0)
            return np.full(size, 1e15)

        if n_sp == 0:
            grad_rho = np.zeros(0)
        else:
            dDp = self._dDp_drho(fit, rho)
            dlog_H = self._dlog_det_H_drho(fit, rho)
            dlog_S = self._dlog_det_S_drho(rho, S_full=fit.S_full)
            grad_rho = dDp / phi + dlog_H - dlog_S

        if not include_log_phi:
            return grad_rho

        Mp = float(self._Mp)
        wt = np.ones(self.n)
        Dp = fit.dev + fit.pen
        ls = np.asarray(self.family.ls(self._y_arr, wt, phi), dtype=float)
        ls1 = float(ls[1])    # d ls / d(log φ), already chain-ruled
        d_logphi = -Dp / phi - 2.0 * ls1 - Mp
        return np.concatenate([grad_rho, [d_logphi]])

    def _reml_hessian(self, rho: np.ndarray, log_phi: float = 0.0,
                              fit: "_FitState | None" = None,
                              include_log_phi: bool = False) -> np.ndarray:
        """Analytical Hessian of `_reml` (2·V_R units).

        Returns ((n_sp+1) × (n_sp+1)) when ``include_log_phi=True``, else
        (n_sp × n_sp). Wood 2011 §4 for non-Gaussian, with Newton-form W:

          ∂²(2·V_R)/∂ρ_l∂ρ_k = (1/φ)·∂²Dp/∂ρ_l∂ρ_k
                              + ∂²log|H|/∂ρ_l∂ρ_k
                              − ∂²log|S|+/∂ρ_l∂ρ_k

        Pieces:

          ∂²Dp/∂ρ_l∂ρ_k    = δ_lk·g_k − 2·λ_l·λ_k·β̂' S_l A⁻¹ S_k β̂   (Gaussian form)

          ∂²log|S|+/∂ρ_l∂ρ_k = δ_lk·λ_k·tr(S⁺ S_k)
                              − λ_l·λ_k·tr(S⁺ S_l S⁺ S_k)         (Gaussian form)

          ∂²log|H|/∂ρ_l∂ρ_k = −tr(H⁻¹·∂H/∂ρ_l·H⁻¹·∂H/∂ρ_k)
                              + tr(H⁻¹·∂²H/∂ρ_l∂ρ_k)

        with ∂H/∂ρ_l = X' diag(h'·v_l) X + λ_l S_l (v_l := X·dβ_l) and

          ∂²H/∂ρ_l∂ρ_k = X' diag(h''·v_l·v_k + h'·X·d²β_lk) X
                         + δ_lk·λ_l·S_l

        Cross-derivatives wrt log φ:

          ∂²(2·V_R)/∂ρ_k∂log φ = −g_k / φ
          ∂²(2·V_R)/∂log φ²    = Dp/φ − 2·ls'_lmpy_2

        where ``ls'_lmpy_2 = family.ls(y, wt, φ)[2]`` (chain-ruled to log φ).

        For Gaussian-identity (h' ≡ h'' ≡ 0) only the SS Wood block and the
        Gaussian Dp/log|S|+ pieces survive, so the result equals 2·`_reml_hessian`
        in the unprofiled REML formulation (the existing `_reml_hessian`
        operates on the φ-profiled Gaussian path and returns V_R-scale).
        """
        if fit is None:
            fit = self._fit_given_rho(rho)
        n_sp = len(self._slots)
        phi = float(np.exp(log_phi))
        size = n_sp + (1 if include_log_phi else 0)
        if not (np.isfinite(phi) and phi > 0):
            return np.full((size, size), 1e15)
        if n_sp == 0:
            H = np.zeros((size, size))
            if include_log_phi:
                Dp0 = fit.dev + fit.pen
                ls = np.asarray(self.family.ls(self._y_arr,
                                               np.ones(self.n), phi))
                H[0, 0] = Dp0 / phi - 2.0 * float(ls[2])
            return H

        p = self.p
        sp = np.exp(rho)
        X = self._X_full
        S_pinv = self._S_pinv(fit.S_full)

        # Common precomputations.
        M = cho_solve((fit.A_chol, fit.A_chol_lower), X.T)   # (p, n) = H⁻¹ X'
        d_diag = np.einsum("ij,ji->i", X, M)                  # (n,)  diag(X H⁻¹ X')
        P = X @ M                                              # (n, n) X H⁻¹ X'
        Rsq = P * P                                            # (n, n) elementwise

        db_drho = self._dbeta_drho(fit, rho)                   # (p, n_sp)
        dw_deta = self._dw_deta(fit)                           # (n,)
        d2w_deta2 = self._d2w_deta2(fit)                       # (n,)
        d2b = self._d2beta_drho_drho(fit, rho, db_drho=db_drho,
                                     dw_deta=dw_deta)          # (p, n_sp, n_sp)
        v = X @ db_drho                                        # (n, n_sp)
        hv = dw_deta[:, None] * v                              # h'·v_l, shape (n, n_sp)

        # Per-slot blocks reused for ∂²Dp / log|S|+ / log|H| Gaussian-style traces.
        AinvS_block: list[np.ndarray] = []
        SpinvS_block: list[np.ndarray] = []
        Sbeta_full = np.zeros((n_sp, p))
        AinvSbeta = np.empty((n_sp, p))
        diag_MtSM: list[np.ndarray] = []   # diag(M' S_k_full M) = (n,) for each k
        g = np.zeros(n_sp)
        tr_AinvS = np.zeros(n_sp)
        tr_SpinvS = np.zeros(n_sp)
        A_inv = cho_solve((fit.A_chol, fit.A_chol_lower), np.eye(p))
        for k, slot in enumerate(self._slots):
            a, b = slot.col_start, slot.col_end
            beta_k = fit.beta[a:b]
            Sb = slot.S @ beta_k
            Sbeta_full[k, a:b] = Sb
            AinvSbeta[k] = cho_solve(
                (fit.A_chol, fit.A_chol_lower), Sbeta_full[k]
            )
            g[k] = sp[k] * float(beta_k @ Sb)
            AinvS_block.append(A_inv[:, a:b] @ slot.S)
            SpinvS_block.append(S_pinv[:, a:b] @ slot.S)
            tr_AinvS[k] = float(np.einsum("ij,ji->", A_inv[a:b, a:b], slot.S))
            tr_SpinvS[k] = float(np.einsum("ij,ji->", S_pinv[a:b, a:b], slot.S))
            # diag(M' S_k_full M)_i = M[a:b, i]' · S_k · M[a:b, i]
            SkM = slot.S @ M[a:b, :]                          # (m_k, n)
            diag_MtSM.append(np.einsum("ji,ji->i", M[a:b, :], SkM))

        # Hessian assembly — symmetric loop.
        H2 = np.zeros((n_sp, n_sp))
        for i in range(n_sp):
            a_i, b_i = self._slots[i].col_start, self._slots[i].col_end
            for j in range(i, n_sp):
                a_j, b_j = self._slots[j].col_start, self._slots[j].col_end

                # ∂²Dp/∂ρ_i∂ρ_j: same family-agnostic form as Gaussian.
                bSiAinvSj_b = float(Sbeta_full[i] @ AinvSbeta[j])
                d2Dp = -2.0 * sp[i] * sp[j] * bSiAinvSj_b

                # tr(H⁻¹·∂H/∂ρ_i·H⁻¹·∂H/∂ρ_j) — four pieces.
                # WW: (h'·v_i)' · Rsq · (h'·v_j).
                tr_WW = float(hv[:, i] @ (Rsq @ hv[:, j]))
                # WS: tr(H⁻¹·A_i·H⁻¹·S_j) = (h'·v_i)' · diag_MtSM[j].
                tr_WS = float(hv[:, i] @ diag_MtSM[j])
                tr_SW = float(hv[:, j] @ diag_MtSM[i])
                # SS: tr(H⁻¹·S_i·H⁻¹·S_j) — Gaussian block trick.
                tr_SS = float(np.einsum(
                    "ab,ba->",
                    AinvS_block[i][a_j:b_j, :],
                    AinvS_block[j][a_i:b_i, :],
                ))
                tr_HinvHpHinvHp = (
                    tr_WW
                    + sp[j] * tr_WS
                    + sp[i] * tr_SW
                    + sp[i] * sp[j] * tr_SS
                )

                # tr(H⁻¹·∂²H/∂ρ_i∂ρ_j).
                #   X'·diag(h''·v_i·v_j)·X contribution: Σ d_i·h''·v_i·v_j.
                #   X'·diag(h'·X·d²β_ij)·X        contribution: Σ d_i·h'·(X·d²β_ij).
                Xd2b = X @ d2b[:, i, j]                       # (n,)
                tr_d2H = (
                    float(np.sum(d_diag * d2w_deta2 * v[:, i] * v[:, j]))
                    + float(np.sum(d_diag * dw_deta * Xd2b))
                )
                # δ_lk·λ_l·tr(H⁻¹·S_l) is the off-square diagonal term.
                d2logH_ij = -tr_HinvHpHinvHp + tr_d2H

                # ∂²log|S|+/∂ρ_i∂ρ_j Gaussian form.
                tr_SpSiSpSj = float(np.einsum(
                    "ab,ba->",
                    SpinvS_block[i][a_j:b_j, :],
                    SpinvS_block[j][a_i:b_i, :],
                ))
                d2logS_ij = -sp[i] * sp[j] * tr_SpSiSpSj

                cross_2VR = d2Dp / phi + d2logH_ij - d2logS_ij
                if i == j:
                    # Diagonal also picks up the δ_lk·g_k from ∂²Dp,
                    # δ_lk·λ_l·tr(H⁻¹·S_l) from ∂²H, and δ_lk·λ_k·tr(S⁺ S_k)
                    # from ∂²log|S|+.
                    H2[i, i] = (
                        cross_2VR
                        + g[i] / phi
                        + sp[i] * tr_AinvS[i]
                        - sp[i] * tr_SpinvS[i]
                    )
                else:
                    H2[i, j] = H2[j, i] = cross_2VR

        if not include_log_phi:
            return H2

        # Augment with log φ row/col.
        H_aug = np.zeros((n_sp + 1, n_sp + 1))
        H_aug[:n_sp, :n_sp] = H2
        for k in range(n_sp):
            cross = -g[k] / phi
            H_aug[k, n_sp] = cross
            H_aug[n_sp, k] = cross
        Dp = fit.dev + fit.pen
        ls = np.asarray(self.family.ls(self._y_arr, np.ones(self.n), phi))
        H_aug[n_sp, n_sp] = Dp / phi - 2.0 * float(ls[2])
        return H_aug

    def _outer_newton(
        self, theta0: np.ndarray, *, include_log_phi: bool,
        criterion: str = "REML",
        max_iter: int = 200, conv_tol: float = 1e-6,
        max_step: float = 5.0, max_half: int = 30,
    ) -> np.ndarray:
        """Unified analytical Newton on V_R(ρ, log φ) or V_g/V_u(ρ) — mgcv's gam.outer.

        Damped Newton with eigen-clamp on H, step cap, backtracking line
        search, and mgcv's two-part outer convergence test (Newton.r):

            max(|g_k|)   ≤ score_scale · conv_tol · 5
            |Δscore|     ≤ score_scale · conv_tol

        with ``score_scale = |scale.est| + |score|`` for GCV/UBRE and
        ``score_scale = |log(scale.est)| + |score|`` for REML.  The
        tolerance default ``1e-6`` matches mgcv's ``newton$conv.tol``.
        Works for any family — PIRLS inner solve degenerates to one
        Cholesky for Gaussian-identity (W=I, z=y).

        ``theta`` layout: ρ first, then a single log φ column when
        ``include_log_phi`` is set (unknown-scale REML). For known-scale
        REML (Poisson, Binomial) log φ is fixed at 0; for GCV.Cp log φ is
        always off the outer vector.

        ``criterion`` selects the objective:
        - ``"REML"``: minimizes V_R via ``_reml`` (returns 2·V_R, hence
          the 0.5 scaling), ``_reml_grad``, ``_reml_hessian``.
        - ``"GCV"``: minimizes V_g (scale-unknown) or V_u (scale-known)
          via ``_gcv``, ``_gcv_grad``, ``_gcv_hessian``. ``include_log_phi``
          must be False (GCV does not put log φ in the outer vector — φ̂
          is the Pearson estimate post-fit, not optimized).
        """
        if criterion not in ("REML", "GCV"):
            raise ValueError(f"criterion must be 'REML' or 'GCV', got {criterion!r}")
        if criterion == "GCV" and include_log_phi:
            raise ValueError("GCV path does not include log φ in outer θ.")

        n_sp = len(self._slots)
        theta = np.asarray(theta0, dtype=float).copy()

        def _split(t):
            if include_log_phi:
                return t[:n_sp], float(t[n_sp])
            return t, 0.0

        if criterion == "REML":
            def _eval(t):
                rho_t, lp_t = _split(t)
                try:
                    fit_t = self._fit_given_rho(rho_t)
                except Exception:
                    return float("inf"), None
                val_2VR = float(self._reml(rho_t, lp_t, fit=fit_t))
                return val_2VR / 2.0, fit_t
            def _grad(rho, log_phi, fit):
                return 0.5 * self._reml_grad(
                    rho, log_phi, fit=fit, include_log_phi=include_log_phi
                )
            def _hess(rho, log_phi, fit):
                return 0.5 * self._reml_hessian(
                    rho, log_phi, fit=fit, include_log_phi=include_log_phi
                )
        else:  # GCV
            def _eval(t):
                rho_t, _ = _split(t)
                try:
                    fit_t = self._fit_given_rho(rho_t)
                except Exception:
                    return float("inf"), None
                val = float(self._gcv(rho_t, fit=fit_t))
                return val, fit_t
            def _grad(rho, log_phi, fit):
                return self._gcv_grad(rho, fit=fit)
            def _hess(rho, log_phi, fit):
                return self._gcv_hessian(rho, fit=fit)

        is_reml = (criterion == "REML")

        def _score_scale(fit_, val):
            # mgcv's score.scale: |scale.est| + |score| (GCV/UBRE) or
            # |log(scale.est)| + |score| (REML). scale.est is mgcv's
            # Pearson estimator; for known-scale families it is 1.
            if self.family.scale_known:
                scale_est = 1.0
            else:
                y_arr = self._y_arr
                mu_arr = fit_.mu
                V_arr = self.family.variance(mu_arr)
                pearson = float(np.sum((y_arr - mu_arr) ** 2 / V_arr))
                fit_F_ = self._fisher_view(fit_)
                A_inv_ = cho_solve(
                    (fit_F_.A_chol, fit_F_.A_chol_lower), np.eye(self.p)
                )
                w_F = fit_F_.w
                if w_F is None or np.allclose(w_F, 1.0):
                    XtWX_ = self._XtX
                else:
                    Xw_ = self._X_full * np.sqrt(w_F)[:, None]
                    XtWX_ = Xw_.T @ Xw_
                tau_ = float(np.trace(A_inv_ @ XtWX_))
                df_resid_ = max(self.n - tau_, 1.0)
                scale_est = pearson / df_resid_
            if is_reml:
                # log(scale.est); guard against scale_est ≤ 0
                scale_est_safe = max(scale_est, 1e-300)
                return abs(np.log(scale_est_safe)) + abs(val)
            return abs(scale_est) + abs(val)

        f_prev, fit = _eval(theta)
        if fit is None:
            return theta

        for it in range(max_iter):
            rho, log_phi = _split(theta)
            grad = _grad(rho, log_phi, fit)
            H = _hess(rho, log_phi, fit)
            H = 0.5 * (H + H.T)

            # Eigen-clamp to PD: |w| with a tiny floor so the quadratic
            # model is positive-definite even on saddle/flat regions.
            w_eig, V_eig = np.linalg.eigh(H)
            w_max = float(np.abs(w_eig).max()) if w_eig.size > 0 else 1.0
            eps = max(w_max * 1e-7, 1e-12)
            w_pd = np.where(np.abs(w_eig) > eps, np.abs(w_eig), eps)
            d = -V_eig @ ((V_eig.T @ grad) / w_pd)

            d_norm = float(np.abs(d).max())
            if d_norm > max_step:
                d *= max_step / d_norm

            alpha = 1.0
            descent = False
            for _ in range(max_half):
                theta_try = theta + alpha * d
                f_try, fit_try = _eval(theta_try)
                if np.isfinite(f_try) and f_try < f_prev - 1e-14 * abs(f_prev):
                    descent = True
                    break
                alpha *= 0.5

            if not descent:
                break
            theta = theta_try
            df = abs(f_try - f_prev)
            f_old = f_prev
            f_prev = f_try
            fit = fit_try

            # mgcv's two-part stopping test (Newton.r):
            #   max(|grad|) ≤ score_scale·conv_tol·5
            #   |Δscore|    ≤ score_scale·conv_tol
            score_scale = _score_scale(fit, f_prev)
            if (
                float(np.abs(grad).max()) <= score_scale * conv_tol * 5.0
                and df <= score_scale * conv_tol
            ):
                break
            if df < 1e-12 * (1.0 + abs(f_prev)):
                break

        return theta

    def _S_pinv(self, S_full: np.ndarray) -> np.ndarray:
        """Pseudo-inverse of Sλ on its fixed range space.

        Eigendecompose Sλ and take the top ``penalty_rank`` eigenpairs,
        same convention as ``_log_det_S_pos`` so derivatives stay
        consistent with the determinant. Used by ``_reml_grad`` to
        compute ``∂log|S|+/∂ρ_k = λ_k tr(S^+ S_k)``.
        """
        r = self._penalty_rank
        if r <= 0:
            return np.zeros_like(S_full)
        Sλ = 0.5 * (S_full + S_full.T)
        w, V = np.linalg.eigh(Sλ)
        order = np.argsort(w)[::-1]
        w_top = np.clip(w[order[:r]], 1e-300, None)
        V_top = V[:, order[:r]]
        return (V_top / w_top) @ V_top.T

    def _fisher_view(self, fit: "_FitState") -> "_FitState":
        """Return a Fisher-W view of a PIRLS-converged fit.

        mgcv's GCV/UBRE score and reported m$edf use the Fisher weight
        ``W_F = μ_η²/V`` (gam.fit3.r:644), while the REML log|H+S| term
        uses the Newton "exact" weight ``W_N = α·μ_η²/V`` (gdi2.c). At
        PIRLS-converged β̂ both Fisher and Newton solve the same penalized
        score equation so β̂ is invariant; only the W that multiplies X
        in ``X'WX + Sλ`` differs. This helper rebuilds the Fisher
        factorization on top of the same β̂.

        For canonical-link or Fisher-fallback fits Newton ≡ Fisher and we
        return ``fit`` unchanged. ``is_fisher_fallback=True`` is set on
        the returned view so ``_dw_deta`` / ``_d2w_deta2`` skip the α'/α
        terms (consistent with W_F not carrying an α factor).
        """
        family = self.family
        eta = fit.eta
        mu = fit.mu
        # Canonical-link short circuit: α≡1 by canonical identity ⇒ W_F = W_N.
        if fit.is_fisher_fallback:
            return fit
        mu_eta = family.link.mu_eta(eta)
        V = family.variance(mu)
        W_F = mu_eta ** 2 / V
        if np.allclose(W_F, fit.w):
            return fit
        sqW_F = np.sqrt(W_F)
        Xw = self._X_full * sqW_F[:, None]
        XtWX_F = Xw.T @ Xw
        A_F = XtWX_F + fit.S_full
        A_F = 0.5 * (A_F + A_F.T)
        A_F_chol, lower = cho_factor(A_F, lower=False)
        log_det_A_F = 2.0 * float(np.sum(np.log(np.abs(np.diag(A_F_chol)))))
        return _FitState(
            beta=fit.beta, dev=fit.dev, pen=fit.pen,
            A_chol=A_F_chol, A_chol_lower=lower,
            S_full=fit.S_full, log_det_A=log_det_A_F,
            eta=eta, mu=mu, w=W_F, z=fit.z, alpha=np.ones(self.n),
            is_fisher_fallback=True,
        )

    def _dbeta_drho(self, fit: "_FitState",
                    rho: np.ndarray) -> np.ndarray:
        """Implicit-function-theorem derivative ∂β̂/∂ρ_k at PIRLS-converged β̂.

        The penalized score equation `s(β̂) = ∂ℓ/∂β |_β̂ - Sλ(ρ) β̂ = 0`
        differentiated in ρ_k gives, with H = -∂²ℓ_p/∂β∂β' = X'WX + Sλ
        (Newton-form W) at converged β̂:

            ∂β̂/∂ρ_k = -λ_k · H⁻¹ · S_k · β̂

        This holds for any family/link as long as PIRLS uses Newton weights
        (so X'WX = -∂²ℓ/∂β∂β' at β̂); for canonical links Newton ≡ Fisher
        and the formula reduces to the Gaussian case used implicitly in
        ``_reml_hessian``'s ``AinvSbeta``. Returns a (p, n_sp) array.
        """
        n_sp = len(self._slots)
        if n_sp == 0:
            return np.zeros((self.p, 0))
        sp = np.exp(rho)
        out = np.empty((self.p, n_sp))
        for k, slot in enumerate(self._slots):
            a, b = slot.col_start, slot.col_end
            Sk_beta_full = np.zeros(self.p)
            Sk_beta_full[a:b] = slot.S @ fit.beta[a:b]
            Ainv_Skb = cho_solve((fit.A_chol, fit.A_chol_lower), Sk_beta_full)
            out[:, k] = -sp[k] * Ainv_Skb
        return out

    def _dw_deta(self, fit: "_FitState") -> np.ndarray:
        """∂w_i/∂η_i at PIRLS-converged β̂. Length-n.

        PIRLS Newton weights are w(μ) = α(μ)·μ_eta(μ)²/V(μ) with
        α(μ) = 1 + (y-μ)·B(μ), B(μ) = V'/V + g''·μ_eta. Differentiating:

            ∂(log w)/∂μ = α'/α − 2·g''·μ_eta − V'/V
            α'(μ)       = −B(μ) + (y-μ)·B'(μ)
            B'(μ)       = V''/V − (V'/V)² + g'''·μ_eta − (g'')²·μ_eta²

        and dw/dη = (dw/dμ)·μ_eta = w·μ_eta·∂(log w)/∂μ.

        For canonical links the Newton form gives α≡1 (B≡0 by canonical
        identity g'V=1), so α'/α=0 and only the (-2·g''·μ_eta − V'/V)
        terms survive — that's the Fisher derivative. For
        ``fit.is_fisher_fallback`` we explicitly drop the α'/α term to
        stay consistent with the α=1 override the PIRLS path applied.
        """
        link = self.family.link
        family = self.family
        y = self._y_arr
        mu = fit.mu
        eta = fit.eta
        w = fit.w
        alpha = fit.alpha

        mu_eta = link.mu_eta(eta)
        V = family.variance(mu)
        Vp = family.dvar(mu)
        Vpp = family.d2var(mu)
        g2 = link.d2link(mu)
        g3 = link.d3link(mu)

        # α'/α term — set to zero for the Fisher fallback path.
        if fit.is_fisher_fallback:
            alpha_prime_over_alpha = np.zeros_like(mu)
        else:
            B = Vp / V + g2 * mu_eta
            Bp = Vpp / V - (Vp / V) ** 2 + g3 * mu_eta - g2 ** 2 * mu_eta ** 2
            alpha_prime = -B + (y - mu) * Bp
            alpha_prime_over_alpha = alpha_prime / alpha

        dlogw_dmu = alpha_prime_over_alpha - 2.0 * g2 * mu_eta - Vp / V
        return w * mu_eta * dlogw_dmu

    def _d2beta_drho_drho(self, fit: "_FitState", rho: np.ndarray,
                          db_drho: np.ndarray | None = None,
                          dw_deta: np.ndarray | None = None) -> np.ndarray:
        """∂²β̂/∂ρ_l∂ρ_k at PIRLS-converged β̂. Returns a (p, n_sp, n_sp) array.

        Differentiating dβ_k = -λ_k·H⁻¹·S_k·β̂ in ρ_l and using the IFT
        identity ∂H⁻¹/∂ρ_l = -H⁻¹·(∂H/∂ρ_l)·H⁻¹:

            ∂²β̂/∂ρ_l∂ρ_k = δ_lk · dβ_k
                          − H⁻¹ · (∂H/∂ρ_l) · dβ_k
                          − λ_k · H⁻¹ · S_k · dβ_l

        with ∂H/∂ρ_l = X'·diag(h'·v_l)·X + λ_l·S_l (v_l := X·dβ_l).
        Symmetric in (l, k) by construction of the formula:
            ∂²β̂/∂ρ_l∂ρ_k = δ_lk·dβ_k
                          − H⁻¹·X'·(h' · v_l · v_k)
                          − λ_l · H⁻¹·S_l·dβ_k
                          − λ_k · H⁻¹·S_k·dβ_l
        — the two S terms swap when (l, k) swap; the X'·(h'·v_l·v_k) term
        is invariant under the swap. Symmetry is exploited in the loop.

        For Gaussian-identity, h' ≡ 0 so the W-derivative term drops and
        the result reduces to the standard penalty-only IFT formula.
        """
        n_sp = len(self._slots)
        if n_sp == 0:
            return np.zeros((self.p, 0, 0))
        if db_drho is None:
            db_drho = self._dbeta_drho(fit, rho)
        sp = np.exp(rho)
        X = self._X_full
        v = X @ db_drho                     # (n, n_sp): v_l = X·dβ_l

        # h'(η) — only present for PIRLS fits (fit.w not None). Gaussian fast
        # path doesn't reach this method.
        if dw_deta is None:
            dw_deta = self._dw_deta(fit)

        # Per-slot S_k·dβ_k[a:b] in the embedded p-vector, stored once.
        Skdb_full = np.zeros((n_sp, self.p, n_sp))
        for k, slot in enumerate(self._slots):
            a, b = slot.col_start, slot.col_end
            for ll in range(n_sp):
                Skdb_full[k, a:b, ll] = slot.S @ db_drho[a:b, ll]

        out = np.empty((self.p, n_sp, n_sp))
        for k in range(n_sp):
            for l in range(k, n_sp):
                # H⁻¹·X'·(h' · v_l · v_k)  — the W-deriv contribution.
                rhs_W = X.T @ (dw_deta * v[:, l] * v[:, k])
                # H⁻¹·S_l·dβ_k (full p-vector, only nonzero at slot l's range)
                # and H⁻¹·S_k·dβ_l, embedded already in Skdb_full.
                rhs = (
                    rhs_W
                    + sp[l] * Skdb_full[l, :, k]
                    + sp[k] * Skdb_full[k, :, l]
                )
                # The implicit-function-theorem formula above:
                #   ∂²β̂/∂ρ_l∂ρ_k = δ_lk·dβ_k − H⁻¹·rhs_combined
                d2 = -cho_solve(
                    (fit.A_chol, fit.A_chol_lower), rhs
                )
                if l == k:
                    d2 = d2 + db_drho[:, k]
                out[:, l, k] = d2
                if l != k:
                    out[:, k, l] = d2
        return out

    def _d2w_deta2(self, fit: "_FitState") -> np.ndarray:
        """∂²w_i/∂η_i² at PIRLS-converged β̂. Length-n.

        Differentiating h(η) := w(η) twice (with y, ρ fixed; only η varies):

            d log h / dη   = μ_eta · D                where D = α'/α − 2 g'' μ_eta − V'/V
            d²h/dη²        = h · μ_eta² · (D² + D' − D · g'' · μ_eta)

        with D' = ∂D/∂μ:

            D' = α''/α − (α'/α)² − 2 g''' μ_eta + 2 (g'')² μ_eta² − V''/V + (V'/V)²
            α''(μ) = −2 B' + (y−μ) · B''
            B''(μ) = V'''/V − 3 V'·V''/V² + 2 V'³/V³
                     + g'''' μ_eta − 3 g'' g''' μ_eta² + 2 (g'')³ μ_eta³

        For the Fisher fallback path (PIRLS forced α=1 because Newton-w<0),
        α'/α and α''/α are both dropped — same convention as ``_dw_deta``.
        """
        link = self.family.link
        family = self.family
        y = self._y_arr
        mu = fit.mu
        eta = fit.eta
        w = fit.w
        alpha = fit.alpha

        mu_eta = link.mu_eta(eta)
        V = family.variance(mu)
        Vp = family.dvar(mu)
        Vpp = family.d2var(mu)
        Vppp = family.d3var(mu)
        g2 = link.d2link(mu)
        g3 = link.d3link(mu)
        g4 = link.d4link(mu)

        Vp_V = Vp / V
        Vpp_V = Vpp / V

        # B(μ) = V'/V + g''·μ_eta and its first derivative — already used in
        # `_dw_deta` for α'.
        Bp = Vpp_V - Vp_V ** 2 + g3 * mu_eta - g2 ** 2 * mu_eta ** 2
        # Second derivative B''(μ) = ∂B'/∂μ.
        Bpp = (
            Vppp / V - 3.0 * Vp * Vpp / (V * V) + 2.0 * Vp ** 3 / V ** 3
            + g4 * mu_eta - 3.0 * g2 * g3 * mu_eta ** 2
            + 2.0 * g2 ** 3 * mu_eta ** 3
        )

        if fit.is_fisher_fallback:
            alpha_prime_over_alpha = np.zeros_like(mu)
            alpha_pp_over_alpha = np.zeros_like(mu)
        else:
            B = Vp_V + g2 * mu_eta
            alpha_prime = -B + (y - mu) * Bp
            alpha_prime_over_alpha = alpha_prime / alpha
            alpha_pp = -2.0 * Bp + (y - mu) * Bpp
            alpha_pp_over_alpha = alpha_pp / alpha

        D = alpha_prime_over_alpha - 2.0 * g2 * mu_eta - Vp_V
        Dp = (
            alpha_pp_over_alpha - alpha_prime_over_alpha ** 2
            - 2.0 * g3 * mu_eta + 2.0 * g2 ** 2 * mu_eta ** 2
            - Vpp_V + Vp_V ** 2
        )
        return w * mu_eta ** 2 * (D ** 2 + Dp - D * g2 * mu_eta)

    def _dlog_det_S_drho(self, rho: np.ndarray,
                         S_pinv: np.ndarray | None = None,
                         S_full: np.ndarray | None = None) -> np.ndarray:
        """∂log|Sλ|+/∂ρ_k = λ_k · tr(S⁺ S_k). Length-n_sp.

        S⁺ is the rank-stable pseudo-inverse from `_S_pinv` (top
        ``penalty_rank`` eigenpairs of Sλ). For exact-rank-stable
        scenarios this matches the existing term in `_reml_grad`.
        """
        n_sp = len(self._slots)
        if n_sp == 0:
            return np.zeros(0)
        if S_pinv is None:
            if S_full is None:
                S_full = self._build_S_lambda(rho)
            S_pinv = self._S_pinv(S_full)
        sp = np.exp(rho)
        out = np.empty(n_sp)
        for k, slot in enumerate(self._slots):
            a, b = slot.col_start, slot.col_end
            tr_SpinvSk = float(np.einsum(
                "ij,ji->", S_pinv[a:b, a:b], slot.S
            ))
            out[k] = sp[k] * tr_SpinvSk
        return out

    def _dlog_det_H_drho(self, fit: "_FitState", rho: np.ndarray,
                         db_drho: np.ndarray | None = None) -> np.ndarray:
        """∂log|H|/∂ρ_k where H = X'WX + Sλ at converged β̂. Length-n_sp.

        Determinant identity: ∂log|H|/∂ρ_k = tr(H⁻¹ ∂H/∂ρ_k).

            ∂H/∂ρ_k = X' diag(∂w/∂ρ_k) X + λ_k S_k

        Trace decomposition with d_i := (X H⁻¹ X')_{ii} (length-n):

            tr(H⁻¹ X' diag(∂w/∂ρ_k) X) = Σ_i d_i · (∂w_i/∂ρ_k)
            ∂w_i/∂ρ_k = (∂w/∂η)_i · (X · ∂β̂/∂ρ_k)_i

        For Gaussian-identity, ∂w/∂η ≡ 0, and the first term vanishes —
        recovering the existing `λ_k · tr(H⁻¹ S_k)` form in `_reml_grad`.
        """
        n_sp = len(self._slots)
        if n_sp == 0:
            return np.zeros(0)
        X = self._X_full
        sp = np.exp(rho)

        # diag(X H⁻¹ X') in O(n·p²): solve H · M = X' for each obs row,
        # then row-wise einsum. We compute H⁻¹ X' as a (p, n) matrix once.
        Hinv_Xt = cho_solve((fit.A_chol, fit.A_chol_lower), X.T)
        d = np.einsum("ij,ji->i", X, Hinv_Xt)   # diag(X H⁻¹ X'), shape (n,)

        # For Gaussian-identity (PIRLS not used) fit.w is None — the
        # caller never reaches this path. PIRLS-converged fits always
        # have w populated.
        dw_deta = self._dw_deta(fit)

        if db_drho is None:
            db_drho = self._dbeta_drho(fit, rho)

        # ∂η/∂ρ has shape (n, n_sp); ∂w/∂ρ = dw_deta[:, None] · ∂η/∂ρ.
        deta_drho = X @ db_drho                  # (n, n_sp)
        dw_drho = dw_deta[:, None] * deta_drho   # (n, n_sp)

        out = np.empty(n_sp)
        for k, slot in enumerate(self._slots):
            a, b = slot.col_start, slot.col_end
            # tr(H⁻¹ S_k): same block trick as `_reml_grad`.
            Hinv_block = cho_solve(
                (fit.A_chol, fit.A_chol_lower), np.eye(self.p)
            )[a:b, a:b]
            tr_Hinv_Sk = float(np.einsum("ij,ji->", Hinv_block, slot.S))
            out[k] = float(np.sum(d * dw_drho[:, k])) + sp[k] * tr_Hinv_Sk
        return out

    def _dDp_drho(self, fit: "_FitState",
                  rho: np.ndarray) -> np.ndarray:
        """∂Dp/∂ρ_k at PIRLS-converged β̂. Length-n_sp.

        Dp = -2·ℓ(β̂) + β̂'Sλ β̂ (deviance + penalty). Differentiating in ρ_k
        and applying β̂(ρ) chain rule:

            ∂Dp/∂ρ_k = (∂(-2ℓ)/∂β |_β̂) · ∂β̂/∂ρ_k
                     + 2·β̂' Sλ · ∂β̂/∂ρ_k
                     + λ_k · β̂' S_k β̂

        At convergence the penalized score is zero: -∂ℓ/∂β |_β̂ + Sλ β̂ = 0,
        i.e. ∂ℓ/∂β |_β̂ = Sλ β̂. Substituting cancels the first two terms:

            ∂Dp/∂ρ_k = λ_k · β̂' S_k β̂

        Same closed form as the Gaussian special case (`g_k` in `_reml_grad`).
        Holds for any family with PIRLS-converged β̂.
        """
        n_sp = len(self._slots)
        if n_sp == 0:
            return np.zeros(0)
        sp = np.exp(rho)
        out = np.empty(n_sp)
        for k, slot in enumerate(self._slots):
            a, b = slot.col_start, slot.col_end
            beta_k = fit.beta[a:b]
            out[k] = sp[k] * float(beta_k @ slot.S @ beta_k)
        return out

    def _gcv(self, rho: np.ndarray, fit: "_FitState | None" = None) -> float:
        """GCV (scale-unknown) or UBRE/Mallows-Cp (scale-known). Wood 2017 §4.4.

            scale_unknown:  V_g = n · D / (n − τ)²
            scale_known:    V_u = D/n + 2·τ/n − 1     (φ ≡ 1)

        with D = Σ family.dev_resid(y, μ̂, wt) the deviance and
        τ = tr((X'W_F X + Sλ)⁻¹ X'W_F X) the Fisher-W effective degrees of
        freedom at PIRLS-converged β̂. mgcv's GCV/UBRE plugs in Fisher
        W_F = μ_η²/V here, not the Newton W_N = α·μ_η²/V used in the REML
        log|H+S| term (verified empirically against trees+Gamma+log:
        τ_F = 4.4222538 = mgcv m$edf, V_g(τ_F) = 0.008082356 = mgcv GCV).
        For canonical links Fisher ≡ Newton; for Gaussian-identity W = I
        and this collapses to D=rss, τ=tr(A⁻¹ X'X), bit-identical to the
        pre-Stage-2 closed form.
        """
        if fit is None:
            fit = self._fit_given_rho(rho)
        fit_F = self._fisher_view(fit)
        n = self.n
        if fit_F.w is None or np.allclose(fit_F.w, 1.0):
            XtWX = self._XtX
        else:
            Xw = self._X_full * np.sqrt(fit_F.w)[:, None]
            XtWX = Xw.T @ Xw
        A_inv = cho_solve((fit_F.A_chol, fit_F.A_chol_lower), np.eye(self.p))
        edf_total = float(np.trace(A_inv @ XtWX))
        if self.family.scale_known:
            return fit.dev / n + 2.0 * edf_total / n - 1.0
        denom = n - edf_total
        if denom <= 0:
            return 1e15
        return n * fit.dev / (denom * denom)

    def _gcv_grad(self, rho: np.ndarray,
                  fit: "_FitState | None" = None) -> np.ndarray:
        """Analytical gradient of `_gcv`. Length n_sp. Wood 2008 §4.

            scale_unknown:  ∂V_g/∂ρ_k = n·∂D/∂ρ_k / (n−τ)²
                                       + 2·n·D·∂τ/∂ρ_k / (n−τ)³
            scale_known:    ∂V_u/∂ρ_k = ∂D/∂ρ_k / n + 2·∂τ/∂ρ_k / n

        Pieces (PIRLS-converged β̂):

          ∂D/∂ρ_k = −2·(Sλ β̂)' · ∂β̂/∂ρ_k       (Newton IFT for ∂β̂/∂ρ_k)

          τ = tr(A_F⁻¹ X'W_F X) with A_F = X'W_F X + Sλ, W_F = μ_η²/V
              (Fisher; mgcv gam.fit3.r:644).
          ∂τ/∂ρ_k = (d − s)' · hv_F,k − λ_k · tr(A_F⁻¹ S_k F_F)

        with d = diag(X A_F⁻¹ X'), s = (X A_F⁻¹ X')² · W_F (row-sum),
        F_F = A_F⁻¹ X'W_F X, hv_F,k = ∂W_F/∂ρ_k = dW_F/dη · (X·∂β̂/∂ρ_k).

        β̂'s ρ-dependence comes from the Newton IFT (since the penalized
        score's β-Jacobian at β̂ is the Newton H = X'W_N X + Sλ, regardless
        of which W enters the score function being optimized), so
        `_dbeta_drho(fit, rho)` keeps the original Newton ``fit.A_chol``.
        For Gaussian-identity hv ≡ 0 ⇒ standard `−λ_k·tr(A⁻¹ S_k F)` form.
        For Gamma+log dW_F/dη ≡ 0 ⇒ same simpler form.
        """
        if fit is None:
            fit = self._fit_given_rho(rho)
        fit_F = self._fisher_view(fit)
        n_sp = len(self._slots)
        if n_sp == 0:
            return np.zeros(0)

        sp = np.exp(rho)
        n, p = self.n, self.p
        X = self._X_full
        family = self.family

        # Fisher X'W_F X (= self._XtX when W_F ≡ 1, e.g. Gaussian-identity).
        w_F = fit_F.w if fit_F.w is not None else np.ones(n)
        if np.allclose(w_F, 1.0):
            XtWX_F = self._XtX
        else:
            Xw = X * np.sqrt(w_F)[:, None]
            XtWX_F = Xw.T @ Xw

        A_F_inv = cho_solve((fit_F.A_chol, fit_F.A_chol_lower), np.eye(p))
        F_F = A_F_inv @ XtWX_F
        edf_total = float(np.trace(F_F))

        # ∂D/∂ρ_k via chain through β̂ (Newton IFT — uses Newton fit.A_chol).
        db_drho = self._dbeta_drho(fit, rho)              # (p, n_sp)
        Sλ_beta = fit.S_full @ fit.beta                    # (p,)
        dD_drho = -2.0 * (Sλ_beta @ db_drho)               # (n_sp,)

        # ∂τ/∂ρ_k. M_F = A_F⁻¹·X', P_F = X·M_F.
        M_F = cho_solve((fit_F.A_chol, fit_F.A_chol_lower), X.T)
        P_F = X @ M_F
        d_diag = np.einsum("ij,ji->i", X, M_F)             # diag(P_F)
        # Penalty piece: −λ_k · tr(A_F⁻¹·S_k·F_F).
        pen_piece = np.empty(n_sp)
        for k, slot in enumerate(self._slots):
            a, b = slot.col_start, slot.col_end
            AinvSk = A_F_inv[:, a:b] @ slot.S
            pen_piece[k] = -sp[k] * float(
                np.einsum("ij,ji->", AinvSk, F_F[a:b, :])
            )

        # W_F-deriv piece: (d − s)' hv_F,k. dW_F/dη = 0 for Gaussian-identity
        # and for Gamma+log (W_F ≡ 1) ⇒ skipped via the all-close check.
        if family.name == "gaussian" and family.link.name == "identity":
            w_piece = np.zeros(n_sp)
        else:
            dw_deta = self._dw_deta(fit_F)                 # (n,) — Fisher form
            v = X @ db_drho                                # (n, n_sp)
            hv = dw_deta[:, None] * v                      # (n, n_sp)
            Rsq = P_F * P_F
            s = Rsq @ w_F
            w_piece = (d_diag - s) @ hv                    # (n_sp,)

        dtau_drho = w_piece + pen_piece

        if family.scale_known:
            return dD_drho / n + 2.0 * dtau_drho / n
        denom = n - edf_total
        if denom <= 0:
            return np.zeros(n_sp)
        return (
            n * dD_drho / (denom * denom)
            + 2.0 * n * fit.dev * dtau_drho / (denom**3)
        )

    def _gcv_hessian(self, rho: np.ndarray,
                     fit: "_FitState | None" = None) -> np.ndarray:
        """Analytical Hessian of `_gcv`. Shape (n_sp, n_sp). Wood 2008 §4.

        scale_unknown:
            V_g = n D / (n−τ)²
            ∂²V_g/∂ρ_l∂ρ_k = n·∂²D/(n−τ)²
                            + 2n·(∂D⊗∂τ + ∂τ⊗∂D)/(n−τ)³
                            + 2n·D·∂²τ/(n−τ)³
                            + 6n·D·(∂τ⊗∂τ)/(n−τ)⁴
        scale_known:
            V_u = D/n + 2τ/n − 1
            ∂²V_u/∂ρ_l∂ρ_k = ∂²D/n + 2·∂²τ/n

        Pieces (PIRLS-converged β̂):

          ∂²D/∂ρ_l∂ρ_k = 2 λ_l λ_k β̂' S_l A_N⁻¹ S_k β̂
                        − 2 (∂β̂/∂ρ_l)' Sλ (∂β̂/∂ρ_k)
                        − 2 (Sλβ̂)' ∂²β̂/(∂ρ_l ∂ρ_k)

            All β̂-derivatives use Newton A_N = X'W_N X + Sλ (the IFT
            Hessian); ``_d2beta_drho_drho`` internally calls ``_dw_deta``
            on the Newton fit — kept that way.

          ∂²τ/∂ρ_l∂ρ_k uses Fisher A_F, F_F = A_F⁻¹ X'W_F X, and Fisher
          W-derivatives dW_F/dη, d²W_F/dη² (mgcv gam.fit3.r:644). The
          d²w_lk = d²W_F/dη² · v_l v_k + dW_F/dη · X·∂²β̂/(∂ρ_l ∂ρ_k)
          term mixes Fisher (dW_F/dη) with Newton (∂²β̂/∂ρ²) — both are
          correct for their respective roles.

        Gaussian-identity: hv ≡ 0 and d²w ≡ 0, so Q_k ≡ 0 and the W-deriv
        terms collapse to ``2 λ_l λ_k tr[A⁻¹ S_l A⁻¹ S_k F] − δ_lk·λ_k·
        tr[A⁻¹ S_k F]``. For Gamma+log Fisher W_F ≡ 1 ⇒ same closed form
        with A_F = X'X + Sλ.
        """
        if fit is None:
            fit = self._fit_given_rho(rho)
        fit_F = self._fisher_view(fit)
        n_sp = len(self._slots)
        if n_sp == 0:
            return np.zeros((0, 0))

        sp = np.exp(rho)
        n, p = self.n, self.p
        X = self._X_full
        family = self.family

        # Fisher X'W_F X for τ.
        w_F = fit_F.w if fit_F.w is not None else np.ones(n)
        if np.allclose(w_F, 1.0):
            XtWX_F = self._XtX
        else:
            Xw = X * np.sqrt(w_F)[:, None]
            XtWX_F = Xw.T @ Xw

        # Fisher precomputations for τ.
        A_F_inv = cho_solve((fit_F.A_chol, fit_F.A_chol_lower), np.eye(p))
        M_F = cho_solve((fit_F.A_chol, fit_F.A_chol_lower), X.T)   # (p, n)
        P_F = X @ M_F                                               # (n, n)
        d_diag = np.einsum("ij,ji->i", X, M_F)                      # diag(P_F)
        Rsq = P_F * P_F
        s = Rsq @ w_F
        F_F = A_F_inv @ XtWX_F                                      # (p, p)
        edf_total = float(np.trace(F_F))

        # First-derivative ingredients. ∂β̂/∂ρ uses Newton A_N (fit.A_chol).
        db_drho = self._dbeta_drho(fit, rho)                  # (p, n_sp)
        Sλβ = fit.S_full @ fit.beta                            # (p,)
        dD_drho = -2.0 * (Sλβ @ db_drho)                       # (n_sp,)

        # W-derivative arrays. Two distinct chains:
        #   Fisher (W_F): for τ-related ingredients (hv_F, d²W_F/dη²).
        #   Newton (W_N): for ∂²β̂/∂ρ² IFT inside `_d2beta_drho_drho`.
        # For canonical or Fisher-fallback fits these coincide.
        dw_deta_F = self._dw_deta(fit_F)                       # (n,) Fisher
        d2w_deta2_F = self._d2w_deta2(fit_F)                   # (n,) Fisher
        dw_deta_N = self._dw_deta(fit)                         # (n,) Newton
        v = X @ db_drho                                        # (n, n_sp)
        hv = dw_deta_F[:, None] * v                            # (n, n_sp)

        # Per-slot block precomputations.
        AinvS_block: list[np.ndarray] = []
        Sbeta_full = np.zeros((n_sp, p))
        AinvSbeta = np.empty((n_sp, p))
        tr_AinvSk_F = np.zeros(n_sp)
        for k, slot in enumerate(self._slots):
            a, b = slot.col_start, slot.col_end
            AinvS_block.append(A_F_inv[:, a:b] @ slot.S)
            beta_k = fit.beta[a:b]
            Sb = slot.S @ beta_k
            Sbeta_full[k, a:b] = Sb
            # Note: the bSAS_b piece of ∂²D uses Newton A (the IFT Hessian),
            # since it expresses (∂β̂/∂ρ_l)' Sλ (∂β̂/∂ρ_k) and ∂β̂/∂ρ uses A_N⁻¹.
            AinvSbeta[k] = cho_solve(
                (fit.A_chol, fit.A_chol_lower), Sbeta_full[k]
            )
            tr_AinvSk_F[k] = float(np.einsum(
                "ij,ji->", AinvS_block[k], F_F[a:b, :]
            ))

        pen_piece = -sp * tr_AinvSk_F                          # (n_sp,)
        w_piece = (d_diag - s) @ hv                            # (n_sp,)
        dtau_drho = w_piece + pen_piece

        # ---- ∂²D/∂ρ_l∂ρ_k — uses Newton A throughout β̂-derivatives. -----
        # bSAS_b[l, k] = β̂' S_l A_N⁻¹ S_k β̂ (already symmetric).
        bSAS_b = Sbeta_full @ AinvSbeta.T                      # (n_sp, n_sp)
        Sλ_db = fit.S_full @ db_drho                            # (p, n_sp)
        db_Sλ_db = db_drho.T @ Sλ_db                            # (n_sp, n_sp)
        d2b = self._d2beta_drho_drho(
            fit, rho, db_drho=db_drho, dw_deta=dw_deta_N
        )                                                      # (p, n_sp, n_sp)
        Sλβ_d2b = np.einsum("p,pij->ij", Sλβ, d2b)              # (n_sp, n_sp)

        sp_outer = np.outer(sp, sp)
        d2D = (
            2.0 * sp_outer * bSAS_b
            - 2.0 * db_Sλ_db
            - 2.0 * Sλβ_d2b
        )
        d2D = 0.5 * (d2D + d2D.T)

        # ---- ∂²τ/∂ρ_l∂ρ_k — Fisher A_F, F_F, dW_F. ----------------------
        # Y_k = A_F⁻¹ P_F,k = M_F·diag(hv_k)·X + λ_k · A_F⁻¹ S_k_full
        # U_k = A_F⁻¹ Q_F,k = M_F·diag(hv_k)·X
        Y_full = np.empty((n_sp, p, p))
        U_full = np.empty((n_sp, p, p))
        for k in range(n_sp):
            a, b = self._slots[k].col_start, self._slots[k].col_end
            MhX_k = M_F @ (hv[:, k:k+1] * X)
            U_full[k] = MhX_k
            Y_k = MhX_k.copy()
            Y_k[:, a:b] += sp[k] * AinvS_block[k]
            Y_full[k] = Y_k

        d2tau = np.zeros((n_sp, n_sp))
        for ll in range(n_sp):
            for k in range(ll, n_sp):
                YlYk = Y_full[ll] @ Y_full[k]
                T_a = float(np.einsum("ij,ji->", YlYk, F_F))
                if ll == k:
                    T_b = T_a
                else:
                    YkYl = Y_full[k] @ Y_full[ll]
                    T_b = float(np.einsum("ij,ji->", YkYl, F_F))
                T1_T2 = T_a + T_b

                T4 = float(np.einsum("ij,ji->", Y_full[k], U_full[ll]))
                T5 = float(np.einsum("ij,ji->", Y_full[ll], U_full[k]))

                # d²W_F_lk = d²W_F/dη² · v_l v_k + dW_F/dη · X·∂²β̂/(∂ρ_l ∂ρ_k).
                # Fisher W-derivatives; Newton ∂²β̂/∂ρ² (Newton IFT).
                Xd2b_lk = X @ d2b[:, ll, k]
                d2w_lk = (
                    d2w_deta2_F * v[:, ll] * v[:, k]
                    + dw_deta_F * Xd2b_lk
                )
                T6_minus_T3B = float((d_diag - s) @ d2w_lk)
                delta_S = -sp[k] * tr_AinvSk_F[k] if ll == k else 0.0

                val = T1_T2 - T4 - T5 + T6_minus_T3B + delta_S
                d2tau[ll, k] = val
                if ll != k:
                    d2tau[k, ll] = val

        d2tau = 0.5 * (d2tau + d2tau.T)

        # ---- Compose criterion Hessian --------------------------------
        if family.scale_known:
            return d2D / n + 2.0 * d2tau / n

        denom = n - edf_total
        if denom <= 0:
            return np.full((n_sp, n_sp), 1e15)

        Dn = float(fit.dev)
        dD_dτ = np.outer(dD_drho, dtau_drho)
        dτ_dτ = np.outer(dtau_drho, dtau_drho)
        H = (
            n * d2D / (denom * denom)
            + 2.0 * n * (dD_dτ + dD_dτ.T) / (denom**3)
            + 2.0 * n * Dn * d2tau / (denom**3)
            + 6.0 * n * Dn * dτ_dτ / (denom**4)
        )
        return H

    def _db_drho(self, rho: np.ndarray, beta: np.ndarray,
                 A_chol, A_chol_lower) -> np.ndarray:
        """Analytical ∂β/∂ρ_k = -exp(ρ_k)·A⁻¹ S_k β, returned as (p, n_sp).

        Differentiate A(ρ) β = X'y wrt ρ_k: ∂A/∂ρ_k = exp(ρ_k) S_k since
        A = X'X + Σ_k exp(ρ_k) S_k. The k-th slot's S is k×k embedded at
        its block's column range, so the RHS is non-zero only there.
        """
        p = self.p
        n_sp = len(self._slots)
        db = np.zeros((p, n_sp))
        for k, slot in enumerate(self._slots):
            a, b = slot.col_start, slot.col_end
            sp_k = float(np.exp(rho[k]))
            v = np.zeros(p)
            v[a:b] = -sp_k * (slot.S @ beta[a:b])
            db[:, k] = cho_solve((A_chol, A_chol_lower), v)
        return db

    def _compute_edf12(self, rho: np.ndarray, fit: "_FitState",
                       sigma_squared: float, A_inv: np.ndarray,
                       A_inv_XtWX: np.ndarray, edf: np.ndarray,
                       H_aug: np.ndarray | None):
        """mgcv's edf1 (frequentist tr(2F−F²) bound) and edf2 (sp-uncertainty
        corrected). Wood 2017 §6.11.3. Returns ``(edf2_per_coef, edf1_per_coef)``.

        edf2 = diag((σ² A⁻¹ + Vc1 + Vc2) · X'WX) / σ², where

          - Vc1 = (∂β̂/∂ρ) · Vr · (∂β̂/∂ρ)ᵀ     (β̂'s ρ-dependence)
          - Vc2 = σ² Σ_{i,j} Vr[i,j] M_i M_j^T    (Cholesky-derivative bit)

        with M_k = ∂L^{-T}/∂ρ_k. Vr is the marginal covariance of ρ̂,
        taken as the top-left block of pinv(H_aug) (this equals the
        Schur complement of the augmented REML Hessian — same thing as
        inverting the profiled-σ² Hessian, mathematically). Falls back
        to the profiled Hessian when H_aug is unavailable (GCV / no
        smooths). For Gaussian + identity, dw/dρ vanishes so the Vc2
        formula above is the full mgcv expression — matches
        ``gam.fit3.post.proc``'s Vp + Vc1 + Vc2 decomposition.
        """
        F = A_inv_XtWX
        edf1 = 2.0 * np.diag(F) - np.einsum("ij,ji->i", F, F)

        n_sp = len(self._slots)
        if n_sp == 0:
            return edf.copy(), edf1

        db = self._db_drho(rho, fit.beta, fit.A_chol, fit.A_chol_lower)
        Vr = self._compute_Vr(rho, H_aug)
        # mgcv splits Vr by component: Vc1 uses pinv(H_aug) on positive
        # eigenspace; Vc2 uses (H_aug + 0.1·I)^{-1} — a weak prior on log
        # smoothing parameters (gam.fit3.post.proc line 1011). Without
        # this prior on Vc2, edf2 drifts ~1e-3 above mgcv.
        Vr_reg = self._compute_Vr(rho, H_aug, prior_var=0.1)

        Vc1 = db @ Vr @ db.T
        Vc2 = self._compute_Vc2(rho, fit, Vr_reg, sigma_squared)

        # diag((σ²A_F⁻¹ + Vc1 + Vc2)·X'W_F X)/σ² = edf + diag((Vc1 + Vc2)·
        # X'W_F X)/σ². Fisher W_F to stay consistent with the edf metric
        # used at gam.fit3.r:644 (and with the Fisher A_inv_XtWX our caller
        # passes in). For Gaussian-identity W_F ≡ I and X'W_F X = X'X.
        if fit.is_fisher_fallback:
            W_F_view = fit.w
        else:
            family = self.family
            mu_eta = family.link.mu_eta(fit.eta)
            V = family.variance(fit.mu)
            W_F_view = mu_eta ** 2 / V
        if W_F_view is None or np.allclose(W_F_view, 1.0):
            XtWX = self._XtX
        else:
            Xw = self._X_full * np.sqrt(W_F_view)[:, None]
            XtWX = Xw.T @ Xw
        if sigma_squared > 0 and np.isfinite(sigma_squared):
            Vc = Vc1 + Vc2
            edf2 = edf + np.einsum("ij,ij->i", Vc, XtWX) / sigma_squared
        else:
            edf2 = edf.copy()

        # Total-sum cap only. mgcv's gam.fit3.post.proc deliberately does
        # not cap element-wise — individual edf2[i] can exceed edf1[i] as
        # long as the sum stays ≤ sum(edf1). Element-wise capping was a
        # bug in an earlier version here that pushed sum(edf2) below
        # sum(edf), the wrong direction for an sp-uncertainty correction.
        if edf2.sum() > edf1.sum():
            edf2 = edf1.copy()
        return edf2, edf1

    def _compute_Vr(self, rho: np.ndarray,
                    H_aug: np.ndarray | None,
                    prior_var: float | None = None) -> np.ndarray:
        """Marginal covariance of ρ̂ — top-left ρρ block of inverse of H_aug.

        ``prior_var=None`` (default): pseudo-inverse with positive-eigenvalue
        projection — used for Vc1 and vcomp CIs. When H_aug is given, this
        is the Schur complement of the augmented Hessian; without it, invert
        the ρ-only profiled Hessian directly. Project onto the positive
        eigenspace before inverting (near sp bounds the surface is locally
        flat and tiny eigenvalues would blow up).

        ``prior_var > 0``: regularized inverse where eigenvalues are
        replaced by ``max(λ, 0) + prior_var`` before inverting — used for
        Vc2 to mirror mgcv's ``1/(d+1/10)`` prior on log smoothing
        parameters (gam.fit3.post.proc line 1011, "exp(4·var^.5) gives
        approx multiplicative range"). Without this, edf2 on bs='re' /
        nested-RE models drifts ~1e-3 above mgcv.
        """
        n_sp = len(self._slots)
        if H_aug is not None:
            w, V = np.linalg.eigh(H_aug)
            if prior_var is not None:
                d_reg = np.where(w > 0, w, 0.0) + float(prior_var)
                H_inv = (V / d_reg) @ V.T
                return H_inv[:n_sp, :n_sp]
            w_max = float(w.max()) if w.size > 0 else 0.0
            keep = (w > w_max * 1e-7) if w_max > 0 else np.zeros_like(w, dtype=bool)
            if not keep.any():
                return np.zeros((n_sp, n_sp))
            Vk = V[:, keep]
            H_inv = (Vk / w[keep]) @ Vk.T
            return H_inv[:n_sp, :n_sp]
        # GCV / no-H_aug fallback: ρρ block of the (ρ, log φ) joint Hessian
        # at log φ = 0. For Gaussian-identity REML this used to call the
        # Gaussian-profiled `_reml_hessian`; the joint Hessian's ρρ block
        # equals 2× that profiled Hessian up to the rank-1 Schur term, which
        # is fine for the GCV path (mgcv defines edf2 differently for GCV
        # anyway — this is a best-effort sp-uncertainty correction).
        H_full = 0.5 * self._reml_hessian(rho, 0.0, include_log_phi=False)
        H = 0.5 * (H_full + H_full.T)
        w, V = np.linalg.eigh(H)
        if prior_var is not None:
            d_reg = np.where(w > 0, w, 0.0) + float(prior_var)
            return (V / d_reg) @ V.T
        w_max = float(w.max()) if w.size > 0 else 0.0
        keep = (w > w_max * 1e-7) if w_max > 0 else np.zeros_like(w, dtype=bool)
        if not keep.any():
            return np.zeros((n_sp, n_sp))
        Vk = V[:, keep]
        return (Vk / w[keep]) @ Vk.T

    def _compute_Vc2(self, rho: np.ndarray, fit: "_FitState",
                     Vr: np.ndarray, sigma_squared: float) -> np.ndarray:
        """Cholesky-derivative correction Vc2 = σ² Σ_{i,j} Vr[i,j] M_i M_j^T,
        where M_k = ∂L^{-T}/∂ρ_k and A = L L^T is lmpy's lower-Cholesky of
        ``X'X + Sλ``.

        Differentiating L L^T = A gives ``L^{-1} dA L^{-T}`` whose lower
        triangle (with halved diag) is ``L^{-1} dL`` — the standard
        formula ``dL = L · Φ(L^{-1} dA L^{-T})`` with ``Φ`` zeroing the
        strict upper and halving the diagonal. Then differentiating
        ``L L^{-1} = I``:

            d(L^{-1}) = -L^{-1} dL L^{-1}
            d(L^{-T}) = -L^{-T} (dL)^T L^{-T}     (transpose)

        So M_k = -L^{-T} (dL_k)^T L^{-T}. The ρ-uncertainty in the
        Bayesian draw β̃ = β̂ + σ L^{-T} z propagates as σ Σ_k ε_k M_k z
        with ε ~ N(0, Vr), z ~ N(0, I_p), giving covariance contribution
        σ² Σ_{i,j} Vr[i,j] M_i M_j^T.

        Mirrors mgcv's gam.fit3.post.proc — closes the residual ~0.1 AIC
        gap on bs='re' models that's left after Vc1 alone.
        """
        p = self.p
        n_sp = len(self._slots)
        if n_sp == 0 or sigma_squared <= 0 or not np.isfinite(sigma_squared):
            return np.zeros((p, p))
        # scipy's cho_factor leaves the unused upper triangle untouched
        # (random memory), so explicitly mask before using as a triangular
        # operand — solve_triangular respects `lower=True` but np.tril for
        # the explicit L matmul below would otherwise pull garbage in.
        L = np.tril(fit.A_chol)

        M = np.empty((n_sp, p, p))
        for k, slot in enumerate(self._slots):
            a, b = slot.col_start, slot.col_end
            sp_k = float(np.exp(rho[k]))
            # dA_k = sp_k · S_k embedded at the slot's column range.
            dA = np.zeros((p, p))
            dA[a:b, a:b] = sp_k * slot.S
            # X = L^{-1} dA L^{-T} — two triangular solves.
            Y = solve_triangular(L, dA, lower=True)
            X = solve_triangular(L, Y.T, lower=True).T
            # Φ(X): strict_lower(X) + 0.5·diag(X). Symmetric in floating
            # point because X is symmetric (since dA is symmetric), so we
            # build it from the lower triangle directly.
            Phi = np.tril(X, -1)
            np.fill_diagonal(Phi, 0.5 * np.diag(X))
            dL = L @ Phi
            # M_k = -L^{-T} (dL)^T L^{-T}. Compute as two triangular
            # solves: G = (dL)^T L^{-T} = (L^{-1} dL)^T, then solve
            # L^T M_k = -G.
            G = solve_triangular(L, dL, lower=True).T
            M[k] = solve_triangular(L.T, -G, lower=False)

        # Vc2[a,b] = Σ_{i,j} Vr[i,j] M_i[a,c] M_j[b,c] — contract over
        # the trailing axis of both M operands.
        Vc2 = np.einsum("ij,iac,jbc->ab", Vr, M, M)
        return sigma_squared * Vc2

    def _compute_vcomp(self) -> pl.DataFrame:
        """Build the variance-component table mgcv calls ``gam.vcomp``.

        For each smoothing-param slot k, σ_k = σ/√sp_k is the implied
        random-effect std.dev (literal for ``bs='re'``; a parametrization
        for other smooths). CIs come from the delta method on
        log(σ_k) = ½(log σ² − ρ_k) using the joint REML Hessian wrt
        (ρ, log σ²) — only meaningful under REML, so for GCV we return
        point estimates with NaN bounds. Reuses the augmented Hessian
        cached on ``self._H_aug`` (set in ``__init__``).
        """
        n_sp = len(self._slots)
        scale_sd = float(self.sigma) if np.isfinite(self.sigma) else float("nan")

        if n_sp == 0:
            return pl.DataFrame({
                "name": ["scale"],
                "std_dev": [scale_sd],
                "lower": [float("nan")],
                "upper": [float("nan")],
            })

        names = [slot.block.label for slot in self._slots] + ["scale"]
        sd2 = np.concatenate([
            np.array([self.sigma_squared / max(s, 1e-300) for s in self.sp]),
            [self.sigma_squared],
        ])
        log_sd = 0.5 * np.log(np.clip(sd2, 1e-300, None))
        sd = np.exp(log_sd)

        # GCV / point-estimate-only path: no Hessian-derived CIs.
        H = self._H_aug
        if H is None or self.method != "REML" or not np.isfinite(self.sigma_squared):
            nan_col = [float("nan")] * len(sd)
            return pl.DataFrame({
                "name": names, "std_dev": sd.tolist(),
                "lower": nan_col, "upper": nan_col,
            })

        # Pseudo-invert on the positive eigenspace, same threshold as edf2.
        w, V = np.linalg.eigh(H)
        w_max = float(w.max()) if w.size > 0 else 0.0
        keep = (w > w_max * 1e-7) if w_max > 0 else np.zeros_like(w, dtype=bool)
        Hinv = np.zeros_like(H)
        if keep.any():
            Vk = V[:, keep]
            Hinv = (Vk / w[keep]) @ Vk.T

        # J: log(σ_k) = -0.5·ρ_k + 0.5·log σ² for k < last; log(σ_scale) =
        # 0.5·log σ². Last column is the log σ² coefficient throughout.
        m = n_sp + 1
        J = np.zeros((m, m))
        J[np.arange(n_sp), np.arange(n_sp)] = -0.5
        J[:, -1] = 0.5

        Vc = J @ Hinv @ J.T
        se = np.sqrt(np.maximum(np.diag(Vc), 0.0))
        z = float(norm.ppf(0.975))
        lower = np.exp(log_sd - z * se)
        upper = np.exp(log_sd + z * se)
        return pl.DataFrame({
            "name": names,
            "std_dev": sd.tolist(),
            "lower": lower.tolist(),
            "upper": upper.tolist(),
        })

    # -----------------------------------------------------------------------
    # Public post-fit API
    # -----------------------------------------------------------------------

    def _deviance_residuals(self, y, mu, wt) -> np.ndarray:
        """``sign(y - μ)·√(per-obs deviance)`` — mgcv's default residual."""
        d_i = self.family.dev_resids(y, mu, wt)
        d_i = np.maximum(d_i, 0.0)            # FP cleanup near zero
        return np.sign(y - mu) * np.sqrt(d_i)

    def residuals_of(self, type: str = "deviance") -> np.ndarray:
        """GLM residuals of the requested ``type``.

        Mirrors ``residuals.glm`` / ``residuals.gam`` in R.

        Parameters
        ----------
        type : {"deviance", "pearson", "working", "response"}
            - ``"deviance"`` (default): ``sign(y-μ)·√(per-obs deviance)``.
            - ``"pearson"``: ``(y-μ)·√(wt / V(μ))``.
            - ``"working"``: ``(y-μ) · g'(μ)`` (η-scale residual).
            - ``"response"``: ``y - μ``.
        """
        if type not in ("deviance", "pearson", "working", "response"):
            raise ValueError(
                f"type must be one of 'deviance', 'pearson', 'working', "
                f"'response'; got {type!r}"
            )
        y = self._y_arr
        mu = self.fitted_values
        wt = self._wt
        if type == "response":
            return y - mu
        if type == "deviance":
            return self._deviance_residuals(y, mu, wt)
        V = self.family.variance(mu)
        if type == "pearson":
            return (y - mu) * np.sqrt(wt / np.maximum(V, 0.0))
        # working: (y-μ) · g'(μ) = (y-μ) / (dμ/dη)
        eta = self.linear_predictors
        dmu_deta = self.family.link.mu_eta(eta)
        return (y - mu) / dmu_deta

    def predict(self, newdata: pl.DataFrame | None = None) -> np.ndarray:
        """Return in-sample fitted values ``ŷ = Xβ̂``.

        Out-of-sample prediction (``newdata != None``) requires a
        mgcv-style ``PredictMat`` that evaluates each smooth's stored
        basis (knots, Lanczos eigenvectors, sum-to-zero constraint) at
        the new covariate values. That machinery isn't part of
        ``lmpy.formula`` yet, so this v1 raises for new data rather than
        return fuzzy results from re-materializing the basis over
        ``[train, new]`` — for tp in particular, the basis is genuinely
        data-dependent and the re-materialized X differs from the fit's.
        """
        if newdata is None:
            return self.fitted
        if self._expanded.smooths:
            raise NotImplementedError(
                "predict(newdata=...) for models with smooth terms is not "
                "yet supported — needs a PredictMat implementation in "
                "lmpy.formula. Use m.fitted for in-sample predictions."
            )
        from .formula import materialize  # local to avoid cycle at module load

        X_new = materialize(self._expanded, newdata).to_numpy().astype(float)
        return X_new @ self._beta

    # ------------- printing ------------------------------------------------

    def __repr__(self) -> str:
        out = [
            f"Family: {self.family.name}",
            f"Link function: {self.family.link.name}",
            "",
            f"Formula: {self.formula}",
            "",
            "Coefficients:",
            format_df(self.bhat),
        ]
        return "\n".join(out)

    def __str__(self) -> str:
        return self.__repr__()

    def summary(self, digits: int = 4) -> None:
        """mgcv-style summary: parametric table + smooth-edf table + fit stats."""
        out = [
            "",
            f"Family: {self.family.name}",
            f"Link function: {self.family.link.name}",
            "",
            f"Formula: {self.formula}",
            "",
        ]

        # -- parametric table (lm-style) -----------------------------------
        if self.p_param > 0:
            out.append("Parametric coefficients:")
            est = self._beta[:self.p_param]
            se  = self._se[:self.p_param]
            with np.errstate(divide="ignore", invalid="ignore"):
                t_stats = est / se
            if self.df_residuals > 0 and np.isfinite(self.df_residuals):
                pv = 2 * t_dist.sf(np.abs(t_stats), self.df_residuals)
            else:
                pv = np.full_like(t_stats, np.nan)
            sig = significance_code(pv)
            tbl = pl.DataFrame({
                "": self.parametric_columns,
                "Estimate":   np.round(est, digits),
                "Std. Error": np.round(se, digits),
                "t value":    np.round(t_stats, digits),
                "Pr(>|t|)":   np.round(pv, digits),
                " ":          sig,
            })
            out.append(format_df(tbl))
            out.append("---")
            out.append(
                "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1"
            )
            out.append("")

        # -- smooth-edf table ----------------------------------------------
        if self._blocks:
            out.append("Approximate significance of smooth terms:")
            rows_label: list[str] = []
            rows_edf:   list[float] = []
            rows_refdf: list[float] = []
            rows_F:     list[float] = []
            rows_p:     list[float] = []
            for b, (a, bcol) in zip(self._blocks, self._block_col_ranges):
                beta_b = self._beta[a:bcol]
                Vp_b   = self.Vp[a:bcol, a:bcol]
                X_b    = self._X_full[:, a:bcol]
                edf_b  = float(self.edf[a:bcol].sum())
                # mgcv `testStat` (summary.r, default p.type=0): work in
                # the smooth's design-orthonormal basis, then truncate
                # the spectral pseudo-inverse at rank k1 ≈ ⌈edf1⌉.
                # Without the X-rotation, low-edf smooths (≈line) get
                # huge F because `Vp_b` has near-zero eigenvalues that
                # the tolerance-based truncate keeps; the rank-truncate
                # in the R-rotated basis is what makes Tr scale with
                # signal-to-noise rather than basis size.
                edf1_b = float(self.edf1[a:bcol].sum()) if hasattr(self, "edf1") else edf_b
                ref_df = max(edf1_b, 1e-8)
                _, R = np.linalg.qr(X_b, mode="reduced")
                V_eff = R @ Vp_b @ R.T
                V_eff = 0.5 * (V_eff + V_eff.T)
                d, U = np.linalg.eigh(V_eff)
                # eigh returns ascending; flip to descending.
                d = d[::-1]
                U = U[:, ::-1]
                k = max(0, int(np.floor(edf1_b)))
                if edf1_b > k + 0.05 or k == 0:
                    k += 1
                k1 = min(max(k, 1), d.size)
                d_top = np.maximum(d[:k1], 1e-300)
                vec = U[:, :k1].T @ (R @ beta_b)
                Tr = float(np.sum(vec**2 / d_top))
                F = Tr / ref_df
                p_val = (
                    float(f_dist.sf(F, ref_df, self.df_residuals))
                    if self.df_residuals > 0 else float("nan")
                )
                rows_label.append(b.label)
                rows_edf.append(edf_b)
                rows_refdf.append(edf1_b)
                rows_F.append(F)
                rows_p.append(p_val)
            sig = significance_code(rows_p)
            sm_tbl = pl.DataFrame({
                "":        rows_label,
                "edf":     np.round(rows_edf, digits),
                "Ref.df":  np.round(rows_refdf, digits),
                "F":       np.round(rows_F, digits),
                "p-value": np.round(rows_p, digits),
                " ":       sig,
            })
            out.append(format_df(sm_tbl))
            out.append("---")
            out.append(
                "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1"
            )
            out.append("")

        # -- fit stats ------------------------------------------------------
        out.append(
            f"R-sq.(adj) = {self.r_squared_adjusted:.3g}  "
            f"Deviance explained = {self.deviance_explained * 100:.1f}%"
        )
        if self.method == "REML":
            out.append(
                f"-REML = {self.REML_criterion / 2:.4f}  "
                f"Scale est. = {self.sigma_squared:.4g}  n = {self.n}"
            )
        else:
            out.append(
                f"GCV = {self.GCV_score:.4g}  "
                f"Scale est. = {self.sigma_squared:.4g}  n = {self.n}"
            )
        print("\n".join(out))


# --------------------------------------------------------------------------
# module-private helpers
# --------------------------------------------------------------------------


def _row_frame(values: np.ndarray, columns: list[str]) -> pl.DataFrame:
    flat = np.asarray(values).reshape(-1)
    return pl.DataFrame({c: [float(flat[i])] for i, c in enumerate(columns)})


def _apply_gam_side(blocks: list[SmoothBlock]) -> list[SmoothBlock]:
    """Apply mgcv's ``gam.side`` identifiability surgery.

    For each block ``b`` whose variable set strictly contains another
    block's (e.g. ``te(x1, x2)`` over ``s(x1) + s(x2)``), some columns of
    ``X_b`` are linearly dependent on the union of the smaller smooths'
    designs plus the intercept. mgcv finds those columns via
    ``fixDependence`` (a QR with column pivoting on the residual after
    projecting out the smaller smooths) and **deletes** them — both from
    ``X_b`` and from the rows/cols of each ``S_b[j]``. For a default
    ``te(x1, x2)`` with ``s(x1) + s(x2)`` marginals, this drops exactly 2
    columns (24 → 22), matching ``ncol(model.matrix(m))``.

    Random-effect smooths (``bs='re'``) carry ``side.constrain=FALSE`` in
    mgcv: their identity penalty already identifies the fit even with a
    rank-deficient X, so gam.side neither constrains them nor includes
    them in X1 when constraining other blocks. Replicating that here
    matters for `s(Worker, bs='re') + s(Machine, Worker, bs='re')` style
    nestings — dropping the 6 dependent interaction columns shifts the
    REML surface (different log|A|, log|S|+) and lands at a different
    optimum than mgcv. Skipping the surgery keeps the design at p=27
    (matching mgcv) at the cost of a rank-deficient X that's still PD
    once Sλ = λ·I is added in the re block.
    """
    if len(blocks) < 2:
        return blocks
    var_sets = [frozenset(b.term) for b in blocks]
    n = int(np.asarray(blocks[0].X).shape[0])
    out: list[SmoothBlock] = []
    for i, b in enumerate(blocks):
        if not _side_constrain(b):
            out.append(b)
            continue
        my_vars = var_sets[i]
        Xb = np.asarray(b.X, dtype=float)
        # X1 = intercept + every strict-subset, side-constrained block's
        # design — exactly what `gam.side` builds before `fixDependence`.
        cols_X1 = [np.ones((n, 1))]
        for j, other in enumerate(blocks):
            if i == j or not _side_constrain(other):
                continue
            if var_sets[j] and var_sets[j] < my_vars:
                cols_X1.append(np.asarray(other.X, dtype=float))
        if len(cols_X1) == 1:
            out.append(b)
            continue
        X1 = np.concatenate(cols_X1, axis=1)
        ind = _fix_dependence(X1, Xb)
        if not ind:
            out.append(b)
            continue
        keep = [c for c in range(Xb.shape[1]) if c not in ind]
        new_X = Xb[:, keep]
        new_S = []
        for Sj in b.S:
            Sj = np.asarray(Sj, dtype=float)
            new_S.append(Sj[np.ix_(keep, keep)])
        out.append(SmoothBlock(
            label=b.label, term=b.term, cls=b.cls, X=new_X, S=new_S,
        ))
    return out


def _side_constrain(b: SmoothBlock) -> bool:
    """Mirrors mgcv's ``smooth$side.constrain``. Random-effect smooths
    (``re.smooth.spec``) opt out — their identity penalty handles ID."""
    return b.cls != "re.smooth.spec"


def _fix_dependence(X1: np.ndarray, X2: np.ndarray,
                    tol: float = float(np.finfo(float).eps) ** 0.5) -> list[int]:
    """Find columns of ``X2`` that are linearly dependent on ``X1``.

    Mirrors mgcv's ``fixDependence(X1, X2, tol)`` (non-strict mode):

    1. ``Q1 R1 = X1`` (QR of X1).
    2. Project X2 onto the orthogonal complement of X1's column space
       and take the bottom block of ``Q1ᵀ X2`` (rows ``r+1..n``).
    3. QR of that residual *with column pivoting*. Trailing columns
       whose mean abs over the diagonal block falls below
       ``|R1[0,0]| · tol`` are the dependent ones — return their pivot
       indices in X2.
    """
    n, r = X1.shape
    Q1, R1 = np.linalg.qr(X1, mode="complete")
    if R1.size == 0 or n <= r:
        return []
    R11 = abs(R1[0, 0]) if R1.shape[0] > 0 else 1.0
    QtX2 = Q1.T @ X2
    residual = QtX2[r:, :]
    if residual.shape[0] == 0:
        return []
    # column-pivoted QR via scipy (numpy's qr lacks pivoting)
    from scipy.linalg import qr as scipy_qr
    Q2, R2, piv = scipy_qr(residual, mode="economic", pivoting=True)
    nrows = R2.shape[0]
    r_full = nrows
    r0 = r_full
    while r0 > 0 and float(np.mean(np.abs(R2[r0 - 1: r_full, r0 - 1: r_full]))) < R11 * tol:
        r0 -= 1
    r0 += 1
    if r0 > r_full:
        return []
    return [int(p) for p in piv[r0 - 1: r_full]]


def _sym_rank(S: np.ndarray) -> int:
    """Numerical rank of a symmetric matrix via eigendecomposition."""
    if S.size == 0:
        return 0
    w = np.linalg.eigvalsh(0.5 * (S + S.T))
    if w.size == 0:
        return 0
    tol = max(1e-12, w.max() * 1e-10) if w.max() > 0 else 1e-12
    return int(np.sum(w > tol))


class _PenaltySlot:
    """One smoothing-param slot: the k×k S matrix and its col range in the
    full design. Each SmoothBlock contributes len(S_list) slots."""
    __slots__ = ("block", "col_start", "col_end", "S")

    def __init__(self, *, block: SmoothBlock, col_start: int, col_end: int,
                 S: np.ndarray):
        self.block = block
        self.col_start = col_start
        self.col_end = col_end
        self.S = S


class _FitState:
    """Fit-at-one-ρ bundle, populated by either the Gaussian closed-form
    solver or the PIRLS loop. ``rss`` is kept as an alias for ``dev`` so
    the Gaussian-only post-fit code reads cleanly; for non-Gaussian
    families ``rss`` is the deviance (``rss == dev``)."""
    __slots__ = (
        "beta", "eta", "mu", "w", "z", "alpha",
        "dev", "pen", "rss",
        "A_chol", "A_chol_lower",
        "S_full", "log_det_A",
        "is_fisher_fallback",
    )

    def __init__(self, *, beta, dev, pen, A_chol, A_chol_lower,
                 S_full, log_det_A,
                 eta=None, mu=None, w=None, z=None, alpha=None,
                 is_fisher_fallback=False):
        self.beta = beta
        self.dev = dev
        self.rss = dev               # back-compat alias for Gaussian path
        self.pen = pen
        self.eta = eta
        self.mu = mu
        self.w = w
        self.z = z
        self.alpha = alpha
        self.A_chol = A_chol
        self.A_chol_lower = A_chol_lower
        self.S_full = S_full
        self.log_det_A = log_det_A
        # True iff PIRLS forced α=1 at convergence because Newton's
        # α formula produced a w<0. In that case dα/dμ is taken as 0
        # for derivative purposes (the analytical α'(μ) is not
        # consistent with the override).
        self.is_fisher_fallback = is_fisher_fallback
