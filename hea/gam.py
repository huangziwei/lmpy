"""Generalized additive model — mgcv-style penalized regression with
REML/GCV smoothing-parameter selection.

Built on hea.formula's ``parse → expand → materialize / materialize_smooths``
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

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.stats import f as f_dist, norm, t as t_dist

from .family import Family, Gaussian
from .formula import BasisSpec, SmoothBlock, _eval_atom, materialize_smooths
from .design import prepare_design
from .lm import _label_top_n, _lowess, _qq_plot
from .utils import (
    _dig_tst,
    format_df,
    format_pval,
    format_signif,
    format_signif_jointly,
    significance_code,
)

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
    method : {"REML", "ML", "GCV.Cp"}, default "REML"
        Smoothing-parameter selection criterion. ``"ML"`` is Laplace
        marginal likelihood — like REML but does not profile out the
        unpenalized fixed effects. Useful for ``anova(m1, m2)``-style
        likelihood-ratio comparisons across different fixed-effect
        structures, where REML scores aren't comparable.
    sp : None or array-like, optional
        If given, fix smoothing parameters at these (non-negative)
        values and skip optimization. Length must match the total number
        of penalty slots across all smooth blocks.
    select : bool, default False
        Mirror of mgcv's ``select=TRUE``. When ``True``, an extra penalty
        is added to each smooth term over its null-space directions, so
        the smoothing-parameter selection can shrink any term entirely
        to zero — i.e., perform model selection alongside smoothness
        estimation. Each smooth gains one additional smoothing parameter.

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

    Attributes (method="ML" only)
    -----------------------------
    ML_criterion : float
        Optimized Laplace-approximate ML criterion, ``-2·V_ML(ρ̂)``.
        Differs from ``REML_criterion`` by a ``Mp·log(2π·φ)`` constant
        — comparable across different fixed-effect structures.

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
        offset: np.ndarray | list | None = None,
        gamma: float = 1.0,
        select: bool = False,
    ):
        if method not in ("REML", "ML", "GCV.Cp"):
            raise ValueError(
                f"method must be 'REML', 'ML', or 'GCV.Cp', got {method!r}"
            )
        if not (np.isfinite(gamma) and gamma > 0):
            raise ValueError(f"gamma must be a positive finite number, got {gamma!r}")

        self.formula = formula
        self.method = method
        self._select = bool(select)
        # mgcv's smoothing-strength multiplier. ``gamma > 1`` produces
        # smoother fits by inflating the apparent edf cost in the GCV/UBRE
        # criterion, or by dividing the data-fit term in REML. Wood §4.6
        # recommends ``gamma=1.4`` as a reasonable default for over-fitting
        # protection. Stored on self and threaded into the criterion
        # functions (_reml, _gcv, ...) and their gradients/hessians.
        self._gamma = float(gamma)
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

        # Sum any ``offset(...)`` atoms from the formula plus the kwarg
        # offset. mgcv's gam adds these to η just like glm does:
        # η = X·β + offset for both fitting and prediction.
        off = (np.zeros(n) if offset is None
               else np.asarray(offset, dtype=float).flatten())
        if off.shape != (n,):
            raise ValueError(f"offset must have length {n}, got {off.shape}")
        for off_node in d.expanded.offsets:
            blk = _eval_atom(off_node, d.data)
            off = off + blk.values.flatten().astype(float)
        self._offset = off

        sb_lists = materialize_smooths(d.expanded, d.data) if d.expanded.smooths else []
        blocks: list[SmoothBlock] = [b for group in sb_lists for b in group]
        # mgcv: select=TRUE adds a null-space penalty per smooth inside
        # smoothCon — i.e., before gam.side. Mirror that order so the
        # subsequent column drops (gam.side) restrict Sf to the kept-cols
        # subspace exactly the way mgcv does.
        if self._select:
            blocks = _add_null_space_penalties(blocks)
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
        # Set by `_outer_newton` when the optimizer runs. None for the
        # no-smooth and fixed-`sp` paths — `gam.check()` skips the
        # convergence block in those cases.
        self._outer_info: dict | None = None
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
            # For unknown-scale families fit by (RE)ML, set log φ̂ to the
            # profile-out value — the same value the (ρ, log φ) outer
            # optimizer would converge to at this sp. Keeps `sigma_squared`
            # and the score consistent with the free-optimization path
            # bit-for-bit when sp= is fed back in.
            #   REML: φ̂ = Dp/(n−Mp)  (Mp·log φ in score; profiles out fixed effects)
            #   ML:   φ̂ = Dp/n      (no Mp·log φ; treats β as deterministic)
            if (not self.family.scale_known) and method in ("REML", "ML"):
                Dp = float(fit.dev + fit.pen)
                denom = max(float(n - self._Mp), 1.0) if method == "REML" else max(float(n), 1.0)
                self._log_phi_hat = float(
                    np.log(max(Dp / denom, 1e-300))
                )
        else:
            # Unified outer optimization. PIRLS inner solve + general
            # `_reml(ρ, log φ)` + analytical Newton, family-agnostic.
            # ``include_log_phi`` is True for unknown-scale (Gaussian, Gamma,
            # IG): θ = (ρ, log φ). False for known-scale (Poisson, Binomial):
            # θ = ρ with log φ ≡ 0. mgcv's gam.outer behaves the same way.
            family = self.family
            include_log_phi = (not family.scale_known) and method in ("REML", "ML")
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

            if method in ("REML", "ML"):
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
                criterion="REML" if method in ("REML", "ML") else "GCV",
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
        # Fisher working weights — needed by reTest (Wood 2013) so summary()
        # can rebuild X'WX without re-running PIRLS. None ↔ unit weights.
        self._fisher_w = (
            np.asarray(fit_F.w, dtype=float).copy() if fit_F.w is not None else None
        )
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
        # families, mgcv reports `m$sig2 = m$scale = scale.est` (the
        # Pearson/deviance estimator, gam.fit3.r:606), regardless of method.
        # This differs from `m$reml.scale = exp(log φ̂)` — the optimizer's
        # converged scale that enters the score formula. For REML on
        # Gaussian-identity the two coincide at the optimum (FOC enforces
        # Dp/(n−Mp) = dev/(n−edf)); for ML they differ since φ̂_ML = Dp/n.
        # ``_log_phi_hat`` is preserved separately for score evaluation.
        df_resid = float(n - edf_total)
        if df_resid > 0 and not self.family.scale_known:
            V = self.family.variance(fit.mu)
            pearson_scale = float(np.sum(wt * (y - fit.mu) ** 2 / V)) / df_resid
        else:
            pearson_scale = 1.0 if self.family.scale_known else float("nan")
        self._pearson_scale = pearson_scale
        scale = 1.0 if self.family.scale_known else pearson_scale
        sigma_squared = scale                 # alias kept for back-compat
        sigma = float(np.sqrt(sigma_squared)) if np.isfinite(sigma_squared) and sigma_squared >= 0 else float("nan")

        Vp = sigma_squared * A_inv
        Ve = sigma_squared * A_inv_XtWX @ A_inv

        # ------------- coefficient basis change (G_P) -----------------------
        # When a smooth's predict basis differs from its fit basis (today
        # only ``t2`` with null_dim ≥ 1), β was fit in a basis that doesn't
        # match what ``predict_mat`` returns. ``estimate.gam`` (mgcv,
        # smooth.r:264-267) handles this with a single ``coefficients <-
        # G$P %*% coefficients`` (and ``Vp <- G$P Vp G$P^T``) post-fit.
        # ``G_P`` is identity except: each remapped block's columns rotate
        # by ``M`` and contribute ``X̄ · β_block`` into the intercept row,
        # encoding ``X_fit = 1·X̄ + X_predict @ M`` exactly. With this in
        # place ``X_fit @ β_partial = X_predict @ (M β_partial) + (X̄ ·
        # β_partial)·1`` — so the in-sample η is unchanged and out-of-sample
        # ``predict_mat(new) @ G_P @ β_partial`` equals what the fit basis
        # would have produced.
        intercept_idx: Optional[int] = (
            column_names.index("(Intercept)") if has_intercept else None
        )
        if any(b.spec is not None and b.spec.coef_remap is not None for b in blocks):
            G_P = np.eye(p)
            for b, (a_col, b_col) in zip(blocks, block_col_ranges):
                if b.spec is None or b.spec.coef_remap is None:
                    continue
                M_b, X_bar_b = b.spec.coef_remap
                G_P[a_col:b_col, a_col:b_col] = M_b
                if intercept_idx is not None:
                    G_P[intercept_idx, a_col:b_col] = X_bar_b
            beta = G_P @ beta
            Vp = G_P @ Vp @ G_P.T
            Ve = G_P @ Ve @ G_P.T

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

        # Penalized hat-matrix diagonal h_ii = w_i·(X·A_F⁻¹·X')_ii — mgcv's
        # `m$hat`, sums to edf_total. Plus rstandard.gam-style standardized
        # residuals: r / (σ̂·√(1−h)). For Gaussian-identity fit_F.w is None ⇒
        # unit weights. Cached here so plot_* methods don't recompute.
        w_F = fit_F.w if fit_F.w is not None else np.ones(n)
        HX = X @ A_inv
        self.leverage = (HX * X).sum(axis=1) * w_F
        sigma_for_std = sigma if np.isfinite(sigma) and sigma > 0 else 1.0
        denom = sigma_for_std * np.sqrt(np.clip(1.0 - self.leverage, 1e-12, None))
        V_mu = self.family.variance(mu)
        pearson_res = (y - mu) * np.sqrt(self._wt / np.maximum(V_mu, 0.0))
        self.std_dev_residuals = self.residuals / denom
        self.std_pearson_residuals = pearson_res / denom
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
            method in ("REML", "ML")
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

        if method in ("REML", "ML"):
            if n_sp > 0:
                # `_reml` returns -2·V_R (REML) or -2·V_ML (ML); `summary()`'s
                # `/2` recovers mgcv's `-REML`/`-ML` display value. Scale-known
                # families (Poisson, Binomial) substitute log φ = 0; scale-
                # unknown read the outer-optimizer's (or sp= path's profile-out)
                # log φ̂.
                log_phi_hat = (
                    self._log_phi_hat if self._log_phi_hat is not None else 0.0
                )
                score = float(self._reml(rho_hat, log_phi_hat, fit=fit))
            else:
                score = float("nan")
            if method == "REML":
                self.REML_criterion = score
            else:
                self.ML_criterion = score
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
        off = self._offset
        n, p = self.n, self.p
        Sλ = self._build_S_lambda(rho)
        Sλ = 0.5 * (Sλ + Sλ.T)
        wt = np.ones(n)                 # prior weights = 1 (offset is plumbed; prior-w lands later)

        # ``eta`` here is the *offset-stripped* β-only predictor X·β; the
        # full linear predictor is ``eta + off``. Mirrors glm._irls. We
        # solve weighted LS on (z - off) ~ X to recover β each step.

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
        eta = link.link(mu) - off       # β-only η
        beta = np.zeros(p)

        mu_null_const = float(np.average(mu, weights=wt))
        eta_null_full = link.link(np.full(n, mu_null_const))
        # Solve null_coef from X·null_coef = (full η at null) − offset.
        null_coef, *_ = np.linalg.lstsq(X, eta_null_full - off, rcond=None)
        eta_null = X @ null_coef
        mu_null = link.linkinv(eta_null + off)
        if not (link.valideta(eta_null + off) and family.validmu(mu_null)):
            # Constant-η projection drifted out of valid region — only
            # plausible for an X with no near-constant column. Fall back
            # to zeros; if the canonical link rejects η=off the user will
            # still get a clear error from the validity step-halver below
            # rather than silent divergence.
            null_coef = np.zeros(p)
            eta_null = np.zeros(n)
            mu_null = link.linkinv(off)
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
        while not (link.valideta(eta + off) and family.validmu(mu)):
            ii += 1
            if ii > 20:
                raise FloatingPointError(
                    "PIRLS init: cannot find valid starting μ̂"
                )
            eta = 0.9 * eta + 0.1 * eta_old
            mu = link.linkinv(eta + off)

        eps = 1e-8
        max_it = 50
        for it in range(max_it):
            eta_full = eta + off
            mu_eta_v = link.mu_eta(eta_full)
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
            # Working response, offset-stripped: z = (full η + (y-μ)/μ_η) - off.
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
            eta_new = X @ start         # β-only η
            if np.any(~np.isfinite(start)):
                raise FloatingPointError("non-finite β in PIRLS")

            mu_new = link.linkinv(eta_new + off)
            # If μ leaves the family's valid region, halve the step toward
            # the previous iterate (mgcv "inner loop 2").
            ii = 0
            while not (link.valideta(eta_new + off) and family.validmu(mu_new)):
                ii += 1
                if ii > max_it:
                    raise FloatingPointError("PIRLS step halving failed (validity)")
                start = 0.5 * (start + beta_old)
                eta_new = 0.5 * (eta_new + eta_old)
                mu_new = link.linkinv(eta_new + off)

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
                mu_new = link.linkinv(eta_new + off)
                if not (link.valideta(eta_new + off) and family.validmu(mu_new)):
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
        eta_full = eta + off
        mu_eta_v = link.mu_eta(eta_full)
        V = family.variance(mu)
        d2g = link.d2link(mu)
        alpha = 1.0 + (y - mu) * (family.dvar(mu) / V + d2g * mu_eta_v)
        alpha = np.where(alpha == 0.0, np.finfo(float).eps, alpha)
        z = eta + (y - mu) / (mu_eta_v * alpha)   # offset-stripped working response
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

        # ``eta`` here is offset-stripped; downstream consumers
        # (linear_predictors, predict, residuals_of) expect the full
        # linear predictor — return ``eta + off``.
        return _FitState(
            beta=beta, dev=dev, pen=pen,
            A_chol=A_chol, A_chol_lower=lower,
            S_full=Sλ, log_det_A=log_det_A,
            eta=eta + off, mu=mu, w=w, z=z, alpha=alpha,
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

    def _ml_logdet_adj(self, fit: "_FitState"):
        """Adjustment to convert log|H+S| (REML) → log|H_pp+S_pp| (ML).

        Direct port of mgcv ``MLpenalty1`` (gdi.c:1532-1680): for ML the
        Laplace approximation marginalises only over the *range* of Sλ
        (dropping Mp null-space columns of the QR factor R before the
        log-det). For REML it uses the full Hessian.

        Identity used here (block determinant on (range, null) basis):

            log|A_pp| = log|A| + log|U_nᵀ A⁻¹ U_n|

        where U_n is an orthonormal basis for null(Sλ), Mp = dim null(Sλ).

        Returns (logdet_adj, M_inv, B) with B = A⁻¹U_n (q×Mp) and
        M = U_nᵀ B (Mp×Mp). The latter two feed the gradient correction
        in ``_dlog_det_H_drho_ml``. ``logdet_adj = log|M|`` is added to
        ``fit.log_det_A`` to obtain log|H_pp + S_pp|.
        """
        Mp = self._Mp
        if Mp == 0:
            return 0.0, None, None
        # Null basis from eigendecomp of Sλ. Bottom Mp eigenvalues are
        # exactly 0 by construction (structural null space), so taking the
        # bottom-Mp eigenvectors picks out a stable U_n regardless of ρ.
        Sλ_sym = 0.5 * (fit.S_full + fit.S_full.T)
        w, V = np.linalg.eigh(Sλ_sym)
        U_n = V[:, :Mp]
        B = cho_solve((fit.A_chol, fit.A_chol_lower), U_n)
        M = U_n.T @ B
        sign, logdet_M = np.linalg.slogdet(M)
        if sign <= 0 or not np.isfinite(logdet_M):
            return 0.0, None, None
        M_inv = np.linalg.inv(M)
        return float(logdet_M), M_inv, B

    def _reml(self, rho: np.ndarray, log_phi: float = 0.0,
                      fit: "_FitState | None" = None) -> float:
        """Laplace-approximate (RE)ML in 2·V units, family/link-agnostic.

        Direct port of mgcv's gam.fit3.r:616 with `remlInd ∈ {1, 0}`:

            2·V = Dp/φ − 2·ls0 + log|H_*| − log|Sλ|_+
                  − remlInd·Mp·(log(2π·φ) − log γ)

        where the Hessian log-determinant differs by method:

            REML: log|H_*| = log|X'WX + Sλ|             (full)
            ML  : log|H_*| = log|U_rᵀ(X'WX + Sλ)U_r|    (range only)

        with U_r an orthonormal basis for range(Sλ). For ML the Laplace
        approximation marginalises only over the penalised subspace, so
        the Mp null-space directions are dropped — see mgcv's
        ``MLpenalty1`` in gdi.c:1532-1680. We compute the range-space
        log-det as ``fit.log_det_A + log|U_nᵀ A⁻¹ U_n|`` (block
        determinant identity, with U_n the null basis).

        ``remlInd = 1`` for ``method="REML"`` (mgcv's default; profiles
        out the unpenalized fixed-effect null-space prior of dimension
        Mp). ``remlInd = 0`` for ``method="ML"`` (treats those β as
        deterministic — score is comparable across different fixed-
        effect structures, suitable for likelihood-ratio tests).

        Dp = fit.dev + β̂'Sλβ̂ at PIRLS-converged β̂ and
        ls0 = family.ls(y, wt, φ)[0]. ``fit.log_det_A`` is the un-φ-scaled
        log|X'WX + Sλ|; the φ-coefficients of the prior-normalisation term
        and the Hessian/penalty Jacobi cancel everywhere except the
        −Mp·log(2π·φ) prior-rank term — see the Laplace derivation in
        Wood 2017 §6.6.

        Reduction-to-Gaussian (REML): profile out φ̂ = Dp/(n−Mp) and
        substitute. With Gaussian ls0 = −n·log(2πφ)/2 (wt=1),

            2·V_R(φ̂) = (n−Mp)·(1 + log(2π·Dp/(n−Mp)))
                       + log|A| − log|S|_+

        which equals ``_reml(rho)`` exactly under method="REML". For
        method="ML" the analogous profile-out is φ̂ = Dp/n.
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
        log_det_H = fit.log_det_A
        if self.method == "ML":
            adj, _, _ = self._ml_logdet_adj(fit)
            log_det_H = log_det_H + adj
        # mgcv (gam.fit3.r:622): ``gamma`` divides the data-fit piece
        # (Dp/φ − 2·ls0) and adds a +Mp·log(γ) constant to compensate the
        # −Mp·log(2πφ) prior-rank term so the criterion stays consistent
        # with the partially-profiled likelihood interpretation. For
        # method="ML", remlInd=0 drops both Mp pieces — β is treated as
        # deterministic, so there is no fixed-effect prior to integrate out.
        gamma = self._gamma
        reml_ind = 1.0 if self.method == "REML" else 0.0
        return (
            (Dp / phi - 2.0 * ls0) / gamma
            + log_det_H
            - log_det_S
            - reml_ind * (Mp * float(np.log(2.0 * np.pi * phi))
                          - Mp * float(np.log(gamma)))
        )

    def _reml_grad(self, rho: np.ndarray, log_phi: float = 0.0,
                           fit: "_FitState | None" = None,
                           include_log_phi: bool = False) -> np.ndarray:
        """Analytical gradient of `_reml` (2·V_R units).

        Length n_sp if `include_log_phi=False`, else n_sp+1 with log_phi
        appended. Wood 2011 §4 + mgcv gam.fit3.r:622, 630:

            ∂(2·V_R)/∂ρ_k    = (∂Dp/∂ρ_k)/φ + ∂log|H|/∂ρ_k − ∂log|S|+/∂ρ_k
            ∂(2·V_R)/∂log φ  = −Dp/φ − 2·ls'_hea − Mp

        ls'_hea is the d/d(log φ) chain-rule output from `family.ls(y, wt, φ)[1]`
        (hea convention, see family.py:338 docstring).
        """
        if fit is None:
            fit = self._fit_given_rho(rho)
        n_sp = len(self._slots)
        phi = float(np.exp(log_phi))
        if not (np.isfinite(phi) and phi > 0):
            size = n_sp + (1 if include_log_phi else 0)
            return np.full(size, 1e15)

        gamma = self._gamma
        if n_sp == 0:
            grad_rho = np.zeros(0)
        else:
            dDp = self._dDp_drho(fit, rho)
            dlog_H = self._dlog_det_H_drho(fit, rho)
            dlog_S = self._dlog_det_S_drho(rho, S_full=fit.S_full)
            # method="ML" uses the range-only Hessian log-det. With the
            # block-determinant identity log|H_pp+S_pp| = log|H+S| + log|M|
            # (M = U_nᵀ A⁻¹ U_n), the gradient picks up
            #     ∂log|M|/∂ρ_k = −tr(M⁻¹ · B′(∂A/∂ρ_k)B)
            # ∂A/∂ρ_k = X′·diag(∂w/∂ρ_k)·X + λ_k·S_k (the W-dep term is
            # nonzero for non-canonical families like binomial), so the
            # correction has two pieces. Mirrors mgcv's ``MLpenalty1`` →
            # ``get_ddetXWXpS`` in gdi.c, which fills trA1 with the
            # ML-version derivatives via the same projected-Hessian logic.
            if self.method == "ML":
                _, M_inv, B = self._ml_logdet_adj(fit)
                if M_inv is not None:
                    sp = np.exp(rho)
                    Y = self._X_full @ B                  # (n, Mp)
                    Y_Minv = Y @ M_inv                    # (n, Mp)
                    q = np.einsum("ij,ij->i", Y, Y_Minv)  # (n,) y_i' M⁻¹ y_i
                    dw_deta = self._dw_deta(fit)
                    db_drho = self._dbeta_drho(fit, rho)
                    deta_drho = self._X_full @ db_drho     # (n, n_sp)
                    dw_drho = dw_deta[:, None] * deta_drho # (n, n_sp)
                    for k, slot in enumerate(self._slots):
                        a, b = slot.col_start, slot.col_end
                        Bk = B[a:b, :]
                        Pk = Bk.T @ slot.S @ Bk
                        # −tr(M⁻¹ Y′ diag(dw/dρ_k) Y) − λ_k tr(M⁻¹ P_k)
                        dlog_H[k] += (
                            -float(np.sum(dw_drho[:, k] * q))
                            - sp[k] * float(np.einsum("ij,ji->", M_inv, Pk))
                        )
            # ∂Dp/∂ρ comes from the data-fit term, so γ divides it; the
            # log|H| / log|S|+ Jacobi pieces are γ-independent.
            grad_rho = dDp / (phi * gamma) + dlog_H - dlog_S

        if not include_log_phi:
            return grad_rho

        Mp = float(self._Mp)
        wt = np.ones(self.n)
        Dp = fit.dev + fit.pen
        ls = np.asarray(self.family.ls(self._y_arr, wt, phi), dtype=float)
        ls1 = float(ls[1])    # d ls / d(log φ), already chain-ruled
        # Data-fit pieces (-Dp/φ - 2·ls1) divide by γ; the -Mp piece comes
        # from -Mp·log(2πφ) (γ-independent) and is REML-only — under
        # method="ML" remlInd=0 drops it (gam.fit3.r:628).
        reml_ind = 1.0 if self.method == "REML" else 0.0
        d_logphi = (-Dp / phi - 2.0 * ls1) / gamma - reml_ind * Mp
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
          ∂²(2·V_R)/∂log φ²    = Dp/φ − 2·ls'_hea_2

        where ``ls'_hea_2 = family.ls(y, wt, φ)[2]`` (chain-ruled to log φ).

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
        gamma = self._gamma
        if n_sp == 0:
            H = np.zeros((size, size))
            if include_log_phi:
                Dp0 = fit.dev + fit.pen
                ls = np.asarray(self.family.ls(self._y_arr,
                                               np.ones(self.n), phi))
                H[0, 0] = (Dp0 / phi - 2.0 * float(ls[2])) / gamma
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

                cross_2VR = d2Dp / (phi * gamma) + d2logH_ij - d2logS_ij
                if i == j:
                    # Diagonal also picks up the δ_lk·g_k from ∂²Dp,
                    # δ_lk·λ_l·tr(H⁻¹·S_l) from ∂²H, and δ_lk·λ_k·tr(S⁺ S_k)
                    # from ∂²log|S|+. Only the ∂²Dp piece is γ-scaled.
                    H2[i, i] = (
                        cross_2VR
                        + g[i] / (phi * gamma)
                        + sp[i] * tr_AinvS[i]
                        - sp[i] * tr_SpinvS[i]
                    )
                else:
                    H2[i, j] = H2[j, i] = cross_2VR

        if not include_log_phi:
            return H2

        # Augment with log φ row/col. Cross / log φ² come from the data-fit
        # term (Dp/φ − 2·ls0), so they scale by 1/γ.
        H_aug = np.zeros((n_sp + 1, n_sp + 1))
        H_aug[:n_sp, :n_sp] = H2
        for k in range(n_sp):
            cross = -g[k] / (phi * gamma)
            H_aug[k, n_sp] = cross
            H_aug[n_sp, k] = cross
        Dp = fit.dev + fit.pen
        ls = np.asarray(self.family.ls(self._y_arr, np.ones(self.n), phi))
        H_aug[n_sp, n_sp] = (Dp / phi - 2.0 * float(ls[2])) / gamma
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
            self._outer_info = {
                "conv": "initial fit failed", "iter": 0,
                "grad": np.zeros_like(theta), "hess": np.zeros((theta.size, theta.size)),
                "score": float(f_prev), "score_scale": float("nan"),
            }
            return theta

        conv_text = "iteration limit reached"
        last_grad = np.zeros_like(theta)
        last_hess = np.zeros((theta.size, theta.size))
        it_done = 0
        for it in range(max_iter):
            rho, log_phi = _split(theta)
            grad = _grad(rho, log_phi, fit)
            H = _hess(rho, log_phi, fit)
            H = 0.5 * (H + H.T)
            last_grad, last_hess = grad, H

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
                conv_text = "step failed"
                it_done = it + 1
                break
            theta = theta_try
            df = abs(f_try - f_prev)
            f_old = f_prev
            f_prev = f_try
            fit = fit_try
            it_done = it + 1

            # mgcv's two-part stopping test (Newton.r):
            #   max(|grad|) ≤ score_scale·conv_tol·5
            #   |Δscore|    ≤ score_scale·conv_tol
            score_scale = _score_scale(fit, f_prev)
            if (
                float(np.abs(grad).max()) <= score_scale * conv_tol * 5.0
                and df <= score_scale * conv_tol
            ):
                conv_text = "full convergence"
                break
            if df < 1e-12 * (1.0 + abs(f_prev)):
                conv_text = "full convergence"
                break

        # Recompute final grad/hess at the converged θ so the diagnostics
        # reflect the *accepted* step (last_grad above is from the iter's
        # entry θ, which after a successful step is one back from final).
        rho_f, log_phi_f = _split(theta)
        last_grad = _grad(rho_f, log_phi_f, fit)
        last_hess = _hess(rho_f, log_phi_f, fit)
        last_hess = 0.5 * (last_hess + last_hess.T)
        self._outer_info = {
            "conv": conv_text,
            "iter": it_done,
            "grad": last_grad,
            "hess": last_hess,
            "score": float(f_prev),
            "score_scale": float(_score_scale(fit, f_prev)),
        }
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
        # mgcv (gam.fit3.r): ``gamma`` inflates the apparent edf cost in
        # the criterion: V_g = n·D / (n − γ·τ)²; V_u = D/n + 2·γ·τ/n − 1.
        gamma = self._gamma
        if self.family.scale_known:
            return fit.dev / n + 2.0 * gamma * edf_total / n - 1.0
        denom = n - gamma * edf_total
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

        # ``gamma`` inflates τ in the criterion: V_g = n·D/(n−γ·τ)²,
        # V_u = D/n + 2γτ/n − 1. Chain-rule the τ-derivative pieces by γ.
        gamma = self._gamma
        if family.scale_known:
            return dD_drho / n + 2.0 * gamma * dtau_drho / n
        denom = n - gamma * edf_total
        if denom <= 0:
            return np.zeros(n_sp)
        return (
            n * dD_drho / (denom * denom)
            + 2.0 * n * gamma * fit.dev * dtau_drho / (denom**3)
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
        # ``gamma`` inflates the τ-coefficient in V_u and V_g; chain-rule
        # picks up γ at every τ-derivative encounter.
        gamma = self._gamma
        if family.scale_known:
            return d2D / n + 2.0 * gamma * d2tau / n

        denom = n - gamma * edf_total
        if denom <= 0:
            return np.full((n_sp, n_sp), 1e15)

        Dn = float(fit.dev)
        dD_dτ = np.outer(dD_drho, dtau_drho)
        dτ_dτ = np.outer(dtau_drho, dtau_drho)
        H = (
            n * d2D / (denom * denom)
            + 2.0 * n * gamma * (dD_dτ + dD_dτ.T) / (denom**3)
            + 2.0 * n * gamma * Dn * d2tau / (denom**3)
            + 6.0 * n * (gamma ** 2) * Dn * dτ_dτ / (denom**4)
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

    def _test_stat_type0(
        self,
        X_b: np.ndarray,
        V_b: np.ndarray,
        beta_b: np.ndarray,
        rank: float,
    ) -> tuple[float, float]:
        """mgcv ``testStat`` with ``type = 0`` (summary.r default).

        Returns ``(stat, rank_out)`` where ``stat`` is the d-statistic and
        ``rank_out`` is the (possibly truncated) rank used as the F numerator
        d.f. The "fractional rank" correction blends the k-th and (k+1)-th
        whitened eigenvectors via a 2×2 symmetric square root so the test
        respects a non-integer reference d.f. Equivalent type=1 (rounded
        rank) gives a discontinuous F as edf1 crosses an integer; type=0
        is what mgcv summary actually calls.
        """
        # QR on the smooth's design block, then rotate Vp into that basis.
        _, R = np.linalg.qr(X_b, mode="reduced")
        V_rot = R @ V_b @ R.T
        V_rot = 0.5 * (V_rot + V_rot.T)
        d_eig, U = np.linalg.eigh(V_rot)
        # Descending order, mgcv sign convention (first row >= 0).
        d_eig = d_eig[::-1]
        U = U[:, ::-1]
        siv = np.sign(U[0, :])
        siv = np.where(siv == 0, 1.0, siv)
        U = U * siv

        k = max(0, int(np.floor(rank)))
        nu = abs(rank - k)
        k1 = k + 1 if nu > 0 else k

        # mgcv's effective-rank guard: if eigenvalue tail is below
        # max·eps^0.9, drop them and shrink k1.
        if d_eig.size > 0 and d_eig[0] > 0:
            r_est = int(np.sum(d_eig > d_eig[0] * np.finfo(float).eps ** 0.9))
        else:
            r_est = 0
        if r_est < k1:
            k1 = k = r_est
            nu = 0.0
            rank = float(r_est)

        if k1 == 0 or U.shape[1] == 0:
            return 0.0, float(rank)

        vec = U[:, :k1].copy()

        if nu > 0 and k > 0:
            # Whiten cols 0 .. k-2 (R: cols 1..k-1) by 1/sqrt(eigenvalue).
            if k > 1:
                scales = 1.0 / np.sqrt(d_eig[:k - 1])
                vec[:, :k - 1] = vec[:, :k - 1] * scales[np.newaxis, :]
            b12 = 0.5 * nu * (1.0 - nu)
            b12 = float(np.sqrt(max(b12, 0.0)))
            B = np.array([[1.0, b12], [b12, nu]], dtype=float)
            ev = np.diag(d_eig[k - 1:k + 1] ** -0.5)
            B = ev @ B @ ev
            eb_d, eb_v = np.linalg.eigh(B)
            rB = eb_v @ np.diag(np.sqrt(np.maximum(eb_d, 0.0))) @ eb_v.T
            cols_orig = vec[:, k - 1:k + 1].copy()
            # vec1 negates the first of the two cols before rB.
            cols_neg = cols_orig.copy()
            cols_neg[:, 0] = -cols_neg[:, 0]
            vec[:, k - 1:k + 1] = cols_orig @ rB.T
            vec1 = vec.copy()
            vec1[:, k - 1:k + 1] = cols_neg @ rB.T
        else:
            if k == 0:
                # Degenerate: scale all of vec by 1/sqrt(d_eig[0]).
                if d_eig[0] > 0:
                    vec = vec * (1.0 / np.sqrt(d_eig[0]))
            else:
                scales = 1.0 / np.sqrt(d_eig[:k])
                vec = vec * scales[np.newaxis, :]
            vec1 = vec

        Rp = R @ beta_b
        proj = vec.T @ Rp
        stat = float(np.sum(proj ** 2))
        return stat, float(rank)

    def _recov_no_re(self, m_idx: int) -> np.ndarray:
        """Port of ``mgcv:::recov`` for the no-RE case (re=∅, m>0).

        Returns ``Rm`` such that ``Rm' Rm`` is the m-th block's Schur
        complement of A = X'WX + Sλ — i.e. the inverse of ``A⁻¹[m,m]``,
        the precision of β̂_m after profiling out the rest. Built by stacking
        the model-matrix R factor (chol(X'WX)) on top of the penalty
        square-root, reordering target cols last, then taking the bottom-right
        block of the QR's R.
        """
        p = self.p
        a, bcol = self._block_col_ranges[m_idx]
        k = bcol - a
        # X'WX from stored Fisher working weights.
        if self._fisher_w is None:
            XtWX = self._X_full.T @ self._X_full
        else:
            Xw = self._X_full * np.sqrt(self._fisher_w)[:, None]
            XtWX = Xw.T @ Xw
        # Cholesky of X'WX: R_factor' R_factor = X'WX (upper-triangular).
        # Use eigendecomp+jitter when XtWX is borderline-PSD (gam.side
        # rank-trim, near-singular weights, etc.).
        try:
            R_factor = np.linalg.cholesky(XtWX).T
        except np.linalg.LinAlgError:
            ev, U = np.linalg.eigh(0.5 * (XtWX + XtWX.T))
            ev = np.clip(ev, 0.0, None)
            R_factor = (U * np.sqrt(ev)).T
        # Penalty square-root sqrtS such that sqrtS' sqrtS = Sλ.
        S_lam = self._build_S_lambda(self._rho_hat)
        ev, U = np.linalg.eigh(0.5 * (S_lam + S_lam.T))
        max_ev = ev.max() if ev.size else 0.0
        keep = ev > max(max_ev, 0.0) * 1e-12
        if keep.any():
            sqrtS = (U[:, keep] * np.sqrt(ev[keep])).T
        else:
            sqrtS = np.zeros((0, p), dtype=float)
        LRB = np.vstack([R_factor, sqrtS])
        # Reorder columns: target block last.
        target = list(range(a, bcol))
        other = [j for j in range(p) if j < a or j >= bcol]
        perm = other + target
        LRB_perm = LRB[:, perm]
        _, R_qr = np.linalg.qr(LRB_perm, mode="reduced")
        return R_qr[-k:, -k:]

    def _re_test(
        self, m_idx: int, beta_b: np.ndarray, Vp_b: np.ndarray
    ) -> tuple[float, float, float]:
        """Port of ``mgcv:::reTest`` (no-RE branch). Returns ``(stat, pval,
        rank)``. Uses ``psum_chisq`` (Davies 1980) for the p-value.

        - Wood (2013) "On p-values for smooth components of an extended GAM",
          Biometrika 100(1), 221–228.
        - mgcv source: ``R/mgcv.r``.
        """
        from ._pchisqsum import psum_chisq
        sig2 = float(self.sigma_squared) if np.isfinite(self.sigma_squared) \
            and self.sigma_squared > 0 else 1.0
        Rm = self._recov_no_re(m_idx)
        # Ve[ind, ind] half-square-root via eigendecomp.
        Ve_b = self.Ve[
            self._block_col_ranges[m_idx][0]:self._block_col_ranges[m_idx][1],
            self._block_col_ranges[m_idx][0]:self._block_col_ranges[m_idx][1],
        ]
        ev_b, U_b = np.linalg.eigh(0.5 * (Ve_b + Ve_b.T))
        ev_b = np.clip(ev_b, 0.0, None)
        B = U_b * np.sqrt(ev_b)
        d = Rm @ beta_b
        stat = float((d * d).sum() / sig2)
        M = Rm @ B
        ev = np.linalg.eigvalsh(0.5 * ((M.T @ M) + (M.T @ M).T) / sig2)
        ev = np.clip(ev, 0.0, None)
        max_ev = ev.max() if ev.size else 0.0
        rank = int(np.sum(ev > max(max_ev, 0.0) * np.finfo(float).eps ** 0.8))
        if self.family.scale_known:
            pval = psum_chisq(stat, ev) if ev.size else float("nan")
        else:
            k_df = max(1, int(round(self.df_residuals)))
            lb = np.concatenate([ev, np.array([-stat / k_df])])
            df = np.concatenate(
                [np.ones(ev.size, dtype=int), np.array([k_df], dtype=int)]
            )
            pval = psum_chisq(0.0, lb, df) if ev.size else float("nan")
        return stat, float(pval), float(rank)

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
        where M_k = ∂L^{-T}/∂ρ_k and A = L L^T is hea's lower-Cholesky of
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
        if H is None or self.method not in ("REML", "ML") or not np.isfinite(self.sigma_squared):
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

    def predict(
        self,
        newdata: pl.DataFrame | None = None,
        type: str = "response",
        se_fit: bool = False,
        offset: np.ndarray | list | None = None,
    ):
        """Predict from the fitted GAM — :func:`predict.gam` parity.

        ``type='response'`` returns ``μ̂ = g⁻¹(X_new β̂ + offset)``;
        ``type='link'`` returns ``η̂ = X_new β̂ + offset``. With
        ``se_fit=True``, also returns the standard error: link-scale SE is
        ``√diag(X · Vp · Xᵀ)`` (offset is constant so it doesn't affect
        SE); response-scale SE multiplies by ``|dμ/dη|`` (delta method,
        same as mgcv).

        ``Vp`` is the Bayesian posterior covariance (``self.Vp``) — mgcv's
        default for ``se.fit`` since smoothing-parameter shrinkage makes the
        frequentist ``Ve`` over-confident at the posterior mode.

        With ``newdata`` and a formula offset, the offset is re-evaluated
        against ``newdata`` (mirrors ``predict.gam``). Pass ``offset=`` to
        override or to add an offset on top of the formula offset.
        """
        if type not in ("link", "response"):
            raise ValueError(
                f"type must be 'link' or 'response'; got {type!r}"
            )

        if newdata is None:
            X_new = self._X_full
            off_new = self._offset
        else:
            from .formula import materialize  # local to avoid cycle

            X_param = materialize(self._expanded, newdata).to_numpy().astype(float)
            cols = [X_param]
            for b in self._blocks:
                if b.spec is None:
                    raise RuntimeError(
                        f"smooth block {b.label!r} (cls={b.cls!r}) has no "
                        "BasisSpec; predict(newdata=...) requires every smooth "
                        "to carry one."
                    )
                cols.append(np.asarray(b.spec.predict_mat(newdata), dtype=float))
            X_new = np.concatenate(cols, axis=1) if len(cols) > 1 else X_param
            n_new = X_new.shape[0]
            # Re-evaluate any formula offset(...) atoms against newdata
            # — predict.gam does the same.
            off_new = np.zeros(n_new)
            for off_node in self._expanded.offsets:
                blk = _eval_atom(off_node, newdata)
                off_new = off_new + blk.values.flatten().astype(float)
        if offset is not None:
            extra = np.asarray(offset, dtype=float).flatten()
            if extra.shape != off_new.shape:
                raise ValueError(
                    f"offset must have length {off_new.shape[0]}, got {extra.shape}"
                )
            off_new = off_new + extra
        eta = X_new @ self._beta + off_new

        fit = eta if type == "link" else self.family.link.linkinv(eta)

        if not se_fit:
            return fit

        # Var(η̂_i) = X_i · Vp · X_iᵀ; rowwise via einsum.
        var_eta = np.einsum("ij,jk,ik->i", X_new, self.Vp, X_new)
        se_link = np.sqrt(np.maximum(var_eta, 0.0))
        if type == "link":
            return fit, se_link
        # Delta method: Var(μ̂) ≈ (dμ/dη)² · Var(η̂).
        mu_eta_v = self.family.link.mu_eta(eta)
        return fit, np.abs(mu_eta_v) * se_link

    def vis(
        self,
        view: tuple[str, str] | list[str] | None = None,
        cond: dict | None = None,
        n_grid: int = 30,
        type: str = "link",
        se: bool = False,
        too_far: float = 0.0,
    ) -> "VisResult":
        """2D model-surface viewer — :func:`vis.gam` parity.

        Builds an ``n_grid × n_grid`` grid over two ``view`` covariates, holds
        every other variable at its "typical" value (median for numeric, mode
        for factor — same as mgcv's ``variable.summary``), calls
        :meth:`predict` on the grid, and returns the surface as a
        :class:`VisResult` (which carries a ``.plot()`` method).

        Parameters
        ----------
        view : tuple of 2 str, optional
            Pair of covariate names to vary. If ``None``, picks the first two
            variables in ``self.data`` that have more than one unique value.
        cond : dict, optional
            Override the typical-value default for any non-view variable, e.g.
            ``cond={"sex": "M", "age": 50}``.
        n_grid : int
            Grid resolution per axis (default 30, matching mgcv).
        type : {"link", "response"}
            Scale of the returned fit/SE — ``"link"`` is η̂, ``"response"``
            applies the inverse link.
        se : bool
            If ``True``, also compute pointwise SE on the grid.
        too_far : float
            Mask grid points whose normalized distance to any data point
            exceeds this threshold (replaces fit/se with ``NaN``). 0 = no
            masking. Mirrors mgcv's ``exclude.too.far``.
        """
        if type not in ("link", "response"):
            raise ValueError(
                f"type must be 'link' or 'response'; got {type!r}"
            )

        vs = self._var_summary()

        if view is None:
            view = []
            # Iterate RHS variables in formula order (vs is built that way) —
            # mgcv's vis.gam picks the first two with variation, same idea.
            for name in vs:
                if _has_variation(self.data[name]):
                    view.append(name)
                    if len(view) == 2:
                        break
            if len(view) < 2:
                raise ValueError(
                    "could not auto-pick `view`: need at least two RHS "
                    "variables with more than one unique value"
                )
        else:
            view = list(view)
            if len(view) != 2:
                raise ValueError(
                    f"view must be a pair of variable names; got {view!r}"
                )
            for v in view:
                if v not in self.data.columns:
                    raise ValueError(
                        f"view variable {v!r} not in data; available: "
                        f"{list(self.data.columns)}"
                    )

        m1 = _grid_axis(self.data[view[0]], n_grid)
        m2 = _grid_axis(self.data[view[1]], n_grid)
        n1, n2 = len(m1), len(m2)

        # meshgrid with indexing='ij' so that reshape(n1, n2) puts m1 on axis 0
        # and m2 on axis 1 — i.e. fit[i, j] is the prediction at (m1[i], m2[j]).
        M1, M2 = np.meshgrid(m1, m2, indexing="ij")
        v1 = M1.ravel()
        v2 = M2.ravel()
        n_pts = n1 * n2

        cond = dict(cond or {})
        cols: dict[str, object] = {}
        for name in self.data.columns:
            if name == view[0]:
                cols[name] = v1
            elif name == view[1]:
                cols[name] = v2
            elif name in cond:
                cols[name] = np.repeat(cond[name], n_pts)
            elif name in vs:
                cols[name] = np.repeat(vs[name], n_pts)
            else:
                # Variable wasn't profiled by var_summary (e.g. an offset column
                # or a non-formula column) — leave it out; predict only
                # references columns named in the formula.
                continue

        # Re-impose the original schema (factor levels, dtypes) so PredictMat's
        # factor matching still works on the grid frame.
        new_df = pl.DataFrame(
            {
                k: (v if isinstance(v, pl.Series) else pl.Series(k, v))
                for k, v in cols.items()
            }
        )
        for name in new_df.columns:
            src = self.data[name]
            if src.dtype != new_df[name].dtype:
                new_df = new_df.with_columns(new_df[name].cast(src.dtype))

        if se:
            fit, se_arr = self.predict(new_df, type=type, se_fit=True)
        else:
            fit = self.predict(new_df, type=type, se_fit=False)
            se_arr = None

        if too_far > 0.0:
            mask = _too_far_mask(
                v1, v2, self.data[view[0]], self.data[view[1]], too_far
            )
            fit = np.array(fit, dtype=float, copy=True)
            fit[mask] = np.nan
            if se_arr is not None:
                se_arr = np.array(se_arr, dtype=float, copy=True)
                se_arr[mask] = np.nan

        fit_grid = np.asarray(fit, dtype=float).reshape(n1, n2)
        se_grid = (
            np.asarray(se_arr, dtype=float).reshape(n1, n2)
            if se_arr is not None
            else None
        )
        return VisResult(
            view=(view[0], view[1]),
            m1=np.asarray(m1),
            m2=np.asarray(m2),
            fit=fit_grid,
            se=se_grid,
            type=type,
        )

    def _var_summary(self) -> dict:
        """mgcv ``variable.summary`` parity: typical value per variable.

        Restricted to RHS variables of the formula (so we don't include the
        response or stray data columns). Numeric → median; factor/string →
        modal level.
        """
        from .formula import referenced_columns  # local to avoid cycle

        rhs_vars = referenced_columns(self._expanded)
        out: dict = {}
        for name in self.data.columns:
            if name not in rhs_vars:
                continue
            col = self.data[name]
            if _is_factor_like_col(col):
                vals = col.drop_nulls()
                if len(vals) == 0:
                    continue
                # Mode: most frequent level. polars `.mode()` returns all ties;
                # take the first to get a deterministic single value.
                out[name] = vals.mode().to_list()[0]
            else:
                arr = col.drop_nulls().to_numpy().astype(float)
                if arr.size == 0:
                    continue
                out[name] = float(np.median(arr))
        return out

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
        # mgcv (summary.gam): when scale.estimated, t/Pr(>|t|) on residual.df;
        # otherwise (binomial/poisson with φ ≡ 1) Wald z/Pr(>|z|).
        scale_known = bool(self.family.scale_known)
        if self.p_param > 0:
            out.append("Parametric coefficients:")
            est = self._beta[:self.p_param]
            se  = self._se[:self.p_param]
            with np.errstate(divide="ignore", invalid="ignore"):
                t_stats = est / se
            if scale_known:
                pv = 2 * norm.sf(np.abs(t_stats))
                stat_col = "z value"
                pcol = "Pr(>|z|)"
            elif self.df_residuals > 0 and np.isfinite(self.df_residuals):
                pv = 2 * t_dist.sf(np.abs(t_stats), self.df_residuals)
                stat_col = "t value"
                pcol = "Pr(>|t|)"
            else:
                pv = np.full_like(t_stats, np.nan)
                stat_col = "t value"
                pcol = "Pr(>|t|)"
            sig = significance_code(pv)
            est_s, se_s = format_signif_jointly([est, se], digits=digits)
            tbl = pl.DataFrame({
                "": self.parametric_columns,
                "Estimate":   est_s,
                "Std. Error": se_s,
                stat_col:     format_signif(t_stats, digits=digits),
                pcol:         format_pval(pv, digits=_dig_tst(digits)),
                " ":          sig,
            })
            out.append(format_df(
                tbl,
                align={c: "right" for c in
                       ("Estimate", "Std. Error", stat_col, pcol)},
            ))
            out.append("---")
            out.append(
                "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1"
            )
            out.append("")

        # -- smooth-edf table ----------------------------------------------
        # mgcv summary.gam dispatches per-smooth on null.space.dim and on
        # scale.estimated. Under select=TRUE the null-space penalty makes
        # null.space.dim == 0 for every smooth ⇒ reTest path (Wood 2013).
        # Without select=TRUE we fall back to testStat (the type=0 fractional
        # rank routine, _test_stat_type0). Output column header switches
        # F↔Chi.sq, and Ref.df reports the rank actually used in the test.
        if self._blocks:
            out.append("Approximate significance of smooth terms:")
            rows_label: list[str] = []
            rows_edf:   list[float] = []
            rows_refdf: list[float] = []
            rows_stat:  list[float] = []
            rows_p:     list[float] = []
            for m_idx, (b, (a, bcol)) in enumerate(
                zip(self._blocks, self._block_col_ranges)
            ):
                beta_b = self._beta[a:bcol]
                Vp_b   = self.Vp[a:bcol, a:bcol]
                X_b    = self._X_full[:, a:bcol]
                edf_b  = float(self.edf[a:bcol].sum())
                edf1_b = float(self.edf1[a:bcol].sum()) if hasattr(self, "edf1") else edf_b
                p_b = bcol - a
                if self._select:
                    # reTest path — null.space.dim==0 for every block.
                    stat, p_val, ref_df = self._re_test(m_idx, beta_b, Vp_b)
                    if scale_known:
                        # Chi.sq column = stat. F column = stat / rank.
                        col_stat = stat
                    else:
                        col_stat = stat / max(ref_df, 1e-8)
                else:
                    rank_in = float(min(p_b, edf1_b))
                    Tr, ref_df = self._test_stat_type0(X_b, Vp_b, beta_b, rank_in)
                    if scale_known:
                        # mgcv testStat with res.df=-1 uses chi^2 with df=rank
                        # for integer rank (the fractional path averages two
                        # psum_chisq calls; we approximate with rounded rank
                        # for known-scale select=False — rare in practice).
                        col_stat = Tr
                        df_int = max(1, int(round(ref_df)))
                        from scipy.stats import chi2 as _chi2_dist
                        p_val = float(_chi2_dist.sf(Tr, df_int))
                    else:
                        F = Tr / max(ref_df, 1e-8)
                        col_stat = F
                        p_val = (
                            float(f_dist.sf(F, ref_df, self.df_residuals))
                            if self.df_residuals > 0 else float("nan")
                        )
                rows_label.append(b.label)
                rows_edf.append(edf_b)
                rows_refdf.append(float(ref_df))
                rows_stat.append(col_stat)
                rows_p.append(p_val)
            sig = significance_code(rows_p)
            stat_col = "Chi.sq" if scale_known else "F"
            sm_tbl = pl.DataFrame({
                "":        rows_label,
                "edf":     format_signif(rows_edf, digits=digits),
                "Ref.df":  format_signif(rows_refdf, digits=digits),
                stat_col:  format_signif(rows_stat, digits=digits),
                "p-value": format_pval(rows_p, digits=_dig_tst(digits)),
                " ":       sig,
            })
            out.append(format_df(
                sm_tbl,
                align={c: "right" for c in
                       ("edf", "Ref.df", stat_col, "p-value")},
            ))
            out.append("---")
            out.append(
                "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1"
            )
            out.append("")

        # -- fit stats ------------------------------------------------------
        # mgcv: `formatC(r.sq, digits=3, width=5)`, `formatC(dev.expl*100,
        # digits=3)`, `formatC(REML/GCV/scale, digits=5)`. Match that.
        out.append(
            f"R-sq.(adj) = {self.r_squared_adjusted:.3g}  "
            f"Deviance explained = {self.deviance_explained * 100:.3g}%"
        )
        if self.method == "REML":
            out.append(
                f"-REML = {self.REML_criterion / 2:.5g}  "
                f"Scale est. = {self.sigma_squared:.5g}  n = {self.n}"
            )
        elif self.method == "ML":
            out.append(
                f"-ML = {self.ML_criterion / 2:.5g}  "
                f"Scale est. = {self.sigma_squared:.5g}  n = {self.n}"
            )
        else:
            out.append(
                f"GCV = {self.GCV_score:.5g}  "
                f"Scale est. = {self.sigma_squared:.5g}  n = {self.n}"
            )
        print("\n".join(out))

    def _k_check(
        self,
        type: str = "deviance",
        subsample: int = 5000,
        n_rep: int = 200,
        seed: int | None = None,
    ) -> pl.DataFrame | None:
        """Port of mgcv's ``k.check`` — basis-dimension test per smooth.

        For each smooth block, pair each residual with neighbours in
        covariate space and compare the mean squared first difference
        against a permutation null. A small ``k-index`` (≪ 1) and small
        p-value indicate the basis is too small to absorb the signal.

        1-D smooths: sort residuals by the covariate, take ``diff``.
        Multi-D smooths: average over the 3 nearest neighbours by
        Euclidean distance in raw covariate space. mgcv additionally
        rescales axes for tensor smooths via ``PredictMat`` gradient
        norms; hea has no PredictMat yet, so tensor (``te``/``ti``/
        ``t2``) k-indexes are not on mgcv's rescaled axes — the
        qualitative "k-index < 1" warning still applies.

        Returns a polars DataFrame with columns ``""``, ``"k'"``,
        ``"edf"``, ``"k-index"``, ``"p-value"`` (one row per smooth
        block), or ``None`` if there are no smooths.
        """
        if not self._blocks:
            return None

        rsd = self.residuals_of(type=type)
        n_full = len(rsd)
        rng = np.random.default_rng(seed)

        # Optional subsample (mgcv's `k.sample`). The same row indices
        # subset both residuals and the per-smooth covariate columns so
        # the neighbour graph stays consistent.
        if n_full > subsample:
            idx = rng.choice(n_full, size=subsample, replace=False)
            rsd = rsd[idx]
        else:
            idx = np.arange(n_full)
        nr = len(rsd)
        rsd_sq_mean = float(np.mean(rsd ** 2))
        if rsd_sq_mean <= 0:
            rsd_sq_mean = 1.0

        rows: list[tuple[str, float, float, float, float]] = []
        for b, (a, bcol) in zip(self._blocks, self._block_col_ranges):
            kc = float(bcol - a)
            edf_b = float(self.edf[a:bcol].sum())
            var_names = list(b.term)

            ok = bool(var_names)
            cols: list[np.ndarray] = []
            for v in var_names:
                if v not in self.data.columns:
                    ok = False
                    break
                s = self.data[v]
                if not s.dtype.is_numeric():
                    ok = False
                    break
                cols.append(s.to_numpy().astype(float)[idx])
            if not ok:
                rows.append((b.label, kc, edf_b, float("nan"), float("nan")))
                continue

            if len(cols) == 1:
                order = np.argsort(cols[0], kind="stable")
                rsd_o = rsd[order]
                v_obs = float(np.mean(np.diff(rsd_o) ** 2) / 2)
                ve = np.empty(n_rep)
                for i in range(n_rep):
                    shuf = rng.permutation(rsd)
                    ve[i] = np.mean(np.diff(shuf) ** 2) / 2
            else:
                from scipy.spatial import cKDTree
                Xnn = np.column_stack(cols)
                nn = 3
                # k=nn+1, skip column 0 (self at distance 0).
                tree = cKDTree(Xnn)
                _, ni = tree.query(Xnn, k=nn + 1)
                ni = ni[:, 1:]
                e_parts = [rsd - rsd[ni[:, j]] for j in range(nn)]
                v_obs = float(np.mean(np.concatenate(e_parts) ** 2) / 2)
                ve = np.empty(n_rep)
                for i in range(n_rep):
                    shuf = rng.permutation(rsd)
                    parts = [shuf - shuf[ni[:, j]] for j in range(nn)]
                    ve[i] = np.mean(np.concatenate(parts) ** 2) / 2

            p_val = float(np.mean(ve < v_obs))
            k_index = v_obs / rsd_sq_mean
            rows.append((b.label, kc, edf_b, float(k_index), p_val))

        return pl.DataFrame({
            "":        [r[0] for r in rows],
            "k'":      [r[1] for r in rows],
            "edf":     [r[2] for r in rows],
            "k-index": [r[3] for r in rows],
            "p-value": [r[4] for r in rows],
        })

    def check(
        self,
        type: str = "deviance",
        k_sample: int = 5000,
        k_rep: int = 200,
        seed: int | None = None,
    ) -> None:
        """mgcv-style ``gam.check``: convergence diagnostics + ``k.check`` table.

        Prints (no plotting — this is a non-graphical port):

        - Method / optimizer line.
        - Convergence status, iterations, gradient range.
        - Score and scale at the optimum.
        - Hessian positive-definiteness and eigenvalue range.
        - Per-smooth basis-dimension check table from ``_k_check``.

        Parameters
        ----------
        type : {"deviance", "pearson", "response"}
            Residual type passed to ``_k_check``. Default matches mgcv.
        k_sample : int
            Maximum residuals to use for the basis check
            (mgcv's ``k.sample``).
        k_rep : int
            Permutation reps for the k-check p-value
            (mgcv's ``k.rep``).
        seed : int | None
            Seeds the permutations and subsample for reproducibility.
            ``None`` uses fresh randomness each call.
        """
        out: list[str] = []

        # --- method / optimizer header ---
        method_label = self.method
        out.append(f"Method: {method_label}   Optimizer: outer newton")

        # --- convergence info from _outer_newton ---
        info = self._outer_info
        if info is None:
            if not self._blocks:
                out.append("Model required no smoothing parameter selection")
            else:
                out.append(
                    "Smoothing parameters fixed by user — no outer optimization."
                )
        else:
            iters = info["iter"]
            plural = "" if iters == 1 else "s"
            out.append(f"{info['conv']} after {iters} iteration{plural}.")
            grad = np.asarray(info["grad"])
            if grad.size > 0:
                out.append(
                    f"Gradient range [{float(grad.min()):.7g},"
                    f"{float(grad.max()):.7g}]"
                )
            score = info["score"]
            scale = self.sigma_squared
            out.append(f"(score {score:.7g} & scale {scale:.7g}).")
            H = np.asarray(info["hess"])
            if H.size > 0:
                ev = np.linalg.eigvalsh(0.5 * (H + H.T))
                ev_min, ev_max = float(ev.min()), float(ev.max())
                pd_text = (
                    "Hessian positive definite, "
                    if ev_min > 0
                    else "Hessian not positive definite, "
                )
                out.append(
                    f"{pd_text}eigenvalue range [{ev_min:.7g},{ev_max:.7g}]."
                )
        out.append(f"Model rank = {self.p} / {self.p}")
        out.append("")

        # --- basis dimension check ---
        ktab = self._k_check(
            type=type, subsample=k_sample, n_rep=k_rep, seed=seed,
        )
        if ktab is not None:
            out.append(
                "Basis dimension (k) checking results. Low p-value "
                "(k-index<1) may"
            )
            out.append(
                "indicate that k is too low, especially if edf is close to k'."
            )
            out.append("")
            kc_vals  = ktab["k'"].to_list()
            edf_vals = ktab["edf"].to_list()
            ki_vals  = ktab["k-index"].to_list()
            pv_vals  = ktab["p-value"].to_list()
            sig = significance_code(pv_vals)
            disp = pl.DataFrame({
                "":        ktab[""].to_list(),
                "k'":      format_signif(kc_vals,  digits=3, min_decimals=2),
                "edf":     format_signif(edf_vals, digits=3, min_decimals=2),
                "k-index": format_signif(ki_vals,  digits=3, min_decimals=2),
                "p-value": format_pval(pv_vals,    digits=2),
                " ":       sig,
            })
            out.append(format_df(
                disp,
                align={c: "right" for c in ("k'", "edf", "k-index", "p-value")},
            ))
            out.append("---")
            out.append(
                "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1"
            )

        print("\n".join(out))

    # ----- diagnostic plots -----------------------------------------------
    #
    # Match the graphical half of mgcv's gam.check + R's plot.glm:
    # - x-axis on residual panels = η̂ (linear predictors), labeled
    #   "Predicted values".
    # - panels 1/2/3 use deviance residuals (residuals.gam default).
    # - panel 5 (leverage) uses standardized Pearson residuals on y, with
    #   Cook's-distance contours scaled by edf_total.
    #
    # Per-smooth effect curves (mgcv's plot.gam) and the 2D fitted-surface
    # view (vis.gam) are separate plot methods, added in later passes.

    def plot_observed_fitted(
        self, ax=None, figsize=None,
        facecolor="none", edgecolor="black", label_n=3,
    ):
        if ax is None:
            _fig, ax = plt.subplots(figsize=figsize)
        y = self._y_arr
        yhat = self.fitted_values
        ax.scatter(yhat, y, facecolor=facecolor, edgecolor=edgecolor)
        lo = float(min(y.min(), yhat.min()))
        hi = float(max(y.max(), yhat.max()))
        ax.plot([lo, hi], [lo, hi], color="black", linestyle="--")
        _label_top_n(ax, yhat, y, scores=y - yhat, n=label_n)
        ax.set_xlabel("Fitted (μ̂)")
        ax.set_ylabel("Observed")
        ax.set_title("Observed vs. Fitted")
        return ax

    def plot_residuals(
        self, ax=None, figsize=None,
        facecolor="none", edgecolor="black",
        smooth=True, label_n=3,
    ):
        if ax is None:
            _fig, ax = plt.subplots(figsize=figsize)
        eta = self.linear_predictors
        r = self.residuals_of("deviance")
        ax.scatter(eta, r, facecolor=facecolor, edgecolor=edgecolor)
        ax.axhline(0, color="black", linestyle="--")
        if smooth:
            xs, ys = _lowess(eta, r)
            ax.plot(xs, ys, color="red", linewidth=1.0)
        _label_top_n(ax, eta, r, scores=r, n=label_n)
        ax.set_xlabel("Predicted values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals vs. Fitted Plot")
        return ax

    def plot_qq(self, ax=None, figsize=None, label_n=3):
        if ax is None:
            _fig, ax = plt.subplots(figsize=figsize)
        _qq_plot(
            ax, self.std_dev_residuals, label_n=label_n,
            ylabel="Std. deviance resid.",
        )
        return ax

    def plot_scale_location(
        self, ax=None, figsize=None,
        facecolor="none", edgecolor="black",
        smooth=True, label_n=3,
    ):
        if ax is None:
            _fig, ax = plt.subplots(figsize=figsize)
        eta = self.linear_predictors
        s = np.sqrt(np.abs(self.std_dev_residuals))
        ax.scatter(eta, s, facecolor=facecolor, edgecolor=edgecolor)
        if smooth:
            xs, ys = _lowess(eta, s)
            ax.plot(xs, ys, color="red", linewidth=1.0)
        _label_top_n(ax, eta, s, scores=self.std_dev_residuals, n=label_n)
        ax.set_xlabel("Predicted values")
        ax.set_ylabel(r"$\sqrt{|\mathrm{Std.\ deviance\ resid.}|}$")
        ax.set_title("Scale-Location")
        return ax

    def plot_leverage(
        self, ax=None, figsize=None,
        facecolor="none", edgecolor="black",
        cook_levels=(0.5, 1.0),
        smooth=True, label_n=3,
    ):
        if ax is None:
            _fig, ax = plt.subplots(figsize=figsize)
        h = self.leverage
        r = self.std_pearson_residuals
        ax.scatter(h, r, facecolor=facecolor, edgecolor=edgecolor)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        if smooth:
            xs, ys = _lowess(h, r)
            ax.plot(xs, ys, color="red", linewidth=1.0)
        # Cook's contours for GAM: D_i = (r²/k)·h/(1−h), k = edf_total —
        # the GAM analogue of GLM's `rank(X)` for the Bayesian penalized
        # hat matrix. Solving for r: r = ±sqrt(c·k·(1−h)/h).
        k = max(float(self.edf_total), 1.0)
        ymin, ymax = ax.get_ylim()
        h_max = float(np.clip(h.max() * 1.1, 1e-3, 0.999))
        h_grid = np.linspace(1e-3, h_max, 200)
        for c in cook_levels:
            rline = np.sqrt(c * k * (1 - h_grid) / h_grid)
            ax.plot(h_grid, rline, color="red", linestyle="--", linewidth=0.8)
            ax.plot(h_grid, -rline, color="red", linestyle="--", linewidth=0.8)
        ax.set_ylim(ymin, ymax)
        cook = (r ** 2 / k) * h / np.clip(1 - h, 1e-12, None)
        _label_top_n(ax, h, r, scores=cook, n=label_n)
        ax.set_xlabel("Leverage")
        ax.set_ylabel("Std. Pearson resid.")
        ax.set_title("Residuals vs. Leverage")
        return ax

    def plot_smooth(
        self,
        select=None,
        n_cols=2,
        figsize=None,
        color="black",
        band_color="black",
        band_alpha=0.2,
        rug=True,
        partial_residuals=False,
        n_grid: int = 40,
        too_far: float = 0.1,
        all_terms: bool = False,
        ax=None,
    ):
        """Per-smooth effect plots — the hea port of mgcv's ``plot.gam``.

        Auto-dispatches by smooth dimensionality:

        - **1D** ``s(x)`` → curve of ``f̂(x_i)`` with a 2·SE band, optional
          rug and partial residuals.
        - **2D** ``s(x,y)`` / ``te(x,y)`` → contour plot of ``f̂(x,y)``:
          bold contours for the estimate, dashed for ``f̂ − SE``, dotted
          for ``f̂ + SE`` (matches mgcv's ``sp.contour`` lty=1/2/3 — note
          Wood 2017 Fig. 4.14's caption inverts the SE assignments
          relative to the actual mgcv code). Data locations overlaid as a
          scatter.

        With ``all_terms=True``, parametric terms get their own panels
        (mgcv's ``plot.gam(..., all.terms=TRUE)`` behavior):

        - Factor term → horizontal-bar termplot, one bar per level
          (reference level pinned at 0), with ±SE dashed bars and a rug.
        - Numeric term → linear partial effect ``β·x`` with a 2·SE band.

        Multi-block factor-by smooths (e.g. ``s(x, by=g)`` for each level
        of ``g``) appear as separate panels — same as mgcv.

        Parameters
        ----------
        select : int | None
            If set, plot just the ``select``-th panel (0-indexed in the
            order panels would otherwise appear). Default plots every
            plottable panel in a grid. Required when ``ax=`` is given and
            the model has more than one plottable panel.
        n_cols : int
            Columns in the grid layout when ``select`` is None and ``ax`` is
            None.
        partial_residuals : bool
            (1D only) Overlay partial residuals (working residual + ``f̂_i``).
        rug : bool
            (1D only) Draw a rug of x-values at the bottom of each panel.
        n_grid : int
            (2D only) Per-axis grid resolution. Default 40 (mgcv uses 30).
        too_far : float
            (2D only) Mask grid points whose normalized distance to the
            nearest data point exceeds this threshold (mgcv's
            ``exclude.too.far``). Default 0.1 matches mgcv's plot.gam
            default; set to 0 to disable masking.
        all_terms : bool
            Also include parametric terms (factor / numeric, excluding the
            intercept) — Wood 2017 Fig. 4.15 layout.
        ax : matplotlib Axes | None
            If given, draw the (single) selected panel into this axes
            instead of building a new figure. The model must resolve to
            exactly one panel — either via ``select=`` or because there
            is only one plottable panel. Returns ``ax`` in that case
            (single-panel return convention); otherwise returns ``fig``.

        Returns
        -------
        Figure when building the multi-panel grid; Axes when ``ax=`` is
        provided.

        Notes
        -----
        Smooths of dimension ≥3, factor-smooth interactions (``bs="fs"``),
        and random-effect smooths (``bs="re"``) are still skipped. For ≥3D
        viewing use :meth:`vis` with ``view=`` to pick a 2D slice.
        """
        # Plottable panels: a list of dispatch records, each a tuple where
        # the first element is a discriminator string. Two kinds:
        #   ("smooth", block, a, bcol)
        #   ("param",  term_label, col_indices, kind)  kind ∈ {"factor", "numeric"}
        plottable: list[tuple] = []
        for idx, (b, (a, bcol)) in enumerate(
            zip(self._blocks, self._block_col_ranges)
        ):
            if len(b.term) not in (1, 2):
                continue
            if b.cls in ("re.smooth.spec", "fs.interaction", "sz.interaction"):
                continue
            plottable.append(("smooth", b, a, bcol))

        if all_terms and self._expanded.terms:
            param_cols = self.parametric_columns
            col_index_of = {c: i for i, c in enumerate(param_cols)}
            used = {"(Intercept)"} if "(Intercept)" in col_index_of else set()
            for term in self._expanded.terms:
                label = term.label
                term_cols = [
                    c for c in param_cols
                    if c not in used and (c == label or c.startswith(label))
                ]
                if not term_cols:
                    continue
                used.update(term_cols)
                col_idx = [col_index_of[c] for c in term_cols]
                # Classify the underlying variable: factor (Enum/Categorical/Utf8)
                # vs numeric. Skip terms whose variable can't be resolved
                # (interactions, transformed terms) — those need bespoke
                # rendering and aren't supported here yet.
                if label in self.data.columns:
                    dt = self.data[label].dtype
                    if dt in (pl.Enum, pl.Categorical, pl.Utf8):
                        plottable.append(("param", label, col_idx, "factor"))
                    elif dt.is_numeric():
                        plottable.append(("param", label, col_idx, "numeric"))
                    # else: skip (datetime, list, etc.)

        if not plottable:
            raise ValueError(
                "no plottable panels in this model; "
                "≥3D / fs / re smooths aren't supported here — try vis()"
            )

        if select is not None:
            if not (0 <= select < len(plottable)):
                raise IndexError(
                    f"select={select} out of range; "
                    f"have {len(plottable)} plottable panel(s)"
                )
            plottable = [plottable[select]]

        wr_all = (
            self.residuals_of("working") if partial_residuals else None
        )

        def draw_panel(ax_, item):
            kind = item[0]
            if kind == "smooth":
                _, block, a, bcol = item
                edf_b = float(self.edf[a:bcol].sum())
                label_inner = block.label.rstrip(")")
                title = f"{label_inner},{round(edf_b, 2):g})"
                if len(block.term) == 1:
                    self._plot_smooth_1d(
                        ax_, block, a, bcol,
                        color=color, band_color=band_color,
                        band_alpha=band_alpha,
                        rug=rug, partial_residuals=partial_residuals,
                        wr_all=wr_all, ylabel=title,
                    )
                else:
                    self._plot_smooth_2d(
                        ax_, block, a, bcol,
                        color=color, n_grid=n_grid, too_far=too_far,
                        title=title,
                    )
            else:  # "param"
                _, term_label, col_idx, term_kind = item
                if term_kind == "factor":
                    self._plot_parametric_factor(
                        ax_, term_label, col_idx,
                        color=color, rug=rug,
                    )
                else:
                    self._plot_parametric_numeric(
                        ax_, term_label, col_idx,
                        color=color, band_color=band_color,
                        band_alpha=band_alpha, rug=rug,
                    )

        # Single-panel target: draw into the user-supplied ax and return it.
        if ax is not None:
            if len(plottable) != 1:
                raise ValueError(
                    f"ax= requires exactly one panel; have {len(plottable)} "
                    f"plottable panel(s). Pass select= to pick one."
                )
            draw_panel(ax, plottable[0])
            return ax

        n_plots = len(plottable)
        if n_plots == 1:
            fig, ax_one = plt.subplots(figsize=figsize or (5, 4))
            axes_arr = np.array([[ax_one]])
            n_rows, n_cols_eff = 1, 1
        else:
            n_cols_eff = min(n_cols, n_plots)
            n_rows = (n_plots + n_cols_eff - 1) // n_cols_eff
            if figsize is None:
                figsize = (5 * n_cols_eff, 4 * n_rows)
            fig, axes = plt.subplots(n_rows, n_cols_eff, figsize=figsize)
            axes_arr = np.atleast_2d(axes)
            if n_rows == 1 and axes_arr.shape[0] != 1:
                axes_arr = axes_arr.reshape(1, -1)

        for plot_i, item in enumerate(plottable):
            r, c = divmod(plot_i, n_cols_eff)
            draw_panel(axes_arr[r, c], item)

        # Hide unused grid cells.
        for plot_i in range(n_plots, axes_arr.size):
            r, c = divmod(plot_i, n_cols_eff)
            axes_arr[r, c].set_visible(False)

        fig.tight_layout()
        return fig

    def _plot_smooth_1d(
        self, ax, block, a, bcol, *,
        color, band_color, band_alpha, rug, partial_residuals,
        wr_all, ylabel,
    ):
        """1D smooth panel: curve + 2·SE band + optional rug / partial residuals."""
        cov_name = block.term[0]
        x = self.data[cov_name].to_numpy().astype(float).flatten()
        B = block.X
        beta = self._beta[a:bcol]
        Vp = self.Vp[a:bcol, a:bcol]
        fhat = B @ beta
        # Var(f̂_i) = B_i · Vp · B_iᵀ; rowwise.
        var_f = ((B @ Vp) * B).sum(axis=1)
        se_f = np.sqrt(np.clip(var_f, 0.0, None))

        # Factor-by basis is zero outside the level: filter to where
        # the smooth is actually evaluated, otherwise we get a flat-0
        # line through the masked rows.
        active = np.any(np.abs(B) > 0, axis=1)
        xa = x[active]
        fa = fhat[active]
        sa = se_f[active]

        order = np.argsort(xa)
        xs, fs, ses = xa[order], fa[order], sa[order]

        ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
        ax.fill_between(
            xs, fs - 2 * ses, fs + 2 * ses,
            color=band_color, alpha=band_alpha, linewidth=0,
        )
        ax.plot(xs, fs, color=color, linewidth=1.0)

        if partial_residuals:
            pr = wr_all[active] + fa
            ax.scatter(
                xa, pr, facecolor="none", edgecolor="grey",
                s=10, alpha=0.5,
            )

        if rug:
            ymin = ax.get_ylim()[0]
            ax.plot(
                xa, np.full_like(xa, ymin), "|",
                color="black", markersize=6, alpha=0.6,
            )

        ax.set_xlabel(cov_name)
        ax.set_ylabel(ylabel)

    def _plot_smooth_2d(
        self, ax, block, a, bcol, *,
        color, n_grid, too_far, title,
    ):
        """2D smooth panel: three-contour view (estimate / +SE / −SE) plus
        data-location scatter. Mirrors mgcv's ``plot.gam`` for ``s(x,y)`` /
        ``te(x,y)`` smooths: bold = f̂, **dashed = f̂−SE**, **dotted = f̂+SE**
        (matches the lty=1/2/3 assignments in mgcv's ``sp.contour``; note
        Wood 2017 Fig. 4.14's caption swaps these — the code is the truth).
        Levels are shared across the three layers (so the same contour
        value lines up bold/dashed/dotted, ±SE apart) and labeled with
        their numeric value (mgcv default).
        """
        from matplotlib.ticker import MaxNLocator

        x_name, y_name = block.term
        x_data = self.data[x_name].to_numpy().astype(float)
        y_data = self.data[y_name].to_numpy().astype(float)

        x_grid = np.linspace(np.nanmin(x_data), np.nanmax(x_data), n_grid)
        y_grid = np.linspace(np.nanmin(y_data), np.nanmax(y_data), n_grid)
        XX, YY = np.meshgrid(x_grid, y_grid)
        grid_df = pl.DataFrame({
            x_name: XX.flatten(),
            y_name: YY.flatten(),
        })

        # Smooth-only basis at the grid; β and Vp slices restricted to the
        # block so the contours show f̂(x,y), not the full η.
        B = np.asarray(block.spec.predict_mat(grid_df), dtype=float)
        beta = self._beta[a:bcol]
        Vp = self.Vp[a:bcol, a:bcol]
        fit = (B @ beta).reshape(XX.shape)
        var_f = np.einsum("ij,jk,ik->i", B, Vp, B).reshape(XX.shape)
        se_f = np.sqrt(np.maximum(var_f, 0.0))

        if too_far > 0.0:
            mask = _too_far_mask(
                XX.flatten(), YY.flatten(),
                self.data[x_name], self.data[y_name], too_far,
            ).reshape(XX.shape)
            fit = np.where(mask, np.nan, fit)
            se_f = np.where(mask, np.nan, se_f)

        # Pick mgcv-style "pretty" round levels covering the union of
        # f̂, f̂+SE and f̂−SE so the same level value renders bold/dashed/
        # dotted across the three layers (mgcv plot.gam convention).
        zmin = float(np.nanmin(fit - se_f))
        zmax = float(np.nanmax(fit + se_f))
        # nbins=15 lets the locator choose a 0.2-spaced step over a [-1, 1]
        # range (matches mgcv's plot.gam default density).
        levels = MaxNLocator(nbins=15, steps=[1, 2, 5, 10]).tick_values(zmin, zmax)

        # ``linestyles="solid"`` overrides matplotlib's default of switching
        # negative-valued contours to dashed (rcParams["contour.negative_
        # linestyle"]) — R's contour() doesn't do that, so the bold lines
        # would otherwise visually mix with the f̂−SE dashed layer.
        cs_fit = ax.contour(XX, YY, fit,        levels=levels,
                            colors=color, linestyles="solid", linewidths=1.4)
        ax.contour(XX, YY, fit - se_f,          levels=levels,
                   colors=color, linestyles="--", linewidths=0.6)  # lty=2
        ax.contour(XX, YY, fit + se_f,          levels=levels,
                   colors=color, linestyles=":",  linewidths=0.6)  # lty=3
        ax.clabel(cs_fit, inline=True, fontsize=8, fmt="%g")
        ax.scatter(x_data, y_data, s=10, color=color)
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.set_title(title)

    def _plot_parametric_factor(
        self, ax, label: str, col_idx: list[int], *, color, rug: bool,
    ):
        """Termplot for a factor parametric term — Wood 2017 Fig. 4.15
        right panel. Reference level pinned at 0 (default treatment
        contrasts); other levels show β̂ as a solid horizontal bar with
        ±SE dashed bars. Optional rug along the bottom (one tick per
        observation, aggregated by level).
        """
        series = self.data[label]
        if isinstance(series.dtype, pl.Enum):
            levels = list(series.dtype.categories)
        elif isinstance(series.dtype, pl.Categorical):
            levels = sorted(series.unique().drop_nulls().to_list())
        else:
            # Utf8 fallback — sort alphabetically (matches R's default).
            levels = sorted(series.unique().drop_nulls().to_list())

        ests = [0.0]
        ses = [0.0]
        cols = self.parametric_columns
        for lvl in levels[1:]:
            col_name = f"{label}{lvl}"
            if col_name in cols:
                i = cols.index(col_name)
                ests.append(float(self._beta[i]))
                ses.append(float(np.sqrt(max(self.Vp[i, i], 0.0))))
            else:
                ests.append(float("nan"))
                ses.append(float("nan"))

        half = 0.35
        for i, (est, s) in enumerate(zip(ests, ses)):
            xL, xR = i - half, i + half
            ax.plot([xL, xR], [est, est], color=color, linewidth=1.2)
            if s > 0:
                ax.plot([xL, xR], [est + s, est + s], color=color,
                        linestyle="--", linewidth=0.7)
                ax.plot([xL, xR], [est - s, est - s], color=color,
                        linestyle="--", linewidth=0.7)

        if rug:
            # Spread rug ticks within each level so the count is visible
            # (mgcv's plot.gam uses ``rug(jitter(x))`` for the same effect;
            # we lay them out deterministically across [i±half_rug] instead
            # of jittering randomly).
            pos = {lv: i for i, lv in enumerate(levels)}
            obs_levels = self.data[label].drop_nulls().to_list()
            counts: dict = {}
            for v in obs_levels:
                if v in pos:
                    counts[v] = counts.get(v, 0) + 1
            half_rug = 0.2
            xs_list: list[float] = []
            for lvl in levels:
                n_obs = counts.get(lvl, 0)
                if n_obs == 0:
                    continue
                i = pos[lvl]
                if n_obs == 1:
                    xs_list.append(float(i))
                else:
                    xs_list.extend(
                        np.linspace(i - half_rug, i + half_rug, n_obs).tolist()
                    )
            if xs_list:
                xs = np.asarray(xs_list)
                ymin = ax.get_ylim()[0]
                ax.plot(xs, np.full_like(xs, ymin), "|",
                        color="black", markersize=6, alpha=0.6)

        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([str(l) for l in levels])
        ax.set_xlabel(label)
        ax.set_ylabel(f"Partial for {label}")

    def _plot_parametric_numeric(
        self, ax, label: str, col_idx: list[int], *,
        color, band_color, band_alpha, rug: bool,
    ):
        """Linear partial effect for a numeric parametric term — ``β̂·x``
        with a 2·SE band (mgcv's termplot for non-factor terms).
        """
        i = col_idx[0]
        beta_x = float(self._beta[i])
        se_x = float(np.sqrt(max(self.Vp[i, i], 0.0)))
        x = self.data[label].drop_nulls().to_numpy().astype(float)
        x_grid = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        fhat = beta_x * x_grid
        se_fhat = se_x * np.abs(x_grid)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
        ax.fill_between(
            x_grid, fhat - 2 * se_fhat, fhat + 2 * se_fhat,
            color=band_color, alpha=band_alpha, linewidth=0,
        )
        ax.plot(x_grid, fhat, color=color, linewidth=1.0)
        if rug:
            ymin = ax.get_ylim()[0]
            ax.plot(x, np.full_like(x, ymin), "|",
                    color="black", markersize=6, alpha=0.6)
        ax.set_xlabel(label)
        ax.set_ylabel(f"Partial for {label}")

    def plot(self, figsize=None, smooth=True, label_n=3):
        """4-panel diagnostic, matching the graphical part of gam.check.

        Per-smooth effect curves (mgcv's plot.gam) and the 2D fitted-surface
        viewer (vis.gam) live in separate methods and are not part of this
        diagnostic panel.
        """
        if figsize is None:
            figsize = (10, 8)
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        self.plot_residuals(ax=axes[0, 0], smooth=smooth, label_n=label_n)
        self.plot_qq(ax=axes[0, 1], label_n=label_n)
        self.plot_scale_location(ax=axes[1, 0], smooth=smooth, label_n=label_n)
        self.plot_leverage(ax=axes[1, 1], smooth=smooth, label_n=label_n)
        fig.tight_layout()
        return fig


# --------------------------------------------------------------------------
# module-private helpers
# --------------------------------------------------------------------------


def _row_frame(values: np.ndarray, columns: list[str]) -> pl.DataFrame:
    flat = np.asarray(values).reshape(-1)
    return pl.DataFrame({c: [float(flat[i])] for i, c in enumerate(columns)})


def _add_null_space_penalties(blocks: list[SmoothBlock]) -> list[SmoothBlock]:
    """Mirror mgcv's ``null.space.penalty=TRUE`` (``gam(..., select=TRUE)``).

    For each block, append a rank-``(p − rank_S)`` matrix that penalizes the
    null-space directions of the existing combined penalty ``Σⱼ Sⱼ`` to the
    block's ``S`` list. With this extra penalty plus its own smoothing
    parameter, the term can be shrunk to zero — that's the whole point of
    ``select=TRUE``. After augmentation the per-block combined penalty is
    full-rank, so the smooth's null-space dim is zero and ``_Mp`` collapses
    to ``p_param``.

    Implements the ``need.full`` eigendecomposition branch of mgcv's
    ``smoothCon`` (R/smooth.r): ``St = Σⱼ Sⱼ``, eigendecompose, take the
    eigenvectors ``U`` with eigenvalues below ``max_eig · ε^0.66``, and use
    ``Sf = U Uᵀ`` (the projection onto the null space). Mgcv's fast path
    for ``nsm=1`` plus a diagonal-canonical ``S`` produces the same ``Sf``
    when applicable; this routine takes the eigen path unconditionally,
    which is bit-equal up to LAPACK's choice of basis for repeated
    eigenvalues — and ``U Uᵀ`` is invariant to that choice.

    No rescaling: mgcv assigns ``S.scale = 1`` to ``Sf`` (left at unit
    norm), in contrast to the per-S ``maXX/‖S‖`` rescaling that
    ``_scale_penalty`` applied to the original penalties.
    """
    eps = float(np.finfo(float).eps)
    threshold_factor = eps ** 0.66
    out: list[SmoothBlock] = []
    for b in blocks:
        S_list = [np.asarray(s, dtype=float) for s in b.S]
        if not S_list:
            out.append(b)
            continue
        St = S_list[0].copy()
        for Sj in S_list[1:]:
            St += Sj
        eigvals, eigvecs = np.linalg.eigh(St)
        max_eig = float(eigvals.max()) if eigvals.size else 0.0
        if max_eig <= 0.0:
            out.append(b)
            continue
        null_mask = eigvals < max_eig * threshold_factor
        if not bool(np.any(null_mask)):
            out.append(b)
            continue
        U = eigvecs[:, null_mask]
        Sf = U @ U.T
        Sf = 0.5 * (Sf + Sf.T)
        out.append(SmoothBlock(
            label=b.label, term=b.term, cls=b.cls,
            X=b.X, S=S_list + [Sf], spec=b.spec,
        ))
    return out


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
        if b.spec is None:
            new_spec = None
        else:
            # Compose with any prior keep_cols so re-running gam.side is idempotent.
            keep_arr = np.asarray(keep, dtype=np.intp)
            prior = b.spec.keep_cols
            new_keep = keep_arr if prior is None else prior[keep_arr]
            new_spec = BasisSpec(
                raw=b.spec.raw, by=b.spec.by, absorb=b.spec.absorb,
                keep_cols=new_keep,
            )
        out.append(SmoothBlock(
            label=b.label, term=b.term, cls=b.cls, X=new_X, S=new_S,
            spec=new_spec,
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


# ---------------------------------------------------------------------------
# vis.gam helpers + result
# ---------------------------------------------------------------------------


def _is_factor_like_col(col: pl.Series) -> bool:
    return col.dtype in (pl.Categorical, pl.Enum, pl.String, pl.Utf8, pl.Object)


def _has_variation(col: pl.Series) -> bool:
    vals = col.drop_nulls()
    if len(vals) <= 1:
        return False
    return vals.n_unique() > 1


def _grid_axis(col: pl.Series, n_grid: int) -> np.ndarray:
    """Build a 1D grid for one ``view`` axis.

    Numeric: ``linspace(min, max, n_grid)``. Factor: the levels (truncated
    to ``n_grid`` if there are more, or each level repeated to fill the
    grid otherwise — same shape as mgcv's ``fac.seq``)."""
    if _is_factor_like_col(col):
        from .formula import _factor_levels  # local import to avoid cycle

        levels = list(_factor_levels(col))
        fn = len(levels)
        if fn >= n_grid:
            return np.array(levels[:n_grid], dtype=object)
        # Repeat each level ⌊n_grid/fn⌋ times then pad the tail with the
        # last level — mirrors mgcv's fac.seq.
        ln = n_grid // fn
        out = np.array([lev for lev in levels for _ in range(ln)] +
                       [levels[-1]] * (n_grid - ln * fn), dtype=object)
        return out
    arr = col.drop_nulls().to_numpy().astype(float)
    return np.linspace(float(arr.min()), float(arr.max()), n_grid)


def _too_far_mask(
    g1: np.ndarray, g2: np.ndarray,
    d1: pl.Series, d2: pl.Series,
    dist: float,
) -> np.ndarray:
    """Port of mgcv's ``exclude.too.far``.

    Normalize grid + data to the grid's [0, 1] box, compute each grid
    point's nearest-data-point distance, return a boolean mask of grid
    points farther than ``dist``. Factor view axes are not supported by
    mgcv's distance metric — we return all-False for those.
    """
    if _is_factor_like_col(d1) or _is_factor_like_col(d2):
        return np.zeros(g1.shape[0], dtype=bool)

    g1 = np.asarray(g1, dtype=float)
    g2 = np.asarray(g2, dtype=float)
    d1 = d1.drop_nulls().to_numpy().astype(float)
    d2 = d2.drop_nulls().to_numpy().astype(float)
    # mgcv normalizes by the grid's range, then both grid + data live in
    # the grid's [0, 1] box (data outside [0, 1] is preserved as-is).
    g1_min, g1_max = g1.min(), g1.max()
    g2_min, g2_max = g2.min(), g2.max()
    g1_span = g1_max - g1_min if g1_max > g1_min else 1.0
    g2_span = g2_max - g2_min if g2_max > g2_min else 1.0
    g1n = (g1 - g1_min) / g1_span
    g2n = (g2 - g2_min) / g2_span
    d1n = (d1 - g1_min) / g1_span
    d2n = (d2 - g2_min) / g2_span
    # Pairwise squared distance — fine for n_grid² ≈ 900 × n data.
    dx = g1n[:, None] - d1n[None, :]
    dy = g2n[:, None] - d2n[None, :]
    min_dist = np.sqrt((dx * dx + dy * dy).min(axis=1))
    return min_dist > dist


class VisResult:
    """Output of :meth:`gam.vis`.

    Attributes
    ----------
    view : (str, str)
        The two covariate names the surface is over.
    m1, m2 : 1D ndarray
        Axis values, length ``n_grid`` each (numeric: linspace; factor: levels).
    fit : (n_grid, n_grid) ndarray
        ``fit[i, j]`` is the prediction at ``(m1[i], m2[j])``. ``NaN`` where
        ``too_far`` masked the grid.
    se : (n_grid, n_grid) ndarray, optional
        Pointwise SE if ``vis(se=True)``; otherwise ``None``.
    type : "link" | "response"
        Scale of fit and se.
    """

    __slots__ = ("view", "m1", "m2", "fit", "se", "type")

    def __init__(self, *, view, m1, m2, fit, se, type):
        self.view = view
        self.m1 = m1
        self.m2 = m2
        self.fit = fit
        self.se = se
        self.type = type

    def __repr__(self) -> str:
        z = self.fit
        return (
            f"VisResult(view={self.view}, n_grid=({len(self.m1)},{len(self.m2)}), "
            f"type={self.type!r}, "
            f"fit range=[{np.nanmin(z):.4g}, {np.nanmax(z):.4g}], "
            f"se={'yes' if self.se is not None else 'no'})"
        )

    def plot(
        self,
        kind: str = "contour",
        ax=None,
        figsize: tuple | None = None,
        cmap: str = "viridis",
        levels: int = 20,
        contour_levels=None,
        se_mult: float = 0.0,
        elev: float = 30.0,
        azim: float = -60.0,
        zlabel: str | None = None,
        aspect: str | float | None = "equal",
        colorbar: bool = True,
        clabel: bool = False,
        clabel_kwargs: dict | None = None,
    ):
        """Render the surface.

        ``kind="contour"`` draws a filled contour with overlaid lines;
        ``kind="persp"`` draws a 3D wireframe (mgcv's default). When
        ``se_mult > 0`` and ``se`` is present, persp also draws ±``se_mult``·SE
        envelopes (same convention as ``vis.gam(se=...)``).

        ``aspect`` (contour only): ``"equal"`` (default — one data-unit on
        x takes the same screen length as one on y, so ticks are visually
        the same size), ``"square"`` (square plotting box regardless of
        data ranges), a float (height/width ratio), or ``None`` (matplotlib
        default).
        """
        if kind not in ("contour", "persp"):
            raise ValueError(f"kind must be 'contour' or 'persp'; got {kind!r}")

        x_lab, y_lab = self.view
        z_lab = zlabel or (
            "linear predictor" if self.type == "link" else "response"
        )

        # Numeric coords for plotting — factor axes get plotted at their
        # ordinal positions with the level names as ticks.
        m1_num, m1_ticks = _axis_for_plot(self.m1)
        m2_num, m2_ticks = _axis_for_plot(self.m2)

        if kind == "contour":
            if ax is None:
                _fig, ax = plt.subplots(figsize=figsize or (6, 5))
            # M1 (rows, axis 0) → x; M2 (cols, axis 1) → y; transpose so that
            # contourf's (x, y, Z) call has Z[j, i] for x=m1[i], y=m2[j].
            X, Y = np.meshgrid(m1_num, m2_num, indexing="xy")
            Z = self.fit.T
            cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)
            # Bump line strength when labels are on so the numbers sit on
            # readable lines, not faint guides.
            line_alpha = 0.8 if clabel else 0.5
            line_width = 0.6 if clabel else 0.4
            line_levels = contour_levels if contour_levels is not None else levels
            cs = ax.contour(X, Y, Z, levels=line_levels, colors="black",
                            linewidths=line_width, alpha=line_alpha)
            if clabel:
                kw = dict(inline=True, fontsize=8, fmt="%.2f")
                if clabel_kwargs:
                    kw.update(clabel_kwargs)
                ax.clabel(cs, **kw)
            if colorbar:
                plt.colorbar(cf, ax=ax, label=z_lab)
            ax.set_xlabel(x_lab)
            ax.set_ylabel(y_lab)
            if m1_ticks is not None:
                ax.set_xticks(m1_num)
                ax.set_xticklabels(m1_ticks, rotation=45, ha="right")
            if m2_ticks is not None:
                ax.set_yticks(m2_num)
                ax.set_yticklabels(m2_ticks)
            if aspect == "square":
                ax.set_box_aspect(1)
            elif aspect == "equal":
                ax.set_aspect("equal")
            elif isinstance(aspect, (int, float)):
                ax.set_box_aspect(float(aspect))
            return ax

        # persp: 3D wireframe
        if ax is None:
            fig = plt.figure(figsize=figsize or (7, 6))
            ax = fig.add_subplot(111, projection="3d")
        X, Y = np.meshgrid(m1_num, m2_num, indexing="ij")
        Z = self.fit
        ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.85,
                        linewidth=0.3, edgecolor="black")
        if se_mult > 0 and self.se is not None:
            ax.plot_wireframe(X, Y, Z + se_mult * self.se,
                              color="red", linewidth=0.3, alpha=0.5)
            ax.plot_wireframe(X, Y, Z - se_mult * self.se,
                              color="green", linewidth=0.3, alpha=0.5)
        ax.set_xlabel(x_lab)
        ax.set_ylabel(y_lab)
        ax.set_zlabel(z_lab)
        ax.view_init(elev=elev, azim=azim)
        if m1_ticks is not None:
            ax.set_xticks(m1_num)
            ax.set_xticklabels(m1_ticks)
        if m2_ticks is not None:
            ax.set_yticks(m2_num)
            ax.set_yticklabels(m2_ticks)
        return ax


def _axis_for_plot(m: np.ndarray):
    """Return (numeric_positions, tick_labels_or_None) for a vis axis.

    Factor axes get integer positions and string tick labels; numeric axes
    return themselves and ``None``."""
    if m.dtype.kind in ("U", "S", "O"):
        return np.arange(len(m), dtype=float), [str(v) for v in m]
    return np.asarray(m, dtype=float), None
