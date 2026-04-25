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
from scipy.optimize import minimize
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
    fitted, residuals : np.ndarray
        Length-n arrays, ``ŷ = Xβ̂`` and ``y − ŷ``.
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
        method: str = "REML",
        sp: np.ndarray | None = None,
        family: Family | None = None,
    ):
        if method not in ("REML", "GCV.Cp"):
            raise ValueError(f"method must be 'REML' or 'GCV.Cp', got {method!r}")

        self.formula = formula
        self.method = method
        self.family = Gaussian() if family is None else family
        # GCV is only meaningful when scale is known (Poisson, binomial). For
        # unknown-scale non-Gaussian families (Gamma, gaussian-non-identity,
        # etc.), scale enters the criterion non-analytically and mgcv requires
        # REML/ML — we follow.
        if (method == "GCV.Cp"
                and not (self.family.name == "gaussian"
                         and self.family.link.name == "identity")):
            raise NotImplementedError(
                "GCV.Cp is currently only supported for Gaussian-identity; "
                "use method='REML' for non-Gaussian families."
            )
        # `_is_strictly_additive` flags the Gaussian-identity fast path: a
        # single closed-form solve gives β̂(λ); PIRLS would converge in one
        # iteration anyway. We keep the closed-form path so the existing
        # Gaussian regression suite stays bit-identical.
        self._is_strictly_additive = (
            self.family.name == "gaussian"
            and self.family.link.name == "identity"
        )

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
            obj = self._reml if method == "REML" else self._gcv
            # Coarse coordinate-wise scan to seed Newton. Two passes:
            # first pick the best uniform ρ across all dimensions, then
            # walk one coordinate at a time fixing the others at the
            # current best. Cheap (~ n_sp × |grid| extra evals) and fixes
            # two failure modes that hit on real data:
            #   - GCV's narrow valley (edf enters denom quadratically) —
            #     line search from ρ=0 routinely overshoots it straight to
            #     the lower bound.
            #   - REML with tensor / overlap smooths where the optimum is
            #     non-uniform (one λ tiny, another huge), so a uniform
            #     starting point picks the wrong basin.
            grid = np.array([-12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0])
            best_val, best_rho0 = np.inf, np.zeros(n_sp)
            for g in grid:
                rho_try = np.full(n_sp, g)
                val = obj(rho_try)
                if np.isfinite(val) and val < best_val:
                    best_val, best_rho0 = val, rho_try.copy()
            # one coordinate-wise refinement pass
            cur_rho = best_rho0.copy()
            cur_val = best_val
            for j in range(n_sp):
                for g in grid:
                    rho_try = cur_rho.copy()
                    rho_try[j] = g
                    val = obj(rho_try)
                    if np.isfinite(val) and val < cur_val:
                        cur_val, cur_rho = val, rho_try
            if method == "REML":
                rho_hat = self._outer_newton(cur_rho)
            else:
                # GCV: keep L-BFGS-B (no analytical Hessian for GCV yet).
                bounds = [(-30.0, 30.0)] * n_sp
                res = minimize(
                    obj, cur_rho, method="L-BFGS-B", bounds=bounds,
                    options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 500},
                )
                rho_hat = res.x
                self._optim = res
            self.sp = np.exp(rho_hat)
            fit = self._fit_given_rho(rho_hat)

        # Unpack fit results.
        beta = fit.beta
        rss = fit.rss
        pen = fit.pen
        A_chol = fit.A_chol
        A_chol_lower = fit.A_chol_lower
        Sλ = fit.S_full
        log_det_A = fit.log_det_A

        self._rho_hat = rho_hat

        # Posterior β covariance Vp = σ²·A⁻¹. We get A⁻¹ once via
        # cho_solve(I) rather than via diag-tricks, since we need the full
        # matrix for Ve, per-coef SEs, and predict().
        A_inv = cho_solve((A_chol, A_chol_lower), np.eye(p))
        # Ve = σ² A⁻¹ XᵀX A⁻¹ (frequentist); edf = diag(A⁻¹ XᵀX) (coefficient-
        # space influence). The two expressions share A⁻¹ XᵀX so compute once.
        A_inv_XtX = A_inv @ XtX
        # Per-coefficient edf = diag(F) where F = A⁻¹ XᵀX. F is not
        # symmetric, so individual diag entries can be negative — mgcv
        # reports them verbatim (matches m$edf), and the per-smooth sum
        # remains non-negative and interpretable.
        edf = np.diag(A_inv_XtX).copy()
        edf_total = float(edf.sum())
        # df.residual used in mgcv = n - edf_total. For Gaussian:
        #   σ̂² = rss / (n - edf_total)
        # (mgcv also uses this to rescale GCV; REML uses (n - Mp)). We store
        # both for summary output.
        df_resid = float(n - edf_total)
        sigma_squared = rss / df_resid if df_resid > 0 else float("nan")
        sigma = float(np.sqrt(sigma_squared)) if np.isfinite(sigma_squared) else float("nan")

        Vp = sigma_squared * A_inv
        Ve = sigma_squared * A_inv_XtX @ A_inv

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

        fitted = X @ beta
        residuals = y - fitted
        self.fitted = fitted
        self.residuals = residuals
        self.sigma = sigma
        self.sigma_squared = sigma_squared
        self.df_residuals = df_resid
        self.rss = float(rss)
        self.deviance = float(rss)

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

        if has_intercept and tss > 0:
            r_squared = 1.0 - rss / tss
            # mgcv's adjusted R² uses df.residual and n - 1 like lm:
            # 1 - (1 - R²) (n - 1) / df.residual
            r_squared_adjusted = (
                1.0 - (1.0 - r_squared) * (n - 1) / df_resid
                if df_resid > 0 else float("nan")
            )
        else:
            r_squared = 1.0 - rss / yty if yty > 0 else float("nan")
            r_squared_adjusted = (
                1.0 - (1.0 - r_squared) * n / df_resid
                if df_resid > 0 else float("nan")
            )
        self.r_squared = float(r_squared)
        self.r_squared_adjusted = float(r_squared_adjusted)

        # Log-likelihood at the Gaussian MLE σ² = rss/n (concentrated form),
        # matching mgcv's logLik.gam — `$sig2` is the unbiased rss/(n-edf)
        # and is reported separately, but logLik/AIC profile σ² out at the
        # MLE, so plugging the unbiased σ² in here would double-count the
        # df penalty. (lm.compute_loglikelihood uses the same formula.)
        if rss > 0:
            loglike = -0.5 * n * (np.log(rss / n) + np.log(2 * np.pi) + 1.0)
        else:
            loglike = float("nan")
        self.loglike = float(loglike)
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
            log_sigma2_hat = float(np.log(sigma_squared))
            H_aug = self._reml_hessian_augmented(rho_hat, log_sigma2_hat)
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
        # only. +1 is for the residual scale (sc.p in mgcv).
        if n_sp > 0:
            edf2_per_coef, edf1_per_coef = self._compute_edf12(
                rho_hat, fit, sigma_squared, A_inv, A_inv_XtX, edf, H_aug,
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
        # logLik.gam additionally caps df at length(coef) — relevant only at
        # the totally-unpenalized end where edf2 ≈ p; keep it for parity.
        df_for_aic = min(self.edf2_total, float(p))
        self.npar = df_for_aic + 1.0
        self.AIC = -2.0 * loglike + 2.0 * self.npar
        self.BIC = -2.0 * loglike + float(np.log(n)) * self.npar

        if method == "REML":
            if n_sp > 0:
                self.REML_criterion = float(self._reml(rho_hat))
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

    def _fit_given_rho(self, rho: np.ndarray) -> "_FitState":
        """Dispatch to the Gaussian-identity closed form or PIRLS depending
        on the family. The Gaussian-identity path is preserved verbatim so
        the existing test suite stays bit-identical."""
        if self._is_strictly_additive:
            return self._fit_given_rho_gaussian(rho)
        return self._fit_given_rho_pirls(rho)

    def _fit_given_rho_gaussian(self, rho: np.ndarray) -> "_FitState":
        """Solve β̂(λ) = (XᵀX + Sλ)⁻¹ Xᵀy by Cholesky at one ρ.

        Also returns the Cholesky factor and log|A| for downstream use
        (REML uses log|A|; post-fit uses the factor for A⁻¹). Adds a tiny
        ridge if A is singular — rare, only at pathological ρ extrema.
        """
        Sλ = self._build_S_lambda(rho)
        A = self._XtX + Sλ
        # Symmetrize defensively against FP drift from the outer products.
        A = 0.5 * (A + A.T)
        try:
            A_chol, lower = cho_factor(A, lower=True, overwrite_a=False)
        except np.linalg.LinAlgError:
            ridge = 1e-8 * np.trace(A) / self.p
            A_chol, lower = cho_factor(
                A + ridge * np.eye(self.p), lower=True, overwrite_a=False,
            )
        beta = cho_solve((A_chol, lower), self._Xty)
        rss = self._yty - 2.0 * float(self._Xty @ beta) + float(beta @ self._XtX @ beta)
        pen = float(beta @ Sλ @ beta)
        # log|A| = 2 Σ log diag(L)
        log_det_A = 2.0 * float(np.log(np.abs(np.diag(A_chol))).sum())
        # Pad GLM-state fields with the Gaussian-identity values so the
        # downstream generic code works either way.
        eta = self._X_full @ beta
        mu = eta.copy()
        w = np.ones(self.n)
        z = self._y_arr.copy()
        alpha = np.ones(self.n)
        return _FitState(
            beta=beta, dev=max(rss, 0.0), pen=pen,
            A_chol=A_chol, A_chol_lower=lower,
            S_full=Sλ, log_det_A=log_det_A,
            eta=eta, mu=mu, w=w, z=z, alpha=alpha,
        )

    def _fit_given_rho_pirls(self, rho: np.ndarray) -> "_FitState":
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
            d2g = link.d2link(mu)
            alpha = 1.0 + (y - mu) * (family.dvar(mu) / V + d2g * mu_eta_v)
            # mgcv: clamp α=0 to ε to avoid division by zero in z-formula.
            alpha = np.where(alpha == 0.0, np.finfo(float).eps, alpha)

            z = eta + (y - mu) / (mu_eta_v * alpha)
            w = alpha * mu_eta_v ** 2 / V

            # Some non-Fisher Newton steps can produce w<0; mgcv's recovery
            # is to fall back to Fisher (α=1, w=mu_eta²/V). Trees+Gamma+log
            # has α=y/μ>0 always, but the fallback is cheap insurance.
            if np.any(w < 0):
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
        mu_eta_v = link.mu_eta(eta)
        V = family.variance(mu)
        d2g = link.d2link(mu)
        alpha = 1.0 + (y - mu) * (family.dvar(mu) / V + d2g * mu_eta_v)
        alpha = np.where(alpha == 0.0, np.finfo(float).eps, alpha)
        z = eta + (y - mu) / (mu_eta_v * alpha)
        w = alpha * mu_eta_v ** 2 / V
        if np.any(w < 0):
            alpha = np.ones(n)
            z = eta + (y - mu) / mu_eta_v
            w = mu_eta_v ** 2 / V

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

    def _reml(self, rho: np.ndarray) -> float:
        """Laplace-approximate REML criterion (-2·V_R). Wood 2017 §6.6."""
        fit = self._fit_given_rho(rho)
        penalized_ss = fit.rss + fit.pen
        if penalized_ss <= 0:
            return 1e15
        n_minus_Mp = self.n - self._Mp
        if n_minus_Mp <= 0:
            return 1e15
        log_det_S = self._log_det_S_pos(rho)
        # Constants dropped since ρ is the only argument; what remains is
        #   (n - Mp) log(rss + pen) + log|A| - log|Sλ|_+
        # plus a σ² contribution from profiling: (n - Mp) log((rss+pen)/
        # (n - Mp)) which is the same up to an additive constant. Adding
        # (n - Mp) log(2π) + (n - Mp) keeps the scale comparable with R's
        # `-2·gam_fit$gcv.ubre`.
        return (
            n_minus_Mp * np.log(penalized_ss / n_minus_Mp)
            + fit.log_det_A
            - log_det_S
            + n_minus_Mp * (1.0 + float(np.log(2 * np.pi)))
        )

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

    def _reml_grad(self, rho: np.ndarray,
                   fit: "_FitState | None" = None) -> np.ndarray:
        """Analytical gradient ∂(2·V_R^prof)/∂ρ — matches ∂(_reml)/∂ρ.

        For Gaussian + identity REML, profiling σ² out gives

            2·V_R^prof = (n-Mp)·log(rss+pen) + log|A| − log|S|+ + const

        Term-by-term derivatives (Wood 2011 §4):

            ∂(rss+pen)/∂ρ_k = λ_k · β̂' S_k β̂          (pen contribution g_k)
            ∂log|A|/∂ρ_k    = λ_k · tr(A⁻¹ S_k)
            ∂log|S|+/∂ρ_k   = λ_k · tr(S⁺ S_k)         (rank-stable case)

        Combining:

            ∂(2·V_R)/∂ρ_k = (n−Mp)·g_k/(rss+pen)
                          + λ_k·tr(A⁻¹ S_k) − λ_k·tr(S⁺ S_k)

        Replaces FD jacobian for the ρ-optimization (and feeds into the
        analytical Hessian later).
        """
        if fit is None:
            fit = self._fit_given_rho(rho)
        n_sp = len(self._slots)
        if n_sp == 0:
            return np.zeros(0)
        n_minus_Mp = self.n - self._Mp
        rs = fit.rss + fit.pen
        if rs <= 0 or n_minus_Mp <= 0:
            return np.zeros(n_sp)

        A_inv = cho_solve((fit.A_chol, fit.A_chol_lower), np.eye(self.p))
        S_pinv = self._S_pinv(fit.S_full)

        grad = np.zeros(n_sp)
        for k, slot in enumerate(self._slots):
            a, b = slot.col_start, slot.col_end
            sp_k = float(np.exp(rho[k]))
            beta_k = fit.beta[a:b]
            g_k = sp_k * float(beta_k @ slot.S @ beta_k)
            # S_k has support only on the slot's column range, so
            # tr(A⁻¹ S_k) reduces to a contraction with the (a:b, a:b)
            # block of A⁻¹. Same for S⁺.
            tr_AinvSk = float(np.einsum("ij,ji->", A_inv[a:b, a:b], slot.S))
            tr_SpinvSk = float(np.einsum("ij,ji->", S_pinv[a:b, a:b], slot.S))
            grad[k] = (
                n_minus_Mp * g_k / rs
                + sp_k * tr_AinvSk
                - sp_k * tr_SpinvSk
            )
        return grad

    def _reml_hessian(self, rho: np.ndarray,
                      fit: "_FitState | None" = None) -> np.ndarray:
        """Analytical Hessian ∂²V_R/∂ρ_i ∂ρ_j — Wood 2011 §4.

        Differentiating ``2·V_R^prof = (n−Mp)·log(rs) + log|A| − log|S|+
        + const`` (with rs = rss + pen) twice in ρ. Pieces:

            ∂rs/∂ρ_k       = g_k = λ_k β̂' S_k β̂
            ∂²rs/∂ρ_i∂ρ_j  = δ_ij g_i − 2 λ_i λ_j β̂' S_i A⁻¹ S_j β̂
            ∂log|A|/∂ρ_k   = λ_k tr(A⁻¹ S_k)
            ∂²log|A|/∂ρ_i∂ρ_j = δ_ij λ_i tr(A⁻¹ S_i)
                              − λ_i λ_j tr(A⁻¹ S_i A⁻¹ S_j)
            ∂log|S|+/∂ρ_k  = λ_k tr(S⁺ S_k)
            ∂²log|S|+/∂ρ_i∂ρ_j = δ_ij λ_i tr(S⁺ S_i)
                                − λ_i λ_j tr(S⁺ S_i S⁺ S_j)

        Combining and using ∂(log f)/∂ρ_k = ∂f/∂ρ_k / f for the rs term,

            H[i,j] of 2·V_R = (n−Mp)/rs · [δ_ij g_i
                              − 2 λ_i λ_j β̂' S_i A⁻¹ S_j β̂
                              − g_i g_j / rs]
                            + δ_ij λ_i tr(A⁻¹ S_i)
                            − λ_i λ_j tr(A⁻¹ S_i A⁻¹ S_j)
                            − δ_ij λ_i tr(S⁺ S_i)
                            + λ_i λ_j tr(S⁺ S_i S⁺ S_j)

        Notice the δ_ij terms are exactly the gradient — the diagonal of
        the Hessian is grad[i] plus the "off-diagonal" terms evaluated at
        i=j. We exploit that to avoid duplicating code.

        Returned in V_R scale (halved) — the convention ``_compute_Vr`` /
        vcomp expect (mgcv's gam.fit3.post.proc inverts V_R Hessian to
        get the asymptotic covariance of ρ̂).
        """
        if fit is None:
            fit = self._fit_given_rho(rho)
        n_sp = len(self._slots)
        if n_sp == 0:
            return np.zeros((0, 0))
        n_minus_Mp = self.n - self._Mp
        rs = fit.rss + fit.pen
        if rs <= 0 or n_minus_Mp <= 0:
            return np.zeros((n_sp, n_sp))

        p = self.p
        sp = np.exp(rho)
        A_inv = cho_solve((fit.A_chol, fit.A_chol_lower), np.eye(p))
        S_pinv = self._S_pinv(fit.S_full)

        # Per-slot precomputation. AinvS_block[k] = (A⁻¹ S_k) restricted
        # to the columns where it's nonzero (slot k's range), shape
        # (p, m_k); SpinvS_block analogous. Sbeta_full[k] = embedded
        # p-vector S_k β̂. AinvSbeta[k] = A⁻¹ (S_k β̂) full p-vector for
        # the β̂' S_i A⁻¹ S_j β̂ term.
        AinvS_block: list[np.ndarray] = []
        SpinvS_block: list[np.ndarray] = []
        Sbeta_full = np.zeros((n_sp, p))
        AinvSbeta = np.empty((n_sp, p))
        g = np.zeros(n_sp)
        tr_AinvS = np.zeros(n_sp)
        tr_SpinvS = np.zeros(n_sp)
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

        # 2·V_R gradient — same as ``_reml_grad``; the diagonal of the
        # 2·V_R Hessian is grad2[i] plus the i=j evaluation of the
        # quadratic-in-(i,j) terms below.
        grad2 = (n_minus_Mp / rs) * g + sp * tr_AinvS - sp * tr_SpinvS

        H2 = np.zeros((n_sp, n_sp))
        for i in range(n_sp):
            a_i, b_i = self._slots[i].col_start, self._slots[i].col_end
            for j in range(i, n_sp):
                a_j, b_j = self._slots[j].col_start, self._slots[j].col_end
                # β̂' S_i A⁻¹ S_j β̂ = (S_i β̂)' (A⁻¹ S_j β̂).
                bSiAinvSj_b = float(Sbeta_full[i] @ AinvSbeta[j])
                # tr(A⁻¹ S_i A⁻¹ S_j) via the embedded-block trick: rows
                # of AinvS_block[i] at j's column range × rows of
                # AinvS_block[j] at i's column range, contracted on both
                # axes (the standard tr(MN) = einsum('ij,ji->',M,N) but
                # restricted to the nonzero stripes).
                tr_AA = float(np.einsum(
                    "ab,ba->",
                    AinvS_block[i][a_j:b_j, :],
                    AinvS_block[j][a_i:b_i, :],
                ))
                tr_SS = float(np.einsum(
                    "ab,ba->",
                    SpinvS_block[i][a_j:b_j, :],
                    SpinvS_block[j][a_i:b_i, :],
                ))
                cross = (
                    (n_minus_Mp / rs) * (
                        -2.0 * sp[i] * sp[j] * bSiAinvSj_b
                        - g[i] * g[j] / rs
                    )
                    - sp[i] * sp[j] * tr_AA
                    + sp[i] * sp[j] * tr_SS
                )
                if i == j:
                    H2[i, i] = cross + grad2[i]
                else:
                    H2[i, j] = H2[j, i] = cross

        return 0.5 * H2

    def _outer_newton(self, rho0: np.ndarray, max_iter: int = 200,
                      tol_grad: float = 1e-8, tol_step: float = 1e-12,
                      max_step: float = 5.0) -> np.ndarray:
        """mgcv-style damped Newton on V_R wrt ρ — gam.outer's `newton()`.

        Per iteration:
          1. Compute analytical g = ∂V_R/∂ρ and H = ∂²V_R/∂ρ∂ρ
             (``_reml_grad`` returns the 2·V_R gradient — halve it; the
             analytical ``_reml_hessian`` already lives in V_R scale).
          2. Eigen-clamp H to PD: replace negative eigenvalues with their
             absolute value, and clamp small/zero eigenvalues to a tiny
             positive floor. This guarantees the Newton direction is a
             descent direction even when H is indefinite (regions of the
             surface that are ridge-shaped between λ-modes).
          3. Step d = -H_pd^{-1} g, capped at max_step in ∞-norm.
          4. Step-halving line search until V_R decreases.
          5. Stop on |g|_∞ < tol_grad or |Δf| / (1 + |f|) < tol_step.

        L-BFGS-B's BFGS approximation drifts off the true Hessian by
        enough that ρ̂ lands ~1e-4 from the analytical stationary point
        — that's the residual gap to mgcv on Machines b2's vcomp. Damped
        Newton on the analytical pieces converges to the same ρ̂ as
        mgcv to FP precision.
        """
        rho = rho0.copy()
        f_prev = self._reml(rho)
        for _ in range(max_iter):
            fit = self._fit_given_rho(rho)
            grad = 0.5 * self._reml_grad(rho, fit)         # V_R gradient
            if np.abs(grad).max() < tol_grad:
                break
            H = self._reml_hessian(rho, fit)               # V_R Hessian
            H = 0.5 * (H + H.T)
            w, V = np.linalg.eigh(H)
            w_max = float(np.abs(w).max()) if w.size > 0 else 1.0
            eps = max(w_max * 1e-7, 1e-12)
            # Negative eigvals → flip sign; small ones → clamp up. This
            # gives a positive-definite quadratic model whose minimizer
            # is a proper Newton step.
            w_pd = np.where(np.abs(w) > eps, np.abs(w), eps)
            d = -V @ ((V.T @ grad) / w_pd)
            d_norm = float(np.abs(d).max())
            if d_norm > max_step:
                d *= max_step / d_norm
            # Backtracking step-halving on V_R = _reml/2 (or just 2·V_R
            # since the comparison only cares about descent).
            alpha = 1.0
            f_new = f_prev
            descent = False
            for _ in range(40):
                rho_new = rho + alpha * d
                f_new = self._reml(rho_new)
                if np.isfinite(f_new) and f_new < f_prev - 1e-14 * abs(f_prev):
                    descent = True
                    break
                alpha *= 0.5
            if not descent:
                break
            rho = rho_new
            df = abs(f_new - f_prev)
            f_prev = f_new
            if df < tol_step * (1.0 + abs(f_prev)):
                break
        return rho

    def _gcv(self, rho: np.ndarray) -> float:
        """Generalized cross-validation score. Wood 2017 §4.4."""
        fit = self._fit_given_rho(rho)
        A_inv = cho_solve((fit.A_chol, fit.A_chol_lower), np.eye(self.p))
        A_inv_XtX = A_inv @ self._XtX
        edf_total = float(np.trace(A_inv_XtX))
        denom = self.n - edf_total
        if denom <= 0:
            return 1e15
        return self.n * fit.rss / (denom * denom)

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
                       A_inv_XtX: np.ndarray, edf: np.ndarray,
                       H_aug: np.ndarray | None):
        """mgcv's edf1 (frequentist tr(2F−F²) bound) and edf2 (sp-uncertainty
        corrected). Wood 2017 §6.11.3. Returns ``(edf2_per_coef, edf1_per_coef)``.

        edf2 = diag((σ² A⁻¹ + Vc1 + Vc2) · X'X) / σ², where

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
        F = A_inv_XtX
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

        # diag((σ²A⁻¹ + Vc1 + Vc2)·X'X)/σ² = edf + diag((Vc1 + Vc2)·X'X)/σ².
        # Each summand is symmetric so einsum('ij,ij->i', M, X'X) gives
        # the diagonal of the matrix product without forming it.
        XtX = self._XtX
        if sigma_squared > 0 and np.isfinite(sigma_squared):
            Vc = Vc1 + Vc2
            edf2 = edf + np.einsum("ij,ij->i", Vc, XtX) / sigma_squared
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
        H = self._reml_hessian(rho)
        H = 0.5 * (H + H.T)
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

    def _reml_hessian_augmented(
        self, rho: np.ndarray, log_sigma2: float,
        fit: "_FitState | None" = None,
    ) -> np.ndarray:
        """Analytical Hessian of V_R wrt (ρ, log σ²) — Wood 2011 §4.

        Differentiating the unprofiled criterion

           2·V_R = (n−Mp)·ζ + rs·exp(−ζ) + log|A| − log|S|+ + const

        (rs = rss + pen, ζ = log σ²) and halving:

           ∂²V_R/∂ζ²    = rs·exp(−ζ) / 2   (= (n−Mp)/2 at ζ̂)
           ∂²V_R/∂ρ_k∂ζ = −g_k·exp(−ζ) / 2 = −g_k/(2σ²)
           ∂²V_R/∂ρ_i∂ρ_j  — same Wood §4 structure as the profiled
                              Hessian, but the rs-term has no chain-rule
                              −g_i g_j/rs piece (we're differentiating
                              rs·exp(−ζ), not log rs).

        At ζ̂ the un-profiling reduces to a rank-1 Schur correction on
        the ρρ block, so we reuse ``_reml_hessian`` (analytical V_R
        Hessian wrt ρ, profiled at ζ̂) and add ``outer(c, c)/d`` where
        c = H_aug[ρ, ζ] and d = H_aug[ζ, ζ]. Off ζ̂ this identity no
        longer holds; the only consumers (vcomp, edf12) always pass
        ``log_sigma2 = log(self.sigma_squared)``, i.e., evaluate at ζ̂,
        so the rank-1 path is exact for them.

        Returned in V_R scale (mgcv's gam.vcomp inverts this).
        """
        if fit is None:
            fit = self._fit_given_rho(rho)
        n_sp = len(self._slots)
        n_minus_Mp = self.n - self._Mp
        sigma2 = float(np.exp(log_sigma2))
        m = n_sp + 1
        H = np.zeros((m, m))
        if n_sp == 0 or sigma2 <= 0 or n_minus_Mp <= 0:
            H[n_sp, n_sp] = 0.5 * n_minus_Mp
            return H

        sp = np.exp(rho)
        g = np.zeros(n_sp)
        for k, slot in enumerate(self._slots):
            a, b = slot.col_start, slot.col_end
            beta_k = fit.beta[a:b]
            g[k] = sp[k] * float(beta_k @ slot.S @ beta_k)

        H_rho_zeta = -0.5 * g / sigma2                       # shape (n_sp,)
        rs = fit.rss + fit.pen
        H_zeta_zeta = 0.5 * rs / sigma2                      # = (n−Mp)/2 at ζ̂

        H_profiled = self._reml_hessian(rho, fit)            # V_R, ρρ-only
        H_rho_rho = H_profiled + np.outer(H_rho_zeta, H_rho_zeta) / H_zeta_zeta

        H[:n_sp, :n_sp] = H_rho_rho
        H[:n_sp, n_sp] = H_rho_zeta
        H[n_sp, :n_sp] = H_rho_zeta
        H[n_sp, n_sp] = H_zeta_zeta
        return H

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
            "Family: gaussian",
            "Link function: identity",
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
            "Family: gaussian",
            "Link function: identity",
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
            rows_k:     list[int]   = []
            rows_F:     list[float] = []
            rows_p:     list[float] = []
            for b, (a, bcol) in zip(self._blocks, self._block_col_ranges):
                beta_b = self._beta[a:bcol]
                Vp_b   = self.Vp[a:bcol, a:bcol]
                edf_b  = float(self.edf[a:bcol].sum())
                # Wald chi²: β'Vp⁻¹β, converted to F = χ²/edf on (edf,
                # df.resid). mgcv's test is a bit more involved (uses
                # Bayesian variance + finite-sample adjustments), but for
                # non-degenerate smooths this is close.
                k = bcol - a
                # pseudo-invert Vp_b via eigen-truncation at rank ≈ edf
                w, U = np.linalg.eigh(Vp_b)
                tol = max(1e-12, w.max() * 1e-8) if w.size and w.max() > 0 else 1e-12
                mask = w > tol
                if mask.any():
                    w_inv = np.where(mask, 1.0 / np.where(mask, w, 1.0), 0.0)
                    Vp_b_pinv = (U * w_inv) @ U.T
                    chisq = float(beta_b @ Vp_b_pinv @ beta_b)
                else:
                    chisq = 0.0
                edf_used = max(edf_b, 1e-8)
                F = chisq / edf_used
                p_val = (
                    float(f_dist.sf(F, edf_used, self.df_residuals))
                    if self.df_residuals > 0 else float("nan")
                )
                rows_label.append(b.label)
                rows_edf.append(edf_b)
                rows_k.append(k)
                rows_F.append(F)
                rows_p.append(p_val)
            sig = significance_code(rows_p)
            sm_tbl = pl.DataFrame({
                "":        rows_label,
                "edf":     np.round(rows_edf, digits),
                "Ref.df":  rows_k,
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
        dev_expl = (
            1.0 - self.rss / self._tss
            if self._tss > 0 else float("nan")
        )
        out.append(
            f"R-sq.(adj) = {self.r_squared_adjusted:.3g}  "
            f"Deviance explained = {dev_expl * 100:.1f}%"
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
    )

    def __init__(self, *, beta, dev, pen, A_chol, A_chol_lower,
                 S_full, log_det_A,
                 eta=None, mu=None, w=None, z=None, alpha=None):
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
