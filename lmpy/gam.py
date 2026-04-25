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
    ):
        if method not in ("REML", "GCV.Cp"):
            raise ValueError(f"method must be 'REML' or 'GCV.Cp', got {method!r}")

        self.formula = formula
        self.method = method

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
            # Coarse coordinate-wise scan to seed L-BFGS-B. Two passes:
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
        # npar = edf_total + 1 (residual variance). Matches R's
        # AIC.gam (which is logLik with df = edf + 1 + 1 for phi).
        self.npar = edf_total + 1.0
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
        return _FitState(
            beta=beta, rss=max(rss, 0.0), pen=pen,
            A_chol=A_chol, A_chol_lower=lower,
            S_full=Sλ, log_det_A=log_det_A,
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

    This is *column deletion* on the original basis, not a rotation: the
    remaining te columns are not orthogonal to the marginal smooths in
    data space, but the combined design is now identifiable. mgcv's
    procedure (`gam.side.R` + `fixDependence`) is what we mirror here.
    """
    if len(blocks) < 2:
        return blocks
    var_sets = [frozenset(b.term) for b in blocks]
    n = int(np.asarray(blocks[0].X).shape[0])
    out: list[SmoothBlock] = []
    for i, b in enumerate(blocks):
        my_vars = var_sets[i]
        Xb = np.asarray(b.X, dtype=float)
        # X1 = intercept + every strict-subset block's design — exactly
        # what `gam.side` builds before calling `fixDependence`.
        cols_X1 = [np.ones((n, 1))]
        for j, other in enumerate(blocks):
            if i == j:
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
    """Fit-at-one-ρ bundle passed from _fit_given_rho to post-fit code."""
    __slots__ = (
        "beta", "rss", "pen",
        "A_chol", "A_chol_lower",
        "S_full", "log_det_A",
    )

    def __init__(self, *, beta, rss, pen, A_chol, A_chol_lower,
                 S_full, log_det_A):
        self.beta = beta
        self.rss = rss
        self.pen = pen
        self.A_chol = A_chol
        self.A_chol_lower = A_chol_lower
        self.S_full = S_full
        self.log_det_A = log_det_A
