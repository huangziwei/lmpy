"""Linear mixed-effects model — lme4-style profiled deviance.

Built on lmpy.formula's ``parse → expand → materialize / materialize_bars``
pipeline. The fixed-effect side comes from ``materialize`` (R-canonical
column names). The random-effect side comes from ``materialize_bars``,
which returns ``Z``, an integer ``Λᵀ`` template, and an initial ``θ``.

We optimize the ML or REML profiled deviance over ``θ`` using L-BFGS-B
(diagonal entries of ``Λ`` constrained to be ≥ 0 for identifiability),
then recover ``β̂``, ``σ̂``, ``SE(β̂)``, and the per-bar variance components
at the optimum.

References
----------
Bates, Mächler, Bolker, Walker (2015), "Fitting Linear Mixed-Effects
Models Using lme4", J. Stat. Software 67(1), §5 ("Profiled Deviance").
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from sksparse.cholmod import (
    CholmodError,
    cho_factor,
    csc_array,
    eye_array,
)

from .formula import materialize_bars
from .design import prepare_design
from .lm import _label_top_n, _lowess, _qq_plot
from .utils import format_df, format_signif, format_signif_jointly

__all__ = ["lme", "Profile"]


def _sparse_Lt_spec(
    template: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute the CSC structure of Λᵀ from the integer template.

    Returns ``(theta_pos, indices, indptr)`` such that for any θ vector,
    ``csc_array((theta[theta_pos], indices, indptr), shape=template.shape)``
    reconstructs Λᵀ. Because the structure is fixed, CHOLMOD can reuse the
    symbolic analysis across every deviance evaluation.
    """
    q = template.shape[0]
    indptr = np.empty(q + 1, dtype=np.int32)
    indptr[0] = 0
    indices_parts: list[np.ndarray] = []
    theta_pos_parts: list[np.ndarray] = []
    for j in range(q):
        col = template[:, j]
        nz_rows = np.nonzero(col)[0]
        indices_parts.append(nz_rows.astype(np.int32))
        theta_pos_parts.append((col[nz_rows] - 1).astype(np.int64))
        indptr[j + 1] = indptr[j] + nz_rows.size
    indices = (
        np.concatenate(indices_parts) if indices_parts
        else np.zeros(0, dtype=np.int32)
    )
    theta_pos = (
        np.concatenate(theta_pos_parts) if theta_pos_parts
        else np.zeros(0, dtype=np.int64)
    )
    return theta_pos, indices, indptr


def _bar_sizes(cnms: dict) -> list[int]:
    """Component count ``c`` per bar (1 for scalar bars, ≥ 2 for vector)."""
    return [
        len(names) if isinstance(names, list) else 1
        for names in cnms.values()
    ]


def _theta_diag_idx(bar_sizes: list[int]) -> list[int]:
    """0-indexed θ positions on the diagonal of any per-level Λᵀ block.

    ``materialize_bars`` packs each c×c upper-triangular Λᵀ block row by
    row: ``θ[off+0] = (0,0)``, ``θ[off+1] = (0,1)``, … . The diagonal
    positions therefore start each row, at cumulative offsets ``c, c-1,
    c-2, …``.
    """
    diag: list[int] = []
    off = 0
    for c in bar_sizes:
        cum = 0
        for i in range(c):
            diag.append(off + cum)
            cum += c - i
        off += c * (c + 1) // 2
    return diag


def _per_bar_relative_cov(theta: np.ndarray, bar_sizes: list[int]) -> list[np.ndarray]:
    """Recover the c×c relative-covariance ``Σ_b = Λ_b Λ_bᵀ`` per bar."""
    blocks: list[np.ndarray] = []
    off = 0
    for c in bar_sizes:
        Lt = np.zeros((c, c))
        idx = 0
        for i in range(c):
            for j in range(i, c):
                Lt[i, j] = theta[off + idx]
                idx += 1
        L = Lt.T
        blocks.append(L @ L.T)
        off += c * (c + 1) // 2
    return blocks


class lme:
    """Linear mixed-effects model, fit by ML or REML profiled deviance.

    Parameters
    ----------
    formula : str
        lme4-style mixed model formula, e.g.
        ``"Reaction ~ 1 + Days + (1+Days|Subject)"``.
    data : polars.DataFrame
        Data table; rows with NA in any referenced column are dropped
        before fitting.
    REML : bool, default True
        Fit by REML (matches lme4's default) or ML.

    Attributes (always set)
    -----------------------
    n, p, q : int
        Sample size, # of fixed-effect coefficients, # of random-effect
        coefficients (= total number of Z columns).
    n_groups : dict[str, int]
        Number of unique levels per (raw) grouping factor.
    sigma : float
        Residual SD (σ̂).
    bhat, se_bhat, t_values : polars.DataFrame
        Fixed-effect estimates / SEs / t-values, one row each, columns
        keyed by R-canonical fixed-effect names (``(Intercept)``,
        ``MachineB``, …).
    sd_re : dict[str, np.ndarray]
        Per-bar component SDs. Keyed by the disambiguated bar key from
        ``ReTerms.cnms`` (e.g. ``"Subject"``, ``"Subject.1"``). Length
        equals the bar's component count (1 for scalar bars).
    corr_re : dict[str, np.ndarray | None]
        Per-bar correlation matrix. ``None`` for scalar bars; a c×c
        matrix for vector bars.
    npar : int
        Total parameter count (fixed effects + θ + 1 residual variance);
        used for likelihood ratio tests.

    Attributes (REML=True only)
    ---------------------------
    REML_criterion : float
        Optimized REML criterion, ``-2 log L_REML``.

    Attributes (REML=False only)
    ----------------------------
    deviance : float
        Optimized ML deviance, ``-2 log L``.
    loglike : float
    df_resid : int
        ``n - npar`` (matches lme4's printed ``df.resid``).

    Attributes (both REML and ML)
    -----------------------------
    AIC, BIC : float
        Information criteria. For ML fits, computed from the ML deviance;
        for REML fits, from the REML criterion (matches lme4's ``AIC()``
        and ``BIC()``). REML AIC/BIC are only comparable across models
        with the same fixed-effects structure.
    """

    def __init__(self, formula: str, data: pl.DataFrame, REML: bool = True):
        self.formula = formula
        self.REML = REML

        d = prepare_design(formula, data)
        if not d.expanded.bars:
            raise ValueError(
                f"lme requires at least one random-effect bar; got formula={formula!r}"
            )
        # materialize_bars is called on d.data (response-NA-cleaned) so it
        # applies the same NA-omit policy as materialize() did for X — the
        # resulting Z stays row-aligned with X.
        re = materialize_bars(d.expanded, d.data)
        X_df = d.X
        y = d.y.to_numpy().astype(float)
        X = X_df.to_numpy().astype(float)
        Z = re.Z
        n, p = X.shape
        q = Z.shape[1]

        self.data = d.data
        self.X = X_df
        self.y = y
        self.Z = Z
        self.column_names = list(X_df.columns)
        self.n = n
        self.p = p
        self.q = q
        self._re = re

        bar_sizes = _bar_sizes(re.cnms)
        self._bar_sizes = bar_sizes
        self.n_groups = {g: len(levs) for g, levs in re.flist_levels.items()}

        # ------------- profiled-deviance optimization ----------------------
        #
        # Z and Λᵀ are stored sparse (CSC). The hot step — the Cholesky of
        # ``M = Λ Zᵀ Z Λᵀ + I`` — goes through ``sksparse.cholmod`` (CHOLMOD
        # with AMD reordering). The symbolic factor is computed once on the
        # first factorization and reused by ``factor.factorize(M_new)`` every
        # subsequent call; only the numeric re-factor runs inside the
        # optimizer loop. Without this, InstEval-class fits (q ≈ 4k) sit in
        # dense Cholesky for O(q³) flops per deviance eval.
        template = re.Lambdat
        lt_theta_pos, lt_indices, lt_indptr = _sparse_Lt_spec(template)
        Z_sp = csc_array(Z)
        eye_q_sp = eye_array(q, format="csc")
        XtX = X.T @ X
        Xty = X.T @ y
        yty = float(y @ y)
        log2pi = float(np.log(2.0 * np.pi))

        # Cache pieces profile() and other post-fit methods reuse.
        self._template = template
        self._lt_theta_pos = lt_theta_pos
        self._lt_indices = lt_indices
        self._lt_indptr = lt_indptr
        self._lt_shape = template.shape
        self._Z_sp = Z_sp
        self._eye_q_sp = eye_q_sp
        self._chol_factor = None
        self._XtX = XtX
        self._Xty = Xty
        self._yty = yty
        self._log2pi = log2pi

        diag_set = set(_theta_diag_idx(bar_sizes))
        self._diag_set = diag_set
        bounds = [
            (0.0, None) if i in diag_set else (None, None)
            for i in range(len(re.theta))
        ]
        self._theta_bounds = bounds

        theta0 = re.theta.astype(float).copy()
        res = minimize(
            lambda th: self._ml_deviance(th) if not REML else self._reml_deviance(th),
            theta0, method="L-BFGS-B", bounds=bounds,
            options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 1000},
        )
        theta_hat = res.x
        self.theta = theta_hat
        self._optim = res

        # ------------- recover β̂, σ̂, SE(β̂) at the optimum ------------------
        #
        # Same Cholesky-based profile-deviance math as ``_chol_block``, but
        # we also keep β̂ and û here (the deviance loop discards them).
        # ``F⁻¹ = M⁻¹`` lets us evaluate ``cu' cu`` and ``RZX' RZX`` as inner
        # products against ``M⁻¹(ZLᵀy)`` and ``M⁻¹(ZLᵀX)`` without ever
        # materializing ``cu`` or ``RZX``.
        Lt = self._build_Lt_sparse(theta_hat)
        ZL = Z_sp @ Lt.T
        M = (ZL.T @ ZL + eye_q_sp).tocsc()
        if self._chol_factor is None:
            self._chol_factor = cho_factor(M)
        else:
            self._chol_factor.factorize(M)
        F = self._chol_factor

        # Snapshot Λ and L at the MLE as dense ndarrays (matches m.Z's
        # convention). profile()/_ranef() re-factorize _chol_factor at
        # non-MLE θ, so freezing copies here detaches us from those.
        # L is in CHOLMOD's permuted ordering — lower triangular by
        # construction; that's also the ordering Bates' Fig 2.4 shows.
        self.Lambda = Lt.T.toarray()
        self.L = F.L.toarray()

        ZLty = np.asarray(ZL.T @ y).ravel()
        ZLtX = np.asarray(ZL.T @ X)
        M_inv_ZLty = F.solve(ZLty)
        M_inv_ZLtX = F.solve(ZLtX)
        # See _chol_block for why this reach uses einsum instead of @.
        cu_sq = float(np.einsum("i,i->", ZLty, M_inv_ZLty))
        XtX_eff = XtX - np.einsum("ij,ik->jk", ZLtX, M_inv_ZLtX)
        Rx = np.linalg.cholesky(XtX_eff)
        rhs = Xty - np.einsum("ij,i->j", ZLtX, M_inv_ZLty)
        cb = solve_triangular(Rx, rhs, lower=True)
        beta = solve_triangular(Rx.T, cb, lower=False)
        rss = yty - cu_sq - float(np.einsum("i,i->", cb, cb))
        # spherical random-effect coefficients u = M⁻¹(ZLᵀy − ZLᵀX β)
        self._u = F.solve(ZLty - np.einsum("ij,j->i", ZLtX, beta))

        sigma2 = rss / (n - p) if REML else rss / n
        sigma = float(np.sqrt(sigma2))
        self.sigma = sigma
        self.sigma_squared = sigma2

        # Fitted values ŷ = Xβ̂ + Z Λ û and residuals ε̂ = y − ŷ
        self.fitted = np.einsum("ij,j->i", X, beta) + ZL @ self._u
        self.residuals = y - self.fitted
        # ε̂ / σ̂ — what lme4 calls Pearson / "Scaled residuals"
        self.scaled_residuals = self.residuals / sigma

        # Var(β̂) = σ̂² (XᵀX_eff)⁻¹ = σ̂² R_x⁻ᵀ R_x⁻¹
        Rx_inv = solve_triangular(Rx, np.eye(p), lower=True)
        vcov_beta = sigma2 * np.einsum("ij,ik->jk", Rx_inv, Rx_inv)
        se_beta = np.sqrt(np.diag(vcov_beta))
        self._vcov_beta_arr = vcov_beta
        self.vcov_beta = pl.DataFrame(
            {c: vcov_beta[:, i] for i, c in enumerate(self.column_names)}
        )

        self._beta = beta
        self._se_beta = se_beta
        self.bhat = pl.DataFrame(
            {c: [float(beta[i])] for i, c in enumerate(self.column_names)}
        )
        self.se_bhat = pl.DataFrame(
            {c: [float(se_beta[i])] for i, c in enumerate(self.column_names)}
        )
        t_vals = beta / se_beta
        self.t_values = pl.DataFrame(
            {c: [float(t_vals[i])] for i, c in enumerate(self.column_names)}
        )

        # ------------- per-bar variance components -------------------------
        Sigma_blocks = _per_bar_relative_cov(theta_hat, bar_sizes)
        self.sd_re: dict[str, np.ndarray] = {}
        self.corr_re: dict[str, np.ndarray | None] = {}
        for key, Sigma in zip(re.cnms.keys(), Sigma_blocks):
            d = np.sqrt(np.diag(Sigma))
            self.sd_re[key] = sigma * d
            if Sigma.shape[0] > 1:
                with np.errstate(invalid="ignore", divide="ignore"):
                    corr = Sigma / np.outer(d, d)
                corr = np.where(np.isfinite(corr), corr, 0.0)
                np.fill_diagonal(corr, 1.0)
                self.corr_re[key] = corr
            else:
                self.corr_re[key] = None

        # ------------- summary statistics ----------------------------------
        # npar = fixed-effect coefficients + θ entries + 1 residual variance
        self.npar = p + len(theta_hat) + 1
        opt = float(res.fun)
        if REML:
            self.REML_criterion = opt
        else:
            self.deviance = opt
            self.loglike = -0.5 * opt
            self.df_resid = n - self.npar
        # AIC/BIC use the ML deviance for ML fits and the REML criterion
        # for REML fits, matching lme4's ``AIC.merMod`` / ``BIC.merMod``.
        self.AIC = opt + 2.0 * self.npar
        self.BIC = opt + np.log(n) * self.npar

    # ---- deviance building blocks --------------------------------------
    #
    # These are used both by __init__ (for the initial ML/REML fit) and by
    # profile() (for the per-grid-point re-optimization).

    def _build_Lt_sparse(self, theta: np.ndarray) -> csc_array:
        """Build Λᵀ as a CSC sparse matrix from the precomputed structure.

        The sparsity pattern is fixed by the integer template, so we just
        swap the numeric entries on each call. Same pattern every call is
        what lets CHOLMOD reuse the symbolic analysis."""
        data = np.asarray(theta, dtype=float)[self._lt_theta_pos]
        return csc_array(
            (data, self._lt_indices, self._lt_indptr),
            shape=self._lt_shape, copy=False,
        )

    def _chol_block(
        self, theta: np.ndarray, *,
        y: np.ndarray | None = None, X: np.ndarray | None = None,
        XtX: np.ndarray | None = None, Xty: np.ndarray | None = None,
        yty: float | None = None,
    ) -> tuple[float, float, float] | None:
        """Core Cholesky step. Returns ``(rss, log|Lz|, log|Rx|)`` at β̂_θ,
        or ``None`` if the factorization fails.

        With defaults this uses the original ``X``/``y`` cached on the fit.
        Overrides let ``profile()`` plug in modified designs (e.g. ``y``
        adjusted by a fixed β_j, or ``X`` with a column removed).

        ``log|Lz|`` is computed as ½·``factor.logdet()`` since
        ``Lz Lzᵀ = M`` means ``|M| = |Lz|²``."""
        y = self.y if y is None else y
        X = self.X.to_numpy().astype(float) if X is None else X
        XtX = self._XtX if XtX is None else XtX
        Xty = self._Xty if Xty is None else Xty
        yty = self._yty if yty is None else yty
        Lt = self._build_Lt_sparse(theta)
        ZL = self._Z_sp @ Lt.T
        M = (ZL.T @ ZL + self._eye_q_sp).tocsc()
        try:
            if self._chol_factor is None:
                self._chol_factor = cho_factor(M)
            else:
                self._chol_factor.factorize(M)
        except CholmodError:
            return None
        F = self._chol_factor
        ZLty = np.asarray(ZL.T @ y).ravel()
        M_inv_ZLty = F.solve(ZLty)
        # Apple Accelerate's small-size GEMV/GEMM dispatch is non-deterministic
        # across fresh buffers (~1e-12 noise), which L-BFGS-B's finite-diff
        # gradient amplifies into visibly different θ. einsum sidesteps that
        # BLAS path and stays bit-identical.
        cu_sq = float(np.einsum("i,i->", ZLty, M_inv_ZLty))
        log_det_Lz = 0.5 * F.logdet()
        if X.shape[1] > 0:
            ZLtX = np.asarray(ZL.T @ X)
            M_inv_ZLtX = F.solve(ZLtX)
            XtX_eff = XtX - np.einsum("ij,ik->jk", ZLtX, M_inv_ZLtX)
            try:
                Rx = np.linalg.cholesky(XtX_eff)
            except np.linalg.LinAlgError:
                return None
            rhs = Xty - np.einsum("ij,i->j", ZLtX, M_inv_ZLty)
            cb = solve_triangular(Rx, rhs, lower=True)
            rss = yty - cu_sq - float(np.einsum("i,i->", cb, cb))
            log_det_Rx = float(np.log(np.diag(Rx)).sum())
        else:
            rss = yty - cu_sq
            log_det_Rx = 0.0
        if rss <= 0:
            return None
        return rss, log_det_Lz, log_det_Rx

    def _ml_deviance(
        self, theta: np.ndarray, *,
        sigma_fix: float | None = None,
        y: np.ndarray | None = None, X: np.ndarray | None = None,
        XtX: np.ndarray | None = None, Xty: np.ndarray | None = None,
        yty: float | None = None,
    ) -> float:
        """ML deviance at this θ. Defaults to σ profiled out (σ̂² = rss/n);
        pass ``sigma_fix`` to hold σ at a specific value instead."""
        n = len(self.y) if y is None else len(y)
        r = self._chol_block(
            theta, y=y, X=X, XtX=XtX, Xty=Xty, yty=yty,
        )
        if r is None:
            return 1e15
        rss, log_det_Lz, _ = r
        if sigma_fix is None:
            return 2.0 * log_det_Lz + n * (1.0 + self._log2pi + np.log(rss / n))
        s2 = sigma_fix ** 2
        return 2.0 * log_det_Lz + n * (self._log2pi + np.log(s2)) + rss / s2

    def _reml_deviance(self, theta: np.ndarray) -> float:
        """REML ``-2 log L_REML`` at this θ. β profiles out, then σ."""
        n, p = self.n, self.p
        r = self._chol_block(theta)
        if r is None:
            return 1e15
        rss, log_det_Lz, log_det_Rx = r
        df = n - p
        return (
            2.0 * log_det_Lz + 2.0 * log_det_Rx
            + df * (1.0 + self._log2pi + np.log(rss / df))
        )

    # ---- profile likelihood --------------------------------------------

    def _refit_theta(self, obj_fn, theta_start: np.ndarray) -> tuple[float, np.ndarray]:
        """Re-optimize θ against ``obj_fn(theta) → deviance``."""
        res = minimize(
            obj_fn, theta_start, method="L-BFGS-B", bounds=self._theta_bounds,
            options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 1000},
        )
        return float(res.fun), res.x

    def _dev_with_beta_fixed(
        self, j: int, beta_j_tgt: float, theta_start: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Min ML deviance with β_j = ``beta_j_tgt``. Trick: subtract
        ``x_j · β_j_tgt`` from y and drop column j — the remaining fit has
        the same functional form."""
        X_full = self.X.to_numpy().astype(float)
        x_j = X_full[:, j]
        X_rest = np.delete(X_full, j, axis=1)
        y_adj = self.y - x_j * beta_j_tgt
        XtX_rest = X_rest.T @ X_rest
        Xty_rest = X_rest.T @ y_adj
        yty_adj = float(y_adj @ y_adj)
        return self._refit_theta(
            lambda th: self._ml_deviance(
                th, y=y_adj, X=X_rest,
                XtX=XtX_rest, Xty=Xty_rest, yty=yty_adj,
            ),
            theta_start,
        )

    def _dev_with_sigma_fixed(
        self, sigma_tgt: float, theta_start: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Min ML deviance with σ = ``sigma_tgt`` (β profiles out)."""
        return self._refit_theta(
            lambda th: self._ml_deviance(th, sigma_fix=sigma_tgt),
            theta_start,
        )

    def _dev_with_sd_fixed(
        self, slot_i: int, sd_tgt: float,
        sigma_start: float, theta_start: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Min ML deviance with σ_i = σ · θ[slot_i] pinned at ``sd_tgt``.

        Scalar-bar case: the bar has one θ entry, so pinning ``σ · θ[slot_i]
        = sd_tgt`` is a single nonlinear constraint. We re-parameterize as
        ``(σ, θ_rest)`` with ``θ[slot_i] = sd_tgt / σ`` and minimize jointly."""
        other = [k for k in range(len(self._theta_bounds)) if k != slot_i]
        theta_rest0 = np.array([theta_start[k] for k in other])

        def obj(x):
            sigma = x[0]
            if sigma <= 0:
                return 1e15
            theta = np.zeros(len(self._theta_bounds))
            theta[slot_i] = sd_tgt / sigma
            for k, slot in enumerate(other):
                theta[slot] = x[1 + k]
            return self._ml_deviance(theta, sigma_fix=sigma)

        x0 = np.concatenate([[sigma_start], theta_rest0])
        bounds = [(1e-8, None)] + [self._theta_bounds[k] for k in other]
        res = minimize(
            obj, x0, method="L-BFGS-B", bounds=bounds,
            options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 1000},
        )
        # Reconstruct θ at the optimum for warm-start of neighboring points.
        theta_opt = np.zeros(len(self._theta_bounds))
        sigma_opt = res.x[0]
        theta_opt[slot_i] = sd_tgt / sigma_opt
        for k, slot in enumerate(other):
            theta_opt[slot] = res.x[1 + k]
        return float(res.fun), theta_opt

    def profile(self, n_grid: int = 41) -> "Profile":
        """Compute profile-likelihood curves for σ_i, σ, and each β_j.

        For each parameter we fix it at ``n_grid`` values centered on the
        MLE, re-minimize the ML deviance over the remaining parameters, and
        record ``ζ = sign(v − v̂) · √(d(v) − d̂)``. The result is a
        :class:`Profile` carrying one DataFrame per parameter plus the MLEs.

        For REML fits we first re-fit by ML, per lme4's convention (the LRT
        statistic requires ML). Only scalar bars ``(1|g)`` are supported in
        this first port.
        """
        if any(c > 1 for c in self._bar_sizes):
            raise NotImplementedError(
                "profile() currently requires scalar bars (1|g); "
                "vector bars like (1+x|g) need a different parameterization."
            )
        if self.REML:
            return lme(self.formula, self.data, REML=False).profile(n_grid=n_grid)

        d_hat = self.deviance
        theta_hat = self.theta.copy()
        sigma_hat = self.sigma

        data: dict[str, pl.DataFrame] = {}
        estimate: dict[str, float] = {}

        # Grid widths are picked to comfortably cover |ζ| ≤ 3 (so the
        # 99% CI cutoff at ±2.576 is inside the grid). β uses Wald SE as a
        # scale; σ and σ_i get multiplicative ranges.

        # -- σ_i (one per scalar bar) ---------------------------------------
        slot_offsets = np.cumsum([0] + self._bar_sizes[:-1])
        for i, bar_key in enumerate(self.sd_re):
            sd_i = float(self.sd_re[bar_key][0])
            label = f".sig{i + 1:02d}"
            estimate[label] = sd_i
            grid = np.linspace(1e-3 * sd_i, 3.5 * sd_i, n_grid)
            slot_i = int(slot_offsets[i])
            zetas = np.empty(n_grid)
            theta_warm = theta_hat.copy()
            sigma_warm = sigma_hat
            for k, v in enumerate(grid):
                d_k, theta_warm = self._dev_with_sd_fixed(
                    slot_i, v, sigma_warm, theta_warm,
                )
                zetas[k] = np.sign(v - sd_i) * np.sqrt(max(0.0, d_k - d_hat))
            data[label] = pl.DataFrame({"value": grid, "zeta": zetas})

        # -- σ ----------------------------------------------------------------
        estimate[".sigma"] = sigma_hat
        log_grid = np.linspace(-0.6, 0.6, n_grid) + np.log(sigma_hat)
        sigma_grid = np.exp(log_grid)
        zetas = np.empty(n_grid)
        theta_warm = theta_hat.copy()
        for k, s in enumerate(sigma_grid):
            d_k, theta_warm = self._dev_with_sigma_fixed(s, theta_warm)
            zetas[k] = np.sign(s - sigma_hat) * np.sqrt(max(0.0, d_k - d_hat))
        data[".sigma"] = pl.DataFrame({"value": sigma_grid, "zeta": zetas})

        # -- β_j --------------------------------------------------------------
        for j, name in enumerate(self.column_names):
            beta_j = float(self._beta[j])
            se_j = float(self._se_beta[j])
            estimate[name] = beta_j
            grid = np.linspace(beta_j - 4 * se_j, beta_j + 4 * se_j, n_grid)
            zetas = np.empty(n_grid)
            theta_warm = theta_hat.copy()
            for k, b in enumerate(grid):
                d_k, theta_warm = self._dev_with_beta_fixed(j, b, theta_warm)
                zetas[k] = np.sign(b - beta_j) * np.sqrt(max(0.0, d_k - d_hat))
            data[name] = pl.DataFrame({"value": grid, "zeta": zetas})

        return Profile(data, estimate)

    # ---- lmer-style printing --------------------------------------------

    def _header(self) -> str:
        if self.REML:
            return "Linear mixed model fit by REML"
        return "Linear mixed model fit by maximum likelihood"

    def _fit_criterion_lines(self) -> list[str]:
        if self.REML:
            return [f"REML criterion at convergence: {self.REML_criterion:.4f}"]
        labels = ["AIC", "BIC", "logLik", "-2*log(L)", "df.resid"]
        vals = [
            f"{self.AIC:.4f}",
            f"{self.BIC:.4f}",
            f"{self.loglike:.4f}",
            f"{self.deviance:.4f}",
            f"{self.df_resid}",
        ]
        widths = [max(len(l), len(v)) for l, v in zip(labels, vals)]
        hdr = " ".join(l.rjust(w) for l, w in zip(labels, widths))
        row = " ".join(v.rjust(w) for v, w in zip(vals, widths))
        return [hdr, row]

    def _n_obs_line(self) -> str:
        groups = "; ".join(f"{g}, {n}" for g, n in self.n_groups.items())
        return f"Number of obs: {self.n}, groups:  {groups}"

    @staticmethod
    def _format_col(values: list[float]) -> list[str]:
        """Format a numeric column with shared decimal places (R's format())."""
        strs = [f"{v:.4g}" for v in values]
        if any("e" in s or "E" in s for s in strs):
            return strs
        max_dp = max((len(s.split(".")[1]) for s in strs if "." in s), default=0)
        if max_dp == 0:
            return strs
        return [f"{v:.{max_dp}f}" for v in values]

    def _re_table_lines(self, include_variance: bool) -> list[str]:
        max_corr_cols = 0
        for c in self.corr_re.values():
            if c is not None:
                max_corr_cols = max(max_corr_cols, c.shape[0] - 1)

        # Collect per-bar entries: (group_label, name, sd, variance, corrs)
        entries: list[tuple[str, str, float, float, list[float]]] = []
        for key in self.sd_re:
            names = self._re.cnms[key]
            if not isinstance(names, list):
                names = [names]
            sds = self.sd_re[key]
            corr = self.corr_re.get(key)
            for i, (name, s) in enumerate(zip(names, sds)):
                corrs = [corr[i, j] for j in range(i)] if (corr is not None and i > 0) else []
                entries.append((key if i == 0 else "", name, float(s), float(s) ** 2, corrs))
        entries.append(("Residual", "", float(self.sigma), float(self.sigma_squared), []))

        sd_col = self._format_col([e[2] for e in entries])
        var_col = self._format_col([e[3] for e in entries]) if include_variance else None

        rows: list[list[str]] = []
        for idx, (group, name, _s, _v, corrs) in enumerate(entries):
            # Residual row: blank out the Name cell
            if group == "Residual" and idx == len(entries) - 1:
                name_cell = ""
            else:
                name_cell = name
            row = [group, name_cell]
            if var_col is not None:
                row.append(var_col[idx])
            row.append(sd_col[idx])
            row.extend(f"{c:.2f}" for c in corrs)
            rows.append(row)

        header = ["Groups", "Name"]
        if include_variance:
            header.append("Variance")
        header.append("Std.Dev.")
        if max_corr_cols > 0:
            header.append("Corr")
            header.extend([""] * (max_corr_cols - 1))

        ncols = len(header)
        for r in rows:
            r.extend([""] * (ncols - len(r)))
        widths = [len(h) for h in header]
        for r in rows:
            for i, c in enumerate(r):
                widths[i] = max(widths[i], len(c))

        def fmt(cells: list[str]) -> str:
            return (" " + " ".join(c.ljust(w) for c, w in zip(cells, widths))).rstrip()

        return [fmt(header)] + [fmt(r) for r in rows]

    def _fixef_table(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "coef": self.column_names,
                "Estimate": self._beta.astype(float),
                "Std. Error": self._se_beta.astype(float),
                "t value": (self._beta / self._se_beta).astype(float),
            }
        )

    def _fixef_corr_lines(self) -> list[str]:
        """Correlation-of-fixed-effects block, lme4-style (lower-triangular)."""
        p = self._vcov_beta_arr.shape[0]
        if p <= 1:
            return []
        vcov = self._vcov_beta_arr
        d = np.sqrt(np.diag(vcov))
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = vcov / np.outer(d, d)
        corr = np.where(np.isfinite(corr), corr, 0.0)
        names = ["(Intr)" if n == "(Intercept)" else n for n in self.column_names]
        row_w = max(len(n) for n in names[1:])
        cell_w = max(6, max(len(n) for n in names[: p - 1]))
        header = " " * row_w + " " + " ".join(
            names[j].rjust(cell_w) for j in range(p - 1)
        )
        rows = []
        for i in range(1, p):
            cells = " ".join(f"{corr[i, j]:.3f}".rjust(cell_w) for j in range(i))
            rows.append(names[i].ljust(row_w) + " " + cells)
        return ["Correlation of Fixed Effects:", header] + rows

    def __repr__(self) -> str:
        out = [self._header(), f"Formula: {self.formula}"]
        out.extend(self._fit_criterion_lines())
        out.append("Random effects:")
        out.extend(self._re_table_lines(include_variance=False))
        out.append(self._n_obs_line())
        out.append("Fixed Effects:")
        out.append(format_df(self.bhat))
        return "\n".join(out)

    def __str__(self) -> str:
        return self.__repr__()

    def _scaled_residuals_lines(self) -> list[str]:
        scaled = self.residuals / self.sigma
        qs = np.quantile(scaled, [0.0, 0.25, 0.5, 0.75, 1.0])
        labels = ["Min", "1Q", "Median", "3Q", "Max"]
        vals = [f"{v:.4f}" for v in qs]
        widths = [max(len(l), len(v)) for l, v in zip(labels, vals)]
        hdr = " ".join(l.rjust(w) for l, w in zip(labels, widths))
        row = " ".join(v.rjust(w) for v, w in zip(vals, widths))
        return ["Scaled residuals:", hdr, row]

    def summary(self, digits: int = 4) -> None:
        out = [self._header(), f"Formula: {self.formula}", ""]
        out.extend(self._fit_criterion_lines())
        out.append("")
        out.extend(self._scaled_residuals_lines())
        out.append("")
        out.append("Random effects:")
        out.extend(self._re_table_lines(include_variance=True))
        out.append(self._n_obs_line())
        out.append("")
        out.append("Fixed effects:")
        raw = self._fixef_table().rename({"coef": ""})
        est_arr = raw["Estimate"].to_numpy()
        se_arr  = raw["Std. Error"].to_numpy()
        tval    = raw["t value"].to_numpy()
        est_s, se_s = format_signif_jointly([est_arr, se_arr], digits=digits)
        tbl = pl.DataFrame({
            "":           raw[""].to_list(),
            "Estimate":   est_s,
            "Std. Error": se_s,
            "t value":    format_signif(tval, digits=digits),
        })
        out.append(format_df(
            tbl,
            align={c: "right" for c in ("Estimate", "Std. Error", "t value")},
        ))
        corr_lines = self._fixef_corr_lines()
        if corr_lines:
            out.append("")
            out.extend(corr_lines)
        print("\n".join(out))

    # ---- diagnostic plots ----------------------------------------------

    def _ranef(self):
        """BLUPs in original units with posterior SEs, sliced per bar.

        Returns a list of ``(bar_key, levels, cnames, b_mat, se_mat)`` —
        ``b_mat`` and ``se_mat`` are ``(n_levels, n_components)`` arrays.

        Posterior covariance: ``Var(b̂ | y) = σ² · Λ M⁻¹ Λᵀ``. We pull the
        diagonal in ``O(q²)`` via one dense ``F.solve(Λᵀ_dense)``; ``q``
        well into the thousands triggers heavy work, so this is lazy and
        cached. Defensively re-factorizes ``M`` at θ̂ since callers like
        ``profile()`` over-write the factor during their own optimization.
        """
        cache = getattr(self, "_ranef_cache", None)
        if cache is not None:
            return cache
        Lt = self._build_Lt_sparse(self.theta)
        ZL = self._Z_sp @ Lt.T
        M = (ZL.T @ ZL + self._eye_q_sp).tocsc()
        self._chol_factor.factorize(M)
        F = self._chol_factor
        Lt_dense = Lt.toarray()
        b_full = (Lt_dense.T @ self._u).ravel()
        M_inv_Lt = F.solve(Lt_dense)
        var_b = self.sigma_squared * (Lt_dense * M_inv_Lt).sum(axis=0)
        se_full = np.sqrt(np.clip(var_b, 0.0, None))

        out = []
        Gp = self._re.Gp
        flist = self._re.flist_levels
        for k, key in enumerate(self._re.cnms):
            start, end = Gp[k], Gp[k + 1]
            cnames = self._re.cnms[key]
            cnames = list(cnames) if isinstance(cnames, list) else [cnames]
            c = len(cnames)
            n_levels = (end - start) // c
            b_mat = b_full[start:end].reshape(n_levels, c)
            se_mat = se_full[start:end].reshape(n_levels, c)
            # Recover original group name (lme4 suffixes ".1", ".2" if reused)
            gname = key
            if gname not in flist:
                base, _, tail = key.rpartition(".")
                if tail.isdigit() and base in flist:
                    gname = base
            levels = list(flist[gname])
            out.append((key, levels, cnames, b_mat, se_mat))
        self._ranef_cache = out
        return out

    def _pooled_std_blups(self) -> np.ndarray:
        """All BLUPs concatenated, each component scaled by its model SD.

        Used by the 2×2 ``plot()``'s combined random-effect Q-Q panel.
        """
        out = []
        for key, _levels, _cnames, b_mat, _se in self._ranef():
            sds = self.sd_re[key]
            for j, sd in enumerate(sds):
                if sd > 0:
                    out.append(b_mat[:, j] / float(sd))
        if not out:
            return np.array([])
        return np.concatenate(out)

    def plot_observed_fitted(
        self, ax=None, figsize=None,
        facecolor="none", edgecolor="black", label_n=3,
    ):
        if ax is None:
            _fig, ax = plt.subplots(figsize=figsize)
        y = np.asarray(self.y, dtype=float)
        yhat = np.asarray(self.fitted, dtype=float)
        ax.scatter(yhat, y, facecolor=facecolor, edgecolor=edgecolor)
        lo = float(min(y.min(), yhat.min()))
        hi = float(max(y.max(), yhat.max()))
        ax.plot([lo, hi], [lo, hi], color="black", linestyle="--")
        _label_top_n(ax, yhat, y, scores=self.residuals, n=label_n)
        ax.set_xlabel("Fitted")
        ax.set_ylabel("Observed")
        ax.set_title("Observed vs. Fitted")

    def plot_residuals(
        self, ax=None, figsize=None,
        facecolor="none", edgecolor="black",
        smooth=True, label_n=3,
    ):
        if ax is None:
            _fig, ax = plt.subplots(figsize=figsize)
        yhat = np.asarray(self.fitted, dtype=float)
        r = np.asarray(self.residuals, dtype=float)
        ax.scatter(yhat, r, facecolor=facecolor, edgecolor=edgecolor)
        ax.axhline(0, color="black", linestyle="--")
        if smooth:
            xs, ys = _lowess(yhat, r)
            ax.plot(xs, ys, color="red", linewidth=1.0)
        _label_top_n(ax, yhat, r, scores=r, n=label_n)
        ax.set_xlabel("Fitted")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals vs. Fitted Plot")

    def plot_qq(self, ax=None, figsize=None, label_n=3):
        if ax is None:
            _fig, ax = plt.subplots(figsize=figsize)
        _qq_plot(ax, self.scaled_residuals, label_n=label_n)

    def plot_scale_location(
        self, ax=None, figsize=None,
        facecolor="none", edgecolor="black",
        smooth=True, label_n=3,
    ):
        if ax is None:
            _fig, ax = plt.subplots(figsize=figsize)
        yhat = np.asarray(self.fitted, dtype=float)
        s = np.sqrt(np.abs(self.scaled_residuals))
        ax.scatter(yhat, s, facecolor=facecolor, edgecolor=edgecolor)
        if smooth:
            xs, ys = _lowess(yhat, s)
            ax.plot(xs, ys, color="red", linewidth=1.0)
        _label_top_n(ax, yhat, s, scores=self.scaled_residuals, n=label_n)
        ax.set_xlabel("Fitted")
        ax.set_ylabel(r"$\sqrt{|\mathrm{Std.\ Residuals}|}$")
        ax.set_title("Scale-Location")

    def plot_qq_ranef(
        self, figsize=None,
        *, level: float = 0.95, strip: bool = True,
    ):
        """qqmath of BLUPs with conditional-variance bars (Bates Fig. 1.12).

        Pythonic ``qqmath(ranef(., condVar=TRUE), strip=...)``. BLUPs on the
        x-axis at y = Φ⁻¹((i−0.5)/n) (Hazen plotting position, matches
        lme4); horizontal bars of half-width Φ⁻¹((1+level)/2)·SE (default
        95%); vertical line at x=0. ``strip=False`` suppresses per-panel
        titles.
        """
        from scipy.stats import norm
        z = float(norm.ppf(0.5 + level / 2))
        panels = []
        for key, _levels, cnames, b_mat, se_mat in self._ranef():
            for j, cname in enumerate(cnames):
                panels.append((f"{key}: {cname}", b_mat[:, j], se_mat[:, j]))
        n_panels = len(panels)
        if figsize is None:
            figsize = (3.2 * n_panels, 3.0)
        fig, axes = plt.subplots(1, n_panels, figsize=figsize, squeeze=False)
        axes = axes.ravel()
        for ax, (title, b, se) in zip(axes, panels):
            order = np.argsort(b)
            b_s = b[order]
            se_s = se[order]
            n = len(b_s)
            q = norm.ppf((np.arange(1, n + 1) - 0.5) / n)
            ax.grid(True, color="lightgray", linewidth=0.4)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.errorbar(
                b_s, q, xerr=z * se_s, fmt="o", color="black",
                ecolor="black", markersize=3, linewidth=0.8, capsize=0,
            )
            ax.set_ylabel("Standard normal quantiles")
            ax.set_title(title if strip else "")
        fig.tight_layout()

    def plot_ranef(
        self, figsize=None,
        *, level: float = 0.95, strip: bool = True,
    ):
        """Caterpillar plot — BLUP ± Φ⁻¹((1+level)/2)·SE per level, sorted.

        Pythonic ``dotplot(ranef(., condVar=TRUE))``: defaults to 95%
        prediction intervals to match lme4. ``strip=False`` suppresses
        per-panel titles (Bates Fig. 1.5 convention).
        """
        from scipy.stats import norm
        z = float(norm.ppf(0.5 + level / 2))
        panels = []
        for key, levels, cnames, b_mat, se_mat in self._ranef():
            for j, cname in enumerate(cnames):
                panels.append(
                    (f"{key}: {cname}", b_mat[:, j], se_mat[:, j], levels)
                )
        n_panels = len(panels)
        if figsize is None:
            max_levels = max(len(p[3]) for p in panels)
            height = max(3.0, min(0.18 * max_levels, 12.0))
            figsize = (3.2 * n_panels, height)
        fig, axes = plt.subplots(1, n_panels, figsize=figsize, squeeze=False)
        axes = axes.ravel()
        for ax, (title, b, se, levels) in zip(axes, panels):
            order = np.argsort(b)
            b_sorted = b[order]
            se_sorted = se[order]
            labels_sorted = [str(levels[i]) for i in order]
            n = len(b)
            y_pos = np.arange(n)
            for y in y_pos:
                ax.axhline(y, color="lightgray", linewidth=0.4, zorder=0)
            ax.errorbar(
                b_sorted, y_pos, xerr=z * se_sorted,
                fmt="o", color="black", ecolor="black",
                markersize=3, capsize=0, linewidth=0.8,
            )
            ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
            ax.set_yticks(y_pos)
            if n <= 30:
                ax.set_yticklabels(labels_sorted, fontsize=7)
            else:
                ax.set_yticklabels([])
            ax.set_xlabel("Random Effect")
            ax.set_title(title if strip else "")
        fig.tight_layout()

    def plot(self, figsize=None, smooth=True, label_n=3):
        """4-panel diagnostic: Residuals, Q-Q residuals, Scale-Location, Q-Q BLUPs."""
        if figsize is None:
            figsize = (10, 8)
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        self.plot_residuals(ax=axes[0, 0], smooth=smooth, label_n=label_n)
        self.plot_qq(ax=axes[0, 1], label_n=label_n)
        self.plot_scale_location(ax=axes[1, 0], smooth=smooth, label_n=label_n)
        pooled = self._pooled_std_blups()
        if len(pooled) >= 4:
            _qq_plot(
                axes[1, 1], pooled, label_n=label_n,
                ylabel="Standardized BLUPs (pooled)",
                title="Random-Effects Q-Q",
            )
        else:
            axes[1, 1].set_title("Random-Effects Q-Q (n too small)")
        fig.tight_layout()


def _resolve_transform(t):
    """Map a ``transform=`` argument to a (forward-fn, title-format) pair."""
    if t is None:
        return (lambda x: np.asarray(x)), "{}"
    if callable(t):
        return t, "{}"
    if t == "log":
        return np.log, "log({})"
    if t in ("square", "sq"):
        return np.square, "{}²"
    raise ValueError(f"unknown transform {t!r}; use 'log', 'square', or a callable")


def _invert_zeta(
    vals: np.ndarray, zetas: np.ndarray, target: float,
    *, fallback: float = float("nan"),
) -> float:
    """Linearly interpolate the ζ-curve to find where ζ(v) = target.

    Returns ``fallback`` if ``target`` falls outside the observed ζ range —
    callers pass 0 for variance-component SDs (natural lower bound; matches
    lme4 when the profile flattens to an asymptote above the threshold) and
    NaN for unbounded parameters. Sorts by ζ first so the interpolation
    works even when the curve isn't evaluated on a monotone-in-v grid.
    """
    if target < np.nanmin(zetas) or target > np.nanmax(zetas):
        return fallback
    order = np.argsort(zetas)
    return float(np.interp(target, zetas[order], vals[order]))


class Profile:
    """Profile-likelihood output from :meth:`lme.profile`.

    Attributes
    ----------
    data : dict[str, polars.DataFrame]
        Per-parameter table with columns ``value`` and ``zeta``. Keys are
        ``.sig01``, ``.sig02``, … for variance-component SDs, ``.sigma``
        for the residual SD, and the R-canonical fixed-effect names
        (``(Intercept)``, ``MachineB``, …).
    estimate : dict[str, float]
        MLE for each profiled parameter, keyed the same way.
    """

    def __init__(self, data: dict[str, pl.DataFrame], estimate: dict[str, float]):
        self.data = data
        self.estimate = estimate

    def confint(self, level: float = 0.95) -> pl.DataFrame:
        """Profile-based confidence intervals at ``level`` (default 95%).

        Inverts each ζ-curve at ±Φ⁻¹((1+level)/2). For variance-component
        SDs (``.sig01``, ``.sig02``, …, ``.sigma``) the lower bound clips
        to 0 when the profile flattens to an asymptote above the threshold
        (matches lme4; see book Fig. 1.8). Unbounded parameters return
        ``NaN`` if the curve doesn't cross the threshold within the grid.
        """
        from scipy.stats import norm

        z = float(norm.ppf(0.5 + level / 2))
        lo_lbl = f"{100 * (1 - level) / 2:.1f}%"
        hi_lbl = f"{100 * (0.5 + level / 2):.1f}%"
        names: list[str] = []
        lo: list[float] = []
        hi: list[float] = []
        for name, df in self.data.items():
            v = df["value"].to_numpy()
            s = df["zeta"].to_numpy()
            names.append(name)
            lo_fb = 0.0 if name.startswith(".sig") else float("nan")
            lo.append(_invert_zeta(v, s, -z, fallback=lo_fb))
            hi.append(_invert_zeta(v, s, +z))
        return pl.DataFrame({"parameter": names, lo_lbl: lo, hi_lbl: hi})

    def plot(
        self, absolute: bool = False, figsize: tuple[float, float] | None = None,
        levels: tuple[float, ...] = (0.50, 0.80, 0.90, 0.95, 0.99),
        *,
        which: str | list[str] | None = None,
        transform: str | "Callable[[np.ndarray], np.ndarray]" | None = None,
        ax=None,
    ):
        """Profile zeta plot — the Pythonic replacement for R's
        ``xyplot(profile(...))``. One subplot per parameter; vertical
        gray lines mark the CI cutoffs for each level in ``levels``.

        With ``absolute=True`` plots ``|ζ|`` (matches book Fig. 1.6).

        ``which`` restricts to one parameter (str) or a subset (list).
        ``transform`` re-scales the x-axis: ``"log"`` for log(v),
        ``"square"`` for v², or any callable. CI cutoff verticals are
        forward-transformed too.

        Pass ``ax`` to draw into a pre-existing Axes (requires ``which`` to
        resolve to a single parameter). Useful for Bates Fig. 1.7-style
        layouts::

            fig, axes = plt.subplots(1, 3, sharey=True)
            pr.plot(which=".sigma", transform="log",    ax=axes[0])
            pr.plot(which=".sigma",                     ax=axes[1])
            pr.plot(which=".sigma", transform="square", ax=axes[2])
        """
        import matplotlib.pyplot as plt
        from scipy.stats import norm

        if which is None:
            names = list(self.data.keys())
        elif isinstance(which, str):
            names = [which]
        else:
            names = list(which)
        unknown = [n for n in names if n not in self.data]
        if unknown:
            raise KeyError(
                f"unknown parameter(s) {unknown!r}; available: {list(self.data)!r}"
            )
        if ax is not None and len(names) != 1:
            raise ValueError("ax= requires a single parameter via which='...'")

        fwd, title_fmt = _resolve_transform(transform)

        if ax is not None:
            axes = [ax]
            fig = ax.figure
        else:
            n = len(names)
            fig, axes_obj = plt.subplots(
                1, n, figsize=figsize or (3.2 * n, 3.0), sharey=False,
            )
            axes = [axes_obj] if n == 1 else list(axes_obj)

        for ax_i, name in zip(axes, names):
            df = self.data[name]
            v = df["value"].to_numpy()
            s = df["zeta"].to_numpy()
            x = fwd(v)
            y = np.abs(s) if absolute else s
            ax_i.plot(x, y, "o-", ms=3, lw=1)
            if not absolute:
                ax_i.axhline(0, color="k", lw=0.4)
            lo_fb = 0.0 if name.startswith(".sig") else float("nan")
            for lvl in levels:
                z = float(norm.ppf(0.5 + lvl / 2))
                for tgt in (-z, z):
                    fb = lo_fb if tgt < 0 else float("nan")
                    v_at = _invert_zeta(v, s, tgt, fallback=fb)
                    if np.isfinite(v_at):
                        x_at = fwd(np.asarray(v_at)).item()
                        if np.isfinite(x_at):
                            ax_i.axvline(x_at, color="gray", alpha=0.4, lw=0.5)
            ax_i.set_title(title_fmt.format(name))
            ax_i.set_xlabel(name)
        if ax is None:
            axes[0].set_ylabel("|ζ|" if absolute else "ζ")
            fig.tight_layout()
        return fig

    def plot_density(
        self, npts: int = 201, upper: float = 0.999,
        figsize: tuple[float, float] | None = None,
    ):
        """Profile-implied density plot — Pythonic ``densityplot(profile(...))``.

        For each parameter, plots φ(ζ(v))·|dζ/dv| against v: the Jacobian
        transform of N(0,1) in ζ to a density on the parameter scale.
        ζ(v) is interpolated with a PCHIP spline (monotone-preserving) and
        differentiated analytically. The x-range is restricted to ζ within
        ±Φ⁻¹(``upper``); for variance-component SDs the lower bound is
        clipped to 0.
        """
        import matplotlib.pyplot as plt
        from scipy.interpolate import PchipInterpolator
        from scipy.stats import norm

        names = list(self.data.keys())
        n = len(names)
        fig, axes = plt.subplots(
            1, n, figsize=figsize or (3.2 * n, 3.0), sharey=False,
        )
        if n == 1:
            axes = [axes]

        z_max = float(norm.ppf(upper))
        for ax, name in zip(axes, names):
            df = self.data[name]
            v = df["value"].to_numpy()
            s = df["zeta"].to_numpy()
            order = np.argsort(v)
            v_s, s_s = v[order], s[order]
            spl = PchipInterpolator(v_s, s_s, extrapolate=True)
            lo_fb = 0.0 if name.startswith(".sig") else float("nan")
            v_lo = _invert_zeta(v, s, -z_max, fallback=lo_fb)
            v_hi = _invert_zeta(v, s, +z_max)
            if not np.isfinite(v_lo):
                v_lo = float(v_s[0])
            if not np.isfinite(v_hi):
                v_hi = float(v_s[-1])
            grid = np.linspace(v_lo, v_hi, npts)
            zeta_g = spl(grid)
            dz_dv = spl.derivative()(grid)
            density = norm.pdf(zeta_g) * np.abs(dz_dv)
            ax.plot(grid, density, lw=1)
            ax.set_title(name)
            ax.set_xlabel(name)
        axes[0].set_ylabel("density")
        fig.tight_layout()
        return fig

    def __repr__(self) -> str:
        return f"Profile({list(self.data)})"
