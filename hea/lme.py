"""Linear mixed-effects model — lme4-style profiled deviance.

Built on hea.formula's ``parse → expand → materialize / materialize_bars``
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

from .formula import _eval_atom, materialize_bars
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

        # Sum any `offset(...)` atoms from the formula, then fit on
        # ``y_solve = y - offset`` (R's lme/lmer convention). β̂, û and the
        # variance components are all unchanged by the offset; only the
        # fitted/residual scale shifts. ``self.y`` keeps the original
        # response so plots and diagnostics show the user's data.
        off = np.zeros(n)
        for off_node in d.expanded.offsets:
            off = off + _eval_atom(off_node, d.data).values.flatten().astype(float)
        self._offset = off
        y_solve = y - off

        self.data = d.data
        self.X = X_df
        self.y = y
        self._y_solve = y_solve
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
        Xty = X.T @ y_solve
        yty = float(y_solve @ y_solve)
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

        # Use the offset-stripped response so this final β̂/û recompute is
        # consistent with the cached Xty/yty the optimizer ran on.
        ZLty = np.asarray(ZL.T @ y_solve).ravel()
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

        # Fitted values ŷ = Xβ̂ + Z Λ û + offset (response scale).
        # Residuals = y − ŷ = y_solve − Xβ̂ − Z Λ û (offset cancels).
        self.fitted = np.einsum("ij,j->i", X, beta) + ZL @ self._u + off
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
        self.fixef = self.bhat                            # R-canonical alias
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
        ``Lz Lzᵀ = M`` means ``|M| = |Lz|²``. ``y`` here is offset-stripped
        (``y_solve``); cached ``Xty/yty`` are built from ``y_solve`` to match."""
        y = self._y_solve if y is None else y
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
        # F.logdet() goes through a slow Python wrapper (~210 µs); for the LLᵀ
        # factor cholmod returns by default, log|M| = 2·Σ log diag(L). Bit-
        # identical, ~20× faster.
        log_det_Lz = float(np.log(F.L.diagonal()).sum())
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

    def _post_refit_state(
        self, theta: np.ndarray, *,
        sigma_fix: float | None = None,
        y: np.ndarray | None = None, X: np.ndarray | None = None,
        XtX: np.ndarray | None = None, Xty: np.ndarray | None = None,
        yty: float | None = None,
    ) -> tuple[float, np.ndarray]:
        """At a fixed θ, recover (σ̂, β̂) at the just-found optimum.

        ``profile()`` calls this after each inner θ-refit so each grid
        point carries the full optimized state — needed for ``plot_pairs``
        traces. Cost is one sparse Cholesky + one tri-solve per call.
        ``sigma_fix=None`` profiles σ out (σ̂² = rss/n); pass it explicitly
        when σ was either pinned or optimized as a free variable upstream.
        """
        y_ = self._y_solve if y is None else y
        X_ = self.X.to_numpy().astype(float) if X is None else X
        XtX_ = self._XtX if XtX is None else XtX
        Xty_ = self._Xty if Xty is None else Xty
        yty_ = self._yty if yty is None else yty
        n = len(y_)
        Lt = self._build_Lt_sparse(theta)
        ZL = self._Z_sp @ Lt.T
        M = (ZL.T @ ZL + self._eye_q_sp).tocsc()
        self._chol_factor.factorize(M)
        F = self._chol_factor
        ZLty = np.asarray(ZL.T @ y_).ravel()
        M_inv_ZLty = F.solve(ZLty)
        cu_sq = float(np.einsum("i,i->", ZLty, M_inv_ZLty))
        if X_.shape[1] == 0:
            rss = yty_ - cu_sq
            beta = np.zeros(0)
        else:
            ZLtX = np.asarray(ZL.T @ X_)
            M_inv_ZLtX = F.solve(ZLtX)
            XtX_eff = XtX_ - np.einsum("ij,ik->jk", ZLtX, M_inv_ZLtX)
            Rx = np.linalg.cholesky(XtX_eff)
            rhs = Xty_ - np.einsum("ij,i->j", ZLtX, M_inv_ZLty)
            cb = solve_triangular(Rx, rhs, lower=True)
            beta = solve_triangular(Rx.T, cb, lower=False)
            rss = yty_ - cu_sq - float(np.einsum("i,i->", cb, cb))
        sigma = sigma_fix if sigma_fix is not None else float(np.sqrt(max(rss, 0.0) / n))
        return sigma, beta

    def _dev_with_beta_fixed(
        self, j: int, beta_j_tgt: float, theta_start: np.ndarray,
    ) -> tuple[float, np.ndarray, float, np.ndarray]:
        """Min ML deviance with β_j = ``beta_j_tgt``. Trick: subtract
        ``x_j · β_j_tgt`` from y and drop column j — the remaining fit has
        the same functional form. Returns ``(dev, θ̂, σ̂, β̂)`` where β̂ is
        in the full original column order with ``β_j = beta_j_tgt``."""
        X_full = self.X.to_numpy().astype(float)
        x_j = X_full[:, j]
        X_rest = np.delete(X_full, j, axis=1)
        # ``self._y_solve`` already has the offset removed; subtracting
        # x_j·β_j_tgt on top of that gives the correct adjusted response
        # for the offset-stripped sub-fit.
        y_adj = self._y_solve - x_j * beta_j_tgt
        XtX_rest = X_rest.T @ X_rest
        Xty_rest = X_rest.T @ y_adj
        yty_adj = float(y_adj @ y_adj)
        dev, theta_opt = self._refit_theta(
            lambda th: self._ml_deviance(
                th, y=y_adj, X=X_rest,
                XtX=XtX_rest, Xty=Xty_rest, yty=yty_adj,
            ),
            theta_start,
        )
        sigma_opt, beta_rest = self._post_refit_state(
            theta_opt, y=y_adj, X=X_rest,
            XtX=XtX_rest, Xty=Xty_rest, yty=yty_adj,
        )
        beta_opt = np.empty(self.p)
        beta_opt[j] = beta_j_tgt
        rest_idx = [k for k in range(self.p) if k != j]
        beta_opt[rest_idx] = beta_rest
        return dev, theta_opt, sigma_opt, beta_opt

    def _dev_with_sigma_fixed(
        self, sigma_tgt: float, theta_start: np.ndarray,
    ) -> tuple[float, np.ndarray, float, np.ndarray]:
        """Min ML deviance with σ = ``sigma_tgt`` (β profiles out).
        Returns ``(dev, θ̂, σ_tgt, β̂)``."""
        dev, theta_opt = self._refit_theta(
            lambda th: self._ml_deviance(th, sigma_fix=sigma_tgt),
            theta_start,
        )
        _, beta_opt = self._post_refit_state(theta_opt, sigma_fix=sigma_tgt)
        return dev, theta_opt, float(sigma_tgt), beta_opt

    def _dev_with_sd_fixed(
        self, slot_i: int, sd_tgt: float,
        sigma_start: float, theta_start: np.ndarray,
    ) -> tuple[float, np.ndarray, float, np.ndarray]:
        """Min ML deviance with σ_i = σ · θ[slot_i] pinned at ``sd_tgt``.

        Scalar-bar case: the bar has one θ entry, so pinning ``σ · θ[slot_i]
        = sd_tgt`` is a single nonlinear constraint. We re-parameterize as
        ``(σ, θ_rest)`` with ``θ[slot_i] = sd_tgt / σ`` and minimize jointly.
        Returns ``(dev, θ̂, σ̂, β̂)``."""
        other = [k for k in range(len(self._theta_bounds)) if k != slot_i]
        theta_rest0 = np.array([theta_start[k] for k in other])

        # Guard θ[slot_i] = sd_tgt/σ from blowing up when L-BFGS-B probes
        # very small σ — without this the implied θ becomes O(1e7) and
        # ``M = ΛᵀZᵀZΛ + I`` factorizes with rcond ≈ 1e-15. Cholmod warns
        # and the gradient gets noisy. Cap θ at 1e4 → cond(M) ≲ 1e8, well
        # away from Cholmod's near-singular threshold; the optimum lives
        # at θ_slot ≈ θ_hat ≪ 1e4 anyway, so the cap never binds.
        sigma_lb = max(1e-8, sd_tgt / 1e4)

        def obj(x):
            sigma = x[0]
            if sigma <= 0:
                return 1e15
            theta = np.zeros(len(self._theta_bounds))
            theta[slot_i] = sd_tgt / sigma
            for k, slot in enumerate(other):
                theta[slot] = x[1 + k]
            return self._ml_deviance(theta, sigma_fix=sigma)

        x0 = np.concatenate([[max(sigma_start, sigma_lb)], theta_rest0])
        bounds = [(sigma_lb, None)] + [self._theta_bounds[k] for k in other]
        res = minimize(
            obj, x0, method="L-BFGS-B", bounds=bounds,
            options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 1000},
        )
        # Reconstruct θ at the optimum for warm-start of neighboring points.
        theta_opt = np.zeros(len(self._theta_bounds))
        sigma_opt = float(res.x[0])
        theta_opt[slot_i] = sd_tgt / sigma_opt
        for k, slot in enumerate(other):
            theta_opt[slot] = res.x[1 + k]
        _, beta_opt = self._post_refit_state(theta_opt, sigma_fix=sigma_opt)
        return float(res.fun), theta_opt, sigma_opt, beta_opt

    def _step_adaptive(
        self, *, direction: int, v_start: float, initial_step: float,
        fit_at_v, theta_warm: np.ndarray, sigma_warm: float,
        d_hat: float, v_min: float = -np.inf, v_max: float = np.inf,
        zeta_cutoff: float = 4.0, target_dzeta: float = 0.5,
        max_steps: int = 25,
    ) -> list[tuple]:
        """One-direction adaptive ζ-stepper — matches R's profile.merMod.

        Steps from ``v_start`` along ``direction`` (±1), refits the
        constrained deviance at each step, and adapts the v-step size so
        |Δζ| ≈ ``target_dzeta`` per step. Stops when |ζ| ≥ ``zeta_cutoff``,
        v hits a bound, or ``max_steps`` is exhausted. Critically, this
        avoids the σ → 5·σ̂ extreme-grid points where ``M = ΛᵀZᵀZΛ + I``
        becomes Cholmod-near-singular (rcond ≈ 1e-15), which is what made
        the old fixed-grid profile diverge between Intel Macs and ARM.
        Returns ``(v, ζ, θ, σ, β)`` tuples in stepping order.
        """
        out: list[tuple] = []
        v_curr, zeta_curr, step = v_start, 0.0, float(initial_step)
        for _ in range(max_steps):
            v_try = v_curr + direction * step
            boundary_hit = False
            if v_try <= v_min:
                v_try = v_min + 1e-6 * abs(initial_step)
                boundary_hit = True
            elif v_try >= v_max:
                v_try = v_max - 1e-6 * abs(initial_step)
                boundary_hit = True
            d, theta_opt, sigma_opt, beta_opt = fit_at_v(v_try, theta_warm, sigma_warm)
            if not np.isfinite(d):
                step *= 0.5
                if step < 1e-6 * initial_step:
                    break
                continue
            zeta_try = direction * np.sqrt(max(0.0, d - d_hat))
            out.append((float(v_try), float(zeta_try), theta_opt, sigma_opt, beta_opt))
            theta_warm, sigma_warm = theta_opt, sigma_opt
            if abs(zeta_try) >= zeta_cutoff or boundary_hit:
                break
            dzeta = abs(zeta_try - zeta_curr)
            if dzeta > 1e-6:
                step = float(np.clip(step * (target_dzeta / dzeta), step / 4, step * 4))
            v_curr, zeta_curr = v_try, zeta_try
        return out

    def _profile_param_adaptive(
        self, *, fit_at_v, v_start: float,
        theta_start: np.ndarray, sigma_start: float, beta_start: np.ndarray,
        d_hat: float, initial_step: float,
        v_min: float = -np.inf, v_max: float = np.inf,
        zeta_cutoff: float = 4.0, max_steps_per_dir: int = 25,
    ) -> list[tuple]:
        """Profile one parameter in both ζ-directions + insert the MLE
        row. See :meth:`_step_adaptive`. Output order: most-negative ζ
        first → MLE → most-positive ζ last.
        """
        common = dict(
            initial_step=initial_step, fit_at_v=fit_at_v, d_hat=d_hat,
            v_min=v_min, v_max=v_max, zeta_cutoff=zeta_cutoff,
            max_steps=max_steps_per_dir,
        )
        pos = self._step_adaptive(
            direction=+1, v_start=v_start,
            theta_warm=theta_start.copy(), sigma_warm=sigma_start, **common,
        )
        neg = self._step_adaptive(
            direction=-1, v_start=v_start,
            theta_warm=theta_start.copy(), sigma_warm=sigma_start, **common,
        )
        mle = (float(v_start), 0.0, theta_start.copy(), float(sigma_start),
               beta_start.copy())
        return list(reversed(neg)) + [mle] + pos

    def profile(self, n_grid: int = 41) -> "Profile":
        """Compute profile-likelihood curves for σ_i, σ, and each β_j.

        Uses R's adaptive ζ-stepping (``profile.merMod``): from the MLE
        we step in v with step size adapted so |Δζ| ≈ 0.5 per step,
        stopping each direction when |ζ| ≥ 4 or v hits a bound. ``n_grid``
        is reinterpreted as the max-steps-per-direction cap; in practice
        most parameters terminate after 10–15 steps. The Profile rows
        thus have variable length, sorted by ζ.

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

        bar_keys = list(self.sd_re.keys())
        bar_labels = [f".sig{i + 1:02d}" for i in range(len(bar_keys))]
        slot_offsets = list(np.cumsum([0] + self._bar_sizes[:-1]))
        bar_slots = [int(s) for s in slot_offsets]
        # Column order, also used as the iteration order for profiled params.
        param_names: list[str] = bar_labels + [".sigma"] + list(self.column_names)

        estimate: dict[str, float] = {}
        for lbl, key in zip(bar_labels, bar_keys):
            estimate[lbl] = float(self.sd_re[key][0])
        estimate[".sigma"] = sigma_hat
        for j, name in enumerate(self.column_names):
            estimate[name] = float(self._beta[j])

        def _state_to_row(theta_opt, sigma_opt, beta_opt) -> dict[str, float]:
            """Map (θ̂, σ̂, β̂) at a grid point into the per-parameter row."""
            row: dict[str, float] = {}
            for lbl, slot in zip(bar_labels, bar_slots):
                row[lbl] = float(sigma_opt * theta_opt[slot])
            row[".sigma"] = float(sigma_opt)
            for j, name in enumerate(self.column_names):
                row[name] = float(beta_opt[j])
            return row

        # Adaptive ζ-stepping per parameter — see _step_adaptive. Each
        # call returns rows ordered most-negative-ζ → MLE → most-positive-ζ.
        rows_by_param: dict[str, list[dict[str, float]]] = {p: [] for p in param_names}
        zetas_by_param: dict[str, np.ndarray] = {}

        def _samples_to_storage(samples: list[tuple], lbl: str):
            zetas_by_param[lbl] = np.array([s[1] for s in samples])
            for s in samples:
                rows_by_param[lbl].append(_state_to_row(s[2], s[3], s[4]))

        # -- σ_i (one per scalar bar) ---------------------------------------
        for lbl, slot_i in zip(bar_labels, bar_slots):
            sd_i = estimate[lbl]
            samples = self._profile_param_adaptive(
                fit_at_v=lambda v, th_w, sg_w, _slot=slot_i:
                    self._dev_with_sd_fixed(_slot, v, sg_w, th_w),
                v_start=sd_i, theta_start=theta_hat,
                sigma_start=sigma_hat, beta_start=self._beta,
                d_hat=d_hat, initial_step=0.1 * max(sd_i, 1.0),
                v_min=0.0, max_steps_per_dir=n_grid,
            )
            _samples_to_storage(samples, lbl)

        # -- σ ----------------------------------------------------------------
        samples = self._profile_param_adaptive(
            fit_at_v=lambda v, th_w, sg_w:
                self._dev_with_sigma_fixed(v, th_w),
            v_start=sigma_hat, theta_start=theta_hat,
            sigma_start=sigma_hat, beta_start=self._beta,
            d_hat=d_hat, initial_step=0.1 * sigma_hat,
            v_min=0.0, max_steps_per_dir=n_grid,
        )
        _samples_to_storage(samples, ".sigma")

        # -- β_j --------------------------------------------------------------
        for j, name in enumerate(self.column_names):
            beta_j = estimate[name]
            se_j = float(self._se_beta[j])
            samples = self._profile_param_adaptive(
                fit_at_v=lambda v, th_w, sg_w, _j=j:
                    self._dev_with_beta_fixed(_j, v, th_w),
                v_start=beta_j, theta_start=theta_hat,
                sigma_start=sigma_hat, beta_start=self._beta,
                d_hat=d_hat, initial_step=max(se_j, 1e-3),
                max_steps_per_dir=n_grid,
            )
            _samples_to_storage(samples, name)

        data: dict[str, pl.DataFrame] = {}
        for p in param_names:
            cols: dict[str, list[float]] = {q: [r[q] for r in rows_by_param[p]] for q in param_names}
            cols["zeta"] = list(zetas_by_param[p])
            data[p] = pl.DataFrame(cols)

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

    @property
    def ranef(self) -> dict[str, pl.DataFrame]:
        """BLUPs per random-effect bar — lme4's ``ranef(m)`` shape.

        Returns one polars DataFrame per bar (keyed by bar name, e.g.
        ``"Subject"``, or ``"Subject.1"`` when the same grouping factor
        appears twice). First column carries the level labels under the
        grouping factor's name; remaining columns are the BLUPs, one per
        random-effect component (``(Intercept)``, slope names, …).
        """
        out: dict[str, pl.DataFrame] = {}
        for key, levels, cnames, b_mat, _se in self._ranef():
            gname = key
            if gname not in self.n_groups:
                base, _, tail = key.rpartition(".")
                if tail.isdigit() and base in self.n_groups:
                    gname = base
            cols: dict[str, list] = {gname: list(levels)}
            for j, cn in enumerate(cnames):
                cols[cn] = b_mat[:, j].tolist()
            out[key] = pl.DataFrame(cols)
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
    """Cubic-spline-interpolate the ζ-curve to find where ζ(v) = target.

    Matches R's ``confint(profile(...))`` which uses ``splines::interpSpline``
    on the ``ζ → v`` mapping — linear interpolation across two adjacent
    grid points loses noticeable curvature near ±z (visible as ~0.25
    units of error in the Dyestuff (Intercept) 99% bounds). Falls back to
    linear interp when there are too few points for a cubic.

    Returns ``fallback`` if ``target`` falls outside the observed ζ range —
    callers pass 0 for variance-component SDs (natural lower bound; matches
    lme4 when the profile flattens to an asymptote above the threshold) and
    NaN for unbounded parameters. Sorts by ζ first so the interpolation
    works even when the curve isn't evaluated on a monotone-in-v grid.
    """
    if target < np.nanmin(zetas) or target > np.nanmax(zetas):
        return fallback
    if len(vals) < 4:
        order = np.argsort(zetas)
        return float(np.interp(target, zetas[order], vals[order]))
    # Match R: fit a forward natural cubic spline ζ = f(v), then numerically
    # invert. The forward direction is monotonic and smooth even at .sig
    # boundary corners (where ζ at v=0 is a finite asymptote, not ±∞), so
    # the spline isn't pulled into the oscillations that fitting v(ζ) on
    # the same data triggers. R uses splines::interpSpline + backSpline.
    from scipy.interpolate import CubicSpline
    from scipy.optimize import brentq
    v_order = np.argsort(vals)
    v_sorted, z_sorted = vals[v_order], zetas[v_order]
    fwd = CubicSpline(v_sorted, z_sorted, bc_type="natural", extrapolate=False)
    # Find the bracket: target lies between two consecutive ζ-knots.
    diffs = z_sorted - target
    sign_change = np.where(diffs[:-1] * diffs[1:] <= 0)[0]
    if len(sign_change) == 0:
        return float(np.interp(target, np.sort(zetas), vals[np.argsort(zetas)]))
    i = int(sign_change[0])
    return float(brentq(lambda v: float(fwd(v)) - target, v_sorted[i], v_sorted[i + 1]))


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
            v = df[name].to_numpy()
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
            v = df[name].to_numpy()
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
            v = df[name].to_numpy()
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

    def plot_pairs(
        self, *,
        which: list[str] | None = None,
        transform: str | None = None,
        levels: tuple[float, ...] = (0.50, 0.80, 0.90, 0.95, 0.99),
        figsize: tuple[float, float] | None = None,
    ):
        """Profile pairs plot — port of lme4's ``splom(profile(...))`` (Fig 2.6).

        Lower triangle: bivariate ζ-deviance contours and the two profile
        traces in *ζ-coordinates* ``(ζⱼ, ζᵢ)``, axes clamped to ±max(level).
        Upper triangle: same contours/traces mapped through each
        parameter's backward spline ``v(ζ)`` into *original* parameter
        space ``(vⱼ, vᵢ)``. Diagonal: parameter labels.

        Pass ``transform="log"`` to reproduce Bates Fig 2.7 — the
        equivalent of R's ``splom(log(profile(fm)))``. ζ is invariant
        under monotone reparameterization, so only the upper-triangle
        v-space panels and the diagonal/axis labels change; log is
        applied to variance-component SDs (``.sig*``, ``.sigma``) only,
        leaving fixed-effect parameters on their natural scale.

        The contour at confidence level α is built (Bates, lme4 § 1.5)
        from four anchor points where the level-α curve crosses the
        profile traces. A periodic cubic spline through ``(θ_mean,
        θ_diff)`` gives an angular parameterization; the curve closes
        smoothly via ``(ζᵢ, ζⱼ) = lev · (cos(θ_mean − θ_diff/2),
        cos(θ_mean + θ_diff/2))``. Contour levels default to the lme4
        defaults: √χ²₂(α) for α ∈ {0.50, 0.80, 0.90, 0.95, 0.99}.
        """
        import matplotlib.pyplot as plt
        from scipy.interpolate import CubicSpline, PchipInterpolator
        from scipy.stats import chi2

        if which is None:
            names = list(self.data.keys())
        else:
            names = list(which)
            unknown = [n for n in names if n not in self.data]
            if unknown:
                raise KeyError(
                    f"unknown parameter(s) {unknown!r}; available: {list(self.data)!r}"
                )
        n = len(names)
        if n < 2:
            raise ValueError("plot_pairs needs at least 2 parameters")

        zeta_levels = np.sqrt(chi2.ppf(np.asarray(levels), 2))
        mlev = float(zeta_levels.max())

        # Per-parameter v-transform. Matches R's log.thpr / logProf:
        # log applies to .sig* and .sigma only; fixed effects keep
        # natural scale.
        if transform is None:
            tx_fn: dict[str, "Callable[[np.ndarray], np.ndarray]"] = {
                name: (lambda x: np.asarray(x)) for name in names
            }
            tx_label = {name: name for name in names}
        elif transform == "log":
            tx_fn = {
                name: (np.log if name.startswith(".sig") else (lambda x: np.asarray(x)))
                for name in names
            }
            tx_label = {
                name: (f"log({name})" if name.startswith(".sig") else name)
                for name in names
            }
        else:
            raise ValueError(
                f"unknown transform {transform!r}; use 'log' or None"
            )

        fwd: dict[str, PchipInterpolator] = {}
        bwd: dict[str, PchipInterpolator] = {}
        v_lim: dict[str, tuple[float, float]] = {}
        for name in names:
            df = self.data[name]
            v = df[name].to_numpy()
            s = df["zeta"].to_numpy()
            order = np.argsort(v)
            v_s, s_s = v[order], s[order]
            fwd[name] = PchipInterpolator(v_s, s_s, extrapolate=False)
            order_z = np.argsort(s_s)
            v_t = tx_fn[name](v_s)
            bwd[name] = PchipInterpolator(s_s[order_z], v_t[order_z], extrapolate=False)
            # v-axis limits — match R splom.thpr: backward-spline at ±mlev,
            # then clip to the profile grid range so we never advertise an
            # axis range we don't actually have data for.
            v_lo = bwd[name](-mlev)
            v_hi = bwd[name](+mlev)
            v_t_min, v_t_max = float(v_t.min()), float(v_t.max())
            v_lo = v_t_min if not np.isfinite(v_lo) else float(max(v_lo, v_t_min))
            v_hi = v_t_max if not np.isfinite(v_hi) else float(min(v_hi, v_t_max))
            v_lim[name] = (v_lo, v_hi)

        def _trace_zeta(prof_name: str, other_name: str) -> tuple[np.ndarray, np.ndarray]:
            """Return (ζ_prof, ζ_other) along the trace of profile(prof_name).

            ζ_prof is read directly from the ``zeta`` column; ζ_other is
            obtained by sending the optimum v_other through the forward
            spline of ``other_name`` and dropping NaNs (off-grid points).
            """
            df = self.data[prof_name]
            zp = df["zeta"].to_numpy()
            zo = fwd[other_name](df[other_name].to_numpy())
            keep = ~np.isnan(zo)
            return zp[keep], zo[keep]

        def _sacos(x):
            return np.arccos(np.clip(x, -0.999, 0.999))

        def _ad(xc, yc):
            a = (xc + yc) / 2.0
            d = xc - yc
            return np.sign(d) * a, np.abs(d)

        def _contour_pts(sij, sji, level: float, nseg: int = 101):
            """Generate one bivariate-ζ contour at radius ``level``.

            Returns (n+1, 2) array of (ζ_i, ζ_j) points on the closed curve;
            ``None`` if any anchor falls outside the trace splines' domain.
            """
            try:
                yc1 = _sacos(float(sij(+level)) / level)
                xc2 = _sacos(float(sji(+level)) / level)
                yc3 = _sacos(float(sij(-level)) / level)
                xc4 = _sacos(float(sji(-level)) / level)
            except Exception:
                return None
            if any(np.isnan(v) for v in (yc1, xc2, yc3, xc4)):
                return None
            xs = np.empty(4)
            ys = np.empty(4)
            xs[0], ys[0] = _ad(0.0, yc1)
            xs[1], ys[1] = _ad(xc2, 0.0)
            xs[2], ys[2] = _ad(np.pi, yc3)
            xs[3], ys[3] = _ad(xc4, np.pi)
            order = np.argsort(xs)
            xs_s = xs[order]
            ys_s = ys[order]
            # Close the ring for ``bc_type='periodic'``: append the first
            # knot shifted by one period, with the same y value, so that
            # ``y[0] == y[-1]`` (CubicSpline's periodic precondition).
            xs_p = np.concatenate([xs_s, [xs_s[0] + 2 * np.pi]])
            ys_p = np.concatenate([ys_s, [ys_s[0]]])
            try:
                spl = CubicSpline(xs_p, ys_p, bc_type="periodic")
            except ValueError:
                return None
            theta = np.linspace(xs_s[0], xs_s[0] + 2 * np.pi, nseg + 1)
            tdiff = spl(theta)
            # tauij in lme4:::cont returns (col1, col2) where col1 = lev *
            # cos(θ_mean - θ_diff/2) = ζ_j and col2 = lev * cos(θ_mean +
            # θ_diff/2) = ζ_i. Verify at anchor 1 (θ_m = -θ/2, θ_d = θ):
            # col1 = lev·cos(-θ) = sij(+lev) (the j-coord), col2 = lev =
            # zeta_i at +lev. Stack as (ζ_i, ζ_j) to match downstream.
            zj = level * np.cos(theta - tdiff / 2.0)
            zi = level * np.cos(theta + tdiff / 2.0)
            return np.column_stack([zi, zj])

        # Pre-compute contour data for each (i, j) pair, i < j.
        contours: dict[tuple[int, int], dict] = {}
        for jj in range(1, n):
            for ii in range(jj):
                ni, nj = names[ii], names[jj]
                zi_i, zj_i = _trace_zeta(ni, nj)   # along trace of i
                zj_j, zi_j = _trace_zeta(nj, ni)   # along trace of j
                if len(zi_i) < 4 or len(zj_j) < 4:
                    contours[(ii, jj)] = {}
                    continue
                o_i = np.argsort(zi_i)
                o_j = np.argsort(zj_j)
                # Trace splines extrapolate, matching R's interpSpline + predy
                # in lme4:::cont — splom always renders all length(levels)
                # contours, even when one parameter's profile range stops
                # short of mlev = √χ²₂(0.99) (e.g. an Intercept that's
                # orthogonal to the variance components).
                sij = PchipInterpolator(zi_i[o_i], zj_i[o_i], extrapolate=True)
                sji = PchipInterpolator(zj_j[o_j], zi_j[o_j], extrapolate=True)
                pts_per_level = []
                for lev in zeta_levels:
                    pts = _contour_pts(sij, sji, float(lev))
                    pts_per_level.append(pts)
                contours[(ii, jj)] = dict(
                    sij=sij, sji=sji,
                    trace_i=(zi_i[o_i], zj_i[o_i]),
                    trace_j=(zi_j[o_j], zj_j[o_j]),
                    pts=pts_per_level,
                )

        fig, axes = plt.subplots(
            n, n, figsize=figsize or (2.4 * n, 2.4 * n), squeeze=False,
        )

        def _draw_zeta_panel(ax, info, x_is_i: bool):
            """ζ-space panel. ``x_is_i`` controls which axis is ζ_i."""
            zi_grid_i, zj_at_i = info["trace_i"]
            zi_at_j, zj_grid_j = info["trace_j"]
            if x_is_i:
                ax.plot(zi_grid_i, zj_at_i, "-", lw=0.5, color="black")
                ax.plot(zi_at_j, zj_grid_j, "-", lw=0.5, color="black")
            else:
                ax.plot(zj_at_i, zi_grid_i, "-", lw=0.5, color="black")
                ax.plot(zj_grid_j, zi_at_j, "-", lw=0.5, color="black")
            for pts in info["pts"]:
                if pts is None:
                    continue
                if x_is_i:
                    ax.plot(pts[:, 0], pts[:, 1], "-", lw=0.5, color="black")
                else:
                    ax.plot(pts[:, 1], pts[:, 0], "-", lw=0.5, color="black")
            ax.set_xlim(-1.05 * mlev, 1.05 * mlev)
            ax.set_ylim(-1.05 * mlev, 1.05 * mlev)

        def _draw_v_panel(ax, info, ni, nj, x_is_i: bool):
            """v-space panel. Maps each ζ-coordinate through its backward
            spline to recover v.  ``x_is_i`` controls which axis is v_i."""
            zi_grid_i, zj_at_i = info["trace_i"]
            zi_at_j, zj_grid_j = info["trace_j"]
            vi_i = bwd[ni](zi_grid_i)
            vj_i = bwd[nj](zj_at_i)
            vi_j = bwd[ni](zi_at_j)
            vj_j = bwd[nj](zj_grid_j)
            if x_is_i:
                ax.plot(vi_i, vj_i, "-", lw=0.5, color="black")
                ax.plot(vi_j, vj_j, "-", lw=0.5, color="black")
            else:
                ax.plot(vj_i, vi_i, "-", lw=0.5, color="black")
                ax.plot(vj_j, vi_j, "-", lw=0.5, color="black")
            for pts in info["pts"]:
                if pts is None:
                    continue
                vc_i = bwd[ni](pts[:, 0])
                vc_j = bwd[nj](pts[:, 1])
                ok = ~(np.isnan(vc_i) | np.isnan(vc_j))
                if not ok.any():
                    continue
                if x_is_i:
                    ax.plot(vc_i[ok], vc_j[ok], "-", lw=0.5, color="black")
                else:
                    ax.plot(vc_j[ok], vc_i[ok], "-", lw=0.5, color="black")
            ax.set_xlim(*(v_lim[ni] if x_is_i else v_lim[nj]))
            ax.set_ylim(*(v_lim[nj] if x_is_i else v_lim[ni]))

        # Lattice-splom layout: origin at lower-left, so the parameter
        # at display row ``r`` (matplotlib top-down) is ``names[n-1-r]``
        # and at display column ``c`` is ``names[c]``. The diagonal runs
        # from bottom-left (.sig01) to top-right ((Intercept)).
        for r in range(n):
            for c in range(n):
                ax = axes[r, c]
                ax.tick_params(labelsize=8)
                ax.grid(True, color="lightgray", lw=0.3)
                vid_row = n - 1 - r
                vid_col = c
                if vid_row == vid_col:
                    ax.text(
                        0.5, 0.5, tx_label[names[vid_row]], ha="center", va="center",
                        transform=ax.transAxes, fontsize=12,
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.grid(False)
                    for s in ("top", "right", "bottom", "left"):
                        ax.spines[s].set_visible(True)
                    continue
                ii = min(vid_row, vid_col)
                jj = max(vid_row, vid_col)
                info = contours.get((ii, jj), {})
                if not info:
                    continue
                ni, nj = names[ii], names[jj]
                x_is_i = (vid_col == ii)
                # Lower triangle in display (closer to bottom-left,
                # vid_row < vid_col): ζ-space, per lme4 splom.
                # Upper triangle in display: v-space.
                if vid_row < vid_col:
                    _draw_zeta_panel(ax, info, x_is_i=x_is_i)
                else:
                    _draw_v_panel(ax, info, ni, nj, x_is_i=x_is_i)
                if c == 0:
                    ax.set_ylabel(tx_label[names[vid_row]])
                if r == n - 1:
                    ax.set_xlabel(tx_label[names[vid_col]])

        fig.tight_layout()
        return fig

    def __repr__(self) -> str:
        return f"Profile({list(self.data)})"
