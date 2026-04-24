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

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.optimize import minimize

from .formula import materialize_bars
from .utils import prepare_design

__all__ = ["lme"]


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
    data : pandas.DataFrame
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
    bhat, se_bhat, t_values : pandas.DataFrame
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
    AIC, BIC : float
    df_resid : int
        ``n - npar`` (matches lme4's printed ``df.resid``).
    """

    def __init__(self, formula: str, data: pd.DataFrame, REML: bool = True):
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
        y = d.y.to_numpy(dtype=float)
        X = X_df.to_numpy(dtype=float)
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
        template = re.Lambdat
        nz_mask = template > 0
        theta_pos = template[nz_mask] - 1  # 0-indexed θ position per nz cell
        eye_q = np.eye(q)
        XtX = X.T @ X
        Xty = X.T @ y
        yty = float(y @ y)
        log2pi = float(np.log(2.0 * np.pi))

        def build_Lt(theta: np.ndarray) -> np.ndarray:
            out = np.zeros(template.shape, dtype=float)
            out[nz_mask] = theta[theta_pos]
            return out

        def objective(theta: np.ndarray) -> float:
            Lt = build_Lt(theta)
            ZL = Z @ Lt.T                    # n×q,  Z @ Λ
            M = ZL.T @ ZL + eye_q
            try:
                Lz = np.linalg.cholesky(M)
            except np.linalg.LinAlgError:
                return 1e15
            cu = solve_triangular(Lz, ZL.T @ y, lower=True)
            RZX = solve_triangular(Lz, ZL.T @ X, lower=True)
            XtX_eff = XtX - RZX.T @ RZX
            try:
                Rx = np.linalg.cholesky(XtX_eff)
            except np.linalg.LinAlgError:
                return 1e15
            cb = solve_triangular(Rx, Xty - RZX.T @ cu, lower=True)
            rss = yty - float(cu @ cu) - float(cb @ cb)
            if rss <= 0:
                return 1e15
            log_det_Lz = float(np.log(np.diag(Lz)).sum())
            if REML:
                log_det_Rx = float(np.log(np.diag(Rx)).sum())
                df = n - p
                return (
                    2.0 * log_det_Lz + 2.0 * log_det_Rx
                    + df * (1.0 + log2pi + np.log(rss / df))
                )
            return 2.0 * log_det_Lz + n * (1.0 + log2pi + np.log(rss / n))

        diag_set = set(_theta_diag_idx(bar_sizes))
        bounds = [
            (0.0, None) if i in diag_set else (None, None)
            for i in range(len(re.theta))
        ]

        theta0 = re.theta.astype(float).copy()
        res = minimize(
            objective, theta0, method="L-BFGS-B", bounds=bounds,
            options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 1000},
        )
        theta_hat = res.x
        self.theta = theta_hat
        self._optim = res

        # ------------- recover β̂, σ̂, SE(β̂) at the optimum ------------------
        Lt = build_Lt(theta_hat)
        ZL = Z @ Lt.T
        M = ZL.T @ ZL + eye_q
        Lz = np.linalg.cholesky(M)
        cu = solve_triangular(Lz, ZL.T @ y, lower=True)
        RZX = solve_triangular(Lz, ZL.T @ X, lower=True)
        XtX_eff = XtX - RZX.T @ RZX
        Rx = np.linalg.cholesky(XtX_eff)
        cb = solve_triangular(Rx, Xty - RZX.T @ cu, lower=True)
        beta = solve_triangular(Rx.T, cb, lower=False)
        rss = yty - float(cu @ cu) - float(cb @ cb)
        # spherical random-effect coefficients (kept for diagnostics)
        self._u = solve_triangular(Lz.T, cu - RZX @ beta, lower=False)

        sigma2 = rss / (n - p) if REML else rss / n
        sigma = float(np.sqrt(sigma2))
        self.sigma = sigma
        self.sigma_squared = sigma2

        # Var(β̂) = σ̂² (XᵀX_eff)⁻¹ = σ̂² R_x⁻ᵀ R_x⁻¹
        Rx_inv = solve_triangular(Rx, np.eye(p), lower=True)
        var_beta = sigma2 * (Rx_inv ** 2).sum(axis=0)
        se_beta = np.sqrt(var_beta)

        self.bhat = pd.DataFrame(
            beta.reshape(1, -1), columns=self.column_names, index=["Estimate"],
        )
        self.se_bhat = pd.DataFrame(
            se_beta.reshape(1, -1), columns=self.column_names, index=["Std. Error"],
        )
        self.t_values = pd.DataFrame(
            (beta / se_beta).reshape(1, -1),
            columns=self.column_names, index=["t value"],
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
            self.AIC = self.deviance + 2.0 * self.npar
            self.BIC = self.deviance + np.log(n) * self.npar

    def __repr__(self) -> str:
        kind = "REML" if self.REML else "ML"
        out = [f"lme[{kind}]: {self.formula}"]
        out.append("")
        out.append("Random effects:")
        for key, sd in self.sd_re.items():
            sds = ", ".join(f"{v:.4g}" for v in sd)
            out.append(f"  {key:>22s}: SD=[{sds}]")
            corr = self.corr_re.get(key)
            if corr is not None and corr.shape[0] > 1:
                i, j = np.triu_indices(corr.shape[0], k=1)
                cs = ", ".join(f"{corr[a, b]:.3f}" for a, b in zip(i, j))
                out.append(f"  {'corr':>22s}: [{cs}]")
        out.append(f"  {'Residual':>22s}: SD={self.sigma:.4g}")
        out.append("")
        out.append("Fixed effects:")
        out.append(self.bhat.to_string())
        return "\n".join(out)

    def __str__(self) -> str:
        return self.__repr__()
