import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import norm, t, f
from scipy.optimize import minimize
from formulae import design_matrices


class lme:
    def __init__(
        self,
        formula: str,
        data: pd.DataFrame,
        REML: bool = True,
        method: str = "Nelder-Mead",
    ):

        # meta
        self.formula = formula
        self.data = data
        self.REML = REML
        self.method = method

        # design matrix
        dm = design_matrices(formula, data)
        self.terms = dm.group.terms.keys()
        self.column_names = dm.common.terms.keys()

        self.Zs = [dm.group[key] for key in dm.group.terms.keys()]
        self.y = y = np.array(dm.response.design_vector.astype(float))
        self.X = X = np.array(dm.common.design_matrix)
        self.Z = np.hstack(self.Zs)
        self.XtX = X.T @ X
        self.Xty = X.T @ y

        self.n = len(self.y)
        self.p = self.X.shape[1]  # number of variables in the fixed effect
        self.qs = [
            Z.shape[1] for Z in self.Zs
        ]  # number of variables in each term of the random effect
        self.q = np.sum(self.qs)

        # model degree of freedom
        self.df_model = self.p - 1 if "Intercept" in self.column_names else self.p

        # residual degrees of freedom (n - p)
        self.df_residuals = (
            self.n - self.df_model - 1
            if "Intercept" in self.column_names
            else self.n - self.df_model
        )

        # fit model
        self.re, self.REML_criterion, self.convergence = self.fit(
            REML=REML, method=method
        )
        bhat = self.fe = self.compute_b(self.Xtr, self.ytr)
        self.bhat = pd.DataFrame(
            bhat.reshape(1, -1), columns=self.column_names, index=["Estimate"]
        )
        self.yhat = yhat = self.compute_mu(self.X, self.bhat.values.T)

        residuals = y.flatten() - yhat.flatten()
        self.residuals = pd.DataFrame(residuals[:, None], columns=["residuals"])
        self.rss = np.squeeze(residuals.T @ residuals)
        self.sigma_squared = self.rss / self.df_residuals
        self.sigma = np.sqrt(self.sigma_squared)

        self.XtXinv = self.compute_XtXinv()

        se_bhat, V_bhat = self.compute_se_bhat()
        self.V_bhat = V_bhat
        self.se_bhat = pd.DataFrame(
            se_bhat.reshape(1, -1), columns=self.column_names, index=["Std. Error"]
        )
        # compute confidence interval for β̂
        self.ci_bhat = self.compute_ci_bhat()

        # compute t values of model coefficients
        self.t_values = self.compute_t_values()

        # compute r2 and r2adjusted, aka coefficient of determination
        # aka percentage of variance explained. Noted that the formulae
        # are different for cases with and without intercept
        (
            self.tss,
            self.r_squared,
            self.r_squared_adjusted,
        ) = self.compute_goodness_of_fit()

        # compute log-likelihood
        self.loglike = self.compute_loglikelihood()

        # compute AIC (Akaike Information criterion): -2logL + 2p, p is the total number of parameters
        self.AIC = self.compute_AIC()

        # compute BIC (Bayes Information criterian): -2logL + p * log(n)
        self.BIC = self.compute_BIC()

    def compute_XtXinv(self):
        U, S, Vt = np.linalg.svd(self.XtX, full_matrices=False)
        XtXinv = Vt.T @ np.diag(1 / S) @ U.T
        return XtXinv

    def compute_se_bhat(self):
        V_bhat = self.sigma_squared * self.XtXinv
        se_bhat = np.sqrt(np.diag(V_bhat))[:, None]
        return se_bhat, V_bhat

    def compute_ci_bhat(self, alpha=0.05):

        se_bhat = self.se_bhat.values.T
        bhat = self.bhat.values.T
        ci_bhat = (
            t.ppf(1 - alpha / 2, self.df_residuals) * se_bhat * np.array([-1, 1]) + bhat
        )
        ci_bhat = pd.DataFrame(
            ci_bhat,
            index=self.column_names,
            columns=[f"CI[{alpha/2*100}%]", f"CI[{100-alpha/2*100}]%"],
        )
        return ci_bhat

    def compute_t_values(self):

        t_values = self.bhat.values / self.se_bhat.values

        return pd.DataFrame(t_values, columns=self.column_names, index=["t values"])

    def compute_goodness_of_fit(self):

        y = self.y

        if "Intercept" in self.column_names:
            tss = np.sum((y - y.mean()) ** 2)
            # Eq: r2 = 1 - RSS / TSS = 1 -  sum((ŷ - yi)**2) / sum((y - ȳ)**2)
            r_squared = (1 - self.rss / tss).squeeze()
            # Eq: r2adj = 1 - (1 - r2) * (n - 1) / df_residuals
            r_squared_adjusted = 1 - (1 - r_squared) * (self.n - 1) / (
                self.df_residuals
            )
        else:
            tss = np.sum(y**2)
            # Eq: r2 = 1 - RSS / TSS = 1 -  sum((ŷ - yi)**2) / sum((y)**2)
            r_squared = (1 - self.rss / tss).squeeze()
            # Eq: r2adj = 1 - (1 - r2) * n / df_residuals
            r_squared_adjusted = 1 - (1 - r_squared) * self.n / (self.df_residuals)

        return tss, r_squared, r_squared_adjusted

    def compute_loglikelihood(self):
        return np.squeeze(
            -0.5 * self.n * (np.log(self.rss / self.n) + np.log(2 * np.pi) + 1)
        )

    def compute_AIC(self):
        # add 1 to p to keep consistent with R
        # https://stackoverflow.com/q/37917437
        return -2 * self.loglike + 2 * (self.p + 1)

    def compute_BIC(self):
        # add 1 to p to keep consistent with R
        # https://stackoverflow.com/q/37917437
        return -2 * self.loglike + np.log(self.n) * (self.p + 1)

    def __repr__(self):

        docstring = f"""Formula: {self.formula}\n\n"""
        docstring += "Coefficients:\n"
        docstring += self.bhat.to_string()

        return docstring

    def __str__(self):
        return self.__repr__()

    def compute_phi(self, tau):
        return np.diag(
            np.hstack([np.repeat(tau[i], q_) for i, q_ in enumerate(self.qs)])
        )

    # multivariate normal vector with mean 0 and covariance Z @ phi @ Z.t + I * sigma**2
    def compute_e(self, phi, sigma):
        e = self.Z @ phi**2 @ self.Z.T + np.eye(self.n) * sigma**2
        return e

    # fixed effect
    def compute_b(self, X, y):
        return np.linalg.lstsq(X.T @ X, X.T @ y, rcond=True)[0]

    # fitted / predicted values
    def compute_mu(self, X, b):
        return X @ b

    # cholesky factor
    def compute_L(self, e):
        return np.linalg.cholesky(e)

    # random effect
    def compute_u(self):
        Sigma = self.e / self.sigma**2
        residuals = self.compute_residuals(self.X, self.y)
        re = np.hstack(
            [
                self.tau[i] ** 2
                * (self.Zs[i].T @ np.linalg.inv(Sigma) @ residuals / self.sigma**2)
                for i in range(len(self.qs))
            ]
        )
        return re

    def transform_Xy(self, X, y, L):
        # I don't really get this transformation
        # but without it will cause error w.r.t singular matrix
        # in m-clark's comment, it's transforming dependent linear model
        # into independent

        y = np.linalg.solve(L, self.y)
        X = np.linalg.solve(L, self.X)
        return X, y

    def compute_residuals(self, X, y):
        return y - X @ self.b

    def profiled_deviance(self, params, REML=False):

        """
        Modified from
        https://m-clark.github.io/docs/mixedModels/mixedModelML.html
        """

        n = self.n
        p = self.p

        self.tau = np.exp(params[:-1])
        self.sigma = np.exp(params[-1])

        self.phi = self.compute_phi(self.tau)

        self.e = self.compute_e(self.phi, self.sigma)
        self.L = self.compute_L(self.e)

        self.Xtr, self.ytr = self.transform_Xy(self.X, self.y, self.L)

        self.b = self.compute_b(self.Xtr, self.ytr)
        self.mu = self.compute_mu(self.Xtr, self.b)
        self.residuals_tr = self.compute_residuals(self.Xtr, self.ytr)

        if REML:
            """
            Ref:https://rh8liuqy.github.io/Linear_mixed_model_equations.html
            """
            XtX = self.Xtr.T @ self.Xtr

            ll = (
                (n - p) / 2 * np.log(2 * np.pi)
                + np.linalg.slogdet(self.L)[1]
                + np.linalg.slogdet(np.linalg.cholesky(XtX))[1]
                + self.residuals_tr.T @ self.residuals_tr / 2
            )
        else:
            ll = (
                n / 2 * np.log(2 * np.pi)
                + np.linalg.slogdet(self.L)[1]
                + self.residuals_tr.T @ self.residuals_tr / 2
            )

        return np.squeeze(ll)

    def fit(self, params=None, REML=True, method="Nelder-Mead"):

        if params is None:
            params = np.ones(len(self.Zs) + 1)

        res = minimize(self.profiled_deviance, params, args=(REML), method=method)

        return np.exp(res.x), res.fun, res.success
