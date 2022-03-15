import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from formulaic import model_matrix

from scipy.linalg import qr, lu, cholesky, solve_triangular
from scipy.stats import norm, t, f
from scipy.optimize import minimize

__all__ = ['lm', 'data']

class lm:
    
    def __init__(self,
                formula: str,
                data: pd.DataFrame,
                method: str = 'qr'):
        
        # meta
        self.formula = formula
        self.data = data
        self.method = method
        
        # design matrix
        self.y, self.X = model_matrix(formula, data)

        self.column_names = self.X.columns        
        self.feature_names = self.column_names[1:] if 'Intercept' in self.column_names else self.column_names 
        
        X = self.X.values
        y = self.y.values.flatten()
        
        self.n, self.p = (n, p) = X.shape # n_samples x n_features (intercept included if available)
        
        # model degree of freedom
        self.df_model = self.p - 1 if 'Intercept' in self.column_names else self.p
            
        # residual degrees of freedom (n - p)
        self.df_residuals = self.n - self.df_model - 1 if 'Intercept' in self.column_names else self.n - self.df_model
    
        ##############
        # Estimation #
        ##############
    
        if method == 'nll':
            bhat, self.sigma, self.XtX, self.Xty, self.loss = self.compute_bhat(X, y, 'nll')
        elif method == 'sse':
            bhat, self.XtX, self.Xty, self.loss = self.compute_bhat(X, y, 'sse')
        else:
            bhat, self.XtX, self.Xty = self.compute_bhat(X, y, method)
            
        self.bhat = pd.DataFrame(bhat.reshape(1, -1), columns=self.column_names, index=['Estimate'])
        
        
        # compute predicted (fitted values ŷ = Xβ̂)
        self.yhat = self.compute_yhat()
        yhat = self.yhat.values.flatten()
        
        # compute residuals ϵ̂
        residuals = y - yhat
        self.residuals = pd.DataFrame(residuals[:, None], columns=['residuals'])
        
        # compute residual sum of squares (RSS)
        self.rss = np.squeeze(residuals.T @ residuals)
        
        # compute standard deviation of model coefficients
        # aka Residual SE: σ^2 = RSS / df_residuals
        if method == 'nll':
            self.sigma_squared = self.sigma ** 2
        else:
            self.sigma_squared = self.rss / self.df_residuals
            self.sigma = np.sqrt(self.sigma_squared)
            
        # compute standard error for β̂
        self.XtXinv = self.compute_XtXinv()
        
        se_bhat, V_bhat = self.compute_se_bhat()
        self.V_bhat = V_bhat
        self.se_bhat = pd.DataFrame(se_bhat.reshape(1, -1), columns=self.column_names, index=['Std. Error'])
            
        # compute confidence interval for β̂ 
        self.ci_bhat = self.compute_ci_bhat()
        
        # compute t values of model coefficients
        self.t_values = self.compute_t_values()
        
        # p values
        self.p_values = self.compute_p_values()
        
        # compute r2 and r2adjusted, aka coefficient of determination
        # aka percentage of variance explained. Noted that the formulae
        # are different for cases with and without intercept
        self.tss, self.r_squared, self.r_squared_adjusted = self.compute_goodness_of_fit()
            

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
        docstring += self.bhat.to_string()
        
        return docstring

    def __str__(self):
        
        return self.__repr__()
        
    def compute_XtXinv(self):
        U, S, Vt = np.linalg.svd(self.XtX, full_matrices=False)
        XtXinv = Vt.T @ np.diag(1/S) @ U.T
        return XtXinv
        
    def compute_bhat(self, X, y, method='qr', return_ss=True):

        match method:
            case 'qr':
                bhat, XtX, Xty = _qr(X, y)
            case 'lu':
                bhat, XtX, Xty = _lu(X, y)
            case 'chol':
                bhat, XtX, Xty = _chol(X, y)
            case 'svd':
                bhat, XtX, Xty = _svd(X, y)
            case 'nq':
                bhat, XtX, Xty = _nq(X, y)
            case 'sse':
                bhat, loss = _sse(X, y)
                if return_ss:
                    XtX = X.T @ X
                    Xty = X.T @ y
                    return bhat, XtX, Xty, loss
                else:
                    return bhat
            case 'nll':
                bhat, sigma, loss = _nll(X, y)

                if return_ss:
                    XtX = X.T @ X
                    Xty = X.T @ y
                    return bhat, sigma, XtX, Xty, loss
                else:
                    return bhat
            case _:
                raise ValueError('Please enter a valid method.')
        
        if return_ss:
            return bhat, XtX, Xty
        else:
            return bhat
    
    def compute_se_bhat(self):
        V_bhat = self.sigma_squared * self.XtXinv
        se_bhat = np.sqrt(np.diag(V_bhat))[:, None]    
        return se_bhat, V_bhat
    
    def compute_ci_bhat(self, alpha=0.05):
        
        se_bhat = self.se_bhat.values.T
        bhat = self.bhat.values.T
        ci_bhat = t.ppf(1 - alpha/2, self.df_residuals) * se_bhat * np.array([-1, 1]) + bhat
        ci_bhat = pd.DataFrame(ci_bhat, index=self.column_names, columns=[f'CI[{alpha/2*100}%]', f'CI[{100-alpha/2*100}]%'])
        return ci_bhat
        
    def compute_ci_bhat_bootstrap(self, num_bootstrap=4000, alpha=0.05):

        X = self.X.values
        bhat = self.bhat.values.T
        residuals = self.residuals.values.flatten()
        bhat_stars = np.zeros([num_bootstrap, self.p])
        for i in range(num_bootstrap):
            residuals_star = np.random.choice(residuals, size=len(residuals), replace=True)
            y_star = X @ bhat + residuals_star[:, None]    
            bhat_star = self.compute_bhat(X, y_star.flatten(), return_ss=False)
            bhat_stars[i] = bhat_star

        ci_bhat_bootstrap = np.quantile(bhat_stars, q=[alpha/2, 1-alpha/2], axis=0).T
        self.ci_bhat_bootstrap = ci_bhat_bootstrap = pd.DataFrame(ci_bhat_bootstrap, index=self.column_names, columns=[f'CI[{alpha/2*100}%]', f'CI[{100-alpha/2*100}]%'])
        self.bhat_bootstrap = pd.DataFrame(bhat_stars, columns=self.column_names)
        
        return ci_bhat_bootstrap
    
    def compute_yhat(self, Xnew=None, interval=None, alpha=0.05):
        if Xnew is None:
            X = self.X.values
        else:
            X = self.X.model_spec.get_model_matrix(Xnew).values   
        # compute predicted or fitted values ŷ = Xβ̂
        bhat = self.bhat.values.T
        yhat = X @ bhat
        yhat = pd.DataFrame(yhat, columns=['Fitted'])
        
        match interval:
            case None:
                return yhat
            case True:
                ci_yhat = self.compute_ci_yhat(yhat, Xnew, alpha)
                pi_yhat = self.compute_pi_yhat(yhat, Xnew, alpha)
                return pd.concat([yhat, ci_yhat, pi_yhat], axis=1)
            case 'prediction':
                pi_yhat = self.compute_pi_yhat(yhat, Xnew, alpha)
                return pd.concat([yhat, pi_yhat], axis=1)
            case 'confidence':
                ci_yhat = self.compute_ci_yhat(yhat, Xnew, alpha)
                return pd.concat([yhat, ci_yhat], axis=1)                
            case _:
                raise ValueError("Please enter a valid value: [None, True, 'prediction', 'confidence']")
        
    
    def compute_ci_yhat(self, yhat, Xnew=None, alpha=0.05):
        
        if Xnew is None:
            X = self.X.values
        else:
            X = self.X.model_spec.get_model_matrix(Xnew).values 
        
        sigma = self.sigma
        sigma_squared = self.sigma_squared
        
        V_yhat = X @ self.XtXinv @ X.T * sigma_squared 

        se_yhat_mean = np.sqrt(np.diag(V_yhat)) * sigma
        ci_yhat = t.ppf(1 - alpha/2, self.df_residuals) * se_yhat_mean[:, None] * np.array([-1, 1]) + yhat.values
        ci_yhat = pd.DataFrame(ci_yhat, columns=[f'CI[{alpha/2*100}%]', f'CI[{100-alpha/2*100}]%'])        

        return ci_yhat
    
    def compute_pi_yhat(self, yhat, Xnew=None, alpha=0.05):
        
        if Xnew is None:
            X = self.X.values
        else:
            X = self.X.model_spec.get_model_matrix(Xnew).values 
        
        sigma = self.sigma
        sigma_squared = self.sigma_squared
        
        V_yhat = X @ self.XtXinv @ X.T * sigma_squared 

        se_yhat = np.sqrt( 1 + np.diag(V_yhat)) * sigma
        pi_yhat = t.ppf(1 - alpha/2, self.df_residuals) * se_yhat[:, None] * np.array([-1, 1]) + yhat.values
        pi_yhat = pd.DataFrame(pi_yhat, columns=[f'PI[{alpha/2*100}%]', f'PI[{100-alpha/2*100}]%'])

        return pi_yhat
        
    
    def compute_t_values(self):
    
        t_values = self.bhat.values / self.se_bhat.values
        
        return pd.DataFrame(t_values, columns=self.column_names, index=['t values'])
    
    def compute_p_values(self):
        # compute p values of model coefficients with scipy.stats.t.sf
        # H0: βi==0
        # H1: βi!=0
        p_values = 2 * t.sf(np.abs(self.t_values.values), self.df_residuals)
        return pd.DataFrame(p_values, columns=self.column_names, index=['Pr(>|t|)'])
    
    def compute_goodness_of_fit(self):
        
        y = self.y.values
        
        if 'Intercept' in self.column_names:
            tss = np.sum((y - y.mean()) ** 2) 
            # Eq: r2 = 1 - RSS / TSS = 1 -  sum((ŷ - yi)**2) / sum((y - ȳ)**2)
            r_squared = (1 - self.rss / tss).squeeze()
            # Eq: r2adj = 1 - (1 - r2) * (n - 1) / df_residuals
            r_squared_adjusted = 1 - (1 - r_squared) * (self.n - 1) / (self.df_residuals)
        else:
            tss = np.sum(y**2)
            # Eq: r2 = 1 - RSS / TSS = 1 -  sum((ŷ - yi)**2) / sum((y)**2)
            r_squared = (1 - self.rss / tss).squeeze()
            # Eq: r2adj = 1 - (1 - r2) * n / df_residuals
            r_squared_adjusted = 1 - (1 - r_squared) * self.n / (self.df_residuals)
            
        return tss, r_squared, r_squared_adjusted
            
    def compute_fstats(self):
        if self.df_model != 0:
            fstats = np.squeeze(((self.tss - self.rss) / self.df_model) / (self.rss / self.df_residuals)) 
            f_p_value = f.sf(fstats, self.df_model, self.df_residuals).squeeze()
        else:
            fstats, f_p_value = None, None
        return fstats, f_p_value
        
    def compute_loglikelihood(self):
        return np.squeeze(- 0.5 * self.n * (np.log(self.rss / self.n) + np.log(2 * np.pi) + 1))
    
    def compute_AIC(self):
        # add 1 to p to keep consistent with R
        # https://stackoverflow.com/q/37917437
        return -2 * self.loglike + 2 * (self.p + 1)
    
    def compute_BIC(self):
        # add 1 to p to keep consistent with R
        # https://stackoverflow.com/q/37917437
        return -2 * self.loglike + np.log(self.n) * (self.p + 1)
    
    def predict(self, Xnew=None, interval=None, alpha=0.05):
        return self.compute_yhat(Xnew=Xnew, interval=interval, alpha=alpha)
    
    def summary(self, digits=3, cor=False):
        
        docstring = f"""Formula: {self.formula}\n\n"""
        docstring += "Coefficients:\n"
        
        sig = significance_code(self.p_values.values.T)
        res = pd.concat([self.bhat, self.se_bhat, self.ci_bhat.T, self.t_values, self.p_values], axis=0).T.round(digits)

        res[''] = sig 
        self.results = res
        
        docstring += res.to_string()
        
        docstring += f'\n\nn = {self.n}, p = {self.p}, Residual SE = {np.sqrt(self.sigma_squared):.3f} on {self.df_residuals} DF\n'
        docstring += f'R-Squared = {self.r_squared:.4f}, adjusted R-Squared = {self.r_squared_adjusted:.4f}\n'
        
        if self.fstats is not None:
            docstring += f'F-statistics = {self.fstats:.4f} on {self.df_model} and {self.df_residuals} DF, p-value: {self.f_p_value:{".2f" if self.f_p_value > 1e-5 else "e"}}\n\n'
        
        docstring += f'Log Likelihood = {self.loglike:.4f}, AIC = {self.AIC:.4f}, BIC = {self.BIC:.4f}'
        
        if cor is True:
            
            docstring += f'\n\nCorrelation of Coefficients:\n'
            if 'Intercept' in self.column_names:
                docstring += self.X.drop('Intercept', axis=1).corr().to_string(formatters={col: "{:.2f}".format for col in self.X.columns})
            else:
                docstring += self.X.corr().to_string(formatters={col: "{:.2f}".format for col in self.X.columns})
        
        print(docstring)
        
    def plot_residuals(self, ax=None, figsize=None,
                       facecolor='none', edgecolor='black'):
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(self.yhat['Fitted'], self.residuals, facecolor=facecolor, edgecolor=edgecolor)
        ax.axhline(0, color='black', linestyle='--')
        ax.set_xlabel('Fitted')
        ax.set_ylabel('Residuals')
        ax.set_title('Residuals vs. Fitted Plot')

    def plot_contrast(self, features=None, figsize=None, subplots=None, away_from='median'):

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
            figsize = np.array([4* num_subplots, 3]) 
        
        if subplots is None:
            subplots= (1, num_subplots)
        
        if num_subplots > 1:
            fig, ax = plt.subplots(subplots[0], subplots[1], figsize=figsize)
        else:
            fig, ax = plt.subplots(figsize=figsize)
        ax = np.array([ax]).flatten()
        
 
        for i, name in enumerate(features):

            xx = self.X[name].values

            if away_from == 'median':
                xxbar = np.median(xx)
            elif away_from == 'mean':
                xxbar = np.mean(xx)
            elif away_from == '0':
                xxbar = 0
            else:
                raise ValueError('The Input value for "away_from" is not supported.')

            rj = self.y - self.X.assign(**{name:xxbar}).values @ self.bhat.values.T
            ax[i].scatter(self.X[name], rj, color='gray', facecolor='none', edgecolor='black')
            ax[i].set_xlabel(name)
            ax[i].set_ylabel('Δ' + self.y.columns[0])
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['right'].set_visible(False)


            Vx = (xx - xxbar) ** 2 * self.se_bhat.loc['Std. Error', name]**2
            se = np.sqrt(Vx)
            
            tt = t.ppf(1 - 0.05/2, self.df_residuals)
            yy = (xx -  xxbar) * self.bhat[name].values
            idx_sorted = np.argsort(xx)
            ax[i].plot(xx[idx_sorted], yy[idx_sorted], color='black')
            ax[i].fill_between(xx[idx_sorted], yy[idx_sorted] + tt * se[idx_sorted], yy[idx_sorted]-tt*se[idx_sorted], 
                            alpha=0.5)

        fig.tight_layout()

    def plot_conditional(self, features=None, figsize=None, subplots=None, away_from='median'):
        
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
            figsize = np.array([4* num_subplots, 3]) 
        
        if subplots is None:
            subplots= (1, num_subplots)
        
        if num_subplots > 1:
            fig, ax = plt.subplots(subplots[0], subplots[1], figsize=figsize)
        else:
            fig, ax = plt.subplots(figsize=figsize)
        ax = np.array([ax]).flatten()
        
        for i, name in enumerate(features):
            
            xx = self.X[name].values
            
            if away_from == 'median':
                Xnew = self.X.assign(**{name1:self.X[name1].median() for name1 in self.column_names if name1 != name}).values
            elif away_from == 'mean':
                Xnew = self.X.assign(**{name1:self.X[name1].mean() for name1 in self.column_names if name1 != name}).values
            elif away_from == '0':
                Xnew = self.X.assign(**{name1:0 for name1 in self.column_names if name1 != name}).values
            else:
                raise ValueError('The Input value for "away_from" is not supported.')
                
            rj = self.residuals + Xnew @ self.bhat.values.T
            ax[i].scatter(self.X[name], rj, color='gray', facecolor='none', edgecolor='black')
            ax[i].set_xlabel(name)
            ax[i].set_ylabel(self.y.columns[0])
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['right'].set_visible(False)

            Vx = Xnew @ self.V_bhat @ Xnew.T
            se = np.sqrt(np.diag(Vx))
            
            tt = t.ppf(1 - 0.05/2, self.df_residuals)
            yy = (Xnew @ self.bhat.values.T).flatten()
            
            ax[i].plot(xx[np.argsort(xx)], yy[np.argsort(xx)], color='black')
            ax[i].fill_between(xx[np.argsort(xx)], yy[np.argsort(xx)] + tt * se[np.argsort(xx)], yy[np.argsort(xx)]-tt*se[np.argsort(xx)], 
                            alpha=0.5)

        fig.tight_layout()


#################
# Estimate bhat #
#################


def _nq(X: np.array, 
        y: np.array, 
        return_ss: bool = True):
    """
    Solving the normal equations by directly inverting
    the gram matrix.

    return_ss: return sufficient statistics
    
    """
    XtX = X.T @ X
    Xty = X.T @ y
    b = np.linalg.inv(XtX) @ (Xty)
    if return_ss:
        return b, XtX, Xty
    else:
        return b
    
def _lu(X: np.array, 
        y: np.array, 
        return_ss: bool = True):
    """
    LU decomposition. 
    The same as using numpy.linalg.solve()
    """
    XtX = X.T @ X
    Xty = X.T @ y
    P, L, U = lu(XtX, permute_l=False)
    z = solve_triangular(L, P @ Xty, lower=True)
    b = solve_triangular(U, z)
    
    if return_ss:
        return b, XtX, Xty
    else:
        return b
    
def _chol(X: np.array, 
          y: np.array, 
          return_ss: bool = True):
    """
    Cholesky decomposition.
    """
    XtX = X.T @ X
    Xty = X.T @ y
    L = cholesky(XtX, lower=True)
    b = solve_triangular(L.T, 
            solve_triangular(
                L, Xty, lower=True
    ))
    if return_ss:
        return b, XtX, Xty
    else:
        return b
    
def _svd(X: np.array, 
        y: np.array, 
        return_ss: bool = True):
    """
    Single value decomposition.
    """
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    Sinv = np.diag(1/S)
    b = Vt.T @ Sinv @ U.T @ y
    if return_ss:
        XtX = X.T @ X
        Xty = X.T @ y
        return b, XtX, Xty
    else:
        return b
    
def _qr(X: np.array, 
        y: np.array, 
        return_ss: bool = True):
    """
    QR decomposition.
    """
    
    Q, R = qr(X, mode='economic')
    f = Q.T @ y
    b = solve_triangular(R, f)

    if return_ss:
        XtX = X.T @ X
        Xty = X.T @ y
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
        L = - np.sum(norm.logpdf(y, loc=mu, scale=p[0]))
        return L
    
    res = minimize(cost, p, args=(X, y), method='L-BFGS-B')
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
        L = np.sum((y-mu)**2)
        return L
    
    res = minimize(cost, p, args=(X, y), method='L-BFGS-B')
    return res.x, res.fun

#########
# Utils #
#########

def AIC(*ms):
    
    aic = [m.AIC for m in ms]
    df = [m.p + 1 for m in ms]
    formuli = [m.formula for m in ms]
    
    df = pd.DataFrame(np.vstack([formuli, df, aic]).T, 
            columns=['formula', 'df', 'AIC']).set_index('formula')
    
    return df


def significance_code(p_values):
    
    sig = []
    for p in p_values:

        if p < 0.001:
            sig.append('***')
        elif p < 0.01:
            sig.append('**')
        elif p < 0.05:
            sig.append('*')
        elif p < 0.1:
            sig.append('.')
        else:
            sig.append(' ')
            
    return sig

def data(name, package='R', save_to='./data', overwrite=False):
    
    import urllib.request
    import os
    import polars as pl
    
    datapath = save_to + f'/{package}/'
    
    if not os.path.exists(save_to):
        os.makedirs(save_to)
        
    if not os.path.exists(datapath):
        os.makedirs(datapath)
        
    if os.path.exists(datapath + f'{name}.csv') is True and overwrite is False:
        pass
    else:
        print(f'Downloading {name} (from {package})...')
        url = f'https://raw.githubusercontent.com/huangziwei/lmpy/main/datasets/{package}/{name}.csv'
        urllib.request.urlretrieve(url, datapath + f'{name}.csv')
        
    df = pl.read_csv(datapath + f'{name}.csv', null_values='NA').to_pandas()
    return df