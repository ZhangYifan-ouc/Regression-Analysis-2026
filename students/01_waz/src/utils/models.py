"""
Module: utils.models
Purpose: Core machine learning estimators.
"""
import numpy as np
from scipy import stats


class CustomOLS:
    """自定义OLS回归器 - 与AnalyticalOLS功能相同"""
    
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, X, y):
        if self.fit_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])

        try:
            self.coef_ = np.linalg.solve(X.T @ X, X.T @ y)
        except np.linalg.LinAlgError:
            self.coef_ = np.linalg.lstsq(X.T @ X, X.T @ y, rcond=None)[0]

        if self.fit_intercept:
            self.intercept_ = self.coef_[0]
            self.coef_ = self.coef_[1:]

        return self
    
    def predict(self, X):
        if self.fit_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])
            return X @ np.concatenate([[self.intercept_], self.coef_])
        return X @ self.coef_
    
    def score(self, X, y):
        y_pred = self.predict(X)
        sse = np.sum((y - y_pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        return 1 - (sse / sst) if sst != 0 else 0.0

class AnalyticalOLS:
    """
    Analytical OLS regression model.
    Supports fit, predict, score, and f_test.
    """

    def __init__(self):
        self.coef_ = None
        self.cov_matrix_ = None
        self.sigma2_ = None
        self.df_resid_ = None
        self.residuals_ = None
        self.fitted_values_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n, p = X.shape
        xtx = X.T @ X
        xtx_inv = np.linalg.inv(xtx)
        self.coef_ = xtx_inv @ X.T @ y

        self.fitted_values_ = X @ self.coef_
        self.residuals_ = y - self.fitted_values_

        rss = self.residuals_ @ self.residuals_
        self.df_resid_ = n - p
        self.sigma2_ = rss / self.df_resid_
        self.cov_matrix_ = self.sigma2_ * xtx_inv
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        return X @ self.coef_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        rss = np.sum((y - y_pred) ** 2)
        tss = np.sum((y - np.mean(y)) ** 2)
        return 1 - (rss / tss)

    def f_test(self, C: np.ndarray, d: np.ndarray) -> dict:
        if self.coef_ is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        q = C.shape[0]
        diff = C @ self.coef_ - d
        cov_c = C @ self.cov_matrix_ @ C.T
        cov_c_inv = np.linalg.inv(cov_c)
        f_stat = (diff.T @ cov_c_inv @ diff) / q
        p_value = 1 - stats.f.cdf(f_stat, q, self.df_resid_)

        return {
            "f_stat": f_stat,
            "p_value": p_value,
            "q": q,
            "df_resid": self.df_resid_,
        }


class GradientDescentOLS:
    def __init__(
        self,
        learning_rate: float = 0.01,
        tol: float = 1e-5,
        max_iter: int = 1000,
        gd_type: str = "full_batch",
        batch_fraction: float = 0.1,
    ):
        self.learning_rate = learning_rate
        self.tol = tol
        self.max_iter = max_iter
        self.gd_type = gd_type
        self.batch_fraction = batch_fraction

        self.coef_ = None
        self.loss_history_ = []
        self.full_loss_history_ = []

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int = 42):
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        self.loss_history_ = []
        self.full_loss_history_ = []

        rng = np.random.default_rng(seed)

        if self.gd_type == "full_batch":
            batch_size = n_samples
        elif self.gd_type == "mini_batch":
            batch_size = max(1, int(n_samples * self.batch_fraction))
        else:
            raise ValueError("gd_type must be 'full_batch' or 'mini_batch'")

        for epoch in range(self.max_iter):
            if self.gd_type == "mini_batch":
                indices = rng.choice(n_samples, size=batch_size, replace=False)
                X_batch = X[indices]
                y_batch = y[indices]
            else:
                X_batch = X
                y_batch = y

            y_pred_batch = X_batch @ self.coef_
            error_batch = y_pred_batch - y_batch
            gradient = (2 / len(X_batch)) * (X_batch.T @ error_batch)

            self.coef_ -= self.learning_rate * gradient

            updated_pred_batch = X_batch @ self.coef_
            batch_mse = np.mean((y_batch - updated_pred_batch) ** 2)
            y_pred_full = X @ self.coef_
            full_mse = np.mean((y - y_pred_full) ** 2)
            self.loss_history_.append(batch_mse)
            self.full_loss_history_.append(full_mse)

            if epoch > 0:
                delta = abs(self.full_loss_history_[-1] - self.full_loss_history_[-2])
                if delta < self.tol:
                    break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        sse = np.sum((y - y_pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        return 1 - sse / sst