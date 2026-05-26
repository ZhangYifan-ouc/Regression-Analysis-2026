"""Small regression models maintained as a personal utility library."""
from __future__ import annotations

import numpy as np


class AnalyticalOLS:
    """Ordinary Least Squares solved by the normal equation.

    The class does not add an intercept automatically. Add a column of ones
    before fit/predict if an intercept is needed.
    """

    def __init__(self) -> None:
        self.coef_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AnalyticalOLS":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim != 2:
            raise ValueError("X must be 2-D")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        self.coef_ = np.linalg.lstsq(X, y, rcond=None)[0]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("The model has not been fitted yet")
        X = np.asarray(X, dtype=float)
        return X @ self.coef_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=float).ravel()
        y_pred = self.predict(X)
        sse = np.sum((y - y_pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        if np.isclose(sst, 0.0):
            return 0.0
        return float(1.0 - sse / sst)


class GradientDescentOLS:
    """OLS solved by gradient descent.

    The interface is intentionally simple: fit learns coefficients, predict uses
    the learned coefficients, and score reports R^2. The class does not add an
    intercept internally.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        tol: float = 1e-8,
        max_iter: int = 10000,
        gd_type: str = "full_batch",
        batch_fraction: float = 0.25,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if tol < 0:
            raise ValueError("tol must be non-negative")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if gd_type not in {"full_batch", "mini_batch"}:
            raise ValueError("gd_type must be 'full_batch' or 'mini_batch'")
        if not 0 < batch_fraction <= 1:
            raise ValueError("batch_fraction must be in (0, 1]")

        self.learning_rate = learning_rate
        self.tol = tol
        self.max_iter = max_iter
        self.gd_type = gd_type
        self.batch_fraction = batch_fraction
        self.coef_: np.ndarray | None = None
        self.loss_history_: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int = 42) -> "GradientDescentOLS":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim != 2:
            raise ValueError("X must be 2-D")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")

        n_samples, n_features = X.shape
        batch_size = n_samples
        if self.gd_type == "mini_batch":
            batch_size = max(1, int(round(n_samples * self.batch_fraction)))

        self.coef_ = np.zeros(n_features, dtype=float)
        self.loss_history_ = []
        rng = np.random.default_rng(seed)
        previous_loss = np.inf

        for _ in range(self.max_iter):
            if self.gd_type == "mini_batch":
                idx = rng.choice(n_samples, size=batch_size, replace=False)
                X_batch = X[idx]
                y_batch = y[idx]
            else:
                X_batch = X
                y_batch = y

            error_batch = X_batch @ self.coef_ - y_batch
            gradient = (2.0 / X_batch.shape[0]) * (X_batch.T @ error_batch)
            self.coef_ -= self.learning_rate * gradient

            full_error = X @ self.coef_ - y
            loss = float(np.mean(full_error**2))
            self.loss_history_.append(loss)
            if abs(previous_loss - loss) < self.tol:
                break
            previous_loss = loss

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("The model has not been fitted yet")
        X = np.asarray(X, dtype=float)
        return X @ self.coef_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=float).ravel()
        y_pred = self.predict(X)
        sse = np.sum((y - y_pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        if np.isclose(sst, 0.0):
            return 0.0
        return float(1.0 - sse / sst)


CustomOLS = AnalyticalOLS


class OrdinaryLeastSquares:
    """Week 12 convenience OLS regressor that adds an intercept internally.

    This class is added on top of the Week 10/11 model utilities.  The earlier
    AnalyticalOLS and GradientDescentOLS classes are intentionally kept above,
    so later weeks extend the utils library instead of replacing it.
    """

    def __init__(self) -> None:
        self.coef_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OrdinaryLeastSquares":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).ravel()
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        X_design = np.column_stack([np.ones(X_arr.shape[0]), X_arr])
        self.coef_ = np.linalg.lstsq(X_design, y_arr, rcond=None)[0]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("The model has not been fitted yet")
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        X_design = np.column_stack([np.ones(X_arr.shape[0]), X_arr])
        return X_design @ self.coef_


LinearRegressionWithIntercept = OrdinaryLeastSquares
