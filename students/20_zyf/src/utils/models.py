"""
Module: utils.models
Purpose: Core machine learning estimators.
"""
import numpy as np


class AnalyticalOLS:
    """Analytical solution for OLS using normal equation."""
    
    def __init__(self):
        self.coef_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model using normal equation: beta = (X'X)^{-1}X'y"""
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)
        self.coef_ = np.linalg.solve(X.T @ X, X.T @ y).flatten()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return X @ self.coef_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate R-squared."""
        y = np.asarray(y).reshape(-1)
        y_pred = self.predict(X)
        sse = np.sum((y - y_pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        return 1 - sse / sst


class GradientDescentOLS:
    """Linear regression using gradient descent optimization."""
    
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

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int = 42):
        """Fit the model using gradient descent."""
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)
        
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        self.loss_history_ = []

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

            # Record full loss for comparison
            y_pred_full = X @ self.coef_
            mse = np.mean((y - y_pred_full) ** 2)
            self.loss_history_.append(mse)

            # Check convergence
            if epoch > 0:
                delta = abs(self.loss_history_[-1] - self.loss_history_[-2])
                if delta < self.tol:
                    break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return X @ self.coef_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate R-squared."""
        y = np.asarray(y).reshape(-1)
        y_pred = self.predict(X)
        sse = np.sum((y - y_pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        return 1 - sse / sst