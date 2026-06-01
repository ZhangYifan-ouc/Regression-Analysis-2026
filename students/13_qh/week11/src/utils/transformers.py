"""
Module: utils.transformers
Purpose: Data preprocessing transformers following sklearn API.
"""

import numpy as np


class CustomStandardScaler:
    """Custom Standard Scaler following Transformer API pattern."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray):
        """Compute the mean and standard deviation."""
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Perform standardization."""
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler has not been fitted yet.")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit to data, then transform it."""
        return self.fit(X).transform(X)


class CustomImputer:
    """Custom Imputer for handling missing values."""

    def __init__(self, strategy: str = "mean"):
        self.strategy = strategy
        self.fill_values_ = None

    def fit(self, X: np.ndarray):
        """Compute fill values based on strategy."""
        if self.strategy == "mean":
            self.fill_values_ = np.nanmean(X, axis=0)
        elif self.strategy == "median":
            self.fill_values_ = np.nanmedian(X, axis=0)
        else:
            raise ValueError("Strategy must be 'mean' or 'median'")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Fill missing values."""
        if self.fill_values_ is None:
            raise ValueError("Imputer has not been fitted yet.")
        X_filled = X.copy()
        for i in range(X.shape[1]):
            mask = np.isnan(X[:, i])
            X_filled[mask, i] = self.fill_values_[i]
        return X_filled

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit to data, then transform it."""
        return self.fit(X).transform(X)
