"""
Module: utils.transformers
Purpose: Data preprocessing transformers following sklearn API.
"""

import numpy as np


class CustomStandardScaler:
    """
    Custom Standard Scaler following Transformer API pattern.

    Standardizes features by removing the mean and scaling to unit variance:
        x' = (x - mean) / std
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray):
        """
        Compute the mean and standard deviation to be used for later scaling.

        Parameters:
        -----------
        X : np.ndarray
            Training data

        Returns:
        --------
        self
        """
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # 避免标准差为0
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Perform standardization by centering and scaling.

        Parameters:
        -----------
        X : np.ndarray
            Data to transform

        Returns:
        --------
        np.ndarray : Transformed data
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler has not been fitted yet. Call fit() first.")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit to data, then transform it.

        Parameters:
        -----------
        X : np.ndarray
            Training data

        Returns:
        --------
        np.ndarray : Transformed data
        """
        return self.fit(X).transform(X)
