"""
Module: utils.metrics
Purpose: Evaluation metrics for regression models.
"""

import numpy as np


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error (RMSE).

    RMSE = sqrt(1/n * sum((y_true - y_pred)^2))
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error (MAE).

    MAE = 1/n * sum(|y_true - y_pred|)
    """
    return np.mean(np.abs(y_true - y_pred))


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).

    MAPE = 1/n * sum(|y_true - y_pred| / |y_true|) * 100

    Note: Handles zero or near-zero values in y_true.
    """
    # 避免除以零或极小值
    epsilon = 1e-8
    mask = np.abs(y_true) > epsilon

    if not np.any(mask):
        return float('inf')

    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mape
