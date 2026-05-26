import numpy as np
def calculate_rmse(y_true, y_pred) -> float:
    """Calculate Root Mean Squared Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def calculate_mae(y_true, y_pred) -> float:
    """Calculate Mean Absolute Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return float(np.mean(np.abs(y_true - y_pred)))


def calculate_mape(y_true, y_pred, epsilon: float = 1e-8) -> float:
    """
    Calculate Mean Absolute Percentage Error.

    epsilon is used to avoid division by zero.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denominator = np.where(np.abs(y_true) < epsilon, epsilon, np.abs(y_true))
    mape = np.mean(np.abs((y_true - y_pred) / denominator)) * 100

    return float(mape)