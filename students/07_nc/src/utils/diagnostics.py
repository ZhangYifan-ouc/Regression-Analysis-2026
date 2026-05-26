"""Regression diagnostics maintained in the personal utils library."""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.models import AnalyticalOLS


def add_intercept(X: np.ndarray) -> np.ndarray:
    """Add a leading intercept column to a numeric design matrix."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2-D")
    return np.column_stack([np.ones(X.shape[0]), X])


def calculate_vif(X: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Calculate variance inflation factors using the custom OLS model.

    VIF_j = 1 / (1 - R_j^2), where R_j^2 comes from regressing feature j on the
    remaining features. The function is written directly with numpy and the
    custom AnalyticalOLS rather than relying on statsmodels.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2-D")
    if X.shape[1] != len(feature_names):
        raise ValueError("feature_names length must equal number of X columns")

    rows: list[dict[str, float | str]] = []
    for j, name in enumerate(feature_names):
        y_j = X[:, j]
        other_idx = [i for i in range(X.shape[1]) if i != j]
        if len(other_idx) == 0 or np.isclose(np.var(y_j), 0.0):
            vif = np.inf
        else:
            X_other = add_intercept(X[:, other_idx])
            model = AnalyticalOLS().fit(X_other, y_j)
            r2 = model.score(X_other, y_j)
            vif = np.inf if np.isclose(1.0 - r2, 0.0) else 1.0 / max(1e-12, 1.0 - r2)
        rows.append({"feature": name, "VIF": float(vif)})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)


def correlation_pairs(df: pd.DataFrame, threshold: float = 0.75) -> pd.DataFrame:
    """Return absolute correlations above a threshold for numeric columns."""
    corr = df.select_dtypes(include=[np.number]).corr().abs()
    rows: list[dict[str, float | str]] = []
    cols = list(corr.columns)
    for i, col_a in enumerate(cols):
        for col_b in cols[i + 1 :]:
            value = corr.loc[col_a, col_b]
            if pd.notna(value) and value >= threshold:
                rows.append({"feature_1": col_a, "feature_2": col_b, "abs_corr": float(value)})
    return pd.DataFrame(rows).sort_values("abs_corr", ascending=False).reset_index(drop=True)


def residual_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Basic residual diagnostics for reports."""
    residuals = np.asarray(y_true, dtype=float).ravel() - np.asarray(y_pred, dtype=float).ravel()
    return {
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals)),
        "residual_median": float(np.median(residuals)),
        "residual_p95_abs": float(np.quantile(np.abs(residuals), 0.95)),
    }
