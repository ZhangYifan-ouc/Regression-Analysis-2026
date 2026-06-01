"""
Module: utils.diagnostics
Purpose: Model diagnostics tools for detecting multicollinearity.
"""

import numpy as np


def calculate_vif(X: np.ndarray) -> list:
    """Calculate Variance Inflation Factor (VIF) for each feature."""
    n_features = X.shape[1]
    vif_values = []

    for j in range(n_features):
        y_j = X[:, j]
        X_other = np.delete(X, j, axis=1)
        X_other = np.column_stack([np.ones(len(X_other)), X_other])

        try:
            beta = np.linalg.solve(X_other.T @ X_other, X_other.T @ y_j)
            y_pred = X_other @ beta
            ss_res = np.sum((y_j - y_pred) ** 2)
            ss_tot = np.sum((y_j - np.mean(y_j)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            vif = 1 / (1 - r_squared) if r_squared < 1 else float('inf')
        except np.linalg.LinAlgError:
            vif = float('inf')

        vif_values.append(vif)

    return vif_values


def check_multicollinearity(X: np.ndarray, feature_names: list, threshold: float = 10.0) -> dict:
    """Check for multicollinearity using VIF."""
    vif_values = calculate_vif(X)
    warnings = []
    has_multicollinearity = False

    for i, (name, vif) in enumerate(zip(feature_names, vif_values)):
        if vif > threshold:
            has_multicollinearity = True
            warnings.append(f"特征 '{name}' 的 VIF = {vif:.2f} > {threshold}，存在严重多重共线性")

    return {
        'vif_values': vif_values,
        'warnings': warnings,
        'has_multicollinearity': has_multicollinearity
    }
