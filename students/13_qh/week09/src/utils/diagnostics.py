"""
Module: utils.diagnostics
Purpose: Model diagnostics tools for detecting multicollinearity.
"""

import numpy as np


def calculate_vif(X: np.ndarray) -> list:
    """
    Calculate Variance Inflation Factor (VIF) for each feature.

    VIF_j = 1 / (1 - R_j^2)

    where R_j^2 is the R-squared from regressing feature j on all other features.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix (without intercept column)

    Returns:
    --------
    list : VIF values for each feature
    """
    n_features = X.shape[1]
    vif_values = []

    for j in range(n_features):
        # Feature j as target
        y_j = X[:, j]

        # All other features as predictors
        X_other = np.delete(X, j, axis=1)

        # Add intercept
        X_other = np.column_stack([np.ones(len(X_other)), X_other])

        # Fit OLS: beta = (X^T X)^-1 X^T y
        try:
            beta = np.linalg.solve(X_other.T @ X_other, X_other.T @ y_j)
            y_pred = X_other @ beta

            # Calculate R^2
            ss_res = np.sum((y_j - y_pred) ** 2)
            ss_tot = np.sum((y_j - np.mean(y_j)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            # Calculate VIF
            vif = 1 / (1 - r_squared) if r_squared < 1 else float('inf')
        except np.linalg.LinAlgError:
            vif = float('inf')

        vif_values.append(vif)

    return vif_values


def check_multicollinearity(X: np.ndarray, feature_names: list, threshold: float = 10.0) -> dict:
    """
    Check for multicollinearity using VIF.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix (without intercept column)
    feature_names : list
        Names of features
    threshold : float
        VIF threshold for warning (default: 10.0)

    Returns:
    --------
    dict : {
        'vif_values': list of VIF values,
        'warnings': list of warning messages,
        'has_multicollinearity': bool
    }
    """
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
