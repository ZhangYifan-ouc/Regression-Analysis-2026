import numpy as np
def calculate_vif(X, feature_names=None) -> list[dict]:
    """
    Calculate Variance Inflation Factor for each feature.

    VIF_j = 1 / (1 - R_j^2)

    Returns a list of dictionaries:
    [
        {"feature": "x1", "vif": 12.3},
        ...
    ]
    """
    X = np.asarray(X, dtype=float)

    n_features = X.shape[1]

    if feature_names is None:
        feature_names = [f"x{i}" for i in range(n_features)]

    results = []

    for j in range(n_features):
        y = X[:, j]
        X_others = np.delete(X, j, axis=1)

        if X_others.shape[1] == 0:
            vif = 1.0
        else:
            X_others = np.column_stack([np.ones(X_others.shape[0]), X_others])

            try:
                beta = np.linalg.lstsq(X_others, y, rcond=None)[0]
                y_pred = X_others @ beta

                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)

                if ss_tot == 0:
                    vif = float("inf")
                else:
                    r_squared = 1 - ss_res / ss_tot

                    if r_squared >= 1:
                        vif = float("inf")
                    else:
                        vif = 1 / (1 - r_squared)

            except np.linalg.LinAlgError:
                vif = float("inf")

        results.append(
            {
                "feature": feature_names[j],
                "vif": float(vif),
            }
        )

    return results


def top_vif(vif_results: list[dict], n: int = 10) -> list[dict]:
    """Return top n VIF results sorted from high to low."""
    return sorted(vif_results, key=lambda item: item["vif"], reverse=True)[:n]