import numpy as np
from sklearn.linear_model import LinearRegression

def calculate_vif(X: np.ndarray) -> list:
    """
    计算每个特征的方差膨胀因子 VIF，用于检测多重共线性
    VIF_j = 1 / (1 - R²_j)
    """
    X = np.asarray(X, dtype=np.float64)
    n_features = X.shape[1]
    vif_values = []

    for j in range(n_features):
        X_other = np.delete(X, j, axis=1)
        y_j = X[:, j]

        # ==========================
        # 彻底修复：同时删除 X 和 y 中的 NaN
        # ==========================
        mask = ~np.isnan(y_j)
        for col in range(X_other.shape[1]):
            mask &= ~np.isnan(X_other[:, col])

        X_other = X_other[mask]
        y_j = y_j[mask]

        # 如果数据为空，跳过
        if len(X_other) == 0:
            vif_values.append(99.9999)
            continue

        # OLS回归
        reg = LinearRegression().fit(X_other, y_j)
        r2 = reg.score(X_other, y_j)

        r2 = min(r2, 0.999999)
        vif = 1 / (1 - r2)
        vif_values.append(round(vif, 4))

    return vif_values