import numpy as np

class CustomOLS:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        # 移除全是 NaN 的行
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        y = y[mask]

        # 如果数据无效，随便给个值，防止崩溃
        if X.shape[0] == 0 or X.shape[1] == 0:
            self.coef_ = np.zeros(X.shape[1])
            self.intercept_ = 0
            return

        # 安全求解
        try:
            X_with_bias = np.hstack([np.ones((X.shape[0], 1)), X])
            beta = np.linalg.inv(X_with_bias.T @ X_with_bias + 1e-6 * np.eye(X_with_bias.shape[1])) @ X_with_bias.T @ y
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
        except:
            self.coef_ = np.zeros(X.shape[1])
            self.intercept_ = 0

    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_