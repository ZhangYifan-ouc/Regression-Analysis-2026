import numpy as np

class CustomImputer:
    def __init__(self, fill_value=0):
        self.means_ = None
        self.fill_value = fill_value

    def fit(self, X: np.ndarray):
        n_cols = X.shape[1]
        self.means_ = np.zeros(n_cols)
        for i in range(n_cols):
            col = X[:, i]
            valid = col[~np.isnan(col)]
            if len(valid) > 0:
                self.means_[i] = np.mean(valid)
            else:
                self.means_[i] = self.fill_value
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.means_ is None:
            raise RuntimeError("必须先调用 fit()")
        X_filled = X.copy()
        for i, mean_val in enumerate(self.means_):
            mask = np.isnan(X_filled[:, i])
            if np.any(mask):
                X_filled[mask, i] = mean_val
        return X_filled

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

class CustomStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("必须先调用 fit()")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

class CustomWinsorizer:
    """异常值缩尾处理：将超出上下分位数的值替换为分位数边界值"""
    def __init__(self, lower_quantile=0.01, upper_quantile=0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X: np.ndarray):
        """计算每列的下限和上限分位数"""
        self.lower_bounds_ = np.quantile(X, self.lower_quantile, axis=0)
        self.upper_bounds_ = np.quantile(X, self.upper_quantile, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.lower_bounds_ is None or self.upper_bounds_ is None:
            raise RuntimeError("必须先调用 fit()")
        X_clipped = X.copy()
        for i in range(X.shape[1]):
            lower = self.lower_bounds_[i]
            upper = self.upper_bounds_[i]
            X_clipped[:, i] = np.clip(X_clipped[:, i], lower, upper)
        return X_clipped

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)