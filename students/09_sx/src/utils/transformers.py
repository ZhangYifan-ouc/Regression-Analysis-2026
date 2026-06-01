"""
模块：工具.转换器
用途：存放手写的预处理类
包含：CustomStandardScaler, SimpleImputer, Winsorizer
"""

import numpy as np


class CustomStandardScaler:
    """自定义标准化转换器"""
    
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self._is_fitted = False
    
    def fit(self, X: np.ndarray) -> 'CustomStandardScaler':
        X = np.asarray(X)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        self._is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("请先调用 fit() 方法")
        X = np.asarray(X)
        X_scaled = (X - self.mean_) / self.std_
        return X_scaled
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


class SimpleImputer:
    """简单填补器 - 修复版本"""
    
    def __init__(self, strategy: str = 'mean', fill_value: float = 0.0):
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_ = None
        self._is_fitted = False
    
    def fit(self, X: np.ndarray) -> 'SimpleImputer':
        X = np.asarray(X)
        # 转换为float以处理NaN
        X_float = X.astype(float)
        if self.strategy == 'mean':
            self.statistics_ = np.nanmean(X_float, axis=0)
        elif self.strategy == 'median':
            self.statistics_ = np.nanmedian(X_float, axis=0)
        elif self.strategy == 'constant':
            self.statistics_ = np.full(X.shape[1], self.fill_value)
        else:
            raise ValueError(f"不支持的填补策略: {self.strategy}")
        self._is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("请先调用 fit() 方法")
        X = np.asarray(X).copy().astype(float)
        nan_mask = np.isnan(X)
        for col in range(X.shape[1]):
            X[nan_mask[:, col], col] = self.statistics_[col]
        return X
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


class Winsorizer:
    """
    异常值处理器 - Winsorization（缩尾处理）
    """
    
    def __init__(self, limits: tuple = (0.01, 0.01)):
        self.lower_limit = limits[0]
        self.upper_limit = limits[1]
        self.lower_bounds_ = None
        self.upper_bounds_ = None
        self._is_fitted = False
    
    def fit(self, X: np.ndarray) -> 'Winsorizer':
        X = np.asarray(X)
        X_float = X.astype(float)
        self.lower_bounds_ = np.percentile(X_float, self.lower_limit * 100, axis=0)
        self.upper_bounds_ = np.percentile(X_float, (1 - self.upper_limit) * 100, axis=0)
        self._is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("请先调用 fit() 方法")
        X = np.asarray(X).copy().astype(float)
        for col in range(X.shape[1]):
            X[:, col] = np.clip(X[:, col], self.lower_bounds_[col], self.upper_bounds_[col])
        return X
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)
    
    def get_outlier_mask(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("请先调用 fit() 方法")
        X = np.asarray(X).astype(float)
        mask = (X < self.lower_bounds_) | (X > self.upper_bounds_)
        return mask