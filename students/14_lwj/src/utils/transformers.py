import numpy as np

class CustomStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        # 先处理NaN：计算均值和标准差时自动忽略NaN
        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0, ddof=0)
        
        # 防止标准差为0导致除以0的错误
        self.std_[self.std_ == 0] = 1.0

    def transform(self, X):
        # 先把X里的NaN替换成训练集的均值
        X = np.where(np.isnan(X), self.mean_, X)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)