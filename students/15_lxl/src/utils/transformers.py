"""模块: utils.transformers
用途: 数据预处理转换器 —— CustomStandardScaler、CustomSimpleImputer。

遵循 Transformer 接口规范:
    fit(X)         — 学习参数，返回 self
    transform(X)   — 用学到的参数对 X 进行变换
    fit_transform(X) — fit + transform 一步完成
"""
import numpy as np


class CustomStandardScaler:
    """手写标准化器 (StandardScaler)。

    将每个特征缩放为 均值=0、标准差=1 的分布:
        X_scaled = (X - mean) / std

    使用场景:
        在交叉验证中，必须仅用训练集的 mean/std 来 fit，
        然后用同样的参数去 transform 验证集，防止数据泄漏。
    """

    def __init__(self):
        # 学到的参数（fit 后才有值）
        self.mean_ = None  # 每个特征的均值
        self.std_ = None   # 每个特征的标准差

    def fit(self, X: np.ndarray) -> "CustomStandardScaler":
        """仅计算并保存训练数据的均值和标准差。

        参数:
            X: 形状为 (n_samples, n_features) 的特征矩阵。

        返回:
            self，支持链式调用。
        """
        # axis=0: 沿列方向计算，得到每个特征的统计量
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # 防止标准差为 0（常量特征）导致除以 0
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """使用 fit 阶段学到的参数对 X 进行标准化。

        参数:
            X: 形状为 (n_samples, n_features) 的特征矩阵。

        返回:
            标准化后的特征矩阵。

        异常:
            如果在调用 transform 之前没有先调用 fit，会抛出 AttributeError。
        """
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """先 fit 再 transform，一步完成。

        参数:
            X: 形状为 (n_samples, n_features) 的特征矩阵。

        返回:
            标准化后的特征矩阵。
        """
        return self.fit(X).transform(X)


class CustomSimpleImputer:
    """手写缺失值填充器 (SimpleImputer)。

    用每列的统计量（均值或中位数）填充 NaN。

    使用场景:
        在交叉验证中，必须仅用训练集的统计量来 fit，
        然后用同样的参数去 transform 验证集，防止数据泄漏。
    """

    def __init__(self, strategy: str = "mean"):
        """参数:
            strategy: 填充策略，"mean"（均值）或 "median"（中位数）。
        """
        self.strategy = strategy
        self.fill_values_ = None  # 每列的填充值（fit 后才有值）

    def fit(self, X: np.ndarray) -> "CustomSimpleImputer":
        """计算每列的填充值（均值或中位数），仅使用非 NaN 的样本。"""
        if self.strategy == "mean":
            self.fill_values_ = np.nanmean(X, axis=0)
        elif self.strategy == "median":
            self.fill_values_ = np.nanmedian(X, axis=0)
        else:
            raise ValueError("strategy must be 'mean' or 'median'")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """用 fit 阶段学到的填充值替换 NaN。"""
        X_filled = X.copy()
        for j in range(X.shape[1]):
            mask = np.isnan(X_filled[:, j])
            X_filled[mask, j] = self.fill_values_[j]
        return X_filled

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """先 fit 再 transform，一步完成。"""
        return self.fit(X).transform(X)
