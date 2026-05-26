"""
模块：utils.transformers
用途：特征预处理工具，包含标准化器和统一预处理流水线
"""

import numpy as np
import pandas as pd


class CustomStandardScaler:
    """
    自定义标准化器：z = (x - mean) / std
    严格遵循 scikit-learn 的 Transformer API
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray):
        """
        计算训练集的均值和标准差
        """
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # 防止除零：如果标准差为 0，则设为 1（该特征不变）
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        使用已保存的均值和标准差进行标准化
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("必须先调用 fit 或 fit_transform")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        先拟合再转换
        """
        self.fit(X)
        return self.transform(X)


def preprocess_features(df: pd.DataFrame, target_col: str,
                        cat_cols: list, fit_mode: bool = True,
                        stats: dict = None, has_target: bool = True) -> tuple:
    """
    统一的特征预处理流水线：缺失值填补 + One-Hot 编码 + 异常值缩尾 + 标准化。

    严格遵循无泄露原则：
    - fit_mode=True 时从训练集学习所有参数（中位数、众数、分位数、均值/标准差）
    - fit_mode=False 时使用 stats 中已保存的参数，仅做 transform

    Parameters
    ----------
    df : pd.DataFrame
        包含特征和目标列的数据
    target_col : str
        目标变量列名
    cat_cols : list
        类别特征列名列表
    fit_mode : bool
        True=训练模式（学习参数），False=推断模式（使用已有参数）
    stats : dict
        fit_mode=False 时必须提供，包含 fill_values / quantile_bounds / scaler
    has_target : bool
        数据是否包含目标列（test.csv 可能无目标列）

    Returns
    -------
    (X, y, feature_names, stats_dict) : tuple
        X: 预处理后的特征矩阵, y: 目标变量（无目标时为 None）,
        feature_names: 特征名列表, stats_dict: 学习到的参数
    """
    df = df.copy()

    # 分离目标变量
    if has_target and target_col in df.columns:
        y = df[target_col].values.astype(np.float64)
        df = df.drop(columns=[target_col])
    else:
        y = None
        if target_col in df.columns:
            df = df.drop(columns=[target_col])

    # 布尔/整数列转 float（避免后续 quantile/winsorize 报错）
    bool_cols = df.select_dtypes(include=['bool']).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(float)
    int_cols = df.select_dtypes(include=['int']).columns
    if len(int_cols) > 0:
        df[int_cols] = df[int_cols].astype(float)

    # ---- 缺失值填补：数值列用中位数，类别列用众数 ----
    num_cols = [c for c in df.columns if c not in cat_cols]
    if fit_mode:
        fill_values = {}
        for c in num_cols:
            fill_values[c] = df[c].median()
        for c in cat_cols:
            fill_values[c] = df[c].mode()[0] if not df[c].mode().empty else 'unknown'
    else:
        fill_values = stats['fill_values']

    for c in df.columns:
        if df[c].isnull().any():
            df[c] = df[c].fillna(fill_values[c])

    # ---- One-Hot 编码（drop_first 避免虚拟变量陷阱）----
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=float)

    feature_names = list(df.columns)

    # ---- 异常值缩尾（99 分位数 Winsorization）----
    if fit_mode:
        quantile_bounds = {}
        for c in feature_names:
            upper = df[c].quantile(0.99)
            quantile_bounds[c] = upper
    else:
        quantile_bounds = stats['quantile_bounds']

    for c in feature_names:
        upper = quantile_bounds[c]
        df.loc[df[c] > upper, c] = upper

    X = df.values.astype(np.float64)

    # ---- 标准化（z-score）----
    if fit_mode:
        scaler = CustomStandardScaler()
        X = scaler.fit_transform(X)
    else:
        scaler = stats['scaler']
        X = scaler.transform(X)

    stats_dict = {
        'fill_values': fill_values,
        'quantile_bounds': quantile_bounds,
        'scaler': scaler,
    }

    return X, y, feature_names, stats_dict