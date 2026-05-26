"""
模块：utils.models
核心机器学习模型：解析解OLS（正规方程）和梯度下降OLS。
"""
import numpy as np
from scipy import stats


class AnalyticalOLS:
    """
    解析解线性回归（正规方程）。
    假设 X 已经包含截距列（全1），或由外部添加。
    """
    def __init__(self):
        self.coef_ = None          # 回归系数
        self.cov_matrix_ = None    # 协方差矩阵
        self.sigma2_ = None        # 残差方差
        self.df_resid_ = None      # 残差自由度
        self.resid_ = None          # 残差
        self._feature_names = None  # 特征名称（可选）

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names=None):
        """
        拟合模型：β = (XᵀX)⁻¹ Xᵀy
        """
        n, p = X.shape
        if feature_names is not None:
            self._feature_names = feature_names
        else:
            self._feature_names = [f'X{i}' for i in range(p)]

        # 正规方程
        XtX = X.T @ X
        Xty = X.T @ y
        try:
            self.coef_ = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            # 矩阵奇异时使用伪逆
            self.coef_ = np.linalg.pinv(XtX) @ Xty

        # 残差与自由度
        y_pred = X @ self.coef_
        self.resid_ = y - y_pred
        self.df_resid_ = n - p
        sse = np.sum(self.resid_ ** 2)
        self.sigma2_ = sse / self.df_resid_

        # 协方差矩阵
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(XtX)
        self.cov_matrix_ = self.sigma2_ * XtX_inv
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测 y = Xβ"""
        return X @ self.coef_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """决定系数 R² = 1 - SSE/SST"""
        y_pred = self.predict(X)
        sse = np.sum((y - y_pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        return 1 - sse / sst if sst != 0 else 0.0

    def f_test(self, C: np.ndarray, d: np.ndarray) -> dict:
        """一般线性假设检验 Cβ = d，返回 F 统计量和 p 值"""
        if self.coef_ is None:
            raise ValueError("模型尚未拟合")
        C = np.atleast_2d(C)
        q = C.shape[0]
        diff = C @ self.coef_ - d
        C_cov_Ct = C @ self.cov_matrix_ @ C.T
        try:
            inv = np.linalg.inv(C_cov_Ct)
            f_stat = (diff.T @ inv @ diff) / q
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(C_cov_Ct)
            f_stat = (diff.T @ inv @ diff) / q
        p_value = 1 - stats.f.cdf(f_stat, q, self.df_resid_)
        return {'f_stat': f_stat, 'p_value': p_value, 'df_num': q, 'df_den': self.df_resid_}

    def summary(self) -> str:
        """打印模型摘要"""
        if self.coef_ is None:
            return "模型尚未拟合"
        std_errors = np.sqrt(np.diag(self.cov_matrix_))
        t_stats = self.coef_ / std_errors
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), self.df_resid_))
        lines = []
        lines.append("=" * 70)
        lines.append("                     AnalyticalOLS 回归结果")
        lines.append("=" * 70)
        lines.append(f"残差自由度: {self.df_resid_}")
        lines.append(f"σ̂² (残差方差): {self.sigma2_:.6f}\n")
        lines.append(f"{'变量':<15} {'系数':>12} {'标准误':>12} {'t统计量':>12} {'p值':>12}")
        lines.append("-" * 70)
        for i, (coef, se, t, p) in enumerate(zip(self.coef_, std_errors, t_stats, p_values)):
            name = self._feature_names[i] if i < len(self._feature_names) else f"X{i}"
            lines.append(f"{name:<15} {coef:>12.6f} {se:>12.6f} {t:>12.6f} {p:>12.4e}")
        lines.append("=" * 70)
        return "\n".join(lines)


class GradientDescentOLS:
    """
    梯度下降线性回归。
    支持全批量 (full_batch) 和小批量 (mini_batch)。
    """
    def __init__(
        self,
        learning_rate: float = 0.01,
        tol: float = 1e-5,
        max_iter: int = 1000,
        gd_type: str = "full_batch",
        batch_fraction: float = 0.1,
    ):
        self.learning_rate = learning_rate
        self.tol = tol
        self.max_iter = max_iter
        self.gd_type = gd_type
        self.batch_fraction = batch_fraction
        self.coef_ = None
        self.loss_history_ = []

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int = 42):
        """梯度下降拟合模型，X 应已包含截距列"""
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        self.loss_history_ = []
        rng = np.random.default_rng(seed)

        # 确定批量大小
        if self.gd_type == "full_batch":
            batch_size = n_samples
        elif self.gd_type == "mini_batch":
            batch_size = max(1, int(n_samples * self.batch_fraction))
        else:
            raise ValueError("gd_type 必须是 'full_batch' 或 'mini_batch'")

        for epoch in range(self.max_iter):
            # 小批量采样
            if self.gd_type == "mini_batch":
                indices = rng.choice(n_samples, size=batch_size, replace=False)
                X_batch = X[indices]
                y_batch = y[indices]
            else:
                X_batch = X
                y_batch = y

            # 计算梯度 (MSE)
            y_pred_batch = X_batch @ self.coef_
            error = y_pred_batch - y_batch
            gradient = (2 / len(X_batch)) * (X_batch.T @ error)

            # 更新系数
            self.coef_ -= self.learning_rate * gradient

            # 记录全量损失（用于收敛判断）
            y_pred_full = X @ self.coef_
            mse = np.mean((y - y_pred_full) ** 2)
            self.loss_history_.append(mse)

            # 收敛检查
            if epoch > 0:
                delta = abs(self.loss_history_[-1] - self.loss_history_[-2])
                if delta < self.tol:
                    break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        sse = np.sum((y - y_pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        return 1 - sse / sst if sst != 0 else 0.0