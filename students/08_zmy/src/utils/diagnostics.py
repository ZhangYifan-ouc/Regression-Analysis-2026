import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def calculate_vif(X: np.ndarray) -> list:
    """
    计算每个特征的方差膨胀因子（VIF）。
    参数:
        X: 形状 (n_samples, n_features) 的特征矩阵（不应包含截距列）
    返回:
        vif_list: 每个特征的VIF值列表
    """
    n_features = X.shape[1]
    vif_list = []
    for i in range(n_features):
        y_i = X[:, i]
        X_i = np.delete(X, i, axis=1)
        if X_i.shape[1] == 0:
            vif_list.append(float('inf'))
            continue
        model = LinearRegression().fit(X_i, y_i)
        r2 = model.score(X_i, y_i)
        vif = 1 / (1 - r2) if r2 < 1 else float('inf')
        vif_list.append(vif)
    return vif_list

def plot_residuals(y_true, y_pred, save_path):
    """残差图（真实值 vs 残差）"""
    residuals = y_true - y_pred
    plt.figure(figsize=(8,5))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_correlation_matrix(X, feature_names, save_path):
    import matplotlib.pyplot as plt
    import numpy as np
    corr = np.corrcoef(X.T)
    fig, ax = plt.subplots(figsize=(10,8))
    im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)   # 使用彩色映射
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_yticks(np.arange(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=90)
    ax.set_yticklabels(feature_names)
    plt.colorbar(im, label='Correlation coefficient')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()