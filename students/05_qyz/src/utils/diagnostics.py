"""
模型诊断工具箱 (Model Diagnostics Toolbox)
功能：
1. 计算 VIF（方差膨胀因子）检测多重共线性
2. 彩色终端输出警告
3. 残差图、QQ图、相关矩阵热力图
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.models import AnalyticalOLS
from pathlib import Path


def calculate_vif(X: np.ndarray) -> list:
    """
    计算每个特征的方差膨胀因子 (VIF)
    
    VIF 公式: VIF = 1 / (1 - R²_j)
    其中 R²_j 是第 j 个特征对其他所有特征的回归拟合优度
    
    Parameters:
    -----------
    X : np.ndarray
        特征矩阵（应已包含截距或已标准化）
    
    Returns:
    --------
    list : 每个特征的 VIF 值列表
    """
    n_features = X.shape[1]
    vif_values = []
    
    for j in range(n_features):
        # 第 j 列作为目标变量
        y_j = X[:, j]
        # 其他列作为特征
        X_j = np.delete(X, j, axis=1)
        
        # 添加截距项
        X_with_intercept = np.column_stack([np.ones(len(X_j)), X_j])
        
        try:
            # 使用 OLS 拟合
            model = AnalyticalOLS(fit_intercept=False)  # 已手动添加截距
            model.fit(X_with_intercept, y_j)
            
            # 计算 R²
            y_pred = model.predict(X_with_intercept)
            sse = np.sum((y_j - y_pred) ** 2)
            sst = np.sum((y_j - np.mean(y_j)) ** 2)
            r_squared = 1 - sse / sst
            
            # 计算 VIF
            vif = 1 / (1 - r_squared) if r_squared < 0.999 else float('inf')
            
        except Exception as e:
            print(f"警告: 计算第 {j} 个特征时出错: {e}")
            vif = float('inf')
        
        vif_values.append(vif)
    
    return vif_values


def calculate_vif_dataframe(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    计算 DataFrame 中特征的 VIF，返回带列名的结果
    """
    X = df[feature_cols].values
    vifs = calculate_vif(X)
    
    results = pd.DataFrame({
        '特征': feature_cols,
        'VIF': vifs
    })
    
    return results


def print_vif_warning(vif_results: pd.DataFrame, threshold: float = 10):
    """
    彩色打印 VIF 结果，并给出警告
    
    终端颜色代码：
    \033[91m = 红色
    \033[93m = 黄色
    \033[0m = 重置
    """
    print("\n" + "="*60)
    print("多重共线性诊断 (VIF 分析)")
    print("="*60)
    print(f"VIF 阈值: {threshold} (超过此值表示严重共线性)")
    print("-"*60)
    
    has_severe = False
    
    for _, row in vif_results.iterrows():
        feature = row['特征']
        vif = row['VIF']
        
        if vif > threshold:
            print(f"\033[91m  {feature}: VIF = {vif:.2f} (严重共线性！)\033[0m")
            has_severe = True
        elif vif > 5:
            print(f"\033[93m {feature}: VIF = {vif:.2f} (中度共线性)\033[0m")
        else:
            print(f"   {feature}: VIF = {vif:.2f}")
    
    if has_severe:
        print("\n" + "="*60)
        print("\033[91m 警告！以下特征引发严重多重共线性:\033[0m")
        severe_features = vif_results[vif_results['VIF'] > threshold]['特征'].tolist()
        print(f"\033[91m   {severe_features}\033[0m")
        print("\033[91m   建议: 考虑删除高度相关的特征或使用正则化方法\033[0m")
        print("="*60)
    else:
        print("\n未检测到严重多重共线性")
    
    return has_severe


# ----------------------------------------------------------------
# 以下是新增的 3 个诊断函数
# ----------------------------------------------------------------

def plot_residuals(y_true, y_pred, save_name="residuals.png"):
    """绘制残差图：残差 vs 预测值"""
    res = y_true - y_pred
    plt.figure(figsize=(10, 5))
    plt.scatter(y_pred, res, alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted Values")
    
    res_dir = Path(__file__).parent.parent / "src" / "week11" / "results"
    res_dir.mkdir(exist_ok=True)
    plt.savefig(res_dir / save_name, dpi=150)
    plt.close()


def plot_qq_residuals(y_true, y_pred, save_name="qq_plot.png"):
    """绘制残差正态 Q-Q 图"""
    import scipy.stats as stats
    res = y_true - y_pred
    plt.figure(figsize=(8, 8))
    stats.probplot(res, plot=plt)
    plt.title("Residual Q-Q Plot")
    
    res_dir = Path(__file__).parent.parent / "src" / "week11" / "results"
    res_dir.mkdir(exist_ok=True)
    plt.savefig(res_dir / save_name, dpi=150)
    plt.close()


def plot_correlation_matrix(df, save_name="corr_matrix.png"):
    """绘制特征相关矩阵热力图"""
    df_numeric = df.select_dtypes(include=[np.number])
    corr = df_numeric.corr()
    
    plt.figure(figsize=(12, 10))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    
    res_dir = Path(__file__).parent.parent / "src" / "week11" / "results"
    res_dir.mkdir(exist_ok=True)
    plt.savefig(res_dir / save_name, dpi=150)
    plt.close()