"""
模块：工具.诊断
用途：模型统计诊断工具
包含：方差膨胀因子计算、残差图、相关矩阵
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple


def calculate_vif(X: np.ndarray) -> List[float]:
    """
    计算每个特征的方差膨胀因子（VIF）
    
    公式：VIF_j = 1 / (1 - R_j^2)
    其中 R_j^2 是将第 j 个特征作为因变量对其他所有特征做回归得到的拟合优度
    
    参数：
        X: 特征矩阵，形状为 (n_samples, n_features)
        
    返回：
        vif_values: 每个特征的 VIF 值列表
    """
    n_samples, n_features = X.shape
    vif_values = []
    
    for i in range(n_features):
        y = X[:, i]
        X_others = np.delete(X, i, axis=1)
        X_others_with_intercept = np.column_stack([np.ones(n_samples), X_others])
        
        XTX = X_others_with_intercept.T @ X_others_with_intercept
        XTX_inv = np.linalg.inv(XTX + 1e-10 * np.eye(XTX.shape[0]))
        beta = XTX_inv @ (X_others_with_intercept.T @ y)
        
        y_pred = X_others_with_intercept @ beta
        
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
        vif = 1.0 / (1.0 - r_squared) if r_squared < 1.0 else float('inf')
        
        vif_values.append(vif)
    
    return vif_values


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residuals Diagnostics",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    绘制残差诊断图（加分功能）
    
    包含：
    - 残差 vs 拟合值散点图
    - 残差Q-Q图（正态性检验）
    - 残差直方图
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # 1. 残差 vs 拟合值
    axes[0].scatter(y_pred, residuals, alpha=0.5, edgecolors='none', s=30)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=1)
    axes[0].set_xlabel('Fitted Values', fontsize=10)
    axes[0].set_ylabel('Residuals', fontsize=10)
    axes[0].set_title('Residuals vs Fitted', fontsize=11)
    
    # 2. Q-Q图
    try:
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot (Normality Check)', fontsize=11)
        lines = axes[1].get_lines()
        if len(lines) > 0:
            lines[0].set_marker('o')
            lines[0].set_markersize(3)
            lines[0].set_alpha(0.5)
    except ImportError:
        axes[1].text(0.5, 0.5, 'scipy not installed\ncannot draw Q-Q plot', 
                    ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Q-Q Plot (scipy required)', fontsize=11)
    
    # 3. 残差直方图
    axes[2].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[2].axvline(x=0, color='r', linestyle='--', linewidth=1)
    axes[2].axvline(x=np.mean(residuals), color='orange', linestyle='-', linewidth=1)
    axes[2].set_xlabel('Residuals', fontsize=10)
    axes[2].set_ylabel('Frequency', fontsize=10)
    axes[2].set_title(f'Residuals Distribution\n(mean={np.mean(residuals):.4f}, std={np.std(residuals):.4f})', 
                      fontsize=11)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"残差图已保存至: {save_path}")
    
    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    title: str = "Correlation Matrix",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    绘制相关矩阵热力图（加分功能）
    
    参数：
        df: 包含特征和目标的DataFrame
        feature_cols: 特征列名列表
        target_col: 目标列名
        title: 图表标题
        save_path: 保存路径（可选）
        figsize: 图表大小
        
    返回：
        matplotlib Figure对象
    """
    # 选择特征+目标
    plot_cols = feature_cols + [target_col]
    corr_matrix = df[plot_cols].corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制热力图
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlation Coefficient', fontsize=10)
    
    # 设置标签
    ax.set_xticks(range(len(plot_cols)))
    ax.set_yticks(range(len(plot_cols)))
    ax.set_xticklabels(plot_cols, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(plot_cols, fontsize=9)
    
    # 添加数值标注
    for i in range(len(plot_cols)):
        for j in range(len(plot_cols)):
            text_color = "white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black"
            ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                   ha="center", va="center", 
                   color=text_color,
                   fontsize=8)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"相关矩阵图已保存至: {save_path}")
    
    return fig


def analyze_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    计算残差的统计特征（加分功能）
    
    返回：
        包含残差统计信息的字典
    """
    residuals = y_true - y_pred
    standardized_residuals = residuals / np.std(residuals) if np.std(residuals) > 0 else residuals
    
    stats_dict = {
        'mean': float(np.mean(residuals)),
        'std': float(np.std(residuals)),
        'skew': float(pd.Series(residuals).skew()),
        'kurtosis': float(pd.Series(residuals).kurtosis()),
        'min': float(np.min(residuals)),
        'max': float(np.max(residuals)),
        'q1': float(np.percentile(residuals, 25)),
        'median': float(np.percentile(residuals, 50)),
        'q3': float(np.percentile(residuals, 75)),
        'iqr': float(np.percentile(residuals, 75) - np.percentile(residuals, 25)),
        'pct_outside_2std': float(np.mean(np.abs(standardized_residuals) > 2) * 100),
        'pct_outside_3std': float(np.mean(np.abs(standardized_residuals) > 3) * 100),
    }
    
    return stats_dict