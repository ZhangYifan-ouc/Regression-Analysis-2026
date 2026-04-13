"""
架构说明：表现层与验证层。将纯数值转化为学术洞察。
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用Agg后端，不需要图形界面
import matplotlib.pyplot as plt

def verify_covariance_matrix(X: np.ndarray, beta_samples: np.ndarray, sigma: float):
    """
    将 1000 次模拟产生的"经验协方差"与公式推导出的"理论协方差"进行对齐。
    
    理论协方差：Var(beta_hat) = sigma^2 * (X^T X)^{-1}
    经验协方差：从 1000 个 beta_hat 样本的协方差矩阵
    
    两个矩阵应当惊人地一致（误差仅来自蒙特卡洛随机抽样）！
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, p)
        固定设计矩阵
    beta_samples : ndarray of shape (n_simulations, p)
        蒙特卡洛模拟得到的参数估计样本
    sigma : float
        真实噪音标准差
    """
    # 1. 计算理论协方差矩阵
    XTX_inv = np.linalg.inv(X.T @ X)
    theoretical_cov = sigma**2 * XTX_inv
    
    # 2. 计算经验协方差矩阵
    # np.cov() 需要输入形状为 (k, n)，其中 k 是变量数，n 是观测数
    empirical_cov = np.cov(beta_samples.T)  # T 是因为 beta_samples 形状为 (n_simulations, p)
    
    return theoretical_cov, empirical_cov


def plot_covariance_ellipses(beta_samples_ortho: np.ndarray, beta_samples_coll: np.ndarray, true_beta: np.ndarray):
    """
    将正交特征 (rho=0) 和共线特征 (rho=0.99) 的估计结果画在同一张 2D 散点图上。
    
    视觉对比：
    - 正交特征：圆形散点云
    - 共线特征：倾斜椭圆形散点云
    
    要求：
    - 使用不同的颜色和透明度 (alpha)
    - 标记真实的 Beta 靶心位置
    - 保存为 PNG 图片（非阻塞）
    
    Parameters
    ----------
    beta_samples_ortho : ndarray of shape (1000, 3)
        正交特征条件下的 1000 个 beta_hat 估计
    beta_samples_coll : ndarray of shape (1000, 3)
        共线特征条件下的 1000 个 beta_hat 估计
    true_beta : ndarray of shape (3,)
        真实参数值 [beta_0, beta_1, beta_2]
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ===== 实验 A: 正交特征 (rho = 0.0) =====
    # 提取 beta_1 和 beta_2 列（跳过常数项 beta_0）
    ax1 = axes[0]
    ax1.scatter(beta_samples_ortho[:, 1], beta_samples_ortho[:, 2], 
                alpha=0.4, s=20, color='blue', label='Estimates (ρ=0)')
    ax1.scatter(true_beta[1], true_beta[2], 
                color='red', s=300, marker='*', label='True β', zorder=5, edgecolors='darkred', linewidth=2)
    ax1.set_xlabel(r'$\hat{\beta}_1$', fontsize=13, fontweight='bold')
    ax1.set_ylabel(r'$\hat{\beta}_2$', fontsize=13, fontweight='bold')
    ax1.set_title('Experiment A: Orthogonal Features (ρ=0.0)\nCircular scatter cloud', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.axhline(y=true_beta[2], color='gray', linestyle=':', alpha=0.3)
    ax1.axvline(x=true_beta[1], color='gray', linestyle=':', alpha=0.3)
    
    # ===== 实验 B: 共线特征 (rho = 0.99) =====
    ax2 = axes[1]
    ax2.scatter(beta_samples_coll[:, 1], beta_samples_coll[:, 2], 
                alpha=0.4, s=20, color='darkorange', label='Estimates (ρ=0.99)')
    ax2.scatter(true_beta[1], true_beta[2], 
                color='red', s=300, marker='*', label='True β', zorder=5, edgecolors='darkred', linewidth=2)
    ax2.set_xlabel(r'$\hat{\beta}_1$', fontsize=13, fontweight='bold')
    ax2.set_ylabel(r'$\hat{\beta}_2$', fontsize=13, fontweight='bold')
    ax2.set_title('Experiment B: Highly Collinear Features (ρ=0.99)\nTilted ellipse (negative correlation)', 
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=11, loc='upper right')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.axhline(y=true_beta[2], color='gray', linestyle=':', alpha=0.3)
    ax2.axvline(x=true_beta[1], color='gray', linestyle=':', alpha=0.3)
    
    # ===== 设置整体布局 =====
    plt.tight_layout()
    
    # ===== 保存图片（非阻塞） =====
    plt.savefig('beta_estimates_scatter.png', dpi=300, bbox_inches='tight')
    print("✓ 散点图已保存为 'beta_estimates_scatter.png'")
    
    # ===== 显示（非阻塞） =====
    plt.show(block=False)