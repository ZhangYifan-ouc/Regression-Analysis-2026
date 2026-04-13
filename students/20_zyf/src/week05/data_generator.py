"""
架构说明：本模块只负责“上帝视角”的数据生成。
核心要求：必须将“固定设计矩阵 (Fixed X)”与“动态噪音 (Epsilon)”严格分离！
"""
import numpy as np

def generate_fixed_design_matrix(n_samples: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    """
    生成两列特征 X1 和 X2，其相关系数为 rho。
    
    数学原理：
    - X1 ~ N(0, 1)
    - X2 = rho * X1 + sqrt(1 - rho^2) * Z，其中 Z ~ N(0, 1)
    - 这样保证 corr(X1, X2) = rho
    
    Parameters
    ----------
    n_samples : int
        样本数量
    rho : float
        X1 和 X2 之间的相关系数，范围 [0, 1]
    rng : np.random.Generator
        随机数生成器，保证可重现性
        
    Returns
    -------
    X : ndarray of shape (n_samples, 3)
        设计矩阵，形式为 [1, X1, X2]（包含常数项）
    """
    # 生成第一个特征
    X1 = rng.standard_normal(n_samples)
    
    # 根据相关系数生成第二个特征（线性组合）
    Z = rng.standard_normal(n_samples)
    X2 = rho * X1 + np.sqrt(1 - rho**2) * Z
    
    # 加上常数项（截距）列
    X = np.column_stack([np.ones(n_samples), X1, X2])
    
    return X

def generate_dynamic_response(X: np.ndarray, true_beta: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """
    基于传入的固定 X，生成一次带随机噪音的 y。
    
    数据生成过程 (DGP)：
    y = X @ true_beta + epsilon，其中 epsilon ~ N(0, sigma^2)
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, 3)
        固定的设计矩阵（包含常数项）
    true_beta : ndarray of shape (3,)
        真实的参数向量 [beta_0, beta_1, beta_2]
    sigma : float
        噪音的标准差
    rng : np.random.Generator
        随机数生成器
        
    Returns
    -------
    y : ndarray of shape (n_samples,)
        观测响应值
    """
    n_samples = X.shape[0]
    
    # 生成纯随机噪音
    epsilon = rng.standard_normal(n_samples) * sigma
    
    # 生成响应值
    y = X @ true_beta + epsilon
    
    return y