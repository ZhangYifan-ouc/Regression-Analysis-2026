"""
架构说明：这是蒙特卡洛模拟的心脏模块。
核心难点：如何在 1000 次甚至 100000 次循环中，将算法的时间复杂度降到最低？
"""
import numpy as np
from data_generator import generate_dynamic_response

def run_monte_carlo(X: np.ndarray, true_beta: np.ndarray, sigma: float, n_simulations: int, rng: np.random.Generator) -> np.ndarray:
    """
    执行蒙特卡洛循环，收集所有的 \hat{\beta}。
    
    我们在之前说过“求解参数必须用 `solve` 而非 `inv`”。
    如果在 for 循环里写 `np.linalg.solve(X.T @ X, X.T @ y)`，会导致每次循环都重新分解一遍 X^T X，极其浪费算力！
    
    【你的任务】：请将所有“不随循环改变的计算 (Loop-invariant computations)” 提取到循环外部！
    思考：你是要在循环外算出一个逆矩阵？还是要用 scipy 做一次 Cholesky 分解缓存起来？
    请在你的实验报告中说明你的性能优化选择及原因。
    """
    
    # ===== 循环外的"预计算" (Loop-invariant Precomputation) =====
    # 计算投影矩阵：P = (X^T X)^{-1} X^T
    # 这样 beta_hat = P @ y，避免在循环里重复求逆
    
    XTX_inv = np.linalg.inv(X.T @ X)
    P = XTX_inv @ X.T  # 形状为 (p, n_samples)
    
    # ===== 蒙特卡洛循环 =====
    beta_samples = []
    
    for _ in range(n_simulations):
        # 1. 生成一次新的随机观测值 y = X @ true_beta + epsilon
        y = generate_dynamic_response(X, true_beta, sigma, rng)
        
        # 2. 利用预计算的投影矩阵快速求出 beta_hat
        beta_hat = P @ y
        
        # 3. 存储结果
        beta_samples.append(beta_hat)
    
    return np.array(beta_samples)