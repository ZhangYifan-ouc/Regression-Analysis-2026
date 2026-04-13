import numpy as np
from data_generator import generate_fixed_design_matrix
from simulation import run_monte_carlo
from analysis import verify_covariance_matrix, plot_covariance_ellipses


def main():
    # ========== 全局实验配置 (Configuration) ==========
    N_SAMPLES = 200  # 样本数量
    N_SIMULATIONS = 1000  # 蒙特卡洛模拟次数
    TRUE_BETA = np.array([1.0, 5.0, 3.0])  # 真实参数 [beta_0, beta_1, beta_2]
    SIGMA = 2.0  # 噪音标准差
    RNG = np.random.default_rng(seed=2026)  # 随机数生成器（保证可重现性）
    
    print("="*75)
    print("Week 05: 蒙特卡洛模拟 - 共线性对参数估计的影响")
    print("="*75)
    print(f"\n📊 实验配置:")
    print(f"  - 样本数量: {N_SAMPLES}")
    print(f"  - 模拟次数: {N_SIMULATIONS}")
    print(f"  - 真实参数 β: {TRUE_BETA}")
    print(f"  - 噪音标准差 σ: {SIGMA}")
    
    # ========== 实验 A: 纯净的世界 (正交特征 rho = 0.0) ==========
    print(f"\n{'='*75}")
    print("🔷 实验 A: 正交特征 (ρ=0.0) - 圆形散点云")
    print(f"{'='*75}")
    
    rho_A = 0.0
    X_A = generate_fixed_design_matrix(N_SAMPLES, rho_A, RNG)
    print(f"✓ 生成固定设计矩阵 X (ρ={rho_A})")
    
    beta_hats_A = run_monte_carlo(X_A, TRUE_BETA, SIGMA, N_SIMULATIONS, RNG)
    print(f"✓ 执行蒙特卡洛循环 ({N_SIMULATIONS} 次拟合)")
    
    theoretical_cov_A, empirical_cov_A = verify_covariance_matrix(X_A, beta_hats_A, SIGMA)
    
    print(f"\n理论协方差矩阵 (σ² (X^T X)^{{-1}})，仅显示 β₁, β₂ 的2×2子矩阵:")
    print(theoretical_cov_A[1:, 1:])
    
    print(f"\n经验协方差矩阵 (from {N_SIMULATIONS} simulations):")
    print(empirical_cov_A[1:, 1:])
    
    # 计算相关系数
    corr_A = np.corrcoef(beta_hats_A[:, 1], beta_hats_A[:, 2])[0, 1]
    print(f"\n相关系数: corr(β̂₁, β̂₂) = {corr_A:+.6f}")
    
    # ========== 实验 B: 被诅咒的世界 (多重共线性 rho = 0.99) ==========
    print(f"\n{'='*75}")
    print("🔶 实验 B: 高度共线特征 (ρ=0.99) - 倾斜椭圆形散点云")
    print(f"{'='*75}")
    
    rho_B = 0.99
    X_B = generate_fixed_design_matrix(N_SAMPLES, rho_B, RNG)
    print(f"✓ 生成固定设计矩阵 X (ρ={rho_B})")
    
    beta_hats_B = run_monte_carlo(X_B, TRUE_BETA, SIGMA, N_SIMULATIONS, RNG)
    print(f"✓ 执行蒙特卡洛循环 ({N_SIMULATIONS} 次拟合)")
    
    theoretical_cov_B, empirical_cov_B = verify_covariance_matrix(X_B, beta_hats_B, SIGMA)
    
    print(f"\n理论协方差矩阵 (σ² (X^T X)^{{-1}})，仅显示 β₁, β₂ 的2×2子矩阵:")
    print(theoretical_cov_B[1:, 1:])
    
    print(f"\n经验协方差矩阵 (from {N_SIMULATIONS} simulations):")
    print(empirical_cov_B[1:, 1:])
    
    # 计算相关系数
    corr_B = np.corrcoef(beta_hats_B[:, 1], beta_hats_B[:, 2])[0, 1]
    print(f"\n相关系数: corr(β̂₁, β̂₂) = {corr_B:+.6f}")
    
    # ========== 对比分析 ==========
    print(f"\n{'='*75}")
    print("📈 对比分析: 共线性的灾难性结果")
    print(f"{'='*75}")
    
    var_beta1_A = empirical_cov_A[1, 1]
    var_beta1_B = empirical_cov_B[1, 1]
    var_beta2_A = empirical_cov_A[2, 2]
    var_beta2_B = empirical_cov_B[2, 2]
    
    print(f"\n方差膨胀 (Variance Inflation):")
    print(f"  Var(β̂₁): {var_beta1_B / var_beta1_A:.2f}x 倍增长 (从 {var_beta1_A:.4f} → {var_beta1_B:.4f})")
    print(f"  Var(β̂₂): {var_beta2_B / var_beta2_A:.2f}x 倍增长 (从 {var_beta2_A:.4f} → {var_beta2_B:.4f})")
    
    print(f"\n参数相关性的灾难性转变:")
    print(f"  正交特征 (ρ=0.0) 下:    corr(β̂₁, β̂₂) = {corr_A:+.6f} (接近零)  ← 独立!")
    print(f"  共线特征 (ρ=0.99) 下:  corr(β̂₁, β̂₂) = {corr_B:+.6f}           ← 强烈负相关!")
    print(f"\n💥 共线性不仅膨胀方差，更关键的是摧毁了参数估计的独立性!")
    
    # ========== 终极可视化 ==========
    print(f"\n{'='*75}")
    print("📊 生成对比散点图...")
    print(f"{'='*75}")
    
    plot_covariance_ellipses(beta_hats_A, beta_hats_B, TRUE_BETA)
    
    print(f"\n✅ 实验流水线执行完毕！")
    print(f"   - 图表已保存为 'beta_estimates_scatter.png'")
    print(f"   - 理论与经验协方差矩阵已打印到控制台")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    main()