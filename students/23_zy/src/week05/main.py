# 第5周：多重共线性蒙特卡洛模拟实验
# 23_zy

from data_generator import generate_correlated_data
from simulation import run_simulation
from analysis import print_summary

if __name__ == "__main__":
    # 实验参数
    n_simulations = 1000  # 模拟次数
    n_samples = 100       # 样本量
    rho = 0.9             # 相关系数（多重共线性强度）
    
    print("=" * 60)
    print("开始多重共线性蒙特卡洛模拟实验")
    print(f"模拟次数: {n_simulations}")
    print(f"样本量: {n_samples}")
    print(f"自变量相关系数 ρ: {rho}")
    print("=" * 60)
    print()

    # 运行模拟
    results = run_simulation(n_simulations, n_samples, rho)
    
    # 输出结果
    print_summary(results)