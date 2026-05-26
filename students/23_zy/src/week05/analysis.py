import numpy as np
# 强制设置为无界面模式，只保存图片，不弹出窗口
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=== 程序开始运行 ===")

def calculate_theoretical_cov(X, sigma):
    print("正在计算理论协方差矩阵...")
    X_features = X[:, 1:]
    XtX = X_features.T @ X_features
    XtX_inv = np.linalg.inv(XtX)
    return sigma**2 * XtX_inv

if __name__ == "__main__":
    print("正在加载 simulation_results.npz 文件...")
    try:
        data = np.load("simulation_results.npz")
    except FileNotFoundError:
        print("❌ 错误：找不到 simulation_results.npz 文件！请先运行 simulation.py")
        exit()

    beta_hat_A = data["beta_hat_A"]
    beta_hat_B = data["beta_hat_B"]
    X_A = data["X_A"]
    X_B = data["X_B"]
    sigma = data["sigma"]
    print("文件加载成功！")

    print("\n===== 1. 实验 B 经验协方差矩阵 =====")
    empirical_cov = np.cov(beta_hat_B.T)
    print(empirical_cov)

    print("\n===== 2. 实验 B 理论协方差矩阵 =====")
    theoretical_cov = calculate_theoretical_cov(X_B, sigma)
    print(theoretical_cov)

    print("\n正在生成散点图...")
    plt.figure(figsize=(8, 8))
    plt.scatter(beta_hat_A[:, 0], beta_hat_A[:, 1], alpha=0.5, label="实验A (ρ=0.0, 正交)", color="blue")
    plt.scatter(beta_hat_B[:, 0], beta_hat_B[:, 1], alpha=0.5, label="实验B (ρ=0.99, 共线)", color="red")
    plt.scatter(5, 3, color="black", marker="*", s=200, label="真实系数 β=(5,3)")
    plt.xlabel(r"$\hat{\beta}_1$")
    plt.ylabel(r"$\hat{\beta}_2$")
    plt.title("正交 vs 共线性：回归系数估计值分布")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(3, color="gray", linestyle="--", alpha=0.5)
    plt.axvline(5, color="gray", linestyle="--", alpha=0.5)

    # 保存图片，不弹出窗口
    plt.savefig("beta_scatter.png", dpi=300, bbox_inches="tight")
    print("✅ 散点图已保存为 beta_scatter.png")

    print("\n=== 程序运行结束 ===")