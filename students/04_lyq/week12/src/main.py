import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# 你自己的工具（沿用 Week11 即可）
from utils.metrics import calculate_rmse, calculate_mae

# ====================== 路径设置 ==========================
BASE = "src"
FIG_DIR = os.path.join(BASE, "results/figures")
REPORT_PATH = os.path.join(BASE, "results/summary.md")
os.makedirs(FIG_DIR, exist_ok=True)

# ====================== 全局参数 ==========================
SEED = 42
np.random.seed(SEED)
N_SAMPLES = 200
TEST_SIZE = 0.3
MAX_DEGREE = 18
N_REPEAT = 12
variance_result = {}
# ====================== 数据生成 ==========================
def true_function(x):
    return np.sin(1.5 * np.pi * x) + x

def generate_data(n=200, noise=0.25):
    x = np.linspace(-1, 1, n)
    y = true_function(x) + noise * np.random.randn(n)
    x = x.reshape(-1, 1)
    return x, y

# ====================== 模型工具 ==========================
def poly_model(degree=1):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression())

# ====================== Task A：3个模型对比 =================
def run_candidate_models():
    x, y = generate_data()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=SEED
    )
    degrees = [1, 4, 15]

    x_plot = np.linspace(-1, 1, 200).reshape(-1, 1)
    y_true = true_function(x_plot)

    plt.figure(figsize=(16, 4))
    for i, d in enumerate(degrees):
        model = poly_model(d)
        model.fit(x_train, y_train)
        y_plot = model.predict(x_plot)
        tr_rmse = calculate_rmse(y_train, model.predict(x_train))
        te_rmse = calculate_rmse(y_test, model.predict(x_test))

        plt.subplot(1, 3, i+1)
        plt.scatter(x_train, y_train, s=10, color="gray", alpha=0.5)
        plt.scatter(x_test, y_test, s=10, color="red", alpha=0.5)
        plt.plot(x_plot, y_true, "k--", label="true")
        plt.plot(x_plot, y_plot, "b-", label=f"deg={d}")
        plt.title(f"Deg {d}\nTrain RMSE={tr_rmse:.2f}\nTest RMSE={te_rmse:.2f}")
        plt.grid(alpha=0.3)
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "candidate_models.png"), dpi=150)
    plt.close()

# ====================== Task B：误差曲线 ==================
def run_error_curve():
    x, y = generate_data()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=SEED
    )
    degrees = list(range(1, MAX_DEGREE+1))
    tr_rmses, te_rmses = [], []

    for d in degrees:
        model = poly_model(d)
        model.fit(x_train, y_train)
        tr = calculate_rmse(y_train, model.predict(x_train))
        te = calculate_rmse(y_test, model.predict(x_test))
        tr_rmses.append(tr)
        te_rmses.append(te)

    plt.figure(figsize=(9, 5))
    plt.plot(degrees, tr_rmses, "o-", label="Train RMSE")
    plt.plot(degrees, te_rmses, "o-", label="Test RMSE")
    plt.xlabel("Degree (Complexity)")
    plt.ylabel("RMSE")
    plt.title("Bias-Variance Tradeoff")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(FIG_DIR, "error_curves.png"), dpi=150)
    plt.close()
    return degrees, tr_rmses, te_rmses

# ====================== Task C：Variance 可视化 =============
def run_variance_demo():
    x_plot = np.linspace(-1, 1, 200).reshape(-1, 1)
    y_true = true_function(x_plot)
    degrees = [2, 15]
    colors = ["darkred", "darkblue"]

    plt.figure(figsize=(10, 4))
    for i, d in enumerate(degrees):
        predictions = []
        for _ in range(N_REPEAT):
            x, y = generate_data()
            model = poly_model(d)
            model.fit(x, y)
            predictions.append(model.predict(x_plot))

        mean_pred = np.mean(predictions, axis=0)
        std_all = np.std(predictions, axis=0)
        mean_std = np.mean(std_all)
        max_std = np.max(std_all)

        # 👉 新加：自动存到全局变量
        variance_result[d] = {"mean": mean_std, "max": max_std}

        plt.subplot(1, 2, i+1)
        for yp in predictions:
            plt.plot(x_plot, yp, color=colors[i], alpha=0.3, linewidth=0.8)
        plt.plot(x_plot, y_true, "k--", linewidth=2, label="True")
        plt.plot(x_plot, mean_pred, color=colors[i], linewidth=2, label=f"Deg {d}")
        plt.title(f"Degree {d}\nMean std = {mean_std:.3f}")  # 👉 改成 mean_std
        plt.legend()
        plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "variance_demo.png"), dpi=150)
    plt.close()


# ====================== Task D：RMSE vs MAE 异常值 ==========
def run_loss_outlier():
    y_true = np.array([1, 2, 3, 4, 5, 6, 7])
    y_clean = np.array([1.1, 2.1, 2.9, 3.8, 5.2, 5.9, 7.1])
    y_outlier = y_clean.copy()
    y_outlier[-1] = 18

    res = {
        "clean": {
            "RMSE": calculate_rmse(y_true, y_clean),
            "MAE": calculate_mae(y_true, y_clean)
        },
        "outlier": {
            "RMSE": calculate_rmse(y_true, y_outlier),
            "MAE": calculate_mae(y_true, y_outlier)
        }
    }

    plt.figure(figsize=(8, 4))
    plt.bar(["Clean RMSE", "Clean MAE", "Outlier RMSE", "Outlier MAE"],
            [res["clean"]["RMSE"], res["clean"]["MAE"],
             res["outlier"]["RMSE"], res["outlier"]["MAE"]],
            color=["g", "g", "r", "r"])
    plt.title("RMSE is Sensitive to Outliers")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(FIG_DIR, "loss_outlier_comparison.png"), dpi=150)
    plt.close()
    return res

# ====================== 输出报告 ==========================
def write_summary(degrees, tr, te, loss_res):
    best_idx = np.argmin(te)
    best_deg = degrees[best_idx]
    best_te = te[best_idx]

    # 计算泛化差距 generalization gap
    gaps = [t_e - t_r for t_r, t_e in zip(tr, te)]

    # ===================== B3 自动生成表格 =====================
    table_lines = "| degree | Train RMSE | Test RMSE | Gen Gap |\n"
    table_lines += "|--------|------------|-----------|---------|\n"
    for d, train_r, test_r, gap in zip(degrees, tr, te, gaps):
        table_lines += f"| {d:2d}    | {train_r:.3f}      | {test_r:.3f}     | {gap:.3f}  |\n"

    text = f"""# Week12 Bias-Variance Lab Report

## A3 模型拟合判断
1. **欠拟合模型**：degree = 1，模型结构简单、表达能力不足，无法捕捉数据非线性特征，训练误差与测试误差均偏高。
2. **过拟合模型**：degree = 15，模型复杂度过高，过度学习训练集噪声，曲线抖动严重，泛化能力极差。
3. **上线最优选择**：degree = 4。该模型复杂度适中，有效拟合真实数据规律，测试误差最低，泛化能力稳定，适合实际部署。

##  B3 模型复杂度成绩单
{table_lines}

##  B3 关键结论
1. **测试误差最低的复杂度：degree {best_deg}**
2. **泛化 gap 最大的复杂度：高次项（大约 10~18 区间）**
3. **为什么训练误差最低的模型不一定是最好的模型？**
   训练误差极低通常意味着模型过拟合，它把噪声也学会了，面对新数据时泛化能力很差。

## C3 方差定量统计结果
| 模型阶数 | 平均预测标准差 | 最大预测标准差 |
|----------|----------------|----------------|
| 2        | {variance_result[2]['mean']:.4f} | {variance_result[2]['max']:.4f} |
| 15       | {variance_result[15]['mean']:.4f} | {variance_result[15]['max']:.4f} |

定量结论：低阶模型标准差极小，方差低、稳定性强；高阶模型标准差巨大，高方差明显。

##  C4 High Variance 填空
High variance model 的危险，不是它不会拟合训练集，而是它对 **训练样本的微小波动** 过于敏感。


##  D4 RMSE vs MAE 异常值对比
| 场景 | RMSE | MAE |
|------|------|-----|
| 无异常值 | {loss_res["clean"]["RMSE"]:.3f} | {loss_res["clean"]["MAE"]:.3f} |
| 含异常值 | {loss_res["outlier"]["RMSE"]:.3f} | {loss_res["outlier"]["MAE"]:.3f} |


##  三条核心结论
1. 模型复杂度与误差存在明显的偏差 - 方差权衡。复杂度越低越容易欠拟合、偏差大；复杂度越高训练误差越低，但测试误差会反弹，出现严重过拟合、方差变大。
2. 高方差模型对训练样本的微小波动极其敏感，多次抽样训练出来的拟合曲线差异巨大，泛化能力非常不稳定。
3. RMSE 和 MAE 对异常值敏感度完全不同：RMSE 因平方放大极端误差，对离群点高度敏感；MAE 为线性误差，受异常值干扰更小，更稳健。

##  最能代表过拟合的图
`variance_demo.png` 最能体现过拟合：高次模型曲线剧烈抖动，对微小样本波动反应过度。
原因：这张图把低阶模型（degree=2）和高阶模型（degree=15）多次抽样拟合曲线放在一起对比。低阶模型每次采样拟合出来的曲线几乎重合，十分稳定；而 15 阶高复杂度模型每次换一组训练样本，拟合曲线都会剧烈上下抖动、偏离真实函数，肉眼就能看出它在强行学习样本里的噪声，完美把过拟合、高方差不稳定从抽象概念变成了可视化的直观现象。
##  指标选择
- 用 **RMSE**：业务非常在意单次巨大预测错误、误差近似正态分布、数据异常值少、需要严惩大误差时，优先用 RMSE。因为平方机制会放大大幅度偏差，能敏感反映系统严重失误。
- 用 **MAE**：数据天然存在较多异常值、离群点不可避免、只想看平均真实误差、不想被极端值带偏整体评价时，优先用 MAE。MAE 线性计算误差，不受极端离群点过度干扰，评价更稳健、更贴合普通样本真实误差水平。

## 为什么需要正则化？
模型越复杂，表达能力越强，越容易把训练集里的噪声、随机波动全部学进去，直接导致 high variance 高方差。表现就是：训练集误差极低，但换到新测试数据上预测效果很差，泛化能力崩塌，也就是典型过拟合。
而 Ridge / Lasso 正则化 的核心作用就是：对模型参数大小施加约束，限制模型不要变得太复杂，压缩系数、削弱无用特征的影响。从而降低模型方差，不让模型过度贴合训练噪声，让曲线变得更平滑稳定，在不明显增大偏差的前提下，有效缓解过拟合，提升模型在未知数据上的预测能力。所以在发现高复杂度→高方差→过拟合之后，自然就会想到用正则化来约束模型。
"""

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(text)


# ====================== 主入口 ==========================
def main():
    print("[Stage 1] 3个候选模型对比...")
    run_candidate_models()

    print("[Stage 2] 误差曲线扫描...")
    degrees, tr, te = run_error_curve()

    print("[Stage 3] Variance 可视化...")
    run_variance_demo()

    print("[Stage 4] RMSE vs MAE 异常值对比...")
    loss_res = run_loss_outlier()

    print("[Stage 5] 生成报告...")
    write_summary(degrees, tr, te, loss_res)

    print("✅ 全部完成！图表在 results/figures，报告在 results/summary.md")

if __name__ == "__main__":
    main()
