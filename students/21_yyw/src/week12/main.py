"""
Week 12: The Bias-Variance Visual Lab
用 Python 脚本把偏差-方差权衡"演出来"

执行入口: uv run src/week12/main.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# 将 src 目录加入 sys.path，以便导入 utils
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.metrics import calculate_mae, calculate_rmse

# ── 全局配置 ──────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORT_PATH = RESULTS_DIR / "summary.md"

RANDOM_STATE = 42
N_SAMPLES = 200
NOISE_STD = 0.5
TEST_RATIO = 0.3

plt.rcParams.update(
    {
        "figure.figsize": (10, 6),
        "figure.dpi": 120,
        "font.size": 13,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "legend.fontsize": 11,
        "lines.linewidth": 2,
    }
)


# ── 工具函数 ──────────────────────────────────────────────────────────────────
def true_function(x: np.ndarray) -> np.ndarray:
    """真实函数: sin(1.5x) + 0.3x"""
    return np.sin(1.5 * x) + 0.3 * x


def make_poly_model(degree: int) -> Pipeline:
    """构建多项式回归 Pipeline"""
    return Pipeline(
        [
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("lr", LinearRegression()),
        ]
    )


def generate_data(
    n_samples: int = N_SAMPLES,
    noise_std: float = NOISE_STD,
    test_ratio: float = TEST_RATIO,
    seed: int = RANDOM_STATE,
):
    """生成一维回归数据，返回 train/test 划分和真实函数。"""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, n_samples)
    y_true = true_function(x)
    y_noisy = y_true + rng.normal(0, noise_std, n_samples)

    n_test = int(n_samples * test_ratio)
    indices = rng.permutation(n_samples)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    x_train, y_train = x[train_idx], y_noisy[train_idx]
    x_test, y_test = x[test_idx], y_noisy[test_idx]
    return x_train, y_train, x_test, y_test


# ── Task A: 构造"会过拟合"的可视化舞台 ───────────────────────────────────────
def run_model_complexity_demo(
    x_train, y_train, x_test, y_test
) -> dict:
    """比较 degree=1, 4, 15 三位候选模型，生成 candidate_models.png。"""
    print("[Stage 1] Comparing candidate polynomial models (degree=1, 4, 15)...")

    degrees = [1, 4, 15]
    x_plot = np.linspace(-3, 3, 500)
    y_true_plot = true_function(x_plot)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    results = {}

    for ax, deg in zip(axes, degrees):
        model = make_poly_model(deg)
        model.fit(x_train.reshape(-1, 1), y_train)

        y_pred_train = model.predict(x_train.reshape(-1, 1))
        y_pred_test = model.predict(x_test.reshape(-1, 1))
        y_pred_plot = model.predict(x_plot.reshape(-1, 1))

        rmse_train = calculate_rmse(y_train, y_pred_train)
        rmse_test = calculate_rmse(y_test, y_pred_test)
        results[deg] = {"rmse_train": rmse_train, "rmse_test": rmse_test}

        ax.scatter(x_train, y_train, s=15, alpha=0.5, label="Train", color="steelblue")
        ax.scatter(x_test, y_test, s=15, alpha=0.5, label="Test", color="orange")
        ax.plot(x_plot, y_true_plot, "k--", label="True function", linewidth=1.5)
        ax.plot(x_plot, y_pred_plot, "r-", label=f"Degree {deg}", linewidth=2)
        ax.set_title(f"Degree = {deg}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="upper left", fontsize=9)
        ax.text(
            0.02,
            0.02,
            f"Train RMSE={rmse_train:.3f}\nTest RMSE={rmse_test:.3f}",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
        )

    fig.suptitle("Task A: Candidate Polynomial Models", fontsize=18, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "candidate_models.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> Saved candidate_models.png")
    return results


# ── Task B: 画出完整的复杂度-误差曲线 ────────────────────────────────────────
def run_error_curves(x_train, y_train, x_test, y_test) -> list[dict]:
    """扫描 degree 1~18，画误差曲线并返回数据。"""
    print("[Stage 2] Sweeping model complexity (degree 1 ~ 18)...")

    degrees = list(range(1, 19))
    records = []

    for deg in degrees:
        model = make_poly_model(deg)
        model.fit(x_train.reshape(-1, 1), y_train)
        y_pred_train = model.predict(x_train.reshape(-1, 1))
        y_pred_test = model.predict(x_test.reshape(-1, 1))
        rmse_train = calculate_rmse(y_train, y_pred_train)
        rmse_test = calculate_rmse(y_test, y_pred_test)
        records.append(
            {
                "degree": deg,
                "rmse_train": rmse_train,
                "rmse_test": rmse_test,
                "gap": rmse_test - rmse_train,
            }
        )

    # 画图
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        [r["degree"] for r in records],
        [r["rmse_train"] for r in records],
        "o-",
        label="Train RMSE",
        color="steelblue",
    )
    ax.plot(
        [r["degree"] for r in records],
        [r["rmse_test"] for r in records],
        "s-",
        label="Test RMSE",
        color="orange",
    )
    ax.set_xlabel("Polynomial Degree (Model Complexity)")
    ax.set_ylabel("RMSE")
    ax.set_title("Task B: Error Curves — Train vs Test RMSE")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "error_curves.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> Saved error_curves.png")

    return records


# ── Task C: 用 repeated sampling 把 variance 画出来 ──────────────────────────
def run_variance_demo(
    x_train, y_train, x_test, y_test, n_repeats: int = 15
) -> dict:
    """重复抽样拟合，展示 degree=2 vs degree=15 的 variance 差异。"""
    print("[Stage 3] Variance demo via repeated sampling...")

    degrees = [2, 15]
    x_plot = np.linspace(-3, 3, 500)
    y_true_plot = true_function(x_plot)
    rng = np.random.default_rng(RANDOM_STATE)

    stats = {}

    for deg in degrees:
        all_preds = []
        fig, ax = plt.subplots(figsize=(10, 6))

        for i in range(n_repeats):
            # 重新抽样训练集
            x_sample = rng.uniform(-3, 3, len(x_train))
            y_sample = true_function(x_sample) + rng.normal(0, NOISE_STD, len(x_sample))

            model = make_poly_model(deg)
            model.fit(x_sample.reshape(-1, 1), y_sample)
            y_pred_plot = model.predict(x_plot.reshape(-1, 1))
            all_preds.append(y_pred_plot)

            ax.plot(x_plot, y_pred_plot, alpha=0.35, color="steelblue", linewidth=1)

        all_preds = np.array(all_preds)
        mean_pred = all_preds.mean(axis=0)
        std_pred = all_preds.std(axis=0)

        ax.plot(x_plot, y_true_plot, "k--", linewidth=2, label="True function")
        ax.plot(x_plot, mean_pred, "r-", linewidth=2, label="Mean prediction")
        ax.fill_between(
            x_plot,
            mean_pred - 2 * std_pred,
            mean_pred + 2 * std_pred,
            alpha=0.2,
            color="red",
            label="±2 std",
        )
        ax.set_title(f"Task C: Variance Demo — Degree = {deg}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"variance_demo_deg{deg}.png", bbox_inches="tight")
        plt.close(fig)
        print(f"  -> Saved variance_demo_deg{deg}.png")

        stats[deg] = {
            "mean_prediction_std": float(std_pred.mean()),
            "max_prediction_std": float(std_pred.max()),
            "n_repeats": n_repeats,
        }

    # 同时生成一张合并对比图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for ax, deg in zip(axes, degrees):
        all_preds = []
        for _ in range(n_repeats):
            x_sample = rng.uniform(-3, 3, len(x_train))
            y_sample = true_function(x_sample) + rng.normal(0, NOISE_STD, len(x_sample))
            model = make_poly_model(deg)
            model.fit(x_sample.reshape(-1, 1), y_sample)
            all_preds.append(model.predict(x_plot.reshape(-1, 1)))
        all_preds = np.array(all_preds)
        mean_pred = all_preds.mean(axis=0)
        std_pred = all_preds.std(axis=0)

        for p in all_preds:
            ax.plot(x_plot, p, alpha=0.3, color="steelblue", linewidth=1)
        ax.plot(x_plot, y_true_plot, "k--", linewidth=2, label="True")
        ax.plot(x_plot, mean_pred, "r-", linewidth=2, label="Mean")
        ax.fill_between(
            x_plot,
            mean_pred - 2 * std_pred,
            mean_pred + 2 * std_pred,
            alpha=0.2,
            color="red",
            label="±2 std",
        )
        ax.set_title(f"Degree = {deg}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.text(
            0.02,
            0.02,
            f"Mean std={std_pred.mean():.4f}\nMax std={std_pred.max():.4f}",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
        )

    fig.suptitle("Task C: Variance Comparison — Low vs High Complexity", fontsize=18, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "variance_demo.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> Saved variance_demo.png")

    return stats


# ── Task D: 让异常值攻击 RMSE 与 MAE ────────────────────────────────────────
def run_loss_comparison_demo() -> dict:
    """比较 RMSE 与 MAE 在干净预测和含异常值情况下的表现。"""
    print("[Stage 4] Comparing RMSE vs MAE with outliers...")

    rng = np.random.default_rng(RANDOM_STATE)
    n = 100
    y_true = rng.uniform(0, 10, n)
    noise = rng.normal(0, 0.3, n)
    y_pred_clean = y_true + noise

    # 加入一个巨大异常值
    y_pred_outlier = y_pred_clean.copy()
    outlier_idx = rng.choice(n, size=3, replace=False)
    y_pred_outlier[outlier_idx] += rng.choice([-1, 1], size=3) * rng.uniform(15, 25, 3)

    rmse_clean = calculate_rmse(y_true, y_pred_clean)
    mae_clean = calculate_mae(y_true, y_pred_clean)
    rmse_outlier = calculate_rmse(y_true, y_pred_outlier)
    mae_outlier = calculate_mae(y_true, y_pred_outlier)

    results = {
        "clean": {"rmse": rmse_clean, "mae": mae_clean},
        "outlier": {"rmse": rmse_outlier, "mae": mae_outlier},
        "outlier_indices": outlier_idx.tolist(),
    }

    # 画图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图: 干净预测的误差分布
    errors_clean = y_true - y_pred_clean
    axes[0].hist(errors_clean, bins=25, color="steelblue", alpha=0.8, edgecolor="white")
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[0].set_title("Clean Prediction — Error Distribution")
    axes[0].set_xlabel("Prediction Error (y_true - y_pred)")
    axes[0].set_ylabel("Count")
    axes[0].text(
        0.02,
        0.95,
        f"RMSE={rmse_clean:.3f}\nMAE={mae_clean:.3f}",
        transform=axes[0].transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
    )

    # 右图: 含异常值的误差分布
    errors_outlier = y_true - y_pred_outlier
    axes[1].hist(errors_outlier, bins=25, color="coral", alpha=0.8, edgecolor="white")
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_title("With Outliers — Error Distribution")
    axes[1].set_xlabel("Prediction Error (y_true - y_pred)")
    axes[1].set_ylabel("Count")
    axes[1].text(
        0.02,
        0.95,
        f"RMSE={rmse_outlier:.3f}\nMAE={mae_outlier:.3f}",
        transform=axes[1].transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
    )

    fig.suptitle("Task D: RMSE vs MAE — Impact of Outliers", fontsize=18, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "loss_outlier_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> Saved loss_outlier_comparison.png")

    return results


# ── Task F: 输出总结报告 ──────────────────────────────────────────────────────
def write_summary_report(
    task_a_results: dict,
    task_b_records: list[dict],
    task_c_stats: dict,
    task_d_results: dict,
):
    """生成 results/summary.md。"""
    print("[Stage 5] Writing summary report...")

    # 找到最佳复杂度
    best = min(task_b_records, key=lambda r: r["rmse_test"])
    max_gap = max(task_b_records, key=lambda r: r["gap"])

    # 构造误差曲线表格
    table_rows = ""
    for r in task_b_records:
        table_rows += (
            f"| {r['degree']} | {r['rmse_train']:.4f} | {r['rmse_test']:.4f} "
            f"| {r['gap']:+.4f} |\n"
        )

    # Task A 表格
    a_table = ""
    for deg, vals in task_a_results.items():
        a_table += f"| {deg} | {vals['rmse_train']:.4f} | {vals['rmse_test']:.4f} |\n"

    # Task C 表格
    c_table = ""
    for deg, vals in task_c_stats.items():
        c_table += (
            f"| {deg} | {vals['mean_prediction_std']:.4f} "
            f"| {vals['max_prediction_std']:.4f} | {vals['n_repeats']} |\n"
        )

    # Task D 表格
    d_clean = task_d_results["clean"]
    d_outlier = task_d_results["outlier"]

    report = f"""# Week 12 — Bias-Variance Visual Lab 总结报告

## Task A: 候选模型对比

| Degree | Train RMSE | Test RMSE |
|--------|-----------|----------|
{a_table}

**谁最像欠拟合？** Degree=1。它是一条直线，无法捕捉 sin 曲线的波动，训练和测试误差都偏高。

**谁最像过拟合？** Degree=15。训练 RMSE 极低（几乎完美拟合训练点），但测试 RMSE 明显升高——模型在记忆噪声而非学习规律。

**如果必须选一个上线？** Degree=4。它在训练和测试误差之间取得了最好的平衡，泛化能力最强。

---

## Task B: 复杂度-误差曲线

| Degree | Train RMSE | Test RMSE | Generalization Gap |
|--------|-----------|----------|-------------------|
{table_rows}

**测试误差最低的复杂度：** Degree = {best['degree']}（Test RMSE = {best['rmse_test']:.4f}）

**泛化 gap 最大的复杂度：** Degree = {max_gap['degree']}（Gap = {max_gap['gap']:+.4f}）

**为什么训练误差最低的模型不一定是最好的？**
训练误差只衡量模型对已见数据的拟合程度，而我们需要的是对未见数据的预测能力。高复杂度模型可以完美拟合训练集中的噪声，但这些噪声模式在新数据中不会重复出现，导致测试误差反而升高。这就是偏差-方差权衡的核心：降低偏差的同时会增加方差。

---

## Task C: Variance 可视化（Repeated Sampling）

| Degree | Mean Prediction Std | Max Prediction Std | N Repeats |
|--------|-------------------|-------------------|-----------|
{c_table}

> high variance model 的危险，不是它不会拟合训练集，而是它对 **训练样本的随机波动** 过于敏感。

Degree=2 的多次拟合曲线高度一致（低方差），而 Degree=15 的曲线在不同训练集下形态各异（高方差），尤其在数据稀疏的边界区域波动剧烈。

---

## Task D: RMSE vs MAE 对比

| 场景 | RMSE | MAE |
|------|------|-----|
| Clean Prediction | {d_clean['rmse']:.4f} | {d_clean['mae']:.4f} |
| With Outliers | {d_outlier['rmse']:.4f} | {d_outlier['mae']:.4f} |

**为什么 RMSE 更容易被大错拉高？**
RMSE 对误差做平方，大误差会被放大（平方效应），一个极端异常值就能显著抬高整体 RMSE。MAE 取绝对值，对大误差的惩罚是线性的，因此更稳健。

**如果线上系统偶尔一次大错的代价极高，更想看哪个指标？**
更想看 RMSE。因为它对大误差更敏感，能更好地反映"最坏情况"的风险，适合对极端错误零容忍的场景（如金融风控、医疗诊断）。

**如果数据天然包含较多异常值，会不会重新考虑指标选择？**
会。此时 MAE 更能反映模型在"典型样本"上的表现，不会被少数异常值主导。也可以考虑 Huber Loss 等折中方案。

---

## 必答总结

### 三条最重要的结论

1. **模型复杂度是一把双刃剑**：增加复杂度可以降低偏差（更好地拟合真实函数），但同时增加方差（对训练数据过度敏感）。存在一个最优复杂度使总误差最小。

2. **Variance 是可见的现象，不是抽象概念**：通过 repeated sampling，我们可以清楚地看到高复杂度模型在不同训练集下产生截然不同的预测——这就是 variance 的直观含义。

3. **损失函数的选择是一种风险偏好**：RMSE 对大误差敏感，适合关注极端风险的场景；MAE 更稳健，适合数据含噪声或异常值较多的场景。

### 最能代表过拟合的图

`variance_demo_deg15.png`（或 `variance_demo.png` 中 Degree=15 的子图）最能代表过拟合是可见现象。15 条拟合曲线形态各异，尤其在数据边界区域大幅摆动，直观展示了高方差模型对训练样本的过度敏感——这不是抽象定义，而是肉眼可见的不稳定。

### 指标选择判断

- **更愿意报告 RMSE 的情况**：数据质量高、异常值少、业务对极端误差零容忍（如安全关键系统）。
- **更愿意报告 MAE 的情况**：数据含较多噪声或异常值、需要一个对典型表现更稳健的指标、或需要向非技术受众解释误差含义时（MAE 的单位更直观）。

### 与下一周的连接

> 如果模型复杂度过高会带来 high variance，那么下一步我们为什么自然会想到正则化（Ridge / Lasso）？

正则化通过在损失函数中加入对模型参数大小的惩罚项，限制模型的"自由度"，从而在不降低模型表达能力的前提下控制方差。Ridge 惩罚 L2 范数（缩小系数），Lasso 惩罚 L1 范数（可将部分系数压为零，实现特征选择）。本质上，正则化是在偏差和方差之间寻找更优权衡的自动化手段。
"""

    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"  -> Saved summary report to {REPORT_PATH}")


# ── 主入口 ────────────────────────────────────────────────────────────────────
def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 生成数据
    print("=" * 60)
    print("Week 12: The Bias-Variance Visual Lab")
    print("=" * 60)
    print("[Data] Generating regression data...")
    x_train, y_train, x_test, y_test = generate_data()
    print(f"  -> Train: {len(x_train)} samples, Test: {len(x_test)} samples")

    # Task A
    task_a = run_model_complexity_demo(x_train, y_train, x_test, y_test)

    # Task B
    task_b = run_error_curves(x_train, y_train, x_test, y_test)

    # Task C
    task_c = run_variance_demo(x_train, y_train, x_test, y_test)

    # Task D
    task_d = run_loss_comparison_demo()

    # Task F: 写报告
    write_summary_report(task_a, task_b, task_c, task_d)

    print("=" * 60)
    print("All tasks completed. Check results/ for outputs.")
    print("=" * 60)


if __name__ == "__main__":
    main()
