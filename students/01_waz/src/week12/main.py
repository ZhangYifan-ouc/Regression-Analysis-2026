"""
Week 12: The Bias-Variance Visual Lab
=====================================
用 Python 脚本把偏差-方差权衡"演出来"。

单一入口: python src/week12/main.py

产出:
  - figures/candidate_models.png     (Task A: 三位候选模型对比)
  - figures/error_curves.png         (Task B: 完整复杂度-误差曲线)
  - figures/variance_demo.png        (Task C: 方差可视化)
  - figures/loss_outlier_comparison.png (Task D: RMSE vs MAE)
  - results/report.md                (Task F: 总结报告)
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# ── 路径：导入自己的 utils ───────────────────────────────────────────
HERE = Path(__file__).resolve().parent
SRC = HERE.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.metrics import calculate_rmse, calculate_mae

# ── 常量 ─────────────────────────────────────────────────────────────
FIGURES_DIR = HERE / "results" / "figures"
REPORT_PATH = HERE / "results" / "report.md"
SEED = 20260525

# 中文兼容
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


# ======================== 数据生成 =====================================

def true_function(x: np.ndarray) -> np.ndarray:
    """真实函数: sin(1.5x) + 0.2x, 非线性且有趋势."""
    return np.sin(1.5 * x) + 0.2 * x


def make_noisy_sample(
    n: int = 130,
    noise_std: float = 0.35,
    x_low: float = -3.0,
    x_high: float = 3.0,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """生成一维回归数据: x ~ Uniform, y = f(x) + ε."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(x_low, x_high, n))
    y = true_function(x) + rng.normal(0, noise_std, n)
    return x.reshape(-1, 1), y


def polynomial_model(degree: int) -> Pipeline:
    """构造多项式回归 pipeline (不含 bias 列, 由 LinearRegression 处理截距)."""
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("linreg", LinearRegression()),
    ])


def fit_degree(x_train, y_train, x_eval, degree):
    """在训练集上拟合, 在评估集上预测."""
    model = polynomial_model(degree)
    model.fit(x_train, y_train)
    return model.predict(x_eval), model


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ======================== Task A: 候选模型对比 =========================

def stage_candidate_models(x_train, x_test, y_train, y_test, x_grid, y_true):
    """A1-A2: 比较 degree=1, 4, 15 三位候选模型."""
    print("[Stage 1] 生成三位候选模型对比图 ...")
    degrees = [1, 4, 15]
    records = []

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, d in zip(axes, degrees):
        y_pred_grid, model = fit_degree(x_train, y_train, x_grid, d)
        train_pred = model.predict(x_train)
        test_pred = model.predict(x_test)
        tr_rmse = calculate_rmse(y_train, train_pred)
        te_rmse = calculate_rmse(y_test, test_pred)
        records.append({"degree": d, "train_rmse": tr_rmse, "test_rmse": te_rmse})

        ax.scatter(x_train[:, 0], y_train, s=16, alpha=0.6, label="Train")
        ax.scatter(x_test[:, 0], y_test, s=16, alpha=0.6, label="Test")
        ax.plot(x_grid[:, 0], y_true, "k--", linewidth=2, label="Truth")
        ax.plot(x_grid[:, 0], y_pred_grid, "#d62728", linewidth=2.5, label=f"deg={d}")
        ax.set_title(f"degree={d}\nTrain RMSE={tr_rmse:.3f}  Test RMSE={te_rmse:.3f}")
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[-1].legend(loc="upper left", fontsize=9)
    fig.suptitle("Candidate Models: Which one would you ship?", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "candidate_models.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(records)


# ======================== Task B: 完整复杂度误差曲线 ==================

def stage_error_curves(x_train, x_test, y_train, y_test):
    """B1-B2: 扫描 degree=1..18, 画 train/test RMSE 曲线."""
    print("[Stage 2] 扫描模型复杂度 1-18 ...")
    records = []
    for d in range(1, 19):
        model = polynomial_model(d)
        model.fit(x_train, y_train)
        tr = calculate_rmse(y_train, model.predict(x_train))
        te = calculate_rmse(y_test, model.predict(x_test))
        records.append({"degree": d, "train_rmse": tr, "test_rmse": te})

    df = pd.DataFrame(records)
    df["generalization_gap"] = df["test_rmse"] - df["train_rmse"]
    best_d = int(df.loc[df["test_rmse"].idxmin(), "degree"])
    worst_gap_d = int(df.loc[df["generalization_gap"].idxmax(), "degree"])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["degree"], df["train_rmse"], "o-", lw=2.2, label="Train RMSE")
    ax.plot(df["degree"], df["test_rmse"], "s-", lw=2.2, label="Test RMSE")
    ax.axvline(best_d, color="gray", ls="--", alpha=0.7,
               label=f"Best degree={best_d}")
    ax.set_xlabel("Polynomial Degree")
    ax.set_ylabel("RMSE")
    ax.set_title("Training vs Test Error Across Model Complexity")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "error_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return df, best_d, worst_gap_d


# ======================== Task C: 方差可视化 ==========================

def stage_variance_demo():
    """C1-C3: 重复抽样, 展示 high-variance 模型的曲线波动."""
    print("[Stage 3] 重复抽样可视化方差 ...")
    x_eval = np.linspace(-3, 3, 300).reshape(-1, 1)
    y_eval_true = true_function(x_eval.ravel())

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    degree_data = {}
    for ax, d in zip(axes, [2, 15]):
        collected = []
        for i in range(14):
            xs, ys = make_noisy_sample(n=35, noise_std=0.35, seed=1000 + i)
            yp, _ = fit_degree(xs, ys, x_eval, d)
            collected.append(yp)
            ax.plot(x_eval[:, 0], yp, alpha=0.28, linewidth=1.3)
        stacked = np.vstack(collected)
        degree_data[d] = stacked
        ax.plot(x_eval[:, 0], y_eval_true, "k--", linewidth=3, label="Truth")
        ax.set_title(f"Repeated fits  degree={d}")
        ax.set_xlabel("x")
        ax.legend(loc="upper left")
    axes[0].set_ylabel("Predicted y")
    fig.suptitle("Variance Demo: How much do the curves wobble?", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "variance_demo.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for d, preds in degree_data.items():
        stds = preds.std(axis=0)
        rows.append({
            "degree": d,
            "mean_prediction_std": float(stds.mean()),
            "max_prediction_std": float(stds.max()),
        })
    return pd.DataFrame(rows)


# ======================== Task D: RMSE vs MAE 离群值攻击 ==============

def stage_loss_comparison():
    """D1-D3: 构造干净预测 + 一个离群点, 比较 RMSE/MAE 反应."""
    print("[Stage 4] 比较 RMSE 与 MAE 对离群值的敏感度 ...")
    y_true = np.array([100, 102, 98, 101, 99, 103, 100, 97], dtype=float)
    y_pred_clean = np.array([101, 101, 99, 100, 100, 102, 99, 98], dtype=float)
    y_pred_outlier = y_pred_clean.copy()
    y_pred_outlier[-1] = 80  # 制造一个巨大的预测偏差

    metrics = pd.DataFrame({
        "Scenario": ["Clean prediction", "One large outlier"],
        "RMSE": [
            calculate_rmse(y_true, y_pred_clean),
            calculate_rmse(y_true, y_pred_outlier),
        ],
        "MAE": [
            calculate_mae(y_true, y_pred_clean),
            calculate_mae(y_true, y_pred_outlier),
        ],
    })

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    # 左: 散点对比
    n = len(y_true)
    axes[0].scatter(range(n), y_true, s=80, marker="o", label="True")
    axes[0].scatter(range(n), y_pred_outlier, s=80, marker="x", label="Pred (w/ outlier)")
    axes[0].set_title("One outlier changes one prediction")
    axes[0].set_xlabel("Sample index")
    axes[0].set_ylabel("Value")
    axes[0].legend()

    # 右: 柱状图对比
    w = 0.35
    x_pos = np.arange(len(metrics))
    axes[1].bar(x_pos - w / 2, metrics["RMSE"], w, label="RMSE")
    axes[1].bar(x_pos + w / 2, metrics["MAE"], w, label="MAE")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(metrics["Scenario"], rotation=8)
    axes[1].set_title("Which metric gets hit harder?")
    axes[1].set_ylabel("Metric value")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "loss_outlier_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return metrics


# ======================== Task F: 报告生成 ============================

def format_md_table(df: pd.DataFrame, decimals: int = 3) -> str:
    """将 DataFrame 格式化成 Markdown 表格."""
    r = df.copy()
    for c in r.select_dtypes(include=["number"]).columns:
        r[c] = r[c].round(decimals)
    h = list(r.columns)
    lines = ["| " + " | ".join(h) + " |", "|" + "|".join(["---"] * len(h)) + "|"]
    for row in r.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def write_report(candidate_df, error_df, best_degree, worst_gap_degree,
                 variance_df, loss_df):
    """Task F: 输出 results/report.md."""
    print("[Stage 5] 生成总结报告 ...")

    best_candidate = int(candidate_df.loc[candidate_df["test_rmse"].idxmin(), "degree"])

    report = f"""# Week 12 Report — Bias-Variance Visual Lab

## 1. 三条重要结论

1. **训练误差下降 ≠ 泛化能力提升。** 当模型复杂度超过某个阈值后，训练误差继续下降，但测试误差反而上升——这就是过拟合的可视化证据。
2. **高复杂度模型的预测曲线在不同训练样本之间剧烈摆动，** 这种"不稳定"就是 high variance 的直观表现。
3. **RMSE 对大误差比 MAE 更敏感，** 因为 RMSE 平方化了大偏差，而 MAE 对所有误差一视同仁。选择哪个指标取决于业务对"极端错误"的容忍度。

---

## 2. 最能代表过拟合的图

**`candidate_models.png`** 最能说明"过拟合不是抽象概念"：

- degree=1 (左) 是典型的欠拟合：曲线太平滑，无法捕捉数据的非线性趋势；
- degree=4 (中) 是合理的拟合：曲线跟随趋势但不过度扭曲；
- degree=15 (右) 是典型的过拟合：曲线在训练点之间剧烈震荡，虽然训练误差极低，但测试误差反而上升。

> 过拟合在图上不是公式，而是眼睛能看见的"骚动曲线"。

---

## 3. 候选模型对比 (Task A)

{format_md_table(candidate_df)}

**判定：**
- degree=1 最像**欠拟合** (两条误差都偏高)；
- degree=15 最像**过拟合** (训练误差极低但测试误差上升)；
- 如果今天必须选一个上线，我会押 **degree=4**：它的测试误差最低，泛化 gap 可控。

---

## 4. 完整复杂度-误差曲线 (Task B)

最佳测试 RMSE 出现在 **degree={best_degree}**。
最大的泛化 gap 出现在 **degree={worst_gap_degree}**。

### 前 10 个复杂度的成绩单

{format_md_table(error_df.head(10)[["degree", "train_rmse", "test_rmse", "generalization_gap"]])}

### 全部 18 个复杂度

{format_md_table(error_df[["degree", "train_rmse", "test_rmse", "generalization_gap"]])}

**回答：**
1. 测试误差最低的复杂度是 **degree={best_degree}**；
2. 泛化 gap 最大的是 **degree={worst_gap_degree}**，说明高复杂度虽然能"记住"训练数据，但在测试集上表现最不稳定；
3. **训练误差最低的模型不一定是最好的，** 因为它可能只是在背诵噪声，而非学习信号的规律。

---

## 5. 方差可视化 (Task C)

{format_md_table(variance_df)}

**一句话补全：**

> high variance model 的危险，不是它不会拟合训练集，而是它对 **训练样本的微小变化** 过于敏感。

在 `variance_demo.png` 中，degree=15 的 14 条拟合曲线像散开的羽毛，而 degree=2 的曲线则紧紧抱在一起。这就是 high variance 的视觉定义。

---

## 6. RMSE vs MAE 离群值对比 (Task D)

{format_md_table(loss_df)}

### 业务解释

1. **为什么 RMSE 更容易被大错拉高？**
   RMSE 先对误差平方再取均值——一个 20 的误差在 RMSE 中贡献 400，而在 MAE 中只贡献 20。这就是平方惩罚的力量。

2. **如果线上系统偶尔一次大错的代价极高，更看哪个指标？**
   更看 **RMSE**。因为 RMSE 会把那一次大错放大到不可忽视的程度，迫使模型优化时优先减少极端偏差。

3. **如果数据天然包含较多异常值，会不会重新考虑指标选择？**
   会。如果异常值是数据固有属性而非录入错误，MAE 更合适——它不会被少数极端值劫持，给出的误差评估更稳定、更贴近"典型"情况。

---

## 7. 与下一周的连接

> 如果模型复杂度过高会带来 high variance，那么下一步我们为什么自然会想到正则化（Ridge / Lasso）？

**回答：** 正则化本质上是在损失函数中加入对系数的惩罚项，迫使模型"不敢"把系数设得太大。这等价于：
- 人为限制模型的"自由度"；
- 用一点 bias 换 variance 的下降；
- 让高复杂度模型也能稳定工作。

换句话说，如果 `error_curves.png` 告诉你"测试误差在 degree>8 后开始上升"，那么正则化就是给 degree=15 的模型加上"紧箍咒"，让它表现得像一个自适应复杂度的模型。

---

## 8. 运行说明

```bash
cd students/01_waz
python src/week12/main.py
```

运行后自动生成：
- `results/figures/candidate_models.png`
- `results/figures/error_curves.png`
- `results/figures/variance_demo.png`
- `results/figures/loss_outlier_comparison.png`
- `results/report.md`
"""

    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"  ✓ report.md")


# ======================== 主入口 ======================================

def main():
    """唯一执行入口."""
    print("=" * 60)
    print("  Week 12: Bias-Variance Visual Lab")
    print("=" * 60)

    ensure_dirs()

    # ── 全局数据准备 ──
    x, y = make_noisy_sample(n=130, noise_std=0.35, seed=7)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.35, random_state=42
    )
    x_grid = np.linspace(-3.2, 3.2, 500).reshape(-1, 1)
    y_true_grid = true_function(x_grid.ravel())

    # ── Stage 1: 候选模型 ──
    candidate_df = stage_candidate_models(
        x_train, x_test, y_train, y_test, x_grid, y_true_grid
    )

    # ── Stage 2: 复杂度曲线 ──
    error_df, best_degree, worst_gap_degree = stage_error_curves(
        x_train, x_test, y_train, y_test
    )

    # ── Stage 3: 方差演示 ──
    variance_df = stage_variance_demo()

    # ── Stage 4: 离群值攻击 ──
    loss_df = stage_loss_comparison()

    # ── Stage 5: 报告 ──
    write_report(
        candidate_df, error_df, best_degree, worst_gap_degree,
        variance_df, loss_df,
    )

    print("\n" + "=" * 60)
    print("  ✅ Week 12 全部完成!")
    print(f"  图片 → {FIGURES_DIR}")
    print(f"  报告 → {REPORT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
