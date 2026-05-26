"""
Week 12: Bias-Variance Visual Lab
==================================
用 Python 脚本把偏差-方差权衡"演出来"：
  - Task A: 候选模型对比 (degree=1,4,15)
  - Task B: 完整复杂度-误差曲线 (degree 1→18)
  - Task C: 重复抽样绘制 variance 图
  - Task D: RMSE vs MAE 异常值敏感性对比
  - Task E/F: 分阶段实验流程 + 自动生成 summary.md

运行方式:
    uv run src/week12/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# ---------- 复用自己已有的 utils ----------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.metrics import calculate_rmse, calculate_mae  # noqa: E402

# ---------- 路径与随机种子 ----------
ROOT = Path(__file__).resolve().parent
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"
SEED = 20260526

# ---------- 全局 matplotlib 设置：让图适合投屏 ----------
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "lines.linewidth": 2.0,
    }
)


# ============================================================
#  数据生成
# ============================================================

def true_function(x: np.ndarray) -> np.ndarray:
    """真实非线性函数：sin(1.5*x) + 0.2*x （与 example 略有不同）"""
    return np.sin(1.5 * x) + 0.2 * x


def make_noisy_sample(
    n: int = 120,
    noise_std: float = 0.35,
    x_low: float = -3.0,
    x_high: float = 3.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """生成一维回归数据：从均匀分布抽样 x，加上高斯噪声。"""
    rng = np.random.default_rng(SEED if seed is None else seed)
    x = np.sort(rng.uniform(x_low, x_high, n))
    y = true_function(x) + rng.normal(0, noise_std, n)
    return x.reshape(-1, 1), y


# ============================================================
#  模型工具
# ============================================================

def polynomial_model(degree: int) -> Pipeline:
    """构造多项式回归 pipeline。"""
    return Pipeline(
        [
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("linreg", LinearRegression()),
        ]
    )


def fit_degree(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    degree: int,
) -> tuple[np.ndarray, Pipeline]:
    """在训练集上拟合指定 degree 的多项式，返回对 x_eval 的预测和模型。"""
    model = polynomial_model(degree)
    model.fit(x_train, y_train)
    return model.predict(x_eval), model


def ensure_dirs() -> None:
    """创建 figures 和 results 目录。"""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
#  Task A: 候选模型对比
# ============================================================

def stage_candidate_models(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    x_grid: np.ndarray,
    y_true_grid: np.ndarray,
) -> pd.DataFrame:
    """绘制 degree=1,4,15 三个候选模型的拟合对比图。"""
    records: list[dict[str, float | int]] = []
    degrees = [1, 4, 15]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, degree in zip(axes, degrees):
        y_grid_pred, model = fit_degree(x_train, y_train, x_grid, degree)

        train_pred = model.predict(x_train)
        test_pred = model.predict(x_test)
        train_rmse = calculate_rmse(y_train, train_pred)
        test_rmse = calculate_rmse(y_test, test_pred)

        records.append(
            {"degree": degree, "train_rmse": train_rmse, "test_rmse": test_rmse}
        )

        ax.scatter(x_train[:, 0], y_train, s=18, alpha=0.6, label="train")
        ax.scatter(x_test[:, 0], y_test, s=18, alpha=0.6, label="test")
        ax.plot(
            x_grid[:, 0],
            y_true_grid,
            color="black",
            linewidth=2,
            linestyle="--",
            label="truth",
        )
        ax.plot(
            x_grid[:, 0],
            y_grid_pred,
            color="#d62728",
            linewidth=2.5,
            label=f"degree={degree}",
        )
        ax.set_title(
            f"degree={degree}\ntrain RMSE={train_rmse:.3f}, test RMSE={test_rmse:.3f}"
        )
        ax.set_xlabel("x")

    axes[0].set_ylabel("y")
    axes[-1].legend(loc="upper left", fontsize=10)
    fig.suptitle("Candidate models: which one would you ship?", y=1.03)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "candidate_models.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    return pd.DataFrame(records)


# ============================================================
#  Task B: 完整复杂度-误差曲线
# ============================================================

def stage_error_curves(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[pd.DataFrame, int, int]:
    """扫描 degree 1→18，绘制 train / test RMSE 曲线。"""
    records: list[dict[str, float | int]] = []
    for degree in range(1, 19):
        model = polynomial_model(degree)
        model.fit(x_train, y_train)
        train_pred = model.predict(x_train)
        test_pred = model.predict(x_test)
        records.append(
            {
                "degree": degree,
                "train_rmse": calculate_rmse(y_train, train_pred),
                "test_rmse": calculate_rmse(y_test, test_pred),
            }
        )

    error_df = pd.DataFrame(records)
    error_df["generalization_gap"] = error_df["test_rmse"] - error_df["train_rmse"]
    best_degree = int(error_df.loc[error_df["test_rmse"].idxmin(), "degree"])
    largest_gap_degree = int(
        error_df.loc[error_df["generalization_gap"].idxmax(), "degree"]
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        error_df["degree"],
        error_df["train_rmse"],
        marker="o",
        linewidth=2.2,
        label="train RMSE",
    )
    ax.plot(
        error_df["degree"],
        error_df["test_rmse"],
        marker="o",
        linewidth=2.2,
        label="test RMSE",
    )
    ax.axvline(
        best_degree,
        color="gray",
        linestyle="--",
        alpha=0.75,
        label=f"best degree={best_degree}",
    )
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("RMSE")
    ax.set_title("Training vs test error across model complexity")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "error_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    return error_df, best_degree, largest_gap_degree


# ============================================================
#  Task C: 重复抽样 —— 把 variance 画出来
# ============================================================

def stage_variance_demo() -> pd.DataFrame:
    """固定真实函数，重复抽样拟合多次，对比低/高复杂度模型的曲线稳定性。"""
    x_eval = np.linspace(-3, 3, 300).reshape(-1, 1)
    y_eval_true = true_function(x_eval.ravel())

    degree_predictions: dict[int, np.ndarray] = {}
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

    for ax, degree in zip(axes, [2, 15]):
        collected_predictions = []
        for sample_idx in range(14):
            x_sample, y_sample = make_noisy_sample(
                n=35, noise_std=0.35, seed=1000 + sample_idx
            )
            y_pred, _ = fit_degree(x_sample, y_sample, x_eval, degree)
            collected_predictions.append(y_pred)
            ax.plot(x_eval[:, 0], y_pred, alpha=0.30, linewidth=1.4)

        stacked_predictions = np.vstack(collected_predictions)
        degree_predictions[degree] = stacked_predictions
        ax.plot(
            x_eval[:, 0],
            y_eval_true,
            color="black",
            linewidth=3,
            linestyle="--",
            label="truth",
        )
        ax.set_title(f"Repeated fits with degree={degree}")
        ax.set_xlabel("x")
        ax.legend(loc="upper left")

    axes[0].set_ylabel("predicted y")
    fig.suptitle("Variance demo: how much do the curves wobble?", y=1.03)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "variance_demo.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for degree, predictions in degree_predictions.items():
        pointwise_std = predictions.std(axis=0)
        rows.append(
            {
                "degree": degree,
                "mean_prediction_std": float(pointwise_std.mean()),
                "max_prediction_std": float(pointwise_std.max()),
            }
        )
    return pd.DataFrame(rows)


# ============================================================
#  Task D: RMSE vs MAE —— 异常值攻击对比
# ============================================================

def stage_loss_comparison() -> pd.DataFrame:
    """构造干净预测与含一个离群点的预测，对比 RMSE 和 MAE 的变化。"""
    y_true = np.array([100, 102, 98, 101, 99, 103, 100, 97], dtype=float)
    y_pred_clean = np.array([101, 101, 99, 100, 100, 102, 99, 98], dtype=float)
    y_pred_outlier = y_pred_clean.copy()
    y_pred_outlier[-1] = 80  # 人为制造一个大误差

    metrics_df = pd.DataFrame(
        {
            "scenario": ["clean prediction", "one large outlier"],
            "RMSE": [
                calculate_rmse(y_true, y_pred_clean),
                calculate_rmse(y_true, y_pred_outlier),
            ],
            "MAE": [
                calculate_mae(y_true, y_pred_clean),
                calculate_mae(y_true, y_pred_outlier),
            ],
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    # 左图：散点对比
    axes[0].scatter(range(len(y_true)), y_true, s=85, label="true")
    axes[0].scatter(range(len(y_true)), y_pred_outlier, s=85, label="pred (with outlier)")
    axes[0].set_title("One outlier changes one prediction")
    axes[0].set_xlabel("sample index")
    axes[0].set_ylabel("value")
    axes[0].legend()

    # 右图：柱状图对比 RMSE vs MAE
    width = 0.35
    x_pos = np.arange(len(metrics_df))
    axes[1].bar(x_pos - width / 2, metrics_df["RMSE"], width=width, label="RMSE")
    axes[1].bar(x_pos + width / 2, metrics_df["MAE"], width=width, label="MAE")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(metrics_df["scenario"], rotation=10)
    axes[1].set_title("Which metric gets hit harder?")
    axes[1].set_ylabel("metric value")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "loss_outlier_comparison.png", dpi=180, bbox_inches="tight"
    )
    plt.close(fig)

    return metrics_df


# ============================================================
#  工具：DataFrame → Markdown 表格
# ============================================================

def format_table(df: pd.DataFrame, decimals: int = 3) -> str:
    """将 DataFrame 转为 Markdown 表格字符串。"""
    rounded = df.copy()
    numeric_cols = rounded.select_dtypes(include=["number"]).columns
    rounded[numeric_cols] = rounded[numeric_cols].round(decimals)

    headers = list(rounded.columns)
    separator = ["---"] * len(headers)
    rows = [headers, separator]

    for row in rounded.itertuples(index=False, name=None):
        rows.append([str(value) for value in row])

    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


# ============================================================
#  Task F: 输出 summary.md
# ============================================================

def write_summary(
    candidate_df: pd.DataFrame,
    error_df: pd.DataFrame,
    best_degree: int,
    largest_gap_degree: int,
    variance_df: pd.DataFrame,
    loss_df: pd.DataFrame,
) -> None:
    """汇总所有实验结果，写入 results/summary.md。"""
    best_candidate = int(candidate_df.loc[candidate_df["test_rmse"].idxmin(), "degree"])
    summary = f"""# Week 12 Summary Report

---

## 三条核心结论

1. **训练误差最低 ≠ 泛化能力最强**：从 Task B 的误差曲线可以看出，随着多项式阶数升高，
   训练 RMSE 持续下降甚至趋近于零，但测试 RMSE 在高复杂度区域反而回升。
   这说明模型已经开始"记忆"训练集中的噪声，而不是学习真实规律。

2. **High variance 是可见的**：在 Task C 的重复抽样实验中，degree=15 的拟合曲线在不同
   训练样本上剧烈摆动，而 degree=2 的曲线则相当稳定。所谓"高方差"，就是模型对训练
   数据的微小变化过度敏感。

3. **RMSE 对离群值更敏感**：Task D 中仅仅改变一个预测值，RMSE 就剧烈上升，而 MAE
   的变化幅度要小得多。这是因为 RMSE 对大误差施加了平方惩罚，放大了离群点的影响。

---

## Task A: 候选模型对比

{format_table(candidate_df)}

- **degree=1（欠拟合）**：直线无法捕捉数据的非线性趋势，train 和 test RMSE 都很高。
- **degree=4（较合理）**：曲线平滑地跟踪了真实函数的走向，test RMSE 在三个候选模型中最低。
- **degree=15（过拟合）**：训练误差极低，但曲线在数据点之间剧烈震荡，test RMSE 明显反升。

如果今天必须选一个上线，我会选 **degree={best_candidate}**，因为它测试集表现最好，
且曲线形态与真实函数最为接近，说明泛化能力最强。

---

## Task B: 完整复杂度扫描

最佳测试 RMSE 出现在 **degree={best_degree}**。
最大泛化 gap 出现在 **degree={largest_gap_degree}**。

{format_table(error_df.loc[:, ["degree", "train_rmse", "test_rmse", "generalization_gap"]])}

从表格可以看出：
- 测试误差最低的复杂度是 **degree={best_degree}**。
- 泛化 gap 最大的区域落在高 degree 附近（如 degree={largest_gap_degree}），此时 train RMSE 极低
  而 test RMSE 很高，说明严重过拟合。
- **训练误差最低（degree=18）的模型不是最好的模型**，因为它对噪声也进行了精确拟合，
  导致在新数据上表现很差。

---

## Task C: Variance 定量总结

{format_table(variance_df)}

degree=15 的 `mean_prediction_std` 和 `max_prediction_std` 都远高于 degree=2，
定量印证了"高复杂度模型预测更不稳定"。

> **一句话补全**：high variance model 的危险，不是它不会拟合训练集，而是它对
> **训练样本的具体构成** 过于敏感。

---

## Task D: RMSE vs MAE 离群点攻击

{format_table(loss_df)}

### 业务解释

1. **为什么 RMSE 更容易被大错拉高？**
   RMSE 计算的是误差的**平方**的均值再开根号。一个误差为 17 的离群点，其平方贡献
   高达 289，而 MAE 中该离群点只贡献 17。平方操作是非线性的，对大误差给予了
   不成比例的惩罚。

2. **如果线上系统偶尔一次大错的代价极高，更应关注哪个指标？**
   如果一次大错就可能造成严重后果（如自动驾驶、医疗诊断），更应该关注 **RMSE**，
   因为它对大误差足够敏感，能及时发出警报。

3. **如果数据天然包含较多异常值，是否会重新考虑指标选择？**
   会。如果异常值是数据本身的一部分（如金融市场的极端波动），用 **MAE** 可能更合理，
   因为它不会被少数极值主导，能更稳健地反映模型的典型表现。

---

## 最能代表"过拟合"的图

**`figures/error_curves.png`** 最能代表"过拟合不是抽象概念，而是可见现象"。

原因：图中 train RMSE 和 test RMSE 两条曲线随着 degree 增大而分道扬镳——
训练误差一路向下，测试误差先降后升。这个**U 形测试误差曲线**本身就是过拟合
最经典的视觉证据：模型复杂度超过某一点后，多出来的自由度被用来
"记住"噪声而非"学习"信号。

---

## 与下一周（正则化）的衔接

如果模型复杂度过高会带来 high variance，那么下一步自然会想到正则化（Ridge / Lasso）。
因为正则化的本质就是：**在损失函数中加入对系数大小的惩罚项，主动限制模型复杂度**。
这样一来：

- Ridge（L2 正则化）会收缩所有系数，防止个别系数过大导致曲线剧烈震荡；
- Lasso（L1 正则化）不仅能收缩系数，还能把不重要的系数直接压到零，起到特征选择的作用。

正则化就是在 bias 和 variance 之间做有意识的 trade-off：用一点点 bias 的增加，
换取 variance 的大幅下降，从而提升泛化能力。

---

> 运行命令：`uv run src/week12/main.py`
> 所有评估指标均来自自建 `utils.metrics` 模块 (`calculate_rmse` / `calculate_mae`)。
"""
    (RESULTS_DIR / "summary.md").write_text(summary, encoding="utf-8")


# ============================================================
#  主入口
# ============================================================

def main() -> None:
    """分阶段执行 Week 12 全部实验。"""
    ensure_dirs()

    # ---- 共享数据准备 ----
    print("[Stage 1] Generating synthetic data and candidate model plots...")
    x, y = make_noisy_sample(n=120, noise_std=0.35, seed=7)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.35, random_state=42
    )
    x_grid = np.linspace(-3.2, 3.2, 500).reshape(-1, 1)
    y_true_grid = true_function(x_grid.ravel())

    candidate_df = stage_candidate_models(
        x_train, x_test, y_train, y_test, x_grid, y_true_grid
    )
    print("  -> candidate_models.png saved.")

    # ----
    print("[Stage 2] Sweeping model complexity (degree 1→18)...")
    error_df, best_degree, largest_gap_degree = stage_error_curves(
        x_train, x_test, y_train, y_test
    )
    print(f"  -> Best degree: {best_degree}, Largest gap degree: {largest_gap_degree}")
    print("  -> error_curves.png saved.")

    # ----
    print("[Stage 3] Repeating sampling to visualize variance...")
    variance_df = stage_variance_demo()
    print("  -> variance_demo.png saved.")

    # ----
    print("[Stage 4] Comparing RMSE and MAE under an outlier...")
    loss_df = stage_loss_comparison()
    print("  -> loss_outlier_comparison.png saved.")

    # ----
    print("[Stage 5] Writing markdown summary...")
    write_summary(
        candidate_df, error_df, best_degree, largest_gap_degree, variance_df, loss_df
    )
    print(f"\nDone! Figures saved to: {FIGURES_DIR}")
    print(f"Summary saved to: {RESULTS_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()