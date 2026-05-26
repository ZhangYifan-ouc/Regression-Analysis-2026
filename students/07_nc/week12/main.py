"""Week 12: Bias-Variance Visual Lab.

Run from the student folder with:

    uv run src/week12/main.py

The script is intentionally organized as a small teaching workflow:
1. Build a noisy nonlinear regression data set.
2. Compare three candidate polynomial models.
3. Sweep model complexity and draw train/test RMSE curves.
4. Repeat sampling to visualize high variance.
5. Compare RMSE and MAE under one large outlier.
6. Write a Markdown report.
"""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Allow `from utils.metrics import ...` when executing this file directly.
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.metrics import calculate_mae, calculate_rmse  # noqa: E402


RANDOM_SEED = 2026
N_SAMPLES = 140
CANDIDATE_DEGREES = [1, 4, 15]
SWEEP_DEGREES = list(range(1, 19))
VARIANCE_DEGREES = [2, 15]
VARIANCE_REPEATS = 25

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULT_FIGURES_DIR = RESULTS_DIR / "figures"
LEGACY_FIGURES_DIR = BASE_DIR / "figures"
SUMMARY_PATH = RESULTS_DIR / "summary.md"
REPORT_PATH = RESULTS_DIR / "report.md"


@dataclass
class GeneratedData:
    """Container for the generated regression stage."""

    x: np.ndarray
    y: np.ndarray
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    x_grid: np.ndarray
    y_grid_true: np.ndarray


@dataclass
class CandidateResult:
    """Metrics and predictions for one candidate degree."""

    degree: int
    train_rmse: float
    test_rmse: float
    train_mae: float
    test_mae: float
    grid_prediction: np.ndarray


def true_function(x: np.ndarray) -> np.ndarray:
    """The smooth nonlinear signal hidden behind noisy observations."""
    return np.sin(1.4 * x) + 0.35 * x + 0.20 * np.cos(2.2 * x)


def prepare_output_dirs() -> None:
    """Create clean output directories for a reproducible run."""
    for directory in [RESULTS_DIR, LEGACY_FIGURES_DIR]:
        if directory.exists():
            shutil.rmtree(directory)
    RESULT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    LEGACY_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def generate_data(seed: int = RANDOM_SEED) -> GeneratedData:
    """Generate one-dimensional noisy regression data.

    The full data set has more than 100 samples as required.  A deliberately
    modest training subset makes the overfitting pattern visually clear when a
    very high polynomial degree is used.
    """
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-3.0, 3.0, size=N_SAMPLES))
    noise = rng.normal(0.0, 0.35, size=N_SAMPLES)
    y = true_function(x) + noise

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        train_size=42,
        random_state=seed,
        shuffle=True,
    )
    x_grid = np.linspace(-3.2, 3.2, 500)
    y_grid_true = true_function(x_grid)

    return GeneratedData(
        x=x,
        y=y,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        x_grid=x_grid,
        y_grid_true=y_grid_true,
    )


def make_polynomial_model(degree: int) -> Pipeline:
    """Create a polynomial regression pipeline.

    PolynomialFeatures creates the model complexity, StandardScaler keeps high
    degree terms numerically stable, and LinearRegression fits the final model.
    """
    return Pipeline(
        steps=[
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("scale", StandardScaler()),
            ("linear", LinearRegression()),
        ]
    )


def fit_and_evaluate_degree(data: GeneratedData, degree: int) -> CandidateResult:
    """Fit one polynomial degree and return train/test metrics."""
    model = make_polynomial_model(degree)
    model.fit(data.x_train.reshape(-1, 1), data.y_train)

    train_pred = model.predict(data.x_train.reshape(-1, 1))
    test_pred = model.predict(data.x_test.reshape(-1, 1))
    grid_prediction = model.predict(data.x_grid.reshape(-1, 1))

    return CandidateResult(
        degree=degree,
        train_rmse=calculate_rmse(data.y_train, train_pred),
        test_rmse=calculate_rmse(data.y_test, test_pred),
        train_mae=calculate_mae(data.y_train, train_pred),
        test_mae=calculate_mae(data.y_test, test_pred),
        grid_prediction=grid_prediction,
    )


def run_model_complexity_demo(data: GeneratedData) -> list[CandidateResult]:
    """Compare degree 1, 4, and 15 in one large candidate-model figure."""
    results = [fit_and_evaluate_degree(data, degree) for degree in CANDIDATE_DEGREES]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    for ax, result in zip(axes, results):
        ax.scatter(data.x_train, data.y_train, s=42, alpha=0.85, label="Train points")
        ax.scatter(data.x_test, data.y_test, s=30, alpha=0.55, label="Test points")
        ax.plot(data.x_grid, data.y_grid_true, linewidth=2.6, label="True function")
        ax.plot(data.x_grid, result.grid_prediction, linewidth=2.2, label=f"Degree {result.degree} fit")
        ax.set_title(
            f"Degree {result.degree}\n"
            f"Train RMSE={result.train_rmse:.3f}, Test RMSE={result.test_rmse:.3f}",
            fontsize=13,
        )
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylim(-4.0, 4.0)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("y", fontsize=12)
    axes[-1].legend(loc="best", fontsize=10)
    fig.suptitle("Candidate polynomial models: underfit, balanced fit, and overfit", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_figure(fig, "candidate_models.png")
    return results


def run_error_curve_demo(data: GeneratedData) -> pd.DataFrame:
    """Sweep degree 1 to 18 and record train/test RMSE."""
    rows: list[dict[str, float]] = []
    for degree in SWEEP_DEGREES:
        result = fit_and_evaluate_degree(data, degree)
        rows.append(
            {
                "degree": degree,
                "train_rmse": result.train_rmse,
                "test_rmse": result.test_rmse,
                "generalization_gap": result.test_rmse - result.train_rmse,
            }
        )
    metrics_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(metrics_df["degree"], metrics_df["train_rmse"], marker="o", linewidth=2.4, label="Train RMSE")
    ax.plot(metrics_df["degree"], metrics_df["test_rmse"], marker="o", linewidth=2.4, label="Test RMSE")
    best_row = metrics_df.loc[metrics_df["test_rmse"].idxmin()]
    ax.axvline(best_row["degree"], linestyle="--", linewidth=1.8, alpha=0.7)
    ax.text(
        best_row["degree"] + 0.15,
        best_row["test_rmse"] + 0.05,
        f"Lowest test RMSE\ndegree={int(best_row['degree'])}",
        fontsize=11,
    )
    ax.set_title("Model complexity vs. error: training error keeps falling, test error turns back", fontsize=15)
    ax.set_xlabel("Polynomial degree", fontsize=12)
    ax.set_ylabel("RMSE", fontsize=12)
    ax.set_xticks(SWEEP_DEGREES)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=11)
    fig.tight_layout()
    save_figure(fig, "error_curves.png")
    return metrics_df


def run_variance_demo(seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Repeated sampling demo for low-variance and high-variance models."""
    rng = np.random.default_rng(seed + 100)
    x_grid = np.linspace(-2.6, 2.6, 500)
    y_grid_true = true_function(x_grid)
    all_predictions: dict[int, list[np.ndarray]] = {degree: [] for degree in VARIANCE_DEGREES}

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), sharey=True)
    for ax, degree in zip(axes, VARIANCE_DEGREES):
        for repeat_idx in range(VARIANCE_REPEATS):
            x_sample = np.sort(rng.uniform(-3.0, 3.0, size=42))
            y_sample = true_function(x_sample) + rng.normal(0.0, 0.35, size=x_sample.shape[0])
            model = make_polynomial_model(degree)
            model.fit(x_sample.reshape(-1, 1), y_sample)
            pred = model.predict(x_grid.reshape(-1, 1))
            all_predictions[degree].append(pred)
            ax.plot(x_grid, pred, linewidth=1.0, alpha=0.42)
        ax.plot(x_grid, y_grid_true, color="black", linewidth=3.0, label="True function")
        ax.set_title(f"Repeated sampling, degree={degree}", fontsize=14)
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylim(-3.5, 3.5)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=10)
    axes[0].set_ylabel("Predicted y", fontsize=12)
    fig.suptitle("High variance is visible as unstable fitted curves across repeated samples", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_figure(fig, "variance_demo.png")

    summary_rows: list[dict[str, float]] = []
    for degree, predictions in all_predictions.items():
        pred_matrix = np.vstack(predictions)
        prediction_std = pred_matrix.std(axis=0)
        summary_rows.append(
            {
                "degree": degree,
                "repeats": VARIANCE_REPEATS,
                "mean_prediction_std": float(prediction_std.mean()),
                "max_prediction_std": float(prediction_std.max()),
            }
        )
    return pd.DataFrame(summary_rows)


def run_loss_comparison_demo(seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Construct a clean prediction case and one-outlier case for RMSE/MAE."""
    rng = np.random.default_rng(seed + 200)
    y_true = np.linspace(10, 30, 40)
    y_pred_clean = y_true + rng.normal(0.0, 0.8, size=y_true.shape[0])
    y_pred_outlier = y_pred_clean.copy()
    y_pred_outlier[28] += 16.0

    loss_rows = []
    for scenario, y_pred in [
        ("clean prediction", y_pred_clean),
        ("one large outlier", y_pred_outlier),
    ]:
        loss_rows.append(
            {
                "scenario": scenario,
                "RMSE": calculate_rmse(y_true, y_pred),
                "MAE": calculate_mae(y_true, y_pred),
                "max_abs_error": float(np.max(np.abs(y_true - y_pred))),
            }
        )
    loss_df = pd.DataFrame(loss_rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    x_index = np.arange(y_true.shape[0])
    axes[0].plot(x_index, np.abs(y_true - y_pred_clean), marker="o", linewidth=1.8, label="Clean errors")
    axes[0].plot(x_index, np.abs(y_true - y_pred_outlier), marker="o", linewidth=1.8, label="With one outlier")
    axes[0].set_title("Absolute error by observation", fontsize=14)
    axes[0].set_xlabel("Observation index", fontsize=12)
    axes[0].set_ylabel("Absolute error", fontsize=12)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=10)

    positions = np.arange(len(loss_df))
    width = 0.35
    axes[1].bar(positions - width / 2, loss_df["RMSE"], width=width, label="RMSE")
    axes[1].bar(positions + width / 2, loss_df["MAE"], width=width, label="MAE")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(loss_df["scenario"], rotation=0)
    axes[1].set_title("RMSE reacts more strongly to one large mistake", fontsize=14)
    axes[1].set_ylabel("Metric value", fontsize=12)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=10)
    fig.suptitle("Loss function comparison: one outlier attacks RMSE more than MAE", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_figure(fig, "loss_outlier_comparison.png")
    return loss_df


def save_figure(fig: plt.Figure, filename: str) -> None:
    """Save a figure to both official/compatible figure directories."""
    for directory in [RESULT_FIGURES_DIR, LEGACY_FIGURES_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(directory / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def format_markdown_table(df: pd.DataFrame, float_digits: int = 4) -> str:
    """Convert a DataFrame to a Markdown table without optional dependencies."""
    display_df = df.copy()
    numeric_cols = display_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        numeric_values = pd.to_numeric(display_df[col], errors="coerce")
        if numeric_values.notna().all() and np.allclose(numeric_values, np.round(numeric_values)):
            display_df[col] = numeric_values.astype(int).astype(str)
        else:
            display_df[col] = numeric_values.map(lambda value: f"{value:.{float_digits}f}")

    headers = [str(col) for col in display_df.columns]
    rows = [[str(value) for value in row] for row in display_df.to_numpy()]
    table_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        table_lines.append("| " + " | ".join(row) + " |")
    return "\n".join(table_lines)


def write_summary_report(
    candidate_results: list[CandidateResult],
    error_curve_df: pd.DataFrame,
    variance_summary_df: pd.DataFrame,
    loss_df: pd.DataFrame,
) -> None:
    """Write the narrative Markdown report required by Week 12."""
    candidate_table = pd.DataFrame(
        [
            {
                "degree": result.degree,
                "train_RMSE": result.train_rmse,
                "test_RMSE": result.test_rmse,
                "train_MAE": result.train_mae,
                "test_MAE": result.test_mae,
            }
            for result in candidate_results
        ]
    )
    best_error_row = error_curve_df.loc[error_curve_df["test_rmse"].idxmin()]
    largest_gap_row = error_curve_df.loc[error_curve_df["generalization_gap"].idxmax()]
    lowest_train_row = error_curve_df.loc[error_curve_df["train_rmse"].idxmin()]

    degree_1 = candidate_table.loc[candidate_table["degree"] == 1].iloc[0]
    degree_4 = candidate_table.loc[candidate_table["degree"] == 4].iloc[0]
    degree_15 = candidate_table.loc[candidate_table["degree"] == 15].iloc[0]

    rmse_increase = (
        loss_df.loc[loss_df["scenario"] == "one large outlier", "RMSE"].iloc[0]
        - loss_df.loc[loss_df["scenario"] == "clean prediction", "RMSE"].iloc[0]
    )
    mae_increase = (
        loss_df.loc[loss_df["scenario"] == "one large outlier", "MAE"].iloc[0]
        - loss_df.loc[loss_df["scenario"] == "clean prediction", "MAE"].iloc[0]
    )

    report = f"""# Week 12 Summary: Bias-Variance Visual Lab

## 0. 运行入口与交付物说明

本周作业使用一个可复现 Python 脚本完成所有实验：

```bash
uv run src/week12/main.py
```

脚本会自动完成数据生成、模型拟合、指标计算、图像输出和 Markdown 报告输出。RMSE 与 MAE 均来自我自己的 `src/utils/metrics.py`，核心调用位于 `src/week12/main.py` 的 `fit_and_evaluate_degree()` 与 `run_loss_comparison_demo()`。

主要输出图包括：

- `src/week12/results/figures/candidate_models.png`
- `src/week12/results/figures/error_curves.png`
- `src/week12/results/figures/variance_demo.png`
- `src/week12/results/figures/loss_outlier_comparison.png`

为了兼容作业说明中对图像目录的不同表述，脚本也会把同样的图片复制到 `src/week12/figures/`。

---

## 1. Task A：三位候选模型比较

我生成了一份一维非线性回归数据：

```text
true function = sin(1.4x) + 0.35x + 0.20cos(2.2x)
```

完整样本量为 `{N_SAMPLES}`，满足不少于 100 个样本的要求。训练集使用 42 个样本，其余作为测试集。训练集故意不设得太大，是为了让高阶多项式更容易把训练点的噪声也学进去，从而产生可视化的过拟合现象。

候选模型图：`figures/candidate_models.png`。

### 1.1 三个候选模型成绩单

{format_markdown_table(candidate_table.rename(columns={"train_RMSE": "train RMSE", "test_RMSE": "test RMSE", "train_MAE": "train MAE", "test_MAE": "test MAE"}))}

### 1.2 谁最像欠拟合？

`degree = 1` 最像欠拟合。原因是它只能画出一条直线，但真实函数明显有弯曲形状。它无法表达 `sin` 和 `cos` 造成的非线性变化，因此训练误差和测试误差都较高。本次运行中 degree 1 的 train RMSE 为 `{degree_1['train_RMSE']:.4f}`，test RMSE 为 `{degree_1['test_RMSE']:.4f}`。

### 1.3 谁最像过拟合？

`degree = 15` 最像过拟合。它有足够高的复杂度，可以追逐训练样本中的局部噪声，所以训练误差会比较低；但是这种曲线在训练点之间容易出现不稳定波动，测试集表现不一定最好。本次运行中 degree 15 的 train RMSE 为 `{degree_15['train_RMSE']:.4f}`，test RMSE 为 `{degree_15['test_RMSE']:.4f}`。

### 1.4 如果必须选一个上线，我会先押谁？

我会先选择 `degree = 4`。理由不是它训练误差最低，而是它在真实函数曲线附近比较平滑，能表达非线性，又没有像 degree 15 那样对训练样本剧烈摆动。本次运行中 degree 4 的 train RMSE 为 `{degree_4['train_RMSE']:.4f}`，test RMSE 为 `{degree_4['test_RMSE']:.4f}`，在候选模型中更像一个 bias 和 variance 折中的选择。

---

## 2. Task B：复杂度-误差曲线

我从 `degree = 1` 扫描到 `degree = 18`，分别记录 train RMSE、test RMSE 和 generalization gap。

误差曲线图：`figures/error_curves.png`。

### 2.1 完整成绩单

{format_markdown_table(error_curve_df.rename(columns={"degree": "degree", "train_rmse": "train RMSE", "test_rmse": "test RMSE", "generalization_gap": "generalization gap"}))}

### 2.2 测试误差最低的复杂度是多少？

测试误差最低的是 `degree = {int(best_error_row['degree'])}`，对应 test RMSE 为 `{best_error_row['test_rmse']:.4f}`。这说明模型复杂度不是越高越好，而是存在一个让测试集误差相对较低的中间区域。

### 2.3 泛化 gap 最大在哪里？

本次实验中泛化 gap 最大的是 `degree = {int(largest_gap_row['degree'])}`，gap 为 `{largest_gap_row['generalization_gap']:.4f}`。这类高复杂度模型的典型问题是：训练误差被压得很低，但测试误差没有同步下降，说明模型对训练集的局部细节过于敏感。

### 2.4 为什么训练误差最低的模型不一定最好？

训练误差最低的是 `degree = {int(lowest_train_row['degree'])}`，train RMSE 为 `{lowest_train_row['train_rmse']:.4f}`。但训练误差只衡量模型对已经见过的数据拟合得有多好，它不衡量对未来新样本的稳定性。过高复杂度的模型可能把噪声当成规律，训练误差继续下降，但测试误差反而上升。

---

## 3. Task C：Repeated sampling 画出 variance

我固定真实函数，只改变每次抽到的训练样本。对 `degree = 2` 和 `degree = 15` 各重复抽样并拟合 `{VARIANCE_REPEATS}` 次，然后把多条拟合曲线画到同一张图里。

Variance 图：`figures/variance_demo.png`。

### 3.1 Prediction variance summary

{format_markdown_table(variance_summary_df.rename(columns={"degree": "degree", "repeats": "repeats", "mean_prediction_std": "mean prediction std", "max_prediction_std": "max prediction std"}))}

### 3.2 图像解释

`degree = 2` 的多条曲线通常比较集中，说明换一批训练样本之后，模型形状不会发生太夸张的变化。`degree = 15` 的多条曲线明显更容易分散，尤其在样本较稀疏的边界附近更不稳定。这就是 high variance 的可视化含义：模型不仅拟合训练集，还会对训练样本中的偶然扰动做出过度反应。

### 3.3 一句话补全

> high variance model 的危险，不是它不会拟合训练集，而是它对 **训练样本的随机扰动和抽样变化** 过于敏感。

---

## 4. Task D：让异常值攻击 RMSE 与 MAE

我构造了一组干净预测 `y_pred_clean`，然后只改动一个样本，制造一个很大的预测误差，得到 `y_pred_outlier`。

对比图：`figures/loss_outlier_comparison.png`。

### 4.1 指标对比表

{format_markdown_table(loss_df.rename(columns={"scenario": "scenario", "max_abs_error": "max abs error"}))}

加入一个大异常误差后，RMSE 增加了 `{rmse_increase:.4f}`，MAE 增加了 `{mae_increase:.4f}`。可以看到 RMSE 对单个大错更敏感。

### 4.2 为什么 RMSE 更容易被大错拉高？

RMSE 的内部是平方误差。一个误差如果变成原来的 10 倍，平方误差会变成 100 倍。因此，少数极端错误会在 RMSE 中被放大。MAE 使用绝对误差，不会平方放大，所以对异常值更稳健。

### 4.3 如果线上系统偶尔一次大错代价极高，更想看哪个指标？

如果一次大错的代价极高，我更想报告 RMSE，或者至少同时报告 RMSE 和最大绝对误差。原因是 RMSE 会主动惩罚大错，能提醒我们不要只追求“平均看起来还不错”。

### 4.4 如果数据天然包含较多异常值，会不会重新考虑指标选择？

会。如果异常值本身是真实业务中的常态，而不是数据错误，那么只看 RMSE 可能让模型评估被少数样本主导。此时我会同时报告 MAE、分位数误差，必要时再检查异常值是否代表特殊人群、特殊场景或录入错误。

---

## 5. Task F：必答总结

### 必答 1：三条最重要结论

1. **训练误差下降不等于模型变好。** 随着多项式 degree 增加，模型更容易贴近训练点，但测试误差可能先下降后上升。
2. **High variance 是可以被看见的。** 在 repeated sampling 图中，高阶模型面对不同训练样本会产生明显不同的曲线。
3. **RMSE 和 MAE 代表不同风险偏好。** RMSE 更重视大错，MAE 更稳健、更像普通平均偏差。

### 必答 2：最能代表过拟合的图

我认为 `figures/variance_demo.png` 最能代表“过拟合不是抽象概念，而是可见现象”。因为这张图不是只画一次模型，而是在相同真实函数下重复抽样多次。degree 15 的曲线在不同样本下明显摇摆，说明它不是单纯学到了真实规律，而是对训练样本的偶然性非常敏感。

`figures/error_curves.png` 也很重要，因为它用 train RMSE 和 test RMSE 的分叉展示了过拟合的指标表现：训练误差继续降低，测试误差却开始回升。

### 必答 3：指标选择判断

当业务更害怕少数大错时，我更愿意报告 RMSE。例如金融风控、医疗预警、供应链缺货预警等场景，单次严重错误可能造成很大损失。

当数据中天然有较多异常值，或者我希望给出普通样本上的典型误差水平时，我更愿意报告 MAE。例如房价、收入、国家发展指标等数据常有长尾分布，MAE 更不容易被少数极端点控制。

### 必答 4：与下一周正则化的连接

如果模型复杂度过高会带来 high variance，那么下一步自然会想到正则化。Ridge 和 Lasso 的核心思路是限制模型参数过大，降低模型对训练样本噪声的过度反应。也就是说，正则化不是为了让训练集拟合得更完美，而是为了在可接受的 bias 增加下换取 variance 的下降，从而提高泛化能力。

---

## 6. 代码组织说明

`src/week12/main.py` 按照作业建议组织为多个函数：

- `generate_data()`：生成非线性模拟数据并做 train/test split；
- `run_model_complexity_demo()`：生成三位候选模型图；
- `run_error_curve_demo()`：扫描 degree 1 到 18 并画误差曲线；
- `run_variance_demo()`：重复抽样并画 high variance 图；
- `run_loss_comparison_demo()`：构造异常值并比较 RMSE 与 MAE；
- `write_summary_report()`：写出本 Markdown 总结；
- `main()`：作为唯一入口串联完整实验。

本作业没有依赖 notebook、隐藏变量或手工步骤。所有结果都可以由 `uv run src/week12/main.py` 一次性复现。
"""

    SUMMARY_PATH.write_text(report, encoding="utf-8")
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    """Run the complete Week 12 visual lab."""
    prepare_output_dirs()

    print("[Stage 1] Generating nonlinear regression data...")
    data = generate_data()

    print("[Stage 2] Comparing candidate polynomial models...")
    candidate_results = run_model_complexity_demo(data)

    print("[Stage 3] Sweeping model complexity from degree 1 to 18...")
    error_curve_df = run_error_curve_demo(data)

    print("[Stage 4] Running repeated sampling variance demo...")
    variance_summary_df = run_variance_demo()

    print("[Stage 5] Comparing RMSE and MAE under one large outlier...")
    loss_df = run_loss_comparison_demo()

    print("[Stage 6] Writing Markdown summary report...")
    write_summary_report(candidate_results, error_curve_df, variance_summary_df, loss_df)

    print("Done. Outputs written to:")
    print(f"  - {SUMMARY_PATH.relative_to(BASE_DIR.parents[1])}")
    print(f"  - {RESULT_FIGURES_DIR.relative_to(BASE_DIR.parents[1])}")
    print(f"  - {LEGACY_FIGURES_DIR.relative_to(BASE_DIR.parents[1])}")


if __name__ == "__main__":
    main()
