"""
Week 12 Assignment: The Bias-Variance Visual Lab
说明:
    本脚本优先复用 src/utils/metrics.py 中的 calculate_rmse 和 calculate_mae。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


# ---------------------------------------------------------------------
# 0. 路径设置与指标函数导入
# ---------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parents[0]
RESULTS_DIR = CURRENT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
SUMMARY_PATH = RESULTS_DIR / "summary.md"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from utils.metrics import calculate_mae, calculate_rmse

    METRICS_SOURCE = "src/utils/metrics.py"
except ImportError:
    # 兜底方案：如果本地路径暂时没有配置好，脚本仍能运行。
    # 但报告中会说明本次没有成功导入自己的 metrics.py。
    METRICS_SOURCE = "fallback functions inside main.py"

    def calculate_rmse(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    def calculate_mae(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.mean(np.abs(y_true - y_pred))


# ---------------------------------------------------------------------
# 1. 基础工具函数
# ---------------------------------------------------------------------
def ensure_output_dirs() -> None:
    """创建输出目录。"""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def true_function(x: np.ndarray) -> np.ndarray:
    """
    本周实验使用的真实函数。

    真实函数设计为：
        f(x) = sin(1.5x) + 0.25x

    该函数不是简单直线，而是“正弦波 + 轻微线性趋势”，适合展示：
    - degree=1 的欠拟合；
    - degree=4 的相对均衡；
    - degree=15 的过拟合风险。
    """
    return np.sin(1.5 * x) + 0.25 * x


def generate_data(
    n_samples: int = 160,
    noise_std: float = 0.28,
    random_state: int = 2026,
):
    """
    生成一维非线性回归数据，并划分 train/test。

    数据生成过程：
    1. 从 [-3, 3] 上均匀随机采样 x；
    2. 根据真实函数 f(x)=sin(1.5x)+0.25x 生成无噪声真实值；
    3. 加入正态噪声 epsilon ~ N(0, noise_std^2)；
    4. 得到观测值 y=f(x)+epsilon；
    5. 按 70%/30% 划分训练集和测试集。
    """
    rng = np.random.default_rng(random_state)

    x = rng.uniform(-3.0, 3.0, size=n_samples)
    noise = rng.normal(0.0, noise_std, size=n_samples)
    y = true_function(x) + noise

    X = x.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=random_state,
    )

    x_grid = np.linspace(-3.2, 3.2, 500)
    X_grid = x_grid.reshape(-1, 1)
    y_true_grid = true_function(x_grid)

    data_info = {
        "n_samples": n_samples,
        "noise_std": noise_std,
        "random_state": random_state,
        "x_min": -3.0,
        "x_max": 3.0,
        "test_size": 0.30,
        "train_size": len(X_train),
        "test_count": len(X_test),
        "grid_count": len(x_grid),
        "true_function": "f(x) = sin(1.5x) + 0.25x",
        "noise_distribution": f"epsilon ~ N(0, {noise_std}^2)",
    }

    return X_train, X_test, y_train, y_test, X_grid, y_true_grid, data_info


def build_polynomial_model(degree: int) -> Pipeline:
    """
    构造多项式回归模型。

    Pipeline 顺序：
    1. PolynomialFeatures: 生成 x, x^2, ..., x^degree；
    2. StandardScaler: 标准化多项式特征，缓解高阶特征尺度差异；
    3. LinearRegression: 拟合线性回归模型。
    """
    return Pipeline(
        steps=[
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("scaler", StandardScaler()),
            ("linear", LinearRegression()),
        ]
    )


def fit_and_score_model(
    degree: int,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """拟合指定复杂度模型，并计算训练集与测试集 RMSE。"""
    model = build_polynomial_model(degree)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_rmse = calculate_rmse(y_train, y_train_pred)
    test_rmse = calculate_rmse(y_test, y_test_pred)

    return {
        "degree": degree,
        "model": model,
        "train_rmse": float(train_rmse),
        "test_rmse": float(test_rmse),
        "gap": float(test_rmse - train_rmse),
    }


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    """把列表数据转成 Markdown 表格，避免额外依赖 pandas/tabulate。"""
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------
# 2. Task A：三位候选模型对比
# ---------------------------------------------------------------------
def run_model_complexity_demo(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    X_grid: np.ndarray,
    y_true_grid: np.ndarray,
    candidate_degrees: tuple[int, ...] = (1, 4, 15),
) -> list[dict]:
    """生成 candidate_models.png，并返回候选模型指标。"""
    print("[Stage 1] Comparing candidate polynomial models...")

    results = [
        fit_and_score_model(degree, X_train, X_test, y_train, y_test)
        for degree in candidate_degrees
    ]

    fig, axes = plt.subplots(1, len(candidate_degrees), figsize=(18, 5), sharey=True)

    if len(candidate_degrees) == 1:
        axes = [axes]

    for ax, result in zip(axes, results):
        degree = result["degree"]
        model = result["model"]
        y_grid_pred = model.predict(X_grid)

        ax.scatter(X_train.ravel(), y_train, s=28, alpha=0.75, label="train data")
        ax.scatter(
            X_test.ravel(), y_test, s=28, alpha=0.75, marker="x", label="test data"
        )
        ax.plot(X_grid.ravel(), y_true_grid, linewidth=2.5, label="true function")
        ax.plot(X_grid.ravel(), y_grid_pred, linewidth=2.5, label="fitted curve")

        ax.set_title(
            f"Polynomial degree = {degree}\n"
            f"train RMSE={result['train_rmse']:.3f}, "
            f"test RMSE={result['test_rmse']:.3f}",
            fontsize=12,
        )
        ax.set_xlabel("x")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("y")
    axes[-1].legend(loc="best", fontsize=9)
    fig.suptitle("Task A: Candidate Models under Different Complexity", fontsize=16)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "candidate_models.png", dpi=220)
    plt.close(fig)

    return results


# ---------------------------------------------------------------------
# 3. Task B：完整复杂度-误差曲线
# ---------------------------------------------------------------------
def run_error_curve_demo(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    degrees: range = range(1, 19),
) -> list[dict]:
    """扫描 degree=1 到 18，生成 error_curves.png。"""
    print("[Stage 2] Sweeping model complexity from degree 1 to 18...")

    results = [
        fit_and_score_model(degree, X_train, X_test, y_train, y_test)
        for degree in degrees
    ]

    degree_values = [item["degree"] for item in results]
    train_values = [item["train_rmse"] for item in results]
    test_values = [item["test_rmse"] for item in results]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(degree_values, train_values, marker="o", linewidth=2.2, label="train RMSE")
    ax.plot(degree_values, test_values, marker="s", linewidth=2.2, label="test RMSE")

    best_result = min(results, key=lambda item: item["test_rmse"])
    ax.axvline(best_result["degree"], linestyle="--", linewidth=1.8, alpha=0.8)
    ax.text(
        best_result["degree"] + 0.15,
        best_result["test_rmse"],
        f"lowest test RMSE degree={best_result['degree']}",
        fontsize=10,
    )

    ax.set_title("Task B: Model Complexity vs. RMSE", fontsize=15)
    ax.set_xlabel("Model complexity: polynomial degree", fontsize=12)
    ax.set_ylabel("RMSE", fontsize=12)
    ax.set_xticks(list(degrees))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "error_curves.png", dpi=220)
    plt.close(fig)

    return results


# ---------------------------------------------------------------------
# 4. Task C：Repeated sampling 展示 variance
# ---------------------------------------------------------------------
def run_variance_demo(
    true_func: Callable[[np.ndarray], np.ndarray],
    degrees: tuple[int, ...] = (2, 15),
    n_repeats: int = 20,
    n_train: int = 35,
    noise_std: float = 0.28,
    random_state: int = 2026,
) -> list[dict]:
    """
    重复抽样训练集，叠加画出多条拟合曲线，生成 variance_demo.png。
    """
    print("[Stage 3] Repeated sampling to visualize variance...")

    rng = np.random.default_rng(random_state)
    x_grid = np.linspace(-3.0, 3.0, 500)
    X_grid = x_grid.reshape(-1, 1)
    y_true_grid = true_func(x_grid)

    fig, axes = plt.subplots(1, len(degrees), figsize=(14, 5), sharey=True)
    if len(degrees) == 1:
        axes = [axes]

    summary_rows = []

    for ax, degree in zip(axes, degrees):
        predictions = []

        for _ in range(n_repeats):
            x_train = rng.uniform(-3.0, 3.0, size=n_train)
            y_train = true_func(x_train) + rng.normal(0.0, noise_std, size=n_train)

            model = build_polynomial_model(degree)
            model.fit(x_train.reshape(-1, 1), y_train)

            y_pred_grid = model.predict(X_grid)
            predictions.append(y_pred_grid)

            # 只限制画图范围，避免 degree=15 极端爆炸值把图片完全拉坏。
            # 统计指标仍然使用原始预测值计算。
            y_plot = np.clip(y_pred_grid, -300, 3200)
            ax.plot(x_grid, y_plot, linewidth=1.4, alpha=0.35)

        predictions_array = np.vstack(predictions)
        pred_std = np.std(predictions_array, axis=0)

        summary_rows.append(
            {
                "degree": degree,
                "mean_prediction_std": float(np.mean(pred_std)),
                "max_prediction_std": float(np.max(pred_std)),
            }
        )

        ax.plot(x_grid, y_true_grid, "k--", linewidth=2.5, label="truth")
        ax.set_title(f"Repeated fits with degree={degree}", fontsize=14)
        ax.set_xlabel("x")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left")

    axes[0].set_ylabel("predicted y")
    axes[0].set_ylim(-300, 3300)
    axes[1].set_ylim(-300, 3300)

    fig.suptitle("Variance demo: how much do the curves wobble?", fontsize=16)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "variance_demo.png", dpi=220)
    plt.close(fig)

    return summary_rows


# ---------------------------------------------------------------------
# 5. Task D：异常值攻击 RMSE 与 MAE
# ---------------------------------------------------------------------
def run_loss_comparison_demo() -> list[dict]:
    """
    构造 clean prediction 与 one large outlier，比较 RMSE 和 MAE。
    """
    print("[Stage 4] Comparing RMSE and MAE under one large outlier...")

    # 构造一个小而直观的例子
    y_true = np.array([100, 102, 98, 101, 99, 103, 100, 97], dtype=float)

    # 干净预测：整体误差都比较小
    y_pred_clean = np.array([101, 101, 99, 100, 100, 102, 99, 98], dtype=float)

    # 在 clean prediction 基础上，只制造一个大离群误差
    y_pred_outlier = y_pred_clean.copy()
    y_pred_outlier[7] = 80.0

    clean_rmse = float(calculate_rmse(y_true, y_pred_clean))
    clean_mae = float(calculate_mae(y_true, y_pred_clean))
    outlier_rmse = float(calculate_rmse(y_true, y_pred_outlier))
    outlier_mae = float(calculate_mae(y_true, y_pred_outlier))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：true vs pred，突出只有一个预测点被异常值影响
    idx = np.arange(len(y_true))
    axes[0].scatter(idx, y_true, s=120, label="true")
    axes[0].scatter(idx, y_pred_outlier, s=120, label="pred")
    axes[0].set_title("One outlier changes one prediction", fontsize=14)
    axes[0].set_xlabel("sample index")
    axes[0].set_ylabel("value")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    # 右图：clean vs outlier 时 RMSE 和 MAE 的变化
    labels = ["clean prediction", "one large outlier"]
    x = np.arange(len(labels))
    width = 0.34

    axes[1].bar(x - width / 2, [clean_rmse, outlier_rmse], width, label="RMSE")
    axes[1].bar(x + width / 2, [clean_mae, outlier_mae], width, label="MAE")
    axes[1].set_title("Which metric gets hit harder?", fontsize=14)
    axes[1].set_ylabel("metric value")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=10)
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "loss_outlier_comparison.png", dpi=220)
    plt.close(fig)

    return [
        {
            "scenario": "clean prediction",
            "rmse": clean_rmse,
            "mae": clean_mae,
        },
        {
            "scenario": "one large outlier",
            "rmse": outlier_rmse,
            "mae": outlier_mae,
        },
    ]


# ---------------------------------------------------------------------
# 6. Task F：写出 summary.md
# ---------------------------------------------------------------------
def write_summary_report(
    data_info: dict,
    candidate_results: list[dict],
    error_curve_results: list[dict],
    variance_results: list[dict],
    loss_results: list[dict],
) -> None:
    """根据实验结果自动写入 Markdown 报告。"""
    print("[Stage 5] Writing summary report...")

    best_test_result = min(error_curve_results, key=lambda item: item["test_rmse"])
    largest_gap_results = sorted(
        error_curve_results,
        key=lambda item: item["gap"],
        reverse=True,
    )[:3]

    # 三位候选模型的上线选择固定为 degree=4。
    # 原因不是机械看某一次误差，而是结合曲线形态、过拟合风险和解释稳定性。
    production_choice_degree = 4
    production_choice = next(
        item for item in candidate_results if item["degree"] == production_choice_degree
    )

    complexity_rows = [
        [
            item["degree"],
            f"{item['train_rmse']:.4f}",
            f"{item['test_rmse']:.4f}",
            f"{item['gap']:.4f}",
        ]
        for item in error_curve_results
    ]

    variance_rows = [
        [
            item["degree"],
            f"{item['mean_prediction_std']:.4f}",
            f"{item['max_prediction_std']:.4f}",
        ]
        for item in variance_results
    ]

    loss_rows = [
        [
            item["scenario"],
            f"{item['rmse']:.4f}",
            f"{item['mae']:.4f}",
        ]
        for item in loss_results
    ]

    gap_degree_text = ", ".join(
        f"degree={item['degree']} (gap={item['gap']:.4f})"
        for item in largest_gap_results
    )

    report = f"""# Week 12 Assignment Summary: Bias-Variance Visual Lab

本报告由 `src/week12/main.py` 自动生成。  
本次实验中的 RMSE 和 MAE 优先来自 `{METRICS_SOURCE}`。

---

## 1. 实验目标

本周实验的目标不是背诵 bias-variance 的定义，而是通过 Python 脚本让现象直接发生。  
我主要完成了四组实验：

1. 比较不同复杂度的多项式模型，观察欠拟合与过拟合；
2. 扫描 degree=1 到 degree=18，画出训练误差与测试误差曲线；
3. 通过 repeated sampling 展示 high variance 在图上的表现；
4. 人为加入异常值，比较 RMSE 和 MAE 对大误差的敏感性。

---

## 2. 数据是怎么生成的

本实验没有使用外部真实数据，而是自己构造了一维非线性回归数据。这样做的好处是：真实函数已知，可以清楚地比较模型拟合曲线和真实曲线之间的差异。

### 2.1 真实函数

本实验设定的真实函数为：

```text
{data_info["true_function"]}
```

这个函数由正弦项和轻微线性趋势组成。它不是简单直线，因此可以观察到低复杂度模型的欠拟合；同时它又是平滑函数，因此高阶多项式如果出现剧烈波动，就可以被识别为过拟合。

### 2.2 随机采样与噪声

具体数据生成过程如下：

1. 在区间 [{data_info["x_min"]}, {data_info["x_max"]}] 上均匀随机采样 x；
2. 根据真实函数计算无噪声的函数值 f(x)；
3. 加入正态分布随机噪声：`{data_info["noise_distribution"]}`；
4. 得到最终观测值：`y = f(x) + epsilon`；
5. 使用 `train_test_split` 将数据划分为训练集和测试集。

### 2.3 数据生成参数

| 项目 | 设置 |
| --- | --- |
| 总样本量 | {data_info["n_samples"]} |
| 训练集样本量 | {data_info["train_size"]} |
| 测试集样本量 | {data_info["test_count"]} |
| 测试集比例 | {data_info["test_size"]} |
| x 采样范围 | [{data_info["x_min"]}, {data_info["x_max"]}] |
| 噪声标准差 | {data_info["noise_std"]} |
| 随机种子 | {data_info["random_state"]} |
| 画平滑曲线使用的 grid 点数 | {data_info["grid_count"]} |

使用固定随机种子是为了保证每次运行 `main.py` 后得到的图和表一致，便于复现和检查。

---

## 3. 代码组织说明

`main.py` 被组织成一个可复现实验工作流，而不是把所有代码堆在一起。主要函数如下：

| 函数名 | 作用 |
| --- | --- |
| `ensure_output_dirs()` | 创建 `results/figures/` 输出目录 |
| `true_function()` | 定义本实验使用的真实非线性函数 |
| `generate_data()` | 生成模拟数据，并划分训练集和测试集 |
| `build_polynomial_model()` | 构建多项式回归 Pipeline |
| `fit_and_score_model()` | 拟合指定 degree 的模型，并计算 train/test RMSE |
| `run_model_complexity_demo()` | 完成 Task A，生成三位候选模型对比图 |
| `run_error_curve_demo()` | 完成 Task B，扫描 degree=1 到 18 并生成误差曲线 |
| `run_variance_demo()` | 完成 Task C，通过 repeated sampling 展示 variance |
| `run_loss_comparison_demo()` | 完成 Task D，比较 RMSE 和 MAE 对异常值的反应 |
| `write_summary_report()` | 自动写出 `summary.md` |
| `main()` | 唯一运行入口，按顺序执行全部实验 |

程序运行顺序为：

```text
生成数据
→ 比较 degree=1/4/15 三个候选模型
→ 扫描完整复杂度误差曲线
→ 重复抽样展示 high variance
→ 构造异常值比较 RMSE 和 MAE
→ 自动生成 Markdown 报告
```
这种组织方式保证了作业可以通过一个入口完整复现。
---

## 4. Task A：三位候选模型对比

输出图像：`results/figures/candidate_models.png`

| degree | train RMSE | test RMSE | 判断 |
| --- | --- | --- | --- |
"""

    for item in candidate_results:
        degree = item["degree"]
        if degree == 1:
            comment = "欠拟合：模型太简单，无法捕捉真实函数的非线性变化"
        elif degree == 4:
            comment = "复杂度适中：能捕捉主要非线性趋势，同时曲线相对平滑"
        elif degree == 15:
            comment = "过拟合：训练误差低，但曲线过于灵活，容易学习噪声"
        else:
            comment = "候选模型"
        report += (
            f"| {degree} | {item['train_rmse']:.4f} | "
            f"{item['test_rmse']:.4f} | {comment} |\n"
        )

    report += f"""
### 4.1 谁最像欠拟合？

**degree=1 最像欠拟合。**  
因为 degree=1 只能拟合一条直线，而本实验的真实函数是 `sin(1.5x)+0.25x`，明显包含弯曲的非线性趋势。它不能充分捕捉数据的主要形状，所以训练误差和测试误差都相对较高。

### 4.2 谁最像过拟合？

**degree=15 最像过拟合。**  
degree=15 的模型非常灵活，可以生成很多不必要的弯曲。虽然它可能在训练集上取得较低误差，但这种低误差不一定代表模型真正学到了稳定规律，更可能是把训练样本中的随机噪声也学进去了。

### 4.3 如果必须选一个上线，我会选谁？

如果今天必须从 degree=1、degree=4、degree=15 中选择一个上线，我会先选择 **degree=4**。

原因是：degree=4 在三位候选模型中更好地平衡了拟合能力和泛化稳定性。它比 degree=1 更能捕捉非线性趋势，同时又不像 degree=15 那样出现明显过拟合风险。本实验中 degree=4 的 train RMSE 为 **{production_choice["train_rmse"]:.4f}**，test RMSE 为 **{production_choice["test_rmse"]:.4f}**，整体表现更加适合作为上线候选。

这里不能简单地说“谁某一次 test RMSE 最低就上线”。如果模型曲线已经明显抖动，或者在 repeated sampling 中表现出 high variance，即使某一次测试划分中结果看起来不错，也不应直接认为它更可靠。

---

## 5. Task B：复杂度-误差曲线

输出图像：`results/figures/error_curves.png`

{markdown_table(["degree", "train RMSE", "test RMSE", "generalization gap"], complexity_rows)}

### 5.1 测试误差最低的复杂度

在 degree=1 到 degree=18 的扫描中，测试误差最低的复杂度是 **degree={best_test_result["degree"]}**，对应 test RMSE = **{best_test_result["test_rmse"]:.4f}**。

### 5.2 泛化 gap 最大的位置

泛化 gap 最大的复杂度主要出现在：

**{gap_degree_text}**

泛化 gap 指的是：

```text
generalization gap = test RMSE - train RMSE
```

当 gap 明显变大时，说明模型在训练集上表现很好，但在测试集上表现明显变差，这通常意味着模型开始过度适应训练数据。

### 5.3 为什么训练误差最低的模型不一定最好？

训练误差只反映模型对已见样本的拟合程度。模型越复杂，越容易贴近训练点，因此 train RMSE 往往会下降。  
但是测试误差反映的是模型面对新数据时的表现。如果模型复杂度过高，它可能会把训练数据中的噪声也当成规律，从而导致测试误差上升。因此，训练误差最低的模型不一定是最适合上线的模型。

---

## 6. Task C：Repeated Sampling 展示 Variance

输出图像：`results/figures/variance_demo.png`

{markdown_table(["degree", "mean prediction std", "max prediction std"], variance_rows)}

### 6.1 图像观察

在 repeated sampling 实验中，我保持真实函数不变，只改变每一次抽到的训练样本。  
如果模型稳定，那么不同训练样本得到的拟合曲线应该比较接近；如果模型 high variance，那么只要训练样本稍微变化，拟合曲线就会明显改变。

从图中可以看到：

- degree=2 的多条曲线相对集中，说明它对训练样本变化不太敏感；
- degree=15 的多条曲线明显更分散，部分区域甚至出现剧烈波动，说明它具有更强的 high variance。

### 6.2 一句话回答

high variance model 的危险，不是它不会拟合训练集，而是它对 **训练样本的随机变化** 过于敏感。

---

## 7. Task D：异常值攻击 RMSE 与 MAE

输出图像：`results/figures/loss_outlier_comparison.png`

{markdown_table(["scenario", "RMSE", "MAE"], loss_rows)}

### 7.1 为什么 RMSE 更容易被大错拉高？

RMSE 在计算时会先对误差平方，再取平均和开方。  
平方操作会放大大误差的影响。例如，一个误差为 10 的样本，在平方后会变成 100，因此一个严重错误就可能显著抬高整体 RMSE。

相比之下，MAE 只计算绝对误差，不会进行平方放大，所以它对异常值更加稳健。

### 7.2 如果线上系统偶尔一次大错的代价极高，更应该看哪个指标？

如果线上系统偶尔一次大错的代价极高，我会更重视 **RMSE**。  
因为 RMSE 会更严厉地惩罚大误差，更容易暴露模型中少数严重预测错误的风险。

### 7.3 如果数据天然包含较多异常值，会不会重新考虑指标选择？

会。  
如果数据本身包含较多异常值，而这些异常值不一定代表模型真正失败，那么只看 RMSE 可能会过度放大少数样本的影响。这种情况下，我会更重视 **MAE**，或者同时报告 RMSE 和 MAE：RMSE 用来看大错风险，MAE 用来看普通样本上的典型误差水平。

---

## 8. 本周最重要的三条结论

1. **训练误差下降不代表测试误差一定下降。** 随着多项式 degree 增加，模型更容易拟合训练集，因此 train RMSE 通常下降；但如果模型复杂度过高，test RMSE 可能上升。
2. **High variance 在图上表现为多次训练曲线明显分散。** 当训练样本稍微变化，高复杂度模型的拟合曲线也明显变化，说明它对数据扰动过于敏感。
3. **RMSE 和 MAE 体现不同的风险偏好。** RMSE 更关注大错风险，MAE 更稳健，更能反映普通样本的平均误差。

---

## 9. 哪张图最能代表“过拟合是可见现象”

我认为 `error_curves.png` 最能代表过拟合。  
因为它直接显示出：随着 degree 增加，train RMSE 可以继续下降，但 test RMSE 在某个阶段后不再下降，甚至明显上升。这说明过拟合不是抽象概念，而是可以通过训练误差和测试误差的分离被观察到的现象。

同时，`candidate_models.png` 也能直观看出 degree=15 的曲线过于灵活，容易围绕训练样本产生不必要的波动；`variance_demo.png` 则进一步说明高阶模型对训练样本变化更加敏感。

---

## 10. 指标选择判断

- 当我更关心**大误差风险**，或者业务中少数严重预测错误会带来很高损失时，我更愿意报告 **RMSE**。
- 当数据中存在较多异常值，或者我更想描述模型在普通样本上的平均表现时，我更愿意报告 **MAE**。
- 实际项目中，我通常会同时报告 RMSE 和 MAE：RMSE 用来观察大错风险，MAE 用来观察整体稳健表现。

---

## 11. 与下一周 Ridge / Lasso 的连接

如果模型复杂度过高会带来 high variance，那么下一步自然会想到正则化。  
Ridge 和 Lasso 的核心思路是在拟合训练数据的同时限制模型参数不要过大，从而降低模型对训练噪声的敏感性。换句话说，正则化不是单纯追求更低的训练误差，而是在训练误差和模型稳定性之间做平衡，因此它正好是应对 high variance 的自然工具。
"""

    SUMMARY_PATH.write_text(report, encoding="utf-8")


# ---------------------------------------------------------------------
# 7. 主入口
# ---------------------------------------------------------------------
def main() -> None:
    """Week 12 作业唯一执行入口。"""
    ensure_output_dirs()

    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_grid,
        y_true_grid,
        data_info,
    ) = generate_data()

    candidate_results = run_model_complexity_demo(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        X_grid=X_grid,
        y_true_grid=y_true_grid,
    )

    error_curve_results = run_error_curve_demo(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    variance_results = run_variance_demo(true_func=true_function)

    loss_results = run_loss_comparison_demo()

    write_summary_report(
        data_info=data_info,
        candidate_results=candidate_results,
        error_curve_results=error_curve_results,
        variance_results=variance_results,
        loss_results=loss_results,
    )

    print("\nDone.")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"Summary saved to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
