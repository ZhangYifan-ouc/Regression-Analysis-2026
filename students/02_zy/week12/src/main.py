import shutil
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


WEEK12_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = WEEK12_DIR / "src"
RESULTS_DIR = WEEK12_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

sys.path.insert(0, str(SRC_DIR))

from utils.metrics import calculate_mae, calculate_rmse


RANDOM_SEED = 42


def ensure_directories() -> None:
    """
    Prepare output folders.

    Every time the script runs, old results are removed and new results are created.
    """
    if RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def true_function(x: np.ndarray) -> np.ndarray:
    """
    Define the true nonlinear function.

    This function is the real pattern we want models to learn.
    """
    return np.sin(x) + 0.25 * x


def generate_data(n_samples: int = 140):
    """
    Generate one-dimensional nonlinear regression data.

    Output:
        X_train, X_test, y_train, y_test, x_grid, y_true_grid

    This is the stage for demonstrating underfitting and overfitting.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    X = rng.uniform(-3, 3, size=n_samples)
    noise = rng.normal(0, 0.28, size=n_samples)
    y = true_function(X) + noise

    X = X.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.35,
        random_state=RANDOM_SEED,
    )

    x_grid = np.linspace(-3.2, 3.2, 500).reshape(-1, 1)
    y_true_grid = true_function(x_grid.ravel())

    return X_train, X_test, y_train, y_test, x_grid, y_true_grid


def build_polynomial_model(degree: int) -> Pipeline:
    """
    Build a polynomial regression model.

    degree controls model complexity.
    """
    return Pipeline(
        [
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("linear", LinearRegression()),
        ]
    )


def evaluate_model(model, X_train, y_train, X_test, y_test) -> dict:
    """
    Fit model and calculate train/test RMSE.
    """
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    return {
        "train_rmse": calculate_rmse(y_train, train_pred),
        "test_rmse": calculate_rmse(y_test, test_pred),
    }


def run_candidate_models_demo() -> list[dict]:
    """
    Task A:
    Compare degree = 1, 4, 15.

    Output:
        figures/candidate_models.png
    """
    X_train, X_test, y_train, y_test, x_grid, y_true_grid = generate_data()

    candidate_degrees = [1, 4, 15]
    records = []

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, degree in zip(axes, candidate_degrees):
        model = build_polynomial_model(degree)
        metrics = evaluate_model(model, X_train, y_train, X_test, y_test)

        y_grid_pred = model.predict(x_grid)

        ax.scatter(X_train.ravel(), y_train, label="Train data", alpha=0.75)
        ax.scatter(X_test.ravel(), y_test, label="Test data", alpha=0.75)
        ax.plot(x_grid.ravel(), y_true_grid, label="True function", linewidth=2)
        ax.plot(x_grid.ravel(), y_grid_pred, label=f"Degree {degree}", linewidth=2)

        ax.set_title(
            f"Degree = {degree}\n"
            f"Train RMSE = {metrics['train_rmse']:.3f}, "
            f"Test RMSE = {metrics['test_rmse']:.3f}"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(fontsize=8)

        records.append(
            {
                "degree": degree,
                "train_rmse": metrics["train_rmse"],
                "test_rmse": metrics["test_rmse"],
            }
        )

    fig.suptitle("Candidate Polynomial Models: Underfitting vs Overfitting", fontsize=16)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "candidate_models.png", dpi=200)
    plt.close(fig)

    return records


def run_error_curve_demo() -> pd.DataFrame:
    """
    Task B:
    Scan degree = 1 to 18 and plot train/test RMSE curves.

    Output:
        figures/error_curves.png
    """
    X_train, X_test, y_train, y_test, _, _ = generate_data()

    records = []

    for degree in range(1, 19):
        model = build_polynomial_model(degree)
        metrics = evaluate_model(model, X_train, y_train, X_test, y_test)

        records.append(
            {
                "degree": degree,
                "train_rmse": metrics["train_rmse"],
                "test_rmse": metrics["test_rmse"],
                "generalization_gap": metrics["test_rmse"] - metrics["train_rmse"],
            }
        )

    df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(df["degree"], df["train_rmse"], marker="o", label="Train RMSE")
    ax.plot(df["degree"], df["test_rmse"], marker="o", label="Test RMSE")

    ax.set_title("Model Complexity vs Error")
    ax.set_xlabel("Polynomial Degree")
    ax.set_ylabel("RMSE")
    ax.set_xticks(df["degree"])
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "error_curves.png", dpi=200)
    plt.close(fig)

    return df


def run_variance_demo(n_repeats: int = 20) -> pd.DataFrame:
    """
    Task C:
    Repeated sampling demo.

    We fit degree = 2 and degree = 15 many times.
    If the fitted curves change a lot across samples, the model has high variance.

    Output:
        figures/variance_demo.png
    """
    rng = np.random.default_rng(RANDOM_SEED)

    x_grid = np.linspace(-3.2, 3.2, 500).reshape(-1, 1)
    y_true_grid = true_function(x_grid.ravel())

    degrees = [2, 15]
    summary_records = []

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, degree in zip(axes, degrees):
        predictions = []

        for _ in range(n_repeats):
            X = rng.uniform(-3, 3, size=45)
            noise = rng.normal(0, 0.30, size=45)
            y = true_function(X) + noise
            X = X.reshape(-1, 1)

            model = build_polynomial_model(degree)
            model.fit(X, y)

            y_grid_pred = model.predict(x_grid)
            predictions.append(y_grid_pred)

            ax.plot(x_grid.ravel(), y_grid_pred, alpha=0.35)

        predictions = np.asarray(predictions)

        prediction_std = np.std(predictions, axis=0)
        mean_prediction_std = float(np.mean(prediction_std))
        max_prediction_std = float(np.max(prediction_std))

        ax.plot(x_grid.ravel(), y_true_grid, linewidth=3, label="True function")
        ax.set_title(
            f"Degree = {degree}\n"
            f"Mean prediction std = {mean_prediction_std:.3f}"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_ylim(-4, 4)
        ax.legend()

        summary_records.append(
            {
                "degree": degree,
                "mean_prediction_std": mean_prediction_std,
                "max_prediction_std": max_prediction_std,
            }
        )

    fig.suptitle("Repeated Sampling: Visualizing Model Variance", fontsize=16)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "variance_demo.png", dpi=200)
    plt.close(fig)

    return pd.DataFrame(summary_records)


def run_loss_comparison_demo() -> pd.DataFrame:
    """
    Task D:
    Compare RMSE and MAE under one large outlier.

    Output:
        figures/loss_outlier_comparison.png
    """
    rng = np.random.default_rng(RANDOM_SEED)

    n = 80
    y_true = np.linspace(20, 100, n)

    y_pred_clean = y_true + rng.normal(0, 3, size=n)

    y_pred_outlier = y_pred_clean.copy()
    y_pred_outlier[-1] = y_true[-1] + 80

    records = [
        {
            "scenario": "Clean prediction",
            "RMSE": calculate_rmse(y_true, y_pred_clean),
            "MAE": calculate_mae(y_true, y_pred_clean),
        },
        {
            "scenario": "One large outlier",
            "RMSE": calculate_rmse(y_true, y_pred_outlier),
            "MAE": calculate_mae(y_true, y_pred_outlier),
        },
    ]

    df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(df))
    width = 0.35

    ax.bar(x - width / 2, df["RMSE"], width, label="RMSE")
    ax.bar(x + width / 2, df["MAE"], width, label="MAE")

    ax.set_xticks(x)
    ax.set_xticklabels(df["scenario"])
    ax.set_ylabel("Error value")
    ax.set_title("RMSE vs MAE Under One Large Outlier")
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "loss_outlier_comparison.png", dpi=200)
    plt.close(fig)

    return df


def dataframe_to_markdown(df: pd.DataFrame, digits: int = 4) -> str:
    """
    Convert dataframe to markdown table without using tabulate.
    """
    df = df.copy()

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].round(digits)

    columns = df.columns.tolist()

    lines = []

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"

    lines.append(header)
    lines.append(separator)

    for _, row in df.iterrows():
        values = [str(row[col]) for col in columns]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)

def write_summary_report(
    candidate_records: list[dict],
    error_curve_df: pd.DataFrame,
    variance_df: pd.DataFrame,
    loss_df: pd.DataFrame,
) -> None:
    """
    Task F:
    Write Markdown summary report.
    """
    summary_path = RESULTS_DIR / "summary.md"

    candidate_df = pd.DataFrame(candidate_records)

    best_row = error_curve_df.loc[error_curve_df["test_rmse"].idxmin()]
    largest_gap_row = error_curve_df.loc[error_curve_df["generalization_gap"].idxmax()]

    content = "\n".join(
        [
            "# Week 12：Bias-Variance Visual Lab 总结报告",
            "",
            "## 1. 本周实验目标",
            "",
            "本周作业的目标不是直接背诵偏差和方差的定义，而是通过 Python 脚本生成图像和表格，观察模型复杂度、训练误差、测试误差、模型方差以及 RMSE 和 MAE 对异常值的不同反应。",
            "",
            "本脚本运行后自动生成四张图：",
            "",
            "- `figures/candidate_models.png`：三个候选模型的拟合效果；",
            "- `figures/error_curves.png`：模型复杂度与训练/测试误差曲线；",
            "- `figures/variance_demo.png`：重复抽样下的高方差现象；",
            "- `figures/loss_outlier_comparison.png`：RMSE 和 MAE 面对异常值时的变化。",
            "",
            "## 2. Task A：三个候选模型对比",
            "",
            dataframe_to_markdown(candidate_df),
            "",
            "从图像上看，`degree = 1` 的模型太简单，只能拟合一条直线，无法捕捉真实函数的弯曲趋势，因此最像欠拟合；`degree = 15` 的模型非常复杂，能够贴近训练点，但曲线容易出现不必要的剧烈波动，因此最像过拟合；`degree = 4` 的模型在复杂度和稳定性之间更折中，如果必须选择一个上线，我会优先选择 `degree = 4`，因为它既能捕捉非线性趋势，又不会像高阶模型那样对训练数据过于敏感。",
            "",
            "## 3. Task B：复杂度-误差曲线",
            "",
            "完整 degree 扫描结果如下：",
            "",
            dataframe_to_markdown(error_curve_df),
            "",
            f"测试误差最低的复杂度是 `degree = {int(best_row['degree'])}`，对应 test RMSE 为 `{best_row['test_rmse']:.4f}`。",
            "",
            f"泛化 gap 最大出现在 `degree = {int(largest_gap_row['degree'])}` 附近，对应 gap 为 `{largest_gap_row['generalization_gap']:.4f}`。",
            "",
            "训练误差通常会随着模型复杂度增加而下降，因为复杂模型更容易贴合训练数据。但是测试误差不一定一直下降，当模型开始学习训练集中的噪声时，测试误差反而会上升。因此，训练误差最低的模型不一定是最好的模型。",
            "",
            "## 4. Task C：Repeated Sampling 与 High Variance",
            "",
            dataframe_to_markdown(variance_df),
            "",
            "在 repeated sampling 图中，低阶模型的多条拟合曲线通常比较集中，而高阶模型的多条曲线变化更剧烈。这说明高复杂度模型对训练样本变化更加敏感，也就是 high variance。",
            "",
            "一句话总结：high variance model 的危险，不是它不会拟合训练集，而是它对 **训练数据的微小变化** 过于敏感。",
            "",
            "## 5. Task D：RMSE 与 MAE 面对异常值的差异",
            "",
            dataframe_to_markdown(loss_df),
            "",
            "RMSE 会先平方误差再开方，因此一个很大的错误会被平方放大，所以 RMSE 更容易被大错拉高。MAE 使用绝对误差，虽然也会受到异常值影响，但反应没有 RMSE 那么剧烈。",
            "",
            "如果线上系统中偶尔一次大错的代价极高，我更想看 RMSE，因为它会更明显地惩罚大错误。如果数据天然包含较多异常值，并且这些异常值不一定代表模型失败，而可能是数据本身噪声较大，我会同时看 MAE，因为 MAE 更稳健，更能反映一般情况下的平均误差。",
            "",
            "## 6. 本周最重要的三条结论",
            "",
            "1. 模型复杂度增加时，训练误差通常会下降，但测试误差可能先下降后上升，这就是过拟合开始出现的现象。",
            "2. high variance 在图上表现为：同样的真实函数下，只要训练样本稍微变化，高复杂度模型的拟合曲线就会明显改变。",
            "3. RMSE 比 MAE 更容易被异常值拉高，因此 RMSE 更关注大错风险，而 MAE 更适合描述一般水平的平均误差。",
            "",
            "## 7. 最能代表过拟合的图",
            "",
            "我认为 `figures/error_curves.png` 最能代表过拟合不是抽象概念，而是可见现象。因为它清楚展示了训练误差持续下降，但测试误差在某个复杂度之后不再下降，甚至开始上升。这个图说明模型并不是越复杂越好。",
            "",
            "## 8. 指标选择判断",
            "",
            "如果业务场景非常害怕大错误，例如医疗费用预测、风控损失预测或重要预算预测，我更愿意报告 RMSE，因为 RMSE 会更重视大误差。如果数据中天然包含较多异常值，或者我想解释模型在一般样本上的平均表现，我更愿意报告 MAE，因为 MAE 更稳健、更容易向业务方解释。",
            "",
            "## 9. 与下一周正则化的连接",
            "",
            "如果模型复杂度过高会带来 high variance，那么下一步自然会想到正则化，例如 Ridge 和 Lasso。正则化的作用是限制模型系数不要过大，从而降低模型对训练样本噪声的过度敏感，让模型在训练误差和泛化能力之间取得更稳定的平衡。",
            "",
        ]
    )

    summary_path.write_text(content, encoding="utf-8")


def main() -> None:
    ensure_directories()

    print("===== Week 12 Bias-Variance Visual Lab Started =====")

    print("[Stage 1] Comparing candidate polynomial models...")
    candidate_records = run_candidate_models_demo()

    print("[Stage 2] Sweeping model complexity from degree 1 to 18...")
    error_curve_df = run_error_curve_demo()

    print("[Stage 3] Running repeated sampling variance demo...")
    variance_df = run_variance_demo()

    print("[Stage 4] Comparing RMSE and MAE under outlier attack...")
    loss_df = run_loss_comparison_demo()

    print("[Stage 5] Writing summary report...")
    write_summary_report(
        candidate_records=candidate_records,
        error_curve_df=error_curve_df,
        variance_df=variance_df,
        loss_df=loss_df,
    )

    print("===== Week 12 Finished =====")
    print("Figures saved to:", FIGURES_DIR)
    print("Summary saved to:", RESULTS_DIR / "summary.md")


if __name__ == "__main__":
    main()