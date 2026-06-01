"""
Week 12 Assignment: Bias-Variance Visual Lab

运行命令：
    uv run src/week12/main.py

运行后会生成：
    src/week12/figures/candidate_models.png
    src/week12/figures/error_curves.png
    src/week12/figures/variance_demo.png
    src/week12/figures/loss_outlier_comparison.png
    src/week12/results/summary.md
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


# ============================================================
# 1. 路径设置
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 2. 指标函数
# ============================================================

def calculate_rmse(y_true, y_pred):
    """
    计算 RMSE。

    RMSE = sqrt(mean((y_true - y_pred)^2))

    特点：
    误差会先平方，所以 RMSE 对大误差、异常值更加敏感。
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return math.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_mae(y_true, y_pred):
    """
    计算 MAE。

    MAE = mean(abs(y_true - y_pred))

    特点：
    MAE 只看绝对误差，不会平方放大大误差，所以比 RMSE 更稳健。
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def dataframe_to_markdown_table(df, float_digits=4):
    """
    把 pandas DataFrame 转成 Markdown 表格。

    这里不用 df.to_markdown()，是为了避免额外依赖 tabulate。
    """
    headers = list(df.columns)

    def format_value(value):
        if isinstance(value, float):
            return f"{value:.{float_digits}f}"
        return str(value)

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for _, row in df.iterrows():
        values = [format_value(row[col]) for col in headers]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


# ============================================================
# 3. 数据生成与模型工具
# ============================================================

def true_function(x):
    """
    构造真实函数。

    这里使用 sin(x) + 线性趋势。
    这样做的好处是：真实关系不是一条直线，方便观察欠拟合和过拟合。
    """
    return np.sin(x) + 0.25 * x


def generate_data(n_samples=160, noise_std=0.35, random_state=42):
    """
    生成一维非线性回归数据，并划分训练集和测试集。
    """
    rng = np.random.default_rng(random_state)

    x = rng.uniform(0, 10, size=n_samples)
    x = np.sort(x)

    y_clean = true_function(x)
    y = y_clean + rng.normal(0, noise_std, size=n_samples)

    x_train, x_test, y_train, y_test = train_test_split(
        x.reshape(-1, 1),
        y,
        test_size=0.3,
        random_state=random_state,
    )

    return x, y_clean, x_train, x_test, y_train, y_test


def build_polynomial_model(degree):
    """
    构建多项式回归模型。

    PolynomialFeatures:
        把原始 x 扩展成 x、x^2、x^3 等多项式特征。

    LinearRegression:
        在线性回归框架下拟合这些多项式特征。
    """
    model = Pipeline(
        steps=[
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("linear", LinearRegression()),
        ]
    )
    return model


def fit_and_evaluate(degree, x_train, x_test, y_train, y_test):
    """
    对某个 degree 的多项式模型进行训练，并计算训练集和测试集 RMSE。
    """
    model = build_polynomial_model(degree)
    model.fit(x_train, y_train)

    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    train_rmse = calculate_rmse(y_train, train_pred)
    test_rmse = calculate_rmse(y_test, test_pred)

    return model, train_rmse, test_rmse


# ============================================================
# 4. Task A：候选模型对比
# ============================================================

def run_model_complexity_demo():
    """
    比较三个复杂度不同的模型：
    degree=1：一般代表欠拟合
    degree=4：相对合理
    degree=15：容易过拟合
    """
    print("[Stage 1] Comparing candidate polynomial models...")

    x, y_clean, x_train, x_test, y_train, y_test = generate_data()

    degrees = [1, 4, 15]
    x_grid = np.linspace(0, 10, 400).reshape(-1, 1)
    y_true_grid = true_function(x_grid.ravel())

    records = []

    plt.figure(figsize=(11, 7))

    plt.scatter(
        x_train.ravel(),
        y_train,
        s=35,
        alpha=0.75,
        label="Training points",
    )

    plt.scatter(
        x_test.ravel(),
        y_test,
        s=35,
        alpha=0.75,
        marker="x",
        label="Testing points",
    )

    plt.plot(
        x_grid.ravel(),
        y_true_grid,
        linewidth=3,
        label="True function",
    )

    for degree in degrees:
        model, train_rmse, test_rmse = fit_and_evaluate(
            degree,
            x_train,
            x_test,
            y_train,
            y_test,
        )

        y_grid_pred = model.predict(x_grid)

        plt.plot(
            x_grid.ravel(),
            y_grid_pred,
            linewidth=2,
            label=f"degree={degree}, train RMSE={train_rmse:.3f}, test RMSE={test_rmse:.3f}",
        )

        records.append(
            {
                "degree": degree,
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
                "generalization_gap": test_rmse - train_rmse,
            }
        )

    plt.title("Candidate Models: Underfitting, Reasonable Fit, and Overfitting")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(fontsize=9)
    plt.tight_layout()

    output_path = os.path.join(FIGURES_DIR, "candidate_models.png")
    plt.savefig(output_path, dpi=200)
    plt.close()

    return pd.DataFrame(records)


# ============================================================
# 5. Task B：复杂度-误差曲线
# ============================================================

def run_error_curve_demo():
    """
    从 degree=1 到 degree=18 扫描模型复杂度。

    观察：
    训练误差是否持续下降？
    测试误差是否也持续下降？
    什么时候出现过拟合迹象？
    """
    print("[Stage 2] Sweeping model complexity...")

    x, y_clean, x_train, x_test, y_train, y_test = generate_data()

    records = []

    for degree in range(1, 19):
        model, train_rmse, test_rmse = fit_and_evaluate(
            degree,
            x_train,
            x_test,
            y_train,
            y_test,
        )

        records.append(
            {
                "degree": degree,
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
                "generalization_gap": test_rmse - train_rmse,
            }
        )

    error_df = pd.DataFrame(records)

    plt.figure(figsize=(10, 6))

    plt.plot(
        error_df["degree"],
        error_df["train_rmse"],
        marker="o",
        linewidth=2,
        label="Train RMSE",
    )

    plt.plot(
        error_df["degree"],
        error_df["test_rmse"],
        marker="o",
        linewidth=2,
        label="Test RMSE",
    )

    plt.title("Model Complexity vs. Error")
    plt.xlabel("Polynomial degree")
    plt.ylabel("RMSE")
    plt.xticks(range(1, 19))
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(FIGURES_DIR, "error_curves.png")
    plt.savefig(output_path, dpi=200)
    plt.close()

    return error_df


# ============================================================
# 6. Task C：Repeated Sampling 展示 variance
# ============================================================

def run_variance_demo(n_repeats=20):
    """
    固定真实函数，只改变训练样本，重复训练模型。

    如果模型复杂度很高，那么不同训练样本会导致拟合曲线差异很大。
    这就是 high variance 的直观表现。
    """
    print("[Stage 3] Demonstrating high variance by repeated sampling...")

    degrees = [2, 15]
    x_grid = np.linspace(0, 10, 400).reshape(-1, 1)
    y_true_grid = true_function(x_grid.ravel())

    summary_records = []

    plt.figure(figsize=(12, 5))

    for subplot_index, degree in enumerate(degrees, start=1):
        all_predictions = []

        plt.subplot(1, 2, subplot_index)

        plt.plot(
            x_grid.ravel(),
            y_true_grid,
            linewidth=3,
            label="True function",
        )

        for seed in range(n_repeats):
            _, _, x_train, _, y_train, _ = generate_data(
                n_samples=70,
                noise_std=0.35,
                random_state=100 + seed,
            )

            model = build_polynomial_model(degree)
            model.fit(x_train, y_train)

            y_grid_pred = model.predict(x_grid)
            all_predictions.append(y_grid_pred)

            plt.plot(
                x_grid.ravel(),
                y_grid_pred,
                linewidth=1,
                alpha=0.55,
            )

        prediction_array = np.vstack(all_predictions)
        prediction_std = prediction_array.std(axis=0)

        summary_records.append(
            {
                "degree": degree,
                "mean_prediction_std": float(np.mean(prediction_std)),
                "max_prediction_std": float(np.max(prediction_std)),
            }
        )

        plt.title(f"Repeated Sampling: degree={degree}")
        plt.xlabel("x")
        plt.ylabel("prediction")
        plt.legend(fontsize=9)

    plt.tight_layout()

    output_path = os.path.join(FIGURES_DIR, "variance_demo.png")
    plt.savefig(output_path, dpi=200)
    plt.close()

    return pd.DataFrame(summary_records)


# ============================================================
# 7. Task D：异常值对 RMSE 和 MAE 的影响
# ============================================================

def run_loss_comparison_demo():
    """
    比较 RMSE 和 MAE 对异常值的不同反应。

    操作：
    先构造一组正常预测；
    再人为加入一个很大的预测错误；
    比较 RMSE 和 MAE 的变化。
    """
    print("[Stage 4] Comparing RMSE and MAE under one large outlier...")

    rng = np.random.default_rng(123)

    y_true = np.linspace(10, 100, 40)
    y_pred_clean = y_true + rng.normal(0, 3, size=len(y_true))

    y_pred_outlier = y_pred_clean.copy()
    y_pred_outlier[-1] = y_pred_outlier[-1] + 80

    records = [
        {
            "scenario": "clean prediction",
            "rmse": calculate_rmse(y_true, y_pred_clean),
            "mae": calculate_mae(y_true, y_pred_clean),
        },
        {
            "scenario": "one large outlier",
            "rmse": calculate_rmse(y_true, y_pred_outlier),
            "mae": calculate_mae(y_true, y_pred_outlier),
        },
    ]

    loss_df = pd.DataFrame(records)

    plt.figure(figsize=(8, 6))

    x_labels = loss_df["scenario"].tolist()
    x_pos = np.arange(len(x_labels))
    width = 0.35

    plt.bar(
        x_pos - width / 2,
        loss_df["rmse"],
        width,
        label="RMSE",
    )

    plt.bar(
        x_pos + width / 2,
        loss_df["mae"],
        width,
        label="MAE",
    )

    plt.title("RMSE vs. MAE: Impact of One Large Outlier")
    plt.xlabel("Prediction scenario")
    plt.ylabel("Error value")
    plt.xticks(x_pos, x_labels)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(FIGURES_DIR, "loss_outlier_comparison.png")
    plt.savefig(output_path, dpi=200)
    plt.close()

    return loss_df


# ============================================================
# 8. Task E / F：写总结报告
# ============================================================

def write_summary_report(candidate_df, error_df, variance_df, loss_df):
    """
    写出 Markdown 总结报告。
    """
    print("[Stage 5] Writing summary report...")

    best_test_row = error_df.loc[error_df["test_rmse"].idxmin()]
    largest_gap_row = error_df.loc[error_df["generalization_gap"].idxmax()]

    candidate_table = dataframe_to_markdown_table(candidate_df)
    error_table = dataframe_to_markdown_table(error_df)
    variance_table = dataframe_to_markdown_table(variance_df)
    loss_table = dataframe_to_markdown_table(loss_df)

    report_parts = []

    report_parts.append("# Week 12 Summary Report\n\n")

    report_parts.append("## 1. 本周作业目标\n\n")
    report_parts.append(
        "本周我主要用 Python 脚本把 bias-variance 相关现象可视化出来。"
        "和单纯背概念相比，这次作业更强调通过图像和数值结果观察模型复杂度变化、"
        "过拟合现象、high variance 现象，以及 RMSE 和 MAE 对异常值的不同反应。\n\n"
    )
    report_parts.append("本次脚本的唯一运行入口是：\n\n")
    report_parts.append("```bash\nuv run src/week12/main.py\n```\n\n")

    report_parts.append("## 2. Task A：三个候选模型对比\n\n")
    report_parts.append(
        "本部分比较了三个不同复杂度的多项式模型：degree=1、degree=4、degree=15。\n\n"
    )
    report_parts.append("生成图像：`src/week12/figures/candidate_models.png`\n\n")
    report_parts.append("### 候选模型结果表\n\n")
    report_parts.append(candidate_table + "\n\n")
    report_parts.append(
        "degree=1 的模型最像欠拟合，因为它只有一条比较简单的直线，"
        "连数据中的弯曲趋势都没有很好地跟上。degree=15 的模型最像过拟合，"
        "它在训练数据附近会表现得很灵活，但这种灵活不一定代表真的学到了稳定规律，"
        "反而可能是在追着训练样本里的噪声跑。如果必须选择一个上线，"
        "我会优先选择 degree=4，因为它在复杂度和泛化能力之间相对更平衡。\n\n"
    )

    report_parts.append("## 3. Task B：复杂度-误差曲线\n\n")
    report_parts.append(
        "本部分扫描了 degree=1 到 degree=18 的模型复杂度，并分别计算 train RMSE、"
        "test RMSE 和 generalization gap。\n\n"
    )
    report_parts.append("生成图像：`src/week12/figures/error_curves.png`\n\n")
    report_parts.append("### 复杂度成绩单\n\n")
    report_parts.append(error_table + "\n\n")
    report_parts.append(
        f"测试误差最低的复杂度是 degree={int(best_test_row['degree'])}，"
        f"test RMSE={best_test_row['test_rmse']:.4f}。\n\n"
    )
    report_parts.append(
        f"泛化 gap 最大的位置大概在 degree={int(largest_gap_row['degree'])}，"
        f"generalization gap={largest_gap_row['generalization_gap']:.4f}。\n\n"
    )
    report_parts.append(
        "从结果可以看出，训练误差通常会随着模型复杂度增加而下降，"
        "但这不代表模型一定更好。训练误差低只能说明模型更会拟合训练集，"
        "不能保证它在新数据上也稳定。如果测试误差开始上升，"
        "就说明模型可能已经开始过拟合了。\n\n"
    )

    report_parts.append("## 4. Task C：Repeated Sampling 展示 variance\n\n")
    report_parts.append(
        "本部分固定真实函数，只改变训练样本，分别对 degree=2 和 degree=15 "
        "重复抽样训练多次。这样可以观察模型对训练样本变化是否敏感。\n\n"
    )
    report_parts.append("生成图像：`src/week12/figures/variance_demo.png`\n\n")
    report_parts.append("### Variance 数值总结\n\n")
    report_parts.append(variance_table + "\n\n")
    report_parts.append(
        "high variance model 的危险，不是它不会拟合训练集，"
        "而是它对训练样本中的小变化和噪声过于敏感。\n\n"
    )

    report_parts.append("## 5. Task D：RMSE 和 MAE 对异常值的不同反应\n\n")
    report_parts.append(
        "本部分构造了一组干净预测结果，然后人为加入一个很大的预测错误，"
        "比较 RMSE 和 MAE 的变化。\n\n"
    )
    report_parts.append("生成图像：`src/week12/figures/loss_outlier_comparison.png`\n\n")
    report_parts.append("### RMSE / MAE 对比表\n\n")
    report_parts.append(loss_table + "\n\n")
    report_parts.append(
        "RMSE 更容易被大错拉高，因为它会先把误差平方，再求平均。"
        "一个特别大的误差经过平方以后，会在整体指标里占很大比重。"
        "如果线上系统偶尔一次大错的代价极高，我会更想看 RMSE。"
        "但是如果数据本身天然包含很多异常值，MAE 可能更稳一些，"
        "因为它不会像 RMSE 那样被少数极端错误强烈拉动。\n\n"
    )

    report_parts.append("## 6. 本周最重要的三条结论\n\n")
    report_parts.append(
        "第一，模型复杂度变高以后，训练误差往往会下降，但测试误差不一定下降，"
        "所以不能只看训练集表现。\n\n"
    )
    report_parts.append(
        "第二，过拟合不是一个抽象概念，它可以通过拟合曲线和误差曲线直接看出来。"
        "尤其是高阶多项式模型，它可能把训练集附近拟合得很好，"
        "但在整体趋势上变得很不稳定。\n\n"
    )
    report_parts.append(
        "第三，RMSE 和 MAE 代表了不同的风险偏好。"
        "RMSE 更关注大错，MAE 更关注平均意义上的普通误差。\n\n"
    )

    report_parts.append("## 7. 最能代表过拟合的图\n\n")
    report_parts.append(
        "我认为 error_curves.png 最能代表过拟合。因为它同时展示了 train RMSE "
        "和 test RMSE 的变化：当复杂度继续增加时，训练误差可能还在下降，"
        "但测试误差不一定同步下降，甚至可能变差。这个现象比单独看一张拟合曲线更直观。\n\n"
    )

    report_parts.append("## 8. 指标选择判断\n\n")
    report_parts.append(
        "如果我更关心模型有没有出现特别严重的大错，我会报告 RMSE。"
        "例如医疗费用预测、金融风险预测这类场景，一次大错可能就会带来较大影响。\n\n"
    )
    report_parts.append(
        "如果我面对的数据本来就有较多异常点，或者我更关心模型的一般平均表现，"
        "我会报告 MAE。因为 MAE 比 RMSE 更不容易被少数极端值带偏。\n\n"
    )

    report_parts.append("## 9. 与下一周正则化的连接\n\n")
    report_parts.append(
        "如果模型复杂度过高会带来 high variance，那么下一步自然会想到正则化。"
        "Ridge 和 Lasso 的作用可以理解为给模型复杂度加一个约束，"
        "让模型不要为了追逐训练集中的噪声而把参数调得过于极端。"
        "这样做的目的不是让训练误差最低，而是希望模型在新数据上更稳定。\n"
    )

    report = "".join(report_parts)

    output_path = os.path.join(RESULTS_DIR, "summary.md")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    return output_path


# ============================================================
# 9. 主函数
# ============================================================

def main():
    print("=" * 60)
    print("Week 12 Assignment: Bias-Variance Visual Lab")
    print("=" * 60)

    candidate_df = run_model_complexity_demo()
    error_df = run_error_curve_demo()
    variance_df = run_variance_demo()
    loss_df = run_loss_comparison_demo()

    report_path = write_summary_report(
        candidate_df=candidate_df,
        error_df=error_df,
        variance_df=variance_df,
        loss_df=loss_df,
    )

    print("\nAll tasks completed.")
    print("Generated files:")
    print(f"1. {os.path.join(FIGURES_DIR, 'candidate_models.png')}")
    print(f"2. {os.path.join(FIGURES_DIR, 'error_curves.png')}")
    print(f"3. {os.path.join(FIGURES_DIR, 'variance_demo.png')}")
    print(f"4. {os.path.join(FIGURES_DIR, 'loss_outlier_comparison.png')}")
    print(f"5. {report_path}")


if __name__ == "__main__":
    main()