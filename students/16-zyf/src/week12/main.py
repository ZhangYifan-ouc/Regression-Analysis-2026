import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

from src.utils.metrics import (
    calculate_rmse,
    calculate_mae,
)


RESULTS_DIR = "src/week12/results"
FIGURES_DIR = "src/week12/results/figures"


# =========================
# 准备结果目录
# =========================
def prepare_dirs():
    if os.path.exists(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)

    os.makedirs(FIGURES_DIR, exist_ok=True)


# =========================
# 真实函数
# =========================
def true_function(x):
    return np.sin(x) + 0.3 * x


# =========================
# 生成模拟数据
# =========================
def generate_data(
    n_samples=150,
    random_state=42,
):
    rng = np.random.default_rng(random_state)

    X = rng.uniform(
        -3,
        3,
        size=n_samples,
    )

    noise = rng.normal(
        0,
        0.35,
        size=n_samples,
    )

    y = true_function(X) + noise

    X = X.reshape(-1, 1)

    return train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=random_state,
    )


# =========================
# 构建多项式模型
# =========================
def build_poly_model(degree):
    return Pipeline(
        [
            (
                "poly",
                PolynomialFeatures(
                    degree=degree
                ),
            ),
            (
                "linear",
                LinearRegression(),
            ),
        ]
    )


# =========================
# Task A
# 三个候选模型
# =========================
def run_candidate_models(
    X_train,
    X_test,
    y_train,
    y_test,
):
    degrees = [1, 4, 15]

    x_grid = np.linspace(
        -3,
        3,
        400,
    ).reshape(-1, 1)

    rows = []

    plt.figure(
        figsize=(10, 6)
    )

    plt.scatter(
        X_train,
        y_train,
        label="Train",
        alpha=0.7,
    )

    plt.scatter(
        X_test,
        y_test,
        label="Test",
        alpha=0.7,
    )

    plt.plot(
        x_grid,
        true_function(
            x_grid.ravel()
        ),
        label="True function",
        linewidth=2,
    )

    for degree in degrees:

        model = build_poly_model(
            degree
        )

        model.fit(
            X_train,
            y_train,
        )

        y_train_pred = model.predict(
            X_train
        )

        y_test_pred = model.predict(
            X_test
        )

        train_rmse = calculate_rmse(
            y_train,
            y_train_pred,
        )

        test_rmse = calculate_rmse(
            y_test,
            y_test_pred,
        )

        rows.append(
            {
                "degree": degree,
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
            }
        )

        label = (
            f"degree={degree}, "
            f"train={train_rmse:.3f}, "
            f"test={test_rmse:.3f}"
        )

        plt.plot(
            x_grid,
            model.predict(
                x_grid
            ),
            label=label,
            linewidth=2,
        )

    plt.title(
        "Candidate Models"
    )

    plt.xlabel("x")
    plt.ylabel("y")

    plt.legend()

    plt.tight_layout()

    path = os.path.join(
        FIGURES_DIR,
        "candidate_models.png",
    )

    plt.savefig(
        path,
        dpi=300,
    )

    plt.close()

    return pd.DataFrame(rows)


# =========================
# Task B
# 模型复杂度扫描
# =========================
def run_error_curves(
    X_train,
    X_test,
    y_train,
    y_test,
):
    rows = []

    for degree in range(
        1,
        19,
    ):

        model = build_poly_model(
            degree
        )

        model.fit(
            X_train,
            y_train,
        )

        y_train_pred = model.predict(
            X_train
        )

        y_test_pred = model.predict(
            X_test
        )

        train_rmse = calculate_rmse(
            y_train,
            y_train_pred,
        )

        test_rmse = calculate_rmse(
            y_test,
            y_test_pred,
        )

        rows.append(
            {
                "degree": degree,
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
                "generalization_gap":
                    test_rmse
                    - train_rmse,
            }
        )

    df = pd.DataFrame(
        rows
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        df["degree"],
        df["train_rmse"],
        marker="o",
        label="Train RMSE",
    )

    plt.plot(
        df["degree"],
        df["test_rmse"],
        marker="o",
        label="Test RMSE",
    )

    plt.title(
        "Complexity vs Error"
    )

    plt.xlabel(
        "Polynomial Degree"
    )

    plt.ylabel(
        "RMSE"
    )

    plt.legend()

    plt.tight_layout()

    path = os.path.join(
        FIGURES_DIR,
        "error_curves.png",
    )

    plt.savefig(
        path,
        dpi=300,
    )

    plt.close()

    return df


# =========================
# Task C
# High Variance Demo
# =========================
def run_variance_demo(
    n_repeats=12,
    n_samples=60,
    random_state=42,
):
    rng = np.random.default_rng(
        random_state
    )

    x_grid = np.linspace(
        -3,
        3,
        400,
    ).reshape(-1, 1)

    degrees = [2, 15]

    summary_rows = []

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
        sharey=True,
    )

    for ax, degree in zip(
        axes,
        degrees,
    ):

        predictions = []

        for _ in range(
            n_repeats
        ):

            X = rng.uniform(
                -3,
                3,
                size=n_samples,
            )

            noise = rng.normal(
                0,
                0.35,
                size=n_samples,
            )

            y = (
                true_function(X)
                + noise
            )

            X = X.reshape(
                -1,
                1,
            )

            model = build_poly_model(
                degree
            )

            model.fit(
                X,
                y,
            )

            y_grid_pred = model.predict(
                x_grid
            )

            predictions.append(
                y_grid_pred
            )

            ax.plot(
                x_grid,
                y_grid_pred,
                alpha=0.4,
            )

        predictions = np.array(
            predictions
        )

        pred_std = predictions.std(
            axis=0
        )

        summary_rows.append(
            {
                "degree": degree,
                "mean_prediction_std":
                    float(
                        np.mean(
                            pred_std
                        )
                    ),
                "max_prediction_std":
                    float(
                        np.max(
                            pred_std
                        )
                    ),
            }
        )

        ax.plot(
            x_grid,
            true_function(
                x_grid.ravel()
            ),
            linewidth=3,
            label="True",
        )

        ax.set_title(
            f"degree={degree}"
        )

        ax.legend()

    plt.tight_layout()

    path = os.path.join(
        FIGURES_DIR,
        "variance_demo.png",
    )

    plt.savefig(
        path,
        dpi=300,
    )

    plt.close()

    return pd.DataFrame(
        summary_rows
    )


# =========================
# Task D
# RMSE vs MAE
# =========================
def run_loss_outlier_demo():
    rng = np.random.default_rng(
        42
    )

    y_true = np.linspace(
        50,
        150,
        30,
    )

    y_pred_clean = (
        y_true
        + rng.normal(
            0,
            5,
            size=len(y_true),
        )
    )

    y_pred_outlier = (
        y_pred_clean.copy()
    )

    y_pred_outlier[5] = (
        y_true[5]
        + 120
    )

    clean_rmse = calculate_rmse(
        y_true,
        y_pred_clean,
    )

    clean_mae = calculate_mae(
        y_true,
        y_pred_clean,
    )

    outlier_rmse = calculate_rmse(
        y_true,
        y_pred_outlier,
    )

    outlier_mae = calculate_mae(
        y_true,
        y_pred_outlier,
    )

    df = pd.DataFrame(
        [
            {
                "scenario":
                    "clean",
                "RMSE":
                    clean_rmse,
                "MAE":
                    clean_mae,
            },
            {
                "scenario":
                    "outlier",
                "RMSE":
                    outlier_rmse,
                "MAE":
                    outlier_mae,
            },
        ]
    )

    x = np.arange(
        len(df)
    )

    width = 0.35

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        x - width / 2,
        df["RMSE"],
        width,
        label="RMSE",
    )

    plt.bar(
        x + width / 2,
        df["MAE"],
        width,
        label="MAE",
    )

    plt.xticks(
        x,
        df["scenario"],
    )

    plt.title(
        "RMSE vs MAE"
    )

    plt.legend()

    plt.tight_layout()

    path = os.path.join(
        FIGURES_DIR,
        "loss_outlier_comparison.png",
    )

    plt.savefig(
        path,
        dpi=300,
    )

    plt.close()

    return df
# =========================
# markdown 转换
# =========================
def dataframe_to_markdown(
    df,
    digits=4,
):
    return df.round(
        digits
    ).to_markdown(
        index=False
    )


# =========================
# 写总结报告
# =========================
def write_summary_report(
    candidate_df,
    error_df,
    variance_df,
    loss_df,
):
    best_degree = int(
        error_df.loc[
            error_df[
                "test_rmse"
            ].idxmin(),
            "degree",
        ]
    )

    max_gap_degree = int(
        error_df.loc[
            error_df[
                "generalization_gap"
            ].idxmax(),
            "degree",
        ]
    )

    summary_path = os.path.join(
        RESULTS_DIR,
        "summary.md",
    )

    content = f"""
# Week 12 Bias-Variance Visual Lab Summary

## 1. 本周最重要的三条结论

1. 模型复杂度增加时，训练误差通常下降，但测试误差不一定下降。
2. high variance model 的危险，不是它不会拟合训练集，而是它对训练样本极其敏感。
3. RMSE 比 MAE 更容易被极端错误放大，因此代表不同风险偏好。

---

## 2. Task A：三个候选模型

本实验比较：

- degree=1
- degree=4
- degree=15

结果如下：

{dataframe_to_markdown(candidate_df)}

从 candidate_models.png 可以看到：

- degree=1 更像欠拟合
- degree=15 更像过拟合
- degree=4 往往是更平衡的复杂度

如果需要上线，我会优先选择 degree=4。

---

## 3. Task B：复杂度扫描

完整结果：

{dataframe_to_markdown(error_df)}

测试误差最低复杂度：

degree = {best_degree}

泛化 gap 最大复杂度：

degree = {max_gap_degree}

训练误差持续下降，并不意味着模型越来越好。

因为模型可能正在：

- 记忆训练集
- 学习噪声
- 泛化能力下降

---

## 4. Task C：High Variance 可视化

Repeated sampling 结果：

{dataframe_to_markdown(variance_df)}

high variance model 的危险：

它不是不会拟合训练集。

相反：

它拟合得太好。

但每次训练数据轻微变化，

模型曲线都会剧烈改变。

degree=15 的波动明显比 degree=2 更严重。

---

## 5. Task D：RMSE 与 MAE

异常值实验：

{dataframe_to_markdown(loss_df)}

观察：

- MAE 变化较平稳
- RMSE 会被极端错误明显拉高

原因：

RMSE 使用平方误差：

large error²

因此：

大错惩罚更重。

MAE 则更加稳健。

---

## 6. 最能代表过拟合的图

我认为：

error_curves.png

最能代表 overfitting。

因为：

- train RMSE 持续下降
- test RMSE 先降后升

说明：

模型开始记忆噪声，

而不是学习真实规律。

---

## 7. 与下一周正则化的联系

高复杂度模型容易：

- variance 增大
- 对噪声敏感
- 泛化下降

下一周的：

- Ridge
- Lasso

本质目的就是：

限制模型复杂度，

降低 variance，

提高 generalization。
"""

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            content
        )

    print(
        f"总结报告已保存到: {summary_path}"
    )


# =========================
# 主程序
# =========================
def main():

    print(
        "[Stage0] Preparing..."
    )

    prepare_dirs()

    print(
        "[Stage1] Generating data..."
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = generate_data()

    print(
        "[Stage2] Candidate models..."
    )

    candidate_df = run_candidate_models(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    print(
        "[Stage3] Error curves..."
    )

    error_df = run_error_curves(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    print(
        "[Stage4] Variance demo..."
    )

    variance_df = run_variance_demo()

    print(
        "[Stage5] Loss demo..."
    )

    loss_df = run_loss_outlier_demo()

    print(
        "[Stage6] Writing report..."
    )

    write_summary_report(
        candidate_df,
        error_df,
        variance_df,
        loss_df,
    )

    print(
        "\nWeek12 finished."
    )


# =========================
# 程序入口
# =========================
if __name__ == "__main__":
    main()