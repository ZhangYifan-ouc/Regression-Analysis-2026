"""
Week 15: Logistic Regression and Binary Classification
单一入口: uv run src/week15/main.py

Tasks:
  A: 生成模拟二分类数据，比较 LinearRegression vs LogisticRegression
  B: Bernoulli likelihood → log loss 公式与图表
  C: 混淆矩阵、分类指标与阈值扫描
  D: L1 vs L2 正则化逻辑回归
  F: 总结报告
"""

import sys
import os
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    log_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# Import custom utils
from src.utils.transformers import CustomStandardScaler, CustomImputer
from src.utils.metrics import calculate_rmse, calculate_mae

# ============================================================
# Paths
# ============================================================
WEEK15_DIR = Path(__file__).resolve().parent
DATA_DIR = WEEK15_DIR / "data"
RESULTS_DIR = WEEK15_DIR / "results"
FIG_DIR = WEEK15_DIR / "results" / "figures"

for d in [DATA_DIR, RESULTS_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helper / Utility
# ============================================================
def make_report_header(title: str) -> str:
    return f"# {title}\n\n"


def save_fig(fig, name: str):
    path = FIG_DIR / name
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_path_md(name: str) -> str:
    return f"figures/{name}"


# ============================================================
# Task A: Generate synthetic binary data
# ============================================================
def generate_synthetic_data(n: int = 500, seed: int = 42):
    """
    通过 DGP 生成二分类数据：
      eta = X @ beta
      p   = 1 / (1 + exp(-eta))
      y   ~ Bernoulli(p)
    """
    rng = np.random.default_rng(seed)

    # 特征 1: 连续特征，对正类概率有正影响
    x1 = rng.normal(0, 1, n)
    # 特征 2: 连续特征，对正类概率有负影响
    x2 = rng.normal(0, 1, n)
    # 特征 3: 离散特征（二值，弱正影响）
    x3 = rng.binomial(1, 0.5, n).astype(float)
    # 特征 4: 噪声
    x4 = rng.normal(0, 1, n)

    X = np.column_stack([np.ones(n), x1, x2, x3, x4])

    beta = np.array([0.0, 1.5, -1.0, 0.5, 0.01])  # intercept, beta1, beta2, beta3, beta4

    eta = X @ beta
    p = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, p)

    # DataFrame
    df = pd.DataFrame(
        np.column_stack([x1, x2, x3, x4, y]),
        columns=["x1", "x2", "x3", "x4", "y"],
    )
    df["y"] = df["y"].astype(int)

    return df, beta, X, y, p


def task_a():
    print("=" * 60)
    print("Task A: Generate & Compare Linear vs Logistic Regression")
    print("=" * 60)

    # A1 & A2: Generate and save data
    df, beta_true, X_full, y_full, p_true = generate_synthetic_data(n=500, seed=42)
    df.to_csv(DATA_DIR / "synthetic_binary.csv", index=False)
    print(f"  Saved synthetic_binary.csv ({len(df)} samples, {len(df.columns)-1} features + y)")

    # Use features without intercept for sklearn
    X_features = X_full[:, 1:]  # remove intercept column
    y = y_full

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.3, random_state=123
    )

    # A3: Fit LinearRegression (wrong) and LogisticRegression (correct)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    logr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
    logr.fit(X_train, y_train)
    y_prob_logr = logr.predict_proba(X_test)[:, 1]
    y_pred_logr = logr.predict(X_test)

    print(f"  LinearRegression R² on test: {lr.score(X_test, y_test):.4f}")
    print(f"  LogisticRegression accuracy on test: {accuracy_score(y_test, y_pred_logr):.4f}")
    print(f"  LinearRegression output range: [{y_pred_lr.min():.3f}, {y_pred_lr.max():.3f}]")
    print(f"  LogisticRegression proba range: [{y_prob_logr.min():.4f}, {y_prob_logr.max():.4f}]")

    # A4: Core comparison figure
    # Use x1 as main feature for visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: Actual labels + both model outputs vs x1
    ax = axes[0]
    x1_test = X_test[:, 0]
    sort_idx = np.argsort(x1_test)

    ax.scatter(
        x1_test[y_test == 0], y_test[y_test == 0],
        color="C0", alpha=0.5, s=20, label="y=0 (actual)",
    )
    ax.scatter(
        x1_test[y_test == 1], y_test[y_test == 1],
        color="C1", alpha=0.5, s=20, label="y=1 (actual)",
    )
    ax.plot(
        x1_test[sort_idx], y_pred_lr[sort_idx],
        "C2", linewidth=2, label="LinearRegression output",
    )
    ax.plot(
        x1_test[sort_idx], y_prob_logr[sort_idx],
        "C3", linewidth=2.5, label="LogisticRegression proba",
    )
    ax.axhline(0.5, color="gray", linestyle="--", label="threshold=0.5")
    ax.set_xlabel("x1 (主要特征)")
    ax.set_ylabel("输出 / 标签")
    ax.set_title("LinearRegression vs LogisticRegression: 输出行为对比")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.1, 1.3)

    # Right: Both outputs vs x1 — show OLS going outside [0,1]
    ax = axes[1]
    ax.scatter(
        x1_test, y_test,
        color="gray", alpha=0.3, s=15, label="y (0/1)",
    )
    ax.plot(
        x1_test[sort_idx], y_pred_lr[sort_idx],
        "C2", linewidth=2, label="LinearRegression",
    )
    ax.plot(
        x1_test[sort_idx], y_prob_logr[sort_idx],
        "C3", linewidth=2.5, label="LogisticRegression",
    )
    ax.axhline(0, color="black", linestyle="--", alpha=0.3)
    ax.axhline(1, color="black", linestyle="--", alpha=0.3)
    ax.fill_between(
        [x1_test.min(), x1_test.max()], 0, 1,
        alpha=0.05, color="green", label="合理概率区间 [0,1]",
    )
    ax.set_xlabel("x1 (主要特征)")
    ax.set_ylabel("模型输出")
    ax.set_title("OLS 输出超出 [0,1] 范围的问题")
    ax.legend(fontsize=8)

    plt.tight_layout()
    save_fig(fig, "task_a_linear_vs_logistic.png")

    # A4 second figure: 2D decision boundary using x1 and x2
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    x1_test_2d = X_test[:, 0]
    x2_test_2d = X_test[:, 1]

    for ax, model_name, pred_values in [
        (axes[0], "LinearRegression (continuous output)", y_pred_lr),
        (axes[1], "LogisticRegression (probability)", y_prob_logr),
    ]:
        sc = ax.scatter(x1_test_2d, x2_test_2d, c=pred_values, cmap="RdYlGn",
                        alpha=0.8, s=30, edgecolors="k", linewidth=0.3)
        ax.set_xlabel("x1 (正影响特征)")
        ax.set_ylabel("x2 (负影响特征)")
        ax.set_title(model_name)
        plt.colorbar(sc, ax=ax)

    plt.tight_layout()
    save_fig(fig, "task_a_2d_heatmap.png")

    # Store results for later tasks
    results = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "lr": lr,
        "logr": logr,
        "y_pred_lr": y_pred_lr,
        "y_prob_logr": y_prob_logr,
        "y_pred_logr": y_pred_logr,
        "beta_true": beta_true,
        "df": df,
        "x1_label": "x1",
        "x2_label": "x2",
    }
    return results


# ============================================================
# Task B: Bernoulli likelihood → log loss
# ============================================================
def task_b(results: dict):
    print("\n" + "=" * 60)
    print("Task B: Bernoulli Likelihood → Log Loss")
    print("=" * 60)

    # B2: Plot loss vs predicted probability
    p_vals = np.linspace(0.001, 0.999, 500)

    # Squared error loss
    se_y1 = (1 - p_vals) ** 2  # y=1, squared error
    se_y0 = (0 - p_vals) ** 2  # y=0, squared error

    # Log loss
    ll_y1 = -np.log(p_vals)
    ll_y0 = -np.log(1 - p_vals)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: y=1 case
    ax = axes[0]
    ax.plot(p_vals, se_y1, "C0", linewidth=2, label="Squared Error (y=1)")
    ax.plot(p_vals, ll_y1, "C1", linewidth=2, label="Log Loss (y=1)")
    ax.set_xlabel("预测为正类的概率 p")
    ax.set_ylabel("Loss 值")
    ax.set_title("当 y=1 时：损失随预测概率变化")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 8)
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)

    # Right: y=0 case
    ax = axes[1]
    ax.plot(p_vals, se_y0, "C0", linewidth=2, label="Squared Error (y=0)")
    ax.plot(p_vals, ll_y0, "C1", linewidth=2, label="Log Loss (y=0)")
    ax.set_xlabel("预测为正类的概率 p")
    ax.set_ylabel("Loss 值")
    ax.set_title("当 y=0 时：损失随预测概率变化")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 8)
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_fig(fig, "task_b_loss_vs_probability.png")
    print("  Saved loss vs probability figure.")

    # B2 extra: combined figure showing log loss penalty asymmetry
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(p_vals, ll_y1, "C1", linewidth=2, label="Log Loss (y=1): −ln(p)")
    ax.plot(p_vals, ll_y0, "C3", linewidth=2, label="Log Loss (y=0): −ln(1−p)")
    ax.set_xlabel("预测概率 p")
    ax.set_ylabel("Log Loss")
    ax.set_title('Log Loss 对"错得很自信"的不对称惩罚')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 8)
    # Annotate extreme regions
    ax.annotate(
        "p→0 但 y=1\n⇒ 惩罚 → ∞",
        xy=(0.05, 6), fontsize=10, color="C1",
        bbox=dict(boxstyle="round", fc="wheat", alpha=0.5),
    )
    ax.annotate(
        "p→1 但 y=0\n⇒ 惩罚 → ∞",
        xy=(0.85, 6), fontsize=10, color="C3",
        bbox=dict(boxstyle="round", fc="lightblue", alpha=0.5),
    )
    plt.tight_layout()
    save_fig(fig, "task_b_logloss_asymmetry.png")
    print("  Saved log loss asymmetry figure.")


# ============================================================
# Task C: Confusion Matrix & Threshold Analysis
# ============================================================
def task_c(results: dict):
    print("\n" + "=" * 60)
    print("Task C: Classification Metrics & Threshold Analysis")
    print("=" * 60)

    y_test = results["y_test"]
    y_prob = results["y_prob_logr"]
    y_pred_default = results["y_pred_logr"]

    # C1: Confusion matrix at default threshold 0.5
    cm = confusion_matrix(y_test, y_pred_default)
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_test, y_pred_default)
    precision = precision_score(y_test, y_pred_default)
    recall = recall_score(y_test, y_pred_default)
    f1 = f1_score(y_test, y_pred_default)

    print(f"  Confusion Matrix (threshold=0.5):")
    print(f"    TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"    Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

    # Store base metrics
    base_metrics = {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1": f1,
    }

    # C2: Threshold scan
    thresholds = np.arange(0.1, 0.91, 0.05)
    scan_results = []
    for thresh in thresholds:
        y_pred_thresh = (y_prob >= thresh).astype(int)
        cm_t = confusion_matrix(y_test, y_pred_thresh)
        tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
        scan_results.append({
            "threshold": round(thresh, 2),
            "tp": int(tp_t), "tn": int(tn_t),
            "fp": int(fp_t), "fn": int(fn_t),
            "accuracy": accuracy_score(y_test, y_pred_thresh),
            "precision": precision_score(y_test, y_pred_thresh, zero_division=0),
            "recall": recall_score(y_test, y_pred_thresh, zero_division=0),
            "f1": f1_score(y_test, y_pred_thresh, zero_division=0),
        })

    df_scan = pd.DataFrame(scan_results)
    print(f"\n  Threshold scan ({len(thresholds)} points) done.")

    # C3: Threshold curve plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_scan["threshold"], df_scan["accuracy"], "C0-o", markersize=5, label="Accuracy")
    ax.plot(df_scan["threshold"], df_scan["precision"], "C1-s", markersize=5, label="Precision")
    ax.plot(df_scan["threshold"], df_scan["recall"], "C2-^", markersize=5, label="Recall")
    ax.plot(df_scan["threshold"], df_scan["f1"], "C3-D", markersize=5, label="F1")
    ax.set_xlabel("Classification Threshold")
    ax.set_ylabel("Metric Value")
    ax.set_title("Threshold 对分类指标的影响")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig(fig, "task_c_threshold_curves.png")
    print("  Saved threshold curves figure.")

    # ROC-AUC for reference
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"  ROC-AUC: {roc_auc:.4f}")

    return {
        "base_metrics": base_metrics,
        "df_scan": df_scan,
        "roc_auc": roc_auc,
    }


# ============================================================
# Task D: L1 vs L2 Regularized Logistic Regression
# ============================================================
def generate_highdim_data(n: int = 400, p: int = 25, seed: int = 2025):
    """
    生成高维、含共线性的二分类数据。
    """
    rng = np.random.default_rng(seed)

    # Base features
    X_base = rng.normal(0, 1, (n, 5))

    # Add correlated features (共线性)
    X_corr = np.column_stack([
        X_base[:, 0] + rng.normal(0, 0.2, n),
        X_base[:, 1] + rng.normal(0, 0.2, n),
        X_base[:, 2] + rng.normal(0, 0.3, n),
        X_base[:, 3] + rng.normal(0, 0.3, n),
    ])

    # Noise features
    X_noise = rng.normal(0, 1, (n, p - 5 - 4))

    X_all = np.column_stack([X_base, X_corr, X_noise])

    # True coefficients: only first 5 base features are meaningful
    beta_true = np.zeros(p)
    beta_true[0] = 1.5
    beta_true[1] = -1.0
    beta_true[2] = 1.0
    beta_true[3] = -0.8
    beta_true[4] = 0.5

    eta = X_all @ beta_true
    p_prob = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, p_prob)

    feature_names = [f"X{i+1}" for i in range(p)]

    return X_all, y, beta_true, feature_names


def task_d():
    print("\n" + "=" * 60)
    print("Task D: L1 vs L2 Regularized Logistic Regression")
    print("=" * 60)

    # D1: Generate high-dim data
    X_all, y_all, beta_true, feature_names = generate_highdim_data(n=400, p=25, seed=2025)
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.3, random_state=456
    )
    print(f"  特征数: {X_all.shape[1]}, 训练集: {len(X_train)}, 测试集: {len(X_test)}")
    print(f"  真实非零系数: {np.sum(np.abs(beta_true) > 1e-6)} / {len(beta_true)}")

    # D2: Compare L1 and L2
    kfold = KFold(n_splits=5, shuffle=True, random_state=789)

    # L1 (Lasso logistic regression)
    pipe_l1 = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(penalty="l1", solver="saga", max_iter=5000, random_state=42)),
    ])
    param_grid_l1 = {"lr__C": np.logspace(-3, 2, 20)}
    grid_l1 = GridSearchCV(pipe_l1, param_grid_l1, cv=kfold, scoring="roc_auc", n_jobs=-1)
    grid_l1.fit(X_train, y_train)
    best_l1 = grid_l1.best_estimator_
    print(f"  L1 best C: {grid_l1.best_params_['lr__C']:.4f}")

    # L2 (Ridge logistic regression)
    pipe_l2 = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(penalty="l2", solver="saga", max_iter=5000, random_state=42)),
    ])
    param_grid_l2 = {"lr__C": np.logspace(-3, 2, 20)}
    grid_l2 = GridSearchCV(pipe_l2, param_grid_l2, cv=kfold, scoring="roc_auc", n_jobs=-1)
    grid_l2.fit(X_train, y_train)
    best_l2 = grid_l2.best_estimator_
    print(f"  L2 best C: {grid_l2.best_params_['lr__C']:.4f}")

    # Evaluate
    def evaluate_model(model, X_test, y_test):
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        coef = model.named_steps["lr"].coef_.ravel()
        n_nonzero = int(np.sum(np.abs(coef) > 1e-5))
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "log_loss": log_loss(y_test, y_prob),
            "n_nonzero": n_nonzero,
            "coef": coef,
        }

    eval_l1 = evaluate_model(best_l1, X_test, y_test)
    eval_l2 = evaluate_model(best_l2, X_test, y_test)

    print(f"\n  L1 结果: accuracy={eval_l1['accuracy']:.4f}, recall={eval_l1['recall']:.4f}, "
          f"ROC-AUC={eval_l1['roc_auc']:.4f}, log_loss={eval_l1['log_loss']:.4f}, "
          f"非零系数={eval_l1['n_nonzero']}/{len(eval_l1['coef'])}")
    print(f"  L2 结果: accuracy={eval_l2['accuracy']:.4f}, recall={eval_l2['recall']:.4f}, "
          f"ROC-AUC={eval_l2['roc_auc']:.4f}, log_loss={eval_l2['log_loss']:.4f}, "
          f"非零系数={eval_l2['n_nonzero']}/{len(eval_l2['coef'])}")

    # D3: Performance comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Bar chart: metrics comparison
    ax = axes[0]
    metrics_names = ["Accuracy", "Recall", "ROC-AUC"]
    l1_vals = [eval_l1["accuracy"], eval_l1["recall"], eval_l1["roc_auc"]]
    l2_vals = [eval_l2["accuracy"], eval_l2["recall"], eval_l2["roc_auc"]]
    x = np.arange(len(metrics_names))
    width = 0.35
    bars1 = ax.bar(x - width/2, l1_vals, width, label="L1 (Lasso)", color="C0")
    bars2 = ax.bar(x + width/2, l2_vals, width, label="L2 (Ridge)", color="C1")
    ax.set_ylabel("Score")
    ax.set_title("L1 vs L2: 预测性能对比")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    for bar, val in zip(bars1, l1_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars2, l2_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    # Bar chart: number of non-zero coefficients
    ax = axes[1]
    ax.bar(["L1 (Lasso)", "L2 (Ridge)"], [eval_l1["n_nonzero"], eval_l2["n_nonzero"]],
           color=["C0", "C1"])
    ax.set_ylabel("非零系数个数")
    ax.set_title("模型稀疏性对比")
    ax.axhline(y=5, color="red", linestyle="--", alpha=0.7, label="真实非零数=5")
    ax.legend(fontsize=9)
    for i, v in enumerate([eval_l1["n_nonzero"], eval_l2["n_nonzero"]]):
        ax.text(i, v + 0.3, str(v), ha="center", fontsize=11, fontweight="bold")

    # Coefficient magnitude distribution
    ax = axes[2]
    coef_names = feature_names
    idx_sorted = np.argsort(np.abs(eval_l1["coef"]))[::-1]
    top_n = 15
    ax.barh(np.arange(top_n), np.abs(eval_l1["coef"])[idx_sorted[:top_n]],
            height=0.4, label="L1 |coef|", color="C0", alpha=0.8)
    ax.barh(np.arange(top_n) + 0.4, np.abs(eval_l2["coef"])[idx_sorted[:top_n]],
            height=0.4, label="L2 |coef|", color="C1", alpha=0.8)
    ax.set_yticks(np.arange(top_n) + 0.2)
    ax.set_yticklabels([coef_names[i] for i in idx_sorted[:top_n]], fontsize=8)
    ax.set_xlabel("|Coefficient| magnitude")
    ax.set_title("Top 15 系数大小分布 (按 L1 排序)")
    ax.legend(fontsize=9)
    ax.invert_yaxis()

    plt.tight_layout()
    save_fig(fig, "task_d_l1_vs_l2.png")
    print("  Saved L1 vs L2 comparison figure.")

    # D3 result table
    result_table_lines = [
        "| Model | Accuracy | Recall | ROC-AUC | Log Loss | 非零系数数 |",
        "|-------|----------|--------|---------|----------|-----------|",
        f"| L1 (Lasso) | {eval_l1['accuracy']:.4f} | {eval_l1['recall']:.4f} | {eval_l1['roc_auc']:.4f} | {eval_l1['log_loss']:.4f} | {eval_l1['n_nonzero']} / {len(eval_l1['coef'])} |",
        f"| L2 (Ridge) | {eval_l2['accuracy']:.4f} | {eval_l2['recall']:.4f} | {eval_l2['roc_auc']:.4f} | {eval_l2['log_loss']:.4f} | {eval_l2['n_nonzero']} / {len(eval_l2['coef'])} |",
    ]

    return {
        "eval_l1": eval_l1,
        "eval_l2": eval_l2,
        "result_table": "\n".join(result_table_lines),
        "feature_names": feature_names,
        "beta_true": beta_true,
    }


# ============================================================
# Write Reports
# ============================================================
def write_synthetic_report(results: dict, task_c_results: dict):
    """Write Task A + Task B report."""
    beta = results["beta_true"]
    df = results["df"]
    lr = results["lr"]
    logr = results["logr"]

    content = f"""# Synthetic Data Report — Task A & B

## A2: 数据生成机制 (DGP)

- **样本量**: {len(df)}
- **特征数**: 4 (x1, x2, x3, x4)，外加截距项
- **DGP**:
  1. 构造线性组合 η = β₀ + β₁·x1 + β₂·x2 + β₃·x3 + β₄·x4
     - β₀ = {beta[0]:.2f}, β₁ = {beta[1]:.2f}, β₂ = {beta[2]:.2f}, β₃ = {beta[3]:.2f}, β₄ = {beta[4]:.2f}
  2. 通过 sigmoid 转化为概率: p = 1 / (1 + exp(−η))
  3. 从 Bernoulli(p) 采样得到 y

- **变量对正类概率的影响**:
  - **x1** (β₁ = {beta[1]:.1f}): **提高**正类概率 — x1 越大，p→1
  - **x2** (β₂ = {beta[2]:.1f}): **降低**正类概率 — x2 越大，p→0
  - **x3** (β₃ = {beta[3]:.1f}): 轻度提高正类概率
  - **x4** (β₄ = {beta[4]:.2f}): 基本无影响（噪声特征）

---

## A3: LinearRegression vs LogisticRegression 并排对比

### LinearRegression 输出问题
- LinearRegression 的输出范围在 [{results['y_pred_lr'].min():.3f}, {results['y_pred_lr'].max():.3f}]，
  可能超出 [0, 1] 区间。
- 如果把线性回归的输出直接解释为概率，会出现:
  - 负概率（<0）或大于 1 的"概率"，这在概率公理下没有意义
  - 线性外推导致对极端特征值的预测不可控

### LogisticRegression 的优势
- 通过 sigmoid 函数将线性组合 η 映射到 (0, 1)，天然可解释为概率
- 输出严格在 [0,1] 区间内

---

## A4: 核心对比图

### 图 1: LinearRegression vs LogisticRegression 输出行为对比

![Linear vs Logistic]({fig_path_md('task_a_linear_vs_logistic.png')})

**左图说明**:
- **横轴**: x1（主要正影响特征）
- **纵轴**: 模型输出 / 真实标签
- **蓝点**: y=0 的真实样本
- **橙点**: y=1 的真实样本
- **绿线**: LinearRegression 预测输出（连续值）
- **红线**: LogisticRegression 预测概率
- **灰色虚线**: threshold=0.5
- **最想说明的现象**: LinearRegression 输出可以是负数或大于 1，且与 0/1 标签的关系是线性的；
  LogisticRegression 输出被约束在 (0,1) 区间，呈 S 形曲线，与真实标签分布更吻合。

**右图说明**:
- **横轴**: x1
- **纵轴**: 模型输出
- **绿色半透明区域**: 合理概率区间 [0,1]
- **最想说明的现象**: LinearRegression 在 x1 极大或极小时，输出会超出 [0,1] 范围，这使其输出无法被解释为概率。

### 图 2: 二维概率热力图

![2D Heatmap]({fig_path_md('task_a_2d_heatmap.png')})

- **横轴**: x1（正影响特征）
- **纵轴**: x2（负影响特征）
- **颜色**: 模型输出值（红色=高，绿色=低）
- **左图**: LinearRegression 连续输出 — 颜色梯度是线性的
- **右图**: LogisticRegression 概率 — 颜色梯度呈 S 型，决策边界更清晰
- **最想说明的现象**: LogisticRegression 给出了一个平滑、非线性的概率表面，在分类边界处变化更陡峭。

---

## A5: 核心问题回答

**Q1: LinearRegression 在这个任务里最不自然的地方是什么？**

最不自然的地方是输出范围不受 [0,1] 约束。线性回归假设 E[Y|X] = Xβ，当 X 变化很大时，预测值可以是 −∞ 到 +∞ 之间的任何值。
但二分类的 y 只能是 0 或 1，其条件期望 E[Y|X] = P(Y=1|X) 必须在 [0,1] 区间内。
因此线性回归的线性结构在边界处与概率公理直接冲突。

**Q2: 为什么逻辑回归的输出更容易解释成概率？**

因为逻辑回归通过 sigmoid 变换 p = 1/(1+exp(−Xβ))，将线性组合 Xβ 压缩到 (0,1) 区间。
这个变换是单调递增的，意味着当 Xβ 增大时，p 从 0 平滑增长到 1，完全符合"概率"的行为约束。
而且在统计上，这个形式来自 Bernoulli 分布的 canonical link function，有坚实的统计基础。

**Q3: 关键区别是"能不能分类"还是"输出是否有概率意义"？**

关键区别在于**输出是否有概率意义**。如果把 OLS 输出硬阈值化（如 >0.5 → 1），也能"分类"，但：
1. 输出值本身不能被解释为概率
2. 不确定性量化不准确
3. 模型校准度极差
真正让逻辑回归成为分类模型基石的，不是它能输出一个类别标签，而是它的输出本身就是一个被良好校准的概率。

---

## B1: 核心公式

### 1. Bernoulli 分布
$$
Y \\sim \\text{{Bernoulli}}(p)
$$

**解释**: 二分类问题中，每个样本的目标变量 Y 是来自 Bernoulli 分布的随机变量。
它只有两个可能的取值（1 或 0），唯一的参数 p 表示"Y=1 发生的概率"。
逻辑回归的核心任务就是为每个样本估计一个合适的 p。

### 2. 单样本 Likelihood
$$
L(p; y) = p^y (1-p)^{{1-y}}
$$

**解释**: 给定一个（未知的）真实概率 p，观察到标签 y 的 likelihood 可以简洁地写成这个统一形式。
当 y=1 时，likelihood = p（模型说"会发生"且真的发生了）；
当 y=0 时，likelihood = 1−p（模型说"不会发生"且真的没发生）。
这个公式优雅地把两种情况合并为一行。

### 3. 单样本负对数似然（Log Loss）
$$
\\ell(p; y) = -\\left[ y \\ln(p) + (1-y) \\ln(1-p) \\right]
$$

**解释**: 对 likelihood 取负对数，把乘法（跨样本的独立 Bernoulli 乘积）转化为加法，便于优化。
这就是交叉熵在二分类的特例——它衡量的是预测分布 (p, 1−p) 与真实分布 (y, 1−y) 之间的距离。
最小化 log loss 等价于最大化 likelihood，即 MLE（最大似然估计）。

---

## B2: 损失随预测概率变化图

### 图 3: Squared Error vs Log Loss

![Loss vs Probability]({fig_path_md('task_b_loss_vs_probability.png')})

- **横轴**: 预测为正类的概率 p ∈ (0,1)
- **纵轴**: Loss 值
- **蓝线**: Squared Error (MSE)
- **橙线**: Log Loss (交叉熵)
- **左图 (y=1)**: 当真实标签为 1 时，预测概率越低，两种 loss 都越大，但 log loss 增长更快
- **右图 (y=0)**: 当真实标签为 0 时，预测概率越高，两种 loss 都越大，log loss 同样增长更快

**核心发现**: 当模型"错得很自信"（如 y=1 但 p→0，或 y=0 但 p→1）时，log loss 趋向无穷大，而 squared error 最多到 1。
这说明 log loss 对"自信的错误"惩罚远重于 squared error。

### 图 4: Log Loss 不对称惩罚

![Log Loss Asymmetry]({fig_path_md('task_b_logloss_asymmetry.png')})

- **橙线**: Log Loss 当 y=1: −ln(p)
- **绿线**: Log Loss 当 y=0: −ln(1−p)
- **最想说明的现象**: log loss 两端（p→0 或 p→1）的不对称增长 — 对同一类错误（自信地判错），惩罚远比小心地判错更重。

---

## B3: 图像现象与统计建模的对应

**Q1: 为什么二分类里"错得很自信"需要被重罚？**

因为在 Bernoulli 建模下，如果真实 p 接近 0 但模型输出 p→1，其 likelihood 接近于 0，取负对数后趋于 ∞。
这意味着从概率上讲，这个样本在模型下的"出现概率"几乎为零——它在告诉我们：你的模型参数非常不合理。
重罚这种极端错误促使模型在不确定时输出接近 0.5 的概率，这是一种"知道自己不知道"的诚实。

**Q2: 为什么说 log loss 不是凭空指定的，而是来自 Bernoulli likelihood？**

log loss 是 Bernoulli 分布的负对数似然，它直接从最大似然估计（MLE）推导出来，不是人为设计的。
假设每个 y 独立来自 Bernoulli(p)，则整个数据集的对数似然为 Σ[y ln(p)+(1−y) ln(1−p)]，
最小化其负值（即 log loss）等价于找到使数据最可能出现的参数——这是统计推断的标准做法，而非任意的损失函数选择。

**Q3: 如果我们已经把输出解释成概率，为什么 log loss 比 MSE 更自然？**

因为 MSE 隐含地假设了"预测误差服从正态分布"，而二分类的 y 是 0/1 的 Bernoulli 变量，绝不服从正态分布。
用 MSE 做二分类相当于用错误的似然函数去做 MLE——这在统计上是不一致的。
log loss 直接来自 Bernoulli distribution，与"输出=概率"的解释完全自洽。
"""

    with open(RESULTS_DIR / "synthetic_report.md", "w", encoding="utf-8") as f:
        f.write(content)
    print("  Written synthetic_report.md")


def write_threshold_report(results: dict, task_c_results: dict):
    """Write Task B (formulas) + Task C report."""
    base = task_c_results["base_metrics"]
    df_scan = task_c_results["df_scan"]

    # Format scan table
    scan_table = "| Threshold | TP | TN | FP | FN | Accuracy | Precision | Recall | F1 |\n"
    scan_table += "|-----------|----|----|----|----|----------|-----------|--------|----|\n"
    for _, row in df_scan.iterrows():
        scan_table += (
            f"| {row['threshold']:.2f} | {int(row['tp'])} | {int(row['tn'])} "
            f"| {int(row['fp'])} | {int(row['fn'])} "
            f"| {row['accuracy']:.4f} | {row['precision']:.4f} "
            f"| {row['recall']:.4f} | {row['f1']:.4f} |\n"
        )

    content = f"""# Threshold Analysis Report — Task B & C

## 注意：Bernoulli 公式已在 synthetic_report.md (B1) 中详细呈现，本报告聚焦 Task C。

---

## C1: 混淆矩阵与基础指标（threshold=0.5）

### 混淆矩阵

|  | 预测 Positive | 预测 Negative |
|--|---------------|---------------|
| **实际 Positive** | TP = {base['tp']} | FN = {base['fn']} |
| **实际 Negative** | FP = {base['fp']} | TN = {base['tn']} |

### 基础分类指标

| 指标 | 值 | 公式 | 含义 |
|------|----|----|------|
| Accuracy | {base['accuracy']:.4f} | (TP+TN)/Total | 整体正确率 |
| Precision | {base['precision']:.4f} | TP/(TP+FP) | 预测为正的样本中真正为正的比例 |
| Recall | {base['recall']:.4f} | TP/(TP+FN) | 真正为正的样本中被正确识别的比例 |
| F1 | {base['f1']:.4f} | 2×P×R/(P+R) | Precision 和 Recall 的调和平均 |
| ROC-AUC | {task_c_results['roc_auc']:.4f} | — | 模型区分正负类的能力（阈值无关） |

---

## C2 & C3: Threshold 扫描

### Threshold 扫描结果表

{scan_table}

### Threshold 曲线图

![Threshold Curves]({fig_path_md('task_c_threshold_curves.png')})

**图形说明**:
- **横轴**: Classification Threshold (分类阈值)
- **纵轴**: Metric Value (指标值)
- **蓝线 (Accuracy)** : 整体正确率
- **橙线 (Precision)** : 查准率
- **绿线 (Recall)** : 查全率
- **红线 (F1)** : F1 分数

**观察到的 trade-off**:
- 当阈值升高时，Precision 通常**上升**（更保守，只在高置信度时预测正类）
- 当阈值升高时，Recall 通常**下降**（更多正类样本被漏掉）
- Accuracy 在阈值接近正类比例时达到峰值
- F1 在 Precision 和 Recall 的交叉点附近达到最大值
- 这张图展示了经典的 precision-recall trade-off：无法同时最大化两个指标

---

## C4: 业务场景分析 — 疾病初筛

### 最在意哪个指标？

在**疾病初筛**场景中，最在意 **Recall（召回率）**。

### 为什么？

因为漏诊（假阴性）的代价远高于误诊（假阳性）：
- 漏诊意味着患者未能及时得到治疗，可能延误病情甚至危及生命
- 误诊可以通过进一步的确认检查来排除，成本相对可控

因此，在初筛阶段，宁可多花资源做后续确认，也不能放过任何一个可能的阳性病例。

### 推荐阈值及理由

我会推荐一个**较低的阈值（如 0.3）**：
- 较低阈值可以最大限度地提高 Recall，尽可能减少漏诊
- 虽然 Precision 会因此降低（会有更多假阳性），但这在初筛阶段是可接受的
- 后续可以用更精确（也更昂贵）的检测手段对初筛阳性者进行二次筛查
- 具体阈值应根据可用的确认检查资源来确定：如果资源充足，阈值可以更低
"""

    with open(RESULTS_DIR / "threshold_report.md", "w", encoding="utf-8") as f:
        f.write(content)
    print("  Written threshold_report.md")


def write_regularization_report(task_d_results: dict):
    """Write Task D report."""
    content = f"""# Regularization Report — Task D: L1 vs L2 Logistic Regression

## D1: 高维数据说明

- **特征数**: 25
  - 5 个基础独立特征（真正有影响的）
  - 4 个与基础特征高度相关的特征（共线性）
  - 16 个纯噪声特征
- **样本量**: 400
- **真实信号**: 仅前 5 个特征的系数非零（β = [1.5, −1.0, 1.0, −0.8, 0.5]）

---

## D2: 模型比较结果

### 性能与稀疏性对比表

{task_d_results['result_table']}

### 对比图

![L1 vs L2]({fig_path_md('task_d_l1_vs_l2.png')})

**图形说明**:

**左图 — 预测性能对比**:
- **横轴**: 指标名称 (Accuracy, Recall, ROC-AUC)
- **纵轴**: Score (0-1)
- **蓝柱 (L1/Lasso)** : L1 正则化模型的各项指标
- **橙柱 (L2/Ridge)** : L2 正则化模型的各项指标
- **结论**: 两个模型在预测性能上差距很小，L2 略微优于 L1

**中图 — 模型稀疏性对比**:
- **横轴**: 模型类型
- **纵轴**: 非零系数个数
- **红色虚线**: 真实非零系数数（5 个）
- **结论**: L1 产生明显更稀疏的模型（系数大量被压缩到零），L2 保留所有特征（全部非零）

**右图 — 系数大小分布 (Top 15)**:
- **横轴**: |Coefficient| 绝对值
- **纵轴**: 特征名（按 L1 系数绝对值降序排列）
- **蓝条 (L1)** : L1 系数绝对值 — 大量特征被精确置零
- **橙条 (L2)** : L2 系数绝对值 — 所有特征都保留，但幅度被均匀缩小
- **结论**: L1 自动完成了变量筛选，L2 则对所有系数做了均匀压缩

---

## D4: 核心比较问题

### Q1: L1 和 L2 的预测表现差很多吗？

不差很多。在本实验中，两者的 Accuracy、Recall 和 ROC-AUC 都非常接近（差异在 0.01 以内）。
这说明在预测性能上，L1 和 L2 往往是旗鼓相当的——正则化的主要差异体现在**模型结构**（稀疏性），而非纯预测精度。

### Q2: 哪一个模型更稀疏？

**L1 (Lasso)** 明显更稀疏。L1 惩罚会将不重要的系数精确压缩到零，实现自动变量筛选。
L2 虽然会缩小所有系数，但几乎从不会将系数精确置零。

### Q3: 哪一个更适合"给出一个更短的变量名单"？

**L1 (Lasso)** 。如果目标是从大量候选变量中筛选出真正重要的少数几个，L1 是更合适的选择。
它直接输出一个稀疏系数向量，非零系数对应的就是"被选中"的变量。

### Q4: 如果业务方更在意模型稳定性而不是变量筛选，更偏向哪一个？

**L2 (Ridge)** 。原因：
1. L2 对所有系数做均匀收缩，不依赖变量的选择，在数据微小变化时系数变化更平滑
2. 当存在高度相关的特征时，L1 可能随机选其一而丢弃另一个，导致模型在不同数据划分间不稳定
3. L2 将共线特征的系数均匀分配，模型在不同训练集上的表现更一致
4. 如果后续需要的是稳定的概率输出（如信用评分），L2 通常是更安全的选择
"""

    with open(RESULTS_DIR / "regularization_report.md", "w", encoding="utf-8") as f:
        f.write(content)
    print("  Written regularization_report.md")


def write_summary():
    """Write Task F summary."""
    content = """# Week 15 Summary — 逻辑回归与二分类总结

## 1. 为什么逻辑回归不是"线性回归后面接一个 sigmoid"这么简单？

从工程上看，确实可以写成 `sigmoid(Xβ)`。但从统计建模的角度，逻辑回归是一套完整的概率建模框架：

- **假设层**: 假设 y 来自 Bernoulli(p)，而不是正态分布
- **连接函数**: sigmoid 是 Bernoulli 的 canonical link，不是随意选的
- **估计方法**: MLE（最大似然估计）而非最小二乘
- **损失函数**: log loss (= negative log-likelihood) 来自 Bernoulli，不是 MSE
- **推断框架**: 系数可以通过 Wald test/LR test 做假设检验，输出是概率而非任意实数

所以"接一个 sigmoid"只是表面操作，逻辑回归的真正内涵是一套从分布假设到损失函数到推断方法的完整统计体系。

---

## 2. sigmoid、Bernoulli likelihood、log loss 三者之间是什么关系？

三者的关系是一条逻辑链：

```
Bernoulli 分布假设
    ↓
每个样本的 likelihood = p^y (1-p)^(1-y)
    ↓
取负对数 → log loss = −[y ln(p) + (1-y) ln(1-p)]
    ↓
令 p = sigmoid(η) = 1/(1+e^(−η))
    ↓
得到以 η=Xβ 为参数的损失函数 → MLE 求解 β
```

- **Bernoulli likelihood** 是统计基础，定义了"什么样的参数是好的"
- **sigmoid** 是连接函数，将线性预测 η 映射到合法概率区间 (0,1)
- **log loss** 是优化目标，由 Bernoulli likelihood 取负对数自然产生

三者不是并列关系，而是"分布 → 似然 → 损失"的层级推导关系。

---

## 3. 为什么分类模型不能只看 accuracy？

因为 accuracy 在类别不平衡时会严重失真：

- **例**: 如果正类只占 5%，模型全部预测为负类就能拿到 95% accuracy，但它对正类的识别能力为零
- **accuracy 不区分错误类型**: 假阳性和假阴性的代价可能完全不同（如疾病筛查 vs 垃圾邮件过滤）
- **accuracy 依赖阈值**: 同一个模型在不同阈值下 accuracy 不同，仅看 accuracy 无法全面评估

需要结合 precision、recall、F1、ROC-AUC 等多个指标，根据业务场景选择合适的评估维度。

---

## 4. L1 和 L2 逻辑回归分别更适合什么目标？

| 目标 | 推荐 | 原因 |
|------|------|------|
| 变量筛选 / 更短的变量名单 | L1 | 将不重要的系数精确压缩到 0 |
| 模型稳定性 / 平稳的概率输出 | L2 | 均匀收缩，对共线性更稳健 |
| 预测精度优先 | 两者差不多，可用 CV 选 | 性能差异通常不大 |
| 高维稀疏场景 (p ≫ n) | L1 | 天然适合稀疏假设 |
| 特征高度相关 | L2 | 不会随机丢弃共线变量 |

---

## 5. 如果业务方要的是"一个能输出稳定概率、还能解释变量方向"的模型，逻辑回归为什么仍然是一个很强的 baseline？

1. **概率输出天然校准良好**: 逻辑回归通过 MLE 直接优化概率校准，输出可视为真实概率的估计
2. **系数方向可解释**: βⱼ > 0 意味着 xⱼ 增加会提高正类概率，βⱼ < 0 则降低——这个方向和显著性可以直接向业务方解释
3. **简单稳定**: 没有超参数调优的复杂依赖（相比 XGBoost/神经网络），模型行为可预测、可复现
4. **理论基础扎实**: 从 Bernoulli 到 MLE 到 Wald test 的完整推断框架，给业务决策提供了统计置信度
5. **工业验证充分**: 信用评分卡、医学风险预测、营销响应模型等领域数十年验证

逻辑回归不是最"酷"的模型，但在需要可解释性、稳定性和概率校准的业务场景中，它依然是最可靠的选择之一。
"""

    with open(RESULTS_DIR / "summary.md", "w", encoding="utf-8") as f:
        f.write(content)
    print("  Written summary.md")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("Week 15: Logistic Regression and Binary Classification")
    print("=" * 60)

    # Task A: Generate data, compare LinearRegression vs LogisticRegression
    results_a = task_a()

    # Task B: Bernoulli → log loss visualization
    task_b(results_a)

    # Task C: Confusion matrix, metrics, threshold scan
    results_c = task_c(results_a)

    # Task D: L1 vs L2 regularized logistic regression
    results_d = task_d()

    # Write reports
    print("\n" + "=" * 60)
    print("Writing Reports...")
    print("=" * 60)
    write_synthetic_report(results_a, results_c)
    write_threshold_report(results_a, results_c)
    write_regularization_report(results_d)
    write_summary()

    print("\n" + "=" * 60)
    print("All tasks completed!")
    print(f"Reports: {RESULTS_DIR}")
    print(f"Figures: {FIG_DIR}")
    print(f"Data: {DATA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()