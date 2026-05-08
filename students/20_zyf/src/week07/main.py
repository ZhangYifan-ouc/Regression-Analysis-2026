"""
Module: week07.main
Purpose: Cross-validation, tuning, and generalization analysis.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.models import AnalyticalOLS, GradientDescentOLS


def rmse(y_true, y_pred):
    """Calculate RMSE."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def task_cross_validation(X, y):
    """Task 2: 5-Fold CV on AnalyticalOLS."""
    print("\n" + "=" * 60)
    print("Task 2: 5-Fold Cross-Validation on AnalyticalOLS")
    print("=" * 60)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    r2_scores = []
    rmse_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = AnalyticalOLS().fit(X_train, y_train)
        preds = model.predict(X_val)

        fold_r2 = r2_score(y_val, preds)
        fold_rmse = rmse(y_val, preds)

        r2_scores.append(fold_r2)
        rmse_scores.append(fold_rmse)

        print(f"  Fold {fold}: R² = {fold_r2:.4f}, RMSE = {fold_rmse:.4f}")

    print("-" * 40)
    print(f"  Average CV R²:   {np.mean(r2_scores):.4f} (±{np.std(r2_scores):.4f})")
    print(f"  Average CV RMSE: {np.mean(rmse_scores):.4f} (±{np.std(rmse_scores):.4f})")
    
    return np.mean(r2_scores), np.mean(rmse_scores)


def task_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    """Task 3: Tune learning rate for GradientDescentOLS."""
    print("\n" + "=" * 60)
    print("Task 3: Hyperparameter Tuning (Learning Rate)")
    print("=" * 60)
    
    print("\n  Fixed parameters: gd_type='mini_batch', batch_fraction=0.2, tol=1e-5, max_iter=1000")
    print("-" * 60)
    print(f"  {'Learning Rate':<15} {'Val R²':<12} {'Val RMSE':<12}")
    print("-" * 60)

    learning_rates = [0.1, 0.01, 0.001, 0.0001, 1e-5]
    best_lr = None
    best_score = -np.inf
    results = []

    for lr in learning_rates:
        model = GradientDescentOLS(
            learning_rate=lr,
            tol=1e-5,
            max_iter=1000,
            gd_type="mini_batch",
            batch_fraction=0.2,
        ).fit(X_train, y_train)

        val_preds = model.predict(X_val)
        val_r2 = r2_score(y_val, val_preds)
        val_rmse_val = rmse(y_val, val_preds)

        results.append((lr, val_r2, val_rmse_val))
        print(f"  {lr:<15.0e} {val_r2:<12.4f} {val_rmse_val:<12.4f}")

        if val_r2 > best_score:
            best_score = val_r2
            best_lr = lr

    print("-" * 60)
    print(f"\n  ✓ Selected best learning rate: {best_lr}")
    print(f"    Best Validation R²: {best_score:.4f}")
    
    return best_lr, results


def task_final_comparison(X_train, y_train, X_val, y_val, X_test, y_test, best_lr):
    """Task 3: Final comparison on Test set."""
    print("\n" + "=" * 60)
    print("Task 3: Final Comparison on Test Set")
    print("=" * 60)
    
    # Train GradientDescentOLS with best learning rate
    gd_model = GradientDescentOLS(
        learning_rate=best_lr,
        tol=1e-5,
        max_iter=1000,
        gd_type="mini_batch",
        batch_fraction=0.2,
    ).fit(X_train, y_train)

    # Train AnalyticalOLS
    analytical_model = AnalyticalOLS().fit(X_train, y_train)

    # Predict on test set
    gd_preds = gd_model.predict(X_test)
    ols_preds = analytical_model.predict(X_test)

    # Calculate metrics
    gd_r2 = r2_score(y_test, gd_preds)
    gd_rmse_val = rmse(y_test, gd_preds)
    ols_r2 = r2_score(y_test, ols_preds)
    ols_rmse_val = rmse(y_test, ols_preds)

    print(f"\n  {'Model':<25} {'Test R²':<12} {'Test RMSE':<12}")
    print("  " + "-" * 50)
    print(f"  {'GradientDescentOLS':<25} {gd_r2:<12.4f} {gd_rmse_val:<12.4f}")
    print(f"  {'AnalyticalOLS':<25} {ols_r2:<12.4f} {ols_rmse_val:<12.4f}")
    
    return {
        "gd_r2": gd_r2,
        "gd_rmse": gd_rmse_val,
        "ols_r2": ols_r2,
        "ols_rmse": ols_rmse_val,
    }


def task_plot_learning_curve(X_train, y_train, results_dir: Path):
    """Task 4: Plot learning curves comparing full batch vs mini batch."""
    print("\n" + "=" * 60)
    print("Task 4: Learning Curve Comparison")
    print("=" * 60)
    
    # Full batch GD
    model_full = GradientDescentOLS(
        learning_rate=0.01,
        gd_type="full_batch",
        max_iter=300,
    ).fit(X_train, y_train)

    # Mini batch GD
    model_mini = GradientDescentOLS(
        learning_rate=0.01,
        gd_type="mini_batch",
        batch_fraction=0.1,
        max_iter=300,
    ).fit(X_train, y_train)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(model_full.loss_history_, label="Full Batch GD", color="steelblue", linewidth=2)
    plt.plot(model_mini.loss_history_, label="Mini-Batch GD (10%)", color="darkorange", linewidth=2, alpha=0.8)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("MSE Loss", fontsize=12)
    plt.title("Learning Curve: Full Batch vs Mini-Batch GD", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = results_dir / "learning_curve_full_vs_mini.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"\n  ✓ Learning curve saved to: {output_path}")
    
    return model_full, model_mini


def generate_report(cv_results, tuning_results, test_results, best_lr, results_dir):
    """Generate summary report."""
    report = f"""# Week 7 实验报告

## 1. 任务概述

本周实现了梯度下降优化的线性回归模型 (`GradientDescentOLS`)，并与解析解 OLS (`AnalyticalOLS`) 进行了全面对比。

## 2. Task 2: 5-Fold Cross-Validation (AnalyticalOLS)

| 指标 | 数值 |
|------|------|
| 平均 R² | {cv_results[0]:.4f} |
| 平均 RMSE | {cv_results[1]:.4f} |

## 3. Task 3: 超参数调优 (Learning Rate)

最佳学习率: **{best_lr}**

### 各学习率验证集表现

| Learning Rate | Val R² | Val RMSE |
|---------------|--------|----------|
"""
    
    for lr, r2, rmse_val in tuning_results:
        report += f"| {lr:.0e} | {r2:.4f} | {rmse_val:.4f} |\n"
    
    report += f"""

---
*报告自动生成*
"""
    
    report_path = results_dir / "summary_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\n  ✓ Summary report saved to: {report_path}")


def main():
    """Main experiment flow."""
    print("\n" + "=" * 60)
    print("  Week 7: Optimization Engine & Generalization Quest")
    print("=" * 60)
    
    # Setup paths
    base_dir = Path(__file__).parent.parent.parent
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Data path: go up 4 levels from main.py to reach project root
    project_root = Path(__file__).parent.parent.parent.parent.parent
    data_path = project_root / "homework" / "week06" / "data" / "q3_marketing.csv"
    
    # Load data
    print("\n  Loading data from: q3_marketing.csv")
    df = pd.read_csv(data_path)
    
    # Define features and target
    feature_cols = ["TV_Budget", "Radio_Budget", "SocialMedia_Budget"]
    target_col = "Sales"
    
    X = df[feature_cols].to_numpy()
    y = df[target_col].to_numpy()
    
    print(f"  Data shape: X = {X.shape}, y = {y.shape}")
    print(f"  Features: {feature_cols}")
    print(f"  Target: {target_col}")
    
    # ============================================
    # Task 2: 5-Fold CV for AnalyticalOLS
    # ============================================
    # Add intercept column
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    cv_results = task_cross_validation(X_with_intercept, y)
    
    # ============================================
    # Task 3: Train / Val / Test split
    # ============================================
    print("\n" + "=" * 60)
    print("Data Split: Train 60% / Validation 20% / Test 20%")
    print("=" * 60)
    
    # First split: 60% train, 40% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    # Second split: 50% of temp = 20% each for val and test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")
    
    # ============================================
    # Task 4: Feature Scaling (prevent data leakage)
    # ============================================
    print("\n" + "=" * 60)
    print("Task 4: Feature Scaling (Data Leakage Prevention)")
    print("=" * 60)
    
    # Fit scaler on TRAIN ONLY
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print("  ✓ Scaler fitted on Train set only")
    print("  ✓ Applied same transformation to Val and Test")
    
    # Add intercept column AFTER scaling (don't scale the intercept!)
    X_train_scaled = np.column_stack([np.ones(len(X_train_scaled)), X_train_scaled])
    X_val_scaled = np.column_stack([np.ones(len(X_val_scaled)), X_val_scaled])
    X_test_scaled = np.column_stack([np.ones(len(X_test_scaled)), X_test_scaled])
    
    print(f"  ✓ Added intercept column (all ones)")
    print(f"  Final shapes: Train {X_train_scaled.shape}, Val {X_val_scaled.shape}, Test {X_test_scaled.shape}")
    
    # ============================================
    # Task 3: Hyperparameter tuning
    # ============================================
    best_lr, tuning_results = task_hyperparameter_tuning(
        X_train_scaled, y_train, X_val_scaled, y_val
    )
    
    # ============================================
    # Task 3: Final comparison on Test set
    # ============================================
    test_results = task_final_comparison(
        X_train_scaled, y_train, 
        X_val_scaled, y_val,
        X_test_scaled, y_test, 
        best_lr
    )
    
    # ============================================
    # Task 4: Learning curve
    # ============================================
    task_plot_learning_curve(X_train_scaled, y_train, results_dir)
    
    # ============================================
    # Generate report
    # ============================================
    generate_report(cv_results, tuning_results, test_results, best_lr, results_dir)
    
    print("\n" + "=" * 60)
    print("  ✓ All tasks completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()