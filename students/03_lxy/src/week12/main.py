# students/03_1xy/src/week12/main.py
"""
Week 12: Bias-Variance Visual Lab
偏差-方差可视化实验
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 设置路径
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.metrics import calculate_mae, calculate_rmse

# 设置随机种子
np.random.seed(42)

# 设置 matplotlib - 使用英文避免字体问题
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 定义结果目录
RESULTS_DIR = CURRENT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def ensure_directories() -> None:
    """Ensure result directories exist"""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[Init] Results directory: {RESULTS_DIR}")


def generate_data(n_samples: int = 200, noise_std: float = 0.2, test_size: float = 0.3):
    """Generate synthetic regression data"""
    X = np.random.uniform(0, 2, n_samples)
    X_sorted = np.linspace(0, 2, 1000)
    
    def true_function(x):
        return np.sin(2 * np.pi * x) + 0.5 * x
    
    y_true = true_function(X)
    noise = np.random.normal(0, noise_std, n_samples)
    y = y_true + noise
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    print(f"[Data] Training: {len(X_train)} samples, Test: {len(X_test)} samples")
    
    return {
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'X_sorted': X_sorted,
        'y_true_sorted': true_function(X_sorted)
    }


def task_a_candidate_models(data: dict):
    """Task A: Compare candidate models (degree=1, 4, 15)"""
    print("\n" + "="*60)
    print("[Task A] Comparing Candidate Models (degree=1, 4, 15)")
    print("="*60)
    
    degrees = [1, 4, 15]
    results = {}
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, degree in enumerate(degrees):
        model = Pipeline([
            ('poly', PolynomialFeatures(degree)),
            ('linear', LinearRegression())
        ])
        model.fit(data['X_train'].reshape(-1, 1), data['y_train'])
        
        y_train_pred = model.predict(data['X_train'].reshape(-1, 1))
        y_test_pred = model.predict(data['X_test'].reshape(-1, 1))
        
        train_rmse = calculate_rmse(data['y_train'], y_train_pred)
        test_rmse = calculate_rmse(data['y_test'], y_test_pred)
        
        results[degree] = {'train_rmse': train_rmse, 'test_rmse': test_rmse}
        
        X_plot = np.linspace(0, 2, 500)
        y_plot_pred = model.predict(X_plot.reshape(-1, 1))
        
        ax = axes[idx]
        ax.scatter(data['X_train'], data['y_train'], alpha=0.6, s=20, color='blue', label='Training')
        ax.scatter(data['X_test'], data['y_test'], alpha=0.6, s=20, color='orange', label='Test')
        ax.plot(data['X_sorted'], data['y_true_sorted'], 'k--', label='True Function', linewidth=2)
        ax.plot(X_plot, y_plot_pred, color='red', linewidth=2, label=f'Degree={degree}')
        
        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('y', fontsize=11)
        ax.set_title(f'Degree {degree}\nTrain RMSE={train_rmse:.3f}, Test RMSE={test_rmse:.3f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = FIGURES_DIR / 'candidate_models.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] Saved: {save_path}")
    for degree in degrees:
        print(f"  Degree {degree}: Train RMSE={results[degree]['train_rmse']:.4f}, "
              f"Test RMSE={results[degree]['test_rmse']:.4f}")
    
    return results


def task_b_error_curves(data: dict):
    """Task B: Scan model complexity (degree=1 to 18)"""
    print("\n" + "="*60)
    print("[Task B] Scanning Model Complexity (degree=1 to 18)")
    print("="*60)
    
    degrees = range(1, 19)
    train_errors = []
    test_errors = []
    
    print("  Degree | Train RMSE | Test RMSE | Gap")
    print("  " + "-" * 45)
    
    for degree in degrees:
        model = Pipeline([
            ('poly', PolynomialFeatures(degree)),
            ('linear', LinearRegression())
        ])
        model.fit(data['X_train'].reshape(-1, 1), data['y_train'])
        
        y_train_pred = model.predict(data['X_train'].reshape(-1, 1))
        y_test_pred = model.predict(data['X_test'].reshape(-1, 1))
        
        train_rmse = calculate_rmse(data['y_train'], y_train_pred)
        test_rmse = calculate_rmse(data['y_test'], y_test_pred)
        gap = test_rmse - train_rmse
        
        train_errors.append(train_rmse)
        test_errors.append(test_rmse)
        
        if degree % 3 == 0:
            print(f"  {degree:3d}   | {train_rmse:9.4f} | {test_rmse:8.4f} | {gap:8.4f}")
    
    best_degree = degrees[np.argmin(test_errors)]
    best_error = min(test_errors)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(degrees, train_errors, 'o-', label='Train RMSE', 
           linewidth=2, markersize=6, color='blue')
    ax.plot(degrees, test_errors, 's-', label='Test RMSE', 
           linewidth=2, markersize=6, color='red')
    
    ax.axvline(x=best_degree, color='green', linestyle='--', alpha=0.5, 
              label=f'Best Complexity: degree={best_degree}')
    ax.scatter(best_degree, best_error, color='green', s=100, zorder=5)
    
    ax.set_xlabel('Model Complexity (Polynomial Degree)', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('Model Complexity vs Error Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = FIGURES_DIR / 'error_curves.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n  [OK] Saved: {save_path}")
    print(f"  [OK] Best complexity: degree={best_degree}, Test RMSE={best_error:.4f}")
    
    return {'best_degree': best_degree, 'train_errors': train_errors, 'test_errors': test_errors}


def task_c_variance_demo(data: dict):
    """Task C: Demonstrate variance through repeated sampling"""
    print("\n" + "="*60)
    print("[Task C] Demonstrating High Variance (20 repeated samplings)")
    print("="*60)
    
    degrees = [2, 15]
    n_repeats = 20
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    variance_stats = {}
    
    def true_function(x):
        return np.sin(2 * np.pi * x) + 0.5 * x
    
    for idx, degree in enumerate(degrees):
        all_predictions = []
        
        print(f"  Processing degree={degree}...")
        
        for repeat in range(n_repeats):
            X = np.random.uniform(0, 2, 200)
            y_true = true_function(X)
            noise = np.random.normal(0, 0.2, 200)
            y = y_true + noise
            
            X_train_r, _, y_train_r, _ = train_test_split(
                X, y, test_size=0.3, random_state=repeat
            )
            
            model = Pipeline([
                ('poly', PolynomialFeatures(degree)),
                ('linear', LinearRegression())
            ])
            model.fit(X_train_r.reshape(-1, 1), y_train_r)
            
            X_plot = np.linspace(0, 2, 200)
            y_pred = model.predict(X_plot.reshape(-1, 1))
            all_predictions.append(y_pred)
        
        all_predictions = np.array(all_predictions)
        mean_pred = np.mean(all_predictions, axis=0)
        std_pred = np.std(all_predictions, axis=0)
        
        mean_std = np.mean(std_pred)
        max_std = np.max(std_pred)
        variance_stats[degree] = {'mean_std': mean_std, 'max_std': max_std}
        
        ax = axes[idx]
        X_plot = np.linspace(0, 2, 200)
        y_true_plot = true_function(X_plot)
        
        for pred in all_predictions[:10]:
            ax.plot(X_plot, pred, 'b-', alpha=0.2, linewidth=1)
        
        ax.plot(X_plot, y_true_plot, 'k-', linewidth=3, label='True Function')
        ax.fill_between(X_plot, mean_pred - std_pred, mean_pred + std_pred, 
                       alpha=0.3, color='red', label='±1 Std Dev')
        ax.plot(X_plot, mean_pred, 'r--', linewidth=2, label='Mean Prediction')
        
        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('y', fontsize=11)
        ax.set_title(f'Degree {degree}\nMean Std={mean_std:.3f}, Max Std={max_std:.3f}')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = FIGURES_DIR / 'variance_demo.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] Saved: {save_path}")
    print(f"  [OK] Variance stats: degree=2 -> mean_std={variance_stats[2]['mean_std']:.4f}")
    print(f"                    degree=15 -> mean_std={variance_stats[15]['mean_std']:.4f}")
    
    return variance_stats


def task_d_loss_comparison():
    """Task D: RMSE vs MAE sensitivity to outliers"""
    print("\n" + "="*60)
    print("[Task D] Comparing RMSE and MAE Sensitivity to Outliers")
    print("="*60)
    
    n_points = 100
    np.random.seed(42)
    y_true = np.random.normal(10, 2, n_points)
    y_pred_clean = y_true + np.random.normal(0, 0.5, n_points)
    
    y_pred_outlier = y_pred_clean.copy()
    outlier_idx = 5
    y_pred_outlier[outlier_idx] = y_true[outlier_idx] + 20
    
    rmse_clean = calculate_rmse(y_true, y_pred_clean)
    mae_clean = calculate_mae(y_true, y_pred_clean)
    rmse_outlier = calculate_rmse(y_true, y_pred_outlier)
    mae_outlier = calculate_mae(y_true, y_pred_outlier)
    
    rmse_change = (rmse_outlier - rmse_clean) / rmse_clean * 100
    mae_change = (mae_outlier - mae_clean) / mae_clean * 100
    
    print(f"  Clean: RMSE={rmse_clean:.4f}, MAE={mae_clean:.4f}")
    print(f"  With outlier: RMSE={rmse_outlier:.4f} (+{rmse_change:.1f}%), MAE={mae_outlier:.4f} (+{mae_change:.1f}%)")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    errors_clean = y_pred_clean - y_true
    errors_outlier = y_pred_outlier - y_true
    
    ax1 = axes[0]
    ax1.boxplot([errors_clean, errors_outlier], labels=['Clean', 'With Outlier'], widths=0.6)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Prediction Error', fontsize=11)
    ax1.set_title('Error Distribution Comparison', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    x_pos = np.arange(2)
    width = 0.35
    
    rmse_bars = ax2.bar(x_pos - width/2, [rmse_clean, rmse_outlier], 
                       width, label='RMSE', color='blue', alpha=0.7)
    mae_bars = ax2.bar(x_pos + width/2, [mae_clean, mae_outlier], 
                      width, label='MAE', color='orange', alpha=0.7)
    
    for bar in rmse_bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom')
    for bar in mae_bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom')
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(['Clean', 'With Outlier'])
    ax2.set_ylabel('Error', fontsize=11)
    ax2.set_title(f'RMSE vs MAE\nRMSE Change: +{rmse_change:.1f}%, MAE Change: +{mae_change:.1f}%',
                 fontsize=11)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = FIGURES_DIR / 'loss_outlier_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] Saved: {save_path}")
    
    return {
        'rmse_clean': rmse_clean, 'rmse_outlier': rmse_outlier,
        'mae_clean': mae_clean, 'mae_outlier': mae_outlier,
        'rmse_change': rmse_change, 'mae_change': mae_change
    }


def write_summary_report(task_a_results, task_b_results, task_c_results, task_d_results):
    """Generate summary report"""
    print("\n" + "="*60)
    print("[Report] Generating summary.md")
    print("="*60)
    
    report_path = RESULTS_DIR / "summary.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Week 12: Bias-Variance Visual Lab Report\n\n")
        
        f.write("## Three Core Conclusions\n\n")
        f.write("1. **Optimal model complexity exists**: Train error decreases continuously, ")
        f.write(f"but test error first decreases then increases. Best complexity: degree={task_b_results['best_degree']}\n\n")
        f.write("2. **High variance model behavior**: High complexity model (degree=15) shows ")
        f.write(f"much larger prediction variance (mean std={task_c_results[15]['mean_std']:.4f}) ")
        f.write(f"compared to low complexity model (degree=2, mean std={task_c_results[2]['mean_std']:.4f})\n\n")
        f.write("3. **RMSE is more sensitive to outliers**: Single large error causes ")
        f.write(f"RMSE to increase by {task_d_results['rmse_change']:.1f}%, ")
        f.write(f"while MAE increases by only {task_d_results['mae_change']:.1f}%\n\n")
        
        f.write("## Task A: Candidate Models\n\n")
        f.write("| Degree | Train RMSE | Test RMSE | Diagnosis |\n")
        f.write("|--------|------------|-----------|-----------|\n")
        for degree in [1, 4, 15]:
            train_rmse = task_a_results[degree]['train_rmse']
            test_rmse = task_a_results[degree]['test_rmse']
            diagnosis = "Underfitting" if degree == 1 else ("Good" if degree == 4 else "Overfitting")
            f.write(f"| {degree} | {train_rmse:.4f} | {test_rmse:.4f} | {diagnosis} |\n")
        
        f.write("\n## Task B: Error Curves\n\n")
        f.write(f"**Best complexity (lowest test error)**: degree={task_b_results['best_degree']}\n\n")
        
        f.write("## Task C: Variance Analysis\n\n")
        f.write("| Degree | Mean Prediction Std | Max Prediction Std |\n")
        f.write("|--------|--------------------|--------------------|\n")
        f.write(f"| 2 | {task_c_results[2]['mean_std']:.4f} | {task_c_results[2]['max_std']:.4f} |\n")
        f.write(f"| 15 | {task_c_results[15]['mean_std']:.4f} | {task_c_results[15]['max_std']:.4f} |\n\n")
        
        f.write("**One sentence completion**:\n\n")
        f.write("> High variance model's danger is not that it fails to fit training data, ")
        f.write("but that it is too sensitive to **random fluctuations in training samples**.\n\n")
        
        f.write("## Task D: RMSE vs MAE\n\n")
        f.write("| Scenario | RMSE | MAE |\n")
        f.write("|----------|------|-----|\n")
        f.write(f"| Clean | {task_d_results['rmse_clean']:.4f} | {task_d_results['mae_clean']:.4f} |\n")
        f.write(f"| With Outlier | {task_d_results['rmse_outlier']:.4f} | {task_d_results['mae_outlier']:.4f} |\n")
        f.write(f"| Change | +{task_d_results['rmse_change']:.1f}% | +{task_d_results['mae_change']:.1f}% |\n\n")
        
        f.write("**Why use RMSE?** - When large errors are extremely costly\n\n")
        f.write("**Why use MAE?** - When data contains many outliers\n\n")
        
        f.write("## Connection to Next Week\n\n")
        f.write("**Why regularization (Ridge/Lasso)?**\n\n")
        f.write("After observing overfitting with high complexity models, we need to control ")
        f.write("model complexity without completely discarding high-degree features. ")
        f.write("Regularization adds penalty terms to the loss function:\n")
        f.write("- **Ridge (L2)**: Penalizes squared coefficients\n")
        f.write("- **Lasso (L1)**: Penalizes absolute coefficients (feature selection)\n\n")
        f.write("This is the natural next step: **improving generalization without increasing model capacity**.\n")
    
    print(f"  [OK] Saved report: {report_path}")


def main() -> None:
    """Main function"""
    print("\n" + "="*60)
    print("Week 12: The Bias-Variance Visual Lab")
    print("="*60)
    
    ensure_directories()
    
    data = generate_data(n_samples=200, noise_std=0.2, test_size=0.3)
    
    task_a_results = task_a_candidate_models(data)
    task_b_results = task_b_error_curves(data)
    task_c_results = task_c_variance_demo(data)
    task_d_results = task_d_loss_comparison()
    
    write_summary_report(task_a_results, task_b_results, task_c_results, task_d_results)
    
    print("\n" + "="*60)
    print("[Done] Week 12 Assignment Completed!")
    print(f"[Path] Results saved in: {RESULTS_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()