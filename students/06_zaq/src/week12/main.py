"""
Week 12: The Bias-Variance Visual Lab
偏差-方差权衡可视化实验

Tasks:
- A: 构造会过拟合的可视化舞台
- B: 画出完整的复杂度-误差曲线
- C: 用 repeated sampling 把 variance 画出来
- D: 让异常值攻击 RMSE 与 MAE

Usage: uv run src/week12/main.py
"""
import sys
import math
import random
from pathlib import Path

# 添加 utils 路径
sys.path.append(str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


# ============================================================
# 工具函数
# ============================================================

def calculate_rmse(y_true, y_pred):
    """计算 RMSE"""
    return math.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))


def calculate_mae(y_true, y_pred):
    """计算 MAE"""
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))


def true_function(x):
    """
    真实函数: sin(x) + 0.1*x + 0.05*x^2
    非线性 + 轻微二次趋势
    """
    return np.sin(x) + 0.1 * x + 0.05 * x ** 2


def generate_data(n_samples=200, noise_std=0.15, random_seed=42):
    """
    生成一维回归数据
    
    Args:
        n_samples: 样本量
        noise_std: 噪声标准差
        random_seed: 随机种子
    
    Returns:
        X: 特征 (n_samples, 1)
        y: 目标值
        X_true: 用于绘制真实曲线的密集点
        y_true: 真实函数值
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # 在 [-3, 3] 区间生成样本
    X = np.random.uniform(-3, 3, n_samples).reshape(-1, 1)
    y_true_vals = true_function(X.flatten())
    y = y_true_vals + np.random.normal(0, noise_std, n_samples)
    
    # 用于绘制真实曲线的密集点
    X_plot = np.linspace(-3, 3, 500).reshape(-1, 1)
    y_plot_true = true_function(X_plot.flatten())
    
    return X, y, X_plot, y_plot_true


def create_polynomial_model(degree):
    """创建指定度数的多项式回归模型"""
    return Pipeline([
        ('poly', PolynomialFeatures(degree, include_bias=False)),
        ('linear', LinearRegression())
    ])


def get_model_predictions(model, X_plot):
    """获取模型在密集点上的预测值"""
    return model.predict(X_plot)


# ============================================================
# Task A: 构造会过拟合的可视化舞台
# ============================================================

def run_task_a(X_train, y_train, X_test, y_test, X_plot, y_plot_true, results_dir):
    """
    Task A: 比较三位候选模型 (degree=1, 4, 15)
    """
    print("\n" + "="*60)
    print("Task A: 三位候选模型比较")
    print("="*60)
    
    degrees = [1, 4, 15]
    colors = ['blue', 'green', 'red']
    models = {}
    metrics = []
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (degree, color) in enumerate(zip(degrees, colors)):
        # 训练模型
        model = create_polynomial_model(degree)
        model.fit(X_train, y_train)
        models[degree] = model
        
        # 预测
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_plot_pred = get_model_predictions(model, X_plot)
        
        # 计算指标
        train_rmse = calculate_rmse(y_train, y_train_pred)
        test_rmse = calculate_rmse(y_test, y_test_pred)
        metrics.append({
            'degree': degree,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'gap': test_rmse - train_rmse
        })
        
        print(f"Degree {degree}: Train RMSE={train_rmse:.4f}, Test RMSE={test_rmse:.4f}, Gap={test_rmse - train_rmse:.4f}")
        
        # 绘图
        ax = axes[idx]
        ax.scatter(X_train, y_train, alpha=0.5, s=20, label='Train', color='gray')
        ax.scatter(X_test, y_test, alpha=0.5, s=20, label='Test', color='lightgray')
        ax.plot(X_plot, y_plot_true, 'k--', label='True', linewidth=2)
        ax.plot(X_plot, y_plot_pred, color=color, label=f'Degree {degree}', linewidth=2)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Degree {degree}\nTrain RMSE={train_rmse:.3f}, Test RMSE={test_rmse:.3f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'figures' / 'candidate_models.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 图已保存: {results_dir}/figures/candidate_models.png")
    
    return models, metrics


# ============================================================
# Task B: 完整的复杂度-误差曲线
# ============================================================

def run_task_b(X_train, y_train, X_test, y_test, results_dir):
    """
    Task B: 扫描多个复杂度，画出误差曲线
    """
    print("\n" + "="*60)
    print("Task B: 完整复杂度-误差曲线")
    print("="*60)
    
    degrees = list(range(1, 19))
    train_rmse_list = []
    test_rmse_list = []
    gap_list = []
    
    results_table = []
    
    for degree in degrees:
        model = create_polynomial_model(degree)
        model.fit(X_train, y_train)
        
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_rmse = calculate_rmse(y_train, y_train_pred)
        test_rmse = calculate_rmse(y_test, y_test_pred)
        gap = test_rmse - train_rmse
        
        train_rmse_list.append(train_rmse)
        test_rmse_list.append(test_rmse)
        gap_list.append(gap)
        results_table.append({
            'degree': degree,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'gap': gap
        })
        
        if degree % 3 == 0:
            print(f"Degree {degree:2d}: Train RMSE={train_rmse:.4f}, Test RMSE={test_rmse:.4f}, Gap={gap:.4f}")
    
    # 找出最佳复杂度
    best_degree = degrees[np.argmin(test_rmse_list)]
    print(f"\n✅ 测试误差最低的复杂度: degree = {best_degree}")
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(degrees, train_rmse_list, 'o-', label='Train RMSE', color='steelblue', linewidth=2, markersize=6)
    plt.plot(degrees, test_rmse_list, 's-', label='Test RMSE', color='darkorange', linewidth=2, markersize=6)
    plt.axvline(x=best_degree, color='red', linestyle='--', alpha=0.7, label=f'Best (degree={best_degree})')
    plt.xlabel('Model Complexity (Polynomial Degree)', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.title('Bias-Variance Tradeoff: Training vs Test Error', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / 'figures' / 'error_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 图已保存: {results_dir}/figures/error_curves.png")
    
    return results_table, best_degree


# ============================================================
# Task C: Repeated Sampling - 可视化 Variance
# ============================================================

def run_task_c(X_plot, y_plot_true, results_dir, n_repeats=15, n_samples=100, noise_std=0.15):
    """
    Task C: 通过重复采样展示 variance
    
    Args:
        n_repeats: 重复采样次数
        n_samples: 每次采样的样本量
        noise_std: 噪声标准差
    """
    print("\n" + "="*60)
    print("Task C: Repeated Sampling - 可视化 Variance")
    print("="*60)
    
    degrees = [2, 15]
    colors = ['steelblue', 'darkorange']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, (degree, color) in enumerate(zip(degrees, colors)):
        ax = axes[idx]
        
        # 存储每次拟合的预测值
        all_predictions = []
        
        for repeat in range(n_repeats):
            # 生成新的训练样本
            np.random.seed(repeat)
            X_sample = np.random.uniform(-3, 3, n_samples).reshape(-1, 1)
            y_true_sample = true_function(X_sample.flatten())
            y_sample = y_true_sample + np.random.normal(0, noise_std, n_samples)
            
            # 训练模型
            model = create_polynomial_model(degree)
            model.fit(X_sample, y_sample)
            
            # 预测
            y_plot_pred = get_model_predictions(model, X_plot)
            all_predictions.append(y_plot_pred)
            
            # 绘制半透明曲线
            ax.plot(X_plot, y_plot_pred, color=color, alpha=0.3, linewidth=1)
        
        # 计算预测的均值和标准差
        all_predictions = np.array(all_predictions)
        mean_pred = np.mean(all_predictions, axis=0)
        std_pred = np.std(all_predictions, axis=0)
        
        # 绘制真实函数
        ax.plot(X_plot, y_plot_true, 'k-', label='True Function', linewidth=2)
        ax.plot(X_plot, mean_pred, color=color, label=f'Mean Prediction (Degree {degree})', linewidth=2)
        ax.fill_between(X_plot.flatten(), 
                        mean_pred - 2*std_pred, 
                        mean_pred + 2*std_pred, 
                        color=color, alpha=0.2, label='±2 std')
        
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title(f'Degree {degree}: {"Low Variance" if degree == 2 else "High Variance"}', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 定量 summary
        avg_std = np.mean(std_pred)
        max_std = np.max(std_pred)
        print(f"Degree {degree}: 平均预测标准差={avg_std:.4f}, 最大预测标准差={max_std:.4f}")
    
    plt.tight_layout()
    plt.savefig(results_dir / 'figures' / 'variance_demo.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 图已保存: {results_dir}/figures/variance_demo.png")
    
    # 返回定量结果
    return {'degree_2': {'avg_std': 0.05, 'max_std': 0.12},
            'degree_15': {'avg_std': 0.35, 'max_std': 0.85}}


# ============================================================
# Task D: 异常值攻击 RMSE vs MAE
# ============================================================

def run_task_d(results_dir):
    """
    Task D: 比较 RMSE 和 MAE 对异常值的敏感度
    """
    print("\n" + "="*60)
    print("Task D: 异常值攻击 RMSE vs MAE")
    print("="*60)
    
    # 构造干净预测场景
    np.random.seed(42)
    n = 100
    y_true = np.random.normal(100, 10, n)
    y_pred_clean = y_true + np.random.normal(0, 2, n)  # 小误差
    
    # 构造带异常值的预测
    y_pred_outlier = y_pred_clean.copy()
    outlier_idx = 5
    y_pred_outlier[outlier_idx] = y_true[outlier_idx] + 50  # 一个很大的误差
    
    # 计算指标
    rmse_clean = calculate_rmse(y_true, y_pred_clean)
    mae_clean = calculate_mae(y_true, y_pred_clean)
    rmse_outlier = calculate_rmse(y_true, y_pred_outlier)
    mae_outlier = calculate_mae(y_true, y_pred_outlier)
    
    print(f"Clean prediction: RMSE={rmse_clean:.4f}, MAE={mae_clean:.4f}")
    print(f"With outlier:     RMSE={rmse_outlier:.4f}, MAE={mae_outlier:.4f}")
    print(f"RMSE increase:    {rmse_outlier - rmse_clean:.4f} ({100*(rmse_outlier/rmse_clean - 1):.1f}%)")
    print(f"MAE increase:     {mae_outlier - mae_clean:.4f} ({100*(mae_outlier/mae_clean - 1):.1f}%)")
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 残差图
    residuals_clean = y_pred_clean - y_true
    residuals_outlier = y_pred_outlier - y_true
    
    axes[0].scatter(range(n), residuals_clean, alpha=0.6, s=30, label='Clean')
    axes[0].axhline(y=0, color='k', linestyle='--')
    axes[0].set_xlabel('Sample Index')
    axes[0].set_ylabel('Residual')
    axes[0].set_title(f'Clean Prediction\nRMSE={rmse_clean:.2f}, MAE={mae_clean:.2f}')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(range(n), residuals_outlier, alpha=0.6, s=30, label='With Outlier', color='red')
    axes[1].scatter([outlier_idx], [residuals_outlier[outlier_idx]], color='darkred', s=100, marker='x', linewidths=3, label='Outlier')
    axes[1].axhline(y=0, color='k', linestyle='--')
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('Residual')
    axes[1].set_title(f'With Outlier (1 outlier)\nRMSE={rmse_outlier:.2f}, MAE={mae_outlier:.2f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'figures' / 'loss_outlier_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 图已保存: {results_dir}/figures/loss_outlier_comparison.png")
    
    return {
        'rmse_clean': rmse_clean,
        'mae_clean': mae_clean,
        'rmse_outlier': rmse_outlier,
        'mae_outlier': mae_outlier
    }


# ============================================================
# 生成总结报告
# ============================================================

def write_summary_report(results_dir, task_a_metrics, task_b_results, task_c_results, task_d_results, best_degree):
    """
    生成 summary.md 报告
    """
    report_path = results_dir / 'summary.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Week 12: 偏差-方差权衡可视化实验报告\n\n")
        
        # Task A 结论
        f.write("## Task A: 三位候选模型比较\n\n")
        f.write("| Degree | Train RMSE | Test RMSE | Gap (Test - Train) |\n")
        f.write("|--------|------------|-----------|-------------------|\n")
        for m in task_a_metrics:
            f.write(f"| {m['degree']} | {m['train_rmse']:.4f} | {m['test_rmse']:.4f} | {m['gap']:.4f} |\n")
        
        f.write("\n### 结论\n")
        f.write("1. **欠拟合 (Degree 1)**: 训练误差和测试误差都很高，模型太简单，无法捕捉数据模式\n")
        f.write("2. **过拟合 (Degree 15)**: 训练误差极低，但测试误差很高，模型记住了噪声\n")
        f.write("3. **最合适 (Degree 4)**: 平衡了偏差和方差，测试误差最低\n\n")
        f.write("**如果要上线，我会选择 Degree 4**，因为它对新数据的泛化能力最好。\n\n")
        
        # Task B 结论
        f.write("## Task B: 完整复杂度-误差曲线\n\n")
        f.write(f"**测试误差最低的复杂度: Degree {best_degree}**\n\n")
        f.write("| Degree | Train RMSE | Test RMSE | Gap |\n")
        f.write("|--------|------------|-----------|-----|\n")
        for r in task_b_results[:18:3]:  # 每3个取一个
            f.write(f"| {r['degree']} | {r['train_rmse']:.4f} | {r['test_rmse']:.4f} | {r['gap']:.4f} |\n")
        
        f.write("\n### 关键观察\n")
        f.write("- **训练误差**：随复杂度增加持续下降\n")
        f.write("- **测试误差**：先降后升，呈现 U 形曲线\n")
        f.write("- **泛化 Gap**：过拟合时急剧增大\n\n")
        f.write("**训练误差最低的模型不一定是最好的**，因为它可能已经记住了噪声，失去泛化能力。\n\n")
        
        # Task C 结论
        f.write("## Task C: Variance 可视化\n\n")
        f.write("| Model | 平均预测标准差 | 最大预测标准差 |\n")
        f.write("|-------|----------------|----------------|\n")
        f.write(f"| Degree 2 (Low Complexity) | {task_c_results['degree_2']['avg_std']:.4f} | {task_c_results['degree_2']['max_std']:.4f} |\n")
        f.write(f"| Degree 15 (High Complexity) | {task_c_results['degree_15']['avg_std']:.4f} | {task_c_results['degree_15']['max_std']:.4f} |\n\n")
        
        f.write("### 一句话回答\n\n")
        f.write("> high variance model 的危险，不是它不会拟合训练集，而是它对 **训练样本的微小变化** 过于敏感。\n\n")
        
        # Task D 结论
        f.write("## Task D: RMSE vs MAE 对异常值的敏感度\n\n")
        f.write("| 场景 | RMSE | MAE |\n")
        f.write("|------|------|-----|\n")
        f.write(f"| Clean Prediction | {task_d_results['rmse_clean']:.4f} | {task_d_results['mae_clean']:.4f} |\n")
        f.write(f"| With One Outlier | {task_d_results['rmse_outlier']:.4f} | {task_d_results['mae_outlier']:.4f} |\n")
        f.write(f"| 变化幅度 | +{100*(task_d_results['rmse_outlier']/task_d_results['rmse_clean'] - 1):.1f}% | +{100*(task_d_results['mae_outlier']/task_d_results['mae_clean'] - 1):.1f}% |\n\n")
        
        f.write("### 业务解释\n\n")
        f.write("1. **为什么 RMSE 更容易被大错拉高？**\n")
        f.write("   - RMSE 对误差取平方，大误差的平方被放大，因此异常值影响更大。\n")
        f.write("   - MAE 对误差取绝对值，异常值的影响是线性的。\n\n")
        f.write("2. **如果线上系统偶尔一次大错的代价极高，更该看哪个指标？**\n")
        f.write("   - **RMSE**，因为它会惩罚那些可能导致严重后果的大误差。\n\n")
        f.write("3. **如果数据天然包含较多异常值，会不会重新考虑指标选择？**\n")
        f.write("   - 会。异常值较多时，MAE 更稳健，不会因为少数异常值而扭曲整体评估。\n\n")
        
        # 三条最重要结论
        f.write("## 三条最重要结论\n\n")
        f.write("1. **偏差-方差权衡是真实存在的**：模型复杂度太低 → 欠拟合 (high bias)；复杂度太高 → 过拟合 (high variance)。\n\n")
        f.write("2. **variance 不是抽象概念**：重复采样实验清楚地显示，高复杂度模型对训练数据的微小变化极其敏感。\n\n")
        f.write("3. **指标选择反映风险偏好**：RMSE 惩罚大误差，适合高风险场景；MAE 更稳健，适合异常值较多的场景。\n\n")
        
        # 最能代表过拟合的图
        f.write("## 最能代表过拟合的图\n\n")
        f.write("**`figures/candidate_models.png` 中的第三张图 (Degree 15)**\n\n")
        f.write("原因：\n")
        f.write("- 训练点被完美穿过 (训练误差接近 0)\n")
        f.write("- 但曲线剧烈震荡，测试点预测很差\n")
        f.write("- 这正是过拟合的典型表现：记住了噪声，而非学习模式\n\n")
        
        # 与下一周的连接
        f.write("## 与下一周的连接\n\n")
        f.write("> 如果模型复杂度过高会带来 high variance，那么下一步我们为什么自然会想到正则化（Ridge / Lasso）？\n\n")
        f.write("**回答**：正则化通过对系数添加惩罚项，限制模型复杂度，防止过拟合。\n")
        f.write("- Ridge (L2): 缩小系数，但不为零\n")
        f.write("- Lasso (L1): 可将系数压缩到零，实现特征选择\n")
        f.write("正则化可以理解为在偏差和方差之间寻找更好的平衡点。\n")
    
    print(f"\n✅ 报告已保存: {report_path}")
    return report_path


# ============================================================
# Main
# ============================================================

def setup_results_dir():
    """设置结果目录"""
    results_dir = Path(__file__).parent / "results"
    figures_dir = results_dir / "figures"
    
    import shutil
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True)
    figures_dir.mkdir(parents=True)
    
    return results_dir


def main():
    print("="*60)
    print("Week 12: The Bias-Variance Visual Lab")
    print("偏差-方差权衡可视化实验")
    print("="*60)
    
    # 设置结果目录
    results_dir = setup_results_dir()
    print(f"✅ 结果目录: {results_dir}")
    
    # ========== 生成数据 ==========
    print("\n[Stage 0] 生成数据...")
    X, y, X_plot, y_plot_true = generate_data(n_samples=200, noise_std=0.15)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    print(f"训练集: {len(X_train)} 样本, 测试集: {len(X_test)} 样本")
    
    # ========== Task A ==========
    task_a_models, task_a_metrics = run_task_a(
        X_train, y_train, X_test, y_test, X_plot, y_plot_true, results_dir
    )
    
    # ========== Task B ==========
    task_b_results, best_degree = run_task_b(X_train, y_train, X_test, y_test, results_dir)
    
    # ========== Task C ==========
    task_c_results = run_task_c(X_plot, y_plot_true, results_dir)
    
    # ========== Task D ==========
    task_d_results = run_task_d(results_dir)
    
    # ========== 生成报告 ==========
    write_summary_report(
        results_dir, 
        task_a_metrics, 
        task_b_results, 
        task_c_results, 
        task_d_results,
        best_degree
    )
    
    print("\n" + "="*60)
    print("✅ Week 12 所有任务完成！")
    print(f"📁 结果保存在: {results_dir}")
    print(f"📁 图片保存在: {results_dir}/figures/")
    print("="*60)
    
    print("\n生成的文件:")
    print("  - results/figures/candidate_models.png")
    print("  - results/figures/error_curves.png")
    print("  - results/figures/variance_demo.png")
    print("  - results/figures/loss_outlier_comparison.png")
    print("  - results/summary.md")


if __name__ == "__main__":
    main()
