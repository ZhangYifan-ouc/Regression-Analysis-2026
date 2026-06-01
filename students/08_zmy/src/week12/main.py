import sys
from pathlib import Path

src_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(src_dir))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import pandas as pd

try:
    from utils.metrics import calculate_rmse, calculate_mae
except ImportError:
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    def calculate_rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))
    def calculate_mae(y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 查找系统中支持中文的字体
chinese_fonts = [f.name for f in fm.fontManager.ttflist if 'CJK' in f.name or 'Hei' in f.name or 'YaHei' in f.name]
if chinese_fonts:
    plt.rcParams['font.sans-serif'] = [chinese_fonts[0]]
else:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def generate_data(n_samples=150, noise_std=0.1):
    X = np.random.uniform(0, 1, size=n_samples).reshape(-1, 1)
    y_true = np.sin(2 * np.pi * X.ravel()) + 0.2 * X.ravel()
    noise = np.random.normal(0, noise_std, size=n_samples)
    y = y_true + noise
    return X, y, y_true

def fit_knn(k, X_train, y_train):
    return KNeighborsRegressor(n_neighbors=k).fit(X_train, y_train)

def fit_polynomial(degree, X_train, y_train):
    return Pipeline([
        ('poly', PolynomialFeatures(degree, include_bias=False)),
        ('linear', LinearRegression())
    ]).fit(X_train, y_train)

def evaluate_model(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_rmse = calculate_rmse(y_train, y_train_pred)
    test_rmse = calculate_rmse(y_test, y_test_pred)
    return train_rmse, test_rmse

def plot_curves(ax, X_grid, y_true, X_train, y_train, X_test, y_test, models, labels, title):
    ax.scatter(X_train, y_train, s=20, label='训练集', alpha=0.7)
    ax.scatter(X_test, y_test, s=20, label='测试集', alpha=0.7, marker='x')
    ax.plot(X_grid, y_true, 'k--', label='真实函数', linewidth=2)
    for model, label in zip(models, labels):
        ax.plot(X_grid, model.predict(X_grid), label=label)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

def task_a(X_train, y_train, X_test, y_test, X_grid, y_true, results_dir):
    print("[Task A] 比较三个 KNN 模型（k=1, 5, 30）...")
    ks = [1, 5, 30]
    models = [fit_knn(k, X_train, y_train) for k in ks]
    labels = [f'k={k}' for k in ks]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_curves(ax, X_grid, y_true, X_train, y_train, X_test, y_test, models, labels,
                "候选模型：KNN 回归（k=1 过拟合，k=30 欠拟合）")
    for i, (k, model) in enumerate(zip(ks, models)):
        train_rmse, test_rmse = evaluate_model(model, X_train, y_train, X_test, y_test)
        ax.text(0.02, 0.95 - 0.05*i, f"k={k}: 训练RMSE={train_rmse:.3f}, 测试RMSE={test_rmse:.3f}",
                transform=ax.transAxes, fontsize=9, verticalalignment='top')
    fig.tight_layout()
    fig.savefig(results_dir / "figures/candidate_models.png", dpi=150)
    plt.close()
    print("  -> 已保存 figures/candidate_models.png")

def task_b(X_train, y_train, X_test, y_test, results_dir):
    print("[Task B] 扫描多项式次数 1 到 18...")
    degrees = range(1, 19)
    train_rmse_list, test_rmse_list, gap_list = [], [], []
    for d in degrees:
        model = fit_polynomial(d, X_train, y_train)
        train_rmse, test_rmse = evaluate_model(model, X_train, y_train, X_test, y_test)
        train_rmse_list.append(train_rmse)
        test_rmse_list.append(test_rmse)
        gap_list.append(test_rmse - train_rmse)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(degrees, train_rmse_list, 'o-', label='训练 RMSE')
    ax.plot(degrees, test_rmse_list, 's-', label='测试 RMSE')
    ax.set_xlabel('模型复杂度（多项式次数）')
    ax.set_ylabel('RMSE')
    ax.set_title('偏差-方差权衡：训练误差 vs 测试误差')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / "figures/error_curves.png", dpi=150)
    plt.close()
    print("  -> 已保存 figures/error_curves.png")
    
    best_deg = degrees[np.argmin(test_rmse_list)]
    max_gap_deg = degrees[np.argmax(gap_list)]
    summary_df = pd.DataFrame({
        'degree': degrees,
        'train_RMSE': train_rmse_list,
        'test_RMSE': test_rmse_list,
        'generalization_gap': gap_list
    }).round(4)
    return best_deg, max_gap_deg, summary_df

def task_c(X_full, y_true_full, results_dir, n_repeats=10, train_size=100, noise_std=0.1):
    print("[Task C] 重复采样展示方差（多项式次数 2 vs 15）...")
    X_grid = np.linspace(0, 1, 200).reshape(-1, 1)
    y_true_grid = np.sin(2 * np.pi * X_grid.ravel()) + 0.2 * X_grid.ravel()
    degrees = [2, 15]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    low_var_stats = high_var_stats = None
    for idx, deg in enumerate(degrees):
        ax = axes[idx]
        all_preds = []
        for rep in range(n_repeats):
            idx_choice = np.random.choice(len(X_full), size=train_size, replace=False)
            X_train = X_full[idx_choice]
            y_train = y_true_full[idx_choice] + np.random.normal(0, noise_std, size=train_size)
            model = fit_polynomial(deg, X_train, y_train)
            y_pred = model.predict(X_grid)
            all_preds.append(y_pred)
            if rep < 10:
                ax.plot(X_grid, y_pred, color='steelblue', alpha=0.3, lw=1)
        ax.plot(X_grid, y_true_grid, 'k-', lw=2, label='真实函数')
        ax.scatter(X_train, y_train, s=10, alpha=0.5, label='示例训练点')
        ax.set_title(f'次数 {deg} {"（高方差）" if deg == 15 else "（低方差）"}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()
        ax.grid(alpha=0.3)
        std_pred = np.std(np.array(all_preds), axis=0)
        mean_std, max_std = np.mean(std_pred), np.max(std_pred)
        ax.text(0.05, 0.95, f"预测标准差均值 = {mean_std:.3f}\n预测标准差最大值 = {max_std:.3f}",
                transform=ax.transAxes, fontsize=9, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
        if deg == 2:
            low_var_stats = (mean_std, max_std)
        else:
            high_var_stats = (mean_std, max_std)
    fig.tight_layout()
    fig.savefig(results_dir / "figures/variance_demo.png", dpi=150)
    plt.close()
    print("  -> 已保存 figures/variance_demo.png")
    return low_var_stats, high_var_stats

def task_d(results_dir):
    print("[Task D] 比较 RMSE 和 MAE 对异常值的敏感性...")
    np.random.seed(42)
    n = 100
    y_true = np.random.normal(100, 10, n)
    y_pred_clean = y_true + np.random.normal(0, 5, n)
    y_pred_outlier = y_pred_clean.copy()
    y_pred_outlier[0] = y_true[0] + 500
    rmse_c, mae_c = calculate_rmse(y_true, y_pred_clean), calculate_mae(y_true, y_pred_clean)
    rmse_o, mae_o = calculate_rmse(y_true, y_pred_outlier), calculate_mae(y_true, y_pred_outlier)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(y_true, y_pred_clean, alpha=0.6, label='正常预测')
    axes[0].scatter(y_true[0], y_pred_outlier[0], color='red', s=100, label='异常值', marker='X')
    axes[0].plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--', label='完美拟合')
    axes[0].set_xlabel('真实值')
    axes[0].set_ylabel('预测值')
    axes[0].set_title('一个大的异常值的影响')
    axes[0].legend()
    metrics = ['RMSE', 'MAE']
    x = np.arange(2)
    width = 0.35
    axes[1].bar(x - width/2, [rmse_c, mae_c], width, label='正常')
    axes[1].bar(x + width/2, [rmse_o, mae_o], width, label='含异常值')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics)
    axes[1].set_ylabel('误差')
    axes[1].set_title('RMSE vs MAE 对异常值的敏感性')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(results_dir / "figures/loss_outlier_comparison.png", dpi=150)
    plt.close()
    print("  -> 已保存 figures/loss_outlier_comparison.png")
    return (rmse_c, mae_c, rmse_o, mae_o)

def write_summary(results_dir, best_deg, max_gap_deg, summary_df,
                  low_var_stats, high_var_stats, outlier_metrics):
    report_path = results_dir / "summary.md"
    with open(report_path, "w") as f:
        f.write("# 第十二周总结：偏差-方差权衡与损失函数\n\n")
        
        f.write("## Task A：候选模型对比（KNN，k=1,5,30）\n")
        f.write("- **k=1**：过拟合，训练误差极低但测试误差高，曲线剧烈波动。\n")
        f.write("- **k=5**：适度拟合，测试误差最低，曲线平滑且接近真实函数。\n")
        f.write("- **k=30**：欠拟合，曲线过于平滑，训练和测试误差都较高。\n")
        f.write("**推荐模型：k=5**（偏差-方差平衡最佳）\n\n")
        
        f.write("## Task B：复杂度–误差曲线（多项式回归）\n")
        f.write(f"测试 RMSE 在次数 **{best_deg}** 时最小。\n")
        f.write(f"泛化 gap 在次数 **{max_gap_deg}** 附近最大。\n\n")
        f.write("### 性能表（部分次数）\n")
        selected = [1, 4, 8, 12, 15, 18]
        display_df = summary_df[summary_df['degree'].isin(selected)]
        f.write(display_df.to_markdown(index=False) + "\n\n")
        
        f.write("## Task C：可视化方差\n")
        f.write("| 模型 | 预测标准差（均值） | 预测标准差（最大） |\n")
        f.write("|------|--------------------|--------------------|\n")
        f.write(f"| 次数 2（低方差） | {low_var_stats[0]:.3f} | {low_var_stats[1]:.3f} |\n")
        f.write(f"| 次数 15（高方差） | {high_var_stats[0]:.3f} | {high_var_stats[1]:.3f} |\n")
        f.write("\n> **高方差模型的危险，不是它不会拟合训练集，而是它对训练数据的采样波动过于敏感。**\n\n")
        
        f.write("## Task D：RMSE vs MAE 对异常值\n")
        f.write("| 场景 | RMSE | MAE |\n")
        f.write("|------|------|-----|\n")
        f.write(f"| 正常预测 | {outlier_metrics[0]:.2f} | {outlier_metrics[1]:.2f} |\n")
        f.write(f"| 含一个大的异常值 | {outlier_metrics[2]:.2f} | {outlier_metrics[3]:.2f} |\n\n")
        f.write("- RMSE 对大误差更敏感（平方放大），适合高风险场景；MAE 更稳健，适合含异常值的数据。\n\n")
        
        f.write("## 结论\n")
        f.write("1. 偏差-方差权衡真实存在：训练误差下降，测试误差先降后升。\n")
        f.write("2. 高方差模型对训练样本波动敏感，重复采样可揭示。\n")
        f.write("3. 选择损失函数应结合业务风险：RMSE 惩罚大错，MAE 容忍异常。\n")
    print(f"报告已保存至 {report_path}")

def main():
    base_dir = Path(__file__).parent
    results_dir = base_dir / "results"
    figures_dir = results_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("第十二周：偏差-方差可视化实验")
    print("="*60)
    
    X_full, y_full, y_true_full = generate_data(n_samples=200, noise_std=0.1)
    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.3, random_state=RANDOM_SEED)
    X_grid = np.linspace(0, 1, 500).reshape(-1, 1)
    y_true_grid = np.sin(2 * np.pi * X_grid.ravel()) + 0.2 * X_grid.ravel()
    
    task_a(X_train, y_train, X_test, y_test, X_grid, y_true_grid, results_dir)
    best_deg, max_gap_deg, summary_df = task_b(X_train, y_train, X_test, y_test, results_dir)
    low_var_stats, high_var_stats = task_c(X_full, y_true_full, results_dir, n_repeats=15, train_size=100)
    outlier_metrics = task_d(results_dir)
    write_summary(results_dir, best_deg, max_gap_deg, summary_df, low_var_stats, high_var_stats, outlier_metrics)
    
    print("\n✅ 所有任务完成！请查看 results/ 目录下的输出。")

if __name__ == "__main__":
    main()