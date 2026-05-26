from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# 100% 复用你自己写的评估指标
from utils.metrics import calculate_rmse, calculate_mae

# 设置全局字体和样式，适合投屏
plt.rcParams["font.size"] = 12
plt.rcParams["figure.dpi"] = 150
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

# ==================== 工具函数 ====================
def init_results_dir(results_dir: Path, figures_dir: Path):
    """自动清理并初始化结果目录"""
    if results_dir.exists():
        import shutil
        shutil.rmtree(results_dir)
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    print(f"📁 结果目录已初始化: {results_dir}")
    print(f"📁 图表目录已初始化: {figures_dir}")

def true_function(x: np.ndarray) -> np.ndarray:
    """真实非线性函数：sin(x) + 线性趋势"""
    return np.sin(x) + 0.1 * x

def generate_data(n_samples: int = 100, random_state: int = 42) -> tuple:
    """生成带噪声的回归数据"""
    np.random.seed(random_state)
    x = np.linspace(0, 10, n_samples).reshape(-1, 1)
    y = true_function(x) + np.random.normal(0, 0.3, n_samples).reshape(-1, 1)
    return train_test_split(x, y, test_size=0.3, random_state=random_state)

# ==================== Task A: 候选模型比较 ====================
def run_model_complexity_demo(X_train, X_test, y_train, y_test, figures_dir: Path):
    """比较1、4、15次多项式模型"""
    print("\n[Stage 1] 比较候选多项式模型...")
    
    degrees = [1, 4, 15]
    colors = ["#ff6b6b", "#4ecdc4", "#ffd166"]
    labels = ["Degree 1 (欠拟合)", "Degree 4 (适中)", "Degree 15 (过拟合)"]
    
    plt.figure(figsize=(12, 8))
    
    # 绘制真实函数和数据点
    x_plot = np.linspace(0, 10, 1000).reshape(-1, 1)
    plt.plot(x_plot, true_function(x_plot), "k--", linewidth=2, label="真实函数")
    plt.scatter(X_train, y_train, c="blue", alpha=0.6, label="训练集")
    plt.scatter(X_test, y_test, c="red", alpha=0.6, label="测试集")
    
    results = []
    
    for degree, color, label in zip(degrees, colors, labels):
        # 创建多项式回归管道
        model = Pipeline([
            ("poly", PolynomialFeatures(degree=degree)),
            ("linear", LinearRegression())
        ])
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 预测
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_plot_pred = model.predict(x_plot)
        
        # 计算误差
        train_rmse = calculate_rmse(y_train, y_train_pred)
        test_rmse = calculate_rmse(y_test, y_test_pred)
        
        results.append({
            "degree": degree,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse
        })
        
        # 绘制拟合曲线
        plt.plot(x_plot, y_plot_pred, color=color, linewidth=2, 
                 label=f"{label}\nTrain RMSE: {train_rmse:.3f}\nTest RMSE: {test_rmse:.3f}")
    
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("不同复杂度多项式模型的拟合效果对比")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "candidate_models.png")
    plt.close()
    
    print("✅ 候选模型对比图已生成: candidate_models.png")
    return results

# ==================== Task B: 复杂度-误差曲线 ====================
def run_error_curves_demo(X_train, X_test, y_train, y_test, figures_dir: Path):
    """扫描1到18次多项式，绘制训练和测试误差曲线"""
    print("\n[Stage 2] 扫描模型复杂度...")
    
    max_degree = 18
    degrees = range(1, max_degree + 1)
    train_rmse_list = []
    test_rmse_list = []
    
    for degree in degrees:
        model = Pipeline([
            ("poly", PolynomialFeatures(degree=degree)),
            ("linear", LinearRegression())
        ])
        
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_rmse = calculate_rmse(y_train, y_train_pred)
        test_rmse = calculate_rmse(y_test, y_test_pred)
        
        train_rmse_list.append(train_rmse)
        test_rmse_list.append(test_rmse)
    
    # 绘制误差曲线
    plt.figure(figsize=(12, 8))
    plt.plot(degrees, train_rmse_list, "b-o", label="训练误差 (Train RMSE)")
    plt.plot(degrees, test_rmse_list, "r-o", label="测试误差 (Test RMSE)")
    
    # 标记测试误差最低点
    min_test_idx = np.argmin(test_rmse_list)
    min_degree = degrees[min_test_idx]
    min_rmse = test_rmse_list[min_test_idx]
    plt.scatter(min_degree, min_rmse, c="green", s=100, zorder=5, 
                label=f"最佳复杂度: degree={min_degree}, RMSE={min_rmse:.3f}")
    
    plt.xlabel("模型复杂度 (多项式次数)")
    plt.ylabel("RMSE")
    plt.title("模型复杂度与误差的关系")
    plt.xticks(degrees)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "error_curves.png")
    plt.close()
    
    # 生成成绩单表格
    results_df = pd.DataFrame({
        "degree": degrees,
        "train_rmse": train_rmse_list,
        "test_rmse": test_rmse_list,
        "generalization_gap": np.array(test_rmse_list) - np.array(train_rmse_list)
    })
    
    print("✅ 复杂度-误差曲线已生成: error_curves.png")
    return results_df

# ==================== Task C: 方差演示 ====================
def run_variance_demo(figures_dir: Path, n_repeats: int = 10):
    """通过重复抽样展示高方差模型的不稳定性"""
    print("\n[Stage 3] 演示模型方差...")
    
    degrees = [2, 15]
    x_plot = np.linspace(0, 10, 1000).reshape(-1, 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    axes = [ax1, ax2]
    
    summary_stats = []
    
    for i, degree in enumerate(degrees):
        ax = axes[i]
        predictions = []
        
        # 重复抽样训练10次
        for repeat in range(n_repeats):
            # 每次生成不同的训练集
            x = np.linspace(0, 10, 100).reshape(-1, 1)
            y = true_function(x) + np.random.normal(0, 0.3, 100).reshape(-1, 1)
            
            model = Pipeline([
                ("poly", PolynomialFeatures(degree=degree)),
                ("linear", LinearRegression())
            ])
            
            model.fit(x, y)
            y_pred = model.predict(x_plot)
            predictions.append(y_pred.flatten())
            
            ax.plot(x_plot, y_pred, alpha=0.5, linewidth=1)
        
        # 绘制真实函数
        ax.plot(x_plot, true_function(x_plot), "k--", linewidth=2, label="真实函数")
        
        # 计算预测标准差
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # 绘制置信区间
        ax.fill_between(x_plot.flatten(), 
                        mean_pred - 2*std_pred, 
                        mean_pred + 2*std_pred, 
                        alpha=0.2, color="gray")
        
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Degree = {degree} (重复{n_repeats}次)")
        ax.legend()
        
        # 统计信息
        mean_std = np.mean(std_pred)
        max_std = np.max(std_pred)
        summary_stats.append({
            "degree": degree,
            "mean_prediction_std": mean_std,
            "max_prediction_std": max_std
        })
    
    plt.tight_layout()
    plt.savefig(figures_dir / "variance_demo.png")
    plt.close()
    
    print("✅ 方差演示图已生成: variance_demo.png")
    return pd.DataFrame(summary_stats)

# ==================== Task D: RMSE与MAE对比 ====================
def run_loss_comparison_demo(figures_dir: Path):
    """比较异常值对RMSE和MAE的影响"""
    print("\n[Stage 4] 比较RMSE与MAE对异常值的敏感性...")
    
    # 构造干净预测
    np.random.seed(42)
    y_true = np.random.normal(10, 2, 100)
    y_pred_clean = y_true + np.random.normal(0, 0.5, 100)
    
    # 构造有一个大异常值的预测
    y_pred_outlier = y_pred_clean.copy()
    y_pred_outlier[0] = 100  # 制造一个非常大的误差
    
    # 计算指标
    clean_rmse = calculate_rmse(y_true, y_pred_clean)
    clean_mae = calculate_mae(y_true, y_pred_clean)
    outlier_rmse = calculate_rmse(y_true, y_pred_outlier)
    outlier_mae = calculate_mae(y_true, y_pred_outlier)
    
    # 绘制对比图
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(y_true))
    plt.scatter(x, y_true, c="blue", alpha=0.6, label="真实值")
    plt.scatter(x, y_pred_clean, c="green", alpha=0.6, label="干净预测")
    plt.scatter(x[0], y_pred_outlier[0], c="red", s=100, zorder=5, label="异常值预测")
    
    plt.xlabel("样本索引")
    plt.ylabel("值")
    plt.title("异常值对预测的影响")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "loss_outlier_comparison.png")
    plt.close()
    
    # 生成对比表格
    results_df = pd.DataFrame({
        "场景": ["干净预测", "含一个大异常值"],
        "RMSE": [clean_rmse, outlier_rmse],
        "MAE": [clean_mae, outlier_mae],
        "RMSE变化倍数": [1.0, outlier_rmse / clean_rmse],
        "MAE变化倍数": [1.0, outlier_mae / clean_mae]
    })
    
    print("✅ 损失函数对比图已生成: loss_outlier_comparison.png")
    return results_df

# ==================== 生成总结报告 ====================
def write_summary_report(results_dir: Path, 
                         candidate_results, 
                         error_curves_df, 
                         variance_stats_df, 
                         loss_comparison_df):
    """自动生成完整的总结报告"""
    print("\n[Stage 5] 生成总结报告...")
    
    # 找到最佳复杂度
    min_test_idx = error_curves_df["test_rmse"].idxmin()
    best_degree = error_curves_df.loc[min_test_idx, "degree"]
    best_test_rmse = error_curves_df.loc[min_test_idx, "test_rmse"]
    
    # 找到泛化gap最大的复杂度
    max_gap_idx = error_curves_df["generalization_gap"].idxmax()
    max_gap_degree = error_curves_df.loc[max_gap_idx, "degree"]
    max_gap = error_curves_df.loc[max_gap_idx, "generalization_gap"]
    
    report_content = f"""# 偏差-方差权衡可视化实验报告

## 1. 候选模型比较
我们比较了1次、4次和15次多项式模型的拟合效果：

| 模型复杂度 | 训练RMSE | 测试RMSE | 拟合状态 |
|------------|----------|----------|----------|
| Degree 1 | {candidate_results[0]['train_rmse']:.3f} | {candidate_results[0]['test_rmse']:.3f} | 欠拟合 |
| Degree 4 | {candidate_results[1]['train_rmse']:.3f} | {candidate_results[1]['test_rmse']:.3f} | 适中 |
| Degree 15 | {candidate_results[2]['train_rmse']:.3f} | {candidate_results[2]['test_rmse']:.3f} | 过拟合 |

### 问题回答
1. **谁最像欠拟合？** Degree 1模型最像欠拟合，它的训练误差和测试误差都很高，无法捕捉数据中的非线性关系。
2. **谁最像过拟合？** Degree 15模型最像过拟合，它的训练误差非常低，但测试误差显著升高。
3. **如果今天必须选一个上线，你会先押谁？为什么？** 我会选择Degree 4模型，因为它在训练误差和测试误差之间取得了最好的平衡，泛化能力最强。

## 2. 复杂度-误差曲线
我们扫描了从1到18次的所有多项式模型，得到了完整的误差曲线：

### 关键发现
- 测试误差最低的复杂度是 **degree = {best_degree}**，对应的测试RMSE为 **{best_test_rmse:.3f}**
- 泛化gap最大的复杂度是 **degree = {max_gap_degree}**，对应的gap为 **{max_gap:.3f}**
- 随着模型复杂度增加，训练误差持续下降，但测试误差先下降后上升，呈现U型曲线

### 为什么训练误差最低的模型不一定是最好的模型？
训练误差最低的模型通常是复杂度最高的模型，它过度拟合了训练集中的噪声和随机波动，而不是学习到了数据的真实规律。这种模型在新的、未见过的数据上表现会很差，泛化能力弱。

## 3. 方差演示
我们通过重复抽样10次，比较了低复杂度和高复杂度模型的稳定性：

| 模型复杂度 | 平均预测标准差 | 最大预测标准差 |
|------------|----------------|----------------|
| Degree 2 | {variance_stats_df.loc[0, 'mean_prediction_std']:.3f} | {variance_stats_df.loc[0, 'max_prediction_std']:.3f} |
| Degree 15 | {variance_stats_df.loc[1, 'mean_prediction_std']:.3f} | {variance_stats_df.loc[1, 'max_prediction_std']:.3f} |

### 一句话结论
> high variance model 的危险，不是它不会拟合训练集，而是它对 **训练数据的微小变化** 过于敏感。

## 4. RMSE与MAE对比
我们比较了异常值对RMSE和MAE的影响：

| 场景 | RMSE | MAE | RMSE变化倍数 | MAE变化倍数 |
|------|------|-----|--------------|-------------|
| 干净预测 | {loss_comparison_df.loc[0, 'RMSE']:.3f} | {loss_comparison_df.loc[0, 'MAE']:.3f} | 1.00 | 1.00 |
| 含一个大异常值 | {loss_comparison_df.loc[1, 'RMSE']:.3f} | {loss_comparison_df.loc[1, 'MAE']:.3f} | {loss_comparison_df.loc[1, 'RMSE变化倍数']:.2f} | {loss_comparison_df.loc[1, 'MAE变化倍数']:.2f} |

### 业务解释
1. **为什么RMSE更容易被大错拉高？** 因为RMSE对误差取了平方，大误差会被放大很多倍，对最终结果的影响远大于小误差。
2. **如果线上系统偶尔一次大错的代价极高，你更想看哪个指标？** 我会更关注RMSE，因为它能更好地反映大错误的影响，帮助我们识别和避免灾难性错误。
3. **如果数据天然包含较多异常值，你会不会重新考虑指标选择？** 会的，我会优先使用MAE，因为它对异常值更稳健，不会被少数极端值过度影响，能更真实地反映模型的整体表现。

## 5. 核心结论
### 三条最重要的结论
1. 模型复杂度与泛化能力之间存在权衡：过于简单的模型会欠拟合，过于复杂的模型会过拟合。
2. 高方差模型的特点是对训练数据的微小变化非常敏感，在不同的训练集上会得到差异很大的结果。
3. RMSE和MAE代表了不同的风险偏好：RMSE更关注大错误，MAE更关注平均错误，应根据业务场景选择合适的指标。

### 最能代表过拟合的图
我认为 **`variance_demo.png`** 最能代表"过拟合不是抽象概念，而是可见现象"。这张图清晰地展示了高复杂度模型在不同训练集上的拟合曲线差异巨大，直观地体现了过拟合模型的不稳定性和高方差特性。

### 指标选择判断
- **什么时候更愿意报告RMSE？** 当大错误的代价远高于小错误时，例如金融风控、医疗诊断等场景，我们需要特别关注和避免大的预测错误。
- **什么时候更愿意报告MAE？** 当数据中存在较多异常值，或者我们更关心模型的平均表现时，例如销量预测、用户行为分析等场景。

### 与下一周的连接
如果模型复杂度过高会带来high variance，那么下一步我们自然会想到正则化（Ridge / Lasso），因为正则化通过给模型参数添加惩罚项，限制了模型的复杂度，从而降低了模型的方差，提高了泛化能力。正则化是解决过拟合问题最常用和最有效的方法之一。
"""

    with open(results_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print("✅ 总结报告已生成: summary.md")

# ==================== 主函数 ====================
def main():
    # 定义路径
    base_dir = Path(__file__).parent
    results_dir = base_dir / "results"
    figures_dir = results_dir / "figures"
    
    # 1. 初始化结果目录
    init_results_dir(results_dir, figures_dir)
    
    # 2. 生成数据
    X_train, X_test, y_train, y_test = generate_data()
    print(f"📊 数据生成完成，训练集: {len(X_train)}个样本，测试集: {len(X_test)}个样本")
    
    # 3. 运行所有实验
    candidate_results = run_model_complexity_demo(X_train, X_test, y_train, y_test, figures_dir)
    error_curves_df = run_error_curves_demo(X_train, X_test, y_train, y_test, figures_dir)
    variance_stats_df = run_variance_demo(figures_dir)
    loss_comparison_df = run_loss_comparison_demo(figures_dir)
    
    # 4. 生成总结报告
    write_summary_report(results_dir, candidate_results, error_curves_df, variance_stats_df, loss_comparison_df)
    
    print("\n" + "="*50)
    print("🎉 所有实验完成！所有图表和报告已保存到 results/ 文件夹")
    print("="*50)

if __name__ == "__main__":
    main()