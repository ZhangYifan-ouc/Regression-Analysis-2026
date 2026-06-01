"""
Module: milestone2.main
Purpose: The Pipeline & The Leakage-Free Generalization
"""

import sys
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.models import GradientDescentOLS
from utils.metrics import calculate_rmse, calculate_mae, calculate_mape
from utils.transformers import CustomStandardScaler


def setup_results_dir() -> Path:
    """动态清理 results/ 文件夹。"""
    results_dir = Path(__file__).parent.parent.parent / "results"
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def load_and_preprocess_data():
    """加载数据，分离特征和目标。"""
    data_path = Path(__file__).parent.parent.parent.parent / "week09" / "data" / "dirty_marketing.csv"
    df = pd.read_csv(data_path)

    # 处理分类变量：One-Hot 编码，drop_first=True
    df = pd.get_dummies(df, columns=['Region'], drop_first=True, dtype=int)

    # 分离特征和目标
    feature_cols = [col for col in df.columns if col != 'Sales']
    X = df[feature_cols].values.astype(float)
    y = df['Sales'].values

    return X, y, feature_cols


def bad_cross_validation(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Task 3: 危险的诱惑 - 制造数据泄露。

    全局预处理：先对全量数据进行标准化和缺失值填补，再做交叉验证。
    这会导致数据泄露！
    """
    print("\n" + "=" * 60)
    print("Task 3: 危险的诱惑 - 全局预处理（存在数据泄露）")
    print("=" * 60)

    # 全局缺失值填补（使用全量数据的均值）
    col_means = np.nanmean(X, axis=0)
    X_filled = np.where(np.isnan(X), col_means, X)

    # 全局标准化
    scaler = CustomStandardScaler()
    X_scaled = scaler.fit_transform(X_filled)

    # 添加截距列
    X_with_intercept = np.column_stack([np.ones(len(X_scaled)), X_scaled])

    # 5-Fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []
    mae_scores = []
    mape_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_with_intercept), start=1):
        X_train, X_val = X_with_intercept[train_idx], X_with_intercept[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = GradientDescentOLS(
            learning_rate=0.0001,
            tol=1e-5,
            max_iter=2000,
            gd_type="full_batch",
        ).fit(X_train, y_train)

        y_pred = model.predict(X_val)

        fold_rmse = calculate_rmse(y_val, y_pred)
        fold_mae = calculate_mae(y_val, y_pred)
        fold_mape = calculate_mape(y_val, y_pred)

        rmse_scores.append(fold_rmse)
        mae_scores.append(fold_mae)
        mape_scores.append(fold_mape)

        print(f"第 {fold} 折: RMSE={fold_rmse:.4f}, MAE={fold_mae:.4f}, MAPE={fold_mape:.2f}%")

    avg_rmse = np.mean(rmse_scores)
    avg_mae = np.mean(mae_scores)
    avg_mape = np.mean(mape_scores)

    print(f"\n平均 RMSE: {avg_rmse:.4f}")
    print(f"平均 MAE: {avg_mae:.4f}")
    print(f"平均 MAPE: {avg_mape:.2f}%")

    return {
        'rmse': avg_rmse,
        'mae': avg_mae,
        'mape': avg_mape,
        'rmse_scores': rmse_scores,
        'mae_scores': mae_scores,
        'mape_scores': mape_scores,
    }


def good_cross_validation(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Task 4: 坚不可摧的护城河 - 防泄漏流水线。

    在 CV 循环内部进行预处理，确保验证集完全未见过。
    """
    print("\n" + "=" * 60)
    print("Task 4: 坚不可摧的护城河 - 防泄漏流水线")
    print("=" * 60)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []
    mae_scores = []
    mape_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 1. 使用训练集的均值填补缺失值
        train_means = np.nanmean(X_train, axis=0)
        X_train_filled = np.where(np.isnan(X_train), train_means, X_train)
        X_val_filled = np.where(np.isnan(X_val), train_means, X_val)  # 使用训练集的均值！

        # 2. 使用训练集拟合 Scaler
        scaler = CustomStandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_filled)

        # 3. 使用训练集的 Scaler 转换验证集
        X_val_scaled = scaler.transform(X_val_filled)

        # 4. 添加截距列
        X_train_with_intercept = np.column_stack([np.ones(len(X_train_scaled)), X_train_scaled])
        X_val_with_intercept = np.column_stack([np.ones(len(X_val_scaled)), X_val_scaled])

        # 5. 训练模型
        model = GradientDescentOLS(
            learning_rate=0.0001,
            tol=1e-5,
            max_iter=2000,
            gd_type="full_batch",
        ).fit(X_train_with_intercept, y_train)

        # 6. 预测并评估
        y_pred = model.predict(X_val_with_intercept)

        fold_rmse = calculate_rmse(y_val, y_pred)
        fold_mae = calculate_mae(y_val, y_pred)
        fold_mape = calculate_mape(y_val, y_pred)

        rmse_scores.append(fold_rmse)
        mae_scores.append(fold_mae)
        mape_scores.append(fold_mape)

        print(f"第 {fold} 折: RMSE={fold_rmse:.4f}, MAE={fold_mae:.4f}, MAPE={fold_mape:.2f}%")

    avg_rmse = np.mean(rmse_scores)
    avg_mae = np.mean(mae_scores)
    avg_mape = np.mean(mape_scores)

    print(f"\n平均 RMSE: {avg_rmse:.4f}")
    print(f"平均 MAE: {avg_mae:.4f}")
    print(f"平均 MAPE: {avg_mape:.2f}%")

    return {
        'rmse': avg_rmse,
        'mae': avg_mae,
        'mape': avg_mape,
        'rmse_scores': rmse_scores,
        'mae_scores': mae_scores,
        'mape_scores': mape_scores,
    }


def plot_comparison(bad_results: dict, good_results: dict, results_dir: Path):
    """绘制对比柱状图。"""
    metrics = ['RMSE', 'MAE', 'MAPE']
    bad_values = [bad_results['rmse'], bad_results['mae'], bad_results['mape']]
    good_values = [good_results['rmse'], good_results['mae'], good_results['mape']]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, bad_values, width, label='With Leakage (Bad)', color='salmon')
    bars2 = ax.bar(x + width/2, good_values, width, label='Leakage-Free (Good)', color='steelblue')

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Values')
    ax.set_title('Comparison: With Data Leakage vs Leakage-Free CV')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(results_dir / "leakage_analysis.png", dpi=150)
    plt.close()


def main():
    """主函数。"""
    print("=" * 60)
    print("Milestone 2: The Pipeline & The Leakage-Free Generalization")
    print("=" * 60)

    # 设置结果目录
    results_dir = setup_results_dir()

    # 加载数据
    X, y, feature_cols = load_and_preprocess_data()
    print(f"\n数据加载完成: {X.shape[0]} 个样本, {X.shape[1]} 个特征")
    print(f"特征: {feature_cols}")

    # Task 3: 有数据泄露的交叉验证
    bad_results = bad_cross_validation(X, y)

    # Task 4: 无数据泄露的交叉验证
    good_results = good_cross_validation(X, y)

    # 绘制对比图
    plot_comparison(bad_results, good_results, results_dir)

    # 生成评估报告
    with open(results_dir / "evaluation_comparison.md", "w", encoding="utf-8") as f:
        f.write("# Milestone 2: 评估对比报告\n\n")
        f.write("## 实验概述\n\n")
        f.write("本实验对比了两种交叉验证方式：\n")
        f.write("1. **Task 3 (有数据泄露)**: 全局预处理后再做交叉验证\n")
        f.write("2. **Task 4 (无数据泄露)**: 在 CV 循环内部进行预处理\n\n")

        f.write("## 数据说明\n\n")
        f.write("- 数据来源: homework/week09/data/dirty_marketing.csv\n")
        f.write(f"- 样本数量: {len(y)}\n")
        f.write(f"- 特征数量: {X.shape[1]}\n")
        f.write(f"- 特征列表: {feature_cols}\n\n")

        f.write("## 评估指标对比\n\n")
        f.write("| 指标 | Task 3 (有泄露) | Task 4 (无泄露) | 差异 |\n")
        f.write("|------|-----------------|-----------------|------|\n")
        f.write(f"| RMSE | {bad_results['rmse']:.4f} | {good_results['rmse']:.4f} | {good_results['rmse'] - bad_results['rmse']:.4f} |\n")
        f.write(f"| MAE | {bad_results['mae']:.4f} | {good_results['mae']:.4f} | {good_results['mae'] - bad_results['mae']:.4f} |\n")
        f.write(f"| MAPE | {bad_results['mape']:.2f}% | {good_results['mape']:.2f}% | {good_results['mape'] - bad_results['mape']:.2f}% |\n\n")

        f.write("## 详细结果\n\n")
        f.write("### Task 3 (有数据泄露)\n")
        for i, (rmse, mae, mape) in enumerate(zip(bad_results['rmse_scores'], bad_results['mae_scores'], bad_results['mape_scores']), 1):
            f.write(f"- 第 {i} 折: RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%\n")

        f.write("\n### Task 4 (无数据泄露)\n")
        for i, (rmse, mae, mape) in enumerate(zip(good_results['rmse_scores'], good_results['mae_scores'], good_results['mape_scores']), 1):
            f.write(f"- 第 {i} 折: RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%\n")

        f.write("\n## 思考题\n\n")
        f.write("### 为什么 Task 3 的'好成绩'是致命的？\n\n")
        f.write("Task 3 存在数据泄露，原因如下：\n\n")
        f.write("1. **全局均值填补**: 使用全量数据（包括验证集）的均值填补缺失值，导致验证集信息泄露\n")
        f.write("2. **全局标准化**: 使用全量数据的均值和标准差进行标准化，验证集的分布信息被泄露到训练过程中\n\n")
        f.write("这种'好看'的成绩是致命的，因为：\n")
        f.write("- 模型在部署时会遇到真正的'未见过'的数据\n")
        f.write("- 真实世界的性能会比交叉验证的结果差很多\n")
        f.write("- 这种虚假的高分会误导业务决策\n\n")

        f.write("### 业务解读\n\n")
        f.write(f"基于 Task 4（无泄露）的结果：\n")
        f.write(f"- **MAE = {good_results['mae']:.2f} 元**: 模型上线后，每天的广告预算预测平均存在约 {good_results['mae']:.2f} 元的真实误差\n")
        f.write(f"- **MAPE = {good_results['mape']:.2f}%**: 预测误差约为实际值的 {good_results['mape']:.2f}%\n\n")
        f.write("应该给老板看 Task 4 的'差成绩'，因为这才是模型在真实世界中的预期表现。\n")

    print(f"\n结果已保存至: {results_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
