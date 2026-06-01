"""
Module: week11.main
Purpose: Dual Inference Sprint — Synthetic-to-Real Regression Workflow
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
from utils.transformers import CustomStandardScaler, CustomImputer
from utils.diagnostics import calculate_vif, check_multicollinearity


def setup_results_dir() -> Path:
    """动态清理 results/ 文件夹。"""
    results_dir = Path(__file__).parent / "results"
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def plot_synthetic_data_analysis(df: pd.DataFrame, results_dir: Path):
    """生成模拟数据的分析图表。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. TV_Budget vs Sales
    axes[0, 0].scatter(df['TV_Budget'], df['Sales'], alpha=0.5, s=20)
    axes[0, 0].set_xlabel('TV_Budget')
    axes[0, 0].set_ylabel('Sales')
    axes[0, 0].set_title('TV_Budget vs Sales')

    # 2. Radio_Budget vs Sales
    axes[0, 1].scatter(df['Radio_Budget'], df['Sales'], alpha=0.5, s=20, color='orange')
    axes[0, 1].set_xlabel('Radio_Budget')
    axes[0, 1].set_ylabel('Sales')
    axes[0, 1].set_title('Radio_Budget vs Sales')

    # 3. Online_Budget vs Sales
    axes[1, 0].scatter(df['Online_Budget'], df['Sales'], alpha=0.5, s=20, color='green')
    axes[1, 0].set_xlabel('Online_Budget')
    axes[1, 0].set_ylabel('Sales')
    axes[1, 0].set_title('Online_Budget vs Sales')

    # 4. Season distribution
    season_counts = df['Season'].value_counts()
    axes[1, 1].bar(season_counts.index, season_counts.values, color=['steelblue', 'orange', 'green', 'red'])
    axes[1, 1].set_xlabel('Season')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Season Distribution')

    plt.tight_layout()
    plt.savefig(results_dir / "synthetic_data_analysis.png", dpi=150)
    plt.close()


def plot_coefficient_comparison(synthetic_results: dict, kaggle_results: dict, results_dir: Path):
    """生成系数对比图。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 模拟数据系数
    features_syn = synthetic_results['feature_names'][:3]  # 只取前3个连续特征
    coefs_syn = synthetic_results['coefficients'][:3]
    axes[0].barh(features_syn, coefs_syn, color='steelblue')
    axes[0].set_xlabel('Coefficient Value')
    axes[0].set_title('Synthetic Data: Feature Coefficients')
    axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    # Kaggle 数据系数（取前6个特征）
    features_kag = kaggle_results['feature_names'][:6]
    coefs_kag = kaggle_results['coefficients'][:6]
    axes[1].barh(features_kag, coefs_kag, color='coral')
    axes[1].set_xlabel('Coefficient Value')
    axes[1].set_title('Kaggle Data: Feature Coefficients')
    axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(results_dir / "coefficient_comparison.png", dpi=150)
    plt.close()


def plot_kaggle_data_exploration(df: pd.DataFrame, results_dir: Path):
    """生成 Kaggle 数据的探索性分析图表。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Hours_Studied vs Exam_Score
    axes[0, 0].scatter(df['Hours_Studied'], df['Exam_Score'], alpha=0.3, s=10)
    axes[0, 0].set_xlabel('Hours_Studied')
    axes[0, 0].set_ylabel('Exam_Score')
    axes[0, 0].set_title('Hours_Studied vs Exam_Score')

    # 2. Attendance vs Exam_Score
    axes[0, 1].scatter(df['Attendance'], df['Exam_Score'], alpha=0.3, s=10, color='orange')
    axes[0, 1].set_xlabel('Attendance')
    axes[0, 1].set_ylabel('Exam_Score')
    axes[0, 1].set_title('Attendance vs Exam_Score')

    # 3. Exam_Score distribution
    axes[1, 0].hist(df['Exam_Score'], bins=30, color='steelblue', alpha=0.7)
    axes[1, 0].set_xlabel('Exam_Score')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Exam_Score Distribution')

    # 4. Parental_Involvement vs Exam_Score
    df.boxplot(column='Exam_Score', by='Parental_Involvement', ax=axes[1, 1])
    axes[1, 1].set_xlabel('Parental_Involvement')
    axes[1, 1].set_ylabel('Exam_Score')
    axes[1, 1].set_title('Parental_Involvement vs Exam_Score')
    plt.suptitle('')

    plt.tight_layout()
    plt.savefig(results_dir / "kaggle_data_exploration.png", dpi=150)
    plt.close()


def plot_metrics_comparison(synthetic_results: dict, kaggle_results: dict, results_dir: Path):
    """生成指标对比图。"""
    metrics = ['RMSE', 'MAE', 'MAPE']
    synthetic_values = [synthetic_results['rmse'], synthetic_results['mae'], synthetic_results['mape']]
    kaggle_values = [kaggle_results['rmse'], kaggle_results['mae'], kaggle_results['mape']]

    # 归一化以便对比
    synthetic_norm = [v / max(synthetic_values) for v in synthetic_values]
    kaggle_norm = [v / max(kaggle_values) for v in kaggle_values]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, synthetic_norm, width, label='Synthetic Data', color='steelblue')
    bars2 = ax.bar(x + width/2, kaggle_norm, width, label='Kaggle Data', color='coral')

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Normalized Values')
    ax.set_title('Metrics Comparison: Synthetic vs Kaggle')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    plt.tight_layout()
    plt.savefig(results_dir / "metrics_comparison.png", dpi=150)
    plt.close()


# =====================================================================
# Task A: 模拟数据生成与分析
# =====================================================================

def generate_synthetic_data(n_samples: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    生成模拟回归数据（广告预算与销售额）。

    DGP (Data Generating Process):
    - TV_Budget: 广告预算（正向影响）
    - Radio_Budget: 广告预算（正向影响）
    - Online_Budget: 在线广告预算（正向影响，与 TV 高度相关）
    - Season: 季节（类别变量，影响销售额）
    - Sales: 销售额 = 3*TV + 2*Radio + 1.5*Online + 50*Is_Peak + noise
    """
    np.random.seed(seed)

    # 生成特征
    TV_Budget = np.random.uniform(50, 300, n_samples)
    Radio_Budget = np.random.uniform(20, 150, n_samples)

    # Online_Budget 与 TV_Budget 高度相关（共线性）
    Online_Budget = 0.8 * TV_Budget + np.random.normal(0, 20, n_samples)
    Online_Budget = np.clip(Online_Budget, 10, 400)

    # 季节变量
    Season = np.random.choice(['Spring', 'Summer', 'Autumn', 'Winter'], n_samples)
    Is_Peak = np.where(np.isin(Season, ['Summer', 'Winter']), 1, 0)

    # 生成目标变量
    Sales = (
        3.0 * TV_Budget
        + 2.0 * Radio_Budget
        + 1.5 * Online_Budget
        + 50 * Is_Peak
        + np.random.normal(0, 30, n_samples)
    )

    df = pd.DataFrame({
        'TV_Budget': TV_Budget,
        'Radio_Budget': Radio_Budget,
        'Online_Budget': Online_Budget,
        'Season': Season,
        'Sales': Sales
    })

    # 添加缺失值（约 5%）
    for col in ['TV_Budget', 'Radio_Budget', 'Online_Budget']:
        mask = np.random.random(n_samples) < 0.05
        df.loc[mask, col] = np.nan

    # 添加异常值（约 2%）
    outlier_mask = np.random.random(n_samples) < 0.02
    df.loc[outlier_mask, 'TV_Budget'] = df['TV_Budget'].quantile(0.99) * np.random.uniform(2, 4, outlier_mask.sum())

    return df


def run_synthetic_task(df: pd.DataFrame, results_dir: Path) -> dict:
    """Task A: 在模拟数据上完成回归分析。"""
    print("\n" + "=" * 60)
    print("Task A: 模拟数据分析")
    print("=" * 60)

    # 1. 数据预处理
    print("\n--- 数据预处理 ---")

    # One-Hot 编码
    df_encoded = pd.get_dummies(df, columns=['Season'], drop_first=True, dtype=float)

    # 分离特征和目标
    feature_cols = [col for col in df_encoded.columns if col != 'Sales']
    X = df_encoded[feature_cols].values
    y = df_encoded['Sales'].values

    print(f"特征: {feature_cols}")
    print(f"样本数: {len(y)}")

    # 2. VIF 诊断（先填补缺失值）
    print("\n--- VIF 诊断 ---")
    imputer_temp = CustomImputer(strategy='mean')
    X_filled_temp = imputer_temp.fit_transform(X)
    vif_result = check_multicollinearity(X_filled_temp, feature_cols)
    for name, vif in zip(feature_cols, vif_result['vif_values']):
        status = "⚠️ 高" if vif > 10 else "✓ 正常"
        print(f"  {name}: VIF = {vif:.4f} [{status}]")

    if vif_result['has_multicollinearity']:
        print("⚠️ 检测到严重多重共线性问题!")

    # 3. 无泄露交叉验证
    print("\n--- 5-Fold 交叉验证（无泄露）---")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []
    mae_scores = []
    mape_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 使用训练集的参数填补缺失值
        imputer = CustomImputer(strategy='mean')
        X_train_filled = imputer.fit_transform(X_train)
        X_val_filled = imputer.transform(X_val)

        # 使用训练集拟合 Scaler
        scaler = CustomStandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_filled)
        X_val_scaled = scaler.transform(X_val_filled)

        # 添加截距列
        X_train_with_intercept = np.column_stack([np.ones(len(X_train_scaled)), X_train_scaled])
        X_val_with_intercept = np.column_stack([np.ones(len(X_val_scaled)), X_val_scaled])

        # 训练模型
        model = GradientDescentOLS(
            learning_rate=0.0001,
            tol=1e-5,
            max_iter=2000,
            gd_type="full_batch",
        ).fit(X_train_with_intercept, y_train)

        # 预测并评估
        y_pred = model.predict(X_val_with_intercept)
        rmse_scores.append(calculate_rmse(y_val, y_pred))
        mae_scores.append(calculate_mae(y_val, y_pred))
        mape_scores.append(calculate_mape(y_val, y_pred))

        print(f"  第 {fold} 折: RMSE={rmse_scores[-1]:.4f}, MAE={mae_scores[-1]:.4f}, MAPE={mape_scores[-1]:.2f}%")

    avg_rmse = np.mean(rmse_scores)
    avg_mae = np.mean(mae_scores)
    avg_mape = np.mean(mape_scores)

    print(f"\n平均 RMSE: {avg_rmse:.4f}")
    print(f"平均 MAE: {avg_mae:.4f}")
    print(f"平均 MAPE: {avg_mape:.2f}%")

    # 4. 获取系数（使用全量数据训练）
    imputer = CustomImputer(strategy='mean')
    X_filled = imputer.fit_transform(X)
    scaler = CustomStandardScaler()
    X_scaled = scaler.fit_transform(X_filled)
    X_with_intercept = np.column_stack([np.ones(len(X_scaled)), X_scaled])
    model = GradientDescentOLS(learning_rate=0.0001, max_iter=2000).fit(X_with_intercept, y)
    coefficients = model.coef_[1:]  # 去掉截距

    return {
        'rmse': avg_rmse,
        'mae': avg_mae,
        'mape': avg_mape,
        'vif_values': vif_result['vif_values'],
        'feature_names': feature_cols,
        'coefficients': coefficients,
    }


# =====================================================================
# Task B: Kaggle 真实数据分析
# =====================================================================

def load_kaggle_data() -> pd.DataFrame:
    """加载 Kaggle 数据。"""
    data_path = Path(__file__).parent / "data" / "StudentPerformanceFactors.csv"
    df = pd.read_csv(data_path)
    return df


def run_kaggle_task(df: pd.DataFrame, results_dir: Path) -> dict:
    """Task B: 在 Kaggle 数据上完成回归分析。"""
    print("\n" + "=" * 60)
    print("Task B: Kaggle 真实数据分析")
    print("=" * 60)

    # 1. 数据探索
    print("\n--- 数据探索 ---")
    print(f"数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    print(f"\n缺失值:\n{df.isnull().sum()}")

    # 2. 数据预处理
    print("\n--- 数据预处理 ---")

    # 分离数值和类别特征
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    target_col = 'Exam_Score'

    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)

    print(f"数值特征: {numeric_cols}")
    print(f"类别特征: {categorical_cols}")

    # One-Hot 编码
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=float)

    # 分离特征和目标
    feature_cols = [col for col in df_encoded.columns if col != target_col]
    X = df_encoded[feature_cols].values
    y = df_encoded[target_col].values

    print(f"编码后特征数: {len(feature_cols)}")

    # 3. VIF 诊断（只对数值特征）
    print("\n--- VIF 诊断（数值特征）---")
    X_numeric = df[numeric_cols].values
    vif_result = check_multicollinearity(X_numeric, numeric_cols)
    for name, vif in zip(numeric_cols, vif_result['vif_values']):
        status = "⚠️ 高" if vif > 10 else "✓ 正常"
        print(f"  {name}: VIF = {vif:.4f} [{status}]")

    # 4. 无泄露交叉验证
    print("\n--- 5-Fold 交叉验证（无泄露）---")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []
    mae_scores = []
    mape_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 使用训练集的参数填补缺失值
        imputer = CustomImputer(strategy='mean')
        X_train_filled = imputer.fit_transform(X_train)
        X_val_filled = imputer.transform(X_val)

        # 使用训练集拟合 Scaler
        scaler = CustomStandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_filled)
        X_val_scaled = scaler.transform(X_val_filled)

        # 添加截距列
        X_train_with_intercept = np.column_stack([np.ones(len(X_train_scaled)), X_train_scaled])
        X_val_with_intercept = np.column_stack([np.ones(len(X_val_scaled)), X_val_scaled])

        # 训练模型
        model = GradientDescentOLS(
            learning_rate=0.0001,
            tol=1e-5,
            max_iter=2000,
            gd_type="full_batch",
        ).fit(X_train_with_intercept, y_train)

        # 预测并评估
        y_pred = model.predict(X_val_with_intercept)
        rmse_scores.append(calculate_rmse(y_val, y_pred))
        mae_scores.append(calculate_mae(y_val, y_pred))
        mape_scores.append(calculate_mape(y_val, y_pred))

        print(f"  第 {fold} 折: RMSE={rmse_scores[-1]:.4f}, MAE={mae_scores[-1]:.4f}, MAPE={mape_scores[-1]:.2f}%")

    avg_rmse = np.mean(rmse_scores)
    avg_mae = np.mean(mae_scores)
    avg_mape = np.mean(mape_scores)

    print(f"\n平均 RMSE: {avg_rmse:.4f}")
    print(f"平均 MAE: {avg_mae:.4f}")
    print(f"平均 MAPE: {avg_mape:.2f}%")

    # 5. 获取系数
    imputer = CustomImputer(strategy='mean')
    X_filled = imputer.fit_transform(X)
    scaler = CustomStandardScaler()
    X_scaled = scaler.fit_transform(X_filled)
    X_with_intercept = np.column_stack([np.ones(len(X_scaled)), X_scaled])
    model = GradientDescentOLS(learning_rate=0.0001, max_iter=2000).fit(X_with_intercept, y)
    coefficients = model.coef_[1:]  # 去掉截距

    return {
        'rmse': avg_rmse,
        'mae': avg_mae,
        'mape': avg_mape,
        'vif_values': vif_result['vif_values'],
        'feature_names': feature_cols,
        'numeric_features': numeric_cols,
        'coefficients': coefficients,
        'data_shape': df.shape,
    }


# =====================================================================
# 报告生成
# =====================================================================

def write_synthetic_report(results: dict, dgp_info: dict, results_dir: Path):
    """生成模拟数据报告。"""
    with open(results_dir / "synthetic_report.md", "w", encoding="utf-8") as f:
        f.write("# Task A: 模拟数据分析报告\n\n")
        f.write("## 数据生成机制 (DGP)\n\n")
        f.write(dgp_info['description'])
        f.write("\n\n")

        f.write("## VIF 诊断结果\n\n")
        f.write("| 特征 | VIF | 状态 |\n")
        f.write("|------|-----|------|\n")
        for name, vif in zip(results['feature_names'], results['vif_values']):
            status = "⚠️ 高" if vif > 10 else "✓ 正常"
            f.write(f"| {name} | {vif:.4f} | {status} |\n")

        f.write("\n## 模型系数\n\n")
        f.write("| 特征 | 系数 | 预期方向 |\n")
        f.write("|------|------|----------|\n")
        for name, coef in zip(results['feature_names'], results['coefficients']):
            expected = dgp_info['expected_directions'].get(name, '未知')
            f.write(f"| {name} | {coef:.4f} | {expected} |\n")

        f.write("\n## 交叉验证结果\n\n")
        f.write(f"- 平均 RMSE: {results['rmse']:.4f}\n")
        f.write(f"- 平均 MAE: {results['mae']:.4f}\n")
        f.write(f"- 平均 MAPE: {results['mape']:.2f}%\n\n")

        f.write("## 推测分析\n\n")
        f.write("### 系数方向一致性\n")
        f.write("对比模型识别出的系数方向与 DGP 设定：\n\n")
        for name, coef in zip(results['feature_names'], results['coefficients']):
            expected_sign = dgp_info['expected_signs'].get(name, 1)
            actual_sign = 1 if coef > 0 else -1
            consistent = "✓ 一致" if expected_sign == actual_sign else "✗ 不一致"
            f.write(f"- {name}: 预期{'正' if expected_sign > 0 else '负'}向, 实际{'正' if actual_sign > 0 else '负'}向 [{consistent}]\n")

        f.write("\n### 共线性影响\n")
        f.write("由于 Online_Budget 与 TV_Budget 高度相关，可能导致系数估计不稳定。\n")


def write_kaggle_report(results: dict, results_dir: Path):
    """生成 Kaggle 数据报告。"""
    with open(results_dir / "kaggle_report.md", "w", encoding="utf-8") as f:
        f.write("# Task B: Kaggle 真实数据分析报告\n\n")
        f.write("## 数据集信息\n\n")
        f.write("- **数据集名称**: Student Performance Factors\n")
        f.write("- **预测目标**: Exam_Score（考试成绩）\n")
        f.write(f"- **数据形状**: {results['data_shape']}\n")
        f.write("- **业务含义**: 每行代表一个学生的学习特征和考试成绩\n\n")

        f.write("## VIF 诊断结果\n\n")
        f.write("| 特征 | VIF | 状态 |\n")
        f.write("|------|-----|------|\n")
        for name, vif in zip(results['numeric_features'], results['vif_values']):
            status = "⚠️ 高" if vif > 10 else "✓ 正常"
            f.write(f"| {name} | {vif:.4f} | {status} |\n")

        f.write("\n## 交叉验证结果\n\n")
        f.write(f"- 平均 RMSE: {results['rmse']:.4f}\n")
        f.write(f"- 平均 MAE: {results['mae']:.4f}\n")
        f.write(f"- 平均 MAPE: {results['mape']:.2f}%\n\n")

        f.write("## 业务解读\n\n")
        f.write(f"- **MAE = {results['mae']:.2f}**: 模型预测的考试成绩平均误差约 {results['mae']:.2f} 分\n")
        f.write(f"- **MAPE = {results['mape']:.2f}%**: 预测误差约为实际成绩的 {results['mape']:.2f}%\n\n")

        f.write("## 风险分析\n\n")
        f.write("1. **数据质量**: 部分字段存在缺失值，需要谨慎处理\n")
        f.write("2. **特征工程**: 类别变量编码可能丢失部分信息\n")
        f.write("3. **模型局限**: 线性模型可能无法捕捉非线性关系\n")


def write_summary_comparison(synthetic_results: dict, kaggle_results: dict, results_dir: Path):
    """生成对照总结报告。"""
    with open(results_dir / "summary_comparison.md", "w", encoding="utf-8") as f:
        f.write("# Task C: 模拟数据与真实数据对照总结\n\n")

        f.write("## 指标对比\n\n")
        f.write("| 指标 | 模拟数据 | Kaggle 数据 |\n")
        f.write("|------|----------|-------------|\n")
        f.write(f"| RMSE | {synthetic_results['rmse']:.4f} | {kaggle_results['rmse']:.4f} |\n")
        f.write(f"| MAE | {synthetic_results['mae']:.4f} | {kaggle_results['mae']:.4f} |\n")
        f.write(f"| MAPE | {synthetic_results['mape']:.2f}% | {kaggle_results['mape']:.2f}% |\n\n")

        f.write("## 关键讨论\n\n")

        f.write("### 1. 为什么模拟数据的推测更容易？\n")
        f.write("在模拟数据中，我们知道真实的 DGP（数据生成机制），因此：\n")
        f.write("- 我们知道哪些变量应该正向影响目标\n")
        f.write("- 我们知道哪些变量是高度相关的\n")
        f.write("- 我们可以验证模型是否识别出了正确的模式\n\n")

        f.write("### 2. 为什么真实数据的解释更困难？\n")
        f.write("在真实数据中：\n")
        f.write("- 我们不知道真实的 DGP\n")
        f.write("- 变量之间的关系可能是非线性的\n")
        f.write("- 可能存在未观测的混淆变量\n")
        f.write("- 数据质量问题（缺失值、异常值）更复杂\n\n")

        f.write("### 3. 共线性、缺失值、异常值的影响\n")
        f.write("| 问题 | 模拟数据 | 真实数据 |\n")
        f.write("|------|----------|----------|\n")
        f.write("| 共线性 | 明确构造（Online 与 TV）| 需要诊断发现 |\n")
        f.write("| 缺失值 | 人为添加 5% | 真实缺失 |\n")
        f.write("| 异常值 | 人为添加 2% | 真实异常 |\n\n")

        f.write("### 4. 为什么无泄露交叉验证在真实数据上尤其重要？\n")
        f.write("在真实数据中：\n")
        f.write("- 数据量通常更小，泄露的影响更显著\n")
        f.write("- 我们无法验证模型是否真的学到了正确的模式\n")
        f.write("- 无泄露评估是唯一可信的泛化能力指标\n\n")

        f.write("### 5. utils/ 组件的价值\n")
        f.write("复用自己维护的 utils/ 组件带来了以下好处：\n")
        f.write("- **一致性**: 两个任务使用相同的预处理逻辑\n")
        f.write("- **可复用性**: 新数据集可以快速应用相同流程\n")
        f.write("- **可解释性**: 清楚知道每一步做了什么\n")
        f.write("- **可调试性**: 问题定位更容易\n")


def main():
    """主函数。"""
    print("=" * 60)
    print("Week 11: Dual Inference Sprint")
    print("=" * 60)

    # 设置结果目录
    results_dir = setup_results_dir()

    # Task A: 模拟数据
    print("\n" + "=" * 60)
    print("生成模拟数据...")
    print("=" * 60)
    synthetic_df = generate_synthetic_data(n_samples=500)

    # 保存模拟数据
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    synthetic_df.to_csv(data_dir / "synthetic_regression.csv", index=False)
    print(f"模拟数据已保存至: {data_dir / 'synthetic_regression.csv'}")

    # DGP 信息
    dgp_info = {
        'description': """
本模拟数据设计了一个"广告预算与销售额"的业务场景：

**数据生成公式**：
Sales = 3.0 × TV_Budget + 2.0 × Radio_Budget + 1.5 × Online_Budget + 50 × Is_Peak + ε

其中：
- TV_Budget: TV 广告预算，均匀分布 U(50, 300)
- Radio_Budget: Radio 广告预算，均匀分布 U(20, 150)
- Online_Budget: 在线广告预算，与 TV_Budget 高度相关（0.8 × TV + noise）
- Season: 季节（Spring/Summer/Autumn/Winter），Summer 和 Winter 为旺季
- Is_Peak: 是否旺季（1 或 0）
- ε: 噪声项，N(0, 30)

**人为添加的问题**：
- 缺失值：约 5%
- 异常值：约 2%
- 共线性：Online_Budget 与 TV_Budget 高度相关
""",
        'expected_directions': {
            'TV_Budget': '正向',
            'Radio_Budget': '正向',
            'Online_Budget': '正向',
            'Season_Summer': '正向',
            'Season_Winter': '正向',
            'Season_Autumn': '负向或无影响',
        },
        'expected_signs': {
            'TV_Budget': 1,
            'Radio_Budget': 1,
            'Online_Budget': 1,
            'Season_Summer': 1,
            'Season_Winter': 1,
            'Season_Autumn': -1,
        }
    }

    synthetic_results = run_synthetic_task(synthetic_df, results_dir)
    write_synthetic_report(synthetic_results, dgp_info, results_dir)

    # 生成模拟数据图表
    print("\n生成模拟数据图表...")
    plot_synthetic_data_analysis(synthetic_df, results_dir)

    # Task B: Kaggle 数据
    print("\n" + "=" * 60)
    print("加载 Kaggle 数据...")
    print("=" * 60)
    kaggle_df = load_kaggle_data()
    kaggle_results = run_kaggle_task(kaggle_df, results_dir)
    write_kaggle_report(kaggle_results, results_dir)

    # 生成 Kaggle 数据图表
    print("\n生成 Kaggle 数据图表...")
    plot_kaggle_data_exploration(kaggle_df, results_dir)

    # Task C: 对照总结
    write_summary_comparison(synthetic_results, kaggle_results, results_dir)

    # 生成对比图表
    print("\n生成对比图表...")
    plot_coefficient_comparison(synthetic_results, kaggle_results, results_dir)
    plot_metrics_comparison(synthetic_results, kaggle_results, results_dir)

    print("\n" + "=" * 60)
    print("所有报告和图表已生成至:", results_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
