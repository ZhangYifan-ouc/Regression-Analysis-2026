"""
Module: week9.evaluate
Purpose: Model diagnostics and cross-validation evaluation.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.models import AnalyticalOLS
from utils.diagnostics import calculate_vif, check_multicollinearity


def print_warning(message: str):
    """打印红色警告信息。"""
    print(f"\033[91m[WARNING] {message}\033[0m")


def run_vif_diagnostics(X: np.ndarray, feature_names: list):
    """运行 VIF 多重共线性诊断。"""
    print("\n" + "=" * 60)
    print("多重共线性诊断 (VIF)")
    print("=" * 60)

    result = check_multicollinearity(X, feature_names, threshold=10.0)

    print("\nVIF 值:")
    for name, vif in zip(feature_names, result['vif_values']):
        status = "⚠️ 高" if vif > 10 else "✓ 正常"
        print(f"  {name}: {vif:.4f} [{status}]")

    if result['has_multicollinearity']:
        print("\n" + "=" * 60)
        print_warning("检测到严重多重共线性问题!")
        for warning in result['warnings']:
            print_warning(warning)
        print_warning("建议: 删除或合并高 VIF 的特征，或使用正则化方法")
        print("=" * 60)
    else:
        print("\n✓ 未检测到严重多重共线性问题 (所有 VIF < 10)")


def run_cross_validation(X: np.ndarray, y: np.ndarray, n_splits: int = 5):
    """运行 5-Fold 交叉验证。"""
    print("\n" + "=" * 60)
    print("5-Fold 交叉验证 (AnalyticalOLS)")
    print("=" * 60)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    r2_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = AnalyticalOLS().fit(X_train, y_train)
        preds = model.predict(X_val)
        fold_r2 = r2_score(y_val, preds)
        r2_scores.append(fold_r2)

        print(f"第 {fold} 折: R² = {fold_r2:.4f}")

    avg_r2 = np.mean(r2_scores)
    std_r2 = np.std(r2_scores)

    print(f"\n平均 R²: {avg_r2:.4f} (±{std_r2:.4f})")

    return avg_r2, std_r2


def main():
    """主函数。"""
    print("=" * 60)
    print("第九周：数据急救员与病态模型诊断")
    print("=" * 60)

    # 读取清洗后的数据
    data_path = Path(__file__).parent.parent.parent / "data" / "clean_marketing.csv"
    if not data_path.exists():
        print(f"错误: 找不到清洗后的数据文件 {data_path}")
        print("请先运行 data_prep.py 生成清洗后的数据")
        sys.exit(1)

    df = pd.read_csv(data_path)
    print(f"\n读取数据: {data_path}")
    print(f"数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")

    # 分离特征和目标
    target_col = 'Sales'
    feature_cols = [col for col in df.columns if col != target_col]

    X = df[feature_cols].values
    y = df[target_col].values

    # 1. VIF 诊断
    run_vif_diagnostics(X, feature_cols)

    # 2. 交叉验证
    # 添加截距列
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    avg_r2, std_r2 = run_cross_validation(X_with_intercept, y)

    # 生成报告
    results_dir = Path(__file__).parent.parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    with open(results_dir / "diagnostics_report.md", "w", encoding="utf-8") as f:
        f.write("# 第九周诊断报告\n\n")
        f.write("## 数据概况\n\n")
        f.write(f"- 样本数量: {len(df)}\n")
        f.write(f"- 特征数量: {len(feature_cols)}\n")
        f.write(f"- 特征列表: {feature_cols}\n\n")

        f.write("## VIF 诊断结果\n\n")
        vif_result = check_multicollinearity(X, feature_cols)
        f.write("| 特征 | VIF | 状态 |\n")
        f.write("|------|-----|------|\n")
        for name, vif in zip(feature_cols, vif_result['vif_values']):
            status = "⚠️ 高" if vif > 10 else "✓ 正常"
            f.write(f"| {name} | {vif:.4f} | {status} |\n")

        if vif_result['has_multicollinearity']:
            f.write("\n**警告**: 检测到严重多重共线性问题\n")
            for warning in vif_result['warnings']:
                f.write(f"- {warning}\n")

        f.write("\n## 交叉验证结果\n\n")
        f.write(f"- 平均 R²: {avg_r2:.4f} (±{std_r2:.4f})\n\n")

        f.write("## 思考题\n\n")
        f.write("### 问题: 验证集数据真的算是'完全未见过的陌生数据'吗？\n\n")
        f.write("答案: 不完全是。\n\n")
        f.write("原因: 在 data_prep.py 中，我们使用了全量数据的均值来填补缺失值。")
        f.write("这意味着验证集中的数据已经被'污染'了——它们的缺失值填补使用了包含验证集在内的全量数据的统计量。")
        f.write("这属于一种轻微的数据泄露（data leakage），会导致交叉验证的 R² 偏高。\n\n")
        f.write("正确的做法应该是:\n")
        f.write("1. 在每折交叉验证中，只使用训练折的均值来填补缺失值\n")
        f.write("2. 然后用同样的均值去填补验证折的缺失值\n")
        f.write("3. 这样才能保证验证折是'完全未见过的陌生数据'\n")

    print(f"\n诊断报告已保存至: {results_dir / 'diagnostics_report.md'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
