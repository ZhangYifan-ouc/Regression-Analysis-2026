"""
Week 9 数据预处理 CLI 脚本
=============================
功能：
    1. 将分类变量进行 One-Hot 编码（并丢弃第一列，防止虚拟变量陷阱）
    2. 对数值型特征进行 Winsorization（缩尾处理），将极端值限制在 99 分位数
    3. 处理缺失值，使用列级均值进行填充
    4. 保存清洁后的数据

用法:
    python data_prep.py --input path/to/dirty_data.csv --output path/to/clean_data.csv

示例:
    python data_prep.py --input dirty_marketing.csv --output clean_marketing.csv
"""

import pandas as pd
import numpy as np
import argparse
import sys
import os


def main():
    """
    主函数：CLI 入口，负责数据预处理流水线
    """
    # ========== 第一步：解析命令行参数 ==========
    parser = argparse.ArgumentParser(
        description="数据清洗脚本：处理分类变量、异常值和缺失值"
    )
    parser.add_argument("--input", required=True, help="输入 CSV 文件路径（脏数据）")
    parser.add_argument("--output", required=True, help="输出 CSV 文件路径（干净数据）")
    args = parser.parse_args()

    # ========== 验证输入文件 ==========
    if not os.path.exists(args.input):
        print(f"错误：输入文件 {args.input} 不存在")
        sys.exit(1)

    # ========== 第二步：加载数据 ==========
    df = pd.read_csv(args.input)
    print(f"✓ 已加载数据，形状: {df.shape}")

    # ========== 第三步：分离目标变量 ==========
    # 假设目标变量是 'Sales'
    target_col = 'Sales'
    
    if target_col not in df.columns:
        print(f"错误：找不到目标列 '{target_col}'")
        print(f"可用列: {df.columns.tolist()}")
        sys.exit(1)
    
    # 分离特征和目标
    y = df[target_col]  # 目标变量（最后一列）
    X = df.drop(columns=[target_col])  # 特征矩阵
    
    print(f"  - 目标变量: {target_col}")
    print(f"  - 特征列: {X.columns.tolist()}")

    # ========== 第四步：识别特征列类型 ==========
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    print(f"  - 分类特征列: {categorical_cols}")
    print(f"  - 数值特征列: {numerical_cols}")

    # ========== 第五步：One-Hot 编码（处理分类变量） ==========
    # 防雷：必须使用 drop_first=True 来丢弃第一列分类
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        print(f"✓ 已进行 One-Hot 编码（drop_first=True），特征形状: {X.shape}")

    # ========== 第六步：Winsorization（缩尾处理异常值） ==========
    for col in numerical_cols:
        if col in X.columns:
            percentile_99 = X[col].quantile(0.99)
            count_before = (X[col] > percentile_99).sum()
            X[col] = np.where(X[col] > percentile_99, percentile_99, X[col])
            if count_before > 0:
                print(f"  - {col}: 缩尾处理 {count_before} 个异常值到 {percentile_99:.2f}")

    # ========== 第七步：填补缺失值 ==========
    # 只对特征填补缺失值
    initial_na_count = X.isna().sum().sum()
    X = X.fillna(X.mean())
    print(f"✓ 已填补特征缺失值: {initial_na_count} 个 NaN 值已用列均值替换")

    # ========== 第八步：合并特征和目标 ==========
    # 确保目标变量在最后一列
    df_clean = X.copy()
    df_clean[target_col] = y
    
    print(f"✓ 清洁数据最终形状: {df_clean.shape}")
    print(f"  - 特征列: {df_clean.columns[:-1].tolist()}")
    print(f"  - 目标列: {df_clean.columns[-1]} (最后一列)")

    # ========== 第九步：保存清洁数据 ==========
    df_clean.to_csv(args.output, index=False)
    print(f"✓ 清洁数据已保存到: {args.output}")


if __name__ == "__main__":
    main()