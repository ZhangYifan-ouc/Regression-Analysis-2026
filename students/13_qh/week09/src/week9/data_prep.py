"""
Module: week9.data_prep
Purpose: CLI data preprocessing script for cleaning dirty marketing data.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="数据清洗 CLI 工具")
    parser.add_argument("--input", type=str, required=True, help="输入数据文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出数据文件路径")
    return parser.parse_args()


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """处理缺失值：使用均值填补数值列。"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val)
            print(f"  列 '{col}': 使用均值 {mean_val:.4f} 填补缺失值")
    return df


def handle_outliers(df: pd.DataFrame, quantile: float = 0.99) -> pd.DataFrame:
    """处理异常值：Winsorization，将超过99分位数的值缩尾。"""
    numeric_cols = ['TV_Budget', 'Radio_Budget', 'SocialMedia_Budget']
    for col in numeric_cols:
        if col in df.columns:
            upper_bound = df[col].quantile(quantile)
            n_outliers = (df[col] > upper_bound).sum()
            if n_outliers > 0:
                df[col] = df[col].clip(upper=upper_bound)
                print(f"  列 '{col}': {n_outliers} 个异常值缩尾到 {upper_bound:.4f}")
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """处理分类变量：One-Hot 编码，drop_first=True 避免虚拟变量陷阱。"""
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print(f"  对分类变量进行 One-Hot 编码: {list(categorical_cols)}")
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    return df


def main():
    """主函数：执行数据清洗流程。"""
    args = parse_args()

    print("=" * 60)
    print("数据清洗 CLI 工具")
    print("=" * 60)

    # 读取数据
    print(f"\n读取数据: {args.input}")
    df = pd.read_csv(args.input)
    print(f"原始数据形状: {df.shape}")
    print(f"缺失值统计:\n{df.isnull().sum()}")

    # 1. 处理缺失值
    print("\n--- 步骤1: 处理缺失值 ---")
    df = handle_missing_values(df)

    # 2. 处理异常值
    print("\n--- 步骤2: 处理异常值 (Winsorization) ---")
    df = handle_outliers(df)

    # 3. 处理分类变量
    print("\n--- 步骤3: 处理分类变量 (One-Hot 编码) ---")
    df = encode_categorical(df)

    # 保存清洗后的数据
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n清洗后数据形状: {df.shape}")
    print(f"清洗后数据列: {list(df.columns)}")
    print(f"\n数据已保存至: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
