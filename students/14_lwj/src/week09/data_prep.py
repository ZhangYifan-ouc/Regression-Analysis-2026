import argparse
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize


def main():
    # 1. 命令行参数解析
    parser = argparse.ArgumentParser(description="数据预处理工具：清洗、编码、缩尾、填充缺失值")
    parser.add_argument("--input", required=True, type=str, help="输入脏数据的CSV路径")
    parser.add_argument("--output", required=True, type=str, help="输出清洗后数据的CSV路径")
    args = parser.parse_args()

    # 读取原始数据
    df = pd.read_csv(args.input)
    print(f"✅ 成功读取数据：{df.shape[0]} 行 {df.shape[1]} 列")

    # ======================
    # 2. 缺失值处理：均值/中位数填充
    # ======================
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        # 数值列使用中位数填充（更鲁棒）
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
    print(f"✅ 数值缺失值已使用中位数填充")

    # ======================
    # 3. 异常值处理：缩尾法（Winsorization）
    # 预算列 > 99分位数 → 强制缩到99分位
    # ======================
    budget_col = None
    for col in df.columns:
        if "budget" in col.lower() or "spend" in col.lower():
            budget_col = col
            break

    if budget_col is not None:
        # 缩尾：上限 99%，下限 0%
        df[budget_col] = winsorize(df[budget_col], limits=[0.0, 0.01])
        print(f"✅ 完成预算列【{budget_col}】缩尾处理（99分位截断）")
    else:
        print("⚠️ 未找到预算/花费列，跳过缩尾处理")

    # ======================
    # 4. 分类变量独热编码 + 避免虚拟变量陷阱
    # ======================
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    if len(categorical_cols) > 0:
        # drop_first=True 是关键！避免虚拟变量陷阱
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
        print(f"✅ 分类列独热编码完成：{list(categorical_cols)}")
        print(f"✅ 已自动丢弃第一列，避免虚拟变量陷阱")

    # ======================
    # 5. 保存清洗后的数据
    # ======================
    df.to_csv(args.output, index=False)
    print(f"\🎉 数据预处理完成！")
    print(f"📁 清洗后数据已保存到：{args.output}")
    print(f"📊 最终数据维度：{df.shape[0]} 行 {df.shape[1]} 列")


if __name__ == "__main__":
    main()