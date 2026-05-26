import sys
from pathlib import Path


current_script_path = Path(__file__).resolve()
# week09 目录
week09_dir = current_script_path.parent
# src 根目录（week09的上一级）
src_root_dir = week09_dir.parent
# 把src目录加入Python的模块搜索路径
sys.path.insert(0, str(src_root_dir))


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

from utils.diagnostics import calculate_vif
from utils.models import CustomOLS


def print_red(text: str) -> None:
    """红色警告输出"""
    print(f"\033[91m{text}\033[0m")


def main():
    # 读取预处理后的数据
    df = pd.read_csv("data/clean_marketing.csv")
    print("✅ 读取清洗完成的数据")

    # 划分特征与标签
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    feature_names = df.columns[:-1].tolist()

    # ======================
    # 多重共线性检测（VIF）
    # ======================
    print("\n" + "="*50)
    print("📋 特征 VIF 多重共线性检测")
    print("="*50)
    
    vif_scores = calculate_vif(X)
    high_vif = []
    
    for name, vif in zip(feature_names, vif_scores):
        print(f"特征 {name:22} | VIF = {vif:.2f}")
        if vif > 10:
            high_vif.append((name, vif))

    # 高共线性红色警告
    if high_vif:
        print("\n" + "!" * 60)
        print_red("⚠️  严重警告：发现高多重共线性特征 (VIF > 10)！")
        for name, vif in high_vif:
            print_red(f"   特征 {name} | VIF = {vif:.2f}")
        print_red("💡 建议：删除/合并相关特征，保证模型稳定！")
        print("!" * 60 + "\n")

    # ======================
    # 5折交叉验证基线
    # ======================
    print("="*50)
    print("📈 5-Fold CV 基线评估 (CustomOLS)")
    print("="*50)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_list = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = CustomOLS()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        r2_list.append(r2)
        print(f"Fold {fold} | R² = {r2:.4f}")

    mean_r2 = np.mean(r2_list)
    print(f"\n🎯 5折交叉验证平均 R² = {mean_r2:.4f}")
    print("\n✅ 评估完成！")


if __name__ == "__main__":
    main()