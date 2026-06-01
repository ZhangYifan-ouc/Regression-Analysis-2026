import sys
import os
# 自动把项目路径加入Python，彻底解决 ModuleNotFoundError
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("./students/18_mxt/src"))

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from utils.diagnostics import calculate_vif
from utils.models import AnalyticalOLS

def red_print(text):
    return f"\033[91m{text}\033[0m"

def main():
    df = pd.read_csv("./homework/week09/data/clean_marketing.csv")

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    vif_scores = calculate_vif(X)
    print("\n📊 特征 VIF 值：", vif_scores)

    high_vif_indices = [i for i, v in enumerate(vif_scores) if v > 10]
    if high_vif_indices:
        print(red_print(f"⚠️  警告：特征 {high_vif_indices} 存在严重多重共线性！"))

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_list = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = AnalyticalOLS()
        model.fit(X_train, y_train)
        r2_list.append(model.score(X_test, y_test))

    mean_r2 = np.mean(r2_list)
    print(f"\n✅ 5 折交叉验证平均 R² = {mean_r2:.4f}")

if __name__ == "__main__":
    main()