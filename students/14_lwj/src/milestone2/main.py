import sys
from pathlib import Path
import numpy as np
import pandas as pd
import os
import shutil
from sklearn.model_selection import KFold

# 路径配置
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

from utils.metrics import calculate_rmse, calculate_mae, calculate_mape
from utils.transformers import CustomStandardScaler
from utils.models import CustomOLS  # 只导入 CustomOLS

def reset_results_folder():
    if os.path.exists("results"):
        shutil.rmtree("results")
    os.makedirs("results", exist_ok=True)

def bad_cross_validation(X, y):
    X_filled = X.astype(np.float64).copy()
    global_mean = np.nanmean(X_filled, axis=0)
    for i in range(X_filled.shape[1]):
        mask = np.isnan(X_filled[:, i])
        X_filled[mask, i] = global_mean[i]

    scaler = CustomStandardScaler()
    X_scaled = scaler.fit_transform(X_filled)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_list, mae_list, mape_list = [], [], []

    for train_idx, val_idx in kf.split(X_scaled):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = CustomOLS()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        rmse_list.append(calculate_rmse(y_val, y_pred))
        mae_list.append(calculate_mae(y_val, y_pred))
        mape_list.append(calculate_mape(y_val, y_pred))

    return np.mean(rmse_list), np.mean(mae_list), np.mean(mape_list)

def good_cross_validation(X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_list, mae_list, mape_list = [], [], []

    for train_idx, val_idx in kf.split(X):
        X_train_raw = X[train_idx].astype(np.float64)
        X_val_raw = X[val_idx].astype(np.float64)
        y_train, y_val = y[train_idx], y[val_idx]

        train_mean = np.nanmean(X_train_raw, axis=0)

        X_train_filled = X_train_raw.copy()
        X_val_filled = X_val_raw.copy()
        for i in range(X_train_filled.shape[1]):
            mask = np.isnan(X_train_filled[:, i])
            X_train_filled[mask, i] = train_mean[i]

            mask_val = np.isnan(X_val_filled[:, i])
            X_val_filled[mask_val, i] = train_mean[i]

        scaler = CustomStandardScaler()
        X_train = scaler.fit_transform(X_train_filled)
        X_val = scaler.transform(X_val_filled)

        model = CustomOLS()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        rmse_list.append(calculate_rmse(y_val, y_pred))
        mae_list.append(calculate_mae(y_val, y_pred))
        mape_list.append(calculate_mape(y_val, y_pred))

    return np.mean(rmse_list), np.mean(mae_list), np.mean(mape_list)

def save_report(bad, good):
    content = f"""# 数据泄露对比报告
| 指标 | 有泄露（坏） | 无泄露（好） |
|------|------------|-------------|
| RMSE | {bad[0]:.2f} | {good[0]:.2f} |
| MAE  | {bad[1]:.2f} | {good[1]:.2f} |
| MAPE | {bad[2]:.2f}% | {good[2]:.2f}% |

## 结论
有数据泄露时模型误差更小、分数更好看，但这是不真实的，上线后会翻车。
无泄露的结果虽然误差更大，但代表了模型在真实业务中的泛化能力，是可信的。
"""
    with open("results/evaluation_comparison.md", "w", encoding="utf-8") as f:
        f.write(content)

if __name__ == "__main__":
    reset_results_folder()
    df = pd.read_csv("data/clean_marketing.csv")
    
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.astype(np.float64)

    print("正在运行：有数据泄露（坏）...")
    bad = bad_cross_validation(X, y)

    print("正在运行：无数据泄露（好）...")
    good = good_cross_validation(X, y)

    save_report(bad, good)
    print("\n✅ 全部完成！报告已保存到 results/evaluation_comparison.md")