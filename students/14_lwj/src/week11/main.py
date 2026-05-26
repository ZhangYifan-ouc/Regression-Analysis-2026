import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from pathlib import Path

# ======================== 修复路径 ========================
week11_dir = Path(__file__).parent
src_dir = week11_dir.parent
sys.path.insert(0, str(src_dir))

# ======================== 导入工具包 ========================
from utils.models import CustomOLS
from utils.metrics import calculate_rmse, calculate_mae, calculate_mape
from utils.transformers import CustomStandardScaler
from utils.diagnostics import calculate_vif

# ======================== 路径设置 ========================
BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
RES_DIR = BASE / "results"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

# ======================== 生成模拟数据 ========================
def generate_synthetic_data():
    np.random.seed(42)
    n = 500

    tv = np.random.normal(50, 15, n)
    radio = np.random.normal(30, 10, n)
    season = np.random.choice(["Spring", "Summer", "Autumn", "Winter"], n)
    tv2 = 0.8 * tv + np.random.normal(0, 3, n)

    tv[np.random.choice(n, 50)] = np.nan
    radio[np.random.choice(n, 20)] *= 5

    df = pd.DataFrame({
        "TV": tv,
        "Radio": radio,
        "TV2": tv2,
        "Season": season
    })
    df = pd.get_dummies(df, columns=["Season"], drop_first=True)
    y = 3 * tv + 1.5 * radio + 2 * (season == "Winter") + np.random.normal(0, 8, n)
    df["Sales"] = y

    df.to_csv(DATA_DIR / "synthetic_regression.csv", index=False)
    return df

# ======================== 核心：无泄露CV（终极修复版） ========================
def run_cv_no_leakage(X, y):
    # 强制转成 float64，彻底解决类型问题
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    mask = ~np.isnan(y)
    X = X[mask]
    y = y[mask]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_list, mae_list, mape_list = [], [], []

    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # 用训练集均值填充所有 NaN
        col_means = np.nanmean(X_tr, axis=0)
        X_tr = np.where(np.isnan(X_tr), col_means, X_tr)
        X_val = np.where(np.isnan(X_val), col_means, X_val)

        # 标准化
        scaler = CustomStandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)

        # 训练
        model = CustomOLS()
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)

        rmse_list.append(calculate_rmse(y_val, y_pred))
        mae_list.append(calculate_mae(y_val, y_pred))
        mape_list.append(calculate_mape(y_val, y_pred))

    return np.mean(rmse_list), np.mean(mae_list), np.mean(mape_list)

# ======================== 运行模拟任务 ========================
def run_synthetic():
    generate_synthetic_data()
    df = pd.read_csv(DATA_DIR / "synthetic_regression.csv")
    df_clean = df.dropna()
    X = df_clean.iloc[:, :-1].values
    y = df_clean.iloc[:, -1].values

    vif_values = calculate_vif(X)
    rmse, mae, mape = run_cv_no_leakage(X, y)

    with open(RES_DIR / "synthetic_report.md", "w", encoding="utf-8") as f:
        f.write(f"""# 模拟数据报告
数据生成机制 DGP：
Sales = 3*TV + 1.5*Radio + 2*Winter + 噪声

- RMSE: {rmse:.2f}
- MAE: {mae:.2f}
- MAPE: {mape:.2f}%

VIF 诊断结果: {vif_values}
""")
    print("✅ 模拟数据任务完成")

# ======================== 运行Kaggle任务 ========================
def run_kaggle():
    try:
        df = pd.read_csv(DATA_DIR / "kaggle_housing.csv")
        df = df.dropna(subset=[df.columns[-1]])
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        rmse, mae, mape = run_cv_no_leakage(X, y)

        with open(RES_DIR / "kaggle_report.md", "w", encoding="utf-8") as f:
            f.write(f"""# Kaggle真实数据报告
数据集：加州房价预测
预测目标：房屋价格

- RMSE: {rmse:.2f}
- MAE: {mae:.2f}
- MAPE: {mape:.2f}%
""")
        print("✅ Kaggle任务完成")
    except Exception as e:
        print(f"⚠️ Kaggle数据未放入，已跳过：{e}")

# ======================== 总结报告 ========================
def summary():
    with open(RES_DIR / "summary_comparison.md", "w", encoding="utf-8") as f:
        f.write("""# 模拟数据 vs 真实数据 对比总结
1. 模拟数据：数据生成机制明确，推断结果可控、容易解释。
2. 真实数据：噪声多、缺失多、特征关系复杂，解释难度更高。
3. 共线性、缺失值、异常值在两类数据中都会影响模型稳定性。
4. 无泄露交叉验证是保证评估结果真实可信的关键。
5. 复用自己的 utils 组件，大幅减少重复代码，流程更统一。
""")

# ======================== 主入口 ========================
if __name__ == "__main__":
    print("🚀 Week11 作业开始运行...")
    run_synthetic()
    run_kaggle()
    summary()
    print("\n🎉 全部完成！报告已保存在 results 文件夹")