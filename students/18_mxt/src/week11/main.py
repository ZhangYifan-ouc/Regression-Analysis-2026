# 导入基础库
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# 路径修复：将src目录加入Python路径，解决utils导入问题
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# 导入自定义工具库（全程复用，符合工程规则）
from utils.models import CustomOLS
from utils.metrics import rmse, mae, mape
from utils.transformers import CustomStandardScaler, CustomImputer, winsorize

# 路径配置（严格按作业目录规范）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RES_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# 自定义VIF计算函数（完全复用CustomOLS，不依赖任何外部库）
# 原理：对每个特征，用其他特征做回归，计算R²，VIF=1/(1-R²)
# ------------------------------------------------------------------------------
def calculate_vif(X, feature_names):
    vif_values = []
    n_features = X.shape[1]
    
    for i in range(n_features):
        # 将第i个特征作为因变量，其他特征作为自变量
        y_vif = X[:, i]
        X_vif = np.delete(X, i, axis=1)
        
        # 使用自己的CustomOLS计算R²
        model = CustomOLS()
        model.fit(X_vif, y_vif)
        y_pred = model.predict(X_vif)
        
        ss_total = np.sum((y_vif - np.mean(y_vif)) ** 2)
        ss_residual = np.sum((y_vif - y_pred) ** 2)
        r_squared = 1 - (ss_residual / ss_total)
        
        vif = 1 / (1 - r_squared) if r_squared < 1 else float('inf')
        vif_values.append(round(vif, 2))
    
    return pd.DataFrame({"Feature": feature_names, "VIF": vif_values})

# ------------------------------------------------------------------------------
# Task A：生成模拟数据（符合A1要求：业务场景+强共线性+脏数据）
# ------------------------------------------------------------------------------
def generate_synthetic_data():
    np.random.seed(42)  # 固定随机种子，保证结果可复现
    n = 500  # 样本量≥300，满足要求

    # 特征定义（带业务含义）
    x1 = np.random.normal(30, 10, n)          # x1：每日基础学习时长
    x2 = 0.85 * x1 + np.random.normal(0, 2, n)# x2：有效学习时长（与x1强共线性）
    x3 = np.random.normal(50, 15, n)          # x3：刷题数量
    group = np.random.choice([0, 1, 2], n)    # group：班级类别（0/1/2）

    # DGP真实生成公式（A2要求）
    y = 1.8 * x1 + 0.9 * x2 - 0.6 * x3 + \
        3 * (group == 1) + 5 * (group == 2) + \
        np.random.normal(0, 4, n)

    # 植入真实数据问题（A1要求：缺失值+异常值）
    X = np.column_stack([x1, x2, x3, group])
    mask = np.random.choice([False, True], X.shape, p=[0.96, 0.04])  # 4%缺失值
    X[mask] = np.nan
    X[np.random.choice(n, 6), 2] *= 4  # 在x3中植入6个异常值

    # 保存数据到指定路径（A2要求）
    df = pd.DataFrame(X, columns=["x1", "x2", "x3", "group"])
    df["target"] = y
    df.to_csv(os.path.join(DATA_DIR, "synthetic_regression.csv"), index=False)
    return df

# ------------------------------------------------------------------------------
# Task A：运行模拟数据回归分析（符合A3要求：无泄露CV+复用utils）
# ------------------------------------------------------------------------------
def run_synthetic_task():
    print("▶ 运行 Task A：模拟数据回归分析")
    df = generate_synthetic_data()
    X = df.drop("target", axis=1).values
    y = df["target"].values
    feature_names = df.drop("target", axis=1).columns.tolist()

    # 5折无泄露交叉验证（A3要求）
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_list, mae_list, mape_list = [], [], []

    for tr_idx, te_idx in kf.split(X):
        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]

        # 无泄露预处理：所有步骤仅拟合训练集（工程红线）
        imp = CustomImputer()
        imp.fit(Xtr)
        Xtr = imp.transform(Xtr)
        Xte = imp.transform(Xte)

        Xtr = winsorize(Xtr, (0.02, 0.02))
        Xte = winsorize(Xte, (0.02, 0.02))

        scaler = CustomStandardScaler()
        scaler.fit(Xtr)
        Xtr = scaler.transform(Xtr)
        Xte = scaler.transform(Xte)

        # 使用自定义模型训练
        model = CustomOLS()
        model.fit(Xtr, ytr)
        yp = model.predict(Xte)

        # 使用自定义指标计算
        rmse_list.append(rmse(yte, yp))
        mae_list.append(mae(yte, yp))
        mape_list.append(mape(yte, yp))

    # 计算全量数据的VIF（用于共线性诊断，A3要求）
    # 修复：将fit_transform拆分为fit+transform，匹配你的utils接口
    imp = CustomImputer()
    imp.fit(X)
    X_imp = imp.transform(X)
    
    X_win = winsorize(X_imp, (0.02, 0.02))
    
    scaler = CustomStandardScaler()
    scaler.fit(X_win)
    X_sca = scaler.transform(X_win)
    
    vif_result = calculate_vif(X_sca, feature_names)

    return {
        "RMSE": round(np.mean(rmse_list), 4),
        "MAE": round(np.mean(mae_list), 4),
        "MAPE": round(np.mean(mape_list), 4),
        "VIF": vif_result
    }

# ------------------------------------------------------------------------------
# Task B：运行Kaggle真实数据回归分析（符合B2要求：全流程复用utils）
# ------------------------------------------------------------------------------
def run_kaggle_task():
    print("▶ 运行 Task B：Kaggle房价回归分析")
    # 读取真实数据
    df = pd.read_csv(os.path.join(DATA_DIR, "kaggle_house.csv"))
    use_cols = ["OverallQual", "GrLivArea", "GarageArea", "TotalBsmtSF", "YearBuilt", "SalePrice"]
    df = df[use_cols].dropna()
    X = df.drop("SalePrice", axis=1).values
    y = df["SalePrice"].values

    # 5折无泄露交叉验证（B2要求）
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_list, mae_list, mape_list = [], [], []

    for tr_idx, te_idx in kf.split(X):
        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]

        # 无泄露预处理（与模拟数据流程完全一致）
        imp = CustomImputer()
        imp.fit(Xtr)
        Xtr = imp.transform(Xtr)
        Xte = imp.transform(Xte)

        Xtr = winsorize(Xtr, (0.02, 0.02))
        Xte = winsorize(Xte, (0.02, 0.02))

        scaler = CustomStandardScaler()
        scaler.fit(Xtr)
        Xtr = scaler.transform(Xtr)
        Xte = scaler.transform(Xte)

        # 使用自定义模型训练
        model = CustomOLS()
        model.fit(Xtr, ytr)
        yp = model.predict(Xte)

        # 使用自定义指标计算
        rmse_list.append(rmse(yte, yp))
        mae_list.append(mae(yte, yp))
        mape_list.append(mape(yte, yp))

    return {
        "RMSE": round(np.mean(rmse_list), 4),
        "MAE": round(np.mean(mae_list), 4),
        "MAPE": round(np.mean(mape_list), 4)
    }

# ------------------------------------------------------------------------------
# 主入口（唯一执行入口，符合工程规则）
# ------------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Week11 回归分析工作流")
    print("=" * 60)

    # 运行模拟数据任务
    res_syn = run_synthetic_task()
    print("✅ Task A 结果：", {"RMSE": res_syn["RMSE"], "MAE": res_syn["MAE"], "MAPE": res_syn["MAPE"]})
    print("\n📊 Task A VIF共线性诊断结果：")
    print(res_syn["VIF"].to_string(index=False))

    print("-" * 60)

    # 运行真实数据任务
    res_kag = run_kaggle_task()
    print("✅ Task B 结果：", res_kag)

    print("=" * 60)
    print("🎉 全部流程运行完成！可现场展示！")

if __name__ == "__main__":
    main()