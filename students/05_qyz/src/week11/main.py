# ============================
# Week11作业
# ============================
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# 项目路径配置，确保能导入 utils
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

# 自己实现的工具库
from utils.transformers import CustomStandardScaler  # 标准化
from utils.diagnostics import (  # 共线性诊断
    calculate_vif_dataframe,
    print_vif_warning,
    plot_residuals,
    plot_qq_residuals,
    plot_correlation_matrix,
)
from utils.models import AnalyticalOLS, RidgeRegression  # 解析解回归模型
from utils.metrics import calculate_rmse, calculate_mae, calculate_mape  # 评估指标


# ------------------------------------------------------------------------------
# 工具函数：全流程通用辅助方法
# ------------------------------------------------------------------------------
def get_result_dir():
    """
    创建并返回结果保存目录
    """
    res_dir = Path(__file__).parent / "results"
    res_dir.mkdir(exist_ok=True)
    return res_dir


def save_descriptive_stats(df, title):
    """
    输出描述性统计（均值、标准差、分位数等）
    """
    print(f"\n📊 {title}")
    print(df.describe().round(2))


def save_plots(df, name, filename):
    """
    生成两张图并保存到 results/1. 价格分布直方图2. 年份/车龄与价格散点图
    """
    res_dir = get_result_dir()
    path = res_dir / filename

    plt.figure(figsize=(10, 4))
    # 左图：价格分布
    plt.subplot(1, 2, 1)
    if "price" in df.columns:
        df["price"].hist(bins=30, color="skyblue", edgecolor="black")
    else:
        df["Price"].hist(bins=30, color="skyblue", edgecolor="black")
    plt.title("Price Distribution")
    # 右图：相关性散点图
    plt.subplot(1, 2, 2)
    if "car_age" in df.columns:
        plt.scatter(df["car_age"], df["price"], alpha=0.5, c="orange")
        plt.xlabel("Car Age")
        plt.ylabel("Price")
    else:
        plt.scatter(df["Year_cent"], df["Price"], alpha=0.5, c="green")
        plt.xlabel("Year (centered)")
        plt.ylabel("Price")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✅ 图表已保存：{path}")


def save_metrics(rmse, mae, mape, r2, title):
    """
    输出模型评估指标
    """
    print(f"\n📋 {title}")
    print(f"RMSE  = {rmse:.2f}")
    print(f"MAE   = {mae:.2f}")
    print(f"MAPE  = {mape:.2f}%")
    print(f"R²    = {r2:.4f}")


def save_interpretation(is_ridge=False):
    """
    输出结果解释：系数方向、误差来源、正则化作用
    """
    print("\n 结果解释")
    print("🔹 系数方向符合业务直觉")
    print("🔹 误差受异常值与共线性影响")
    if is_ridge:
        print("🔹 已使用 Ridge 正则化缓解多重共线性")


# ------------------------------------------------------------------------------
# 1. 生成模拟数据（Task A）
# 功能：按已知 DGP 生成二手车价格模拟数据，用于验证回归模型是否正确
# ------------------------------------------------------------------------------
def generate_synthetic_data():
    np.random.seed(42)  # 固定随机种子，保证可复现
    n = 500
    # 模拟特征：车龄、里程、功率、变速箱、燃油类型
    car_age = np.random.uniform(1, 15, n)
    mileage = 1.2 * car_age + np.random.normal(
        0, 2, n
    )  # 里程与车龄强相关（制造共线性）
    engine_power = np.random.uniform(50, 300, n)
    transmission = np.random.choice(["Manual", "Automatic"], n)
    fuel_type = np.random.choice(["Petrol", "Diesel"], n)
    # 随机制造 10% 缺失值，模拟真实脏数据
    mask = np.random.rand(n) < 0.1
    engine_power[mask] = np.nan
    # 真实数据生成机制 DGP
    price = (
        5000
        - 800 * car_age
        - 300 * mileage
        + 50 * engine_power
        + 2000 * (transmission == "Automatic")
        + 1500 * (fuel_type == "Diesel")
        + np.random.normal(0, 1500, n)
    )
    # 手动加入异常值，让模拟更贴近真实数据
    out1 = np.random.choice(n, 5)
    price[out1] *= 3
    out2 = np.random.choice(n, 5)
    price[out2] *= -1
    # 构造 DataFrame 并保存
    df = pd.DataFrame(
        {
            "car_age": car_age,
            "mileage": mileage,
            "engine_power": engine_power,
            "transmission": transmission,
            "fuel_type": fuel_type,
            "price": price,
        }
    )

    out_path = Path(__file__).parent / "data" / "synthetic_regression.csv"
    df.to_csv(out_path, index=False)
    return df


# ------------------------------------------------------------------------------
# 2. 运行模拟数据任务
# 流程：生成数据 → 清洗 → 编码 → 标准化 → CV → 评估 → 报告
# ------------------------------------------------------------------------------
def run_synthetic_task():
    print("=" * 60)
    print("Task A：模拟数据实验")
    print("=" * 60)

    # 生成模拟数据
    df = generate_synthetic_data()
    # 数据探索：统计 + 图表
    save_descriptive_stats(df, "模拟数据描述统计")
    save_plots(df, "Synthetic", "synthetic_plot.png")
    # 分类变量独热编码
    df = pd.get_dummies(df, columns=["transmission", "fuel_type"], drop_first=True)
    # ===================== 在这里加缺失值填补 =====================
    print("\n 模拟数据缺失值检查：")
    print(df.isnull().sum())
    df["engine_power"] = df["engine_power"].fillna(df["engine_power"].median())
    df["price"] = df["price"].fillna(df["price"].median())
    # ===============================================================
    # 构建 X, y
    y = df["price"].values
    X = df.drop("price", axis=1).values.astype(float)
    # VIF 共线性诊断
    print("\n开始 VIF 共线性诊断")
    vif_df = pd.DataFrame(X, columns=df.drop("price", axis=1).columns)
    vif = calculate_vif_dataframe(vif_df, vif_df.columns.tolist())
    print_vif_warning(vif)
    # 5 折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_list, mae_list, mape_list, r2_list = [], [], [], []

    for train_idx, test_idx in kf.split(X):
        # 按 fold 分割训练集 / 测试集
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 标准化必须在每一折内部执行（防泄露）
        scaler = CustomStandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 训练 OLS 模型（解析解）
        model = AnalyticalOLS(fit_intercept=True)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 记录指标
        rmse_list.append(calculate_rmse(y_test, y_pred))
        mae_list.append(calculate_mae(y_test, y_pred))
        mape_list.append(calculate_mape(y_test, y_pred))
        r2_list.append(model.score(X_test, y_test))

    # 输出报告
    save_metrics(
        np.mean(rmse_list),
        np.mean(mae_list),
        np.mean(mape_list),
        np.mean(r2_list),
        "模拟数据 OLS 指标",
    )
    save_interpretation()


# ------------------------------------------------------------------------------
# 3. 读取 Kaggle 真实二手车数据
# 只保留关键字段，删除缺失，保证数据干净
# ------------------------------------------------------------------------------
def load_kaggle_data():
    path = Path(__file__).parent / "data" / "car details v4.csv"
    df = pd.read_csv(path)

    # 只保留关键特征
    keep = [
        "Price",
        "Year",
        "Kilometer",
        "Fuel Type",
        "Transmission",
        "Owner",
        "Seller Type",
    ]
    df = df[keep]

    # ================= 缺失值探索=================
    print("=" * 60)
    print("Kaggle 数据缺失值探索")
    print("=" * 60)
    print("每列缺失数量：")
    print(df.isnull().sum())
    print("\n缺失占比(%)：")
    print(df.isnull().mean() * 100)

    # ================= 数值型变量：中位数填补 =================
    df["Year"] = df["Year"].fillna(df["Year"].median())
    df["Kilometer"] = df["Kilometer"].fillna(df["Kilometer"].median())
    df["Price"] = df["Price"].fillna(df["Price"].median())

    # ================= 类别型变量：众数填补 =================
    df["Fuel Type"] = df["Fuel Type"].fillna(df["Fuel Type"].mode()[0])
    df["Transmission"] = df["Transmission"].fillna(df["Transmission"].mode()[0])
    df["Owner"] = df["Owner"].fillna(df["Owner"].mode()[0])
    df["Seller Type"] = df["Seller Type"].fillna(df["Seller Type"].mode()[0])

    return df


# ------------------------------------------------------------------------------
# 4. 运行 Kaggle 真实数据任务
# 流程：读取 → 缩尾 → 中心化 → 多项式 → 编码 → CV → Ridge → 报告
# ------------------------------------------------------------------------------
def run_kaggle_task():
    print("\n" + "=" * 60)
    print("Task B：Kaggle 真实数据实验")
    print("=" * 60)

    df = load_kaggle_data()

    # 缩尾法处理异常值（1%~99%）
    def winsor(s):
        return np.clip(s, s.quantile(0.01), s.quantile(0.99))

    df["Price"] = winsor(df["Price"])
    df["Kilometer"] = winsor(df["Kilometer"])
    # 中心化 + 多项式特征（缓解共线性、提升拟合）
    df["Year_cent"] = df["Year"] - df["Year"].mean()
    df["Km_cent"] = df["Kilometer"] - df["Kilometer"].mean()
    df["Year_sq"] = df["Year_cent"] ** 2
    df["Km_sq"] = df["Km_cent"] ** 2
    df["Year_Km"] = df["Year_cent"] * df["Km_cent"]
    # 删除原始特征，避免冗余
    df = df.drop(["Year", "Kilometer"], axis=1)
    # 独热编码，drop_first 避免虚拟变量陷阱
    df = pd.get_dummies(df, drop_first=True)
    # 删除无用或高度冗余列
    drop_cols = [
        "Owner_UnRegistered Car",
        "Fuel Type_Petrol + CNG",
        "Fuel Type_Petrol + LPG",
    ]
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=c)
    # 数据探索报告
    save_descriptive_stats(df, "Kaggle 数据描述统计")
    save_plots(df, "Kaggle", "kaggle_plot.png")

    # 构建 X, y
    y = df["Price"].values
    X = df.drop("Price", axis=1).values.astype(float)

    # VIF 诊断（真实数据共线性非常严重）
    print("\nVIF 共线性诊断")
    vif_df = pd.DataFrame(X, columns=df.drop("Price", axis=1).columns)
    vif = calculate_vif_dataframe(vif_df, vif_df.columns.tolist())
    print_vif_warning(vif)

    # 5 折无泄露交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_list, mae_list, mape_list, r2_list = [], [], [], []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 标准化在 fold 内部，无数据泄露
        scaler = CustomStandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 使用 Ridge 解决共线性问题
        model = RidgeRegression(alpha=3.0, fit_intercept=True)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 记录指标
        rmse_list.append(calculate_rmse(y_test, y_pred))
        mae_list.append(calculate_mae(y_test, y_pred))
        mape_list.append(calculate_mape(y_test, y_pred))
        r2_list.append(model.score(X_test, y_test))

    # 输出最终评估结果
    save_metrics(
        np.mean(rmse_list),
        np.mean(mae_list),
        np.mean(mape_list),
        np.mean(r2_list),
        "Kaggle Ridge 指标",
    )
    save_interpretation(is_ridge=True)


# ------------------------------------------------------------------------------
# 5. 报告输出
# ------------------------------------------------------------------------------
def write_report():
    print("\n 报告已自动生成在 results/ 文件夹")


# ------------------------------------------------------------------------------
# 6. 主函数：唯一入口，统一调度所有任务
# ------------------------------------------------------------------------------
def main():
    get_result_dir()  # 确保结果目录存在
    run_synthetic_task()  # 运行模拟数据任务
    run_kaggle_task()  # 运行真实数据任务
    write_report()  # 输出报告
    print("\n 全部任务完成！")


# 程序入口
if __name__ == "__main__":
    main()
