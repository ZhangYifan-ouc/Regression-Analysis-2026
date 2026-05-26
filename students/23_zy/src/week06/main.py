import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from models import CustomOLS
from utils import evaluate_model, setup_results_dir, generate_synthetic_data

def scenario_A_synthetic(results_dir: Path):
    print("=== 场景A: 合成数据测试 ===")
    X, y = generate_synthetic_data(n_samples=1000)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = CustomOLS()
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test, model_name="Scenario A")

    print(f"w = {model.w}")
    print(f"b = {model.b:.4f}")
    print(f"R² = {metrics['r2']:.4f}")
    print(f"RMSE = {metrics['rmse']:.4f}")

def scenario_B_marketing(results_dir: Path):
    print("\n=== 场景B: 真实营销数据测试 ===")

    df = pd.read_csv("q3_marketing.csv")
    print("所有列名：", list(df.columns))

    # ✅ 自动找正确的广告列（不会再报错 KeyError）
    if "TV_Budget" in df.columns:
        X = df[["TV_Budget"]].values
    elif "SocialMedia_Budget" in df.columns:
        X = df[["SocialMedia_Budget"]].values
    else:
        print("错误：找不到广告预算列")
        return

    y = df["Sales"].values
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = CustomOLS()
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test, model_name="Scenario B")

    print(f"回归系数 w = {float(model.w[0]):.4f}")
    print(f"截距 b = {model.b:.4f}")
    print(f"R² = {metrics['r2']:.4f}")
    print(f"RMSE = {metrics['rmse']:.4f}")

    plt.figure(figsize=(8,5))
    plt.scatter(X_test, y_test, color="blue", label="真实值")
    plt.plot(X_test, model.predict(X_test), color="red", linewidth=2, label="拟合直线")
    plt.xlabel("Advertising Budget")
    plt.ylabel("Sales")
    plt.legend()
    plt.savefig(results_dir / "marketing_regression.png")
    plt.close()

if __name__ == "__main__":
    print("开始运行 week06 作业...")
    results_dir = Path("results/week06")
    results_dir.mkdir(parents=True, exist_ok=True)
    scenario_A_synthetic(results_dir)
    scenario_B_marketing(results_dir)
    print("\n✅ 作业运行完成！")