# 导入操作系统相关模块，用于文件/目录操作
import os
# 导入系统模块，用于修改Python导入路径
import sys
# 将项目根目录添加到Python搜索路径，解决utils模块导入报错问题
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入数值计算库，用于生成数据、数学运算
import numpy as np
# 导入数据处理库，用于保存实验结果为CSV文件
import pandas as pd
# 导入绘图库，用于生成可视化图表
import matplotlib.pyplot as plt
# 导入多项式特征构造工具，用于生成多项式回归的特征
from sklearn.preprocessing import PolynomialFeatures
# 导入流水线工具，用于封装多项式+回归的建模流程
from sklearn.pipeline import Pipeline
# 导入线性回归模型，作为基础拟合模型
from sklearn.linear_model import LinearRegression
# 导入数据集划分工具，用于拆分训练集和测试集
from sklearn.model_selection import train_test_split

# 从自己写的工具模块中，复用RMSE、MAE评估指标函数
from utils.metrics import rmse, mae

# -------------------------- 全局配置区 --------------------------
# 设置随机种子，保证实验结果可复现
np.random.seed(42)
# 定义图表保存路径
FIG_DIR = "students/18_mxt/src/week12/figures"
# 定义实验结果保存路径
RESULT_DIR = "students/18_mxt/src/week12/results"
# 自动创建图表和结果目录，若目录已存在则不报错
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# 统一配置Matplotlib绘图样式（字体、字号、分辨率等）
plt.rcParams.update({
    "font.size": 12,              # 全局字体大小
    "axes.titlesize": 14,         # 坐标轴标题大小
    "axes.labelsize": 12,         # 坐标轴标签大小
    "legend.fontsize": 10,        # 图例字体大小
    "figure.dpi": 100,            # 图片分辨率
    "font.family": "DejaVu Sans"  # 字体类型
})

# ------------------------------------------------------------------------------
# Task A: 构造"会过拟合"的可视化舞台
# ------------------------------------------------------------------------------
def generate_data(n_samples=200):
    """生成一维非线性回归数据（带噪声的正弦+线性函数）"""
    # 生成200个均匀分布的x值，范围0~10
    x = np.linspace(0, 10, n_samples)
    # 定义真实数据函数：sin(x) + 0.5x（正弦+线性趋势）
    y_true = np.sin(x) + 0.5 * x
    # 给真实数据添加高斯噪声（均值0，标准差0.3）
    y = y_true + np.random.normal(0, 0.3, n_samples)
    # 划分训练集(70%)和测试集(30%)，固定随机种子保证可复现
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42
    )
    # 返回完整数据：原始x、真实y、训练集、测试集
    return x, y_true, x_train, x_test, y_train, y_test

def run_candidate_models():
    # 打印当前执行阶段，方便查看运行进度
    print("[Stage 1/4] Comparing candidate model complexities...")
    # 调用函数生成实验数据
    x, y_true, x_train, x_test, y_train, y_test = generate_data()
    
    # 定义要对比的3个多项式阶数（欠拟合/最优/过拟合）
    degrees = [1, 4, 15]
    # 定义3个模型对应的绘图颜色
    colors = ["red", "green", "blue"]
    # 定义3个模型的图例标签
    labels = ["Degree 1 (Underfit)", "Degree 4 (Optimal)", "Degree 15 (Overfit)"]
    
    # 创建画布，设置图片大小
    plt.figure(figsize=(12, 6))
    # 绘制真实函数曲线（黑色虚线）
    plt.plot(x, y_true, "k--", label="True Function", linewidth=2)
    # 绘制训练集散点图，半透明显示
    plt.scatter(x_train, y_train, alpha=0.5, label="Train Set")
    # 绘制测试集散点图，用x标记，半透明显示
    plt.scatter(x_test, y_test, alpha=0.5, label="Test Set", marker="x")
    
    # 初始化列表，保存实验结果（阶数、训练RMSE、测试RMSE）
    results = []
    # 遍历3个多项式阶数，分别建模、绘图
    for deg, color, label in zip(degrees, colors, labels):
        # 构建流水线：多项式特征构造 + 线性回归
        model = Pipeline([
            ("poly", PolynomialFeatures(degree=deg)),
            ("reg", LinearRegression())
        ])
        # 训练模型（reshape是因为sklearn要求特征为二维数组）
        model.fit(x_train.reshape(-1, 1), y_train)
        
        # 对训练集、测试集做预测
        y_pred_train = model.predict(x_train.reshape(-1, 1))
        y_pred_test = model.predict(x_test.reshape(-1, 1))
        
        # 调用自定义工具计算RMSE指标
        train_rmse = rmse(y_train, y_pred_train)
        test_rmse = rmse(y_test, y_pred_test)
        # 将结果存入列表
        results.append((deg, train_rmse, test_rmse))
        
        # 生成平滑的x轴数据，用于绘制拟合曲线
        x_plot = np.linspace(0, 10, 1000)
        # 模型预测平滑曲线
        y_plot = model.predict(x_plot.reshape(-1, 1))
        # 绘制当前模型的拟合曲线，并标注指标
        plt.plot(x_plot, y_plot, color=color, 
                 label=f"{label}\nTrain RMSE: {train_rmse:.3f}\nTest RMSE: {test_rmse:.3f}")
    
    # 设置图表标题、坐标轴标签、图例、网格
    plt.title("Comparison of Model Complexity")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(alpha=0.3)
    # 保存图片到指定目录，裁剪多余空白
    plt.savefig(f"{FIG_DIR}/candidate_models.png", bbox_inches="tight")
    # 关闭画布，释放内存
    plt.close()
    
    # 将实验结果保存为CSV文件
    pd.DataFrame(results, columns=["Degree", "Train RMSE", "Test RMSE"]).to_csv(
        f"{RESULT_DIR}/candidate_models_results.csv", index=False
    )
    # 打印保存成功提示
    print(f"✅ Candidate models plot saved: {FIG_DIR}/candidate_models.png")
    # 返回实验结果
    return results

# ------------------------------------------------------------------------------
# Task B: 画出完整的复杂度-误差曲线
# ------------------------------------------------------------------------------
def run_complexity_sweep():
    print("[Stage 2/4] Sweeping model complexity range...")
    # 生成实验数据
    x, y_true, x_train, x_test, y_train, y_test = generate_data()
    
    # 定义最大多项式阶数
    max_degree = 18
    # 生成1~18的阶数列表
    degrees = range(1, max_degree + 1)
    # 初始化列表，保存训练/测试RMSE
    train_rmses = []
    test_rmses = []
    
    # 遍历所有阶数，建模并计算误差
    for deg in degrees:
        # 构建多项式回归流水线
        model = Pipeline([
            ("poly", PolynomialFeatures(degree=deg)),
            ("reg", LinearRegression())
        ])
        model.fit(x_train.reshape(-1, 1), y_train)
        
        # 预测并计算RMSE
        y_pred_train = model.predict(x_train.reshape(-1, 1))
        y_pred_test = model.predict(x_test.reshape(-1, 1))
        
        train_rmses.append(rmse(y_train, y_pred_train))
        test_rmses.append(rmse(y_test, y_pred_test))
    
    # 绘制误差曲线
    plt.figure(figsize=(10, 6))
    # 绘制训练集RMSE曲线（蓝色圆点线）
    plt.plot(degrees, train_rmses, "b-o", label="Train RMSE")
    # 绘制测试集RMSE曲线（红色圆点线）
    plt.plot(degrees, test_rmses, "r-o", label="Test RMSE")
    # 图表样式配置
    plt.title("Model Complexity vs Error")
    plt.xlabel("Polynomial Degree")
    plt.ylabel("RMSE")
    plt.xticks(degrees)
    plt.legend()
    plt.grid(alpha=0.3)
    # 保存图片
    plt.savefig(f"{FIG_DIR}/error_curves.png", bbox_inches="tight")
    plt.close()
    
    # 构造结果表，包含泛化间隙（测试误差-训练误差）
    df = pd.DataFrame({
        "Degree": degrees,
        "Train RMSE": train_rmses,
        "Test RMSE": test_rmses,
        "Generalization Gap": np.array(test_rmses) - np.array(train_rmses)
    })
    # 保存结果到CSV
    df.to_csv(f"{RESULT_DIR}/complexity_results.csv", index=False)
    print(f"✅ Complexity error curve saved: {FIG_DIR}/error_curves.png")
    print(f"✅ Full complexity data saved: {RESULT_DIR}/complexity_results.csv")
    return df

# ------------------------------------------------------------------------------
# Task C: 用重复抽样把 Variance 画出来
# ------------------------------------------------------------------------------
def run_variance_demo():
    print("[Stage 3/4] Demonstrating high variance model behavior...")
    # 重复抽样次数：10次
    n_repeats = 10
    # 对比2个阶数：低阶(低方差)、高阶(高方差)
    degrees = [2, 15]
    # 生成平滑的x轴数据，用于绘制真实函数
    x_true = np.linspace(0, 10, 1000)
    # 真实函数
    y_true = np.sin(x_true) + 0.5 * x_true
    
    # 创建1行2列的子图，分别展示2个阶数的方差效果
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    axes = [ax1, ax2]
    # 保存方差计算结果
    std_results = []
    
    # 遍历2个多项式阶数
    for i, deg in enumerate(degrees):
        ax = axes[i]
        # 绘制真实函数曲线
        ax.plot(x_true, y_true, "k--", label="True Function", linewidth=2)
        
        # 保存10次抽样的预测结果
        all_predictions = []
        # 重复10次：重新抽样数据→训练模型→预测
        for _ in range(n_repeats):
            # 重新生成数据
            x = np.linspace(0, 10, 200)
            y = np.sin(x) + 0.5 * x + np.random.normal(0, 0.3, 200)
            # 重新划分训练集
            x_train, _, y_train, _ = train_test_split(x, y, test_size=0.3)
            
            # 建模训练
            model = Pipeline([
                ("poly", PolynomialFeatures(degree=deg)),
                ("reg", LinearRegression())
            ])
            model.fit(x_train.reshape(-1, 1), y_train)
            y_pred = model.predict(x_true.reshape(-1, 1))
            all_predictions.append(y_pred)
            
            # 绘制单次拟合曲线（半透明叠加）
            ax.plot(x_true, y_pred, alpha=0.5)
        
        # 将10次预测结果转为数组，计算标准差
        all_predictions = np.array(all_predictions)
        # 平均预测标准差（衡量方差大小）
        mean_std = np.mean(np.std(all_predictions, axis=0))
        # 最大预测标准差
        max_std = np.max(np.std(all_predictions, axis=0))
        std_results.append((deg, mean_std, max_std))
        
        # 子图样式配置
        ax.set_title(f"Degree = {deg}\nMean Prediction Std: {mean_std:.3f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        ax.grid(alpha=0.3)
    
    # 总标题+保存图片
    plt.suptitle("Variance Comparison (10 Repeated Samplings)", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/variance_demo.png", bbox_inches="tight")
    plt.close()
    
    # 保存方差结果到CSV
    pd.DataFrame(std_results, columns=["Degree", "Mean Std", "Max Std"]).to_csv(
        f"{RESULT_DIR}/variance_results.csv", index=False
    )
    print(f"✅ Variance comparison plot saved: {FIG_DIR}/variance_demo.png")
    return std_results

# ------------------------------------------------------------------------------
# Task D: 异常值对 RMSE 与 MAE 的影响
# ------------------------------------------------------------------------------
def run_loss_comparison():
    print("[Stage 4/4] Comparing RMSE vs MAE sensitivity to outliers...")
    # 构造干净的真实值：100个正态分布数据
    y_true = np.random.normal(10, 2, 100)
    # 构造干净的预测值：真实值+小噪声
    y_pred_clean = y_true + np.random.normal(0, 0.5, 100)
    
    # 复制干净预测值，构造带异常值的预测值
    y_pred_outlier = y_pred_clean.copy()
    # 手动制造1个极端异常值（将第一个预测值改为100）
    y_pred_outlier[0] = 100
    
    # 计算干净数据的RMSE、MAE
    clean_rmse = rmse(y_true, y_pred_clean)
    clean_mae = mae(y_true, y_pred_clean)
    # 计算带异常值数据的RMSE、MAE
    outlier_rmse = rmse(y_true, y_pred_outlier)
    outlier_mae = mae(y_true, y_pred_outlier)
    
    # 创建画布，绘制对比图
    plt.figure(figsize=(12, 6))
    
    # 子图1：干净数据的预测效果
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred_clean, alpha=0.6)
    # 绘制理想预测对角线（y=x）
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
    plt.title(f"Clean Prediction\nRMSE: {clean_rmse:.3f}\nMAE: {clean_mae:.3f}")
    plt.xlabel("True Value")
    plt.ylabel("Predicted Value")
    plt.grid(alpha=0.3)
    
    # 子图2：带异常值的预测效果
    plt.subplot(1, 2, 2)
    plt.scatter(y_true, y_pred_outlier, alpha=0.6)
    # 红色标记异常值点
    plt.scatter(y_true[0], y_pred_outlier[0], color="red", s=100, label="Outlier")
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
    plt.title(f"With One Outlier\nRMSE: {outlier_rmse:.3f}\nMAE: {outlier_mae:.3f}")
    plt.xlabel("True Value")
    plt.ylabel("Predicted Value")
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 总标题+保存图片
    plt.suptitle("RMSE vs MAE Sensitivity to Outliers", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/loss_outlier_comparison.png", bbox_inches="tight")
    plt.close()
    
    # 构造结果表并保存
    loss_results = pd.DataFrame({
        "Scenario": ["Clean Prediction", "With One Outlier"],
        "RMSE": [clean_rmse, outlier_rmse],
        "MAE": [clean_mae, outlier_mae]
    })
    loss_results.to_csv(f"{RESULT_DIR}/loss_comparison_results.csv", index=False)
    print(f"✅ Loss function comparison plot saved: {FIG_DIR}/loss_outlier_comparison.png")
    return loss_results

# ------------------------------------------------------------------------------
# 主入口函数：执行所有实验
# ------------------------------------------------------------------------------
def main():
    # 打印项目标题，格式化输出
    print("=" * 60)
    print("Week 12: Bias-Variance Tradeoff Visual Lab")
    print("=" * 60)
    print("⚠️  This script only runs experiments and generates plots/data")
    print("⚠️  Please write your report manually to: results/summary.md")
    print("=" * 60 + "\n")
    
    # 按顺序执行4个实验任务
    run_candidate_models()
    run_complexity_sweep()
    run_variance_demo()
    run_loss_comparison()
    
    # 打印完成提示
    print("\n" + "=" * 60)
    print("✅ All experiments completed successfully!")
    print(f"📊 All plots saved to: {FIG_DIR}/")
    print(f"📊 All experiment data saved to: {RESULT_DIR}/")
    print("\n📝 Now you can write your complete results/summary.md report")
    print("📝 based on the generated data and plots")
    print("=" * 60)

# 程序入口：当直接运行此文件时，执行main函数
if __name__ == "__main__":
    main()