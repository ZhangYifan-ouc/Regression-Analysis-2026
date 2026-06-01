import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# ========== 关键：配置中文支持 ==========
# ========== 终极中文配置 ==========
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 导入自定义评估指标
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.metrics import calculate_rmse, calculate_mae


# ====================== 路径初始化（加固，避免创建失败） ======================
BASE = Path(__file__).resolve().parent
RESULT_DIR = BASE / "results"
FIGURE_DIR = RESULT_DIR / "figures"
# 强制递归创建目录
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# 全局绘图样式
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.grid": True
})

# ====================== 实验数据生成 ======================
def generate_data(n=150, noise=0.3):
    np.random.seed(42)
    x = np.linspace(0, 4, n).reshape(-1, 1)
    y_true = np.sin(x).ravel()
    y_noisy = y_true + np.random.normal(0, noise, n)
    return x, y_noisy, y_true

# ====================== Task A：三模型对比 ======================
def run_candidate_models():
    print("[Stage 1] 执行多模型复杂度对比实验...")
    x, y, y_true = generate_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    x_plot = np.linspace(0, 4, 200).reshape(-1, 1)

    degrees = [1, 4, 15]
    colors = ["orange", "green", "red"]
    case_desc = ["Degree1-低复杂度", "Degree4-适中复杂度", "Degree15-高复杂度"]

    plt.scatter(x_train, y_train, label="训练样本", s=30, alpha=0.7)
    plt.scatter(x_test, y_test, label="测试样本", s=30, marker="^", alpha=0.7)
    plt.plot(x_plot, np.sin(x_plot), "b--", linewidth=2.5, label="真实非线性函数 sin(x)")

    model_records = []
    for d, c, desc in zip(degrees, colors, case_desc):
        pipeline = Pipeline([
            ("poly", PolynomialFeatures(degree=d)),
            ("lr", LinearRegression())
        ])
        pipeline.fit(x_train, y_train)
        train_pred = pipeline.predict(x_train)
        test_pred = pipeline.predict(x_test)
        train_rmse = calculate_rmse(y_train, train_pred)
        test_rmse = calculate_rmse(y_test, test_pred)
        plt.plot(x_plot, pipeline.predict(x_plot), color=c, linewidth=2,
                 label=f"{desc} | TrainRMSE:{train_rmse:.2f} TestRMSE:{test_rmse:.2f}")
        model_records.append([d, train_rmse, test_rmse])

    plt.title("不同模型复杂度拟合效果对比（欠拟合/最优/过拟合）")
    plt.xlabel("自变量 X")
    plt.ylabel("因变量 Y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "candidate_models.png", dpi=150)
    plt.close()
    return model_records

# ====================== Task B：完整复杂度误差曲线 ======================
def run_error_curve():
    print("[Stage 2] 扫描1-18阶多项式，生成偏差方差误差曲线...")
    x, y, _ = generate_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    degrees = list(range(1, 19))
    train_rmses, test_rmses, gap_list = [], [], []

    for d in degrees:
        pipeline = Pipeline([
            ("poly", PolynomialFeatures(degree=d)),
            ("lr", LinearRegression())
        ])
        pipeline.fit(x_train, y_train)
        tr_rmse = calculate_rmse(y_train, pipeline.predict(x_train))
        te_rmse = calculate_rmse(y_test, pipeline.predict(x_test))
        train_rmses.append(tr_rmse)
        test_rmses.append(te_rmse)
        gap_list.append(te_rmse - tr_rmse)

    plt.plot(degrees, train_rmses, "o-", color="#2E86AB", label="训练集 RMSE")
    plt.plot(degrees, test_rmses, "o-", color="#A23B72", label="测试集 RMSE")
    plt.xlabel("多项式阶数（模型复杂度）")
    plt.ylabel("RMSE 误差值")
    plt.title("模型复杂度 - 偏差方差权衡曲线")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "error_curves.png", dpi=150)
    plt.close()

    return degrees, train_rmses, test_rmses, gap_list

# ====================== Task C：方差可视化 ======================
def run_variance_demo():
    print("[Stage 3] 执行重复抽样方差对比实验...")
    x_plot = np.linspace(0, 4, 200).reshape(-1, 1)
    true_curve = np.sin(x_plot)
    std_summary = {}

    for degree, title in [(2, "低方差模型 Degree=2"), (15, "高方差模型 Degree=15")]:
        plt.figure()
        plt.plot(x_plot, true_curve, "r--", linewidth=2, label="真实函数")
        pred_all = []
        for seed in range(10):
            x, y, _ = generate_data()
            x_tr, _, y_tr, _ = train_test_split(x, y, test_size=0.3, random_state=seed)
            pipeline = Pipeline([
                ("poly", PolynomialFeatures(degree=degree)),
                ("lr", LinearRegression())
            ])
            pipeline.fit(x_tr, y_tr)
            pred = pipeline.predict(x_plot)
            pred_all.append(pred)
            plt.plot(x_plot, pred, alpha=0.3, color="blue")

        pred_all = np.array(pred_all)
        mean_std = np.mean(np.std(pred_all, axis=0))
        max_std = np.max(np.std(pred_all, axis=0))
        std_summary[degree] = [mean_std, max_std]

        plt.title(title)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / f"variance_demo_{degree}.png", dpi=150)
        plt.close()
    return std_summary

# ====================== Task D：RMSE/MAE 异常值对比 ======================
def run_loss_comparison():
    print("[Stage 4] 执行RMSE、MAE异常值敏感度对比实验...")
    y_true = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y_pred_clean = np.array([1.1, 2.0, 3.2, 3.9, 5.1, 5.8, 7.1, 8.0, 9.0, 10.1])
    y_pred_outlier = y_pred_clean.copy()
    y_pred_outlier[-1] = 100

    res = {
        "clean_rmse": calculate_rmse(y_true, y_pred_clean),
        "clean_mae": calculate_mae(y_true, y_pred_clean),
        "outlier_rmse": calculate_rmse(y_true, y_pred_outlier),
        "outlier_mae": calculate_mae(y_true, y_pred_outlier)
    }

    x = np.array([0, 1])
    plt.bar(x-0.2, [res["clean_rmse"], res["outlier_rmse"]], width=0.4, label="RMSE")
    plt.bar(x+0.2, [res["clean_mae"], res["outlier_mae"]], width=0.4, label="MAE")
    plt.xticks(x, ["正常预测", "含极端异常值"])
    plt.title("异常值对 RMSE、MAE 损失的影响对比")
    plt.ylabel("误差数值")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "loss_outlier_comparison.png", dpi=150)
    plt.close()
    return res

# ====================== 生成完整报告 ======================
def write_full_report(degrees, trs, tes, gaps, var_summary, loss_res):
    print("[Stage 5] 写入实验报告 report.md ...")
    best_idx = np.argmin(tes)
    best_degree = degrees[best_idx]

    report_content = """# Week12 偏差-方差权衡可视化实验报告
## 一、实验目的
本次实验通过可控模拟数据与多项式拟合可视化，直观理解偏差-方差权衡关系。主要验证三点：
1. 模型复杂度如何影响欠拟合与过拟合；
2. 高偏差、高方差的可视化特征与本质；
3. RMSE、MAE 对异常值的敏感度差异及业务选择逻辑；
同时为后续正则化算法学习铺垫基础。

## 二、实验数据构造
1. 采用非线性真实函数：$y = \sin(x)$；
2. 生成150组样本，叠加高斯噪声模拟真实业务数据；
3. 按照7:3比例划分训练集与测试集；
4. 固定随机种子，保证所有实验结果可复现。

## 三、实验一：多复杂度模型对比
### 操作步骤
选取3种典型多项式阶数：1阶、4阶、15阶，分别代表低、中、高模型复杂度。
使用相同训练集拟合模型，相同测试集评估效果，计算RMSE并绘制拟合曲线。

### 实验现象与结论
1. 1阶多项式：曲线简单，无法捕捉非线性规律，属于**高偏差、欠拟合**，训练、测试误差均偏高；
2. 4阶多项式：贴合真实函数趋势，偏差与方差达到平衡，泛化能力最优；
3. 15阶多项式：曲线剧烈震荡，过度学习训练集噪声，属于**高方差、过拟合**，训练误差极低，测试误差显著上升。

### 上线选择
优先选择4阶多项式模型，该模型综合表现最优，兼顾拟合能力与泛化能力。

## 四、实验二：全复杂度误差曲线
### 操作步骤
遍历1~18阶多项式，逐阶训练模型，记录训练RMSE、测试RMSE，计算泛化差距（测试误差-训练误差），绘制误差变化曲线。

### 实验数据表格
| 多项式阶数 | 训练RMSE | 测试RMSE | 泛化Gap |
"""
    # 拼接表格数据
    for d, tr, te, g in zip(degrees, trs, tes, gaps):
        report_content += f"| {d} | {tr:.2f} | {te:.2f} | {g:.2f} |\n"

    report_content += f"""
### 结果分析
1. 复杂度升高，训练误差持续下降，模型对训练数据拟合越来越充分；
2. 测试误差先降后升，本次实验最优复杂度为 {best_degree} 阶；
3. 低复杂度区间泛化差距小，主要问题为欠拟合；高复杂度区间泛化差距持续拉大，出现严重过拟合；
4. 训练误差最小的模型往往泛化能力最差，不能作为上线选择。

## 五、实验三：方差可视化实验
### 操作步骤
固定真实函数，重复10次随机抽取训练集；分别使用2阶（低复杂度）、15阶（高复杂度）模型训练；叠加所有拟合曲线，并计算预测值标准差量化方差。

### 定量结果
- 2阶模型：平均预测标准差 {var_summary[2][0]:.3f}，最大标准差 {var_summary[2][1]:.3f}
- 15阶模型：平均预测标准差 {var_summary[15][0]:.3f}，最大标准差 {var_summary[15][1]:.3f}

### 结论
低复杂度模型曲线稳定，方差小；高复杂度模型曲线波动极大，方差很高。

### 概念填空
high variance model 的危险，不是它不会拟合训练集，而是它对 **训练样本的微小波动** 过于敏感。

## 六、实验四：异常值对损失函数的影响
### 操作步骤
构造一组正常预测数据，再人为修改单个样本制造极端异常值，分别计算两组数据的RMSE与MAE，对比指标变化。

### 实验结果
| 实验场景 | RMSE | MAE |
|----------|------|-----|
| 正常预测 | {loss_res['clean_rmse']:.2f} | {loss_res['clean_mae']:.2f} |
| 含极端异常值 | {loss_res['outlier_rmse']:.2f} | {loss_res['outlier_mae']:.2f} |

### 原理与业务解释
1. RMSE 对误差做平方运算，会放大极端误差，因此对异常值高度敏感；
2. MAE 基于绝对值计算，受单个极端误差影响更小，鲁棒性更强；
3. 业务选择：重视重大错误、数据干净选RMSE；数据异常多、追求指标稳定选MAE。

## 七、核心结论
1. 偏差与方差存在固有权衡，模型复杂度是调节二者的核心手段；
2. 过拟合的本质是模型方差过高，学习到了训练集噪声；
3. 损失函数决定模型的风险偏好，需根据数据特征与业务需求选择。

## 八、代表性可视化图表
`candidate_models.png` 最能体现过拟合现象：高复杂度模型刻意拟合噪声，与真实规律偏离，训练和测试误差差异巨大。

## 九、指标选择场景
- 选用RMSE：数据干净、异常值少，需要严厉惩罚大预测误差；
- 选用MAE：数据噪声多、异常值普遍，要求指标稳定可靠。

## 十、正则化衔接说明
高复杂度模型存在高方差、泛化能力差的问题。正则化（Ridge/Lasso）通过对模型系数施加惩罚，主动降低模型复杂度、压制方差，在小幅提升偏差的前提下，整体提升模型泛化能力，是解决过拟合的常用方案。
"""
    # 写入文件
    md_path = RESULT_DIR / "report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"✅ 报告已成功写入：{md_path}")

# ====================== 主入口 ======================
def main():
    run_candidate_models()
    degrees, trs, tes, gaps = run_error_curve()
    var_std_res = run_variance_demo()
    loss_res = run_loss_comparison()
    write_full_report(degrees, trs, tes, gaps, var_std_res, loss_res)
    print("\n✅ Week12 全部实验执行完毕！")

if __name__ == "__main__":
    main()