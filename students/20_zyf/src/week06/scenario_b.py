"""
Scenario B: Real-World Marketing Data with Multiple Market Instances
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ols_model import CustomOLS


def scenario_B_real_world(results_dir: Path):
    print("🔄 运行场景B（真实数据）...")
    # 定位真实数据文件
    root_dir = Path(__file__).parent.parent.parent.parent.parent
    data_path = root_dir / "homework" / "week06" / "data" / "q3_marketing.csv"
    print(f"📖 从以下位置读取数据: {data_path.absolute()}")
    
    # 读取真实数据
    if not data_path.exists():
        raise FileNotFoundError(f"数据文件未找到: {data_path}")
    # 重要：keep_default_na=False 防止pandas将'NA'识别为NaN
    df = pd.read_csv(data_path, keep_default_na=False)
    print(f"✅ 数据已加载！行数：{len(df)}, 列：{df.columns.tolist()}")

    # 修正列名，匹配真实数据
    X = np.hstack([
        np.ones((len(df), 1)), 
        df[["TV_Budget", "Radio_Budget", "SocialMedia_Budget", "Is_Holiday"]].values
    ])
    y = df["Sales"].values

    # 按地区拆分数据
    mask_na = df["Region"] == "NA"
    mask_eu = df["Region"] == "EU"
    X_na, y_na = X[mask_na], y[mask_na]
    X_eu, y_eu = X[mask_eu], y[mask_eu]
    print(f"✅ 数据已拆分：北美地区（{len(X_na)}行），欧洲地区（{len(X_eu)}行）")

    # 训练模型（适配无数据情况）
    model_na = CustomOLS()
    model_eu = CustomOLS()
    
    if len(X_na) > 0:
        model_na.fit(X_na, y_na)
        print(f"✅ 北美模型训练成功！")
    else:
        print(f"⚠️ 无北美地区数据，跳过北美模型训练")
    
    if len(X_eu) > 0:
        model_eu.fit(X_eu, y_eu)
        print(f"✅ 欧洲模型训练成功！")
    else:
        print(f"⚠️ 无欧洲地区数据，跳过欧洲模型训练")

    # F检验（广告效果：TV、Radio、SocialMedia）
    C = np.array([[0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0]])
    d = np.zeros(3)
    na_f = model_na.f_test(C, d)
    eu_f = model_eu.f_test(C, d)
    print(f"✅ F检验完成！北美F统计量: {na_f['f_stat']}, 欧洲F统计量: {eu_f['f_stat']}")

    # 1. 生成F检验结论（追加到summary_report.md）
    summary_path = results_dir / "summary_report.md"
    with open(summary_path, "a", encoding='utf-8') as f:
        f.write("\n\n# F检验结果（真实营销数据）\n\n")
        f.write("## 广告效果检验（TV+Radio+SocialMedia）\n")
        if len(X_na) > 0:
            f.write(f"- 北美地区：F统计量 = {na_f['f_stat']}, p值 = {na_f['p_value']}\n")
            f.write(f"  解释：{'显著效果（p<0.05）' if na_f['p_value'] < 0.05 else '无显著效果'}\n")
        else:
            f.write(f"- 北美地区：暂无数据\n")
        if len(X_eu) > 0:
            f.write(f"- 欧洲地区：F统计量 = {eu_f['f_stat']}, p值 = {eu_f['p_value']}\n")
            f.write(f"  解释：{'显著效果（p<0.05）' if eu_f['p_value'] < 0.05 else '无显著效果'}\n")
        else:
            f.write(f"- 欧洲地区：暂无数据\n")
    print(f"✅ F检验结论已保存到: {summary_path}")

    # 1.5 同时生成独立的真实数据报告文件
    real_world_path = results_dir / "real_world_report.md"
    with open(real_world_path, "w", encoding='utf-8') as f:
        f.write("# 场景B：真实营销数据分析\n\n")
        f.write("## 数据概览\n")
        f.write(f"- 总样本数：{len(df)}\n")
        f.write(f"- 北美地区（NA）：{len(X_na)}个样本\n")
        f.write(f"- 欧洲地区（EU）：{len(X_eu)}个样本\n\n")
        
        f.write("## 模型系数\n\n")
        if len(X_na) > 0:
            f.write("### 北美市场模型\n")
            f.write("| 参数 | 系数值 |\n")
            f.write("|------|--------|\n")
            f.write(f"| 截距 | {model_na.coef_[0]:.4f} |\n")
            f.write(f"| TV预算 | {model_na.coef_[1]:.4f} |\n")
            f.write(f"| 广播预算 | {model_na.coef_[2]:.4f} |\n")
            f.write(f"| 社交媒体预算 | {model_na.coef_[3]:.4f} |\n")
            f.write(f"| 是否假日 | {model_na.coef_[4]:.4f} |\n")
            f.write(f"| R² | {model_na.score(X_na, y_na):.4f} |\n\n")
        
        if len(X_eu) > 0:
            f.write("### 欧洲市场模型\n")
            f.write("| 参数 | 系数值 |\n")
            f.write("|------|--------|\n")
            f.write(f"| 截距 | {model_eu.coef_[0]:.4f} |\n")
            f.write(f"| TV预算 | {model_eu.coef_[1]:.4f} |\n")
            f.write(f"| 广播预算 | {model_eu.coef_[2]:.4f} |\n")
            f.write(f"| 社交媒体预算 | {model_eu.coef_[3]:.4f} |\n")
            f.write(f"| 是否假日 | {model_eu.coef_[4]:.4f} |\n")
            f.write(f"| R² | {model_eu.score(X_eu, y_eu):.4f} |\n\n")
        
        f.write("## F检验结果\n\n")
        f.write("### 广告效果联合检验\n")
        f.write("**虚无假设：** H₀: β_TV = β_Radio = β_Social = 0（广告无效）\n\n")
        
        if len(X_na) > 0:
            f.write("**北美市场：**\n")
            f.write(f"- F统计量：{na_f['f_stat']:.4f}\n")
            f.write(f"- p值：{na_f['p_value']:.6f}\n")
            f.write(f"- 结论：{'✅ 广告显著有效（p<0.05）' if na_f['p_value'] < 0.05 else '❌ 广告效果不显著'}\n\n")
        
        if len(X_eu) > 0:
            f.write("**欧洲市场：**\n")
            f.write(f"- F统计量：{eu_f['f_stat']:.4f}\n")
            f.write(f"- p值：{eu_f['p_value']:.6f}\n")
            f.write(f"- 结论：{'✅ 广告显著有效（p<0.05）' if eu_f['p_value'] < 0.05 else '❌ 广告效果不显著'}\n\n")
        
        f.write("## 关键发现\n\n")
        if len(X_na) > 0 and len(X_eu) > 0:
            f.write(f"- 北美市场R²：{model_na.score(X_na, y_na):.4f}，欧洲市场R²：{model_eu.score(X_eu, y_eu):.4f}\n")
            f.write(f"- {'北美' if model_na.score(X_na, y_na) > model_eu.score(X_eu, y_eu) else '欧洲'}市场的模型解释力更强\n")
        f.write("- 详细的可视化分析请参考配套的图表文件\n")
    
    print(f"✅ 真实数据报告已保存到: {real_world_path}")
    # 绘制残差散点图（以欧洲地区数据为例）
    plt.figure(figsize=(10, 5))
    if len(X_eu) > 0:
        y_eu_hat = model_eu.predict(X_eu)
        residuals = y_eu - y_eu_hat.flatten()
        plt.scatter(y_eu_hat, residuals, alpha=0.6, color="#2ca02c")
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        plt.title("残差图（欧洲地区销售预测）")
        plt.xlabel("预测销售额")
        plt.ylabel("残差")
        plt.grid(alpha=0.3)
    else:
        plt.text(0.5, 0.5, "欧洲地区无数据", ha="center", va="center")
    
    residual_plot_path = results_dir / "residual_plot.png"
    plt.savefig(residual_plot_path, bbox_inches="tight", dpi=100)
    plt.close()
    print(f"✅ 残差图已保存到: {residual_plot_path}")

    # 绘制F检验对比图
    plt.figure(figsize=(10, 4))
    regions = []
    f_stats = []
    colors = []
    if len(X_na) > 0:
        regions.append("北美地区")
        f_stats.append(na_f["f_stat"])
        colors.append("#1f77b4")
    if len(X_eu) > 0:
        regions.append("欧洲地区")
        f_stats.append(eu_f["f_stat"])
        colors.append("#ff7f0e")
    
    if regions:
        plt.bar(regions, f_stats, color=colors)
    else:
        plt.text(0.5, 0.5, "暂无有效地区数据", ha="center", va="center")
    
    plt.title("F检验：广告效果（电视+广播+社交媒体）")
    plt.ylabel("F统计量")
    plt.grid(axis="y", alpha=0.3)
    ftest_plot_path = results_dir / "ftest_comparison.png"
    plt.savefig(ftest_plot_path, bbox_inches="tight", dpi=100)
    plt.close()
    print(f"✅ F检验图已保存到: {ftest_plot_path}")