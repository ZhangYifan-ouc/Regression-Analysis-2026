from pathlib import Path
from utils import setup_results_dir
from scenario_a import scenario_A_synthetic
from scenario_b import scenario_B_real_world


def main():
    """Main entry point - orchestrates all scenarios and report generation."""
    
    print("\n" + "="*70)
    print("第6周里程碑项目1：推断引擎与真实数据回归分析")
    print("="*70)
    
    # Setup results directory
    results_dir = setup_results_dir()
    
    # Find data file
    # The data should be in ../../data/ relative to this script
    script_dir = Path(__file__).parent
    data_path = script_dir.parent.parent.parent.parent / "homework" / "week06" / "data" / "q3_marketing.csv"
    
    if not data_path.exists():
        print(f"❌ 错误：数据文件未找到 {data_path}")
        return
    
    print(f"\n✅ 数据文件已找到：{data_path}")
    
    # Run Scenario A
    try:
        scenario_A_result = scenario_A_synthetic(results_dir)
        print("\n✓ 场景A已完成")
    except Exception as e:
        print(f"\n✗ 场景A执行失败：{e}")
        import traceback
        traceback.print_exc()
        return
    
    # Run Scenario B
    try:
        scenario_B_result = scenario_B_real_world(results_dir)
        print("\n✓ 场景B已完成")
    except Exception as e:
        print(f"\n✗ 场景B执行失败：{e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create summary report
    summary_path = results_dir / "summary_report.md"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# 第6周里程碑项目1：完整分析总结\n\n")
        f.write("## 项目概述\n")
        f.write("本项目包含以下内容：\n")
        f.write("- **CustomOLS** 类：用NumPy手工实现的OLS回归引擎\n")
        f.write("- **evaluate_model()** 通用函数：展示鸭子类型的应用\n")
        f.write("- **场景A**：合成数据白盒测试\n")
        f.write("- **场景B**：真实数据双市场分析与F检验\n\n")
        f.write("## 核心设计选择\n\n")
        f.write("### 面向对象编程（类方式）\n")
        f.write("我们选择了**类方式**而非过程式函数，原因如下：\n")
        f.write("1. **封装性**：每个模型实例维护自己的状态（coef_、cov_matrix_等）\n")
        f.write("2. **多实例安全**：NA和EU市场的模型能够共存而不相互干扰\n")
        f.write("3. **方法链**：`model.fit().predict()` 提高代码可读性\n")
        f.write("4. **鸭子类型**：与sklearn的API无缝兼容\n\n")
        f.write("### 截距项处理\n")
        f.write("在拟合前，我们在X中添加全1列作为截距项，使其成为一个普通系数。\n")
        f.write("这与sklearn中fit_intercept=True的行为一致。\n\n")
        f.write("## 模块结构\n")
        f.write("- `ols_model.py`：CustomOLS类实现\n")
        f.write("- `evaluator.py`：通用evaluate_model()函数\n")
        f.write("- `scenario_a.py`：合成数据基准测试\n")
        f.write("- `scenario_b.py`：真实营销数据分析\n")
        f.write("- `utils.py`：工具函数（setup_results_dir）\n")
        f.write("- `main.py`：主入口（当前文件）\n\n")
        f.write("## 生成的报告文件\n")
        f.write("- `synthetic_report.md`：场景A结果\n")
        f.write("- `real_world_report.md`：场景B分析\n")
        f.write("- `market_comparison.png`：可视化（实际vs预测、残差图）\n")
        f.write("- `summary_report.md`：本文件\n\n")
    
    print(f"\n✓ Summary report generated: {summary_path}")
    
    print("\n" + "="*70)
    print("所有场景执行完成！")
    print("="*70)
    print(f"\n结果已保存到：{results_dir}")
    print("生成的文件：")
    for file in sorted(results_dir.glob("*")):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()
