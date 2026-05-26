# 🌍 Kaggle Real Data Report — 真实数据回归报告

## 1. 数据信息
- **数据集名称**: Medical Cost Personal Dataset
- **Kaggle 链接**: https://www.kaggle.com/datasets/mirichoi0218/insurance
- **下载日期**: 2026-05-19
- **目标变量**: `charges` (医疗费用, 连续变量, 美元)
- **每行样本**: 一位参保人的人口统计信息及年度医疗费用
- **原始特征**: age, sex, bmi, children, smoker, region, charges

## 2. 选择理由
该数据集混合了数值变量 (age, bmi, children) 和类别变量 (sex, smoker, region),
存在明显的偏态 (charges 右偏)、缺失值少但存在异常值, 且业务含义清晰,
适合作为回归分析的真实场景练习。

## 3. 预处理说明
- 类别变量 (sex, smoker, region) → OneHot 编码
- 缺失值: 在每折 CV 内用训练集均值填补 (CustomImputer)
- 标准化: 在每折 CV 内用训练集参数标准化 (CustomStandardScaler)
- 离群值: 全局 Winsorization (1%-99%), 不学参数, 无泄露

## 4. 交叉验证结果 (CustomOLS, 5-Fold Leak-Free)

| Fold | RMSE | MAE | MAPE(%) |
|------|------|-----|---------|
| 1 | 6005.5039 | 4062.5799 | 40.95 |
| 2 | 5785.5001 | 3912.7386 | 36.21 |
| 3 | 6453.8978 | 4511.9291 | 41.00 |
| 4 | 6088.8728 | 4284.9880 | 49.56 |
| 5 | 5980.1547 | 4209.3103 | 44.28 |
| **Mean** | **6062.7858** | **4196.3092** | **42.40** |

## 5. 对比模型
- CustomOLS 平均 RMSE: 6062.7858, MAE: 4196.3092, MAPE: 42.40%
- GradientDescentOLS 平均 RMSE: 14584.9402, MAE: 13268.7558, MAPE: 207.41%
- sklearn LinearRegression 平均 RMSE: 6079.6911, MAE: 4208.0400, MAPE: 42.51%

## 6. VIF 诊断

| Feature | VIF | 判断 |
|---------|-----|------|
| age | 1.02 | ✅ 正常 |
| bmi | 1.11 | ✅ 正常 |
| children | 1.00 | ✅ 正常 |
| sex_male | 1.01 | ✅ 正常 |
| smoker_yes | 1.01 | ✅ 正常 |
| region_northwest | 1.52 | ✅ 正常 |
| region_southeast | 1.65 | ✅ 正常 |
| region_southwest | 1.53 | ✅ 正常 |

## 7. 系数方向分析

| Feature | Coefficient | 解读 |
|---------|-------------|------|
| age | +3530.5542 | 正向 |
| bmi | +2029.5711 | 正向 |
| children | +547.0492 | 正向 |
| sex_male | -140.9503 | 负向 |
| smoker_yes | +9513.6596 | 正向 |
| region_northwest | -132.2814 | 负向 |
| region_southeast | -292.4826 | 负向 |
| region_southwest | -297.2238 | 负向 |

## 8. 推测与业务解读
- **smoker** 极可能是最强预测变量 (吸烟者费用显著更高)
- **age** 正向影响: 年长者费用更高
- **bmi** 正向: 较高 BMI 与更高费用相关
- 模型平均误差 (RMSE) 代表预测费用与实际费用的典型偏差
- 若用于真实决策, 需注意: 地区/性别差异可能反映系统性偏见而非因果关系

## 9. 上线风险
- 数据为美国参保人群, 不一定适用于其他人群
- smoker 的强效应可能导致模型过度依赖单一变量
- 费用分布严重右偏, MAPE 可能较高