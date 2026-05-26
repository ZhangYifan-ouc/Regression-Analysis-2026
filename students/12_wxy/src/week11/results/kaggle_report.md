# Kaggle 二手车数据报告
## 1. 最稳定影响变量
- Present_Price、Year、Kms_Driven

## 2. 直觉重要但模型不稳定的变量
- 类别变量（Fuel_Type、Transmission）
- 因样本分布不均，系数波动大

## 3. 共线性 & 异常值
- Present_Price 与 Year 强相关
- Kms_Driven 存在极端异常值

## 4. 业务误差解释
RMSE ≈ 1.82 万元
二手车价格预测中属于可接受范围。

## 5. 上线风险
1. 共线性导致特征重要性不可靠
2. 豪车/极端车预测偏差大
3. 市场波动会让模型失效
4. 缺少车况、地区信息

## 6. 模型指标
- RMSE：1.82
- MAE：1.18
- MAPE：129.12%

## 7. VIF 共线性
高VIF变量：['Fuel_Type_Diesel: 19.9', 'Fuel_Type_Petrol: 19.6']
