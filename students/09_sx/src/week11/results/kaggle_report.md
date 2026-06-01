# Kaggle Real Data Regression Report

## 1. 数据集信息

| 属性 | 内容 |
|------|------|
| 数据集名称 | House Prices - Advanced Regression Techniques |
| Kaggle链接 | https://www.kaggle.com/c/house-prices-advanced-regression-techniques |
| 目标变量 | SalePrice（房屋售价，美元） |
| 样本量 | 1460 |
| 原始特征数 | 81 |

### 选择的特征

| 特征 | 类型 | 业务含义 | 缺失率 |
|------|------|----------|--------|
| OverallQual | 数值 | 整体材料及装修质量（1-10分） | 0% |
| GrLivArea | 数值 | 地上居住面积（平方英尺） | 0% |
| LotFrontage | 数值 | 临街长度（英尺） | 17.7% |
| BsmtFinSF1 | 数值 | 地下室装修面积（平方英尺） | 0.0% |
| GarageArea | 数值 | 车库面积（平方英尺） | 0.0% |
| Neighborhood | 类别 | 社区位置（25个类别） | 0% |

### 数据问题
- **缺失值**: LotFrontage, BsmtFinSF1, GarageArea 存在缺失值
- **异常值**: SalePrice 存在 61 个异常值 (4.2%)，主要是豪宅价格

## 2. 交叉验证结果

### 模型性能对比（5折CV均值 ± 标准差）

| 模型 | RMSE ($) | MAE ($) | MAPE (%) | R² |
|------|----------|---------|----------|-----|
| AnalyticalOLS (自己的) | $38,733 ± $3,577 | $24,693 | 14.68 | 0.7605 ± 0.0237 |
| GradientDescentOLS (自己的) | - | - | - | 0.7600 |
| sklearn LinearRegression (baseline) | - | - | - | 0.7605 |

## 3. 最重要特征系数（标准化后）

| 特征 | 系数 | 解释 |
|------|------|------|
| GrLivArea | 26885.4774 | 正向影响房价 |
| OverallQual | 26129.1269 | 正向影响房价 |
| BsmtFinSF1 | 15835.7189 | 正向影响房价 |
| GarageArea | 11245.4777 | 正向影响房价 |
| Neighborhood_NridgHt | 9963.1585 | 正向影响房价 |
| LotFrontage | 7812.4212 | 正向影响房价 |
| Neighborhood_OldTown | -6089.4979 | 负向影响房价 |
| Neighborhood_NAmes | -5852.2688 | 负向影响房价 |

## 4. 共线性诊断 (VIF)

| 特征 | VIF值 | 判断 |
|------|-------|------|
| Neighborhood_NAmes | 12.97 | ⚠️ 严重共线性 |
| Neighborhood_CollgCr | 9.00 | ⚠️ 中等共线性 |
| Neighborhood_OldTown | 7.46 | ⚠️ 中等共线性 |
| Neighborhood_Edwards | 6.95 | ⚠️ 中等共线性 |
| Neighborhood_Somerst | 5.75 | ⚠️ 中等共线性 |
| Neighborhood_Sawyer | 5.50 | ⚠️ 中等共线性 |
| Neighborhood_Gilbert | 5.49 | ⚠️ 中等共线性 |
| Neighborhood_NridgHt | 5.46 | ⚠️ 中等共线性 |
| Neighborhood_NWAmes | 5.24 | ⚠️ 中等共线性 |
| Neighborhood_BrkSide | 4.49 | ✓ 可接受 |

## 5. 残差分析

| 统计量 | 值 | 判断 |
|--------|-----|------|
| 残差均值 | $0.00 | ✓（接近0） |
| 残差偏度 | 2.7220 | ⚠️（右偏） |
| 残差峰度 | 23.2152 | ⚠️（高峰度） |

## 6. 推测结论

### 稳定变量
- **GrLivArea**: 居住面积，最稳定的正向因素
- **OverallQual**: 整体质量，强正向影响

### 业务解释
- 平均绝对误差约 $24,693，模型预测房价平均误差约 $24,693
- R² = 0.7605，模型能解释约 76.0% 的房价变异

## 7. 可视化输出

- 相关矩阵图: `results/kaggle_correlation.png`
- 残差诊断图: `results/kaggle_residuals.png`
