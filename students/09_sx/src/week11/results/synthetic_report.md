# Synthetic Data Regression Report

## 1. 数据生成机制 (DGP)

### 业务场景
- **场景**: 电商平台销售额预测
- **样本量**: 500
- **目标变量**: sales（销售额，单位：元）

### DGP公式
### sales = 3000 + 60*ad_budget + 10*website_traffic + 500*product_quality + 300*customer_service + platform_effect + ε

### 变量影响方向（预期）
| 变量 | 预期系数 | 方向 | 业务含义 |
|------|----------|------|----------|
| ad_budget | +60 | 正向 | 广告预算，投入越多销售额越高 |
| website_traffic | +10 | 正向 | 网站访问量，流量越多销售额越高 |
| product_quality | +500 | 正向 | 产品质量评分，质量越好销售额越高 |
| customer_service | +300 | 正向 | 客服评分，服务越好销售额越高 |
| platform_type_B | -2000 | 负向 | B平台相对于A平台效果更差 |
| platform_type_C | -5000 | 负向 | C平台相对于A平台效果最差 |

### 构造的数据问题
1. **共线性**: `ad_budget` 与 `website_traffic` 高度相关（r≈0.8）
2. **缺失值**: `product_quality` 中10%随机缺失
3. **异常值**: `customer_service` 中5%极端值
4. **量纲差异**: 各特征取值范围不同

## 2. 交叉验证结果

### 模型性能对比（5折CV均值 ± 标准差）

| 模型 | RMSE | MAE | MAPE (%) | R² |
|------|------|-----|----------|-----|
| AnalyticalOLS (自己的) | 1131.49 ± 63.08 | 907.69 | 3.67 | 0.9785 ± 0.0033 |
| GradientDescentOLS (自己的) | 1133.61 ± 58.94 | 909.09 | 3.68 | 0.9784 ± 0.0032 |
| sklearn LinearRegression (baseline) | 1131.49 ± 63.08 | 907.69 | 3.67 | 0.9785 ± 0.0033 |

## 3. 系数分析（标准化后）

### 系数方向一致性检验
| 特征 | 预期方向 | 识别系数 | 方向一致？ |
|------|----------|----------|------------|
| ad_budget | 正向 | 3345.1664 | ✓ |
| website_traffic | 正向 | 3484.9262 | ✓ |
| product_quality | 正向 | 1000.0755 | ✓ |
| customer_service | 正向 | 657.5710 | ✓ |
| platform_type_B | 负向 | -3188.3448 | ✓ |
| platform_type_C | 负向 | -1946.3141 | ✓ |

### 推测结论
- **所有特征方向与DGP一致** ✓
- `ad_budget` 和 `website_traffic` 的系数都为正，符合业务逻辑

## 4. 共线性诊断 (VIF)

| 特征 | VIF值 | 判断 |
|------|-------|------|
| ad_budget | 8.11 | ⚠️ 中等共线性 |
| website_traffic | 8.09 | ⚠️ 中等共线性 |
| product_quality | 1.01 | ✓ 可接受 |
| customer_service | 1.00 | ✓ 可接受 |
| platform_type_B | 1.12 | ✓ 可接受 |
| platform_type_C | 1.13 | ✓ 可接受 |

## 5. 残差分析

| 统计量 | 值 | 判断 |
|--------|-----|------|
| 残差均值 | 0.000000 | ✓ |
| 残差偏度 | 0.1740 | ✓ |
| 残差峰度 | 0.0442 | ✓ |

## 6. 可视化输出

- 相关矩阵图: `results/synthetic_correlation.png`
- 残差诊断图: `results/synthetic_residuals.png`
