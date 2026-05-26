# 📐 Synthetic Data Report — 模拟回归数据报告

## 1. 数据生成机制 (DGP)

**场景**: 房屋价格预测 (简化版)

**特征**:
- `area_sqft`: 房屋面积 (sqft), ~Uniform(500, 4000)
- `bedrooms`: 卧室数量, 与 area_sqft 高度相关 (共线性来源)
- `age_years`: 房龄, ~Gamma(shape=2, scale=8), 右偏分布
- `location`: 地段等级 A / B / C (类别变量)

**目标变量 price_wan (房价, 万元) 生成公式**:
```
y = 50 + 0.8 * area_sqft - 1.2 * age_years
    + 15 * (location==A) + 10 * (location==B) + ε
ε ~ N(0, 15²)
```

**预期方向** (注: location 采用 drop-first 编码, A=基线):
- `area_sqft` → 正向
- `age_years` → 负向
- `bedrooms` → 与面积共线, 系数可能不稳定
- `location_B` → 弱于 A (系数预期为负 vs 基线A)
- `location_C` → 弱于 A (系数预期为负 vs 基线A)

## 2. 注入的「真实世界问题」
- **缺失值**: 5% 的 `age_years` 设为 NaN
- **异常值**: 3% 样本的面积×3 或房龄置为 60-100 年
- **共线性**: bedrooms ≈ 0.003 × area_sqft + noise
- **量纲差异**: 面积 (500-4000) vs 卧室 (1-8) vs 房龄 (0-100)

## 3. 交叉验证结果 (CustomOLS, 5-Fold Leak-Free)

| Fold | RMSE | MAE | MAPE(%) |
|------|------|-----|---------|
| 1 | 141.0216 | 99.8860 | 6.34 |
| 2 | 120.8770 | 78.7761 | 3.90 |
| 3 | 113.2008 | 85.3576 | 4.65 |
| 4 | 89.0937 | 73.2613 | 4.46 |
| 5 | 236.5518 | 82.8547 | 5.97 |
| **Mean** | **140.1490** | **84.0271** | **5.07** |

## 4. 对比模型
- CustomOLS 平均 RMSE: 140.1490
- GradientDescentOLS 平均 RMSE: 1867.2699
- sklearn LinearRegression 平均 RMSE: 155.3365

## 5. VIF 诊断

| Feature | VIF | 判断 |
|---------|-----|------|
| area_sqft | 8.20 | ⚡ 中等 |
| bedrooms | 8.96 | ⚡ 中等 |
| age_years | 1.00 | ✅ 正常 |
| location_B | 2.80 | ✅ 正常 |
| location_C | 8.49 | ⚡ 中等 |

## 6. 系数方向 vs DGP

| Feature | DGP 预期 | 模型系数 | 一致? |
|---------|----------|----------|-------|
| area_sqft | + | +534.4396 (+) | — |
| bedrooms | ~0 (共线) | +151.8219 (+) | — |
| age_years | - | -6.1978 (-) | — |
| location_B | -vsA | -98.8429 (-) | — |
| location_C | -vsA | -119.6625 (-) | — |

## 7. 推测总结
- `area_sqft` 方向与 DGP 一致 (正向), 因标准化量纲不可直接比较绝对值
- `age_years` 方向与 DGP 一致 (负向)
- `bedrooms` 因与 area_sqft 高度共线 (VIF≈9), 系数不稳定
- 位置变量使用 drop-first 编码 (A 为基线), B/C 系数表示与 A 的差异
- 异常值和缺失值在 winsorization + CV 内均值填补后影响可控
- GradientDescentOLS 收敛不佳, 需调整学习率

> 因为知道 DGP, 所以可以精确验证模型识别能力。