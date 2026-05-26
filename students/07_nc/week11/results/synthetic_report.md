# Week 11 Task A：模拟数据回归报告

## 1. 数据生成机制（DGP）

本任务构造了一个 Q4 营销投放与销售额场景。每一行代表一个产品/区域/渠道组合在 Q4 的投放记录，目标变量为 `q4_sales`。

显式设定的目标生成公式为：

```text
q4_sales = 18000
         + 130 * tv_budget
         + 55 * radio_budget
         + 0.20 * search_clicks
         - 21000 * discount_rate
         + 90 * competitor_price_index
         + 160 * brand_index
         + channel_effect
         + season_effect
         + random_noise
```

其中：

- `tv_budget`、`radio_budget`、`search_clicks`、`competitor_price_index`、`brand_index` 应正向影响销售额；
- `discount_rate` 的系数被设为负数，表示过度折扣可能压低收入质量；
- `channel=search/social/partner` 相对于 `offline` 有正向渠道效果；
- `season=holiday/back_to_school` 相对于 `normal` 有正向季节效果；
- 我故意构造了 `radio_budget = 0.82 * tv_budget + noise`，因此 `tv_budget` 与 `radio_budget` 应该高度相关。

主动加入的真实世界问题包括：缺失值、异常值、明显量纲差异、共线性。生成数据已保存到 `week11/data/synthetic_regression.csv`。

## 2. 数据概览

- 样本量：600
- 特征数：8
- 目标变量：`q4_sales`

### 字段质量检查

| column | dtype | missing | missing_rate | unique |
| --- | --- | --- | --- | --- |
| tv_budget | float64 | 27 | 0.0450 | 572 |
| radio_budget | float64 | 0 | 0.0000 | 599 |
| search_clicks | float64 | 21 | 0.0350 | 579 |
| discount_rate | float64 | 0 | 0.0000 | 600 |
| competitor_price_index | float64 | 0 | 0.0000 | 600 |
| brand_index | float64 | 0 | 0.0000 | 600 |
| channel | str | 18 | 0.0300 | 4 |
| season | str | 0 | 0.0000 | 3 |
| q4_sales | float64 | 0 | 0.0000 | 600 |

### 数值变量描述性统计

| feature | count | mean | std | min | 25% | 50% | 75% | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tv_budget | 573.00 | 86.16 | 24.99 | 15.00 | 68.72 | 86.91 | 102.84 | 160.00 |
| radio_budget | 600.00 | 70.28 | 21.49 | 5.00 | 55.70 | 70.50 | 83.97 | 135.92 |
| search_clicks | 579.00 | 28735.91 | 22690.36 | 6326.29 | 17358.07 | 23982.89 | 33047.16 | 290554.43 |
| discount_rate | 600.00 | 0.09 | 0.05 | 0.01 | 0.05 | 0.08 | 0.12 | 0.28 |
| competitor_price_index | 600.00 | 100.40 | 8.76 | 72.88 | 94.66 | 100.90 | 106.26 | 128.87 |
| brand_index | 600.00 | 55.58 | 11.66 | 21.26 | 47.84 | 55.44 | 63.53 | 92.25 |
| q4_sales | 600.00 | 57182.00 | 7981.15 | 37067.71 | 51524.91 | 56977.77 | 61884.15 | 90972.67 |

## 3. 无泄露建模流程

5 折交叉验证中，每一折都按以下顺序处理：

1. 只在训练折上 `fit` 自定义 `RegressionPreprocessor`；
2. 训练折学习中位数填补、winsorization 分位点、标准化均值/标准差、类别编码结构；
3. 验证折只调用 `transform()`，不重新学习任何参数；
4. 主模型使用 `utils.models.AnalyticalOLS`；
5. 指标使用 `utils.metrics.calculate_rmse / calculate_mae / calculate_mape`；
6. VIF 使用 `utils.diagnostics.calculate_vif`。

## 4. 5 折交叉验证结果

### 每折结果（自定义 OLS 主流程）

| fold | RMSE | MAE | MAPE | R2 |
| --- | --- | --- | --- | --- |
| 1 | 5476.945 | 4103.914 | 7.316 | 0.481 |
| 2 | 6390.845 | 4375.955 | 7.807 | 0.474 |
| 3 | 5845.152 | 4029.012 | 6.793 | 0.462 |
| 4 | 5089.874 | 3761.129 | 6.653 | 0.572 |
| 5 | 5393.190 | 4275.689 | 7.698 | 0.476 |

### 指标均值与标准差

| model | metric | mean | std |
| --- | --- | --- | --- |
| Custom OLS workflow | RMSE | 5639.201 | 498.997 |
| Custom OLS workflow | MAE | 4109.140 | 237.954 |
| Custom OLS workflow | MAPE | 7.253 | 0.520 |
| Custom OLS workflow | R2 | 0.493 | 0.045 |
| sklearn Ridge baseline | RMSE | 5638.054 | 501.917 |
| sklearn Ridge baseline | MAE | 4108.937 | 241.560 |
| sklearn Ridge baseline | MAPE | 7.253 | 0.527 |
| sklearn Ridge baseline | R2 | 0.493 | 0.045 |

说明：`sklearn Ridge baseline` 仅作对照，预处理仍使用本作业自己的 transformer；主流程结论以 `Custom OLS workflow` 为准。

## 5. 共线性诊断

### 高相关变量对

| feature_1 | feature_2 | abs_corr |
| --- | --- | --- |
| tv_budget | radio_budget | 0.963 |

### VIF 前 12 项

| feature | VIF |
| --- | --- |
| radio_budget | 8.364 |
| tv_budget | 8.344 |
| season__holiday | 1.899 |
| season__normal | 1.899 |
| channel__search | 1.569 |
| channel__social | 1.498 |
| channel__partner | 1.334 |
| search_clicks | 1.016 |
| brand_index | 1.015 |
| competitor_price_index | 1.011 |
| discount_rate | 1.009 |

`tv_budget` 与 `radio_budget` 的 VIF 明显较高，符合我在 DGP 中主动构造共线性的预期。这说明系数解释时不能只看单个变量的正负和大小，而要把这一组高度相关投放变量作为整体解释。

## 6. 系数方向与 DGP 是否一致

| feature | expected | estimated_coef | estimated_direction | consistent |
| --- | --- | --- | --- | --- |
| tv_budget | 正向 | 2564.608 | 正向 | 是 |
| radio_budget | 正向 | 2032.002 | 正向 | 是 |
| search_clicks | 正向 | 1826.376 | 正向 | 是 |
| discount_rate | 负向 | -1194.044 | 负向 | 是 |
| competitor_price_index | 正向 | 492.637 | 正向 | 是 |
| brand_index | 正向 | 1710.327 | 正向 | 是 |
| channel__search | 正向 | 3178.552 | 正向 | 是 |
| channel__social | 正向 | 2656.266 | 正向 | 是 |
| season__holiday | 正向 | 1000.501 | 正向 | 是 |

整体上，主要变量方向与 DGP 基本一致。`tv_budget` 与 `radio_budget` 可能出现单个系数不稳定，是因为二者被故意设为高度相关；在线性回归里，模型很难稳定地区分“电视投放”和“广播投放”的独立边际贡献。

## 7. 推测结论

- 模拟数据中，因为真实 DGP 已知，所以判断变量方向是否正确相对容易。
- 影响最稳定的是 `search_clicks`、`brand_index`、季节变量和渠道变量，它们与目标的关系不完全依赖共线变量组。
- `tv_budget` 与 `radio_budget` 本来就难以稳定识别，因为二者高度相关，VIF 也验证了这一点。
- RMSE 比 MAE 更容易受到我主动加入的销售额异常值影响，因此报告中同时保留 RMSE、MAE、MAPE 三个指标。

## 8. 图形

- `synthetic_actual_vs_pred.png`
- `synthetic_residuals.png`
