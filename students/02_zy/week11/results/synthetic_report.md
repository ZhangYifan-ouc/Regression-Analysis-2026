# Week 11 Task A：模拟数据回归推测报告

## 1. 数据生成机制 DGP

本任务模拟一个广告投放与销售额之间的业务场景。每一行代表一个市场投放观测样本，目标变量是 `sales`。

我设定的生成公式为：

```text
sales = 200 + 2.5 * tv_budget + 1.2 * online_budget + 0.15 * store_visits - 3.0 * competitor_price + region_effect + noise
```

其中：

- `tv_budget`：电视广告预算，理论上正向影响销售额；
- `online_budget`：线上广告预算，理论上正向影响销售额；
- `store_visits`：门店访问量，理论上正向影响销售额；
- `competitor_price`：竞争对手价格，理论上负向影响本企业销售额；
- `region`：地区类别变量，不同地区有不同的销售基础水平。

为了主动构造共线性，我先生成 `tv_budget`，再令：

```text
online_budget = 0.85 * tv_budget + 随机扰动
```

因此，`tv_budget` 和 `online_budget` 应该存在明显共线性。

## 2. 人为加入的真实世界问题

我在模拟数据中主动加入了以下问题：

- 缺失值：部分 `tv_budget` 和 `competitor_price` 被设置为 NaN；
- 异常值：部分 `online_budget` 和 `store_visits` 被放大；
- 特征量纲差异：预算变量大约几十到几百，而 `store_visits` 大约上千；
- 共线性：`tv_budget` 和 `online_budget` 高度相关。

## 3. 无泄露 5 折交叉验证结果

| 指标 | 数值 |
|---|---:|
| RMSE | 54.9867 |
| MAE | 42.9692 |
| MAPE (%) | 9.6769 |

### 每一折结果

| 折数 | RMSE | MAE | MAPE (%) |
|---:|---:|---:|---:|
| 1 | 47.3617 | 38.1545 | 8.5588 |
| 2 | 58.3223 | 43.1044 | 9.9289 |
| 3 | 55.8334 | 44.1384 | 9.7542 |
| 4 | 57.2382 | 44.5573 | 10.1430 |
| 5 | 56.1779 | 44.8913 | 9.9994 |

## 4. VIF 共线性诊断

| 特征 | VIF |
|---|---:|
| tv_budget | 1.4753 |
| online_budget | 1.4676 |
| region_West | 1.3800 |
| region_North | 1.3597 |
| region_South | 1.3124 |
| competitor_price | 1.0163 |
| store_visits | 1.0157 |

如果某些变量的 VIF 明显大于 10，说明存在严重多重共线性。在本模拟数据中，`tv_budget` 和 `online_budget` 是我主动构造的高度相关变量，因此它们的 VIF 较高是符合预期的。

## 5. 系数方向推测

| 特征 | 系数 | 方向 |
|---|---:|---|
| tv_budget | 63.6813 | 正向 |
| competitor_price | -37.6914 | 负向 |
| region_North | -23.7899 | 负向 |
| region_South | -23.0245 | 负向 |
| store_visits | 19.9464 | 正向 |
| region_West | -13.3323 | 负向 |
| online_budget | 4.2559 | 正向 |

从系数方向看，`tv_budget`、`online_budget` 和 `store_visits` 应该主要呈正向影响，`competitor_price` 应该主要呈负向影响。如果个别系数方向不稳定，主要原因通常是 `tv_budget` 和 `online_budget` 之间存在较强共线性，导致模型难以稳定地区分二者各自的独立贡献。

## 6. 推测结论

在模拟数据中，因为我知道真实的数据生成机制，所以可以检查模型推断是否与 DGP 一致。整体上，模型能够识别主要方向，但高度相关的广告预算变量会带来系数不稳定问题。这说明即使预测误差较低，也不能忽略共线性对解释性的影响。
