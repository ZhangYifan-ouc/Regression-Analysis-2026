# Week 11 Task A：模拟数据回归分析报告

## 1. 数据生成机制 DGP

本部分构造的是一个广告预算影响销售额的模拟业务场景。目标变量是 `sales`，表示销售额。

我设定的真实生成公式大致为：

```text
sales = 80 + 2.5 * tv_budget + 1.2 * online_budget + 1.8 * radio_budget
        - 120 * price_discount + region_effect + noise
```

其中，`tv_budget`、`online_budget`、`radio_budget` 对销售额是正向影响，`price_discount` 在这里被设定为负向影响。地区变量 `region` 通过不同地区效应影响销售额。

## 2. 主动加入的数据问题

- 缺失值：在 `radio_budget` 和 `region` 中人为加入缺失值；
- 异常值：在 `tv_budget` 中人为放大部分样本；
- 共线性：令 `online_budget = 0.85 * tv_budget + 随机扰动`，因此它和 `tv_budget` 高度相关；
- 类别变量：`region` 是非数值型变量，需要进行 one-hot 编码。

## 3. 5 折交叉验证结果

| dataset   |   fold |    RMSE |     MAE |    MAPE |
|:----------|-------:|--------:|--------:|--------:|
| synthetic |      1 | 21.4431 | 18.0488 | 3.57451 |
| synthetic |      2 | 19.6835 | 15.8027 | 3.27015 |
| synthetic |      3 | 21.0591 | 16.1042 | 3.49683 |
| synthetic |      4 | 20.2009 | 14.8807 | 3.1273  |
| synthetic |      5 | 19.3924 | 14.4765 | 3.09174 |

平均 RMSE：20.3558

平均 MAE：15.8626

平均 MAPE：3.3121%

## 4. 平均系数方向

|                |   mean_coefficient |
|:---------------|-------------------:|
| intercept      |          498.184   |
| tv_budget      |           11.8971  |
| online_budget  |           74.929   |
| radio_budget   |           16.5424  |
| price_discount |          -10.0808  |
| region_north   |          -11.9407  |
| region_south   |           -8.26328 |
| region_west    |           -5.09924 |

从平均系数看，大多数变量方向与我设定的 DGP 基本一致。但是由于 `tv_budget` 和 `online_budget` 被人为设置为高度相关，它们各自的系数可能不够稳定，这正是共线性会带来的问题。

## 5. VIF 共线性诊断

| feature        |     VIF |
|:---------------|--------:|
| tv_budget      | 4.08109 |
| online_budget  | 4.08289 |
| radio_budget   | 1.0035  |
| price_discount | 1.00345 |
| region_north   | 1.25418 |
| region_south   | 1.25404 |
| region_west    | 1.29444 |

一般来说，VIF 大于 10 可以认为存在比较明显的共线性风险。在本模拟数据中，如果 `tv_budget` 和 `online_budget` 的 VIF 较高，说明模型确实识别出了我在 DGP 中主动构造的共线性问题。

## 6. 推测结论

由于模拟数据的生成机制是已知的，所以我们可以直接对比模型估计结果和真实设定。整体上，模型能够恢复主要变量的影响方向。但对于高度相关的变量，单个系数的解释要谨慎，因为模型很难把两个高度相关变量的作用完全分开。
