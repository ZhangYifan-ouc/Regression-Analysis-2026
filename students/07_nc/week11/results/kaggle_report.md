# Week 11 Task B：Kaggle 真实数据回归报告

## 1. Kaggle 数据说明

- 数据集名称：Gapminder: Countries Over Time
- Kaggle 页面链接：https://www.kaggle.com/datasets/shraddha4ever20/gapminder-countries-over-time
- 下载/整理日期：2026-05-24
- 使用文件：`week11/data/kaggle_gapminder.csv`
- 预测目标：`lifeExp`，即出生时预期寿命，连续变量，适合回归问题。
- 每一行样本代表：某个国家在某个年份的国家层面观测，包括洲别、人口、GDP per capita 和预期寿命。

我选择这份数据，是因为它有真实业务/公共政策含义：预期寿命会受到经济发展水平、人口规模、时间趋势和地区差异共同影响。相比几乎只需直接套模板的数据，这份数据需要处理强偏态数值变量、高基数国家字段、时间趋势和地区类别解释问题。

## 2. 原始数据概览

- 原始行数：1704
- 清洗后行数：1704
- 原始字段：country, continent, year, lifeExp, pop, gdpPercap

### 字段质量检查

| column | dtype | missing | missing_rate | unique |
| --- | --- | --- | --- | --- |
| country | str | 0 | 0.0000 | 142 |
| continent | str | 0 | 0.0000 | 5 |
| year | int64 | 0 | 0.0000 | 12 |
| lifeExp | float64 | 0 | 0.0000 | 1626 |
| pop | int64 | 0 | 0.0000 | 1704 |
| gdpPercap | float64 | 0 | 0.0000 | 1704 |

### 建模后数值变量描述性统计

| feature | count | mean | std | min | 25% | 50% | 75% | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| year_centered | 1704.000 | 27.500 | 17.265 | 0.000 | 13.750 | 27.500 | 41.250 | 55.000 |
| log_gdp_per_capita | 1704.000 | 8.159 | 1.241 | 5.485 | 7.092 | 8.170 | 9.141 | 11.640 |
| log_population | 1704.000 | 15.766 | 1.605 | 11.002 | 14.843 | 15.765 | 16.790 | 21.000 |
| lifeExp | 1704.000 | 59.474 | 12.917 | 23.599 | 48.198 | 60.712 | 70.846 | 82.603 |

## 3. 清洗与特征处理

本任务做了以下处理：

1. 删除完全重复行；
2. 将 `year`、`lifeExp`、`pop`、`gdpPercap` 转为数值；
3. 删除目标或关键数值字段非正的记录；
4. 构造 `year_centered = year - min(year)`，避免年份原值过大；
5. 对 `pop` 和 `gdpPercap` 做对数变换，得到 `log_population` 和 `log_gdp_per_capita`，缓解强偏态和极端值影响；
6. 保留 `continent` 做类别变量；
7. 不把 `country` 作为模型特征，因为 142 个国家哑变量会让作业中的线性解释过度复杂，且容易引入国家固定效应主导问题。

在每一个 CV 训练折中，自定义 `RegressionPreprocessor` 会重新学习：中位数填补、winsorization 分位点、标准化参数和类别编码结构。验证折只 `transform`，没有泄露。

## 4. 5 折交叉验证结果

### 每折结果（自定义 OLS 主流程）

| fold | RMSE | MAE | MAPE | R2 |
| --- | --- | --- | --- | --- |
| 1 | 5.259 | 4.144 | 7.414 | 0.819 |
| 2 | 6.061 | 4.639 | 8.878 | 0.790 |
| 3 | 5.986 | 4.480 | 8.603 | 0.791 |
| 4 | 5.749 | 4.452 | 8.224 | 0.804 |
| 5 | 5.895 | 4.513 | 8.465 | 0.790 |

### 指标均值与标准差

| model | metric | mean | std |
| --- | --- | --- | --- |
| Custom OLS workflow | RMSE | 5.790 | 0.319 |
| Custom OLS workflow | MAE | 4.446 | 0.183 |
| Custom OLS workflow | MAPE | 8.317 | 0.557 |
| Custom OLS workflow | R2 | 0.799 | 0.013 |
| sklearn Ridge baseline | RMSE | 5.791 | 0.321 |
| sklearn Ridge baseline | MAE | 4.445 | 0.186 |
| sklearn Ridge baseline | MAPE | 8.319 | 0.561 |
| sklearn Ridge baseline | R2 | 0.799 | 0.013 |

`sklearn Ridge baseline` 只是对照组，主要分析仍来自自定义预处理 + 自定义 OLS + 自定义 metrics 的主流程。

## 5. 共线性与诊断

### 高相关变量对

| feature_1 | feature_2 | abs_corr |
| --- | --- | --- |
| log_gdp_per_capita | lifeExp | 0.808 |

### VIF 前 12 项

| feature | VIF |
| --- | --- |
| continent__Europe | 2.381 |
| log_gdp_per_capita | 2.086 |
| continent__Americas | 1.671 |
| continent__Asia | 1.605 |
| log_population | 1.208 |
| year_centered | 1.203 |
| continent__Oceania | 1.154 |

### 残差摘要

| residual_mean | residual_std | residual_median | residual_p95_abs |
| --- | --- | --- | --- |
| -0.000 | 5.769 | 0.303 | 12.120 |

`year_centered` 和 `log_gdp_per_capita` 往往会同时与 `lifeExp` 正相关，因为全球经济发展和医疗改善都随时间推进。VIF 用来提醒我们：如果某些变量互相解释能力过强，单个系数的业务解释就要谨慎。

## 6. 系数方向与真实数据推测

| feature | expected | estimated_coef | estimated_direction | consistent |
| --- | --- | --- | --- | --- |
| year_centered | 正向 | 4.065 | 正向 | 是 |
| log_gdp_per_capita | 正向 | 6.395 | 正向 | 是 |
| log_population | 正向 | 0.219 | 正向 | 是 |
| continent__Americas | 正向 | 8.602 | 正向 | 是 |
| continent__Europe | 正向 | 12.052 | 正向 | 是 |
| continent__Oceania | 正向 | 12.242 | 正向 | 是 |

较稳定的变量通常是：

- `year_centered`：时间推进通常对应公共卫生、医疗和教育改善；
- `log_gdp_per_capita`：经济水平更高通常对应更高预期寿命；
- `continent`：地区差异吸收了很多制度、地理和历史因素。

不稳定或需要谨慎解释的变量包括：

- `log_population`：人口规模本身不必然提高或降低寿命，它更像国家规模控制变量；
- `continent` 的哑变量：它不是直接因果变量，而是很多未观测因素的综合代理；
- `country` 被删除，因为它会让模型更多记住国家身份，而不是学习可解释的经济/人口关系。

## 7. 业务解释与上线风险

平均误差可以理解为模型对国家-年份预期寿命的平均预测偏差。即使 RMSE/MAE 看起来可接受，这个模型也不能直接用于政策因果判断，因为：

1. 数据是国家层面聚合数据，不能推断个人层面的寿命；
2. `continent` 代表的是复杂地区差异，不是可干预变量；
3. 时间趋势、GDP、医疗水平、战争、疫情等因素互相交织；
4. 线性模型可能无法捕捉预期寿命的上限效应和非线性变化。

如果上线，我最担心的是把相关性误读为因果性，以及在极端国家/年份上预测偏差过大。

## 8. 图形

- `kaggle_actual_vs_pred.png`
- `kaggle_residuals.png`
