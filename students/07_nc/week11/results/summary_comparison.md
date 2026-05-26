# Week 11 Task C：模拟数据与 Kaggle 真实数据对照总结

## 1. 指标对照

| model | metric | mean | std |
| --- | --- | --- | --- |
| Synthetic / Custom OLS | RMSE | 5639.201 | 498.997 |
| Synthetic / Custom OLS | MAE | 4109.140 | 237.954 |
| Synthetic / Custom OLS | MAPE | 7.253 | 0.520 |
| Synthetic / Custom OLS | R2 | 0.493 | 0.045 |
| Kaggle / Custom OLS | RMSE | 5.790 | 0.319 |
| Kaggle / Custom OLS | MAE | 4.446 | 0.183 |
| Kaggle / Custom OLS | MAPE | 8.317 | 0.557 |
| Kaggle / Custom OLS | R2 | 0.799 | 0.013 |

## 2. 为什么模拟数据中的“推测”更容易？

模拟数据的 DGP 是我自己写出来的，因此我事先知道哪些变量应该正向影响目标、哪些变量应该负向影响目标，也知道 `tv_budget` 和 `radio_budget` 被故意设置成高度相关。这样一来，模型结果可以直接和真实公式对照：如果系数方向不一致，优先检查噪声、异常值、共线性或预处理。

## 3. 为什么真实数据即使分数还可以，解释也更困难？

Kaggle Gapminder 数据中，`lifeExp` 与 GDP、年份、地区之间存在明显相关，但这些相关并不等于因果。比如洲别哑变量可能同时代表医疗体系、气候、历史、制度、战争风险等大量未观测因素。模型分数说明预测还可以，但不能说明“提高某个变量一定导致寿命变化”。真实数据的解释困难主要来自遗漏变量、聚合数据、时间趋势和变量代理含义不清晰。

## 4. 共线性、缺失值、异常值在两类数据上的影响

- 模拟数据：共线性是主动构造的，所以可以清楚地看到 `tv_budget` 与 `radio_budget` 的 VIF 风险；缺失值和异常值也是可控注入的，主要用于检验自己的清洗流程。
- 真实数据：共线性和异常值来自真实世界结构，例如 GDP 和年份可能共同反映长期发展趋势，人口和 GDP 的分布极端偏态。我们不知道完整 DGP，因此只能用 VIF、残差图和业务常识谨慎判断。

## 5. 为什么无泄露交叉验证在真实数据上尤其重要？

真实数据中的分布差异更复杂。如果在 CV 前对全量数据做均值填补、标准化或分位数截尾，就等于让验证折的信息提前进入训练过程，模型评估会过于乐观。Week11 的主流程在每一折中重新 `fit` 预处理器，然后对验证折只 `transform`，因此验证结果更接近未来新数据场景。

## 6. 自己维护的 utils 组件节省了哪些重复劳动？

本周复用了以下组件：

- `utils.models.AnalyticalOLS`：统一训练和预测接口；
- `utils.metrics.calculate_rmse / calculate_mae / calculate_mape`：统一指标计算；
- `utils.transformers.RegressionPreprocessor`：把缺失值填补、winsorization、标准化、one-hot 编码组织到同一个可复用对象；
- `utils.diagnostics.calculate_vif / correlation_pairs / residual_summary`：统一诊断输出。

因为这些工具已经封装好，模拟数据和 Kaggle 数据可以共用同一套 CV 函数，不需要复制粘贴两份清洗和评估代码。

## 7. 答辩准备要点

1. `main()` 的流程：清空结果目录 → 生成模拟数据 → 跑模拟任务 → 读取 Kaggle 数据 → 跑真实数据任务 → 写对照总结。
2. 无泄露关键：每一折训练集 `fit` 预处理器，验证集只 `transform`。
3. 主模型：自定义 `AnalyticalOLS`，不是 sklearn 一把梭。
4. baseline：`sklearn Ridge` 只是对照组，且仍使用自定义预处理后的特征。
5. 核心函数可解释：`RegressionPreprocessor.fit()` 学习填补值、截尾分位点、标准化参数和类别集合；`transform()` 只复用这些参数。
