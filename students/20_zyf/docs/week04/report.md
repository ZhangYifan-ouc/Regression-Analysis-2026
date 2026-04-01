# Week 04 实验报告：求解器双城记

## 1. 实验目的
- 理解多元线性回归解析解与梯度下降两种主流求解方式的原理与工程实现。
- 对比手写求解器与工业界主流API在低维/高维场景下的性能与精度。

## 2. 实验设置
- 低维场景：$N=10,000$, $P=10$
- 高维场景：$N=10,000$, $P=2,000$
- 比较5种求解器：
	1. AnalyticalSolver（手写解析解）
	2. GradientDescentSolver（手写梯度下降）
	3. statsmodels.OLS
	4. sklearn.LinearRegression
	5. sklearn.SGDRegressor

## 3. 实验结果

| Solver                      | Low Dim Time (s) | Low Dim MSE | High Dim Time (s) | High Dim MSE |
|-----------------------------|------------------|-------------|-------------------|--------------|
| AnalyticalSolver            | 0.134            | 25.57       | 0.733             | 3909.58      |
| GradientDescentSolver       | 0.088            | 16.65       | 29.20             | 2686.04      |
| statsmodels.OLS             | 0.026            | 25.57       | 6.51              | 3909.58      |
| sklearn.LinearRegression    | 0.009            | 25.57       | 2.28              | 3909.58      |
| sklearn.SGDRegressor        | 0.013            | 25.57       | 0.61              | 3909.51      |

## 4. 结果分析
- **低维场景**：所有求解器都能在极短时间内完成，精度接近，梯度下降略优（因未收敛到解析解）。
- **高维场景**：
	- 解析解（AnalyticalSolver、statsmodels、sklearn.LinearRegression）耗时明显增加，尤其是 statsmodels 最慢。
	- 手写梯度下降（GradientDescentSolver）耗时极长，但MSE更低，说明未完全收敛或超参数需调优。
	- sklearn.SGDRegressor 速度最快，且精度与解析解接近。

## 5. 思考题
### 1. 在高维场景下，哪个API崩溃或极其缓慢？
- statsmodels.OLS 在高维下最慢（6.51s），手写梯度下降更慢（29.20s），但未崩溃。
- 若维度更高，statsmodels/解析解类API可能因内存或矩阵不可逆而崩溃。

### 2. 为什么SGDRegressor能在极短时间内完成任务？
- SGDRegressor 采用**随机梯度下降**，每次只用部分样本更新参数，避免了大矩阵运算，内存和算力消耗极低，适合高维大数据。
- 其实现高度优化，默认早停机制，能快速获得较优解。

## 6. 总结
- 解析解适合低维小数据，梯度下降适合高维大数据。
- 工业界API（如sklearn.SGDRegressor）在高维下兼顾速度与精度，是实际工程首选。
