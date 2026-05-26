# Week 10 作业详细解答

## Task 1：评估指标库 `utils/metrics.py`

我手写实现了三个回归预测指标：

1. `calculate_rmse(y_true, y_pred)`：计算均方误差后开平方，对大误差更加敏感；
2. `calculate_mae(y_true, y_pred)`：计算绝对误差平均值，单位和目标变量一致，便于向业务解释；
3. `calculate_mape(y_true, y_pred)`：计算平均绝对百分比误差。代码使用 `epsilon=1e-8` 过滤真实值为 0 或极小的样本，避免除零导致无穷大或误导性百分比。

## Task 2：转换器 API `utils/transformers.py`

`CustomStandardScaler` 按 Transformer 规范实现：

- `fit(X)`：只计算并保存 `self.mean_` 和 `self.std_`；
- `transform(X)`：只复用已经保存的均值和标准差，不重新学习；
- `fit_transform(X)`：组合调用 `fit()` 和 `transform()`。

这个接口的关键是把“学习参数”和“应用参数”分开。Good CV 中验证集只能使用训练集学到的参数，所以验证集不会参与 scaler 的 `fit()`。

## Task 3：Bad Cross Validation（故意制造数据泄露）

`bad_cross_validation()` 的流程是：

1. 读取老师放在 `data/dirty_q4_marketing.csv` 的脏数据；
2. 在全量 `X` 上计算缺失值填补参数、类别编码结构和标准化参数；
3. 对全量数据先完成 `fit_transform()`；
4. 再把处理好的数据送入 5 折交叉验证。

这一步故意制造数据泄露，因为验证折的信息提前参与了全局预处理参数的学习。

本次 Bad CV 平均指标：

- RMSE：1,773.8796
- MAE：1,372.0703
- MAPE：1.9803%

## Task 4：Good Cross Validation（无泄露流水线）

`good_cross_validation()` 的每一折都重新执行一条独立流水线：

1. 先切分 `X_train` 和 `X_val`；
2. 只在 `X_train` 上学习缺失值填补均值和类别编码结构；
3. 只在 `X_train` 上调用 `CustomStandardScaler.fit_transform()`；
4. 对 `X_val` 只调用已经拟合好的 `transform()`；
5. 用 `GradientDescentOLS` 在训练折训练，并在验证折上计算 RMSE、MAE、MAPE。

本次 Good CV 平均指标：

- RMSE：1,738.5033
- MAE：1,353.9687
- MAPE：1.9609%

## Task 5：自动化 I/O 与制品管理

程序唯一入口为：

```bash
uv run src/milestone2/main.py
```

程序启动时会自动清空并重建 `week10/results/` 文件夹，输出：

- `week10/results/evaluation_comparison.md`：Bad CV 和 Good CV 的指标对比；
- `week10/results/week10.md`：本文件，详细作业解答；
- `week10/results/leakage_analysis.png`：有无泄露的误差柱状图。

## Task 6：课堂答辩提纲

### CTO 视角

展示 `CustomStandardScaler` 的 `fit/transform/fit_transform` 结构。重点说明 Good CV 每一折都会重新创建 `fold_plan` 和 `fold_scaler`，并且只用训练折调用 `fit()`；验证折只调用 `transform()`，没有任何二次拟合。

### CMO 视角

更应该汇报 Good CV。按本次运行结果，模型上线后目标列 `Sales` 的平均绝对误差约为 **1,353.9687**，平均相对误差约为 **1.9609%**。这比 Bad CV 更保守，但更接近真实上线环境，因此更适合做预算预测和投放决策。
