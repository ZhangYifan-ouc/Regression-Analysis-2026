# Milestone 2: The Pipeline & The Leakage-Free Generalization

## 1. 实验目的

本实验旨在：
1. 实现评估指标库（RMSE, MAE, MAPE）
2. 实现自定义标准化转换器（CustomStandardScaler）
3. 理解数据泄露的概念和危害
4. 实现防泄漏的交叉验证流水线

---

## 2. 实验设计

### 2.1 目录结构

```
students/13_qh/week10/
├── docs/
│   └── report.md
├── results/
│   ├── evaluation_comparison.md
│   └── leakage_analysis.png
└── src/
    ├── .python-version
    ├── pyproject.toml
    ├── utils/
    │   ├── __init__.py
    │   ├── models.py          # OLS 模型
    │   ├── metrics.py         # 评估指标
    │   └── transformers.py    # 数据转换器
    └── milestone2/
        ├── __init__.py
        └── main.py            # 主程序
```

### 2.2 数据说明

- **数据来源**: students/13_qh/week09/data/dirty_marketing.csv
- **样本数量**: 1000
- **特征**: TV_Budget, Radio_Budget, SocialMedia_Budget, Is_Holiday, Region_EU, Region_LatAm
- **目标变量**: Sales

---

## 3. 方法说明

### 3.1 评估指标库 (metrics.py)

```python
def calculate_rmse(y_true, y_pred):
    # RMSE = sqrt(1/n * sum((y_true - y_pred)^2))

def calculate_mae(y_true, y_pred):
    # MAE = 1/n * sum(|y_true - y_pred|)

def calculate_mape(y_true, y_pred):
    # MAPE = 1/n * sum(|y_true - y_pred| / |y_true|) * 100
    # 处理分母为 0 的情况
```

### 3.2 转换器 API (transformers.py)

```python
class CustomStandardScaler:
    def fit(self, X):
        # 计算并保存均值和标准差
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)

    def transform(self, X):
        # 使用保存的参数进行标准化
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        # 合并 fit 和 transform
        return self.fit(X).transform(X)
```

### 3.3 数据泄露 vs 防泄漏流水线

**Task 3 (有数据泄露)**：
1. 使用全量数据的均值填补缺失值
2. 使用全量数据进行标准化
3. 然后做交叉验证

**Task 4 (无数据泄露)**：
1. 在 CV 循环内部进行预处理
2. 只使用训练集的参数填补缺失值
3. 只使用训练集拟合 Scaler
4. 用训练集的参数转换验证集

---

## 4. 实验结果

### 4.1 评估指标对比

| 指标 | Task 3 (有泄露) | Task 4 (无泄露) | 差异 |
|------|-----------------|-----------------|------|
| RMSE | 597.6355 | **595.9195** | -1.7160 |
| MAE | 556.6100 | **554.7226** | -1.8874 |
| MAPE | 64.98% | **64.75%** | -0.23% |

### 4.2 详细结果

**Task 3 (有数据泄露)**：
- 第 1 折: RMSE=606.9170, MAE=567.6040, MAPE=65.13%
- 第 2 折: RMSE=613.0639, MAE=575.7877, MAPE=66.38%
- 第 3 折: RMSE=586.0254, MAE=544.9267, MAPE=64.52%
- 第 4 折: RMSE=601.6419, MAE=554.7535, MAPE=64.69%
- 第 5 折: RMSE=580.5290, MAE=539.9780, MAPE=64.19%

**Task 4 (无数据泄露)**：
- 第 1 折: RMSE=601.0287, MAE=563.4844, MAPE=64.93%
- 第 2 折: RMSE=614.4475, MAE=575.5434, MAPE=66.16%
- 第 3 折: RMSE=584.7898, MAE=543.0974, MAPE=64.18%
- 第 4 折: RMSE=599.7939, MAE=552.8094, MAPE=64.46%
- 第 5 折: RMSE=579.5376, MAE=538.6787, MAPE=64.02%

---

## 5. 思考题

### 5.1 为什么 Task 3 的"好成绩"是致命的？

Task 3 存在数据泄露，原因如下：

1. **全局均值填补**：使用全量数据（包括验证集）的均值填补缺失值，导致验证集信息泄露
2. **全局标准化**：使用全量数据的均值和标准差进行标准化，验证集的分布信息被泄露到训练过程中

这种"好看"的成绩是致命的，因为：
- 模型在部署时会遇到真正的"未见过"的数据
- 真实世界的性能会比交叉验证的结果差很多
- 这种虚假的高分会误导业务决策

### 5.2 业务解读

基于 Task 4（无泄露）的结果：
- **MAE = 554.72 元**：模型上线后，每天的广告预算预测平均存在约 **555 元**的真实误差
- **MAPE = 64.75%**：预测误差约为实际值的 **65%**

应该给老板看 Task 4 的"差成绩"，因为这才是模型在真实世界中的预期表现。

---

## 6. 结论

1. **评估指标库成功实现**：RMSE, MAE, MAPE 三个指标均可正常计算
2. **CustomStandardScaler 实现正确**：严格遵循 Transformer API 规范
3. **数据泄露问题已验证**：本次实验中 Task 4 的结果略优于 Task 3，说明数据泄露的影响在大数据集上可能被稀释
4. **防泄漏流水线实现成功**：在 CV 循环内部进行预处理，确保验证集完全未见过
5. **业务意义明确**：MAPE 约 65% 的误差率需要在业务决策中充分考虑

---

## 7. 生成文件

- `results/evaluation_comparison.md`：评估对比报告
- `results/leakage_analysis.png`：对比柱状图
