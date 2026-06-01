# Week 13 Assignment: Regularized Regression and Variable Selection
**（第十三周实操：正则化回归与变量筛选）**

## 背景与目标 (Background)
在之前的实操中，我们已经掌握了基础的回归工作流，并体会到了 OLS（普通最小二乘法）在处理真实数据时的局限性。本周，我们将结合课上的核心概念（Ridge、Lasso、Elastic Net 以及前向/后向变量筛选），进一步探索：

1. 当特征高度相关时，OLS 的系数为何会不稳定；
2. 如何通过引入 penalty（正则化项）来约束模型复杂度，牺牲一部分训练集的贴合度以换取更稳定的表现；
3. Ridge、Lasso、Elastic Net 在面对共线性特征时，系数收缩路径有何差异；
4. 如何通过交叉验证（如 GridSearchCV）科学地选择最优超参数 `alpha`；
5. 除了正则化，传统的变量筛选方法（Forward Selection / Backward Elimination）是如何工作的。

本周的关键词是：
- **目标函数 = loss + penalty**；
- **系数收缩与变量筛选**；
- **交叉验证与超参数寻优**；
- **算法对比与稳定性验证**。

---

## 目录规范 (Directory Architecture)
请继续维护你的个人算法库，并为 Week 13 新增一个完整工作区：
```text
students/<your_name>/
├── pyproject.toml
└── src/
    ├── utils/
    │   ├── models.py          <-- 可新增或维护前向/后向选择逻辑
    │   ├── metrics.py         
    │   ├── transformers.py    
    │   └── diagnostics.py     
    └── week13/
        ├── data/
        │   ├── synthetic_correlated.csv   <-- 你自己生成并保存的模拟共线性数据
        │   └── kaggle_*.csv               <-- 你下载的高维/存在共线性的 Kaggle 数据
        ├── results/
        │   ├── synthetic_report.md
        │   ├── kaggle_report.md
        │   └── summary_comparison.md
        └── main.py                        <-- 唯一执行入口
```

---

## 工程总要求 (Engineering Rules)

### Rule 1: 必须复用你自己的 `utils/` 组件
虽然这周我们将大量使用 `sklearn` 的正则化模型，但数据预处理、评估指标计算等环节必须继续使用你自己的 `src/utils/` 组件进行封装。

### Rule 2: `sklearn` 的边界
允许使用的 `sklearn` 功能：
- `Ridge`, `Lasso`, `ElasticNet`
- `GridSearchCV`, `KFold`, `train_test_split`
- `LinearRegression` (作为基准模型或用于变量筛选中的子模型评估)

你需要自己实现（或在你的 `utils` 中补充）：
- 基于交叉验证的前向选择 (Forward Selection) 或后向剔除 (Backward Elimination) 的核心逻辑。

### Rule 3: 单一入口
必须仅通过：
```text
uv run src/week13/main.py
```
完成本周全部流程。

---

## Task A：自己生成数据，观察系数路径与正则化效果
在本任务中，你需要复现课上的核心发现：在存在高度相关特征的场景下，各种方法的行为差异。

### A1. 生成带有明确共线性的模拟回归数据
请参考课上的 `make_correlated_regression_data` 逻辑，自己设计一份数据集。
要求：
1. 样本量不少于 `300`；
2. 至少 `8` 个特征，其中必须显式构造至少一组（包含 3 个及以上）高度相关的特征族，以及若干纯噪声特征；
3. 目标变量 `y` 的真实生成公式（DGP）应当只依赖于部分特征。

### A2. 保存数据并记录 DGP
把生成后的数据保存到 `src/week13/data/synthetic_correlated.csv`。
在 `synthetic_report.md` 中写明：
- 真实的 DGP 是怎样的；
- 哪几个特征是高度相关的；
- 哪几个是纯噪声。

### A3. 核心模型对比与调参（本次作业核心）
这一步是你展现工程素养和理论理解的关键。请严格按照以下步骤完成并在 `synthetic_report.md` 中详尽记录：

1. **正则化前后的稳定性对比**：我们已经知道 OLS 在遇到共线性时系数会很不稳定。请使用 `train_test_split` 进行至少 50 次不同的随机切分，分别用**普通 OLS 和 Ridge（选定一个适中的 `alpha`）**进行拟合。收集你设定的“高度相关特征”在不同切分下的系数，并计算标准差或绘制箱线图（Boxplot）进行**并排对比**。直观地向业务方展示：“引入正则化后，哪怕换一批样本，我们的结论也变得稳定得多”。
2. **提取最优模型参数并建立 Pipeline**：在调参前，务必使用你在 `utils/` 中编写的预处理组件（如自定义的 Scaler）与 `sklearn` 模型组装成 `Pipeline`。并在报告中简要回答：**为什么在使用 Ridge 或 Lasso 之前，必须对特征进行标准化？**
3. **GridSearchCV 寻优与可视化**：
   - 为 Ridge 和 Lasso 设置合理的 `alpha` 搜索空间（例如对数空间 `np.logspace(-4, 3, 50)`）。
   - 为 Elastic Net 设置 `alpha` 和 `l1_ratio` 的二维搜索空间。
   - 运行 5 折交叉验证，提取 CV 结果，并尝试绘制出“CV 验证误差随 `alpha` 变化的折线图”（类似课件中的 U 型曲线），标出误差最低点对应的最优超参数。
4. **模型性格大比拼**：
   - 提取出最佳的 Ridge、Lasso、Elastic Net 模型，并在**测试集**上评估它们的性能 (RMSE, MAE 等)。
   - 提取并打印各最优模型的系数，重点对比它们是如何处理你设定的那组“高度相关特征”的。
   - 回答：Ridge 是不是将它们均匀缩小了？Lasso 是不是只保留了其中一个而把其他的压缩为 0？Elastic Net 是像 Lasso 一样狠，还是像 Ridge 一样保留了整体阵型？这与你在课堂上学到的“模型性格”是否完全一致？

### A4. 自定义变量筛选机制的对比
1. 在你的代码中实现一种传统的变量选择机制（前向选择 Top-K 或后向剔除），跑在这份数据上；
2. 对比 Lasso 自动选择出的非零变量名单与传统筛选机制选出的名单是否一致。

---

## Task B：从 Kaggle 下载真实数据，完成“高维或共线性真实场景下的推测”（可选 / Optional）
*本部分为选做题，鼓励学有余力、希望把正则化应用到复杂真实场景的同学挑战。如果完成，将在本次作业评估中获得加分。*

在真实世界中，变量之间的关联通常是隐含的。

### B1. 自选 Kaggle 回归数据集
寻找一份特征较多（例如特征数 >= 15）或存在潜在共线性的真实数据集。
在报告中说明：
- 数据来源与业务背景；
- 为什么这份数据适合用来练习正则化和变量筛选。

### B2. 用你的工具箱与 sklearn 结合跑完整流程
1. 完成必要的数据清洗与你自定义的预处理（`transformers`）；
2. 分别使用 OLS, Ridge, Lasso, Elastic Net 对数据进行建模；
3. 通过交叉验证严格评估模型在测试集上的表现 (RMSE, MAE 等)；
4. 输出每种模型所侧重的“特征重要度（系数大小）”或保留下来的变量名单。

### B3. 真实数据的“推测”解释
在 `kaggle_report.md` 中回答：
- 与 OLS 相比，正则化方法是否显著提升了验证集的表现？如果没有，可能是为什么？
- Lasso 最终帮你剔除了哪些特征？从业务逻辑上看，这些特征被剔除是否合理？
- 根据模型结果，如果业务方要求提供一份“最关键的 5 个影响因素”名单，你会以什么方法的结果为准？为什么？

---

## Task C：理论与实践总结
在 `summary_comparison.md` 中，基于你在 Task A (及可选的 Task B) 中的实验，讨论以下问题：
1. Lasso 的系数收缩行为在面对高度相关变量组时，有什么潜在的业务风险？Elastic Net 是如何缓解这个问题的？
2. `GridSearchCV` 寻找最低验证误差的超参数，与我们主观追求“越稀疏越好”或“越稳越好”之间，有何异同？
3. 对比传统的前向选择/后向剔除与 Lasso，在计算效率和最终结果上你有何体会？

---

## 交付物要求 (Deliverables)
你最终至少应提交并产出以下内容：
1. `src/week13/main.py`
2. `src/week13/data/synthetic_correlated.csv`
3. `src/week13/results/synthetic_report.md`
4. `src/week13/results/summary_comparison.md`
5. 你选择的 Kaggle 原始或清洗后数据文件 (可选)
6. `src/week13/results/kaggle_report.md` (可选)
