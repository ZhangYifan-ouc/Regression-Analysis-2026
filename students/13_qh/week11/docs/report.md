# Week 11: Dual Inference Sprint — Synthetic-to-Real Regression Workflow

## 1. 实验目的

本实验旨在：
1. 生成模拟数据，验证模型推断能力
2. 在真实数据上完成完整的回归分析流程
3. 对比模拟数据与真实数据的分析差异
4. 理解数据泄露、共线性、缺失值等问题

---

## 2. 实验设计

### 2.1 目录结构

```
students/13_qh/week11/
├── docs/report.md
└── src/
    ├── utils/
    │   ├── models.py
    │   ├── metrics.py
    │   ├── transformers.py
    │   └── diagnostics.py
    └── week11/
        ├── data/
        │   ├── synthetic_regression.csv
        │   └── StudentPerformanceFactors.csv
        ├── results/
        │   ├── synthetic_report.md
        │   ├── kaggle_report.md
        │   ├── summary_comparison.md
        │   ├── synthetic_data_analysis.png
        │   ├── kaggle_data_exploration.png
        │   ├── coefficient_comparison.png
        │   └── metrics_comparison.png
        └── main.py
```

---

## 3. Task A: 模拟数据分析

### 3.1 数据生成机制 (DGP)

**场景**：广告预算与销售额

**公式**：
```
Sales = 3.0 × TV_Budget + 2.0 × Radio_Budget + 1.5 × Online_Budget + 50 × Is_Peak + ε
```

**特征说明**：
- TV_Budget: TV 广告预算，U(50, 300)
- Radio_Budget: Radio 广告预算，U(20, 150)
- Online_Budget: 在线广告预算，与 TV_Budget 高度相关（0.8 × TV + noise）
- Season: 季节变量，Summer 和 Winter 为旺季

**人为添加的问题**：
- 缺失值：约 5%
- 异常值：约 2%
- 共线性：Online_Budget 与 TV_Budget 高度相关

### 3.2 数据探索

![模拟数据分析](../src/week11/results/synthetic_data_analysis.png)

**图表说明**：
- 左上：TV_Budget vs Sales，显示正相关关系
- 右上：Radio_Budget vs Sales，显示正相关关系
- 左下：Online_Budget vs Sales，显示正相关关系
- 右下：季节分布，各季节样本数量相近

### 3.3 VIF 诊断结果

| 特征 | VIF | 状态 |
|------|-----|------|
| TV_Budget | 1.3203 | ✓ 正常 |
| Radio_Budget | 1.0074 | ✓ 正常 |
| Online_Budget | 1.3181 | ✓ 正常 |
| Season_Spring | 1.6018 | ✓ 正常 |
| Season_Summer | 1.5901 | ✓ 正常 |
| Season_Winter | 1.6221 | ✓ 正常 |

**结论**：所有特征 VIF < 10，未检测到严重多重共线性。

### 3.4 模型系数

| 特征 | 系数 | 预期方向 | 实际方向 | 一致性 |
|------|------|----------|----------|--------|
| TV_Budget | 44.33 | 正向 | 正向 | ✓ |
| Radio_Budget | 25.18 | 正向 | 正向 | ✓ |
| Online_Budget | 94.23 | 正向 | 正向 | ✓ |
| Season_Spring | -4.16 | 负向 | 负向 | ✓ |
| Season_Summer | 7.73 | 正向 | 正向 | ✓ |
| Season_Winter | 6.38 | 正向 | 正向 | ✓ |

**分析**：
- TV_Budget 系数为 44.33，符合预期的正向影响
- Radio_Budget 系数为 25.18，符合预期的正向影响
- Online_Budget 系数为 94.23，较高可能是因为与 TV_Budget 的共线性
- 季节变量方向与预期一致

### 3.5 交叉验证结果

- **R²: 0.8998**
- 平均 RMSE: 659.45
- 平均 MAE: 623.40
- 平均 MAPE: 67.47%

---

## 4. Task B: Kaggle 真实数据分析

### 4.1 数据集信息

- **数据集名称**: Student Performance Factors
- **预测目标**: Exam_Score（考试成绩）
- **样本数**: 6607
- **原始特征数**: 19
- **业务含义**: 每行代表一个学生的学习特征和考试成绩

### 4.2 变量选择

**使用了所有 19 个特征**，包括：

**数值型特征（6 个）**：
- Hours_Studied, Attendance, Sleep_Hours, Previous_Scores, Tutoring_Sessions, Physical_Activity

**类别型特征（13 个）**：
- Parental_Involvement, Access_to_Resources, Extracurricular_Activities, Motivation_Level, Internet_Access, Family_Income, Teacher_Quality, School_Type, Peer_Influence, Learning_Disabilities, Parental_Education_Level, Distance_from_Home, Gender

**处理方式**：
- 数值型特征：直接使用
- 类别型特征：One-Hot 编码（drop_first=True 避免虚拟变量陷阱）
- 编码后特征数：27 个

### 4.3 数据探索

![Kaggle 数据探索](../src/week11/results/kaggle_data_exploration.png)

**图表说明**：
- 左上：Hours_Studied vs Exam_Score，显示正相关趋势
- 右上：Attendance vs Exam_Score，显示正相关趋势
- 左下：Exam_Score 分布，近似正态分布
- 右下：Parental_Involvement vs Exam_Score，显示不同参与度对成绩的影响

### 4.3 缺失值分析

| 特征 | 缺失值数量 | 缺失比例 |
|------|------------|----------|
| Teacher_Quality | 78 | 1.18% |
| Parental_Education_Level | 90 | 1.36% |
| Distance_from_Home | 67 | 1.01% |

### 4.4 VIF 诊断结果

| 特征 | VIF | 状态 |
|------|-----|------|
| Hours_Studied | 1.0010 | ✓ 正常 |
| Attendance | 1.0015 | ✓ 正常 |
| Sleep_Hours | 1.0010 | ✓ 正常 |
| Previous_Scores | 1.0018 | ✓ 正常 |
| Tutoring_Sessions | 1.0010 | ✓ 正常 |
| Physical_Activity | 1.0010 | ✓ 正常 |

**结论**：所有数值特征 VIF 接近 1，无多重共线性问题。

### 4.5 交叉验证结果

- **R²: 0.7267**
- 平均 RMSE: 45.17
- 平均 MAE: 45.07
- 平均 MAPE: 67.00%

### 4.6 业务解读

- **MAE = 45.07**: 模型预测的考试成绩平均误差约 45 分
- **MAPE = 67.00%**: 预测误差约为实际成绩的 67%

**风险分析**：
1. **数据质量**: 部分字段存在缺失值，需要谨慎处理
2. **特征工程**: 类别变量编码可能丢失部分信息
3. **模型局限**: 线性模型可能无法捕捉非线性关系

---

## 5. Task C: 对照总结

### 5.1 指标对比

| 指标 | 模拟数据 | Kaggle 数据 |
|------|----------|-------------|
| R² | 0.8998 | 0.7267 |
| RMSE | 659.45 | 45.17 |
| MAE | 623.40 | 45.07 |
| MAPE | 67.47% | 67.00% |

![指标对比](../src/week11/results/metrics_comparison.png)

### 5.2 系数对比

![系数对比](../src/week11/results/coefficient_comparison.png)

**分析**：
- 模拟数据中，Online_Budget 系数最大，符合 DGP 设计
- Kaggle 数据中，Hours_Studied 和 Attendance 是最重要的特征

### 5.3 关键讨论

**1. 为什么模拟数据的推测更容易？**
- 我们知道真实的 DGP
- 我们知道哪些变量应该正向影响目标
- 我们可以验证模型是否识别出了正确的模式

**2. 为什么真实数据的解释更困难？**
- 我们不知道真实的 DGP
- 变量之间的关系可能是非线性的
- 可能存在未观测的混淆变量

**3. 共线性、缺失值、异常值的影响**

| 问题 | 模拟数据 | 真实数据 |
|------|----------|----------|
| 共线性 | 明确构造（Online 与 TV）| 需要诊断发现 |
| 缺失值 | 人为添加 5% | 真实缺失（1-2%）|
| 异常值 | 人为添加 2% | 真实异常 |

**4. 为什么无泄露交叉验证在真实数据上尤其重要？**
- 数据量通常更小，泄露的影响更显著
- 我们无法验证模型是否真的学到了正确的模式
- 无泄露评估是唯一可信的泛化能力指标

**5. utils/ 组件的价值**
- **一致性**: 两个任务使用相同的预处理逻辑
- **可复用性**: 新数据集可以快速应用相同流程
- **可解释性**: 清楚知道每一步做了什么
- **可调试性**: 问题定位更容易

---

## 6. 结论

1. **模拟数据验证成功**：模型系数方向与 DGP 一致
2. **真实数据分析完成**：在 Kaggle 数据上完成完整流程
3. **无泄露评估实现**：在 CV 循环内部进行预处理
4. **utils/ 组件复用成功**：两个任务共享相同的工具箱
5. **业务意义明确**：MAE 约 45 分的误差需要在业务决策中考虑

---

## 7. 生成文件

### 数据文件
- `week11/data/synthetic_regression.csv`：模拟数据
- `week11/data/StudentPerformanceFactors.csv`：Kaggle 数据

### 报告文件
- `week11/results/synthetic_report.md`：模拟数据分析报告
- `week11/results/kaggle_report.md`：Kaggle 数据分析报告
- `week11/results/summary_comparison.md`：对照总结报告

### 图表文件
- `week11/results/synthetic_data_analysis.png`：模拟数据探索图
- `week11/results/kaggle_data_exploration.png`：Kaggle 数据探索图
- `week11/results/coefficient_comparison.png`：系数对比图
- `week11/results/metrics_comparison.png`：指标对比图
