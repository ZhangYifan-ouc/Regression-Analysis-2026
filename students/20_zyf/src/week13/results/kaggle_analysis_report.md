# 第13周作业：正则化回归与变量筛选 - Kaggle数据报告

## 一、数据集信息

### 数据来源
- **数据集**: AI Impact on Jobs and Layoff Risk Dataset
- **来源**: https://www.kaggle.com/datasets/shivasingh4945/ai-impact-on-jobs-and-layoff-risk-dataset
- **样本量**: 20,000行
- **特征数**: 16列

### 数据集背景

这是一个关于AI对就业和裁员风险影响的真实数据集。在高维数据和隐含共线性的真实场景中，正则化方法的应用尤为重要。

### 特征列表

| 特征 | 类型 | 描述 |
|-----|------|-----|
| Age | 整数 | 员工年龄(21-60岁) |
| Education_Level | 分类 | 教育程度 |
| Years_of_Experience | 整数 | 工作年限 |
| Industry | 分类 | 行业领域 |
| Job_Role | 分类 | 工作职位 |
| Company_Size | 分类 | 公司规模 |
| Job_Level | 分类 | 工作级别 |
| Routine_Task_Percentage | 整数 | 重复任务比例(%) |
| Creativity_Requirement | 整数 | 创意需求(0-100) |
| Human_Interaction_Level | 整数 | 人际互动程度(0-100) |
| AI_Adoption_Level | 分类 | AI采纳水平 |
| Number_of_AI_Tools_Used | 整数 | 使用的AI工具数量 |
| AI_Usage_Hours_Per_Week | 整数 | 每周AI使用时数 |
| Tasks_Automated_Percentage | 整数 | 自动化任务比例(%) |
| AI_Training_Hours | 整数 | AI培训时数 |
| **Layoff_Risk** | **分类** | **目标变量(Low/Medium/High)** |

---

## 二、数据预处理

### 2.1 编码方式

所有分类变量使用LabelEncoder进行编码：
- 目标变量 Layoff_Risk: Low=0, Medium=1, High=2 (或其他编码)
- 特征变量分类编码: 按字母顺序编码

### 2.2 特征标准化

使用CustomStandardScaler进行z-score标准化：
$$z = \frac{x - \mu}{\sigma}$$

这确保了正则化项对所有特征的惩罚力度一致。

---

## 三、模型性能对比

### 3.1 测试集结果


| 模型 | RMSE | MAE |
|-----|------|-----|
| OLS          |   0.7287 |   0.6572 |
| Ridge        |   0.7287 |   0.6573 |
| Lasso        |   0.7294 |   0.6578 |


### 3.2 性能分析

- **OLS基准**: 作为线性模型的无正则化基准
- **Ridge**: 通过L₂正则化防止过拟合
- **Lasso**: 通过L₁正则化进行特征选择

在真实数据中，正则化方法通常能在以下方面优于OLS：
1. 泛化性能（测试集误差）
2. 模型稳定性（对新数据的适应能力）
3. 可解释性（Lasso产生的稀疏解）

---

## 四、特征选择分析

### 4.1 Lasso选择的特征

Lasso（通过L₁惩罚）自动进行特征选择，将不重要特征的系数压为零。

**被选中的特征**（系数非零）:

| 排名 | 特征 | 系数 |
|-----|------|------|
| 1 | Tasks_Automated_Percentage | -0.221923 |
| 2 | Job_Level | 0.099857 |
| 3 | Creativity_Requirement | 0.063070 |
| 4 | Industry | -0.047815 |
| 5 | AI_Usage_Hours_Per_Week | -0.045045 |
| 6 | Human_Interaction_Level | 0.038867 |
| 7 | Routine_Task_Percentage | -0.030305 |
| 8 | Years_of_Experience | 0.029182 |
| 9 | Education_Level | 0.020639 |
| 10 | Number_of_AI_Tools_Used | -0.016263 |


### 4.2 Ridge选择的特征重要度

Ridge保留所有特征但对系数进行缩小。根据系数绝对值排序：

| 排名 | 特征 | 系数 |
|-----|------|------|
| 1 | Tasks_Automated_Percentage | -0.226712 |
| 2 | Job_Level | 0.109860 |
| 3 | Creativity_Requirement | 0.068246 |
| 4 | Industry | -0.055624 |
| 5 | AI_Usage_Hours_Per_Week | -0.048767 |
| 6 | Human_Interaction_Level | 0.048743 |
| 7 | Years_of_Experience | 0.039846 |
| 8 | Routine_Task_Percentage | -0.032721 |
| 9 | Education_Level | 0.030648 |
| 10 | Number_of_AI_Tools_Used | -0.021718 |


---

## 五、关键发现与启示

### 5.1 正则化在真实数据中的价值

1. **维度诅咒**: 16个特征可能存在高度相关性，OLS容易过拟合
2. **正则化效果**: Ridge和Lasso通过不同机制处理共线性和过拟合
3. **Lasso的可解释性**: 通过自动特征选择，生成稀疏模型，便于解释

### 5.2 业务应用建议

#### 如果要向业务方解释"最关键的5个影响因素"：

**选择方案对比**:

| 方案 | 方法 | 优点 | 缺点 |
|-----|------|------|------|
| A | Lasso系数 | 自动特征选择，稀疏化 | 结果可能不稳定 |
| B | Ridge系数 | 稳定可靠 | 包含所有特征，难以解释 |
| C | 业务+模型混合 | 结合领域知识 | 需要专家投入 |

**推荐**: 结合**Lasso选出的非零特征**与**Ridge对这些特征的系数排序**，得到既稀疏又稳定的特征重要度排名。

---

## 六、总结

本分析展示了正则化方法在高维真实数据中的应用价值：
1. OLS在高维数据上容易过拟合和系数不稳定
2. Ridge提供稳定的系数估计
3. Lasso进行自动特征选择，减少模型复杂度
4. ElasticNet（若使用）则平衡两者

这些正则化技术是构建可靠、可解释机器学习模型的必备工具。

---

## 七、附件

- 原始数据: `data/ai-impact-jobs-layoff-risk-dataset.csv`
