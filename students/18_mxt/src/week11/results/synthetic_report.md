## 模拟数据回归分析

## 1. 业务场景与数据生成机制（DGP）
本次模拟数据以 **学生考试成绩预测** 为业务场景。

**DGP 公式：**
score = 1.8 * study_hour + 0.9 * effective_study - 0.6 * wrong_questions
3 * (group==1) + 5 * (group==2) + noise
plaintext

- **x1 (study_hour)**：学习时长（连续）
- **x2 (effective_study)**：有效学习时长（连续，由 x1 构造强相关）
- **x3 (wrong_questions)**：错题数量（连续）
- **group**：班级类别变量（1/2/3）

**共线性构造：**
x2 = 0.85 * x1 + 随机扰动
使 x1 与 x2 高度相关。

**加入真实数据问题：**
- 4% 随机缺失值
- 5% 异常值
- 特征量纲差异
- 强多重共线性

## 2. 变量影响方向
- **正向影响**：study_hour、effective_study、group（班级等级）
- **负向影响**：wrong_questions（错题越多成绩越低）

## 3. 建模流程（完全复用自定义 utils）
- 缺失值处理：CustomImputer（中位数填充）
- 异常值处理：winsorize 缩尾
- 标准化：CustomStandardScaler
- 模型：CustomOLS
- 评估：RMSE、MAE、MAPE（自定义实现）
- 诊断：VIF（自定义实现）
- 验证：5 折无泄露交叉验证

## 4. 实验结果
**5 折 CV 结果：**
- RMSE: 8.8857
- MAE: 4.9965
- MAPE: 24.2306%

**VIF 共线性诊断：**
| Feature | VIF    |
|---------|--------|
| x1      | 7.40   |
| x2      | 7.38   |
| x3      | 1.01   |
| group   | 1.01   |

**结论：x1 与 x2 存在严重多重共线性（VIF > 5）。**

## 5. 推断（Inference）结论
- 模型识别的**系数方向与 DGP 完全一致**。
- **x1、x2 无法稳定独立识别**，因为高度共线性导致信息冗余。
- MAPE 偏高由共线性导致，并非模型拟合不足。
- 噪声与共线性是误差主要来源。