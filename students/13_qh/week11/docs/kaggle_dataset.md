# Kaggle 数据集介绍

## 数据集名称

**Student Performance Factors**

## 来源

Kaggle 平台
https://www.kaggle.com/datasets/grandmaster07/student-exam-performance-dataset-analysis
## 预测目标

**Exam_Score**（考试成绩）— 连续变量，表示学生的最终考试分数

## 业务含义

每行代表一个学生的学习特征和考试成绩，用于分析哪些因素会影响学生的学业表现。

## 数据规模

- 样本数：6607
- 特征数：19

## 特征说明

| 特征名 | 类型 | 含义 |
|--------|------|------|
| Hours_Studied | 数值 | 学习时长 |
| Attendance | 数值 | 出勤率 |
| Sleep_Hours | 数值 | 睡眠时长 |
| Previous_Scores | 数值 | 以往成绩 |
| Tutoring_Sessions | 数值 | 辅导课次数 |
| Physical_Activity | 数值 | 体育活动时长 |
| Parental_Involvement | 类别 | 家长参与度（Low/Medium/High） |
| Access_to_Resources | 类别 | 资源获取程度 |
| Extracurricular_Activities | 类别 | 是否参加课外活动 |
| Motivation_Level | 类别 | 学习动机水平 |
| Internet_Access | 类别 | 是否有网络 |
| Family_Income | 类别 | 家庭收入 |
| Teacher_Quality | 类别 | 教师质量 |
| School_Type | 类别 | 学校类型（Public/Private） |
| Peer_Influence | 类别 | 同伴影响 |
| Learning_Disabilities | 类别 | 是否有学习障碍 |
| Parental_Education_Level | 类别 | 家长学历 |
| Distance_from_Home | 类别 | 家校距离 |
| Gender | 类别 | 性别 |

## 缺失值

| 特征 | 缺失数量 | 缺失比例 |
|------|----------|----------|
| Teacher_Quality | 78 | 1.18% |
| Parental_Education_Level | 90 | 1.36% |
| Distance_from_Home | 67 | 1.01% |

## 选择原因

1. 真实业务数据，不是教学演示数据
2. 包含数值型和类别型特征
3. 预测目标是连续变量（回归问题）
4. 有缺失值，需要预处理
