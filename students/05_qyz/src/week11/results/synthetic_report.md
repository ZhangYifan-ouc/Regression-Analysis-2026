# 模拟回归数据实验报告

## A1. 模拟数据生成
### 1. 业务场景
本次模拟数据为 **二手车价格预测数据集**，目标是根据车辆属性预测二手车成交价格。

### 2. 数据规模
- 样本量：500 条（满足 ≥300 要求）

### 3. 特征设计
共 5 个特征：
1. car_age：连续型，车龄（年）
2. mileage：连续型，行驶里程（万公里）
3. engine_power：连续型，发动机功率
4. transmission：类别型，变速箱（Manual/Automatic）
5. fuel_type：类别型，燃油类型（Petrol/Diesel）

满足：
- 至少 4 个特征 
- 至少 2 个连续变量 
- 至少 1 个类别变量 
- 显式构造一组高度相关特征 

### 4. 高度相关特征构造
mileage 由 car_age 直接生成：mileage = 1.2 * car_age + 高斯噪声
使 car_age 与 mileage 呈现强线性相关。

### 5. 目标变量生成公式（显式 DGP）
price = 5000
800 * car_age
300 * mileage
50 * engine_power
2000 * (transmission == "Automatic")
1500 * (fuel_type == "Diesel")
N(0, 1500²)

### 6. 主动加入的真实世界问题（≥2 类）
1. **缺失值**
   - 变量：engine_power
   - 缺失比例：10%
   - 缺失数量：约 53 条

2. **异常值**
   - 变量：price
   - 异常数量：10 条（5 条极高价、5 条负价格）

3. **特征量纲差异明显**
   - car_age：1–15
   - mileage：0–20
   - engine_power：50–300

4. **共线性**
   - car_age ↔ mileage 高度相关

### 7. 数据处理方式
- 对数值特征 engine_power 和 price 使用中位数填补
- 异常值：保留以模拟真实场景
- 类别特征：独热编码
- 共线性：保留特征，使用 VIF 诊断

---

## A2. 数据保存与 DGP 说明
### 1. 数据保存路径
src/week11/data/synthetic_regression.csv

### 2. 变量影响方向
**正向影响价格（越高越贵）**
- engine_power
- transmission_Automatic
- fuel_type_Diesel

**负向影响价格（越高越便宜）**
- car_age
- mileage

### 3. 冗余/高度相关变量
- car_age 和 mileage 高度相关、互为冗余特征。

---

## A3. 建模与评估流程（使用自定义 utils）
### 1. 数据清洗
- 对对数值特征 engine_power 和目标变量 price 使用中位数填补
- 对 transmission、fuel_type 独热编码
- 保留异常值以贴近真实场景

### 2. 预处理
使用自己实现的 `CustomStandardScaler` 完成标准化。

### 3. 模型
使用自己实现的 `AnalyticalOLS`。

### 4. 验证方式
5 折交叉验证，每折仅在训练集拟合 scaler，**无数据泄露**。

### 5. 评估指标（5 折平均）
- -RMSE = 3663.36
- MAE = 1888.90
- MAPE = 145.15%
- R² = 0.6965

### 6. 共线性诊断（VIF）
- car_age VIF = 7.59
- mileage VIF = 7.59
- engine_power VIF = 1.00
- transmission_Manual VIF = 1.00
- fuel_type_Petrol VIF = 1.00

**结论：存在中度多重共线性，无严重共线性。**

---

## A4. 推测（结果分析）
### 1. 模型识别方向与 DGP 是否一致？
**基本一致。**
- car_age、mileage 对价格呈负向影响
- engine_power、自动挡、柴油车呈正向影响

### 2. 系数不准确的原因
- 存在中度共线性（car_age ↔ mileage）
- 数据含噪声与异常值
- 缺失值填补带来轻微分布变化

### 3. 难以稳定识别的变量
**car_age 和 mileage 无法同时稳定识别。**
原因：两个特征高度冗余，模型难以区分真实效应，是典型共线性导致的不稳定性。

---

## 实验结论
本实验完整复现了真实回归建模场景，构造了包含缺失值、异常值、量纲差异、多重共线性的数据集；基于自定义工具库完成了数据清洗、标准化、建模、5 折交叉验证、模型评估与共线性诊断；模型预测方向与真实 DGP 一致，共线性是影响系数稳定性的主要因素。