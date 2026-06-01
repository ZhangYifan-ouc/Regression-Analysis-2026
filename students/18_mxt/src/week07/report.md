# Week 07 实验报告

## 一、实验概述
实现解析解 OLS 与梯度下降 OLS，完成交叉验证、超参数寻优、特征标准化与学习曲线绘制。

## 二、模型实现
1. **AnalyticalOLS**：基于矩阵解析解求解回归系数。
2. **GradientDescentOLS**：支持 full_batch / mini_batch 梯度下降，记录 loss 历史，支持早停。

## 三、实验结果

### Task 2：5 折交叉验证（AnalyticalOLS）
- 平均 R²：0.9079
- 平均 RMSE：72.1430

### Task 3：学习率寻优
尝试学习率：`[0.1, 0.01, 0.001, 0.0001, 1e-5]`
**最佳学习率：0.1**

### Task 4：测试集最终表现
- GradientDescentOLS Test R²：0.8913
- AnalyticalOLS Test R²：0.8906

## 四、关键说明
1. **标准化**：仅在训练集拟合 scaler，避免数据泄露。
2. **梯度下降**：mini_batch 收敛更稳定，full_batch 损失下降更平滑。
3. **结论**：梯度下降与解析解性能几乎一致，证明优化器实现正确。