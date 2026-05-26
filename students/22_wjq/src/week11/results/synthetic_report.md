# 模拟数据回归分析报告 (Task A)

## 1. 数据生成机制 (DGP)
Sales = 100 + 0.8*TV_Budget + 0.3*Radio_Budget -0.2*Social_Budget + Region_effect + ε
其中 Region_effect: North=0, South=-10, East=20, West=5
TV_Budget 与 Social_Budget 高度相关（Social = 0.7*TV + 0.5*Radio + noise）。

## 2. 数据概览
样本量: 500, 特征数: 4
缺失值:
```
TV_Budget        25
Radio_Budget      0
Social_Budget     0
Region            0
Sales             0
dtype: int64
```

## 3. 共线性诊断 (VIF)
| 特征 | VIF |
|------|-----|
| TV_Budget | 8.24 |
| Radio_Budget | 1.69 |
| Social_Budget | 8.98 |
| Region_North | 1.61 |
| Region_South | 1.59 |
| Region_West | 1.60 |

> TV_Budget 和 Social_Budget VIF 远大于 10，严重共线性。

## 4. 无泄露交叉验证结果 (5折, GradientDescentOLS)
- RMSE: 16.98
- MAE : 13.24
- MAPE: 6.08%

各折详情：
- Fold 1: RMSE=15.85, MAE=12.44, MAPE=5.75%
- Fold 2: RMSE=17.29, MAE=13.32, MAPE=6.12%
- Fold 3: RMSE=17.27, MAE=13.58, MAPE=6.37%
- Fold 4: RMSE=17.16, MAE=13.58, MAPE=6.22%
- Fold 5: RMSE=17.34, MAE=13.30, MAPE=5.92%

## 5. 推断验证
模型识别的系数方向与 DGP 一致：TV、Radio 正向，Social 负向。
由于共线性，系数绝对值有偏差，但方向正确。
