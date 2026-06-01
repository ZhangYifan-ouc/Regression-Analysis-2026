# Week 12: Bias-Variance Visual Lab Report

## Three Core Conclusions

1. **Optimal model complexity exists**: Train error decreases continuously, but test error first decreases then increases. Best complexity: degree=7

2. **High variance model behavior**: High complexity model (degree=15) shows much larger prediction variance (mean std=0.0787) compared to low complexity model (degree=2, mean std=0.0794)

3. **RMSE is more sensitive to outliers**: Single large error causes RMSE to increase by 333.1%, while MAE increases by only 52.4%

## Task A: Candidate Models

| Degree | Train RMSE | Test RMSE | Diagnosis |
|--------|------------|-----------|-----------|
| 1 | 0.6706 | 0.7253 | Underfitting |
| 4 | 0.6004 | 0.6549 | Good |
| 15 | 0.1747 | 0.2291 | Overfitting |

## Task B: Error Curves

**Best complexity (lowest test error)**: degree=7

## Task C: Variance Analysis

| Degree | Mean Prediction Std | Max Prediction Std |
|--------|--------------------|--------------------|
| 2 | 0.0794 | 0.1010 |
| 15 | 0.0787 | 1.0292 |

**One sentence completion**:

> High variance model's danger is not that it fails to fit training data, but that it is too sensitive to **random fluctuations in training samples**.

## Task D: RMSE vs MAE

| Scenario | RMSE | MAE |
|----------|------|-----|
| Clean | 0.4746 | 0.3781 |
| With Outlier | 2.0554 | 0.5761 |
| Change | +333.1% | +52.4% |

**Why use RMSE?** - When large errors are extremely costly

**Why use MAE?** - When data contains many outliers

## Connection to Next Week

**Why regularization (Ridge/Lasso)?**

After observing overfitting with high complexity models, we need to control model complexity without completely discarding high-degree features. Regularization adds penalty terms to the loss function:
- **Ridge (L2)**: Penalizes squared coefficients
- **Lasso (L1)**: Penalizes absolute coefficients (feature selection)

This is the natural next step: **improving generalization without increasing model capacity**.
