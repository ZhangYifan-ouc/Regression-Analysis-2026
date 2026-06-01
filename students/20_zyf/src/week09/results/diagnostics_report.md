# 🛠️ Week 9: Model Diagnostics & Cross-Validation

**Generated:** 2026-05-12 21:09:32.102939

## Data Information

[INFO] Loaded cleaned data from: students/20_zyf/results/week09/clean_marketing.csv
[INFO] Data shape: (1000, 7)
[INFO] Columns: ['TV_Budget', 'Online_Video_Budget', 'Radio_Budget', 'Sales', 'Region_North', 'Region_South', 'Region_West']

- **Shape:** 1000 samples × 7 features
- **Features:** TV_Budget, Online_Video_Budget, Radio_Budget, Region_North, Region_South, Region_West
- **Target:** Sales

## Task 1: Multicollinearity Diagnostics

| Feature | VIF Value | Status |
|---------|-----------|--------|
| TV_Budget | 16.7642 | 🔴 SEVERE |
| Online_Video_Budget | 17.3208 | 🔴 SEVERE |
| Radio_Budget | 1.0000 | 🟢 OK |
| Region_North | 1.3052 | 🟢 OK |
| Region_South | 1.3068 | 🟢 OK |
| Region_West | 1.3122 | 🟢 OK |

### ⚠️ WARNING: HIGH MULTICOLLINEARITY DETECTED!

The following features have **VIF > 10**:

- **TV_Budget**: VIF = 16.7642
- **Online_Video_Budget**: VIF = 17.3208

These features show severe multicollinearity and should be investigated.

## Task 2: 5-Fold Cross-Validation

### Cross-Validation Progress

- **Fold 1/5:** R² = 0.990323
- **Fold 2/5:** R² = 0.990102
- **Fold 3/5:** R² = 0.990191
- **Fold 4/5:** R² = 0.989613
- **Fold 5/5:** R² = 0.988447

### Cross-Validation Statistics

| Metric | Value |
|--------|-------|
| Mean R² | 0.989735 |
| Std Dev | 0.000688 |
| Min R² | 0.988447 |
| Max R² | 0.990323 |

### ⭐ Baseline Cross-Validation Score: 0.989735

## Critical Reflection: Data Leakage Issue

### Question

In `data_prep.py`, we imputed all missing values using the mean of the **ENTIRE** dataset.
During 5-fold cross-validation, does the test set represent truly "unseen" data?

### Answer: **NO!** 🚨

### Why This Is a Problem

The imputation leakage occurs because:

1. **Means Computed on Full Dataset**: We computed statistics on all 1000 samples before splitting
2. **Information Leakage**: These means implicitly "leak" information into the test set
3. **Contaminated Test Set**: The test set observations are "contaminated" with statistics
   that include their own values indirectly

### Better Approach (Data Leakage Prevention)

```python
# WRONG (current approach):
mean = df.mean()  # computed on FULL data
for fold in cv.splits():
    df_test = df.iloc[test_idx].fillna(mean)  # test data contaminated!

# RIGHT (proper approach):
for fold in cv.splits():
    train_data, test_data = split(df, fold)
    mean = train_data.mean()  # computed ONLY on train
    train_data = train_data.fillna(mean)
    test_data = test_data.fillna(mean)  # apply same mean to test
```

### Key Takeaway

The **artificially high** $R^2 = 0.9897$ is partly due to this leakage.
In Week 10, we will implement proper preprocessing that prevents this issue
by fitting all transformations ONLY on training data.

---

✅ **Report Generation Complete**
