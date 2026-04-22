"""
Scenario A: Synthetic Data Baseline Test (White-box)
"""

import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from ols_model import CustomOLS
from evaluator import evaluate_model


def scenario_A_synthetic(results_dir: Path):
    """
    Scenario A: Generate synthetic data and verify correctness.
    
    Test our CustomOLS against sklearn's LinearRegression.
    Assert that R^2 calculations match expectations.
    """
    print("\n" + "="*70)
    print("场景A：合成数据基准测试")
    print("="*70)
    
    # Step 1: Generate synthetic data (DGP)
    np.random.seed(42)
    n_samples = 1000
    n_features = 4
    
    # True parameters
    true_beta = np.array([2.0, 0.5, -0.3, 0.8, 0.2])  # [intercept, beta1, beta2, beta3, beta4]
    
    # Generate features (without constant column yet)
    X_raw = np.random.randn(n_samples, n_features)
    
    # Add constant column (intercept)
    X = np.column_stack([np.ones(n_samples), X_raw])
    
    # Generate target: y = X @ beta + noise
    noise = np.random.randn(n_samples) * 0.1
    y = X @ true_beta + noise
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"已生成 {n_samples} 个合成数据样本")
    print(f"  训练集: {X_train.shape}, 测试集: {X_test.shape}")
    print(f"  真实系数: {true_beta}")
    
    # Step 2: Train CustomOLS
    model_custom = CustomOLS()
    model_custom.fit(X_train, y_train)
    
    custom_r2 = model_custom.score(X_test, y_test)
    custom_coef = model_custom.coef_
    
    print(f"\nCustomOLS 结果:")
    print(f"  估计系数: {custom_coef}")
    print(f"  测试R：{custom_r2:.4f}")
    
    # Step 3: Train sklearn's LinearRegression for comparison
    # Note: sklearn handles the intercept separately, so we pass X_raw and fit_intercept=True
    model_sklearn = LinearRegression(fit_intercept=True)
    X_train_raw = X_train[:, 1:]  # Remove constant column
    X_test_raw = X_test[:, 1:]
    model_sklearn.fit(X_train_raw, y_train)
    
    sklearn_r2 = model_sklearn.score(X_test_raw, y_test)
    sklearn_coef = np.concatenate([[model_sklearn.intercept_], model_sklearn.coef_])
    
    print(f"\nsklearn LinearRegression 结果:")
    print(f"  估计系数: {sklearn_coef}")
    print(f"  测试R：{sklearn_r2:.4f}")
    
    # Step 4: Assert correctness
    print(f"\n对比：")
    print(f"  R差异: {abs(custom_r2 - sklearn_r2):.6f}")
    print(f"  系数匹配: {np.allclose(custom_coef, sklearn_coef, atol=1e-6)}")
    
    assert np.allclose(custom_coef, sklearn_coef, atol=1e-6), \
        "CustomOLS and sklearn coefficients do not match!"
    assert np.allclose(custom_r2, sklearn_r2, atol=1e-6), \
        "CustomOLS and sklearn R² do not match!"
    
    # Step 5: Build comparison table
    results_header = "# 场景A：合成数据基准测试\n\n"
    results_header += "| 模型 | 训练时间(秒) | R异 |\n"
    results_header += "|-------|----------------|---------|\n"
    
    result_custom = evaluate_model(
        CustomOLS().fit(X_train, y_train), X_train, y_train, X_test, y_test,
        "CustomOLS"
    )
    
    # For sklearn comparison, we need to reconstruct with the proper format
    model_sklearn_full = LinearRegression(fit_intercept=True)
    model_sklearn_full.fit(X_train_raw, y_train)
    
    # Wrap sklearn model to have consistent interface
    class SklearnWrapper:
        def __init__(self, sklearn_model, X_train_raw):
            self.model = sklearn_model
            self.X_train_raw = X_train_raw
        
        def fit(self, X_train, y_train):
            # Handle both X with constant column and X_raw
            if X_train.shape[1] == self.X_train_raw.shape[1] + 1:
                # X has constant column, remove it
                self.model.fit(X_train[:, 1:], y_train)
            else:
                self.model.fit(X_train, y_train)
            return self
        
        def predict(self, X_test):
            if X_test.shape[1] == self.X_train_raw.shape[1] + 1:
                return self.model.predict(X_test[:, 1:])
            else:
                return self.model.predict(X_test)
        
        def score(self, X_test, y_test):
            if X_test.shape[1] == self.X_train_raw.shape[1] + 1:
                return self.model.score(X_test[:, 1:], y_test)
            else:
                return self.model.score(X_test, y_test)
    
    model_sklearn_wrapped = SklearnWrapper(model_sklearn_full, X_train_raw)
    result_sklearn = evaluate_model(
        model_sklearn_wrapped, X_train, y_train, X_test, y_test,
        "sklearn.LinearRegression"
    )
    
    results_content = results_header + result_custom + result_sklearn
    
    # Save results
    report_path = results_dir / "synthetic_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(results_content)
    
    print(f"\n✓ 功能性测试已保存到 {report_path}")
    
    return {
        'custom_r2': custom_r2,
        'sklearn_r2': sklearn_r2,
        'X_test': X_test,
        'y_test': y_test,
        'X_train': X_train,
        'y_train': y_train
    }
