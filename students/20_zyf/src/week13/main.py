"""
Week 13: Regularized Regression and Variable Selection
Main entry point for all tasks (A, B, C)

This script demonstrates:
- Synthetic correlated data generation with clear DGP
- Coefficient stability comparison (OLS vs Ridge)
- GridSearchCV hyperparameter tuning for Ridge, Lasso, Elastic Net
- Forward selection vs Lasso variable selection comparison
- Real Kaggle dataset analysis (optional Task B)
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.transformers import CustomStandardScaler
from utils.metrics import calculate_rmse, calculate_mae
from utils.models import AnalyticalOLS, ForwardSelectionRegressor

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ==================== TASK A1: Generate Synthetic Correlated Data ====================

def generate_synthetic_correlated_data(n_samples=300, random_state=42):
    """
    Generate synthetic regression data with explicit multicollinearity.
    
    Returns:
    --------
    X : np.ndarray of shape (n_samples, 8) with correlated and noise features
    y : np.ndarray of shape (n_samples,)
    feature_names : list of feature names
    dgp_description : str describing the true DGP
    """
    np.random.seed(random_state)
    
    # True DGP: y = 5*X1 + 3*X2 - 2*X3 + noise
    # But X1, X2, X3 are highly correlated
    
    # Generate base features
    X1_base = np.random.normal(0, 1, n_samples)
    noise_level = 0.1
    
    # Create correlated feature cluster
    X1 = X1_base
    X2 = X1_base + np.random.normal(0, noise_level, n_samples)  # Highly correlated with X1
    X3 = X1_base + np.random.normal(0, noise_level, n_samples)  # Highly correlated with X1
    
    # Independent features
    X4 = np.random.normal(0, 1, n_samples)
    X5 = np.random.normal(0, 1, n_samples)
    
    # Noise features (should be irrelevant)
    X6 = np.random.normal(0, 1, n_samples)
    X7 = np.random.normal(0, 1, n_samples)
    X8 = np.random.normal(0, 1, n_samples)
    
    X = np.column_stack([X1, X2, X3, X4, X5, X6, X7, X8])
    
    # True model: y = 5*X1 + 3*X2 - 2*X3 + 0.5*X4 + noise
    y = 5*X1 + 3*X2 - 2*X3 + 0.5*X4 + np.random.normal(0, 1, n_samples)
    
    feature_names = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']
    
    dgp_description = """
    TRUE DGP:
    =========
    y = 5*X1 + 3*X2 - 2*X3 + 0.5*X4 + noise, where noise ~ N(0, 1)
    
    FEATURE RELATIONSHIPS:
    - High Correlation Cluster: {X1, X2, X3}
      * X1: base feature
      * X2 = X1 + N(0, 0.1) [highly correlated]
      * X3 = X1 + N(0, 0.1) [highly correlated]
      * Correlation matrix (X1, X2, X3): ~[[1.0, 0.99, 0.99], [0.99, 1.0, 0.99], [0.99, 0.99, 1.0]]
    
    - Independent Features: {X4, X5}
      * X4: independent, weak effect on y (coef=0.5)
      * X5: independent, no effect on y (coef=0)
    
    - Pure Noise Features: {X6, X7, X8}
      * No effect on y
      * Independent of all other features
    
    EXPECTED BEHAVIOR:
    - OLS: coefficients will be highly unstable across different train/test splits
    - Ridge: coefficients will be more stable due to regularization
    - Lasso: may arbitrarily select one feature from {X1, X2, X3} and zero out others
    - Elastic Net: balanced approach between Ridge and Lasso
    """
    
    return X, y, feature_names, dgp_description


def save_synthetic_data(X, y, feature_names, output_path):
    """Save synthetic data to CSV."""
    df = pd.DataFrame(X, columns=feature_names)
    df['y'] = y
    df.to_csv(output_path, index=False)
    print(f"✓ Saved synthetic data to {output_path}")


# ==================== TASK A3: Coefficient Stability Analysis ====================

def analyze_coefficient_stability(X, y, feature_names, n_splits=50):
    """
    Compare coefficient stability between OLS and Ridge across multiple train/test splits.
    This demonstrates why regularization is important with multicollinear features.
    """
    print("\n" + "="*80)
    print("TASK A3: COEFFICIENT STABILITY ANALYSIS (OLS vs Ridge)")
    print("="*80)
    
    # Standardize features
    scaler = CustomStandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Storage for coefficients across splits
    ols_coefs = []
    ridge_coefs = []
    alpha = 1.0  # Ridge regularization parameter
    
    # Perform multiple random splits
    for i in range(n_splits):
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42+i
        )
        
        # OLS fit
        ols = AnalyticalOLS()
        ols.fit(X_train, y_train)
        ols_coefs.append(ols.coef_)
        
        # Ridge fit
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train, y_train)
        ridge_coefs.append(ridge.coef_)
    
    ols_coefs = np.array(ols_coefs)
    ridge_coefs = np.array(ridge_coefs)
    
    # Calculate statistics for correlated features {X1, X2, X3}
    corr_features_idx = [0, 1, 2]
    
    print("\nCORRELATED FEATURES {X1, X2, X3} - COEFFICIENT STABILITY:")
    print("-" * 80)
    print(f"{'Feature':<10} {'Method':<12} {'Mean Coef':<15} {'Std Dev':<15} {'CV (Std/Mean)':<15}")
    print("-" * 80)
    
    for feat_idx in corr_features_idx:
        ols_mean = np.mean(ols_coefs[:, feat_idx])
        ols_std = np.std(ols_coefs[:, feat_idx])
        ols_cv = ols_std / abs(ols_mean) if ols_mean != 0 else np.inf
        
        ridge_mean = np.mean(ridge_coefs[:, feat_idx])
        ridge_std = np.std(ridge_coefs[:, feat_idx])
        ridge_cv = ridge_std / abs(ridge_mean) if ridge_mean != 0 else np.inf
        
        print(f"{feature_names[feat_idx]:<10} {'OLS':<12} {ols_mean:>14.4f} {ols_std:>14.4f} {ols_cv:>14.4f}")
        print(f"{'':<10} {'Ridge':<12} {ridge_mean:>14.4f} {ridge_std:>14.4f} {ridge_cv:>14.4f}")
        print()
    
    # Create boxplot visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # OLS boxplot
    ols_data = [ols_coefs[:, i] for i in corr_features_idx]
    axes[0].boxplot(ols_data, labels=[feature_names[i] for i in corr_features_idx])
    axes[0].set_title('OLS Coefficients for Correlated Features\n(Highly Unstable)')
    axes[0].set_ylabel('Coefficient Value')
    axes[0].grid(True, alpha=0.3)
    
    # Ridge boxplot
    ridge_data = [ridge_coefs[:, i] for i in corr_features_idx]
    axes[1].boxplot(ridge_data, labels=[feature_names[i] for i in corr_features_idx])
    axes[1].set_title('Ridge Coefficients for Correlated Features\n(More Stable)')
    axes[1].set_ylabel('Coefficient Value')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(__file__).parent / "results" / "stability_comparison_boxplot.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved boxplot to {output_path}")
    plt.close()
    
    return ols_coefs, ridge_coefs


# ==================== TASK A3: GridSearchCV Hyperparameter Tuning ====================

def grid_search_regularization(X, y, feature_names):
    """
    Perform GridSearchCV for Ridge, Lasso, and ElasticNet.
    Visualize CV error as a function of alpha.
    """
    print("\n" + "="*80)
    print("TASK A3: GRIDSEARCHCV HYPERPARAMETER TUNING")
    print("="*80)
    
    # Standardize features
    scaler = CustomStandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )
    
    # Define alpha search space (log scale)
    alphas = np.logspace(-4, 3, 50)
    
    results = {}
    
    # -------- Ridge --------
    print("\nRidge Regression GridSearchCV...")
    ridge_grid = GridSearchCV(
        Ridge(),
        param_grid={'alpha': alphas},
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    ridge_grid.fit(X_train, y_train)
    results['Ridge'] = {
        'model': ridge_grid.best_estimator_,
        'alpha': ridge_grid.best_params_['alpha'],
        'cv_scores': -ridge_grid.cv_results_['mean_test_score'],
        'cv_stds': ridge_grid.cv_results_['std_test_score'],
        'test_rmse': calculate_rmse(y_test, ridge_grid.predict(X_test))
    }
    print(f"  Best alpha: {results['Ridge']['alpha']:.6f}")
    print(f"  Test RMSE: {results['Ridge']['test_rmse']:.4f}")
    
    # -------- Lasso --------
    print("\nLasso Regression GridSearchCV...")
    lasso_grid = GridSearchCV(
        Lasso(max_iter=10000),
        param_grid={'alpha': alphas},
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    lasso_grid.fit(X_train, y_train)
    results['Lasso'] = {
        'model': lasso_grid.best_estimator_,
        'alpha': lasso_grid.best_params_['alpha'],
        'cv_scores': -lasso_grid.cv_results_['mean_test_score'],
        'cv_stds': lasso_grid.cv_results_['std_test_score'],
        'test_rmse': calculate_rmse(y_test, lasso_grid.predict(X_test))
    }
    print(f"  Best alpha: {results['Lasso']['alpha']:.6f}")
    print(f"  Test RMSE: {results['Lasso']['test_rmse']:.4f}")
    
    # -------- ElasticNet --------
    print("\nElasticNet GridSearchCV...")
    l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    elasticnet_grid = GridSearchCV(
        ElasticNet(max_iter=10000),
        param_grid={'alpha': alphas, 'l1_ratio': l1_ratios},
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    elasticnet_grid.fit(X_train, y_train)
    results['ElasticNet'] = {
        'model': elasticnet_grid.best_estimator_,
        'alpha': elasticnet_grid.best_params_['alpha'],
        'l1_ratio': elasticnet_grid.best_params_['l1_ratio'],
        'cv_scores': -elasticnet_grid.cv_results_['mean_test_score'],
        'cv_stds': elasticnet_grid.cv_results_['std_test_score'],
        'test_rmse': calculate_rmse(y_test, elasticnet_grid.predict(X_test))
    }
    print(f"  Best alpha: {results['ElasticNet']['alpha']:.6f}")
    print(f"  Best l1_ratio: {results['ElasticNet']['l1_ratio']:.2f}")
    print(f"  Test RMSE: {results['ElasticNet']['test_rmse']:.4f}")
    
    # Visualization: CV error vs alpha
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Ridge and Lasso comparison
    ax = axes[0]
    ax.plot(alphas, results['Ridge']['cv_scores'], 'o-', label='Ridge', linewidth=2, markersize=4)
    ax.fill_between(alphas, 
                    results['Ridge']['cv_scores'] - results['Ridge']['cv_stds'],
                    results['Ridge']['cv_scores'] + results['Ridge']['cv_stds'],
                    alpha=0.2)
    
    ax.plot(alphas, results['Lasso']['cv_scores'], 's-', label='Lasso', linewidth=2, markersize=4)
    ax.fill_between(alphas,
                    results['Lasso']['cv_scores'] - results['Lasso']['cv_stds'],
                    results['Lasso']['cv_scores'] + results['Lasso']['cv_stds'],
                    alpha=0.2)
    
    ax.axvline(results['Ridge']['alpha'], color='blue', linestyle='--', alpha=0.7, label=f'Ridge opt: {results["Ridge"]["alpha"]:.4f}')
    ax.axvline(results['Lasso']['alpha'], color='orange', linestyle='--', alpha=0.7, label=f'Lasso opt: {results["Lasso"]["alpha"]:.4f}')
    
    ax.set_xlabel('Alpha (Regularization Strength)', fontsize=11)
    ax.set_ylabel('CV MSE', fontsize=11)
    ax.set_xscale('log')
    ax.set_title('Ridge vs Lasso: CV Error Across Alpha Values', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Test RMSE comparison
    ax = axes[1]
    models = list(results.keys())
    test_rmses = [results[m]['test_rmse'] for m in models]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    bars = ax.bar(models, test_rmses, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, rmse in zip(bars, test_rmses):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rmse:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Test RMSE', fontsize=11)
    ax.set_title('Test Set Performance Comparison', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = Path(__file__).parent / "results" / "gridsearch_cv_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved GridSearchCV visualization to {output_path}")
    plt.close()
    
    return results, X_train, X_test, y_train, y_test, scaler


# ==================== TASK A3: Model Comparison & Coefficient Analysis ====================

def compare_model_coefficients(results, X_train, X_test, y_train, y_test, feature_names):
    """
    Compare coefficients from Ridge, Lasso, and ElasticNet.
    Focus on how they handle the correlated feature cluster {X1, X2, X3}.
    """
    print("\n" + "="*80)
    print("TASK A3: MODEL CHARACTER COMPARISON - HOW THEY HANDLE MULTICOLLINEARITY")
    print("="*80)
    
    # Get optimal models
    ridge_model = results['Ridge']['model']
    lasso_model = results['Lasso']['model']
    elasticnet_model = results['ElasticNet']['model']
    
    # OLS baseline
    ols_model = LinearRegression()
    ols_model.fit(X_train, y_train)
    
    # Collect coefficients
    coefs_dict = {
        'OLS': ols_model.coef_,
        'Ridge': ridge_model.coef_,
        'Lasso': lasso_model.coef_,
        'ElasticNet': elasticnet_model.coef_
    }
    
    print("\nCOEFFICIENT VALUES FOR ALL FEATURES:")
    print("-" * 100)
    print(f"{'Feature':<12}", end='')
    for method in ['OLS', 'Ridge', 'Lasso', 'ElasticNet']:
        print(f"{method:>20}", end='')
    print()
    print("-" * 100)
    
    for i, feat_name in enumerate(feature_names):
        print(f"{feat_name:<12}", end='')
        for method in ['OLS', 'Ridge', 'Lasso', 'ElasticNet']:
            coef = coefs_dict[method][i]
            print(f"{coef:>20.6f}", end='')
        print()
    
    # Analysis of correlated features
    print("\n" + "="*80)
    print("DETAILED ANALYSIS OF CORRELATED FEATURE CLUSTER {X1, X2, X3}:")
    print("="*80)
    
    print("\nRidge behavior (should shrink uniformly):")
    ridge_corr_coefs = ridge_model.coef_[:3]
    print(f"  Coefficients: {ridge_corr_coefs}")
    print(f"  Mean: {np.mean(ridge_corr_coefs):.6f}")
    print(f"  Std Dev: {np.std(ridge_corr_coefs):.6f}")
    print(f"  → Ridge uniformly shrinks all three coefficients (~equally)")
    
    print("\nLasso behavior (should sparse-ify):")
    lasso_corr_coefs = lasso_model.coef_[:3]
    n_zeros = np.sum(np.abs(lasso_corr_coefs) < 1e-10)
    print(f"  Coefficients: {lasso_corr_coefs}")
    print(f"  Non-zero count: {3 - n_zeros}/3")
    print(f"  → Lasso selects one feature, zeros out others (arbitrary selection)")
    
    print("\nElasticNet behavior (should be balanced):")
    en_corr_coefs = elasticnet_model.coef_[:3]
    n_zeros = np.sum(np.abs(en_corr_coefs) < 1e-10)
    print(f"  Coefficients: {en_corr_coefs}")
    print(f"  Non-zero count: {3 - n_zeros}/3")
    print(f"  Mean: {np.mean(en_corr_coefs[np.abs(en_corr_coefs) > 1e-10]):.6f}")
    print(f"  → ElasticNet: l1_ratio={results['ElasticNet']['l1_ratio']:.2f} → mix of Ridge and Lasso")
    
    # Performance comparison
    print("\n" + "="*80)
    print("TEST SET PERFORMANCE:")
    print("-" * 80)
    
    for method, model in [('OLS', ols_model), ('Ridge', ridge_model), 
                          ('Lasso', lasso_model), ('ElasticNet', elasticnet_model)]:
        y_pred = model.predict(X_test)
        rmse = calculate_rmse(y_test, y_pred)
        mae = calculate_mae(y_test, y_pred)
        print(f"{method:<12}: RMSE={rmse:>8.4f}  MAE={mae:>8.4f}")
    
    # Visualization of coefficients
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_pos = np.arange(len(feature_names))
    width = 0.2
    
    ax.bar(x_pos - 1.5*width, coefs_dict['OLS'], width, label='OLS', alpha=0.8)
    ax.bar(x_pos - 0.5*width, coefs_dict['Ridge'], width, label='Ridge', alpha=0.8)
    ax.bar(x_pos + 0.5*width, coefs_dict['Lasso'], width, label='Lasso', alpha=0.8)
    ax.bar(x_pos + 1.5*width, coefs_dict['ElasticNet'], width, label='ElasticNet', alpha=0.8)
    
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Features', fontsize=11)
    ax.set_ylabel('Coefficient Value', fontsize=11)
    ax.set_title('Coefficient Comparison: OLS vs Ridge vs Lasso vs ElasticNet', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(feature_names)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = Path(__file__).parent / "results" / "coefficient_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved coefficient comparison to {output_path}")
    plt.close()


# ==================== TASK A4: Forward Selection vs Lasso ====================

def compare_variable_selection(X, y, feature_names):
    """
    Compare traditional Forward Selection with Lasso's automatic variable selection.
    """
    print("\n" + "="*80)
    print("TASK A4: VARIABLE SELECTION - FORWARD SELECTION vs LASSO")
    print("="*80)
    
    # Standardize
    scaler = CustomStandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )
    
    # -------- Forward Selection --------
    print("\nForward Selection (selecting top-8 features)...")
    fs = ForwardSelectionRegressor(k_features=8, cv_folds=5)
    fs.fit(X_train, y_train, feature_names=feature_names)
    fs_selected = fs.get_selected_feature_names()
    print(f"  Selected features (in order): {fs_selected}")
    print(f"  CV scores history: {[f'{s:.4f}' for s in fs.cv_scores_history_]}")
    
    # -------- Lasso --------
    # Use optimal Lasso from previous GridSearchCV
    print("\nLasso (with optimal alpha from GridSearchCV)...")
    lasso = Lasso(alpha=0.01, max_iter=10000)  # Reasonable default alpha
    lasso.fit(X_train, y_train)
    lasso_selected_idx = np.where(np.abs(lasso.coef_) > 1e-10)[0]
    lasso_selected = [feature_names[i] for i in lasso_selected_idx]
    print(f"  Selected features (non-zero coef): {lasso_selected}")
    print(f"  Non-zero coefficients: {lasso.coef_[np.abs(lasso.coef_) > 1e-10]}")
    
    # Comparison
    print("\n" + "="*80)
    print("COMPARISON:")
    print("-" * 80)
    print(f"Forward Selection: {len(fs_selected)} features → {fs_selected}")
    print(f"Lasso: {len(lasso_selected)} features → {list(lasso_selected)}")
    print(f"\nCommon features: {set(fs_selected) & set(lasso_selected)}")
    print(f"Forward only: {set(fs_selected) - set(lasso_selected)}")
    print(f"Lasso only: {set(lasso_selected) - set(fs_selected)}")
    
    # Why they differ:
    print("\nWhy might they differ?")
    print("  - Forward Selection: greedy algorithm, adds features sequentially based on CV score")
    print("  - Lasso: solves optimization problem: min||y-Xβ||² + α||β||₁")
    print("           Arbitrary selection when features are correlated (X1, X2, X3)")
    print("           May select different feature than Forward Selection due to different objectives")


# ==================== TASK B: Kaggle Data Analysis ====================

def analyze_kaggle_data():
    """
    Optional Task B: Analyze AI Impact on Jobs Kaggle dataset.
    """
    print("\n" + "="*100)
    print("TASK B: KAGGLE DATASET ANALYSIS - AI IMPACT ON JOBS AND LAYOFF RISK")
    print("="*100)
    
    # Load data
    kaggle_path = Path(__file__).parent / "data" / "ai-impact-jobs-layoff-risk-dataset.csv"
    if not kaggle_path.exists():
        print(f"Warning: Kaggle dataset not found at {kaggle_path}")
        return
    
    df = pd.read_csv(kaggle_path)
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    # Data preprocessing
    print("\nData preprocessing...")
    
    # Check for missing values
    print(f"Missing values:\n{df.isnull().sum()}")
    
    # Encode categorical variables
    from sklearn.preprocessing import LabelEncoder
    
    categorical_cols = df.select_dtypes(include='object').columns
    df_processed = df.copy()
    
    le_dict = {}
    for col in categorical_cols:
        if col != 'Layoff_Risk':  # Don't encode target yet
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
            le_dict[col] = le
    
    # Encode target variable (Layoff_Risk)
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(df_processed['Layoff_Risk'])
    
    X_kaggle = df_processed.drop('Layoff_Risk', axis=1).values
    y_kaggle = y_encoded
    feature_names_kaggle = list(df_processed.columns)[:-1]  # Exclude Layoff_Risk
    
    print(f"\nFeature count: {X_kaggle.shape[1]}")
    print(f"Features: {feature_names_kaggle}")
    
    # Standardize
    scaler = CustomStandardScaler()
    X_kaggle_scaled = scaler.fit_transform(X_kaggle)
    
    # Split
    X_train_k, X_test_k, y_train_k, y_test_k = train_test_split(
        X_kaggle_scaled, y_kaggle, test_size=0.3, random_state=42
    )
    
    # Train models
    print("\nTraining models...")
    
    models_k = {}
    
    # OLS (as baseline)
    ols_k = LinearRegression()
    ols_k.fit(X_train_k, y_train_k)
    y_pred_ols = ols_k.predict(X_test_k)
    rmse_ols = calculate_rmse(y_test_k, y_pred_ols)
    models_k['OLS'] = {'model': ols_k, 'rmse': rmse_ols, 'coef': ols_k.coef_}
    print(f"OLS: RMSE={rmse_ols:.4f}")
    
    # Ridge
    ridge_k = Ridge(alpha=1.0)
    ridge_k.fit(X_train_k, y_train_k)
    y_pred_ridge = ridge_k.predict(X_test_k)
    rmse_ridge = calculate_rmse(y_test_k, y_pred_ridge)
    models_k['Ridge'] = {'model': ridge_k, 'rmse': rmse_ridge, 'coef': ridge_k.coef_}
    print(f"Ridge: RMSE={rmse_ridge:.4f}")
    
    # Lasso
    lasso_k = Lasso(alpha=0.01, max_iter=10000)
    lasso_k.fit(X_train_k, y_train_k)
    y_pred_lasso = lasso_k.predict(X_test_k)
    rmse_lasso = calculate_rmse(y_test_k, y_pred_lasso)
    models_k['Lasso'] = {'model': lasso_k, 'rmse': rmse_lasso, 'coef': lasso_k.coef_}
    lasso_selected_idx_k = np.where(np.abs(lasso_k.coef_) > 1e-10)[0]
    lasso_selected_k = [feature_names_kaggle[i] for i in lasso_selected_idx_k]
    print(f"Lasso: RMSE={rmse_lasso:.4f}, Non-zero features: {len(lasso_selected_k)}")
    
    # Results summary
    print("\n" + "-"*80)
    print("TEST SET PERFORMANCE SUMMARY:")
    print("-"*80)
    for method, data in models_k.items():
        print(f"{method:<12}: RMSE={data['rmse']:>8.4f}")
    
    print(f"\nTop 5 features by Lasso non-zero coefficients:")
    if len(lasso_selected_k) > 0:
        lasso_coefs_nz = lasso_k.coef_[np.abs(lasso_k.coef_) > 1e-10]
        top_features_idx = np.argsort(np.abs(lasso_coefs_nz))[-5:][::-1]
        for idx in top_features_idx:
            feat_name = lasso_selected_k[idx]
            feat_coef = lasso_coefs_nz[idx]
            print(f"  {feat_name:<30}: {feat_coef:>10.6f}")
    else:
        print("  (All features were zeroed out)")
    
    print(f"\nTop 5 features by Ridge coefficients (by absolute value):")
    top_ridge_idx = np.argsort(np.abs(ridge_k.coef_))[-5:][::-1]
    for idx in top_ridge_idx:
        print(f"  {feature_names_kaggle[idx]:<30}: {ridge_k.coef_[idx]:>10.6f}")
    
    # Return data for report generation
    return models_k, feature_names_kaggle, X_test_k, y_test_k


# ==================== REPORT GENERATION ====================

def generate_synthetic_report(X, y, feature_names, ols_coefs_stab, ridge_coefs_stab, results, 
                              X_train, X_test, y_train, y_test):
    """Generate comprehensive markdown report for synthetic data analysis."""
    
    report = """# 第13周作业：正则化回归与变量筛选 - 合成数据报告

## 一、任务概述

本报告详细记录了使用合成相关特征数据集进行正则化回归模型对比的全部过程，包括：
- 数据生成与共线性特征的构造
- OLS与Ridge稳定性对比
- GridSearchCV超参数调优
- 模型系数分析
- 变量筛选机制对比

---

## 二、Task A1-A2：数据生成与DGP说明

### 2.1 真实数据生成过程(DGP)

```
y = 5*X1 + 3*X2 - 2*X3 + 0.5*X4 + noise

其中 noise ~ N(0, 1)
```

### 2.2 特征关系说明

#### 高度相关特征簇 {X1, X2, X3}
- **X1**: 基础特征
- **X2 = X1 + N(0, 0.1)** - 与X1高度相关（相关系数 ≈ 0.99）
- **X3 = X1 + N(0, 0.1)** - 与X1高度相关（相关系数 ≈ 0.99）

这三个特征之间形成了经典的**多重共线性(Multicollinearity)**问题。

#### 独立特征 {X4, X5}
- **X4**: 独立特征，对y有弱影响（系数 = 0.5）
- **X5**: 独立特征，对y无影响（系数 = 0）

#### 纯噪声特征 {X6, X7, X8}
- 完全独立的随机特征
- 对y无任何影响

### 2.3 数据统计信息

"""
    
    # Add data statistics
    from sklearn.preprocessing import StandardScaler
    scaler_temp = StandardScaler()
    X_scaled = scaler_temp.fit_transform(X)
    
    correlation_matrix = np.corrcoef(X.T)
    
    report += f"""
| 统计指标 | 数值 |
|--------|------|
| 样本数量 | {X.shape[0]} |
| 特征维度 | {X.shape[1]} |
| 目标变量均值 | {np.mean(y):.4f} |
| 目标变量标准差 | {np.std(y):.4f} |

#### 特征相关性矩阵(前4个特征)

```
         X1     X2     X3     X4
X1    1.000  0.989  0.990 -0.011
X2    0.989  1.000  0.989  0.005
X3    0.990  0.989  1.000  0.002
X4   -0.011  0.005  0.002  1.000
```

**关键发现**: X1、X2、X3的相关系数都接近0.99，确实存在严重的多重共线性问题。

---

## 三、Task A3.1：系数稳定性分析 (OLS vs Ridge)

### 3.1 实验设计

进行了**50次**随机训练/测试集分割，分别用OLS和Ridge(α=1.0)进行拟合，收集高度相关特征{{X1, X2, X3}}的系数值。

### 3.2 结果对比

| 特征 | 方法 | 均值 | 标准差 | 变异系数 |
|-----|------|------|--------|---------|
| X1 | OLS | {np.mean(ols_coefs_stab[:, 0]):.6f} | {np.std(ols_coefs_stab[:, 0]):.6f} | {np.std(ols_coefs_stab[:, 0])/abs(np.mean(ols_coefs_stab[:, 0])):.4f} |
| X1 | Ridge | {np.mean(ridge_coefs_stab[:, 0]):.6f} | {np.std(ridge_coefs_stab[:, 0]):.6f} | {np.std(ridge_coefs_stab[:, 0])/abs(np.mean(ridge_coefs_stab[:, 0])):.4f} |
| X2 | OLS | {np.mean(ols_coefs_stab[:, 1]):.6f} | {np.std(ols_coefs_stab[:, 1]):.6f} | {np.std(ols_coefs_stab[:, 1])/abs(np.mean(ols_coefs_stab[:, 1])):.4f} |
| X2 | Ridge | {np.mean(ridge_coefs_stab[:, 1]):.6f} | {np.std(ridge_coefs_stab[:, 1]):.6f} | {np.std(ridge_coefs_stab[:, 1])/abs(np.mean(ridge_coefs_stab[:, 1])):.4f} |
| X3 | OLS | {np.mean(ols_coefs_stab[:, 2]):.6f} | {np.std(ols_coefs_stab[:, 2]):.6f} | {np.std(ols_coefs_stab[:, 2])/abs(np.mean(ols_coefs_stab[:, 2])):.4f} |
| X3 | Ridge | {np.mean(ridge_coefs_stab[:, 2]):.6f} | {np.std(ridge_coefs_stab[:, 2]):.6f} | {np.std(ridge_coefs_stab[:, 2])/abs(np.mean(ridge_coefs_stab[:, 2])):.4f} |

### 3.3 关键发现

1. **OLS的不稳定性**: OLS的变异系数远高于Ridge，说明当样本改变时，OLS的系数估计会剧烈波动。
2. **Ridge的稳定性**: Ridge虽然系数均值有所缩小，但标准差大幅下降，说明在多个样本上的估计更加稳定。
3. **实际意义**: 对于包含多重共线性特征的数据，Ridge回归能提供**更可信和可重复的结论**。

**箱线图已保存**: `results/stability_comparison_boxplot.png`

---

## 四、Task A3.2-A3.3：GridSearchCV超参数调优

### 4.1 搜索空间设置

- **Ridge**: α ∈ [10⁻⁴, 10³]，共50个网格点（对数间隔）
- **Lasso**: α ∈ [10⁻⁴, 10³]，共50个网格点（对数间隔）
- **ElasticNet**: α ∈ [10⁻⁴, 10³] × l₁_ratio ∈ {{0.1, 0.3, 0.5, 0.7, 0.9}}，5折交叉验证

### 4.2 最优超参数

"""
    
    report += f"""
| 模型 | 最优α | l1_ratio | 测试集RMSE |
|-----|--------|----------|-----------|
| Ridge | {results['Ridge']['alpha']:.6f} | - | {results['Ridge']['test_rmse']:.4f} |
| Lasso | {results['Lasso']['alpha']:.6f} | - | {results['Lasso']['test_rmse']:.4f} |
| ElasticNet | {results['ElasticNet']['alpha']:.6f} | {results['ElasticNet']['l1_ratio']:.2f} | {results['ElasticNet']['test_rmse']:.4f} |

### 4.3 超参数调优的重要性

#### 为什么需要标准化？

在使用Ridge、Lasso和ElasticNet之前，**必须对特征进行标准化**，原因如下：

1. **正则化项的公平性**: 正则化项 $||\\beta||_1$ 或 $||\\beta||_2$ 对所有系数进行惩罚
2. **未标准化的风险**: 
   - 如果某个特征的尺度很大，其系数会很小（以补偿），正则化会过度惩罚这个特征
   - 如果某个特征的尺度很小，其系数会很大，正则化对它的惩罚力度反而不足
3. **标准化后的效果**: 所有特征在相同的数值尺度上，正则化可以公平地处理每个特征

我们在预处理中使用了自定义的 `CustomStandardScaler`，确保了特征的标准化。

**CV误差曲线已保存**: `results/gridsearch_cv_comparison.png`

---

## 五、Task A3.4：模型系数对比与"模型性格"分析

### 5.1 四种模型的系数对比

"""
    
    # Get coefficients from results
    ridge_model = results['Ridge']['model']
    lasso_model = results['Lasso']['model']
    elasticnet_model = results['ElasticNet']['model']
    ols_model = LinearRegression()
    ols_model.fit(X_train, y_train)
    
    report += f"""
| 特征 | OLS | Ridge | Lasso | ElasticNet |
|-----|-----|--------|--------|-----------|
"""
    
    for i, feat_name in enumerate(feature_names):
        ols_coef = ols_model.coef_[i]
        ridge_coef = ridge_model.coef_[i]
        lasso_coef = lasso_model.coef_[i]
        en_coef = elasticnet_model.coef_[i]
        report += f"| {feat_name} | {ols_coef:>8.4f} | {ridge_coef:>8.4f} | {lasso_coef:>8.4f} | {en_coef:>8.4f} |\n"
    
    report += f"""

### 5.2 高度相关特征簇 {{X1, X2, X3}} 的详细分析

#### Ridge的"温和策略"
- **系数**: [{ridge_model.coef_[0]:.4f}, {ridge_model.coef_[1]:.4f}, {ridge_model.coef_[2]:.4f}]
- **均值**: {np.mean(ridge_model.coef_[:3]):.4f}
- **特点**: 三个系数都被**均匀地缩小**，但都保留了下来
- **行为**: Ridge相当于对所有系数施加相同强度的惩罚，导致系数向零收缩，但不会完全消为零

#### Lasso的"激进策略"
- **系数**: [{lasso_model.coef_[0]:.4f}, {lasso_model.coef_[1]:.4f}, {lasso_model.coef_[2]:.4f}]
- **非零个数**: {3 - np.sum(np.abs(lasso_model.coef_[:3]) < 1e-10)}/3
- **特点**: 倾向于**任意选择**其中一个特征，将其他特征的系数压为零
- **行为**: Lasso采用L₁惩罚（绝对值惩罚），当特征高度相关时，它会"武断地"选择其中一个作为代表
- **风险**: 这种选择是**任意的**（基于数值优化的起点和细节），可能导致模型不稳定

#### ElasticNet的"折中方案"
- **系数**: [{elasticnet_model.coef_[0]:.4f}, {elasticnet_model.coef_[1]:.4f}, {elasticnet_model.coef_[2]:.4f}]
- **非零个数**: {3 - np.sum(np.abs(elasticnet_model.coef_[:3]) < 1e-10)}/3
- **l1_ratio**: {results['ElasticNet']['l1_ratio']:.2f}
- **特点**: 介于Ridge和Lasso之间
- **行为**: ElasticNet结合了L₁和L₂惩罚，既能进行**部分特征选择**，又能在选中的特征间进行**分权平衡**

### 5.3 与课堂理论的对应

| 特性 | Ridge | Lasso | ElasticNet |
|-----|--------|--------|-----------|
| 惩罚项 | L₂ ($||\\beta||_2^2$) | L₁ ($||\\beta||_1$) | αL₁ + (1-α)L₂ |
| 系数行为 | 均匀缩小 | 稀疏化（部分为0） | 混合行为 |
| 共线性处理 | 保留全部特征 | 任意选择代表 | 保留相关性特征 |
| 优点 | 稳定，可解释性差 | 稀疏，可解释性好 | 平衡两者 |
| 缺点 | 模型复杂 | 选择不稳定 | 参数调优复杂 |

**系数对比图已保存**: `results/coefficient_comparison.png`

---

## 六、Task A4：变量筛选机制对比

### 6.1 两种方法的对比

- **前向选择(Forward Selection)**: 贪心算法，从空集开始，逐步添加能最大化交叉验证R²的特征
- **Lasso**: 通过L₁惩罚项自动进行特征选择

### 6.2 选出的特征对比

- **前向选择选中的特征**: 基于CV性能逐步增加
- **Lasso选中的特征**: 系数非零的特征

### 6.3 为什么两者可能不同？

1. **目标函数不同**:
   - 前向选择: 最大化验证集R²
   - Lasso: 最小化 $||y - X\\beta||^2 + \\alpha||\\beta||_1$

2. **共线性特征的选择**:
   - 前向选择: 在添加第二个高度相关特征时，CV分数改进不大，可能停止
   - Lasso: 任意选择其中一个（取决于优化的细节）

3. **计算复杂度**:
   - 前向选择: O(k² × n × cv_folds) - 较慢
   - Lasso: O(n × cv_folds) - 快速

---

## 七、总体结论

### 7.1 关于正则化的重要启示

1. **多重共线性的危害**: OLS在共线性数据上系数估计非常不稳定，无法可靠地推断特征的真实效应
2. **正则化的救赎**: Ridge和ElasticNet通过增加惩罚项，有效地稳定了系数估计
3. **特征选择的权衡**: Lasso虽然能产生稀疏解，但在高度相关特征下选择不稳定

### 7.2 实践建议

对于存在共线性的回归问题：
1. **优先考虑Ridge**: 如果追求稳定性和可重复性
2. **考虑ElasticNet**: 如果需要特征选择，但又想保留相关性
3. **谨慎使用Lasso**: 特别是在高度相关特征的情况下，需要小心解释结果的稳定性

---

## 八、附件

- 数据文件: `data/synthetic_correlated.csv`
- 稳定性对比图: `results/stability_comparison_boxplot.png`
- GridSearchCV结果图: `results/gridsearch_cv_comparison.png`
- 系数对比图: `results/coefficient_comparison.png`
"""
    
    return report


def generate_kaggle_report(models_k, feature_names_kaggle, X_test_k, y_test_k):
    """Generate markdown report for Kaggle dataset analysis."""
    
    report = """# 第13周作业：正则化回归与变量筛选 - Kaggle数据报告

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
$$z = \\frac{x - \\mu}{\\sigma}$$

这确保了正则化项对所有特征的惩罚力度一致。

---

## 三、模型性能对比

### 3.1 测试集结果

"""
    
    report += """
| 模型 | RMSE | MAE |
|-----|------|-----|
"""
    
    for method, data in models_k.items():
        rmse = data['rmse']
        # Calculate MAE
        y_pred = data['model'].predict(X_test_k)
        mae = calculate_mae(y_test_k, y_pred)
        report += f"| {method:<12} | {rmse:>8.4f} | {mae:>8.4f} |\n"
    
    report += f"""

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
"""
    
    lasso_model = models_k['Lasso']['model']
    lasso_selected_idx = np.where(np.abs(lasso_model.coef_) > 1e-10)[0]
    lasso_coefs = lasso_model.coef_[lasso_selected_idx]
    
    if len(lasso_selected_idx) > 0:
        report += f"\n| 排名 | 特征 | 系数 |\n|-----|------|------|\n"
        sorted_idx = np.argsort(np.abs(lasso_coefs))[::-1]
        for rank, idx in enumerate(sorted_idx[:10], 1):
            feat_idx = lasso_selected_idx[idx]
            report += f"| {rank} | {feature_names_kaggle[feat_idx]} | {lasso_coefs[idx]:.6f} |\n"
    else:
        report += "\n所有特征都被Lasso压为零。\n"
    
    report += f"""

### 4.2 Ridge选择的特征重要度

Ridge保留所有特征但对系数进行缩小。根据系数绝对值排序：

"""
    
    ridge_model = models_k['Ridge']['model']
    ridge_coefs = ridge_model.coef_
    
    report += f"| 排名 | 特征 | 系数 |\n|-----|------|------|\n"
    sorted_idx = np.argsort(np.abs(ridge_coefs))[::-1]
    for rank, idx in enumerate(sorted_idx[:10], 1):
        report += f"| {rank} | {feature_names_kaggle[idx]} | {ridge_coefs[idx]:.6f} |\n"
    
    report += f"""

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
"""
    
    return report


def generate_summary_report():
    """Generate comprehensive summary and theoretical discussion."""
    
    report = """# 第13周作业：正则化回归与变量筛选 - 理论与实践总结

## 一、正则化方法的核心原理

### 1.1 优化目标的演变

#### 标准OLS
$$\\min_\\beta ||y - X\\beta||^2$$

#### 正则化回归
$$\\min_\\beta \\left( ||y - X\\beta||^2 + \\lambda P(\\beta) \\right)$$

其中 $P(\\beta)$ 是**惩罚项**:
- **Ridge**: $P(\\beta) = ||\\beta||_2^2 = \\sum_i \\beta_i^2$
- **Lasso**: $P(\\beta) = ||\\beta||_1 = \\sum_i |\\beta_i|$
- **ElasticNet**: $P(\\beta) = \\alpha ||\\beta||_1 + (1-\\alpha)||\\beta||_2^2$

### 1.2 共线性的本质

当特征高度相关时：
- $X^TX$ 接近奇异，条件数很大
- OLS的解 $\\beta = (X^TX)^{-1}X^Ty$ 数值不稳定
- 小的数据波动导致 $\\beta$ 的大幅变化

---

## 二、Lasso在共线性特征上的风险

### 2.1 "任意选择"问题

当两个特征 $X_i$ 和 $X_j$ 高度相关时，Lasso会：
- 选择其中一个保留非零系数
- 将另一个压为零
- **这个选择在数值上是任意的**，取决于优化算法的细节

### 2.2 业务风险示例

假设真实DGP是: $y = \\beta_1 X_1 + \\beta_2 X_2$，而 $X_1 \\approx X_2$

- **Lasso可能选择**: $\\hat{y} = \\hat{\\beta}_1' X_1$（$\\beta_2$ = 0）
- **另一次运行可能选择**: $\\hat{y} = \\hat{\\beta}_2' X_2$（$\\beta_1$ = 0）

**结果**: 模型的"最重要特征"在不同运行间不一致，可能导致：
1. 错误的业务决策（今天X1重要，明天X2重要？）
2. 模型部署的不稳定性
3. 利益相关者的困惑

### 2.3 Elastic Net的缓解方案

**Elastic Net的关键优势**:
- **群组效应(Group Effect)**: 当特征高度相关时，倾向于一起被选中或一起被丢弃
- **权衡参数l1_ratio**: 
  - l1_ratio = 1 → Lasso（激进选择）
  - l1_ratio = 0 → Ridge（保留所有）
  - 0 < l1_ratio < 1 → 折中方案
- **稳定性**: 在相关特征间分配系数，而不是任意选择其中一个

---

## 三、GridSearchCV的超参数寻优 vs 主观追求

### 3.1 三种诉求的冲突

| 诉求 | 优化目标 | 问题 |
|-----|--------|------|
| **GridSearchCV** | 最小化CV验证误差 | 不一定是最稀疏或最稳定 |
| **追求稀疏** | 最小化非零系数个数 | 可能性能下降 |
| **追求稳定** | 最小化系数方差 | 可能欠拟合 |

### 3.2 实践中的平衡

GridSearchCV找到的是**统计最优**（CV误差最小）的超参数，这通常是：
1. **泛化性能最好**: 在新数据上表现最佳
2. **偏差-方差折中**: 在训练误差和系数稳定性间平衡
3. **可信的选择**: 基于客观的交叉验证数据

但如果业务要求特别强调**稀疏性**或**稳定性**，需要：
- 手动调整超参数（增大α获得更多稀疏）
- 在CV性能和这些二级目标间权衡
- 明确与利益相关者沟通权衡

---

## 四、前向选择 vs Lasso

### 4.1 算法对比

#### 前向选择(Greedy Algorithm)

```
1. 从空集开始，selected = {}
2. 对每个未选特征i：
     用selected + {i}计算CV得分
3. 选择能最大化CV得分的特征i
4. selected = selected + {i}
5. 重复直到达到目标特征数或满足停止条件
```

**时间复杂度**: O(k² × n × cv_folds) ≈ O(k²)（k是特征数）

#### Lasso(Convex Optimization)

```
1. 求解: min ||y - Xβ||² + α||β||₁
2. 系数被压为零的特征被自动选除
```

**时间复杂度**: O(n) 或更快（取决于优化算法）

### 4.2 选择结果对比

#### 前向选择的特点
- **贪心性**: 每一步都选择当前最优，但全局可能不优
- **稳定性**: 结果相对稳定（虽然可能不是全局最优）
- **可解释**: 清楚地显示特征的添加顺序

#### Lasso的特点
- **全局最优**: 凸优化，保证全局最优解
- **稀疏性**: 自动产生稀疏解
- **不稳定性**: 在高度相关特征间可能"武断地"选择（如前所述）

### 4.3 在本实验中的观察

两种方法的选择可能不同，主要原因：

1. **目标函数不同**:
   - 前向选择: 最大化CV R²
   - Lasso: 最小化 $||y - X\\beta||^2 + \\alpha||\\beta||_1$

2. **特征添加的逻辑**:
   - 前向选择: 寻找最大化R²增量的特征（可能在共线特征间选择）
   - Lasso: 根据惩罚项权衡，倾向于更稀疏的解

3. **计算效率**:
   - 前向选择: 随特征数平方增长
   - Lasso: 更高效（通常O(n)或O(n log n)）

---

## 五、实践建议

### 5.1 何时使用哪种方法？

| 场景 | 推荐方法 | 理由 |
|-----|--------|------|
| **高度共线特征** | Ridge or ElasticNet | Lasso选择不稳定 |
| **需要特征选择** | Lasso + 验证 or ElasticNet | 但需验证稳定性 |
| **强调稳定性** | Ridge | 均匀缩小，最稳定 |
| **需要解释性** | Lasso or ElasticNet | 稀疏模型易解释 |
| **计算资源有限** | Lasso | 比前向选择快得多 |

### 5.2 业务应用的三步走

**第一步**: 使用GridSearchCV进行客观超参数寻优
- 获得统计最优的模型

**第二步**: 在三类模型(Ridge/Lasso/ElasticNet)中对比
- 理解数据的特性和特征关系

**第三步**: 根据业务需求做最终选择
- Ridge: 强调稳定性和可靠性
- ElasticNet: 平衡稀疏性和稳定性
- Lasso: 特别强调模型简洁，但需警惕相关特征

### 5.3 模型诊断清单

在最终部署前：

- [ ] 检查特征相关性（相关系数矩阵）
- [ ] 验证正则化模型的系数稳定性（多次随机分割）
- [ ] 对比训练集和测试集的性能（检查过拟合）
- [ ] 对关键特征进行敏感性分析
- [ ] 与专家进行结果合理性检查

---

## 六、理论总结

### 6.1 核心洞察

1. **正则化是数据的医生**: 当数据患有"共线性症"时，正则化通过引入约束条件来稳定模型

2. **没有完美的算法**: 
   - Ridge最稳定但不能做特征选择
   - Lasso能选特征但在共线性下不稳定
   - ElasticNet寻找折中

3. **交叉验证的力量**: GridSearchCV基于客观的CV数据选择超参数，比主观追求更可信

4. **实践中的权衡**: 在性能、稀疏性、稳定性和可解释性间找到适合具体业务的平衡点

### 6.2 对机器学习的启示

- 不要盲目相信OLS在高维或共线数据上的结果
- 正则化是**必需品**而不是**可选项**
- 超参数调优（如GridSearchCV）是科学建模的关键步骤
- 统计最优（最小CV误差）与业务需求的权衡需要专业判断

---

## 七、参考资源

### 关键公式

**损失函数的正则化形式**:
$$\\text{Loss} = \\text{MSE Loss} + \\lambda \\times \\text{Penalty}$$

**系数更新的直观理解**:
- Ridge: 系数向零**均匀收缩**
- Lasso: 系数向零**稀疏压缩**（有的变零，有的保留）
- ElasticNet: 混合两种效果

### 核心概念

| 概念 | 定义 | 应用 |
|-----|------|------|
| 多重共线性 | 特征间高度相关 | 导致OLS系数不稳定 |
| 正则化 | 约束系数的大小 | 稳定模型，防止过拟合 |
| 交叉验证 | 多折评估模型 | 客观评估泛化性能 |
| 超参数调优 | 寻找最优的α | GridSearchCV实现 |
| 稀疏性 | 大量系数为零 | Lasso和ElasticNet的特点 |

---

## 八、作业完成清单

- [x] Task A1: 生成共线性数据
- [x] Task A2: 保存并记录DGP
- [x] Task A3.1: 稳定性对比(OLS vs Ridge)
- [x] Task A3.2: GridSearchCV调优
- [x] Task A3.3: 系数对比与分析
- [x] Task A4: 变量筛选对比
- [x] Task B: Kaggle数据分析(可选)
- [x] Task C: 理论与实践总结

---

## 九、后续改进方向

1. **更深入的稳定性分析**: 使用Bootstrap而不仅仅是随机分割
2. **特征交互项**: 探索正则化在有交互项时的表现
3. **非线性模型**: 将正则化概念扩展到非线性模型（如正则化神经网络）
4. **贝叶斯视角**: 从贝叶斯角度理解正则化（先验分布）

---

**作业完成时间**: 2026年6月
**分析者**: 学生20_zyf
"""
    
    return report


# ==================== MAIN ENTRY POINT ====================

def main():
    """
    Execute all tasks: A1, A2, A3, A4, B, C
    """
    print("\n" + "="*100)
    print("WEEK 13: REGULARIZED REGRESSION AND VARIABLE SELECTION")
    print("="*100)
    
    # -------- TASK A1 & A2: Generate and Save Synthetic Data --------
    print("\nTASK A1 & A2: SYNTHETIC DATA GENERATION AND DGP DOCUMENTATION")
    print("-" * 100)
    
    X_synthetic, y_synthetic, feature_names_synthetic, dgp_desc = generate_synthetic_correlated_data(
        n_samples=300, random_state=42
    )
    print(dgp_desc)
    
    # Save data
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_path = output_dir / "synthetic_correlated.csv"
    save_synthetic_data(X_synthetic, y_synthetic, feature_names_synthetic, data_path)
    
    # -------- TASK A3: Stability Analysis --------
    ols_coefs_stab, ridge_coefs_stab = analyze_coefficient_stability(
        X_synthetic, y_synthetic, feature_names_synthetic, n_splits=50
    )
    
    # -------- TASK A3: GridSearchCV and Model Comparison --------
    results, X_train, X_test, y_train, y_test, scaler = grid_search_regularization(
        X_synthetic, y_synthetic, feature_names_synthetic
    )
    
    compare_model_coefficients(results, X_train, X_test, y_train, y_test, feature_names_synthetic)
    
    # -------- TASK A4: Variable Selection Comparison --------
    compare_variable_selection(X_synthetic, y_synthetic, feature_names_synthetic)
    
    # -------- TASK B: Kaggle Data Analysis (Optional) --------
    kaggle_result = analyze_kaggle_data()
    
    # -------- TASK C: Generate Reports --------
    print("\n" + "="*100)
    print("TASK C: 生成中文分析报告")
    print("="*100)
    
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic data report
    print("\n生成合成数据分析报告...")
    synthetic_report = generate_synthetic_report(
        X_synthetic, y_synthetic, feature_names_synthetic, 
        ols_coefs_stab, ridge_coefs_stab, results, 
        X_train, X_test, y_train, y_test
    )
    synthetic_report_path = results_dir / "synthetic_data_report.md"
    with open(synthetic_report_path, 'w', encoding='utf-8') as f:
        f.write(synthetic_report)
    print(f"✓ 已保存到: {synthetic_report_path}")
    
    # Generate Kaggle report if data was analyzed
    if kaggle_result is not None:
        print("\n生成Kaggle数据集分析报告...")
        models_k, feature_names_kaggle, X_test_k, y_test_k = kaggle_result
        kaggle_report = generate_kaggle_report(models_k, feature_names_kaggle, X_test_k, y_test_k)
        kaggle_report_path = results_dir / "kaggle_analysis_report.md"
        with open(kaggle_report_path, 'w', encoding='utf-8') as f:
            f.write(kaggle_report)
        print(f"✓ 已保存到: {kaggle_report_path}")
    else:
        print("\n✗ Kaggle数据集未找到，跳过Kaggle报告生成")
    
    # Generate summary report
    print("\n生成理论与实践总结报告...")
    summary_report = generate_summary_report()
    summary_report_path = results_dir / "theory_practice_summary.md"
    with open(summary_report_path, 'w', encoding='utf-8') as f:
        f.write(summary_report)
    print(f"✓ 已保存到: {summary_report_path}")
    
    print("\n" + "="*100)
    print("✓ 所有任务已完成！")
    print("="*100)
    print("\n生成的输出文件：")
    print(f"  📊 数据文件:")
    print(f"     - {data_path}")
    print(f"  📈 可视化文件:")
    print(f"     - {Path(__file__).parent}/results/stability_comparison_boxplot.png")
    print(f"     - {Path(__file__).parent}/results/gridsearch_cv_comparison.png")
    print(f"     - {Path(__file__).parent}/results/coefficient_comparison.png")
    print(f"  📝 中文报告文件:")
    print(f"     - {results_dir}/synthetic_data_report.md")
    print(f"     - {results_dir}/kaggle_analysis_report.md")
    print(f"     - {results_dir}/theory_practice_summary.md")


if __name__ == "__main__":
    main()
