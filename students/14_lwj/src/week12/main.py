import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# --------------------------
# 直接在这里定义RMSE/MAE，不用import了！
# --------------------------
def calculate_rmse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def calculate_mae(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(np.abs(y_true - y_pred))

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 100
})

# ----------------------------------------------------
def generate_data(n_samples=200, noise=0.5, random_state=42):
    np.random.seed(random_state)
    x = np.linspace(0, 10, n_samples)
    y_true = np.sin(x) + 0.2 * x
    y = y_true + np.random.normal(0, noise, size=n_samples)
    x = x.reshape(-1, 1)
    return x, y, y_true

# ----------------------------------------------------
def run_candidate_models(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    degrees = [1, 4, 15]
    plt.figure(figsize=(12, 6))
    plt.scatter(x_train, y_train, label='Train', alpha=0.5, s=20)
    plt.scatter(x_test, y_test, label='Test', alpha=0.5, s=20)

    for d in degrees:
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=d)),
            ('lr', LinearRegression())
        ])
        model.fit(x_train, y_train)
        y_plot = model.predict(x)
        tr_rmse = calculate_rmse(y_train, model.predict(x_train))
        te_rmse = calculate_rmse(y_test, model.predict(x_test))
        plt.plot(x, y_plot, label=f'deg={d} | tr={tr_rmse:.2f}, te={te_rmse:.2f}')

    plt.title('Three Candidate Models (Underfit / Good / Overfit)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('src/week12/results/figures/candidate_models.png')
    plt.close()

# ----------------------------------------------------
def run_model_complexity_demo(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    degrees = list(range(1, 19))
    tr, te = [], []
    for d in degrees:
        model = Pipeline([('p', PolynomialFeatures(d)), ('lr', LinearRegression())])
        model.fit(x_train, y_train)
        tr.append(calculate_rmse(y_train, model.predict(x_train)))
        te.append(calculate_rmse(y_test, model.predict(x_test)))

    plt.figure(figsize=(12, 5))
    plt.plot(degrees, tr, marker='o', label='Train RMSE')
    plt.plot(degrees, te, marker='s', label='Test RMSE')
    plt.title('Model Complexity vs RMSE (Overfit Point)')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('src/week12/results/figures/error_curves.png')
    plt.close()
    return degrees, tr, te

# ----------------------------------------------------
def run_variance_demo(x, y_true):
    plt.figure(figsize=(12, 6))
    for d, c, lab in [(2, 'blue', 'Low Var (deg=2)'), (15, 'red', 'High Var (deg=15)')]:
        for _ in range(10):
            y = y_true + np.random.normal(0, 0.5, size=len(y_true))
            xt, _, yt, _ = train_test_split(x, y, test_size=0.3)
            model = Pipeline([('p', PolynomialFeatures(d)), ('lr', LinearRegression())])
            model.fit(xt, yt)
            plt.plot(x, model.predict(x), color=c, alpha=0.3)
    plt.plot(x, y_true, 'k-', linewidth=3, label='True Function')
    plt.title('Variance: High Complexity = Unstable Predictions')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('src/week12/results/figures/variance_demo.png')
    plt.close()

# ----------------------------------------------------
def run_loss_comparison_demo():
    y_true = np.array([1,2,3,4,5,6,7,8,9,10])
    clean = np.array([1.1,2.1,2.9,4.2,4.8,6.1,6.9,8.2,8.8,10.1])
    outlier = clean.copy()
    outlier[-1] = 100

    r_clean = calculate_rmse(y_true, clean)
    m_clean = calculate_mae(y_true, clean)
    r_out = calculate_rmse(y_true, outlier)
    m_out = calculate_mae(y_true, outlier)

    plt.figure(figsize=(10,5))
    plt.bar(['Clean','Outlier'], [r_clean, r_out], width=0.4, label='RMSE')
    plt.bar(['Clean','Outlier'], [m_clean, m_out], width=0.4, label='MAE')
    plt.title('RMSE is Sensitive to Outliers')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('src/week12/results/figures/loss_outlier_comparison.png')
    plt.close()

    return {
        'clean': {'rmse': r_clean, 'mae': m_clean},
        'outlier': {'rmse': r_out, 'mae': m_out}
    }

# ----------------------------------------------------
def write_summary_report(degrees, tr, te, loss):
    best = degrees[np.argmin(te)]
    s = f"""# Week12 Summary

## 1. Three Key Conclusions
1. As model complexity increases, training error keeps falling but test error eventually rises (overfitting).
2. High complexity brings high variance: predictions become unstable across different samples.
3. RMSE is very sensitive to outliers; MAE is robust.

## 2. Best Overfitting Figure
error_curves.png: test RMSE starts increasing after degree {best}. This is the clearest visual proof of overfitting.

## 3. When to Use RMSE / MAE
- RMSE: when large mistakes are very costly (finance, safety).
- MAE: when data has many outliers or you want stable average error.

## 4. Why Regularization?
High complexity → high variance → overfitting.
Regularization (Ridge/Lasso) limits model complexity to reduce variance.

## Model Complexity Table
| deg | train_rmse | test_rmse |
|-----|------------|-----------|
"""
    for d, a, b in zip(degrees, tr, te):
        s += f"| {d} | {a:.3f} | {b:.3f} |\n"

    s += f"""
## Loss Comparison
| Scene | RMSE | MAE |
|-------|------|-----|
| Clean | {loss['clean']['rmse']:.3f} | {loss['clean']['mae']:.3f} |
| Outlier | {loss['outlier']['rmse']:.3f} | {loss['outlier']['mae']:.3f} |
"""
    with open('src/week12/results/summary.md', 'w', encoding='utf-8') as f:
        f.write(s)

# ----------------------------------------------------
def main():
    print("[Stage 1] Generate data...")
    x, y, y_true = generate_data()

    print("[Stage 2] Candidate models...")
    run_candidate_models(x, y)

    print("[Stage 3] Model complexity curve...")
    degrees, tr, te = run_model_complexity_demo(x, y)

    print("[Stage 4] Variance demo...")
    run_variance_demo(x, y_true)

    print("[Stage 5] Loss comparison...")
    loss = run_loss_comparison_demo()

    print("[Stage 6] Write summary...")
    write_summary_report(degrees, tr, te, loss)

    print("✅ ALL DONE — week12 finished!")

if __name__ == "__main__":
    main()