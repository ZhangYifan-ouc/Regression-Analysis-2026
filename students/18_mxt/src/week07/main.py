import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

from utils.models import AnalyticalOLS, GradientDescentOLS

def task2_cv(X, y):
    print("\n===== Task 2: 5-Fold Cross Validation =====")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_list = []
    rmse_list = []

    for i, (train_idx, val_idx) in enumerate(kf.split(X)):
        Xt, Xv = X[train_idx], X[val_idx]
        yt, yv = y[train_idx], y[val_idx]
        model = AnalyticalOLS().fit(Xt, yt)
        r2 = model.score(Xv, yv)
        rmse_val = rmse(yv, model.predict(Xv))
        r2_list.append(r2)
        rmse_list.append(rmse_val)
        print(f"Fold {i+1} | R2: {r2:.4f} | RMSE: {rmse_val:.4f}")

    print(f"\nAverage R2: {np.mean(r2_list):.4f}")
    print(f"Average RMSE: {np.mean(rmse_list):.4f}")

def task3_tune(X_train, y_train, X_val, y_val):
    print("\n===== Task 3: Learning Rate Tuning =====")
    lrs = [0.1, 0.01, 0.001, 0.0001, 1e-5]
    best_lr = None
    best_r2 = -float("inf")

    for lr in lrs:
        model = GradientDescentOLS(learning_rate=lr, gd_type="mini_batch").fit(X_train, y_train)
        r2 = model.score(X_val, y_val)
        print(f"LR = {lr:8.5f} | Val R2 = {r2:.4f}")
        if r2 > best_r2:
            best_r2 = r2
            best_lr = lr

    print(f"\nBest Learning Rate: {best_lr}")
    return best_lr

def task4_plot_curve(X_train, y_train):
    print("\n===== Task 4: Plot Learning Curve =====")
    Path("results").mkdir(exist_ok=True)

    m1 = GradientDescentOLS(learning_rate=0.01, gd_type="full_batch", max_iter=300).fit(X_train, y_train)
    m2 = GradientDescentOLS(learning_rate=0.01, gd_type="mini_batch", max_iter=300).fit(X_train, y_train)

    plt.figure(figsize=(10,5))
    plt.plot(m1.loss_history_, label="Full Batch")
    plt.plot(m2.loss_history_, label="Mini Batch")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Learning Curve")
    plt.legend()
    plt.savefig("results/learning_curve_full_vs_mini.png", dpi=150)
    plt.close()
    print("✅ Learning curve saved!")

def main():
    df = pd.read_csv("homework/week06/data/q3_marketing.csv")
    
    # ✅ 终极修复：只保留数值列
    X = df.select_dtypes(include=[np.number]).iloc[:, :-1].values
    y = df.select_dtypes(include=[np.number]).iloc[:, -1].values

    X_with_bias = np.c_[np.ones(X.shape[0]), X]
    task2_cv(X_with_bias, y)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    X_train_s = np.c_[np.ones(X_train_s.shape[0]), X_train_s]
    X_val_s = np.c_[np.ones(X_val_s.shape[0]), X_val_s]
    X_test_s = np.c_[np.ones(X_test_s.shape[0]), X_test_s]

    best_lr = task3_tune(X_train_s, y_train, X_val_s, y_val)

    gd = GradientDescentOLS(learning_rate=best_lr, gd_type="mini_batch").fit(X_train_s, y_train)
    ols = AnalyticalOLS().fit(X_train_s, y_train)

    print("\n===== Final Test Performance =====")
    print(f"GradientDescent Test R2: {gd.score(X_test_s, y_test):.4f}")
    print(f"AnalyticalOLS Test R2: {ols.score(X_test_s, y_test):.4f}")

    task4_plot_curve(X_train_s, y_train)

    with open("results/summary_report.md", "w") as f:
        f.write(f"Best LR: {best_lr}\n")
        f.write(f"GD Test R2: {gd.score(X_test_s, y_test):.4f}\n")
        f.write(f"OLS Test R2: {ols.score(X_test_s, y_test):.4f}\n")

    print("\n🎉 WEEK07 ALL TASKS FINISHED!")

if __name__ == "__main__":
    main()
