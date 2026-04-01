import numpy as np
import time
from sklearn.linear_model import LinearRegression, SGDRegressor
import statsmodels.api as sm

# ========== 你的手写求解器 ===========
class AnalyticalSolver:
    def fit(self, X, y):
        # 增加偏置项
        X_ = np.hstack([np.ones((X.shape[0], 1)), X])
        # 用np.linalg.solve替代inv
        self.beta = np.linalg.solve(X_.T @ X_, X_.T @ y)
        return self
    def predict(self, X):
        X_ = np.hstack([np.ones((X.shape[0], 1)), X])
        return X_ @ self.beta

class GradientDescentSolver:
    def __init__(self, lr=1e-3, epochs=1000):
        self.lr = lr
        self.epochs = epochs
    def fit(self, X, y):
        X_ = np.hstack([np.ones((X.shape[0], 1)), X])
        N, P = X_.shape
        self.beta = np.zeros(P)
        for _ in range(self.epochs):
            y_pred = X_ @ self.beta
            grad = (2/N) * X_.T @ (y_pred - y)
            self.beta -= self.lr * grad
        return self
    def predict(self, X):
        X_ = np.hstack([np.ones((X.shape[0], 1)), X])
        return X_ @ self.beta

# ========== 评估函数 ===========
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def benchmark_solver(solver, X, y, X_test, y_test, name):
    start = time.time()
    solver.fit(X, y)
    y_pred = solver.predict(X_test)
    cost = time.time() - start
    error = mse(y_test, y_pred)
    print(f"{name:30s} | Time: {cost:.3f}s | MSE: {error:.6f}")
    return cost, error

# ========== 数据生成 ===========
def gen_data(N, P, noise=0.1, seed=42):
    np.random.seed(seed)
    X = np.random.randn(N, P)
    true_beta = np.random.randn(P + 1)
    X_ = np.hstack([np.ones((N, 1)), X])
    y = X_ @ true_beta + noise * np.random.randn(N)
    return X, y, true_beta

if __name__ == "__main__":
    for N, P, tag in [(10000, 10, 'Low Dim'), (10000, 2000, 'High Dim')]:
        print(f"\n===== {tag} (N={N}, P={P}) =====")
        X, y, _ = gen_data(N, P)
        X_test, y_test, _ = gen_data(2000, P, seed=24)
        # 1. AnalyticalSolver
        benchmark_solver(AnalyticalSolver(), X, y, X_test, y_test, 'AnalyticalSolver')
        # 2. GradientDescentSolver
        benchmark_solver(GradientDescentSolver(lr=1e-3, epochs=500), X, y, X_test, y_test, 'GradientDescentSolver')
        # 3. statsmodels.api.OLS
        start = time.time()
        X_sm = sm.add_constant(X)
        model = sm.OLS(y, X_sm).fit()
        y_pred = model.predict(sm.add_constant(X_test))
        cost = time.time() - start
        error = mse(y_test, y_pred)
        print(f"{'statsmodels.OLS':30s} | Time: {cost:.3f}s | MSE: {error:.6f}")
        # 4. sklearn.linear_model.LinearRegression
        start = time.time()
        lr = LinearRegression()
        lr.fit(X, y)
        y_pred = lr.predict(X_test)
        cost = time.time() - start
        error = mse(y_test, y_pred)
        print(f"{'sklearn.LinearRegression':30s} | Time: {cost:.3f}s | MSE: {error:.6f}")
        # 5. sklearn.linear_model.SGDRegressor
        start = time.time()
        sgd = SGDRegressor(max_iter=500, tol=1e-3)
        sgd.fit(X, y)
        y_pred = sgd.predict(X_test)
        cost = time.time() - start
        error = mse(y_test, y_pred)
        print(f"{'sklearn.SGDRegressor':30s} | Time: {cost:.3f}s | MSE: {error:.6f}")
