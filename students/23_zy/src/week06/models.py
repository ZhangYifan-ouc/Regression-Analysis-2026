import numpy as np

class CustomOLS:
    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X, y):
        X_b = np.hstack([np.ones((len(X), 1)), X])
        theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        self.b = theta[0]
        self.w = theta[1:]  # 形状正确

    def predict(self, X):
        return self.b + X @ self.w