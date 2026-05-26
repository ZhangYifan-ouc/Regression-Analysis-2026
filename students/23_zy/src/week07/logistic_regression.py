import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

class LogisticRegressionGD:
    def __init__(self, lr=0.01, n_iters=1000, penalty=None, C=1.0):
        self.lr = lr
        self.n_iters = n_iters
        self.penalty = penalty  # 'l1', 'l2' or None
        self.C = C
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # 正则化项
            if self.penalty == 'l2':
                dw += (1 / self.C) * self.weights
            elif self.penalty == 'l1':
                dw += (1 / self.C) * np.sign(self.weights)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return [1 if p >= threshold else 0 for p in proba]