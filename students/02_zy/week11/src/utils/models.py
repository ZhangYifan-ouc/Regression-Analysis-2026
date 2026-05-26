import numpy as np
class CustomOLS:
    """
    Ordinary Least Squares regression.

    This model uses the pseudo-inverse, so it can still run when the design
    matrix is close to singular.
    """

    def __init__(self) -> None:
        self.coef_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        X_design = np.column_stack([np.ones(X.shape[0]), X])

        self.coef_ = np.linalg.pinv(X_design.T @ X_design) @ X_design.T @ y

        return self

    def predict(self, X):
        if self.coef_ is None:
            raise ValueError("Model must be fitted before prediction.")

        X = np.asarray(X, dtype=float)
        X_design = np.column_stack([np.ones(X.shape[0]), X])

        return X_design @ self.coef_


class GradientDescentOLS:
    """
    Linear regression trained by gradient descent.

    This class is kept in the toolbox for comparison or later extension.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 5000,
        tolerance: float = 1e-8,
    ) -> None:
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.coef_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        X_design = np.column_stack([np.ones(X.shape[0]), X])
        n_samples, n_features = X_design.shape

        self.coef_ = np.zeros(n_features)
        previous_loss = float("inf")

        for _ in range(self.n_iterations):
            y_pred = X_design @ self.coef_
            error = y_pred - y

            gradient = (2 / n_samples) * (X_design.T @ error)
            self.coef_ -= self.learning_rate * gradient

            current_loss = np.mean(error**2)

            if abs(previous_loss - current_loss) < self.tolerance:
                break

            previous_loss = current_loss

        return self

    def predict(self, X):
        if self.coef_ is None:
            raise ValueError("Model must be fitted before prediction.")

        X = np.asarray(X, dtype=float)
        X_design = np.column_stack([np.ones(X.shape[0]), X])

        return X_design @ self.coef_