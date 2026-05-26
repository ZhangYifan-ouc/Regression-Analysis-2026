import numpy as np
class CustomMeanImputer:
    """
    Mean imputer for numeric features.

    fit(X):
        Learn column means from training data only.

    transform(X):
        Fill missing values using the learned training means.
    """

    def __init__(self) -> None:
        self.mean_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)

        self.mean_ = np.nanmean(X, axis=0)
        self.mean_ = np.where(np.isnan(self.mean_), 0.0, self.mean_)

        return self

    def transform(self, X):
        if self.mean_ is None:
            raise ValueError("CustomMeanImputer must be fitted before transform().")

        X = np.asarray(X, dtype=float).copy()

        nan_rows, nan_cols = np.where(np.isnan(X))

        if len(nan_rows) > 0:
            X[nan_rows, nan_cols] = self.mean_[nan_cols]

        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class CustomStandardScaler:
    """
    Standard scaler.

    fit(X):
        Learn mean and standard deviation from training data only.

    transform(X):
        Standardize data using the learned training statistics.
    """

    def __init__(self) -> None:
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)

        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)

        self.std_ = np.where((self.std_ == 0) | np.isnan(self.std_), 1.0, self.std_)

        return self

    def transform(self, X):
        if self.mean_ is None or self.std_ is None:
            raise ValueError("CustomStandardScaler must be fitted before transform().")

        X = np.asarray(X, dtype=float)

        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class Winsorizer:
    """
    Winsorization transformer.

    It learns lower and upper quantiles from training data only, then clips
    both training and validation features using the same thresholds.
    """

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> None:
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)

        self.lower_bounds_ = np.nanquantile(X, self.lower_quantile, axis=0)
        self.upper_bounds_ = np.nanquantile(X, self.upper_quantile, axis=0)

        return self

    def transform(self, X):
        if self.lower_bounds_ is None or self.upper_bounds_ is None:
            raise ValueError("Winsorizer must be fitted before transform().")

        X = np.asarray(X, dtype=float).copy()

        return np.clip(X, self.lower_bounds_, self.upper_bounds_)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)