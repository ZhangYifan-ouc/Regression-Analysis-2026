"""
Module: utils.transformers
Purpose: Data transformation classes following sklearn-style API.
"""
import numpy as np


class CustomImputer:
    """
    Missing value imputation transformer.
    
    Supports mean and median imputation strategies.
    Follows sklearn Transformer interface:
    - fit(X): Learn imputation values from data
    - transform(X): Apply learned imputation
    - fit_transform(X): Learn and apply in one step
    """
    
    def __init__(self, strategy: str = "mean"):
        """
        Parameters:
        -----------
        strategy : str
            Imputation strategy: 'mean' or 'median'
        """
        if strategy not in ("mean", "median"):
            raise ValueError("strategy must be 'mean' or 'median'")
        self.strategy = strategy
        self.fill_values_ = None
        self.is_fitted_ = False
    
    def fit(self, X: np.ndarray):
        """
        Learn fill values from the data.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
        
        Returns:
        --------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        
        if self.strategy == "mean":
            self.fill_values_ = np.nanmean(X, axis=0)
        else:  # median
            self.fill_values_ = np.nanmedian(X, axis=0)
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fill missing values using learned fill values.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
        
        Returns:
        --------
        X_filled : np.ndarray
        """
        if not self.is_fitted_:
            raise ValueError("Imputer must be fitted before transform. Call fit() first.")
        
        X = np.asarray(X, dtype=np.float64)
        X_filled = X.copy()
        
        for j in range(X_filled.shape[1]):
            mask = np.isnan(X_filled[:, j])
            if np.any(mask):
                X_filled[mask, j] = self.fill_values_[j]
        
        return X_filled
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Learn parameters and apply imputation in one step."""
        self.fit(X)
        return self.transform(X)


class CustomStandardScaler:
    """
    Standardization transformer using z-score normalization.
    
    Follows the sklearn Transformer interface:
    - fit(X): Learn parameters from data
    - transform(X): Apply learned parameters
    - fit_transform(X): Learn and apply in one step
    
    Formula: z = (x - mean) / std
    """
    
    def __init__(self):
        """Initialize the scaler."""
        self.mean_ = None
        self.std_ = None
        self.is_fitted_ = False
    
    def fit(self, X: np.ndarray):
        """
        Learn the mean and standard deviation from the data.
        
        Parameters:
        -----------
        X : np.ndarray
            Training data of shape (n_samples, n_features)
        
        Returns:
        --------
        self : CustomStandardScaler
            Returns self for method chaining
        """
        X = np.asarray(X, dtype=np.float64)
        
        # Handle missing values: compute mean/std ignoring NaN
        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0)
        
        # Avoid division by zero: if std is 0, set to 1
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        
        self.is_fitted_ = True
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply standardization using learned parameters.
        
        Parameters:
        -----------
        X : np.ndarray
            Data to transform of shape (n_samples, n_features)
        
        Returns:
        --------
        X_scaled : np.ndarray
            Transformed data
        """
        if not self.is_fitted_:
            raise ValueError("Scaler must be fitted before transform. Call fit() first.")
        
        X = np.asarray(X, dtype=np.float64)
        
        # Apply z-score normalization: (x - mean) / std
        # Preserve NaN values during transformation
        X_scaled = (X - self.mean_) / self.std_
        
        return X_scaled
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Learn parameters and apply transformation in one step.
        
        Parameters:
        -----------
        X : np.ndarray
            Training data of shape (n_samples, n_features)
        
        Returns:
        --------
        X_scaled : np.ndarray
            Transformed data
        """
        self.fit(X)
        return self.transform(X)
    
    def get_params(self) -> dict:
        """Get fitted parameters."""
        return {
            'mean_': self.mean_,
            'std_': self.std_,
            'is_fitted_': self.is_fitted_
        }
