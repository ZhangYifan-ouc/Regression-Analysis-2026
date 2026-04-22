import numpy as np
import scipy.stats as stats


class CustomOLS:
    """
    Custom implementation of Ordinary Least Squares (OLS) regression.
    
    The power of OOP: Each instance encapsulates its own state, allowing
    multiple independent models to coexist without interfering with each other.
    """
    
    def __init__(self):
        self.coef_ = None  # Beta hat: estimated coefficients
        self.cov_matrix_ = None  # Covariance matrix of beta
        self.sigma2_ = None  # Variance of residuals
        self.df_resid_ = None  # Degrees of freedom for residuals
        self.intercept_ = None  # Explicitly store intercept (first coef)
        self._X_train = None  # Store training data shape info
        self._y_train = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the OLS model.
        
        Args:
            X: (n_samples, n_features) feature matrix (should include constant column)
            y: (n_samples,) target vector
            
        Returns:
            self (for method chaining)
        """
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)
        
        n = X.shape[0]  # Number of samples
        k = X.shape[1]  # Number of features (including intercept column)
        
        # Step 1: Calculate beta_hat = (X^T X)^{-1} X^T y
        XtX = X.T @ X  # (k, k)
        Xty = X.T @ y  # (k, 1)
        
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            raise ValueError("X^T X is singular, cannot fit model")
        
        beta_hat = XtX_inv @ Xty  # (k, 1)
        self.coef_ = beta_hat.flatten()  # Store as 1D array
        self.intercept_ = self.coef_[0]  # First coefficient is intercept
        
        # Step 2: Calculate residuals and sigma2
        y_pred = X @ beta_hat  # (n, 1)
        residuals = y - y_pred  # (n, 1)
        SSE = (residuals ** 2).sum()  # Sum of squared errors
        
        # Step 3: Estimate variance
        self.df_resid_ = n - k  # Degrees of freedom
        self.sigma2_ = SSE / self.df_resid_
        
        # Step 4: Calculate covariance matrix
        # Cov(beta_hat) = sigma2 * (X^T X)^{-1}
        self.cov_matrix_ = self.sigma2_ * XtX_inv
        
        self._X_train = X
        self._y_train = y
        
        return self  # Enable chaining
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: (n_samples, n_features) feature matrix
            
        Returns:
            (n_samples,) predicted values
        """
        if self.coef_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.asarray(X)
        y_pred = X @ self.coef_
        return y_pred
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate R-squared (coefficient of determination).
        
        R^2 = 1 - (SSE / SST)
        
        Args:
            X: (n_samples, n_features) feature matrix
            y: (n_samples,) target vector
            
        Returns:
            R-squared value (between 0 and 1)
        """
        y = np.asarray(y).reshape(-1, 1)
        y_pred = self.predict(X).reshape(-1, 1)
        
        SST = ((y - y.mean()) ** 2).sum()  # Total sum of squares
        SSE = ((y - y_pred) ** 2).sum()  # Residual sum of squares
        
        r2 = 1 - (SSE / SST)
        return r2
    
    def f_test(self, C: np.ndarray, d: np.ndarray) -> dict:
        """
        Perform General Linear Hypothesis Test: H0: C*beta = d
        
        F statistic = (C*beta_hat - d)^T [C*(X^T X)^{-1}*C^T]^{-1} (C*beta_hat - d) / (q * sigma2)
        where q is the number of restrictions (rows of C)
        
        Args:
            C: (q, k) constraint matrix
            d: (q,) constraint vector
            
        Returns:
            dict with 'f_stat' and 'p_value'
        """
        if self.coef_ is None:
            raise ValueError("Model must be fitted before F-test")
        
        C = np.asarray(C)
        d = np.asarray(d).reshape(-1, 1)
        
        beta_hat = self.coef_.reshape(-1, 1)
        q = C.shape[0]  # Number of restrictions
        
        # Numerator: (C*beta_hat - d)^T [C*Cov(beta)*C^T]^{-1} (C*beta_hat - d)
        Cb_minus_d = C @ beta_hat - d  # (q, 1)
        C_cov_Ct = C @ self.cov_matrix_ @ C.T  # (q, q)
        
        try:
            C_cov_Ct_inv = np.linalg.inv(C_cov_Ct)
        except np.linalg.LinAlgError:
            raise ValueError("C*Cov(beta)*C^T is singular")
        
        numerator = (Cb_minus_d.T @ C_cov_Ct_inv @ Cb_minus_d).item()
        
        # F statistic
        f_stat = numerator / (q * self.sigma2_)
        
        # P-value from F-distribution with df1=q, df2=df_resid_
        p_value = 1 - stats.f.cdf(f_stat, dfn=q, dfd=self.df_resid_)
        
        return {
            'f_stat': f_stat,
            'p_value': p_value,
            'q': q,
            'df_resid': self.df_resid_
        }
