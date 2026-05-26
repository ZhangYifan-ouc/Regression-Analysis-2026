import numpy as np

def generate_data(n_samples=1000, rho=0.0, random_state=42):
    np.random.seed(random_state)
    z1 = np.random.randn(n_samples)
    z2 = np.random.randn(n_samples)
    
    X1 = z1
    X2 = rho * z1 + np.sqrt(1 - rho**2) * z2
    
    X = np.column_stack([np.ones(n_samples), X1, X2])
    return X