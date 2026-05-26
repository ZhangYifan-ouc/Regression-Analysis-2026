import numpy as np
from data_generator import generate_data
from sklearn.linear_model import LinearRegression

def run_simulation(X, true_beta, sigma, n_simulations=1000, random_state=42):
    np.random.seed(random_state)
    n_samples = X.shape[0]
    beta_hat_list = []
    
    for _ in range(n_simulations):
        epsilon = np.random.normal(0, sigma, size=n_samples)
        y = X @ true_beta + epsilon
        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)
        beta_hat = model.coef_
        beta_hat_list.append(beta_hat)
    
    return np.array(beta_hat_list)

if __name__ == "__main__":
    n_samples = 1000
    true_beta = np.array([0.0, 5.0, 3.0])
    sigma = 2.0
    n_simulations = 1000

    print("=== 实验A：正交特征 rho=0.0 ===")
    X_A = generate_data(n_samples=n_samples, rho=0.0)
    beta_hat_A = run_simulation(X_A, true_beta, sigma, n_simulations)
    beta_hat_A = beta_hat_A[:, 1:]

    print("=== 实验B：高度共线性 rho=0.99 ===")
    X_B = generate_data(n_samples=n_samples, rho=0.99)
    beta_hat_B = run_simulation(X_B, true_beta, sigma, n_simulations)
    beta_hat_B = beta_hat_B[:, 1:]

    np.savez("simulation_results.npz",
             beta_hat_A=beta_hat_A,
             beta_hat_B=beta_hat_B,
             X_A=X_A,
             X_B=X_B,
             sigma=sigma)
    print("模拟完成，结果已保存！")