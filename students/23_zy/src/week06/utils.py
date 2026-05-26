import numpy as np
import time
from pathlib import Path
import shutil


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name: str) -> dict:
    """通用评价函数（支持 CustomOLS 和 sklearn 模型）"""
    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - start_time
    y_pred = model.predict(X_test)
    
    # 计算 R² 和 RMSE
    r2_score = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

    # 返回评估结果
    return {"r2": r2_score, "rmse": rmse}

def setup_results_dir() -> Path:
    """自动化管理 results 文件夹（存在则清空，不存在则创建）"""
    results_dir = Path(__file__).parent.parent.parent / "results"
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def generate_synthetic_data(n_samples: int = 1000, seed: int = 42) -> tuple:
    """生成合成数据（用于场景A的白盒测试）"""
    np.random.seed(seed)
    X = np.random.randn(n_samples, 2)
    beta_true = np.array([5.0, 3.0])
    noise = np.random.randn(n_samples) * 2.0
    y = X @ beta_true + noise
    return X, y