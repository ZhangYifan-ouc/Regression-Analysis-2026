import time


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name: str) -> str:
    """
    Universal model evaluator using duck typing.
    Works with any model that has .fit(), .predict(), and .score() methods.
    
    Args:
        model: Any model instance with standard sklearn-like API
        X_train, y_train: Training data
        X_test, y_test: Test data
        model_name: Name for display in results
        
    Returns:
        Formatted string row for results table
    """
    start_time = time.perf_counter()
    
    # 1. Train the model
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - start_time
    
    # 2. Evaluate on test set
    r2_score = model.score(X_test, y_test)
    
    # 3. Format result as markdown table row
    result_str = f"| {model_name} | {fit_time:.5f} sec | {r2_score:.4f} |\n"
    
    return result_str
