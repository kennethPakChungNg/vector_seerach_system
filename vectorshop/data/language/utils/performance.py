import time
from functools import wraps

def measure_performance(func):
    """Decorator to measure execution time of functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        print(f"Execution time for {func.__name__}: {execution_time*1000:.2f}ms")
        return result
    return wrapper