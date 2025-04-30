# Suggested by Grok 3

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Callable, Any, Tuple, Dict

def run_with_timeout(
    func: Callable,
    args: Tuple = (),
    kwargs: Dict = None,
    timeout: float = 2.0
) -> Any:
    """
    Run a function with a timeout, supporting both positional and keyword arguments.
    
    Args:
        func: Function to execute.
        args: Tuple of positional arguments to pass to the function.
        kwargs: Dictionary of keyword arguments to pass to the function.
        timeout: Timeout in seconds.
    
    Returns:
        The function's result if completed within the timeout.
    
    Raises:
        TimeoutError: If the function exceeds the timeout.
        Exception: If the function raises any other exception.
    """
    if kwargs is None:
        kwargs = {}
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            raise TimeoutError(f"Function {func.__name__} timed out after {timeout} seconds")
        except Exception as e:
            raise e


# Example usage
def long_running_function(seconds: int) -> str:
    import time

    time.sleep(seconds)
    return f"Slept for {seconds} seconds"


if __name__ == "__main__":
    # Run multiple tasks with timeouts
    tasks = [(long_running_function, (3,)), (long_running_function, (1,))]
    for func, args in tasks:
        try:
            result = run_with_timeout(func, args, timeout=2)
            print(f"Result: {result}")
        except TimeoutError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
