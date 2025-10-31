# Suggested by Grok 3

from concurrent.futures import ProcessPoolExecutor, TimeoutError
from multiprocessing import Process, Queue
from typing import Callable, Any, Tuple, Dict
import signal


def run_in_process(
    func: Callable, args: Tuple, kwargs: Dict, queue: Queue
) -> None:
    """
    Run the function in a separate process and put the result in a queue.
    """
    try:
        result = func(*args, **kwargs)
        queue.put(("success", result))
    except Exception as e:
        queue.put(("error", e))


def run_with_timeout(
    func: Callable, args: Tuple = (), kwargs: Dict = None, timeout: float = 2.0
) -> Any:
    """
    Run a function with a timeout, using ProcessPoolExecutor with forced termination.

    Args:
        func: Function to execute.
        args: Tuple of positional arguments.
        kwargs: Dictionary of keyword arguments.
        timeout: Timeout in seconds.

    Returns:
        The function's result if completed within the timeout.

    Raises:
        TimeoutError: If the function exceeds the timeout.
        Exception: If the function raises any other exception.
    """
    if kwargs is None:
        kwargs = {}

    # Create a queue to communicate results
    queue = Queue()

    # Start the function in a separate process
    process = Process(target=run_in_process, args=(func, args, kwargs, queue))
    process.start()

    try:
        # Wait for the process to complete or timeout
        process.join(timeout)

        if process.is_alive():
            # If still running, terminate the process
            process.terminate()
            process.join()  # Ensure process is cleaned up
            raise TimeoutError(
                f"Function {func.__name__} timed out after {timeout} seconds"
            )

        # Check if the process exited with an error
        if process.exitcode != 0:
            raise RuntimeError(
                f"Process failed with exit code {process.exitcode}"
            )

        # Retrieve result from queue
        if not queue.empty():
            status, result = queue.get()
            if status == "success":
                return result
            else:
                raise result
        else:
            raise RuntimeError("No result returned from process")

    finally:
        # Ensure process is terminated and cleaned up
        if process.is_alive():
            process.terminate()
            process.join()
        process.close()


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
