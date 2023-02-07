import re
import traceback
from functools import wraps
from time import perf_counter
from typing import *

import psutil
import torch


def print_ram_usage() -> None:
    print(
        f"RAM used: {psutil.virtual_memory()[3] / 10**9} GB "
        f"({psutil.virtual_memory()[2]} %)"
    )


def track_time_memory_usage(func):
    """
    Decorator that measures the execution time and the memory allocated (and not
    deallocated) during the execution of a function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Running function {func.__name__}.")
        start_time = perf_counter()
        start_ram = psutil.virtual_memory()[3] / (1024**3)
        start_pct_ram = psutil.virtual_memory()[2]
        print(f"RAM initially used: {start_ram:.4f} GB ({start_pct_ram:.2f} %)")

        result = func(*args, **kwargs)

        end_time = perf_counter()
        end_ram = psutil.virtual_memory()[3] / 10**9
        end_pct_ram = psutil.virtual_memory()[2]
        print(f"RAM used after execution: {end_ram:.4f} GB ({end_pct_ram:.2f} %)")

        print(
            f"Function {func.__name__} took {end_time - start_time:.2f} seconds"
            f" and increased memory use by {(end_ram - start_ram) * 1024:.4f} MB"
            f" ({end_pct_ram - start_pct_ram:.2f} %).\n"
        )
        return result

    return wrapper


def print_tensors_size_in_memory(*tensors: torch.Tensor) -> None:
    stack = traceback.extract_stack()
    code = stack[-2][-1]
    tensor_names = re.compile(r"\((.*?)\).*$").search(code).groups()[0].split(", ")

    for idx, tensor in enumerate(tensors):
        tensor_size = tensor.element_size() * tensor.nelement() / 1024**2
        print(f"Size in memory of tensor {tensor_names[idx]}: {tensor_size:.4f} MB")


def timeit(func: Callable) -> Callable:
    """
    Decorator for timing function execution time.

    :param func: The function to time.
    :return: The wrapped function.
    """

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = perf_counter()
        result = func(*args, **kwargs)
        end_time = perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__} took {total_time:.2f} seconds")
        return result

    return timeit_wrapper


def checkpoint(time_ref: float = perf_counter()) -> Callable[..., None]:
    """
    Closure that stores a time checkpoint that is updated at every call.
    Each call prints the time elapsed since the last checkpoint with a custom message.

    :param time_ref: The time reference to start from. By default, the time of the call will be taken.
    :return: The closure.
    """

    def _closure(message: str = "") -> None:
        """
        Prints the time elapsed since the previous call.

        :param message: Custom message to print. The overall result will be: 'message: time_elapsed'.
        """
        nonlocal time_ref
        current_time = perf_counter()
        if message != "":
            print(f"{message} {current_time - time_ref:.4f} s.")
        time_ref = current_time

    return _closure
