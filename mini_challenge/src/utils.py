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

