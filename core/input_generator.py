# core/input_generator.py

import numpy as np
import random

from typing import overload, Tuple, List, Literal, Union

@overload
def generate_input(op_name: Literal["matmul"]) -> Tuple[np.ndarray, np.ndarray]:
    ...

@overload
def generate_input(op_name: Literal["monte_carlo"]) -> int:
    ...

@overload
def generate_input(op_name: Literal["gaussian_blur"]) -> List[List[float]]:
    ...

def generate_input(
    op_name: str
) -> Union[Tuple[np.ndarray, np.ndarray], int, List[List[float]]]:
    """
    Produce sample inputs matching your baseline impl signatures:
      - matmul(a, b) → Tuple of two ndarrays
      - monte_carlo(n_samples) → int
      - gaussian_blur(image) → list of lists
    """
    if op_name == "matmul":
        A = np.random.rand(200, 300)
        B = np.random.rand(300, 150)
        return (A, B)

    elif op_name == "monte_carlo":
        return 100_000

    elif op_name == "gaussian_blur":
        image = np.random.rand(256, 256)
        return image.tolist()

    else:
        raise ValueError(f"Unknown operation for input generation: {op_name}")
