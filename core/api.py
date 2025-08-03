# core/api.py

import numpy as np
from typing import Optional
from .decorator import dynamic_exec

@dynamic_exec("matmul")
def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Multiply two 2D arrays a and b, returning their matrix product.
    The decorator will dispatch to the best of:
      - baseline (pure Python)
      - vectorized (NumPy)
      - multithreaded
      - multiprocessed
    """
    raise NotImplementedError("Should be overridden by the dynamic selector")


@dynamic_exec("monte_carlo")
def monte_carlo(num_samples: int, seed: Optional[int] = None) -> float:
    """
    Estimate π by sampling `num_samples` points in [-1,1]².
    The decorator will dispatch to the best of:
      - baseline
      - vectorized
      - multithreaded
      - multiprocessed
    """
    raise NotImplementedError("Should be overridden by the dynamic selector")


@dynamic_exec("gaussian_blur")
def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply a Gaussian blur to a 2D image array.
    The decorator will dispatch to the best of:
      - baseline
      - vectorized
      - multithreaded
      - multiprocessed
    """
    raise NotImplementedError("Should be overridden by the dynamic selector")