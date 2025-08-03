# implementations/matmul_vectorized.py
# Synthetic data only; no real application dataset.

import numpy as np
from core.registry import register_impl
from core.types import ImplementationType

@register_impl("matmul", "vectorized", ImplementationType.VECTORIZED)
def matmul_vectorized(a, b):
    """NumPy dot‐product, returns ndarray."""
    A = np.asarray(a)
    B = np.asarray(b)
    # let NumPy handle shape checks and dtype promotion
    return np.dot(A, B)