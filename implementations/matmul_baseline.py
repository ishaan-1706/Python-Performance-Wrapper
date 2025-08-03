# implementations/matmul_baseline.py
# Synthetic data only; no real application dataset.

import numpy as np
from core.registry import register_impl
from core.types import ImplementationType

@register_impl("matmul", "baseline", ImplementationType.BASELINE)
def matmul_baseline(a, b):
    """Pure‐Python nested‐loops matrix multiplication, returns ndarray."""
    # allow Python lists or numpy arrays
    A = np.asarray(a)
    B = np.asarray(b)
    n, k = A.shape
    k2, m = B.shape
    if k != k2:
        raise ValueError(f"Incompatible dimensions: {A.shape} × {B.shape}")

    # compute via lists
    result = [[0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            acc = 0
            for t in range(k):
                acc += A[i, t] * B[t, j]
            result[i][j] = acc

    # convert to ndarray with same dtype as A @ B
    return np.array(result, dtype=np.result_type(A.dtype, B.dtype))