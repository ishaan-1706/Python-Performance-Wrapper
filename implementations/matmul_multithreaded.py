# implementations/matmul_multithreaded.py
# Synthetic data only; no real application dataset.

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from core.registry import register_impl
from core.types import ImplementationType

@register_impl("matmul", "multithreaded", ImplementationType.MULTITHREADED)
def matmul_multithreaded(a, b, max_workers=None):
    """Thread‐pooled row‐wise multiplication, returns ndarray."""
    A = np.asarray(a)
    B = np.asarray(b)
    n, k = A.shape
    k2, m = B.shape
    if k != k2:
        raise ValueError(f"Incompatible dimensions: {A.shape} × {B.shape}")

    result = [[0] * m for _ in range(n)]

    def compute_row(i):
        row = [0] * m
        for j in range(m):
            acc = 0
            for t in range(k):
                acc += A[i, t] * B[t, j]
            row[j] = acc
        result[i] = row

    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        exe.map(compute_row, range(n))

    return np.array(result, dtype=np.result_type(A.dtype, B.dtype))