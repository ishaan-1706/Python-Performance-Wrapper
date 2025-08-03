# implementations/matmul_multiprocessed.py
# Synthetic data only; no real application dataset.

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from core.registry import register_impl
from core.types import ImplementationType

def _compute_row(args):
    i, row_a, B = args
    m = B.shape[1]
    k = row_a.shape[0]
    row_res = [0] * m
    for j in range(m):
        acc = 0
        for t in range(k):
            acc += row_a[t] * B[t, j]
        row_res[j] = acc
    return i, row_res

@register_impl("matmul", "multiprocessed", ImplementationType.MULTIPROCESSED)
def matmul_multiprocessed(a, b, max_workers=None):
    """Process‐pooled row‐wise multiplication, returns ndarray."""
    A = np.asarray(a)
    B = np.asarray(b)
    n, k = A.shape
    k2, m = B.shape
    if k != k2:
        raise ValueError(f"Incompatible dimensions: {A.shape} × {B.shape}")

    # prepare tasks
    tasks = [(i, A[i, :], B) for i in range(n)]
    result = [None] * n

    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        for i, row in exe.map(_compute_row, tasks):
            result[i] = row

    return np.array(result, dtype=np.result_type(A.dtype, B.dtype))