import math
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from core.registry import register_impl
from core.types import ImplementationType
from implementations.gaussian_blur_baseline import _make_kernel

def _worker(args):
    i, padded, w, kernel_size, kernel = args
    row_out = [0.0] * w
    for j in range(w):
        s = 0.0
        for ki in range(kernel_size):
            for kj in range(kernel_size):
                s += kernel[ki][kj] * padded[i+ki][j+kj]
        row_out[j] = s
    return i, row_out

@register_impl("gaussian_blur", "multiprocessed", ImplementationType.MULTIPROCESSED)
def gaussian_blur_multiprocessed(image, kernel_size=5, sigma=1.0, max_workers=None):
    """Process-pooled convolution on row-chunks."""
    h, w = len(image), len(image[0])
    pad = kernel_size // 2
    kernel = _make_kernel(kernel_size, sigma)

    # pad image
    padded = [[0.0] * (w + 2*pad) for _ in range(h + 2*pad)]
    for i in range(h):
        for j in range(w):
            padded[i + pad][j + pad] = image[i][j]

    args = [(i, padded, w, kernel_size, kernel) for i in range(h)]
    out = [None] * h
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        for i, row in exe.map(_worker, args):
            out[i] = row

    # convert list-of-lists into ndarray
    return np.array(out, dtype=float)