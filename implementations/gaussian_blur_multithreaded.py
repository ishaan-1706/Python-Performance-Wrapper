import math
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from core.registry import register_impl
from core.types import ImplementationType
from implementations.gaussian_blur_baseline import _make_kernel

@register_impl("gaussian_blur", "multithreaded", ImplementationType.MULTITHREADED)
def gaussian_blur_multithreaded(image, kernel_size=5, sigma=1.0, max_workers=None):
    """Threaded convolution on row-chunks."""
    h, w = len(image), len(image[0])
    pad = kernel_size // 2
    kernel = _make_kernel(kernel_size, sigma)

    # pad image
    padded = [[0.0] * (w + 2*pad) for _ in range(h + 2*pad)]
    for i in range(h):
        for j in range(w):
            padded[i + pad][j + pad] = image[i][j]

    out = [[0.0] * w for _ in range(h)]

    def process_row(i):
        for j in range(w):
            s = 0.0
            for ki in range(kernel_size):
                for kj in range(kernel_size):
                    s += kernel[ki][kj] * padded[i+ki][j+kj]
            out[i][j] = s

    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        list(exe.map(process_row, range(h)))

    # ensure numpy.ndarray return
    return np.array(out, dtype=float)