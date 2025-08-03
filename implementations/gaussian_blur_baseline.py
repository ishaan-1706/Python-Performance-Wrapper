import math
import numpy as np
from core.registry import register_impl
from core.types import ImplementationType

def _make_kernel(k, sigma):
    center = k // 2
    kernel = [[0.0] * k for _ in range(k)]
    norm = 0.0
    for i in range(k):
        for j in range(k):
            dx, dy = i - center, j - center
            val = math.exp(-(dx*dx + dy*dy) / (2 * sigma * sigma))
            kernel[i][j] = val
            norm += val
    # normalize
    for i in range(k):
        for j in range(k):
            kernel[i][j] /= norm
    return kernel

@register_impl("gaussian_blur", "baseline", ImplementationType.BASELINE)
def gaussian_blur_baseline(image, kernel_size=5, sigma=1.0):
    """Nested-loops convolution."""
    h, w = len(image), len(image[0])
    pad = kernel_size // 2
    kernel = _make_kernel(kernel_size, sigma)

    # pad image
    padded = [[0.0] * (w + 2*pad) for _ in range(h + 2*pad)]
    for i in range(h):
        for j in range(w):
            padded[i + pad][j + pad] = image[i][j]

    # convolve
    out = [[0.0] * w for _ in range(h)]
    for i in range(h):
        for j in range(w):
            s = 0.0
            for ki in range(kernel_size):
                for kj in range(kernel_size):
                    s += kernel[ki][kj] * padded[i + ki][j + kj]
            out[i][j] = s

    # ensure numpy.ndarray return
    return np.array(out, dtype=float)