import numpy as np
from core.registry import register_impl
from core.types import ImplementationType

def _make_kernel_np(k, sigma):
    ax = np.arange(-k//2 + 1., k//2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2)/(2. * sigma**2))
    return kernel / np.sum(kernel)

@register_impl("gaussian_blur", "vectorized", ImplementationType.VECTORIZED)
def gaussian_blur_vectorized(image, kernel_size=5, sigma=1.0):
    """NumPy padded convolution."""
    img = np.asarray(image, dtype=float)
    kernel = _make_kernel_np(kernel_size, sigma)
    pad = kernel_size // 2
    padded = np.pad(img, pad, mode='constant')
    h, w = img.shape
    out = np.zeros_like(img)

    for i in range(h):
        for j in range(w):
            window = padded[i:i+kernel_size, j:j+kernel_size]
            out[i, j] = np.sum(window * kernel)

    # already ndarray, but enforce dtype
    return out.astype(float)