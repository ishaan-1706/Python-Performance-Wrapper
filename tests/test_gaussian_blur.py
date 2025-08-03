import numpy as np
from core.api import gaussian_blur

def test_gaussian_blur_shape():
    img = np.random.rand(10, 15)
    out = gaussian_blur(img, sigma=1.0)
    assert isinstance(out, np.ndarray)
    assert out.shape == img.shape

def test_gaussian_blur_energy_conserved():
    # Sum of pixels should stay roughly the same for normalized kernel
    img = np.zeros((5,5))
    img[2,2] = 1.0
    out = gaussian_blur(img, sigma=1.0)
    # kernel sums to 1, so total sum remains ~1
    assert np.isclose(out.sum(), 1.0, atol=1e-6)