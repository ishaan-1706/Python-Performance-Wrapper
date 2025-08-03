# implementations/monte_carlo_vectorized.py
import numpy as np
from typing import Optional
from core.registry import register_impl
from core.types import ImplementationType

@register_impl("monte_carlo", "vectorized", ImplementationType.VECTORIZED)
def monte_carlo_vectorized(
    n_samples: int,
    seed: Optional[int] = None
) -> float:
    """NumPy vectorized Monte Carlo with optional reproducible seed."""
    if seed is not None:
        np.random.seed(seed)

    xs = np.random.uniform(-1, 1, size=n_samples)
    ys = np.random.uniform(-1, 1, size=n_samples)
    inside = np.count_nonzero(xs*xs + ys*ys <= 1)
    return (inside / n_samples) * 4