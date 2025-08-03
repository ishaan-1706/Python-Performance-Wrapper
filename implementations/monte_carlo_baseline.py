# implementations/monte_carlo_baseline.py
# Synthetic data only; no real application dataset.

import random
from typing import Optional
from core.registry import register_impl
from core.types import ImplementationType

@register_impl("monte_carlo", "baseline", ImplementationType.BASELINE)
def monte_carlo_baseline(
    n_samples: int,
    seed: Optional[int] = None
) -> float:
    """Loop-based Monte Carlo for π, with optional reproducible seed."""
    if seed is not None:
        random.seed(seed)

    inside = 0
    for _ in range(n_samples):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1

    return (inside / n_samples) * 4