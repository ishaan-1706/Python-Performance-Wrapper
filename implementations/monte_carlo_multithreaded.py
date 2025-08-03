# implementations/monte_carlo_multithreaded.py
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple
from core.registry import register_impl
from core.types import ImplementationType

def _thread_task(args: Tuple[int, Optional[int]]) -> int:
    n, seed = args
    rng = random.Random(seed)
    count = 0
    for _ in range(n):
        x = rng.uniform(-1, 1)
        y = rng.uniform(-1, 1)
        if x*x + y*y <= 1:
            count += 1
    return count

@register_impl("monte_carlo", "multithreaded", ImplementationType.MULTITHREADED)
def monte_carlo_multithreaded(
    n_samples: int,
    seed: Optional[int] = None,
    max_workers: Optional[int] = None
) -> float:
    """Thread-pooled Monte Carlo with optional reproducible seed."""
    workers = max_workers or 4
    chunk = n_samples // workers

    if seed is not None:
        master_rng = random.Random(seed)
        seeds = [master_rng.randrange(2**32) for _ in range(workers)]
    else:
        seeds = [None] * workers

    tasks = [(chunk, s) for s in seeds]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = executor.map(_thread_task, tasks)
    total_inside = sum(results)
    return (total_inside / (chunk * workers)) * 4