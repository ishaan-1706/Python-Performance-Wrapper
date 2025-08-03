# core/benchmarker.py

import time

class BenchmarkResult:
    def __init__(self, times):
        self.times = times
        self.avg_time = sum(times) / len(times)

def benchmark(func, args, kwargs, reps=5, warmups=1):
    """
    Run `func(*args, **kwargs)`:
      - warm up `warmups` times (ignore timings)
      - measure it `reps` times
    Returns BenchmarkResult with average duration.
    """
    for _ in range(warmups):
        func(*args, **kwargs)

    timings = []
    for _ in range(reps):
        start = time.perf_counter()
        func(*args, **kwargs)
        timings.append(time.perf_counter() - start)

    return BenchmarkResult(timings)