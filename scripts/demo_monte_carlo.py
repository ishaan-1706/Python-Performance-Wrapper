#scripts/demo_monte_carlo.py

import os
import sys

# 1. Ensure project root is on the import path
project_root = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.insert(0, project_root)

# 2. Force-load all implementations so registry is populated
from core.decorator import _load_all_implementations
_load_all_implementations()

import argparse
import numpy as np
from time import perf_counter

from core.registry import get_impls
from core.types import ImplementationType
from core.api import monte_carlo


def parse_args():
    parser = argparse.ArgumentParser(
        description="Demo: compare baseline vs. dynamic-exec monte_carlo"
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=1_000_000,
        help="Number of Monte Carlo samples"
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=1,
        help="Warm-up runs to ignore"
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=5,
        help="Measured runs per benchmark"
    )
    return parser.parse_args()


def get_baseline_func():
    for name, func, impl_type in get_impls("monte_carlo"):
        if impl_type == ImplementationType.BASELINE:
            return name, func
    raise RuntimeError("No baseline implementation for 'monte_carlo'")


def benchmark(func, args, reps, warmups):
    for _ in range(warmups):
        func(*args)

    times = []
    for _ in range(reps):
        t0 = perf_counter()
        func(*args)
        times.append(perf_counter() - t0)

    return sum(times) / len(times)


def main():
    args = parse_args()
    call_args = (args.mc_samples,)

    # baseline
    base_name, base_func = get_baseline_func()
    t_base = benchmark(base_func, call_args, args.reps, args.warmups)

    # dynamic
    t_dyn_first = benchmark(monte_carlo, call_args, args.reps, args.warmups)
    t_dyn_cached = benchmark(monte_carlo, call_args, args.reps, args.warmups)

    print("\nOperation: monte_carlo")
    print(f"Baseline ({base_name}):        {t_base:.6f} s")
    print(f"Dynamic 1st call (select):    {t_dyn_first:.6f} s")
    print(f"Dynamic cached call:          {t_dyn_cached:.6f} s")
    print(f"Speedup (baseline→cached):    {t_base/t_dyn_cached:.2f}×\n")


if __name__ == "__main__":
    main()