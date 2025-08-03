# scripts/demo_gaussian_blur.py

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
from core.api import gaussian_blur


def parse_args():
    p = argparse.ArgumentParser(
        description="Demo: compare baseline vs. dynamic-exec gaussian_blur"
    )
    p.add_argument(
        "--blur-size", nargs=2, type=int, metavar=("H", "W"),
        default=[512, 512], help="Image size H W"
    )
    p.add_argument(
        "--sigma", type=float, default=1.0,
        help="Gaussian blur sigma/radius"
    )
    p.add_argument(
        "--warmups", type=int, default=1,
        help="Warm-up runs"
    )
    p.add_argument(
        "--reps", type=int, default=5,
        help="Measured runs"
    )
    return p.parse_args()


def get_baseline_func():
    for name, fn, t in get_impls("gaussian_blur"):
        if t == ImplementationType.BASELINE:
            return name, fn
    raise RuntimeError("No baseline for gaussian_blur")


def benchmark(fn, args, reps, warmups):
    for _ in range(warmups):
        fn(*args)
    times = []
    for _ in range(reps):
        t0 = perf_counter()
        fn(*args)
        times.append(perf_counter() - t0)
    return sum(times) / len(times)


def main():
    args = parse_args()
    H, W = args.blur_size
    img = np.random.rand(H, W)

    # derive an odd integer kernel_size from sigma
    kernel_size = 2 * int(3 * args.sigma) + 1

    # call BOTH baseline and dynamic with (img, kernel_size, sigma)
    call_args = (img, kernel_size, args.sigma)

    # baseline
    base_name, base_fn = get_baseline_func()
    t_base = benchmark(base_fn, call_args, args.reps, args.warmups)

    # dynamic
    t_dyn_first = benchmark(gaussian_blur, call_args, args.reps, args.warmups)
    t_dyn_cached = benchmark(gaussian_blur, call_args, args.reps, args.warmups)

    print("\nOperation: gaussian_blur")
    print(f"Baseline ({base_name}):        {t_base:.6f} s")
    print(f"Dynamic 1st call (select):    {t_dyn_first:.6f} s")
    print(f"Dynamic cached call:          {t_dyn_cached:.6f} s")
    print(f"Speedup (baseline→cached):    {t_base/t_dyn_cached:.2f}×\n")


if __name__ == "__main__":
    main()