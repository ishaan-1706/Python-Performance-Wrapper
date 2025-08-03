#scripts/demo.py

import os
import sys

# 1. Ensure project root is on the import path
project_root = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.insert(0, project_root)

# 2. Trigger import of all implementations so registry gets populated
from core.decorator import _load_all_implementations
_load_all_implementations()

import argparse
import numpy as np
from time import perf_counter

from core.registry import get_impls
from core.types import ImplementationType
from core.api import matmul, monte_carlo, gaussian_blur


def parse_args():
    parser = argparse.ArgumentParser(
        description="Demo: compare baseline vs. dynamic-exec performance"
    )
    parser.add_argument(
        "op",
        choices=["matmul", "monte_carlo", "gaussian_blur"],
        help="Which operation to run"
    )
    parser.add_argument(
        "--matmul-shape",
        nargs=3, type=int, metavar=("M", "K", "N"),
        default=[200, 200, 200],
        help="Sizes for matmul: M K N"
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=1_000_000,
        help="Number of Monte Carlo samples"
    )
    parser.add_argument(
        "--blur-size",
        nargs=2, type=int, metavar=("H", "W"),
        default=[512, 512],
        help="Image size for gaussian_blur"
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Gaussian blur sigma/radius"
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


def get_baseline_func(op_name):
    """Find the pure-Python impl in the registry."""
    for name, func, impl_type in get_impls(op_name):
        if impl_type == ImplementationType.BASELINE:
            return name, func
    raise RuntimeError(f"No baseline implementation for '{op_name}'")


def benchmark(func, args, reps, warmups):
    # warm-up
    for _ in range(warmups):
        func(*args)

    # timed runs
    times = []
    for _ in range(reps):
        t0 = perf_counter()
        func(*args)
        times.append(perf_counter() - t0)

    return sum(times) / len(times)


def main():
    args = parse_args()

    # prepare inputs based on chosen operation
    if args.op == "matmul":
        M, K, N = args.matmul_shape
        A = np.random.rand(M, K)
        B = np.random.rand(K, N)
        call_args = (A, B)

    elif args.op == "monte_carlo":
        N = args.mc_samples
        call_args = (N,)

    else:  # gaussian_blur
        H, W = args.blur_size
        img = np.random.rand(H, W)
        call_args = (img, args.sigma)

    # 1. baseline
    base_name, base_func = get_baseline_func(args.op)
    t_base = benchmark(base_func, call_args, args.reps, args.warmups)

    # 2. dynamic first call (selection + cache)
    dyn_func = {
        "matmul": matmul,
        "monte_carlo": monte_carlo,
        "gaussian_blur": gaussian_blur
    }[args.op]
    t_dyn_first = benchmark(dyn_func, call_args, args.reps, args.warmups)

    # 3. dynamic cached call (should be fastest)
    t_dyn_cached = benchmark(dyn_func, call_args, args.reps, args.warmups)

    # report
    print(f"\nOperation: {args.op}")
    print(f"Baseline ({base_name}):        {t_base:.6f} s")
    print(f"Dynamic 1st call (select):    {t_dyn_first:.6f} s")
    print(f"Dynamic cached call:          {t_dyn_cached:.6f} s")
    print(f"Speedup (baseline→cached):    {t_base/t_dyn_cached:.2f}×\n")


if __name__ == "__main__":
    main()