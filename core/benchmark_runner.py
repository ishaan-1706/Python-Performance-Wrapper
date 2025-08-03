import os
import json
import numpy as np

from core.registry import get_impls, list_ops
from core.benchmarker import benchmark
from core.input_generator import generate_input

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_all_benchmarks(reps=10, warmups=2):
    all_results = {}

    for op_name in list_ops():
        print(f"\n🔬 Benchmarking operation: {op_name}")
        sample = generate_input(op_name)

        op_results = {}
        for impl_name, func, _impl_type in get_impls(op_name):
            print(f"  ▶ {impl_name}")

            # unpack for matmul, else wrap in a tuple
            if op_name == "matmul":
                args = sample
            else:
                args = (sample,)

            result = benchmark(func, args=args, kwargs={}, reps=reps, warmups=warmups)

            times = result.times
            mean_time = float(np.mean(times))
            std_dev   = float(np.std(times))

            op_results[impl_name] = {
                "mean_time": mean_time,
                "std_dev":   std_dev,
                "times":     times
            }

        all_results[op_name] = op_results

    out_path = os.path.join(RESULTS_DIR, "benchmark.json")
    with open(out_path, "w") as fp:
        json.dump(all_results, fp, indent=2)
    print(f"\n✅ Results written to {out_path}")

if __name__ == "__main__":
    run_all_benchmarks()
