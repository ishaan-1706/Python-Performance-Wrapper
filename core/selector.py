# core/selector.py

from .registry import get_impls
from .benchmarker import benchmark
from .cache import lookup_cached, cache_result

def _signature(args, kwargs):
    """
    Cre­ates a simple key based on shapes or lengths of args.
    Synthetic data only; no real-world records.
    """
    sig_args = []
    for arg in args:
        if hasattr(arg, "shape"):
            sig_args.append(tuple(arg.shape))
        elif hasattr(arg, "__len__"):
            sig_args.append(len(arg))
        else:
            sig_args.append(None)
    sig_kwargs = tuple(sorted(kwargs.items()))
    return (tuple(sig_args), sig_kwargs)

def choose_impl(op_name, args, kwargs):
    """
    Returns (impl_name, impl_func) for the fastest variant.
    Caches choice after the first benchmark.
    """
    key = (op_name, _signature(args, kwargs))
    cached_name = lookup_cached(key)
    if cached_name:
        # find function by name
        for name, func, _ in get_impls(op_name):
            if name == cached_name:
                return name, func

    # benchmark all registered variants
    candidates = get_impls(op_name)
    results = []
    for name, func, _ in candidates:
        perf = benchmark(func, args, kwargs)
        results.append((perf.avg_time, name, func))

    best_time, best_name, best_func = min(results, key=lambda x: x[0])
    # only store impl_name in JSON cache; function is looked up by name next time
    cache_result(key, (best_name, None))
    return best_name, best_func