# core/decorator.py

import pkgutil
import importlib
from .registry import get_impls
from .selector import choose_impl

# one‐time import guard
_loaded = False

def _load_all_implementations():
    """
    Import every module under implementations/ so that
    @register_impl decorators fire and populate the registry.
    """
    global _loaded
    if _loaded:
        return

    import implementations  # your folder must be a package
    for _, module_name, _ in pkgutil.iter_modules(implementations.__path__):
        importlib.import_module(f"implementations.{module_name}")

    _loaded = True

def dynamic_exec(op_name: str):
    """
    Decorator for the abstract API function:

    - On first call: loads all impls, asks choose_impl(op_name, args, kwargs)
      to benchmark & pick the fastest, then caches that choice.
    - On every call: dispatches straight to the cached impl.
    """
    def decorator(fn):
        def wrapped(*args, **kwargs):
            # ensure all implementations have been registered
            _load_all_implementations()

            # sanity check
            if not get_impls(op_name):
                raise RuntimeError(f"No implementations registered for '{op_name}'")

            # select (and cache) the best impl; returns (name, func)
            impl_name, impl_func = choose_impl(op_name, args, kwargs)

            # dispatch to the chosen implementation
            return impl_func(*args, **kwargs)

        # preserve metadata
        wrapped.__name__ = fn.__name__
        wrapped.__doc__  = fn.__doc__
        return wrapped

    return decorator