from collections import defaultdict
from .types import ImplementationType

# Still initialize as defaultdict for normal usage
_registry = defaultdict(list)

def register_impl(op_name: str, impl_name: str, impl_type: ImplementationType):
    """
    Decorator to register an implementation variant for op_name.
    Uses setdefault() so we never KeyError, even if tests replace _registry with a plain dict.
    """
    def decorator(func):
        # Ensure there's a list at _registry[op_name]
        _registry.setdefault(op_name, []).append((impl_name, func, impl_type))
        return func
    return decorator

def get_impls(op_name: str):
    """
    Return list of (impl_name, func, impl_type) for the given op_name.
    """
    return _registry.get(op_name, [])

def list_ops():
    """
    Return all operation names that have at least one registered implementation.
    """
    return list(_registry.keys())
