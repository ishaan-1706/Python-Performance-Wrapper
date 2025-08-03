# core/cache.py

import json
import os

CACHE_FILE = os.path.join(os.path.dirname(__file__), "..", "exec_cache.json")

def _load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

def lookup_cached(key):
    """
    key should be JSON‐serializable (e.g., tuple).
    Returns (impl_name, func_name) or None.
    """
    return _load_cache().get(str(key))

def cache_result(key, value):
    """
    value is (impl_name, func_reference_path). We store impl_name only,
    since func lookup happens via registry.
    """
    data = _load_cache()
    data[str(key)] = value[0]
    with open(CACHE_FILE, "w") as f:
        json.dump(data, f, indent=2)