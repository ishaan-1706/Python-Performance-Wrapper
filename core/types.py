# core/types.py

from enum import Enum, auto

class ImplementationType(Enum):
    BASELINE       = auto()  # Pure Python
    VECTORIZED     = auto()  # NumPy / vector libs
    MULTITHREADED  = auto()  # ThreadPoolExecutor
    MULTIPROCESSED = auto()  # ProcessPoolExecutor