from typing import Callable
import numpy as np

# Result metrics: fn(result) -> float
# result shape: (num_walks, walk_length), dtype int64
_metrics: dict[str, Callable[[np.ndarray], float]] = {}


def register(name: str):
    """Decorator to register a result metric.

    The decorated function must have the signature:
        fn(result: np.ndarray) -> float
    where result is the walk matrix of shape (num_walks, walk_length).
    """
    def decorator(fn: Callable) -> Callable:
        _metrics[name] = fn
        return fn
    return decorator


def get_metrics() -> dict[str, Callable[[np.ndarray], float]]:
    return _metrics
