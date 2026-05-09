from dataclasses import dataclass
from typing import Callable
import numpy as np

@dataclass
class MethodInfo:
    name: str
    fn: Callable
    supports_no_backtrack: bool

_registry: dict[str, MethodInfo] = {}

def register(name: str, supports_no_backtrack: bool = False):
    """Decorator to register a random walk method.

    The decorated function must have the signature:
        fn(rowptr: np.ndarray, col: np.ndarray, start_nodes: np.ndarray,
           walk_length: int, allow_backtrack: bool) -> np.ndarray
    Returns np.ndarray of shape (num_walks, walk_length) dtype int64.
    Methods that don't support no-backtrack may ignore allow_backtrack.
    """
    def decorator(fn: Callable) -> Callable:
        _registry[name] = MethodInfo(name=name, fn=fn, supports_no_backtrack=supports_no_backtrack)
        return fn
    return decorator

def get_registry() -> dict[str, MethodInfo]:
    return _registry
