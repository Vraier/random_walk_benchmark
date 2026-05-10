from dataclasses import dataclass
from typing import Callable

@dataclass
class MethodInfo:
    name: str
    fn: Callable
    supports_no_backtrack: bool
    supports_parallel: bool  # if True, benchmarked with num_threads=1 and num_threads=N_CORES

_registry: dict[str, MethodInfo] = {}

def register(name: str, supports_no_backtrack: bool = False, supports_parallel: bool = False):
    """Decorator to register a random walk method.

    Uniform signature:
        fn(rowptr, col, start_nodes, walk_length, allow_backtrack, num_threads) -> np.ndarray
        shape (num_walks, walk_length), dtype int64

    Methods that don't support no-backtrack or parallel may ignore those args.
    """
    def decorator(fn: Callable) -> Callable:
        _registry[name] = MethodInfo(
            name=name, fn=fn,
            supports_no_backtrack=supports_no_backtrack,
            supports_parallel=supports_parallel,
        )
        return fn
    return decorator

def get_registry() -> dict[str, MethodInfo]:
    return _registry
