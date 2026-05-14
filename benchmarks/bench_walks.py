from dataclasses import dataclass

import numpy as np
import pytest

import methods.cpp_method       # noqa: F401  # type: ignore[import]
import methods.python_method    # noqa: F401  # type: ignore[import]
import methods.cpython_method   # noqa: F401  # type: ignore[import]
import methods.cython_method    # noqa: F401  # type: ignore[import]
import methods.igraph_method    # noqa: F401  # type: ignore[import]
import methods.numba_method     # noqa: F401  # type: ignore[import]
import methods.pybind11_method  # noqa: F401  # type: ignore[import]
import methods.pyg_method       # noqa: F401  # type: ignore[import]
# import methods.numpy_method   # noqa: F401 - slow, enable explicitly
import metrics.coverage  # noqa: F401 — registers coverage_fraction metric
from graphs.generators import erdos_renyi
from methods import get_registry
from metrics import get_metrics

# --- Base Configuration ---
BASE_N               = 10000
BASE_WALK_LENGTH     = 20
BASE_NUM_WALKS       = 20
BASE_AVG_DEGREE      = 15

# --- Sweep Values ---
SCALE_N_VALUES          = [2000, 6000, 10000, 14000, 18000]
SCALE_WALK_LENGTH_VALUES = [10, 20, 50, 100, 200]
SCALE_NUM_WALKS_VALUES  = [1, 5, 10, 20, 50]
SCALE_THREADS_VALUES    = [1, 2, 4, 8]

@dataclass
class BenchConfig:
    n: int
    walk_length: int
    num_walks_per_node: int
    avg_degree: int


def _cfg(**kwargs) -> BenchConfig:
    defaults = dict(
        n=BASE_N,
        walk_length=BASE_WALK_LENGTH,
        num_walks_per_node=BASE_NUM_WALKS,
        avg_degree=BASE_AVG_DEGREE,
    )
    return BenchConfig(**{**defaults, **kwargs})


def _config_id(c: BenchConfig) -> str:
    return f"n{c.n}-wl{c.walk_length}-nw{c.num_walks_per_node}-deg{c.avg_degree}"


def _graph_to_csr(G):
    import networkx as nx
    n = G.number_of_nodes()
    A = nx.to_scipy_sparse_array(G, nodelist=range(n), format="csr")
    return np.array(A.indptr, dtype=np.int64), np.array(A.indices, dtype=np.int64)


_registry = get_registry()


def _build_test_cases():
    cases = []

    # scale_n: all methods, t=1, allow_backtrack=True, vary n
    for n in SCALE_N_VALUES:
        for method_name in _registry:
            cases.append((method_name, _cfg(n=n), 1, True))

    # scale_walk_length: all methods, t=1, allow_backtrack=True, vary walk_length
    for wl in SCALE_WALK_LENGTH_VALUES:
        for method_name in _registry:
            cases.append((method_name, _cfg(walk_length=wl), 1, True))

    # scale_num_walks: all methods, t=1, allow_backtrack=True, vary num_walks_per_node
    for nw in SCALE_NUM_WALKS_VALUES:
        for method_name in _registry:
            cases.append((method_name, _cfg(num_walks_per_node=nw), 1, True))

    # scale_threads: only parallel methods, vary threads, fixed base config
    for method_name, info in _registry.items():
        if info.supports_parallel:
            for t in SCALE_THREADS_VALUES:
                cases.append((method_name, _cfg(), t, True))

    # backtrack: only no-backtrack methods, vary n, both allow_backtrack values
    for n in SCALE_N_VALUES:
        for method_name, info in _registry.items():
            if info.supports_no_backtrack:
                cases.append((method_name, _cfg(n=n), 1, True))
                cases.append((method_name, _cfg(n=n), 1, False))

    # Deduplicate by cache key (same params may appear in multiple experiments)
    seen = set()
    unique = []
    for case in cases:
        key = _case_id(case)
        if key not in seen:
            seen.add(key)
            unique.append(case)
    return unique


def _case_id(case):
    method_name, cfg, num_threads, allow_backtrack = case
    bt = "" if allow_backtrack else "-nobt"
    return f"{_config_id(cfg)}-{method_name}-t{num_threads}{bt}"


_TEST_CASES = _build_test_cases()


# --- Benchmark ---

@pytest.mark.parametrize("case", _TEST_CASES, ids=_case_id)
def test_walk(benchmark, case):
    method_name, cfg, num_threads, allow_backtrack = case
    fn = _registry[method_name].fn

    G = erdos_renyi(cfg.n, cfg.avg_degree)
    rowptr, col = _graph_to_csr(G)
    start_nodes = np.repeat(np.arange(cfg.n, dtype=np.int64), cfg.num_walks_per_node)

    result = fn(rowptr, col, start_nodes, cfg.walk_length, allow_backtrack, num_threads)
    benchmark.extra_info["num_threads"] = num_threads
    benchmark.extra_info["allow_backtrack"] = allow_backtrack

    assert result.shape == (len(start_nodes), cfg.walk_length)
    assert result.dtype == np.int64

    for metric_name, metric_fn in get_metrics().items():
        benchmark.extra_info[metric_name] = metric_fn(result)

    # Runtime benchmark (warmup handled by pytest-benchmark)
    benchmark(fn, rowptr, col, start_nodes, cfg.walk_length, allow_backtrack, num_threads)
