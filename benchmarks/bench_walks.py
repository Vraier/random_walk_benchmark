import itertools
import os
from dataclasses import dataclass

import numpy as np
import pytest

import methods.cpp_method       # noqa: F401  # type: ignore[import]
import methods.cpython_method   # noqa: F401  # type: ignore[import]
import methods.cython_method    # noqa: F401  # type: ignore[import]
import methods.igraph_method    # noqa: F401  # type: ignore[import]
import methods.numba_method     # noqa: F401  # type: ignore[import]
import methods.pybind11_method  # noqa: F401  # type: ignore[import]
import methods.pyg_method       # noqa: F401  # type: ignore[import]
# import methods.numpy_method   # noqa: F401 - slow, enable explicitly
import metrics.memory
from graphs.generators import erdos_renyi
from methods import get_registry
from metrics import get_metrics

# import metrics.coverage      # uncomment to enable coverage metrics

# --- Sweep Parameters ---
NODE_COUNTS = [2000, 6000, 10000, 14000, 18000]
WALK_LENGTHS = [20, 50]
NUM_WALKS_PER_NODE = [5, 10]
AVG_DEGREES = [15]

_N_CORES = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()


@dataclass
class BenchConfig:
    n: int
    walk_length: int
    num_walks_per_node: int
    avg_degree: int


_ALL_CONFIGS = [
    BenchConfig(n, wl, nw, deg)
    for n, wl, nw, deg in itertools.product(
        NODE_COUNTS, WALK_LENGTHS, NUM_WALKS_PER_NODE, AVG_DEGREES
    )
]


def _config_id(c: BenchConfig) -> str:
    return f"n{c.n}-wl{c.walk_length}-nw{c.num_walks_per_node}-deg{c.avg_degree}"


def _graph_to_csr(G):
    import networkx as nx
    n = G.number_of_nodes()
    A = nx.to_scipy_sparse_array(G, nodelist=range(n), format="csr")
    return np.array(A.indptr, dtype=np.int64), np.array(A.indices, dtype=np.int64)


# --- Build test cases: (method_name, cfg, num_threads) ---
# supports_parallel methods get both num_threads=1 and num_threads=N_CORES.
# Others always get num_threads=1.

_registry = get_registry()


def _build_test_cases():
    cases = []
    for method_name, info in _registry.items():
        threads = [1, _N_CORES] if info.supports_parallel else [1]
        for cfg in _ALL_CONFIGS:
            for t in threads:
                cases.append((method_name, cfg, t))
    return cases


def _case_id(case):
    method_name, cfg, num_threads = case
    t = f"t{num_threads}"
    return f"{_config_id(cfg)}-{method_name}-{t}"


_TEST_CASES = _build_test_cases()


# --- Benchmark ---

@pytest.mark.parametrize("case", _TEST_CASES, ids=_case_id)
def test_walk(benchmark, case):
    method_name, cfg, num_threads = case
    fn = _registry[method_name].fn

    G = erdos_renyi(cfg.n, cfg.avg_degree)
    rowptr, col = _graph_to_csr(G)
    start_nodes = np.repeat(np.arange(cfg.n, dtype=np.int64), cfg.num_walks_per_node)

    # Single run: memory measurement + result metrics
    result, mem_delta_mb = metrics.memory.measure(
        fn, rowptr, col, start_nodes, cfg.walk_length, True, num_threads
    )
    benchmark.extra_info["memory_rss_delta_mb"] = mem_delta_mb
    benchmark.extra_info["num_threads"] = num_threads

    assert result.shape == (len(start_nodes), cfg.walk_length)
    assert result.dtype == np.int64

    for metric_name, metric_fn in get_metrics().items():
        benchmark.extra_info[metric_name] = metric_fn(result)

    # Runtime benchmark (warmup handled by pytest-benchmark)
    benchmark(fn, rowptr, col, start_nodes, cfg.walk_length, True, num_threads)
