import itertools
from dataclasses import dataclass

import numpy as np
import pytest

import methods.cpp_method  # noqa: F401

# import methods.numpy_method  # noqa: F401 - trigger registration
import methods.pyg_method  # noqa: F401
import metrics.memory
from graphs.generators import erdos_renyi
from methods import get_registry
from metrics import get_metrics

# import metrics.coverage      # uncomment to enable coverage metrics

# --- Sweep Parameters ---
NODE_COUNTS = [2000, 4000, 6000]
WALK_LENGTHS = [20, 50]
NUM_WALKS_PER_NODE = [5, 10]
AVG_DEGREES = [15]


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


# --- Fixtures ---


@pytest.fixture(params=_ALL_CONFIGS, ids=_config_id)
def bench_config(request):
    cfg = request.param
    G = erdos_renyi(cfg.n, cfg.avg_degree)
    rowptr, col = _graph_to_csr(G)
    start_nodes = np.repeat(np.arange(cfg.n, dtype=np.int64), cfg.num_walks_per_node)
    return cfg, rowptr, col, start_nodes


# --- Benchmark ---

_registry = get_registry()


@pytest.mark.parametrize("method_name", list(_registry.keys()))
def test_walk(benchmark, bench_config, method_name):
    cfg, rowptr, col, start_nodes = bench_config
    fn = _registry[method_name].fn

    # Single run: memory measurement + result metrics
    result, mem_delta_mb = metrics.memory.measure(
        fn, rowptr, col, start_nodes, cfg.walk_length, True
    )
    benchmark.extra_info["memory_rss_delta_mb"] = mem_delta_mb

    assert result.shape == (len(start_nodes), cfg.walk_length)
    assert result.dtype == np.int64

    for metric_name, metric_fn in get_metrics().items():
        benchmark.extra_info[metric_name] = metric_fn(result)

    # Runtime benchmark (multiple runs with warmup)
    benchmark(fn, rowptr, col, start_nodes, cfg.walk_length, True)
