import numpy as np
import igraph as ig
from methods import register


def _csr_to_igraph(rowptr: np.ndarray, col: np.ndarray) -> ig.Graph:
    n = len(rowptr) - 1
    sources = np.repeat(np.arange(n, dtype=np.int64), np.diff(rowptr))
    edges = list(zip(sources.tolist(), col.tolist()))
    # directed=True: CSR stores both directions for undirected graphs,
    # equivalent for random walks using OUT mode.
    return ig.Graph(n=n, edges=edges, directed=True)


@register("igraph", supports_no_backtrack=False, supports_parallel=False)
def walk(rowptr, col, start_nodes, walk_length, allow_backtrack, num_threads):
    g = _csr_to_igraph(rowptr, col)
    n_walks = len(start_nodes)
    result = np.empty((n_walks, walk_length), dtype=np.int64)
    for i, start in enumerate(start_nodes):
        w = g.random_walk(int(start), walk_length - 1, mode="out")
        result[i] = w
    return result
