import numpy as np
from numba import njit, prange, set_num_threads
from methods import register


@njit(parallel=True, cache=True)
def _walk_parallel(rowptr, col, start_nodes, walk_length):
    n_walks = len(start_nodes)
    result = np.empty((n_walks, walk_length), dtype=np.int64)
    for i in prange(n_walks):
        node = start_nodes[i]
        result[i, 0] = node
        for k in range(1, walk_length):
            rs = rowptr[node]
            re = rowptr[node + 1]
            if rs == re:
                result[i, k] = node
            else:
                node = col[rs + np.random.randint(0, re - rs)]
                result[i, k] = node
    return result


@register("numba", supports_no_backtrack=False, supports_parallel=True)
def walk(rowptr, col, start_nodes, walk_length, allow_backtrack, num_threads):
    set_num_threads(num_threads)
    return _walk_parallel(rowptr, col, start_nodes, walk_length)
