import numpy as np
from numba import njit, prange, set_num_threads
from methods import register


@njit(parallel=True, cache=True)
def _walk_parallel(rowptr, col, start_nodes, walk_length, allow_backtrack):
    n_walks = len(start_nodes)
    result = np.empty((n_walks, walk_length), dtype=np.int64)
    for i in prange(n_walks):
        curr = start_nodes[i]
        result[i, 0] = curr
        prev = np.int64(-1)
        for k in range(1, walk_length):
            rs = rowptr[curr]
            re = rowptr[curr + 1]
            degree = re - rs
            if degree == 0:
                result[i, k] = curr
                continue
            if not allow_backtrack and prev >= 0 and degree > 1:
                idx = np.random.randint(0, degree - 1)
                count = np.int64(0)
                nxt = curr
                for j in range(rs, re):
                    if col[j] == prev:
                        continue
                    if count == idx:
                        nxt = col[j]
                        break
                    count += 1
            else:
                nxt = col[rs + np.random.randint(0, degree)]
            prev = curr
            curr = nxt
            result[i, k] = curr
    return result


@register("numba", supports_no_backtrack=True, supports_parallel=True)
def walk(rowptr, col, start_nodes, walk_length, allow_backtrack, num_threads):
    set_num_threads(num_threads)
    return _walk_parallel(rowptr, col, start_nodes, walk_length, allow_backtrack)
