# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np
from libc.stdint cimport int64_t, uint64_t

def walk_impl(
    np.ndarray[int64_t, ndim=1] rowptr,
    np.ndarray[int64_t, ndim=1] col,
    np.ndarray[int64_t, ndim=1] start_nodes,
    int64_t walk_length,
):
    cdef int64_t n_walks = start_nodes.shape[0]
    cdef np.ndarray[int64_t, ndim=2] result = np.empty((n_walks, walk_length), dtype=np.int64)
    cdef int64_t i, k, node, rs, re
    cdef uint64_t state = 42

    for i in range(n_walks):
        node = start_nodes[i]
        result[i, 0] = node
        for k in range(1, walk_length):
            rs = rowptr[node]
            re = rowptr[node + 1]
            if rs == re:
                result[i, k] = node
            else:
                state = state * <uint64_t>6364136223846793005 + <uint64_t>1442695040888963407
                node = col[rs + <int64_t>(state >> 33) % (re - rs)]
                result[i, k] = node
    return result
