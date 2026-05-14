# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np
from libc.stdint cimport int64_t, uint64_t

def walk_impl(
    np.ndarray[int64_t, ndim=1] rowptr,
    np.ndarray[int64_t, ndim=1] col,
    np.ndarray[int64_t, ndim=1] start_nodes,
    int64_t walk_length,
    bint allow_backtrack,
):
    cdef int64_t n_walks = start_nodes.shape[0]
    cdef np.ndarray[int64_t, ndim=2] result = np.empty((n_walks, walk_length), dtype=np.int64)
    cdef int64_t i, k, j, curr, prev, nxt, rs, re, degree, idx, count
    cdef uint64_t state = 42

    for i in range(n_walks):
        curr = start_nodes[i]
        result[i, 0] = curr
        prev = -1
        for k in range(1, walk_length):
            rs = rowptr[curr]
            re = rowptr[curr + 1]
            degree = re - rs
            if degree == 0:
                result[i, k] = curr
                continue
            if not allow_backtrack and prev >= 0 and degree > 1:
                state = state * <uint64_t>6364136223846793005 + <uint64_t>1442695040888963407
                idx = <int64_t>(state >> 33) % (degree - 1)
                count = 0
                nxt = curr
                for j in range(rs, re):
                    if col[j] == prev:
                        continue
                    if count == idx:
                        nxt = col[j]
                        break
                    count += 1
            else:
                state = state * <uint64_t>6364136223846793005 + <uint64_t>1442695040888963407
                nxt = col[rs + <int64_t>(state >> 33) % degree]
            prev = curr
            curr = nxt
            result[i, k] = curr
    return result
