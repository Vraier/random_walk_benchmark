"""Random walk in pure Python (no numpy ops in the hot path)."""
import random
from methods import register


@register("python", supports_no_backtrack=True, supports_parallel=False)
def walk(rowptr, col, start_nodes, walk_length, allow_backtrack, num_threads):
    rowptr = rowptr.tolist()
    col = col.tolist()
    start_nodes = start_nodes.tolist()
    num_walks = len(start_nodes)

    import numpy as np
    result = [[0] * walk_length for _ in range(num_walks)]

    rng = random.Random(42)
    for i in range(num_walks):
        curr = start_nodes[i]
        result[i][0] = curr
        prev = -1
        for k in range(1, walk_length):
            rs = rowptr[curr]
            re = rowptr[curr + 1]
            degree = re - rs
            if degree == 0:
                result[i][k] = curr
                continue
            if not allow_backtrack and prev >= 0 and degree > 1:
                idx = rng.randint(0, degree - 2)
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
                nxt = col[rng.randint(rs, re - 1)]
            prev = curr
            curr = nxt
            result[i][k] = curr

    return np.array(result, dtype=np.int64)
