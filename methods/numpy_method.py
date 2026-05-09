import numpy as np
from methods import register


@register("numpy", supports_no_backtrack=True)
def walk(
    rowptr: np.ndarray,
    col: np.ndarray,
    start_nodes: np.ndarray,
    walk_length: int,
    allow_backtrack: bool,
) -> np.ndarray:
    num_walks = len(start_nodes)
    walks = np.empty((num_walks, walk_length), dtype=np.int64)
    rng = np.random.default_rng(42)

    for i in range(num_walks):
        step = int(start_nodes[i])
        walks[i, 0] = step
        prev = -1

        for k in range(1, walk_length):
            row_start = int(rowptr[step])
            row_end = int(rowptr[step + 1])

            if row_start == row_end:
                walks[i, k] = step
                continue

            neighbors = col[row_start:row_end]

            if not allow_backtrack and prev >= 0:
                mask = neighbors != prev
                if mask.any():
                    neighbors = neighbors[mask]

            prev = step
            step = int(rng.choice(neighbors))
            walks[i, k] = step

    return walks
