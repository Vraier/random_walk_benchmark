"""Coverage-based result metrics."""
import numpy as np
from metrics import register


@register("coverage_fraction")
def coverage_fraction(result: np.ndarray) -> float:
    """Mean fraction of distinct nodes visited per walk (unique nodes / walk_length).

    1.0 means no revisits; lower values indicate more backtracking/revisiting.
    """
    unique_per_walk = np.array([len(np.unique(result[i])) for i in range(len(result))])
    return float(unique_per_walk.mean() / result.shape[1])
