"""Coverage-based result metrics."""
import numpy as np
# from metrics import register


# @register("coverage_time")
# def coverage_time(result: np.ndarray) -> float:
#     """Average steps until all start nodes have been visited at least once."""
#     num_walks, walk_length = result.shape
#     times = []
#     for walk in result:
#         seen = set()
#         for step, node in enumerate(walk):
#             seen.add(int(node))
#             if len(seen) == walk_length:  # or some target coverage
#                 times.append(step)
#                 break
#         else:
#             times.append(walk_length)
#     return float(np.mean(times))
