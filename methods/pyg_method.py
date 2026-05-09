import numpy as np
import torch
from torch_sparse import SparseTensor
from methods import register


@register("pyg_cpu", supports_no_backtrack=False)
def walk(
    rowptr: np.ndarray,
    col: np.ndarray,
    start_nodes: np.ndarray,
    walk_length: int,
    allow_backtrack: bool,
) -> np.ndarray:
    num_nodes = len(rowptr) - 1
    rowptr_t = torch.from_numpy(rowptr)
    col_t = torch.from_numpy(col)
    adj = SparseTensor(rowptr=rowptr_t, col=col_t, sparse_sizes=(num_nodes, num_nodes))
    start_t = torch.from_numpy(start_nodes)
    result = adj.random_walk(start_t, walk_length - 1)
    return result.numpy()
