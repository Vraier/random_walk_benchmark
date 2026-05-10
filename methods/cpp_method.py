from pathlib import Path
import numpy as np
import torch
from torch.utils.cpp_extension import load_inline
from methods import register

_cpp_source = (Path(__file__).parent.parent / "cpp" / "random_walks.cpp").read_text()

_cpp_module = load_inline(
    name="cpp_walk_module",
    cpp_sources=_cpp_source,
    functions=["walk_cpp_impl"],
    extra_cflags=["-O3", "-fopenmp"],
    extra_ldflags=["-lgomp"],
    verbose=False,
)


@register("cpp_omp", supports_no_backtrack=False, supports_parallel=True)
def walk(rowptr, col, start_nodes, walk_length, allow_backtrack, num_threads):
    rowptr_t = torch.from_numpy(rowptr)
    col_t = torch.from_numpy(col)
    start_t = torch.from_numpy(start_nodes)
    result = _cpp_module.walk_cpp_impl(rowptr_t, col_t, start_t, walk_length, 42, num_threads)
    return result.numpy()
