"""Random walk via Cython (.pyx auto-compiled by pyximport)."""
import numpy as np
import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level="3str")

from methods import cython_walks as _mod  # noqa: E402 — must come after pyximport.install()
from methods import register


@register("cython", supports_no_backtrack=True, supports_parallel=False)
def walk(rowptr, col, start_nodes, walk_length, allow_backtrack, num_threads):
    return _mod.walk_impl(rowptr, col, start_nodes, walk_length, bool(allow_backtrack))
