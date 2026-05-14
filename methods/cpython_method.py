"""Random walk via raw CPython C extension (PyArg_ParseTuple style)."""
import subprocess
import sysconfig
import importlib.util
from pathlib import Path

import numpy as np
from methods import register

_MODULE_NAME = "cpython_walks"
_SRC = Path(__file__).parent.parent / "cpp" / "cpython_walks.c"
_BUILD_DIR = Path(__file__).parent.parent / "_build"
_BUILD_DIR.mkdir(exist_ok=True)

_ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
_OUT = _BUILD_DIR / f"{_MODULE_NAME}{_ext_suffix}"

if not _OUT.exists():
    subprocess.run(
        [
            "gcc", "-O3", "-fPIC", "-shared",
            f"-I{sysconfig.get_path('include')}",
            f"-I{np.get_include()}",
            "-o", str(_OUT),
            str(_SRC),
        ],
        check=True,
    )

_spec = importlib.util.spec_from_file_location(_MODULE_NAME, _OUT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


@register("cpython_c", supports_no_backtrack=True, supports_parallel=False)
def walk(rowptr, col, start_nodes, walk_length, allow_backtrack, num_threads):
    return _mod.walk_impl(rowptr, col, start_nodes, walk_length, int(allow_backtrack))
