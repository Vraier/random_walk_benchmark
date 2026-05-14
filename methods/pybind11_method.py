"""Random walk via explicit pybind11 bindings (no torch build infrastructure)."""
import subprocess
import sysconfig
import importlib.util
from pathlib import Path

import numpy as np
import pybind11
from methods import register

_MODULE_NAME = "rw_pybind11"
_SRC = Path(__file__).parent.parent / "cpp" / "pybind11_walks.cpp"
_BUILD_DIR = Path(__file__).parent.parent / "_build"
_BUILD_DIR.mkdir(exist_ok=True)

_ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
_OUT = _BUILD_DIR / f"{_MODULE_NAME}{_ext_suffix}"

if not _OUT.exists():
    subprocess.run(
        [
            "g++", "-O3", "-fPIC", "-shared", "-std=c++17",
            f"-DMODULE_NAME={_MODULE_NAME}",
            f"-I{sysconfig.get_path('include')}",
            f"-I{pybind11.get_include()}",
            f"-I{np.get_include()}",
            "-o", str(_OUT),
            str(_SRC),
        ],
        check=True,
    )

_spec = importlib.util.spec_from_file_location(_MODULE_NAME, _OUT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


@register("pybind11", supports_no_backtrack=True, supports_parallel=False)
def walk(rowptr, col, start_nodes, walk_length, allow_backtrack, num_threads):
    return _mod.walk_impl(rowptr, col, start_nodes, walk_length, bool(allow_backtrack))
