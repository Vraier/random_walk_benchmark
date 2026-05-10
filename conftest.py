import json
import pytest
from pathlib import Path

CACHE_FILE = Path("results/cache.json")


def _load_cache() -> dict:
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def pytest_configure(config):
    config._rwb_cache = _load_cache()


def pytest_runtest_setup(item):
    cache = getattr(item.config, "_rwb_cache", {})
    # param is e.g. "n2000-wl20-nw5-deg15-cpp_openmp"
    param = getattr(item, "callspec", None) and item.callspec.id
    if param and param in cache:
        pytest.skip(f"cached ({param})")
