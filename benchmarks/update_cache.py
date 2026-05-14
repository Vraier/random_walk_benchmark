"""Merge a new pytest-benchmark JSON run into results/cache.json.

Usage:
    uv run python scripts/update_cache.py results/20260510_120000.json
"""

import json
import sys
from pathlib import Path

CACHE_FILE = Path("results/cache.json")


def main():
    if len(sys.argv) < 2:
        print("Usage: update_cache.py <benchmark.json>")
        sys.exit(1)

    new_path = Path(sys.argv[1])
    with open(new_path) as f:
        new_data = json.load(f)

    cache = {}
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            cache = json.load(f)

    added = 0
    for b in new_data["benchmarks"]:
        key = b["param"]  # e.g. "n2000-wl20-nw5-deg15-cpp_openmp"
        if key not in cache:
            cache[key] = {
                "mean_s":             b["stats"]["mean"],
                "stddev_s":           b["stats"]["stddev"],
                "coverage_fraction":  b["extra_info"].get("coverage_fraction"),
                "param":              key,
            }
            added += 1

    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

    print(f"Cache updated: +{added} new entries ({len(cache)} total) -> {CACHE_FILE}")


if __name__ == "__main__":
    main()
