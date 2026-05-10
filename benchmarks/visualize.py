"""Visualize benchmark results from results/cache.json.

Usage:
    uv run python scripts/visualize.py
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent.parent / "results"
CACHE_FILE = RESULTS_DIR / "cache.json"
# method label ending in -tN where N>1 = parallel run → solid line, else dashed
COLORS = {
    "numpy":      "tab:blue",
    "pyg_cpu":    "tab:orange",
    "cpp_omp":    "tab:green",
    "numba":      "tab:purple",
    "igraph":     "tab:brown",
    "pybind11":   "tab:pink",
    "cpython_c":  "tab:cyan",
    "cython":     "tab:olive",
}


def _base_method(label: str) -> str:
    """Strip thread suffix: 'cpp_openmp-t16' -> 'cpp_openmp'."""
    return re.sub(r"-t\d+$", "", label)


def _is_parallel_run(label: str) -> bool:
    """True if label ends with -tN where N > 1."""
    m = re.search(r"-t(\d+)$", label)
    return bool(m and int(m.group(1)) > 1)


def parse_records(cache: dict) -> list[dict]:
    records = []
    for key, v in cache.items():
        # New format: n...-wl...-nw...-deg...-<method>-t<threads>
        # Old format: n...-wl...-nw...-deg...-<method>   (num_threads=1 implied)
        m = re.match(r"n(\d+)-wl(\d+)-nw(\d+)-deg(\d+)-(.+?)(?:-t(\d+))?$", key)
        if not m:
            continue
        method_base = m.group(5)
        num_threads = int(m.group(6)) if m.group(6) else 1
        method_label = f"{method_base}-t{num_threads}" if num_threads > 1 else method_base
        records.append({
            "n":                  int(m.group(1)),
            "walk_length":        int(m.group(2)),
            "num_walks_per_node": int(m.group(3)),
            "avg_degree":         int(m.group(4)),
            "method":             method_label,
            "num_threads":        num_threads,
            "mean_s":             v["mean_s"],
            "stddev_s":           v["stddev_s"],
            "memory_mb":          v.get("memory_mb") or float("nan"),  # 0 = unreliable
        })
    return records


def aggregate(records, fixed_key, fixed_val, x_key, mean_key, std_key=None, require_complete=False):
    """For records where fixed_key==fixed_val, average mean_key (and std_key)
    over all other varying params, grouped by (method, x_key).

    require_complete: if True, return NaN for any group that contains a NaN value.
    """
    filtered = [r for r in records if r[fixed_key] == fixed_val]
    methods = sorted(set(r["method"] for r in filtered))
    x_vals = sorted(set(r[x_key] for r in filtered))

    result = {}
    for method in methods:
        means, stds = [], []
        for x in x_vals:
            group = [r for r in filtered if r["method"] == method and r[x_key] == x]
            if not group:
                means.append(float("nan"))
                if std_key:
                    stds.append(float("nan"))
                continue
            vals = np.array([r[mean_key] for r in group], dtype=float)
            if require_complete and np.any(np.isnan(vals)):
                means.append(float("nan"))
            else:
                means.append(np.nanmean(vals))
            if std_key:
                sv = np.array([r[std_key] for r in group], dtype=float)
                stds.append(float("nan") if require_complete and np.any(np.isnan(sv)) else np.nanmean(sv))
        result[method] = {"x": x_vals, "mean": np.array(means), "std": np.array(stds) if std_key else None}
    return result


def plot_lines(ax, agg, x_label, y_label, title, scale_y=1.0, yscale="linear"):
    for method, d in agg.items():
        y = d["mean"] * scale_y
        color = COLORS.get(_base_method(method))
        linestyle = "--" if _is_parallel_run(method) else "-"
        ax.plot(d["x"], y, marker="o", label=method, color=color, linewidth=2, linestyle=linestyle)
        if d["std"] is not None:
            std = d["std"] * scale_y
            ax.fill_between(d["x"], y - std, y + std, alpha=0.2, color=color)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_yscale(yscale)
    ax.legend()
    ax.grid(True, alpha=0.3)


def main():
    if not CACHE_FILE.exists():
        print(f"No cache found at {CACHE_FILE}. Run ./run_benchmarks.sh first.")
        sys.exit(1)

    with open(CACHE_FILE) as f:
        cache = json.load(f)

    print(f"Loading cache: {CACHE_FILE} ({len(cache)} entries)")
    records = parse_records(cache)

    max_n = max(r["n"] for r in records)
    max_wl = max(r["walk_length"] for r in records)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Benchmark results", fontsize=13)

    # Runtime vs n (fixed walk_length=max)
    plot_lines(
        axes[0, 0],
        aggregate(records, "walk_length", max_wl, "n", "mean_s", "stddev_s"),
        x_label="Number of nodes",
        y_label="Mean time (ms)",
        title=f"Runtime vs n  [walk_length={max_wl}]",
        scale_y=1000,
        yscale="log",
    )

    # Runtime vs walk_length (fixed n=max)
    plot_lines(
        axes[0, 1],
        aggregate(records, "n", max_n, "walk_length", "mean_s", "stddev_s"),
        x_label="Walk length",
        y_label="Mean time (ms)",
        title=f"Runtime vs walk_length  [n={max_n}]",
        scale_y=1000,
        yscale="log",
    )

    # Memory vs n (fixed walk_length=max)
    plot_lines(
        axes[1, 0],
        aggregate(records, "walk_length", max_wl, "n", "memory_mb", require_complete=True),
        x_label="Number of nodes",
        y_label="Memory RSS delta (MB)",
        title=f"Memory vs n  [walk_length={max_wl}]",
    )

    # Memory vs walk_length (fixed n=max)
    plot_lines(
        axes[1, 1],
        aggregate(records, "n", max_n, "walk_length", "memory_mb", require_complete=True),
        x_label="Walk length",
        y_label="Memory RSS delta (MB)",
        title=f"Memory vs walk_length  [n={max_n}]",
    )

    plt.tight_layout()
    out_path = RESULTS_DIR / "plots.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
