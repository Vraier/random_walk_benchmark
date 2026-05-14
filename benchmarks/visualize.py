"""Visualize benchmark results from results/cache.json.

Usage:
    uv run python benchmarks/visualize.py
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent.parent / "results"
CACHE_FILE = RESULTS_DIR / "cache.json"

BASE_N               = 10000
BASE_WALK_LENGTH     = 20
BASE_NUM_WALKS       = 20
BASE_AVG_DEGREE      = 15
SCALE_N_VALUES          = [2000, 6000, 10000, 14000, 18000]
SCALE_WALK_LENGTH_VALUES = [10, 20, 50, 100, 200]
SCALE_NUM_WALKS_VALUES  = [1, 5, 10, 20, 50]
SCALE_THREADS_VALUES    = [1, 2, 4, 8]

COLORS = {
    "python":     "tab:gray",
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
    """Strip '-nobt' suffix to get base method name for color lookup."""
    return label.removesuffix("-nobt")


def parse_records(cache: dict) -> list[dict]:
    records = []
    for key, v in cache.items():
        # n...-wl...-nw...-deg...-<method>-t<threads>[-nobt]
        m = re.match(r"n(\d+)-wl(\d+)-nw(\d+)-deg(\d+)-(.+?)-t(\d+)(-nobt)?$", key)
        if not m:
            continue
        records.append({
            "n":                  int(m.group(1)),
            "walk_length":        int(m.group(2)),
            "num_walks_per_node": int(m.group(3)),
            "avg_degree":         int(m.group(4)),
            "method":             m.group(5),
            "num_threads":        int(m.group(6)),
            "allow_backtrack":    m.group(7) is None,
            "mean_s":             v["mean_s"],
            "stddev_s":           v["stddev_s"],
            "coverage_fraction":  v.get("coverage_fraction") or float("nan"),
        })
    return records


def _filter(records, **kwargs):
    result = records
    for k, v in kwargs.items():
        result = [r for r in result if r[k] == v]
    return result


def _aggregate(records, x_key, group_key="method"):
    """Group records by group_key and x_key, averaging across any remaining variation."""
    groups = sorted(set(r[group_key] for r in records))
    x_vals = sorted(set(r[x_key] for r in records))

    result = {}
    for g in groups:
        means, stds, mems = [], [], []
        for x in x_vals:
            group = [r for r in records if r[group_key] == g and r[x_key] == x]
            if not group:
                means.append(float("nan"))
                stds.append(float("nan"))
                mems.append(float("nan"))
                continue
            means.append(np.nanmean([r["mean_s"] for r in group]))
            stds.append(np.nanmean([r["stddev_s"] for r in group]))
            cov_vals = np.array([r["coverage_fraction"] for r in group])
            mems.append(float("nan") if np.any(np.isnan(cov_vals)) else np.nanmean(cov_vals))
        result[g] = {
            "x": x_vals,
            "mean_s": np.array(means),
            "stddev_s": np.array(stds),
            "coverage_fraction": np.array(mems),
        }
    return result


def _plot_runtime(ax, agg, x_label, title, linestyles=None):
    for method, d in agg.items():
        y = d["mean_s"] * 1000
        std = d["stddev_s"] * 1000
        color = COLORS.get(_base_method(method))
        ls = (linestyles or {}).get(method, "-")
        ax.plot(d["x"], y, marker="o", label=method, color=color, linewidth=2, linestyle=ls)
        ax.fill_between(d["x"], y - std, y + std, alpha=0.2, color=color)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Mean time (ms)")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_coverage(ax, agg, x_label, title, linestyles=None):
    for method, d in agg.items():
        color = COLORS.get(_base_method(method))
        ls = (linestyles or {}).get(method, "-")
        ax.plot(d["x"], d["coverage_fraction"], marker="o", label=method, color=color, linewidth=2, linestyle=ls)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Coverage fraction (unique nodes / walk_length)")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def _save(fig, name):
    fig.tight_layout()
    path = RESULTS_DIR / f"{name}.png"
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close(fig)


def plot_scale_n(records):
    recs = _filter(records, walk_length=BASE_WALK_LENGTH, num_walks_per_node=BASE_NUM_WALKS,
                   avg_degree=BASE_AVG_DEGREE, num_threads=1, allow_backtrack=True)
    if not recs:
        print("No data for scale_n"); return
    agg = _aggregate(recs, x_key="n")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Scale n  [wl={BASE_WALK_LENGTH}, nw={BASE_NUM_WALKS}, deg={BASE_AVG_DEGREE}]")
    _plot_runtime(ax1, agg, "Number of nodes", "Runtime vs n")
    _plot_coverage(ax2, agg, "Number of nodes", "Coverage vs n")
    _save(fig, "scale_n")


def plot_scale_walk_length(records):
    recs = _filter(records, n=BASE_N, num_walks_per_node=BASE_NUM_WALKS,
                   avg_degree=BASE_AVG_DEGREE, num_threads=1, allow_backtrack=True)
    if not recs:
        print("No data for scale_walk_length"); return
    agg = _aggregate(recs, x_key="walk_length")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Scale walk_length  [n={BASE_N}, nw={BASE_NUM_WALKS}, deg={BASE_AVG_DEGREE}]")
    _plot_runtime(ax1, agg, "Walk length", "Runtime vs walk_length")
    _plot_coverage(ax2, agg, "Walk length", "Coverage vs walk_length")
    _save(fig, "scale_walk_length")


def plot_scale_num_walks(records):
    recs = _filter(records, n=BASE_N, walk_length=BASE_WALK_LENGTH,
                   avg_degree=BASE_AVG_DEGREE, num_threads=1, allow_backtrack=True)
    if not recs:
        print("No data for scale_num_walks"); return
    agg = _aggregate(recs, x_key="num_walks_per_node")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Scale num_walks  [n={BASE_N}, wl={BASE_WALK_LENGTH}, deg={BASE_AVG_DEGREE}]")
    _plot_runtime(ax1, agg, "Walks per node", "Runtime vs num_walks")
    _plot_coverage(ax2, agg, "Walks per node", "Coverage vs num_walks")
    _save(fig, "scale_num_walks")


def plot_scale_threads(records):
    recs = _filter(records, n=BASE_N, walk_length=BASE_WALK_LENGTH,
                   num_walks_per_node=BASE_NUM_WALKS, avg_degree=BASE_AVG_DEGREE, allow_backtrack=True)
    recs = [r for r in recs if r["num_threads"] in SCALE_THREADS_VALUES]
    # Only keep methods that have data for more than one thread count (i.e., parallel methods)
    thread_counts_per_method: dict[str, set] = {}
    for r in recs:
        thread_counts_per_method.setdefault(r["method"], set()).add(r["num_threads"])
    parallel_methods = {m for m, ts in thread_counts_per_method.items() if len(ts) > 1}
    recs = [r for r in recs if r["method"] in parallel_methods]
    if not recs:
        print("No data for scale_threads"); return
    agg = _aggregate(recs, x_key="num_threads")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Scale threads  [n={BASE_N}, wl={BASE_WALK_LENGTH}, nw={BASE_NUM_WALKS}, deg={BASE_AVG_DEGREE}]")
    _plot_runtime(ax1, agg, "Number of threads", "Runtime vs threads")
    _plot_coverage(ax2, agg, "Number of threads", "Coverage vs threads")
    _save(fig, "scale_threads")


def plot_backtrack(records):
    recs = _filter(records, walk_length=BASE_WALK_LENGTH, num_walks_per_node=BASE_NUM_WALKS,
                   avg_degree=BASE_AVG_DEGREE, num_threads=1)
    recs = [r for r in recs if r["n"] in SCALE_N_VALUES]
    if not recs:
        print("No data for backtrack"); return

    # Derive group label: method (bt=True, solid) vs method-nobt (bt=False, dashed)
    for r in recs:
        r["method_bt"] = r["method"] if r["allow_backtrack"] else f"{r['method']}-nobt"

    agg = _aggregate(recs, x_key="n", group_key="method_bt")
    linestyles = {k: "-" if not k.endswith("-nobt") else "--" for k in agg}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Backtrack toggle  [wl={BASE_WALK_LENGTH}, nw={BASE_NUM_WALKS}, deg={BASE_AVG_DEGREE}]")
    _plot_runtime(ax1, agg, "Number of nodes", "Runtime vs n", linestyles)
    _plot_coverage(ax2, agg, "Number of nodes", "Coverage vs n", linestyles)

    # Rebuild legends: one entry per method (no -nobt duplicates) + line style key
    from matplotlib.lines import Line2D
    style_handles = [
        Line2D([0], [0], color="black", linewidth=2, linestyle="-"),
        Line2D([0], [0], color="black", linewidth=2, linestyle="--"),
    ]
    style_labels = ["allow backtrack", "no backtrack"]
    for ax in (ax1, ax2):
        handles, labels = ax.get_legend_handles_labels()
        method_pairs = [(h, l) for h, l in zip(handles, labels) if not l.endswith("-nobt")]
        mh, ml = zip(*method_pairs) if method_pairs else ([], [])
        ax.legend(list(mh) + style_handles, list(ml) + style_labels, fontsize=8)

    _save(fig, "backtrack")


def main():
    if not CACHE_FILE.exists():
        print(f"No cache found at {CACHE_FILE}. Run ./run_benchmarks.sh first.")
        sys.exit(1)

    with open(CACHE_FILE) as f:
        cache = json.load(f)

    print(f"Loading cache: {CACHE_FILE} ({len(cache)} entries)")
    records = parse_records(cache)
    print(f"Parsed {len(records)} records")

    plot_scale_n(records)
    plot_scale_walk_length(records)
    plot_scale_num_walks(records)
    plot_scale_threads(records)
    plot_backtrack(records)


if __name__ == "__main__":
    main()
