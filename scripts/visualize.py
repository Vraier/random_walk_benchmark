"""Visualize benchmark results from a pytest-benchmark JSON file.

Usage:
    uv run python scripts/visualize.py               # uses latest results/*.json
    uv run python scripts/visualize.py results/foo.json
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent.parent / "results"
COLORS = {"numpy": "tab:blue", "pyg_cpu": "tab:orange", "cpp_openmp": "tab:green"}


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def parse_records(data: dict) -> list[dict]:
    records = []
    for b in data["benchmarks"]:
        m = re.match(r"n(\d+)-wl(\d+)-nw(\d+)-deg(\d+)-(.+)", b["param"])
        if not m:
            continue
        records.append({
            "n":                  int(m.group(1)),
            "walk_length":        int(m.group(2)),
            "num_walks_per_node": int(m.group(3)),
            "avg_degree":         int(m.group(4)),
            "method":             m.group(5),
            "mean_s":             b["stats"]["mean"],
            "stddev_s":           b["stats"]["stddev"],
            "memory_mb":          b["extra_info"].get("memory_rss_delta_mb", float("nan")),
        })
    return records


def aggregate(records, fixed_key, fixed_val, x_key, mean_key, std_key=None):
    """For records where fixed_key==fixed_val, average mean_key (and std_key)
    over all other varying params, grouped by (method, x_key)."""
    filtered = [r for r in records if r[fixed_key] == fixed_val]
    methods = sorted(set(r["method"] for r in filtered))
    x_vals = sorted(set(r[x_key] for r in filtered))

    result = {}
    for method in methods:
        means, stds = [], []
        for x in x_vals:
            group = [r for r in filtered if r["method"] == method and r[x_key] == x]
            means.append(np.mean([r[mean_key] for r in group]) if group else float("nan"))
            if std_key:
                stds.append(np.mean([r[std_key] for r in group]) if group else float("nan"))
        result[method] = {"x": x_vals, "mean": np.array(means), "std": np.array(stds) if std_key else None}
    return result


def plot_lines(ax, agg, x_label, y_label, title, scale_y=1.0, yscale="linear"):
    for method, d in agg.items():
        y = d["mean"] * scale_y
        color = COLORS.get(method)
        ax.plot(d["x"], y, marker="o", label=method, color=color, linewidth=2)
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
    if len(sys.argv) > 1:
        json_path = Path(sys.argv[1])
    else:
        jsons = sorted(RESULTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
        if not jsons:
            print("No JSON files found in results/")
            sys.exit(1)
        json_path = jsons[-1]

    print(f"Loading: {json_path}")
    data = load_json(json_path)
    records = parse_records(data)

    max_n = max(r["n"] for r in records)
    max_wl = max(r["walk_length"] for r in records)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Benchmark results — {json_path.stem}", fontsize=13)

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
        aggregate(records, "walk_length", max_wl, "n", "memory_mb"),
        x_label="Number of nodes",
        y_label="Memory RSS delta (MB)",
        title=f"Memory vs n  [walk_length={max_wl}]",
    )

    # Memory vs walk_length (fixed n=max)
    plot_lines(
        axes[1, 1],
        aggregate(records, "n", max_n, "walk_length", "memory_mb"),
        x_label="Walk length",
        y_label="Memory RSS delta (MB)",
        title=f"Memory vs walk_length  [n={max_n}]",
    )

    plt.tight_layout()
    out_path = json_path.with_name(json_path.stem + "_plots.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
