#!/usr/bin/env bash
set -euo pipefail

RESULTS_DIR="results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

mkdir -p "$RESULTS_DIR"

uv run pytest benchmarks/ --benchmark-json="$RESULTS_DIR/${TIMESTAMP}.json" -v
