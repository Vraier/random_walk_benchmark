#!/usr/bin/env bash
set -euo pipefail

RESULTS_DIR="results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
JSON_FILE="$RESULTS_DIR/${TIMESTAMP}.json"

mkdir -p "$RESULTS_DIR"

echo "Running benchmarks (cached tests will be skipped)..."
uv run pytest benchmarks/ --benchmark-json="$JSON_FILE" -v

echo "Updating cache..."
uv run python benchmarks/update_cache.py "$JSON_FILE"
