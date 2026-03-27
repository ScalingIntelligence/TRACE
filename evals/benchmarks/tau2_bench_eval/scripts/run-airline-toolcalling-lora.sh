#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_DIR="$(dirname "$SCRIPT_DIR")"

for i in 0 1; do
    echo "=========================================="
    echo "Run $i / 2 — airline-toolcalling-lora-${i}"
    echo "=========================================="
    python "$EVAL_DIR/main.py" \
        --config "$EVAL_DIR/config-airline-toolcalling-lora.yml" \
        --save-to "airline-toolcalling-lora-det-${i}"
done

python "$EVAL_DIR/main.py" --config "$EVAL_DIR/config-retail-toolcalling-lora.yml" 

echo "All 3 runs complete."
