#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_DIR="$(dirname "$SCRIPT_DIR")"

for i in 0 1; do
    echo "=========================================="
    echo "Run $i / 2 — airline-structured-lora-${i}"
    echo "=========================================="
    python "$EVAL_DIR/main.py" \
        --config "$EVAL_DIR/config-airline-structured-lora.yml" \
        --save-to "airline-structured-lora-det-${i}"
done


python "$EVAL_DIR/main.py" --config "$EVAL_DIR/config-retail-structured-lora.yml" 


echo "All 3 runs complete."
