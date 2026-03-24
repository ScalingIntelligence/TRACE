#!/bin/bash
# Install vLLM 0.15.0 with weighted LoRA support.
#
# 1. Installs official vllm==0.15.0 from PyPI (with all compiled .so files)
# 2. Copies our 9 modified Python files on top
#
# Usage:
#   bash install.sh
#   bash install.sh /usr/bin/python3   # specify python

set -e

PYTHON="${1:-/usr/bin/python3}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="$SCRIPT_DIR/vllm"
DST=$($PYTHON -c "import site; print(site.getusersitepackages())")/vllm

echo "Installing vllm==0.15.0 from PyPI..."
$PYTHON -m pip install vllm==0.15.0

echo ""
echo "Applying weighted LoRA modifications..."
for f in \
  entrypoints/serve/lora/protocol.py \
  entrypoints/serve/lora/weighted_merge.py \
  entrypoints/openai/models/serving.py \
  entrypoints/openai/api_server.py \
  lora/punica_wrapper/punica_base.py \
  lora/punica_wrapper/punica_gpu.py \
  lora/layers/base_linear.py \
  lora/layers/logits_processor.py \
  lora/layers/fused_moe.py; do
  cp "$SRC/$f" "$DST/$f"
  # Delete stale .pyc so Python uses our patched .py
  pyc_dir="$(dirname "$DST/$f")/__pycache__"
  base="$(basename "$f" .py)"
  rm -f "$pyc_dir/${base}".cpython-*.pyc 2>/dev/null
  echo "  Patched: $f"
done

echo ""
$PYTHON -c "
import vllm
from vllm.entrypoints.serve.lora.protocol import CreateWeightedLoRARequest
print(f'vllm {vllm.__version__} + weighted LoRA: OK')
"
echo ""
echo "Server launch:"
echo "  VLLM_ALLOW_RUNTIME_LORA_UPDATING=True vllm serve <model> --enable-lora --max-loras 2 ..."
