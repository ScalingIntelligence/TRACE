#!/bin/bash
# Run GEPA prompt sweep across iter checkpoints: ~500, ~1000, ~3000
# Runs airline first, then retail for each checkpoint.

set -e

SCRIPT_DIR="/home/ubuntu/hangook/games/evals/benchmarks/tau2_bench_eval"
BASE_DIR="/home/ubuntu/hangook/games/outputs/gepa_ablation_oracle"
TEMPLATE_AIRLINE="${SCRIPT_DIR}/config-gepa-airline.yml"
TEMPLATE_RETAIL="${SCRIPT_DIR}/config-gepa-retail.yml"

# Define iter checkpoints: label, tool_calling, precondition, multistep, structured_data
declare -A ITERS
ITERS[500]="502 504 502 502"
ITERS[1000]="1003 1005 1005 1004"
ITERS[3000]="3044 3015 3032 3017"

# Ordered iteration
for ITER_LABEL in 500 1000 3000; do
    read -r TC PC MS SD <<< "${ITERS[$ITER_LABEL]}"

    echo "============================================"
    echo "  GEPA iter ~${ITER_LABEL}"
    echo "  tool_calling=${TC} precondition=${PC} multistep=${MS} structured_data=${SD}"
    echo "============================================"

    # Build prompt file list (YAML format)
    PROMPT_FILES="agent_prompt_files:
  - ${BASE_DIR}/tool_calling/best_prompt_at_${TC}.txt
  - ${BASE_DIR}/precondition/best_prompt_at_${PC}.txt
  - ${BASE_DIR}/multistep/best_prompt_at_${MS}.txt
  - ${BASE_DIR}/structured_data/best_prompt_at_${SD}.txt"

    # --- Airline ---
    CONFIG_AIRLINE="/tmp/config-gepa-airline-${ITER_LABEL}.yml"
    sed -e "s|save_to: gepa-airline-2000|save_to: gepa-airline-${ITER_LABEL}|" \
        "${TEMPLATE_AIRLINE}" > "${CONFIG_AIRLINE}"
    # Replace the agent_prompt_files block (4 lines after the key)
    python3 -c "
import re, sys
with open('${CONFIG_AIRLINE}') as f:
    text = f.read()
new_block = '''${PROMPT_FILES}'''
text = re.sub(r'agent_prompt_files:\n(  - .+\n){4}', new_block + '\n', text)
with open('${CONFIG_AIRLINE}', 'w') as f:
    f.write(text)
"
    echo ""
    echo ">>> Running AIRLINE iter ~${ITER_LABEL}..."
    echo "    Config: ${CONFIG_AIRLINE}"
    echo ""
    cd "${SCRIPT_DIR}"
    python main.py --config "${CONFIG_AIRLINE}"

    # --- Retail ---
    CONFIG_RETAIL="/tmp/config-gepa-retail-${ITER_LABEL}.yml"
    sed -e "s|save_to: gepa-retail-2000|save_to: gepa-retail-${ITER_LABEL}|" \
        "${TEMPLATE_RETAIL}" > "${CONFIG_RETAIL}"
    python3 -c "
import re, sys
with open('${CONFIG_RETAIL}') as f:
    text = f.read()
new_block = '''${PROMPT_FILES}'''
text = re.sub(r'agent_prompt_files:\n(  - .+\n){4}', new_block + '\n', text)
with open('${CONFIG_RETAIL}', 'w') as f:
    f.write(text)
"
    echo ""
    echo ">>> Running RETAIL iter ~${ITER_LABEL}..."
    echo "    Config: ${CONFIG_RETAIL}"
    echo ""
    cd "${SCRIPT_DIR}"
    python main.py --config "${CONFIG_RETAIL}"

done

echo ""
echo "============================================"
echo "  All GEPA sweep runs complete!"
echo "============================================"
