#!/bin/bash
# Sweep classifier_temperature x routing_topp combinations
# Usage: bash single_lora/sweep_cls_topp.sh

set -e
cd /home/ubuntu/hangook/games/evals/benchmarks/tau2_bench_eval

CONFIG=single_lora/config-airline-5skill-orchestrator--cls-topp.yml
BASE_SAVE="../../single_lora/results/sweep"

for TEMP in 1.0 2.0 3.0 5.0; do
  for TOPP in 0.7 0.8 0.9 0.95; do
    TAG="t${TEMP}-p${TOPP}"
    SAVE="${BASE_SAVE}-${TAG}"

    # Update config
    sed -i "s/classifier_temperature:.*/classifier_temperature: ${TEMP}/" "$CONFIG"
    sed -i "s/routing_topp:.*/routing_topp: ${TOPP}/" "$CONFIG"
    sed -i "s|save_to:.*|save_to: ${SAVE}|" "$CONFIG"

    echo "=== Running ${TAG} ==="
    conda run -n games python main.py --config "$CONFIG" 2>&1 | tail -1

    # Wait for output
    OUTFILE="single_lora/results/sweep-${TAG}.json"
    while [ ! -f "$OUTFILE" ]; do sleep 5; done

    # Score
    SCORE=$(python3 -c "
import json
with open('${OUTFILE}') as f:
    data = json.load(f)
solved = sum(1 for s in data['simulations'] if s.get('reward_info',{}).get('reward',0)==1)
print(f'${TAG}: {solved}/{len(data[\"simulations\"])}')
")
    echo "  $SCORE"
  done
done
