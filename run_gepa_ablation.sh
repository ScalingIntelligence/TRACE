#!/usr/bin/env bash
set -euo pipefail

# GEPA Oracle Ablation: all 4 skills in parallel tmux sessions
# Reflects on synthetic envs, validates on skill-specific tau2-bench tasks
#
# Usage:
#   bash run_gepa_ablation.sh
#   bash run_gepa_ablation.sh --check   # just check status of running sessions

MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
AGENT_URL="http://localhost:9001/v1"
USER_URL="http://localhost:9002/v1"
REFLECT_URL="http://localhost:9002/v1"
BUDGET=5000
TRAIN_SEEDS=200
WORKERS=22
CHECKPOINT=500
OUTPUT_DIR="outputs/gepa_ablation_oracle"

SKILLS=(tool_calling precondition structured_data multistep)

# ── Check mode ──────────────────────────────────────────────
if [[ "${1:-}" == "--check" ]]; then
    echo "=== GEPA Ablation Session Status ==="
    for skill in "${SKILLS[@]}"; do
        session="gepa_${skill}"
        if tmux has-session -t "$session" 2>/dev/null; then
            # Grab last 3 lines of output
            last=$(tmux capture-pane -t "$session" -p | grep -v '^$' | tail -3)
            echo ""
            echo "[$session] RUNNING"
            echo "$last"
        else
            echo ""
            echo "[$session] NOT RUNNING"
        fi
    done
    exit 0
fi

# ── Verify vLLM servers ────────────────────────────────────
echo "Checking vLLM servers..."
for url in "$AGENT_URL" "$USER_URL"; do
    if ! curl -s "${url}/models" > /dev/null 2>&1; then
        echo "ERROR: vLLM not responding at ${url}"
        exit 1
    fi
done
echo "Both vLLM servers OK."
echo ""

# ── Launch sessions ─────────────────────────────────────────
for skill in "${SKILLS[@]}"; do
    session="gepa_${skill}"

    # Kill existing session if any
    tmux kill-session -t "$session" 2>/dev/null || true

    cmd="source \$(conda info --base)/etc/profile.d/conda.sh && conda activate games && PYTHONUNBUFFERED=1 python gepa_ablation.py \
        --skill ${skill} \
        --oracle \
        --base-url ${AGENT_URL} \
        --model ${MODEL} \
        --user-base-url ${USER_URL} \
        --user-model ${MODEL} \
        --reflection-base-url ${REFLECT_URL} \
        --reflection-model ${MODEL} \
        --reflection-temperature 0.7 \
        --reflection-max-tokens 4096 \
        --train-seeds ${TRAIN_SEEDS} \
        --max-metric-calls ${BUDGET} \
        --max-workers ${WORKERS} \
        --checkpoint-interval ${CHECKPOINT} \
        --output-dir ${OUTPUT_DIR} \
        2>&1 | tee ${OUTPUT_DIR}/${skill}/run.log"

    # Create output dir
    mkdir -p "${OUTPUT_DIR}/${skill}"

    # Launch in tmux
    tmux new-session -d -s "$session" -c /home/ubuntu/hangook/games "$cmd"
    echo "Launched: ${session} (${skill})"
done

echo ""
echo "All 4 skills launched. Monitor with:"
echo "  bash run_gepa_ablation.sh --check"
echo "  tmux attach -t gepa_tool_calling"
echo "  tail -f ${OUTPUT_DIR}/tool_calling/run.log"
