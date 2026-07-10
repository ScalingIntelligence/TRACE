#!/bin/bash
# Durable phase 1 launcher.
# Everything lives on /workspace PVC so this script is the only thing
# needed to recover from a pod kill.
#
# Usage:
#   bash /workspace/launch_phase1.sh
#
# Resumes from /workspace/trace_pipeline/runs/baseline/preds.json (skips
# already-completed instances).
set -e

WORKSPACE=${WORKSPACE:-/workspace}
PIPE_ROOT=${TRACE_PIPE_ROOT:-$WORKSPACE/trace_pipeline}
CODE=$WORKSPACE/trace_pipeline_code
LOGS=$PIPE_ROOT/logs

mkdir -p "$LOGS" "$PIPE_ROOT/runs/baseline" "$PIPE_ROOT/state"

export HOME=$WORKSPACE/home
export USER=clauderunner

# Load HF_TOKEN (.env has KEY=VALUE format)
if [ -f "$WORKSPACE/home/.env" ]; then
  export $(grep -v '^#' "$WORKSPACE/home/.env" | xargs)
fi
echo "[launch] HF_TOKEN set: ${HF_TOKEN:+yes}"

# 1. vLLM (run with CLEAN python env — /workspace/pylibs tokenizers conflicts with vLLM's transformers)
# -------------------------------------------------------------------------
if curl -sf http://localhost:8000/v1/models -m 5 > /dev/null 2>&1; then
  echo "[launch] vLLM already up"
else
  echo "[launch] starting vLLM DP=8 (clean PYTHONPATH)…"
  PYTHONPATH="" PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  CUDA_VISIBLE_DEVICES=0,1,2,3 nohup vllm serve Qwen/Qwen3.6-27B \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 1 \
    --data-parallel-size 4 \
    --dtype bfloat16 \
    --max-model-len 131072 \
    --enforce-eager \
    --gpu-memory-utilization 0.90 \
    --max-num-batched-tokens 8192 --max-num-seqs 16 \
    --trust-remote-code \
    --served-model-name Qwen/Qwen3.6-27B \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    > "$LOGS/vllm_pod_e.log" 2>&1 &
  echo "[launch] vllm pid=$!"
  echo "[launch] waiting up to 10 min for vLLM ready…"
  for i in $(seq 1 40); do
    if curl -sf http://localhost:8000/v1/models -m 5 > /dev/null 2>&1; then
      echo "[launch] vLLM ready after ${i}x15s"; break
    fi
    sleep 15
  done
fi

# Confirm vLLM is up before starting phase 1
if ! curl -sf http://localhost:8000/v1/models -m 5 > /dev/null 2>&1; then
  echo "[launch] ERROR: vLLM did not come up; aborting phase 1"
  echo "[launch] Check $LOGS/vllm_pod_e.log"
  exit 1
fi

# Now set Python env for phase 1 + snapshot daemon (need /workspace/pylibs for mini-swe-agent)
export PATH=$WORKSPACE/pylibs/bin:$PATH
export PYTHONPATH=$WORKSPACE/pylibs

# 2. Snapshot daemon ----------------------------------------------------
if pgrep -f hf_snapshot_daemon.py > /dev/null; then
  echo "[launch] snapshot daemon already running"
else
  echo "[launch] starting snapshot daemon…"
  cd "$CODE"
  HF_TOKEN=$HF_TOKEN \
    TRACE_PIPE_ROOT=$PIPE_ROOT \
    TRACE_HF_BASELINE_REPO=${TRACE_HF_BASELINE_REPO:-ScalingIntelligence/qwen3.6-swe-base} \
    TRACE_HF_INTERVAL_SEC=${TRACE_HF_INTERVAL_SEC:-600} \
    nohup python3 hf_snapshot_daemon.py > "$LOGS/hf_snapshot.log" 2>&1 &
  echo "[launch] snap daemon pid=$!"
fi

# 3. Phase 1 ------------------------------------------------------------
if pgrep -f phase1_baseline_eval.py > /dev/null; then
  echo "[launch] phase 1 already running"
else
  echo "[launch] starting phase 1 with resume…"
  cd "$CODE"
  HF_TOKEN=$HF_TOKEN \
    MODAL_PROFILE=scalingintelligence \
    TRACE_PIPE_ROOT=$PIPE_ROOT \
    TRACE_PHASE1_CONFIG=$CODE/configs/swebench_qwen36_vllm.yaml \
    TRACE_VLLM_E_URL=http://localhost:8000/v1 \
    TRACE_MODEL=Qwen/Qwen3.6-27B \
    TRACE_PHASE1_WORKERS=${TRACE_PHASE1_WORKERS:-50} \
    TRACE_PHASE1_RUN_ID=${TRACE_PHASE1_RUN_ID:-trace_baseline_qwen36} \
    TRACE_HF_BASELINE_REPO=${TRACE_HF_BASELINE_REPO:-ScalingIntelligence/qwen3.6-swe-base} \
    nohup python3 phase1_baseline_eval.py > "$LOGS/phase1.log" 2>&1 &
  echo "[launch] phase 1 pid=$!"
fi

echo "[launch] ALL UP. Tail logs at $LOGS/phase1.log and $LOGS/hf_snapshot.log"
