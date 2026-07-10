#!/usr/bin/env bash
# SWE-bench Verified eval on Qwen3.6-27B + a TRACE LoRA checkpoint.
#
# Topology: 8×H100 80GB, vLLM = TP=4 × DP=2 (two replicas), each replica
# serves at 128k context with ~10 concurrent sequences. Mini-swe-agent runs
# the agent loop; sandboxes + grading happen on Modal.
#
# Usage:
#   ./eval_iter_on_8gpu.sh iter_50    # eval iter_50 LoRA
#   ./eval_iter_on_8gpu.sh iter_100   # eval iter_100 LoRA
#
# Prereqs:
#   - vLLM 0.22+ installed (with --data-parallel-size support)
#   - mini-swe-agent installed: pip install 'mini-swe-agent[swebench]' modal swebench
#   - Modal auth: modal token set --token-id ak-... --token-secret as-...
#   - HF auth: export HF_TOKEN=hf_...
set -euo pipefail

ITER_NAME="${1:-iter_50}"
HF_REPO="${HF_REPO:-ScalingIntelligence/qwen3.6-trace-semantic-logic-precision-v1}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.6-27B}"
RUN_ROOT="${RUN_ROOT:-/workspace/trace_eval}"
WORKERS="${WORKERS:-80}"
PORT="${PORT:-8000}"

LORA_LOCAL="${RUN_ROOT}/loras/${ITER_NAME}"
RUN_DIR="${RUN_ROOT}/runs/${ITER_NAME}"
CONFIG_YAML="${RUN_ROOT}/configs/swebench_${ITER_NAME}.yaml"

mkdir -p "${LORA_LOCAL}" "${RUN_DIR}" "$(dirname "${CONFIG_YAML}")"

# 1. Download the LoRA subfolder from HF (once; ~640MB)
if [[ ! -f "${LORA_LOCAL}/adapter_model.safetensors" ]]; then
  echo "[1/4] downloading ${HF_REPO}/${ITER_NAME} → ${LORA_LOCAL}"
  python3 - <<PYEOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="${HF_REPO}",
    allow_patterns="${ITER_NAME}/*",
    local_dir="${RUN_ROOT}/loras_tmp",
)
import shutil, os
src = "${RUN_ROOT}/loras_tmp/${ITER_NAME}"
dst = "${LORA_LOCAL}"
for fn in os.listdir(src):
    shutil.copy(os.path.join(src, fn), os.path.join(dst, fn))
print("done")
PYEOF
fi

# 2. Write the agent config (points at the LoRA's served name)
cat > "${CONFIG_YAML}" <<EOF
# Matches phase-1 baseline config, model_name swapped to the LoRA.
agent:
  step_limit: 150
  cost_limit: 3.0

environment:
  environment_class: "swerex_modal"
  startup_timeout: 600.0
  deployment_timeout: 3600.0
  runtime_timeout: 3600.0
  install_pipx: true

model:
  model_name: "hosted_vllm/trace_${ITER_NAME}"
  cost_tracking: "ignore_errors"
  model_kwargs:
    api_base: "http://localhost:${PORT}/v1"
    drop_params: true
    temperature: 0.7
    top_p: 0.95
    parallel_tool_calls: true
    max_tokens: 24000
    extra_body:
      top_k: 20
      chat_template_kwargs:
        enable_thinking: true
EOF

# 3. Launch vLLM with TP=4 × DP=2 (two replicas across 8 GPUs)
echo "[2/4] launching vLLM (TP=4, DP=2, 128k context, 10 seqs/replica) on port ${PORT}"
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
nohup vllm serve "${BASE_MODEL}" \
  --host 0.0.0.0 --port "${PORT}" \
  --tensor-parallel-size 4 \
  --data-parallel-size 2 \
  --dtype bfloat16 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 32768 \
  --max-num-seqs 10 \
  --trust-remote-code \
  --served-model-name "${BASE_MODEL}" \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --enable-lora \
  --max-loras 2 \
  --max-lora-rank 32 \
  > "${RUN_DIR}/vllm.log" 2>&1 &
VLLM_PID=$!
echo "vLLM pid=${VLLM_PID}"

# Wait for vLLM ready (~2-5 min)
echo "waiting for vLLM ready..."
until grep -q "Application startup complete" "${RUN_DIR}/vllm.log" 2>/dev/null; do
  sleep 5
done
echo "vLLM ready"

# 4. Register the LoRA adapter
echo "[3/4] registering LoRA adapter 'trace_${ITER_NAME}'"
curl -sf -X POST "http://localhost:${PORT}/v1/load_lora_adapter" \
  -H "Content-Type: application/json" \
  -d "{\"lora_name\":\"trace_${ITER_NAME}\",\"lora_path\":\"${LORA_LOCAL}\"}"
echo

# Sanity: list models
curl -s "http://localhost:${PORT}/v1/models" | python3 -c "
import json,sys
print('models:', [m['id'] for m in json.load(sys.stdin)['data']])
"

# 5. Run mini-swe-agent over SWE-bench Verified
echo "[4/4] running mini-swe-agent over 500 SWE-bench Verified instances"
python3 -m minisweagent.run.benchmarks.swebench \
  -c swebench \
  -c "${CONFIG_YAML}" \
  --subset verified \
  --split test \
  -o "${RUN_DIR}" \
  -w "${WORKERS}"

# 6. preds.json → preds.jsonl
python3 - <<PYEOF
import json
from pathlib import Path
RUN_DIR = Path("${RUN_DIR}")
pj = RUN_DIR / "preds.json"
data = json.loads(pj.read_text())
with open(RUN_DIR / "preds.jsonl", "w") as f:
    for v in data.values():
        f.write(json.dumps(v) + "\n")
print(f"wrote preds.jsonl ({len(data)} preds)")
PYEOF

# 7. Grade on Modal
echo "[grading] swebench harness on Modal"
python3 -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --split test \
  --predictions_path "${RUN_DIR}/preds.jsonl" \
  --max_workers "${WORKERS}" \
  --run_id "trace_${ITER_NAME}_qwen36" \
  --modal true \
  --report_dir "${RUN_DIR}/modal-reports"

echo
echo "==== DONE ===="
echo "Trajectories: ${RUN_DIR}/"
echo "Grading report: ${RUN_DIR}/modal-reports/"
echo "Stop vLLM:  kill ${VLLM_PID}"
