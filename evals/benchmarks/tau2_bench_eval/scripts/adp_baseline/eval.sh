#!/bin/bash
# Step 4: Evaluate ADP baseline on tau-bench (airline + retail)
#
# Two evaluation modes:
#   A) Merged model (merge LoRA into base, serve as single model)
#   B) LoRA adapter on vLLM (serve base + adapter directly)
#
# We use Mode B for consistency with how we evaluate other LoRA adapters.
#
# Prerequisites:
#   - vLLM server running with LoRA support on port 8080
#   - Separate user LLM server on port 9000 (base model)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="/home/ubuntu/hangook/games/evals/benchmarks/tau2_bench_eval"
ADAPTER_PATH="/home/ubuntu/.cache/huggingface/adp_baseline/sft_lora"

echo "============================================"
echo "  ADP Baseline Evaluation on tau-bench"
echo "============================================"
echo ""

# -----------------------------------------------------------------------
# Option A: Merge LoRA into base and upload, then serve merged model
# -----------------------------------------------------------------------
# Uncomment this block if you prefer to merge first:
#
# echo "Merging LoRA adapter into base model..."
# python /home/ubuntu/hangook/games/merge_and_push_RL.py  # (edit MERGE_JOBS first)
#
# Then launch vLLM with the merged model:
# CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
#     --model tarsur909/adp-baseline-merged \
#     --port 8080 --max-model-len 32000 --dtype bfloat16 \
#     --gpu-memory-utilization 0.9 --tensor-parallel-size 2

# -----------------------------------------------------------------------
# Option B: Serve base model with LoRA adapter (recommended)
# -----------------------------------------------------------------------
echo "=== Server Setup Instructions ==="
echo ""
echo "Terminal 1 — Agent vLLM (base + ADP LoRA adapter):"
echo "  CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \\"
echo "      --model Qwen/Qwen3-30B-A3B-Instruct-2507 \\"
echo "      --enable-lora --max-loras 1 --max-lora-rank 16 \\"
echo "      --lora-modules adp_baseline=${ADAPTER_PATH} \\"
echo "      --port 8080 --max-model-len 32000 --dtype bfloat16 \\"
echo "      --gpu-memory-utilization 0.9"
echo ""
echo "Terminal 2 — User vLLM (base model, no LoRA):"
echo "  CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \\"
echo "      --model Qwen/Qwen3-30B-A3B-Instruct-2507 \\"
echo "      --port 9000 --max-model-len 32000 --dtype bfloat16 \\"
echo "      --gpu-memory-utilization 0.9"
echo ""
echo "Then run this script again with --run flag."
echo ""

if [ "$1" != "--run" ]; then
    echo "Pass --run to actually execute the evaluation."
    exit 0
fi

# -----------------------------------------------------------------------
# Run evaluation
# -----------------------------------------------------------------------

echo "Running airline evaluation..."
cd "${EVAL_DIR}"
python main.py --config <(cat <<'YAML'
domain: airline
agent_llm: vllm://adp_baseline
agent: null

agent_llm_args:
  temperature: 0.0
  max_context_length: 32000
  tokenizer_model: openai/Qwen/Qwen3-30B-A3B-Instruct-2507

user_llm: vllm://Qwen/Qwen3-30B-A3B-Instruct-2507
user_llm_args:
  temperature: 0.0
  max_context_length: 32000
  tokenizer_model: openai/Qwen/Qwen3-30B-A3B-Instruct-2507

user: null
num_trials: 1
max_steps: 50
max_concurrency: 1
seed: 42
verbose: true
save_to: adp-baseline-airline

vllm:
  base_url: http://localhost:8080/v1
  user_base_urls:
    - http://localhost:9000/v1
YAML
)

echo ""
echo "Running retail evaluation..."
python main.py --config <(cat <<'YAML'
domain: retail
agent_llm: vllm://adp_baseline
agent: null

agent_llm_args:
  temperature: 0.0
  max_context_length: 32000
  tokenizer_model: openai/Qwen/Qwen3-30B-A3B-Instruct-2507

user_llm: vllm://Qwen/Qwen3-30B-A3B-Instruct-2507
user_llm_args:
  temperature: 0.0
  max_context_length: 32000
  tokenizer_model: openai/Qwen/Qwen3-30B-A3B-Instruct-2507

user: null
num_trials: 1
max_steps: 50
max_concurrency: 1
seed: 42
verbose: true
save_to: adp-baseline-retail

vllm:
  base_url: http://localhost:8080/v1
  user_base_urls:
    - http://localhost:9000/v1
YAML
)

echo ""
echo "=== Evaluation complete ==="
echo "Results saved to:"
echo "  ${EVAL_DIR}/data/simulations/adp-baseline-airline.json"
echo "  ${EVAL_DIR}/data/simulations/adp-baseline-retail.json"
