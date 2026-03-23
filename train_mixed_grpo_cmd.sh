#!/usr/bin/env bash
set -euo pipefail

# Mixed GRPO training: SDR + multistep + precondition + tau_tool_calling (4 games)
# Warm-start from precondition-v1-40 LoRA
# Agent rollout model: vLLM (base Qwen3-30B) on port 8080
# User simulator: vLLM (base Qwen3-30B) on port 9000

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
NCCL_P2P_DISABLE=1 \
WANDB_API_KEY=f4ef099e7073d103963e5c986e4f818f5a526ee8 \
VLLM_BASE_URLS=http://localhost:8080 \
VLLM_MODEL=tarsur909/precondition-v1-40 \
VLLM_TIMEOUT_S=2000 \
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
VLLM_RPC_TIMEOUT=2000 \
PYTHONUNBUFFERED=1 \
torchrun --nproc_per_node=6 --master-port 29501 train_grpo_optimized.py \
    --games "structured_data_reasoning:0.25,multistep_task:0.25,precondition_check:0.25,tau_tool_calling:0.25" \
    --model tarsur909/precondition-v1-40 \
    --compact-tools \
    --lr 5e-6 \
    --temperature-range "0.4,0.7,1.0" \
    --save-every 5 \
    --groups-per-batch 16 \
    --group-size 8 \
    --dynamic-sampling-max-batches 3 \
    --user-llm-url http://localhost:9000/v1 \
    --user-llm-model "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --sft-coef 0.5