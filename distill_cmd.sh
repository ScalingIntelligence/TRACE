#!/bin/bash
# On-policy distillation (GLM-5 §3.5 / SkyRL-style)
#
# Three teacher modes (uncomment the one you need):
#
# Mode A: Single teacher model on one vLLM server
#   - vLLM port 8080: student base model (rollout generation)
#   - vLLM port 9000: teacher full model (logprob queries)
#
# Mode B: Multiple teacher models on separate vLLM servers (per-skill)
#   - vLLM port 8080: student base model (rollout generation)
#   - vLLM port 9000: teacher for structured_data_reasoning
#   - vLLM port 9001: teacher for tau_tool_calling
#   - vLLM port 9002: teacher for multistep_task
#
# Mode C: Local LoRA adapter as teacher (no extra vLLM server)
#   - vLLM port 8080: student base model (rollout generation)
#
# 5 training GPUs (3,4,5,6,7)

# ---- Mode A: Single teacher on one vLLM server ----
# CUDA_VISIBLE_DEVICES=3,4,5,6,7 \
#   NCCL_P2P_DISABLE=1 \
#   WANDB_API_KEY=f4ef099e7073d103963e5c986e4f818f5a526ee8 \
#   WANDB_PROJECT=games \
#   VLLM_BASE_URLS=http://localhost:8080 \
#   VLLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 \
#   VLLM_TIMEOUT_S=2000 \
#   VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
#   PYTHONUNBUFFERED=1 \
#   torchrun --nproc_per_node=5 --master-port 29501 train_distill.py \
#     --teacher-url http://localhost:9000 \
#     --teacher-model tarsur909/Qwen3-30B-A3B-Instruct-2507-structured-10 \
#     --teacher-concurrency 16 \
#     --loss-type reverse_kl \
#     --game structured_data_reasoning \
#     --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
#     --compact-tools \
#     --groups-per-batch 64 \
#     --temperature 1.0 \
#     --lr 1e-5 \
#     --mini-batch-size 2 \
#     --stats-chunk-size 2 \
#     --save-every 5

# ---- Mode B: Multiple teachers on separate vLLM servers (uncomment to use) ----
CUDA_VISIBLE_DEVICES=5,6,7 \
  NCCL_P2P_DISABLE=1 \
  WANDB_API_KEY=f4ef099e7073d103963e5c986e4f818f5a526ee8 \
  WANDB_PROJECT=games \
  VLLM_BASE_URLS=http://localhost:8080 \
  VLLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 \
  VLLM_TIMEOUT_S=2000 \
  VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
  PYTHONUNBUFFERED=1 \
  torchrun --nproc_per_node=3 --master-port 29501 train_distill.py \
    --teacher-url "structured_data_reasoning=http://localhost:9000,multistep_task=http://localhost:9002,tau_tool_calling=http://localhost:9001" \
    --teacher-model "structured_data_reasoning=tarsur909/Qwen3-30B-A3B-Instruct-2507-structured-10,multistep_task=tarsur909/Qwen3-30B-A3B-Instruct-2507-multistep-task-10,tau_tool_calling=tarsur909/Qwen3-30B-A3B-Instruct-2507-toolcalling-v3-grpo-40" \
    --teacher-concurrency 16 \
    --loss-type ppo_surrogate \
    --games "structured_data_reasoning:0.33,multistep_task:0.33,tau_tool_calling:0.34" \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --compact-tools \
    --groups-per-batch 32 \
    --temperature 1.0 \
    --lr 1e-5 \
    --mini-batch-size 2 \
    --stats-chunk-size 2 \
    --save-every 5 --user-llm-url http://localhost:9004/v1 --user-llm-model "Qwen/Qwen3-30B-A3B-Instruct-2507"

# ---- Mode C: Local LoRA adapter as teacher (uncomment to use) ----
# CUDA_VISIBLE_DEVICES=3,4,5,6,7 \
#   NCCL_P2P_DISABLE=1 \
#   WANDB_API_KEY="${WANDB_API_KEY}" \
#   WANDB_PROJECT=games \
#   VLLM_BASE_URLS=http://localhost:8080 \
#   VLLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 \
#   VLLM_TIMEOUT_S=2000 \
#   VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
#   PYTHONUNBUFFERED=1 \
#   torchrun --nproc_per_node=5 --master-port 29501 train_distill.py \
#     --teacher-adapter /path/to/grpo_ckpt_iter_X \
#     --loss-type reverse_kl \
#     --games "structured_data_reasoning:0.34,tau_tool_calling:0.33,multistep_task:0.33" \
#     --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
#     --compact-tools \
#     --groups-per-batch 64 \
#     --temperature 1.0 \
#     --lr 1e-5 \
#     --mini-batch-size 2 \
#     --stats-chunk-size 2 \
#     --save-every 5
