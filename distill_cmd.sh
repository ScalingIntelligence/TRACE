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
# CUDA_VISIBLE_DEVICES=7 \
#   NCCL_P2P_DISABLE=1 \
#   WANDB_API_KEY=f4ef099e7073d103963e5c986e4f818f5a526ee8 \
#   WANDB_PROJECT=games \
#   VLLM_BASE_URLS=http://localhost:9090 \
#   VLLM_MODEL=tarsur909/merged-tc-sd-ms \
#   VLLM_TIMEOUT_S=2000 \
#   VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
#   PYTHONUNBUFFERED=1 \
#   torchrun --nproc_per_node=1 --master-port 29501 train_distill.py \
#     --teacher-url "structured_data_reasoning=http://localhost:9000,multistep_task=http://localhost:9002,tau_tool_calling=http://localhost:9001" \
#     --teacher-model "structured_data_reasoning=tarsur909/structured_data_reasoning-v5-40,multistep_task=tarsur909/multistep_task-v5-30,tau_tool_calling=tarsur909/tau_tool_calling-v5-40" \
#     --teacher-concurrency 16 \
#     --loss-type ppo_surrogate \
#     --games "structured_data_reasoning:0.33,multistep_task:0.33,tau_tool_calling:0.34" \
#     --model tarsur909/merged-tc-sd-ms \
#     --compact-tools \
#     --groups-per-batch 32 \
#     --temperature 1.0 \
#     --lr 1e-5 \
#     --mini-batch-size 2 \
#     --stats-chunk-size 2 \
#     --save-every 5 --user-llm-url http://localhost:9004/v1 --user-llm-model "Qwen/Qwen3-30B-A3B-Instruct-2507"

# ---- Mode D: Multiple teacher LoRAs on one multi-LoRA vLLM server (uncomment to use) ----
# Like Mode B but uses per-skill LoRA adapters on a single vLLM server
# instead of separate vLLM servers per teacher.
#   - vLLM port 9000: student base model (rollout generation)
#   - vLLM port 9001: teacher base model + LoRA adapters (logprob queries)
#   - vLLM port 9002: user LLM
#   - LoRA names must match --lora-modules on the teacher vLLM server
#
# Teacher vLLM server launch (port 9001):
  # python -m vllm.entrypoints.openai.api_server \
  #   --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  #   --enable-lora --max-loras 4 --max-lora-rank 16 \
  #   --lora-modules \
  #     sdr_teacher=/home/ubuntu/.cache/huggingface/structured_data_reasoning/grpo_ckpt_iter_40_20260308_080312 \
  #     mt_teacher=/home/ubuntu/.cache/huggingface/multistep_task/grpo_ckpt_iter_30_20260318_182920 \
  #     tc_teacher=/home/ubuntu/.cache/huggingface/tau_tool_calling/grpo_ckpt_iter_40_20260311_030533 \
  #     pre_teacher=/home/ubuntu/.cache/huggingface/precondition/grpo_ckpt_iter_40_20260319_035848 \
  #   --port 9001 --max-model-len 32000 \
  #   --gpu-memory-utilization 0.9 --dtype bfloat16
#
CUDA_VISIBLE_DEVICES=4,5,6,7 \
  NCCL_P2P_DISABLE=1 \
  WANDB_API_KEY=f4ef099e7073d103963e5c986e4f818f5a526ee8 \
  WANDB_PROJECT=games \
  VLLM_BASE_URLS=http://localhost:9000 \
  VLLM_MODEL=tarsur909/merged-tc-sd-ms-pre \
  VLLM_TIMEOUT_S=2000 \
  VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
  PYTHONUNBUFFERED=1 \
  torchrun --nproc_per_node=4 --master-port 29501 train_distill.py \
    --teacher-url http://localhost:9001 \
    --teacher-model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --teacher-adapters "structured_data_reasoning=sdr_teacher,multistep_task=mt_teacher,tau_tool_calling=tc_teacher,precondition_check=pre_teacher" \
    --teacher-concurrency 16 \
    --loss-type ppo_surrogate \
    --games "structured_data_reasoning:0.25,multistep_task:0.25,tau_tool_calling:0.25,precondition_check:0.25" \
    --model tarsur909/merged-tc-sd-ms-pre \
    --compact-tools \
    --groups-per-batch 32 \
    --temperature 1.0 \
    --lr 1e-5 \
    --mini-batch-size 2 \
    --stats-chunk-size 2 \
    --save-every 5 --user-llm-url http://localhost:9002/v1 --user-llm-model "Qwen/Qwen3-30B-A3B-Instruct-2507"

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
