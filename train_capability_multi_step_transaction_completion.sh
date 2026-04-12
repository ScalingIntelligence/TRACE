#!/usr/bin/env bash
# Train GRPO on the capability_multi_step_transaction_completion game.
# (Game is auto-registered when game_registry imports
#  capability_multi_step_transaction_completion_game.py.)
set -euo pipefail

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
export NCCL_P2P_DISABLE=1
export WANDB_API_KEY=f4ef099e7073d103963e5c986e4f818f5a526ee8
export WANDB_PROJECT=games
export VLLM_BASE_URLS=http://localhost:8080
export VLLM_MODEL=Qwen/Qwen3.5-2B
export VLLM_TIMEOUT_S=2000
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export PYTHONUNBUFFERED=1

torchrun --nproc_per_node=7 --master-port=29501 -m train \
    --game capability_multi_step_transaction_completion \
    --model Qwen/Qwen3.5-2B \
    --group-size 8 \
    --groups-per-batch 8 \
    "$@"
