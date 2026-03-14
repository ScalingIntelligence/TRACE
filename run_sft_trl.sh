#!/bin/bash
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
NCCL_P2P_DISABLE=1 \
WANDB_API_KEY=f4ef099e7073d103963e5c986e4f818f5a526ee8 \
accelerate launch --num_processes=6 train_sft_trl.py \
  --sft-data "game_rollouts/structured_data_reasoning-Qwen3-30B-A3B-Instruct-2507-structured-v3-grpo-80-d2-n10000-s0.json,game_rollouts/structured_data_reasoning-Qwen3-30B-A3B-Instruct-2507-structured-v3-grpo-80-d3-n10000-s0.json,game_rollouts/structured_data_reasoning-Qwen3-30B-A3B-Instruct-2507-structured-v3-grpo-80-d4-n10000-s0.json,game_rollouts/tau_tool_calling-Qwen3-30B-A3B-Instruct-2507-toolcalling-v3-grpo-40-Qwen3-30B-A3B-Instruct-2507-toolcalling-v3-grpo-40-n1000-s0-airline.json,game_rollouts/tau_tool_calling-Qwen3-30B-A3B-Instruct-2507-toolcalling-v3-grpo-40-Qwen3-30B-A3B-Instruct-2507-toolcalling-v3-grpo-40-n1000-s0-retail.json" \
  --max-samples-per-file "80,200,120,200,200" \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --lora-rank 32 \
  --lora-alpha 32 \
  --lr 2e-5 \
  --mini-batch-size 1 \
  --gradient-accumulation-steps 2 \
  --pack-sequences
