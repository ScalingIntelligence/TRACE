#!/bin/bash
set -e

SFT_DATA="game_rollouts/game_rollouts/structured_data_reasoning-Qwen3-30B-A3B-Instruct-2507-structured-v3-grpo-80-d2-n10000-s0.json,game_rollouts/game_rollouts/structured_data_reasoning-Qwen3-30B-A3B-Instruct-2507-structured-v3-grpo-80-d3-n10000-s0.json,game_rollouts/game_rollouts/structured_data_reasoning-Qwen3-30B-A3B-Instruct-2507-structured-v3-grpo-80-d4-n10000-s0.json,game_rollouts/game_rollouts/tau_tool_calling-Qwen3-30B-A3B-Instruct-2507-toolcalling-v3-grpo-40-Qwen3-30B-A3B-Instruct-2507-toolcalling-v3-grpo-40-n1000-s0-airline.json,game_rollouts/game_rollouts/tau_tool_calling-Qwen3-30B-A3B-Instruct-2507-toolcalling-v3-grpo-40-Qwen3-30B-A3B-Instruct-2507-toolcalling-v3-grpo-40-n1000-s0-retail.json"
MAX_SAMPLES_PER_FILE="40,100,60,100,100"
MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
GUMBEL_K=100
GUMBEL_OUTPUT="gumbel_precomputed_small.pt"

# # ---- Step 1: Precompute Gumbel top-k teacher outputs (multi-GPU) ----
# echo "=== Precomputing Gumbel top-k teacher outputs ==="
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# NCCL_P2P_DISABLE=1 \
# python -m torch.distributed.run --nproc_per_node=8 precompute_gumbel.py \
#   --sft-data "$SFT_DATA" \
#   --max-samples-per-file "$MAX_SAMPLES_PER_FILE" \
#   --model "$MODEL" \
#   --gumbel-k "$GUMBEL_K" \
#   --batch-size 4 \
#   --output "$GUMBEL_OUTPUT"

# ---- Step 2: SFT training with Gumbel KL distillation ----
echo "=== Starting SFT training with KL distillation ==="
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NCCL_P2P_DISABLE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_API_KEY=f4ef099e7073d103963e5c986e4f818f5a526ee8 \
python -m torch.distributed.run --nproc_per_node=8 train_sft.py \
  --sft-data "$SFT_DATA" \
  --max-samples-per-file "$MAX_SAMPLES_PER_FILE" \
  --model "$MODEL" \
  --lora-rank 32 \
  --lora-alpha 32 \
  --lr 2e-5 \
  --mini-batch-size 1 \
  --gradient-accumulation-steps 2 \
  --pack-sequences \
  --gumbel-data "$GUMBEL_OUTPUT" \
  --gumbel-k "$GUMBEL_K" \
  --kl-coef 0.1
