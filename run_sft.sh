#!/bin/bash
set -e

SFT_DATA="game_rollouts/multistep_task_revised-self_expert-n300_airline.json,game_rollouts/multistep_task_revised-self_expert-n300_retail.json,game_rollouts/precondition_check_revised-self_expert-n300_airline.json,game_rollouts/precondition_check_revised-self_expert-n300_retail.json,game_rollouts/structured_data_reasoning_revised-self_expert-n300.json,game_rollouts/tau_tool_calling_revised-self_expert-n300_airline.json,game_rollouts/tau_tool_calling_revised-self_expert-n300_retail.json"
MODEL="tarsur909/precondition-v1-40"

# ---- SFT training (no Gumbel KL distillation) ----
echo "=== Starting SFT training ==="
CUDA_VISIBLE_DEVICES=0,2,4,5,6,7 \
NCCL_P2P_DISABLE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_API_KEY=f4ef099e7073d103963e5c986e4f818f5a526ee8 \
python -m torch.distributed.run --nproc_per_node=6 train_sft.py \
  --sft-data "$SFT_DATA" \
  --max-samples-per-file "50,50,50,50,100,50,50" \
  --model "$MODEL" \
  --lora-rank 32 \
  --lora-alpha 32 \
  --lr 1e-6 \
  --mini-batch-size 1 \
  --gradient-accumulation-steps 2 \
  --pack-sequences
