#!/bin/bash
# Step 3: Train Qwen3-30B-A3B on ADP data with LoRA
#
# Uses 8 GPUs, matching our expert LoRA config (r=16, alpha=16).
# Effective batch size = 1 * 16 (grad accum) * 8 (GPUs) = 128
#
# Expected: ~2-4 hours for 1 epoch depending on dataset size.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/train_config.yaml"

echo "=== ADP Baseline Training ==="
echo "Config: ${CONFIG}"
echo "GPUs: 8x A100-80GB"
echo ""

# Set environment
export WANDB_API_KEY="${WANDB_API_KEY:-f4ef099e7073d103963e5c986e4f818f5a526ee8}"
export WANDB_PROJECT="games"
export NCCL_P2P_DISABLE=1

# Run training with all 8 GPUs
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
    llamafactory-cli train "${CONFIG}"

echo ""
echo "=== Training complete ==="
echo "Adapter saved to: /home/ubuntu/.cache/huggingface/adp_baseline/sft_lora"
echo ""
echo "Next: Upload adapter and run evaluation"
echo "  python upload_adapter.py   # optional: push to HF"
echo "  bash eval.sh               # evaluate on tau-bench"


#   CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
#   WANDB_API_KEY=f4ef099e7073d103963e5c986e4f818f5a526ee8 \                                                                                                                                                                                                                                                                                                                                      
#   torchrun --nproc_per_node=6 train_sft_adp.py \                                                                                                                                                                                                                                                                                                                                                
#       --data /home/ubuntu/hangook/games/evals/benchmarks/tau2_bench_eval/scripts/adp_baseline/data/adp_sft_train.jsonl \                                                                                                                                                                                                                                                                        
#       --max-seq-length 8192 \                                                                                                                                                                                                                                                                                                                                                                   
#       --num-epochs 1 \                                                                                                                                                                                                                                                                                                                                                                          
#       --lr 1e-5 \                                                                                                                                                                                                                                                                                                                                                                               
#       --mini-batch-size 1 \                                                                                                                                                                                                                                                                                                                                                                     
#       --gradient-accumulation-steps 16 --save-every 5


  CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
  WANDB_API_KEY=f4ef099e7073d103963e5c986e4f818f5a526ee8 \
  torchrun --nproc_per_node=6 train_sft_adp.py \
      --data /home/ubuntu/hangook/games/evals/benchmarks/tau2_bench_eval/scripts/adp_baseline/data/adp_sft_train.jsonl \
      --max-seq-length 8192 \
      --num-epochs 1 \
      --lr 1e-5 \
      --mini-batch-size 1 \
      --gradient-accumulation-steps 16 --save-every 5



  CUDA_VISIBLE_DEVICES=6,7 \
  WANDB_API_KEY=f4ef099e7073d103963e5c986e4f818f5a526ee8 \
  torchrun --nproc_per_node=2 train_sft_adp.py \
      --data /home/ubuntu/hangook/games/evals/benchmarks/tau2_bench_eval/scripts/adp_baseline/data/adp_sft_train.jsonl \
      --max-seq-length 8192 \
      --num-epochs 1 \
      --lr 1e-5 \
      --mini-batch-size 4 \
      --gradient-accumulation-steps 8 \
      --save-steps 10 \
      --max-samples 10000 \
      --log-every 2