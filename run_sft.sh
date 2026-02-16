#!/bin/bash
# =============================================================================
# SFT (Supervised Fine-Tuning) Training Pipeline for Qwen3-4B-Instruct
# =============================================================================
# This script:
# 1. Prepares SFT data from tau2-bench trajectories
# 2. Fine-tunes Qwen3-4B-Instruct using LoRA
#
# Uses the same dataset as run_embedding_training.sh for consistency.
#
# Usage: ./run_sft_training.sh
# =============================================================================

set -e  # Exit on error

# =============================================================================
# Conda activation
# =============================================================================
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

# =============================================================================
# GPU Configuration
# =============================================================================
export CUDA_VISIBLE_DEVICES=0

# =============================================================================
# Configuration
# =============================================================================

# Project root
PROJECT_ROOT="/root/games"

# W&B Configuration
WANDB_PROJECT="qwen3-sft"
WANDB_RUN_NAME="sft_tau2bench_$(date +%Y%m%d_%H%M%S)"
USE_WANDB=true  # Set to true to enable W&B logging

# Data paths - tau2-bench results (same as embedding training)
TAU_BENCH_DIR="${PROJECT_ROOT}/tau2-bench/data/tau2/results/final"
SFT_DATA_DIR="${PROJECT_ROOT}/data/sft"

# Input files (all available tau2-bench results - same as embedding training)
INPUT_FILES=(
    "${TAU_BENCH_DIR}/gpt-4.1-2025-04-14_airline_default_gpt-4.1-2025-04-14_4trials.json"
    "${TAU_BENCH_DIR}/gpt-4.1-2025-04-14_retail_default_gpt-4.1-2025-04-14_4trials.json"
    "${TAU_BENCH_DIR}/gpt-4.1-mini-2025-04-14_airline_base_gpt-4.1-2025-04-14_4trials.json"
    "${TAU_BENCH_DIR}/gpt-4.1-mini-2025-04-14_retail_base_gpt-4.1-2025-04-14_4trials.json"
    "${TAU_BENCH_DIR}/o4-mini-2025-04-16_airline_default_gpt-4.1-2025-04-14_4trials.json"
    "${TAU_BENCH_DIR}/o4-mini-2025-04-16_retail_default_gpt-4.1-2025-04-14_4trials.json"
    "${TAU_BENCH_DIR}/claude-3-7-sonnet-20250219_airline_default_gpt-4.1-2025-04-14_4trials.json"
    "${TAU_BENCH_DIR}/claude-3-7-sonnet-20250219_retail_default_gpt-4.1-2025-04-14_4trials.json"
)

# Model settings
MODEL_NAME="Qwen/Qwen3-30B-A3B-Instruct-2507"

# Data preparation settings
MAX_SEQ_LENGTH=10000
TRAIN_VAL_SPLIT=0.1  # 10% for validation
FILTER_SUCCESSFUL=true
MIN_REWARD=0.5
SEED=42

# LoRA settings
USE_LORA=true
LORA_R=64
LORA_ALPHA=128
LORA_DROPOUT=0.05

# Training settings
BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=8  # Effective batch size = 2 * 8 = 16
NUM_EPOCHS=3
LEARNING_RATE=2e-5
WARMUP_RATIO=0.03
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=1.0
LR_SCHEDULER="cosine"

# Logging settings
LOGGING_STEPS=10
EVAL_STEPS=100
SAVE_STEPS=200
SAVE_TOTAL_LIMIT=3

# Checkpoint directory
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints/sft"

# =============================================================================
# Directory Setup
# =============================================================================

cd "$PROJECT_ROOT"

mkdir -p "$SFT_DATA_DIR"
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "logs"

# =============================================================================
# Step 1: Prepare SFT Data
# =============================================================================

echo "============================================================"
echo "Step 1: Preparing SFT Data from tau2-bench trajectories"
echo "============================================================"

TRAIN_FILE="${SFT_DATA_DIR}/train.jsonl"
VAL_FILE="${SFT_DATA_DIR}/val.jsonl"

# Check if SFT data already exists
if [ -f "$TRAIN_FILE" ] && [ -f "$VAL_FILE" ]; then
    echo "SFT data already exists at ${SFT_DATA_DIR}"
    echo "Skipping data preparation..."
else
    echo ""
    echo "Input files:"
    for f in "${INPUT_FILES[@]}"; do
        echo "  - $f"
    done
    echo ""
    echo "Settings:"
    echo "  Train/Val split: ${TRAIN_VAL_SPLIT}"
    if [ "$FILTER_SUCCESSFUL" = true ]; then
        echo "  Filter: reward >= ${MIN_REWARD}"
    fi
    echo ""

    # Build input files argument
    INPUT_ARGS=""
    for f in "${INPUT_FILES[@]}"; do
        INPUT_ARGS="$INPUT_ARGS \"$f\""
    done

    PREP_CMD="python3 preprocessing/prepare_sft_data.py \
        --input $INPUT_ARGS \
        --output-dir \"$SFT_DATA_DIR\" \
        --train-val-split $TRAIN_VAL_SPLIT \
        --seed $SEED"
    
    if [ "$FILTER_SUCCESSFUL" = true ]; then
        PREP_CMD="$PREP_CMD --filter-successful --min-reward $MIN_REWARD"
    fi
    
    eval $PREP_CMD
fi

# Print data statistics
echo ""
echo "=== SFT Data Statistics ==="
if [ -f "${SFT_DATA_DIR}/metadata.json" ]; then
    python3 -c "
import json
with open('${SFT_DATA_DIR}/metadata.json') as f:
    m = json.load(f)
print(f'  Train samples: {m[\"num_train_samples\"]}')
print(f'  Val samples: {m[\"num_val_samples\"]}')
print(f'  Total trajectories: {m[\"total_trajectories\"]}')
"
fi

# =============================================================================
# Step 2: SFT Training
# =============================================================================

echo ""
echo "============================================================"
echo "Step 2: SFT Training on Qwen3-4B-Instruct"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Model: ${MODEL_NAME}"
echo "  Max sequence length: ${MAX_SEQ_LENGTH}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Gradient accumulation: ${GRADIENT_ACCUMULATION_STEPS}"
echo "  Effective batch size: $((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "  Epochs: ${NUM_EPOCHS}"
echo "  Learning rate: ${LEARNING_RATE}"
if [ "$USE_LORA" = true ]; then
    echo "  LoRA rank: ${LORA_R}"
    echo "  LoRA alpha: ${LORA_ALPHA}"
fi
echo "  Checkpoint dir: ${CHECKPOINT_DIR}"
if [ "$USE_WANDB" = true ]; then
    echo "  W&B Project: ${WANDB_PROJECT}"
    echo "  W&B Run: ${WANDB_RUN_NAME}"
fi
echo "============================================================"
echo ""

# Clear CUDA cache
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Build training command
TRAIN_CMD="python3 train/sft_train.py \
    --train-file \"$TRAIN_FILE\" \
    --val-file \"$VAL_FILE\" \
    --model-name \"$MODEL_NAME\" \
    --max-seq-length $MAX_SEQ_LENGTH \
    --output-dir \"$CHECKPOINT_DIR\" \
    --num-epochs $NUM_EPOCHS \
    --batch-size $BATCH_SIZE \
    --gradient-accumulation-steps $GRADIENT_ACCUMULATION_STEPS \
    --learning-rate $LEARNING_RATE \
    --weight-decay $WEIGHT_DECAY \
    --warmup-ratio $WARMUP_RATIO \
    --max-grad-norm $MAX_GRAD_NORM \
    --lr-scheduler-type $LR_SCHEDULER \
    --logging-steps $LOGGING_STEPS \
    --eval-steps $EVAL_STEPS \
    --save-steps $SAVE_STEPS \
    --save-total-limit $SAVE_TOTAL_LIMIT \
    --seed $SEED \
    --bf16"

if [ "$USE_LORA" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --use-lora --lora-r $LORA_R --lora-alpha $LORA_ALPHA --lora-dropout $LORA_DROPOUT"
else
    TRAIN_CMD="$TRAIN_CMD --no-lora"
fi

if [ "$USE_WANDB" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --use-wandb --wandb-project \"$WANDB_PROJECT\" --wandb-run-name \"$WANDB_RUN_NAME\""
fi

# Run training
eval $TRAIN_CMD 2>&1 | tee "logs/sft_training_${WANDB_RUN_NAME}.log"

echo ""
echo "Training complete! Checkpoints saved to: ${CHECKPOINT_DIR}"

