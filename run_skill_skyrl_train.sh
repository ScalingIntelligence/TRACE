#!/usr/bin/env bash
set -x

# =============================================================================
# SkyRL GRPO training for skill games
#
# Layout: colocate_all=true, TP=2 vLLM (3 engines), FSDP across all GPUs
#         cpu_offload=true required for 30B model on 80GB GPUs
#         No KL/ref model (pure GRPO) to halve fwd_logprobs time
#
# Usage:
#   CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 NUM_GPUS=6 bash run_skill_skyrl_train.sh
#   GAME=multistep_task bash run_skill_skyrl_train.sh
# =============================================================================

# -- Configurable via env vars --
: "${GAME:=multistep_task}"
: "${DATA_DIR:=$HOME/data/skill_${GAME}}"
: "${NUM_GPUS:=6}"
: "${MODEL:=Qwen/Qwen3-30B-A3B-Instruct-2507}"
: "${LOGGER:=wandb}"
: "${NUM_TRAIN_SEEDS:=10000}"
: "${NUM_VAL_SEEDS:=1000}"

# Auto-extract WANDB_API_KEY from .netrc if not set
if [ -z "$WANDB_API_KEY" ] && [ -f "$HOME/.netrc" ]; then
  export WANDB_API_KEY=$(awk '/api.wandb.ai/{getline; getline; print $2}' "$HOME/.netrc")
fi
: "${CKPT_DIR:=$HOME/ckpts/skill_${GAME}}"
: "${RUN_NAME:=skill_${GAME}_grpo}"

# -- GRPO structure --
N_SAMPLES=16                  # group_size (rollouts per prompt)
TRAIN_BATCH_SIZE=$((16 * NUM_GPUS))   # groups_per_batch scaled to GPU count
POLICY_MINI_BATCH_SIZE=${TRAIN_BATCH_SIZE}

cd "$(dirname "$0")"

# -- Generate dataset if not already present --
if [ ! -f "${DATA_DIR}/train.parquet" ] || [ ! -f "${DATA_DIR}/validation.parquet" ]; then
  echo "Generating dataset for game=${GAME} in ${DATA_DIR} ..."
  python skill_skyrl_dataset.py \
    --games ${GAME} \
    --num_train_seeds ${NUM_TRAIN_SEEDS} \
    --num_val_seeds ${NUM_VAL_SEEDS} \
    --output_dir "${DATA_DIR}"
fi

python skill_skyrl_train.py \
  data.train_data="['${DATA_DIR}/train.parquet']" \
  data.val_data="['${DATA_DIR}/validation.parquet']" \
  environment.env_class=skill_game \
  \
  trainer.strategy=fsdp2 \
  trainer.placement.colocate_all=true \
  trainer.placement.policy_num_gpus_per_node=${NUM_GPUS} \
  trainer.placement.critic_num_gpus_per_node=${NUM_GPUS} \
  trainer.placement.ref_num_gpus_per_node=${NUM_GPUS} \
  \
  generator.inference_engine.num_engines=1 \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=false \
  generator.inference_engine.remote_urls="['${VLLM_BASE_URL:-http://localhost:8080}']" \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.inference_engine.gpu_memory_utilization=0.9 \
  generator.inference_engine.enforce_eager=false \
  generator.inference_engine.enable_prefix_caching=true \
  generator.inference_engine.enable_chunked_prefill=true \
  generator.inference_engine.engine_init_kwargs='{"max_model_len": 2560}' \
  \
  trainer.policy.model.path=${MODEL} \
  trainer.policy.model.lora.rank=16 \
  trainer.policy.model.lora.alpha=16 \
  \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.use_kl_in_reward=false \
  trainer.algorithm.eps_clip_low=0.2 \
  trainer.algorithm.eps_clip_high=0.2 \
  trainer.algorithm.grpo_norm_by_std=true \
  trainer.algorithm.zero_variance_filter=true \
  \
  trainer.policy.optimizer_config.lr=3e-5 \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  trainer.policy.optimizer_config.weight_decay=0.01 \
  \
  generator.n_samples_per_prompt=${N_SAMPLES} \
  generator.max_turns=1 \
  generator.batched=true \
  generator.sampling_params.max_generate_length=512 \
  generator.sampling_params.temperature=1.0 \
  \
  trainer.epochs=1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=${TRAIN_BATCH_SIZE} \
  trainer.policy_mini_batch_size=${POLICY_MINI_BATCH_SIZE} \
  trainer.micro_train_batch_size_per_gpu=8 \
  trainer.micro_forward_batch_size_per_gpu=16 \
  trainer.max_prompt_length=2048 \
  trainer.eval_batch_size=512 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.ckpt_interval=10 \
  trainer.seed=42 \
  trainer.bf16=true \
  trainer.flash_attn=true \
  trainer.gradient_checkpointing=true \
  trainer.use_sample_packing=true \
  \
  trainer.logger=${LOGGER} \
  trainer.project_name=skill_games \
  trainer.run_name=${RUN_NAME} \
  trainer.resume_mode=null \
  trainer.ckpt_path=${CKPT_DIR} \
  "$@"
