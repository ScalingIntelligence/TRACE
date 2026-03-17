#!/usr/bin/env bash
set -x

# =============================================================================
# SkyRL GRPO training for skill games (structured_data_reasoning)
#
# Hyperparameters based on train_grpo_optimized.py defaults, adjusted per
# SkyRL recipes (examples/train/gsm8k, lora, dapo):
#
#   train_grpo_optimized.py          SkyRL (this script)
#   ───────────────────────          ─────────────────────
#   group_size=16                →   n_samples_per_prompt=16
#   groups_per_batch=8           →   train_batch_size=128
#   lr=1e-5                      →   lr=3e-5 (LoRA needs higher, per SkyRL lora recipe)
#   lora_rank=16, alpha=16       →   rank=32, alpha=32 (SkyRL lora recipe minimum)
#   kl_coef=0                    →   use_kl_loss=true, coef=0.001 (cheap in SkyRL; ref model free)
#   clip_eps=0.2                 →   eps_clip_low=0.2, eps_clip_high=0.2
#   filter_constant_groups       →   zero_variance_filter=true
#   epochs=1                     →   epochs=20 (SkyRL outer loop), update_epochs_per_batch=1
#   temperature=1.0              →   same
#   max_gen_tokens=512           →   same
#
# Usage:
#   # 1. Generate dataset (one-time):
#   conda activate sky_games
#   python skill_skyrl_dataset.py \
#     --games structured_data_reasoning \
#     --num_train_seeds 2000 --num_val_seeds 200 \
#     --output_dir ~/data/skill_structured_data
#
#   # 2. Train (8 GPUs):
#   bash run_skill_skyrl_train.sh
#
#   # Override defaults:
#   NUM_GPUS=4 MODEL=Qwen/Qwen2.5-7B-Instruct bash run_skill_skyrl_train.sh
# =============================================================================

# -- Configurable via env vars --
: "${DATA_DIR:=$HOME/data/skill_structured_data}"
: "${NUM_GPUS:=8}"
: "${MODEL:=Qwen/Qwen3-4B-Instruct-2507}"
: "${LOGGER:=wandb}"

# Auto-extract WANDB_API_KEY from .netrc if not set
if [ -z "$WANDB_API_KEY" ] && [ -f "$HOME/.netrc" ]; then
  export WANDB_API_KEY=$(awk '/api.wandb.ai/{getline; getline; print $2}' "$HOME/.netrc")
fi
: "${CKPT_DIR:=$HOME/ckpts/skill_structured_data}"
: "${RUN_NAME:=skill_structured_data_grpo}"

# -- GRPO structure --
N_SAMPLES=16                  # group_size (rollouts per prompt)
TRAIN_BATCH_SIZE=128          # groups_per_batch=8 (unique prompts per iter)
POLICY_MINI_BATCH_SIZE=128    # SkyRL standard: train_batch_size / 1-4x

cd "$(dirname "$0")"

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
  generator.inference_engine.num_engines=${NUM_GPUS} \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  generator.inference_engine.enforce_eager=false \
  generator.inference_engine.enable_prefix_caching=true \
  generator.inference_engine.enable_chunked_prefill=true \
  generator.inference_engine.engine_init_kwargs='{"max_model_len": 4096}' \
  \
  trainer.policy.model.path=${MODEL} \
  trainer.policy.model.lora.rank=32 \
  trainer.policy.model.lora.alpha=32 \
  \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.001 \
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
  trainer.epochs=20 \
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
