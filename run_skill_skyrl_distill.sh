#!/usr/bin/env bash
set -x

# =============================================================================
# SkyRL On-Policy Distillation — multi-skill with per-skill teacher routing
#
# On-policy distillation (GLM-5 §3.5, Agarwal et al.):
#   1. Student generates on-policy trajectories across skill games
#   2. Per-skill teacher LoRAs (external vLLM server) grade each token
#   3. Per-token reverse KL drives a clipped surrogate loss
#   4. Student policy is updated, repeat
#
# Teacher setup (run BEFORE this script):
#   # Launch teacher vLLM server with all skill LoRA adapters:
#   CUDA_VISIBLE_DEVICES=6 python -m vllm.entrypoints.openai.api_server \
#     --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
#     --enable-lora --max-loras 5 --max-lora-rank 32 \
#     --lora-modules \
#       sdr_teacher=/path/to/structured_data_reasoning_adapter \
#       mt_teacher=/path/to/multistep_task_adapter \
#       tc_teacher=/path/to/tau_tool_calling_adapter \
#     --port 9100 --max-model-len 8192 \
#     --gpu-memory-utilization 0.9 --dtype bfloat16
#
# User LLM (needed for tau_tool_calling, adversarial_policy):
#   CUDA_VISIBLE_DEVICES=7 python -m vllm.entrypoints.openai.api_server \
#     --model Qwen/Qwen3-4B-Instruct-2507 --port 8001 \
#     --max-model-len 4096 --gpu-memory-utilization 0.9 --dtype bfloat16
#
# Run distillation (GPUs 0-5 for training):
#   bash run_skill_skyrl_distill.sh
#
# Single-teacher mode (one teacher for all skills):
#   TEACHER_ADAPTERS="my_teacher_lora" bash run_skill_skyrl_distill.sh
#
# Override games:
#   GAMES="structured_data_reasoning multistep_task" bash run_skill_skyrl_distill.sh
# =============================================================================

# -- Configurable --
: "${GAMES:=structured_data_reasoning multistep_task tau_tool_calling}"
: "${GAME_TAG:=distill_3skill}"
: "${DATA_DIR:=$HOME/data/skill_${GAME_TAG}}"
: "${NUM_GPUS:=6}"                          # GPUs 6=teacher, 7=user LLM
: "${MODEL:=Qwen/Qwen3-30B-A3B-Instruct-2507}"
: "${LOGGER:=wandb}"
: "${CKPT_DIR:=$HOME/ckpts/skill_${GAME_TAG}}"
: "${RUN_NAME:=skill_${GAME_TAG}}"

# -- Teacher configuration --
: "${TEACHER_VLLM_URL:=http://localhost:9100}"
: "${TEACHER_MODEL:=Qwen/Qwen3-30B-A3B-Instruct-2507}"
# Per-skill adapter mapping: "skill_name=lora_name,..."
# Use "__default__" or a single name for single-teacher mode
: "${TEACHER_ADAPTERS:=structured_data_reasoning=sdr_teacher,multistep_task=mt_teacher,tau_tool_calling=tc_teacher}"
: "${TEACHER_CONCURRENCY:=32}"

# -- User LLM (for tau_tool_calling, adversarial_policy) --
: "${USER_LLM_BASE_URL:=http://localhost:8001/v1}"
: "${USER_LLM_MODEL:=Qwen/Qwen3-4B-Instruct-2507}"

# -- Dataset seeds --
: "${NUM_TRAIN_SEEDS:=500}"
: "${NUM_VAL_SEEDS:=50}"

# Export teacher config as env vars (read by SkillDistillationTrainer)
export TEACHER_VLLM_URL
export TEACHER_MODEL
export TEACHER_ADAPTERS
export TEACHER_CONCURRENCY

# Auto-extract WANDB_API_KEY from .netrc if not set
if [ -z "$WANDB_API_KEY" ] && [ -f "$HOME/.netrc" ]; then
  export WANDB_API_KEY=$(awk '/api.wandb.ai/{getline; getline; print $2}' "$HOME/.netrc")
fi

# Restrict training to GPUs 0..NUM_GPUS-1
GPU_LIST=$(seq -s, 0 $((NUM_GPUS - 1)))
export CUDA_VISIBLE_DEVICES=${GPU_LIST}

# -- Distillation structure --
# group_size=1 equivalent: n_samples_per_prompt=1 (each prompt gets 1 rollout)
# For distillation, dense token-level signal replaces group-relative advantages
N_SAMPLES=1
TRAIN_BATCH_SIZE=$((NUM_GPUS * 8))     # 48 for 6 GPUs
POLICY_MINI_BATCH_SIZE=${TRAIN_BATCH_SIZE}

cd "$(dirname "$0")"

# -- Generate dataset if not already present --
if [ ! -f "${DATA_DIR}/train.parquet" ] || [ ! -f "${DATA_DIR}/validation.parquet" ]; then
  echo "Generating dataset for games='${GAMES}' in ${DATA_DIR} ..."
  USER_LLM_BASE_URL=${USER_LLM_BASE_URL} \
  USER_LLM_MODEL=${USER_LLM_MODEL} \
  python skill_skyrl_dataset.py \
    --games ${GAMES} \
    --num_train_seeds ${NUM_TRAIN_SEEDS} \
    --num_val_seeds ${NUM_VAL_SEEDS} \
    --output_dir "${DATA_DIR}"
fi

python skill_skyrl_distill.py \
  data.train_data="['${DATA_DIR}/train.parquet']" \
  data.val_data="['${DATA_DIR}/validation.parquet']" \
  environment.env_class=skill_game \
  environment.skyrl_gym.skill_game.user_llm_base_url=${USER_LLM_BASE_URL} \
  environment.skyrl_gym.skill_game.user_llm_model=${USER_LLM_MODEL} \
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
  generator.inference_engine.engine_init_kwargs='{"max_model_len": 8192}' \
  \
  trainer.policy.model.path=${MODEL} \
  trainer.policy.model.lora.rank=32 \
  trainer.policy.model.lora.alpha=32 \
  \
  trainer.algorithm.advantage_estimator=no_op \
  trainer.algorithm.policy_loss_type=importance_sampling \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.use_kl_in_reward=false \
  trainer.algorithm.eps_clip_low=0.2 \
  trainer.algorithm.eps_clip_high=0.2 \
  trainer.algorithm.grpo_norm_by_std=false \
  trainer.algorithm.zero_variance_filter=false \
  \
  trainer.policy.optimizer_config.lr=1e-5 \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  trainer.policy.optimizer_config.weight_decay=0.01 \
  \
  generator.n_samples_per_prompt=${N_SAMPLES} \
  generator.max_turns=30 \
  generator.batched=false \
  generator.use_conversation_multi_turn=true \
  generator.sampling_params.max_generate_length=1024 \
  generator.sampling_params.temperature=1.0 \
  generator.max_input_length=7168 \
  \
  trainer.epochs=20 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=${TRAIN_BATCH_SIZE} \
  trainer.policy_mini_batch_size=${POLICY_MINI_BATCH_SIZE} \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.micro_forward_batch_size_per_gpu=8 \
  trainer.max_prompt_length=2048 \
  trainer.eval_batch_size=256 \
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
  trainer.project_name=skill_distillation \
  trainer.run_name=${RUN_NAME} \
  trainer.resume_mode=null \
  trainer.ckpt_path=${CKPT_DIR} \
  "$@"
