#!/usr/bin/env bash
set -x

# =============================================================================
# SkyRL GRPO training — 3-game mix
#
# Games: structured_data_reasoning (0.34), multistep_task (0.33), tau_tool_calling (0.33)
#
# Multi-turn games (multistep_task, tau_tool_calling) require:
#   - max_turns=30, batched=false, use_conversation_multi_turn=true
#   - A user LLM server for tau_tool_calling (set USER_LLM_BASE_URL)
#   - GPU 7 reserved for user LLM → training on GPUs 0-6 (7 GPUs)
#
# Prerequisites:
#   1. Launch user LLM server on GPU 7:
#      CUDA_VISIBLE_DEVICES=7 python -m vllm.entrypoints.openai.api_server \
#        --model Qwen/Qwen3-4B-Instruct-2507 --port 8001 \
#        --max-model-len 4096 --gpu-memory-utilization 0.9 --dtype bfloat16
#
#   2. Generate dataset:
#      USER_LLM_BASE_URL=http://localhost:8001/v1 \
#      USER_LLM_MODEL=Qwen/Qwen3-4B-Instruct-2507 \
#      python skill_skyrl_dataset.py \
#        --games structured_data_reasoning multistep_task tau_tool_calling \
#        --num_train_seeds 500 --num_val_seeds 50 \
#        --output_dir ~/data/skill_mix_3game
#
#   3. Train:
#      bash run_skill_skyrl_mix3.sh
# =============================================================================

# -- Configurable --
: "${DATA_DIR:=$HOME/data/skill_mix_3game}"
: "${NUM_GPUS:=7}"                          # GPU 7 reserved for user LLM
: "${MODEL:=Qwen/Qwen3-4B-Instruct-2507}"
: "${LOGGER:=wandb}"
: "${CKPT_DIR:=$HOME/ckpts/skill_mix_3game}"
: "${RUN_NAME:=skill_mix3_grpo}"
: "${USER_LLM_BASE_URL:=http://localhost:8001/v1}"
: "${USER_LLM_MODEL:=Qwen/Qwen3-4B-Instruct-2507}"

# Auto-extract WANDB_API_KEY from .netrc if not set
if [ -z "$WANDB_API_KEY" ] && [ -f "$HOME/.netrc" ]; then
  export WANDB_API_KEY=$(awk '/api.wandb.ai/{getline; getline; print $2}' "$HOME/.netrc")
fi

# Restrict training to GPUs 0-6 (GPU 7 = user LLM server)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6

# -- GRPO structure --
N_SAMPLES=8                   # lower than 16 to keep batch manageable with multi-turn
TRAIN_BATCH_SIZE=56           # 8 groups × 7 (divisible by NUM_GPUS=7)
POLICY_MINI_BATCH_SIZE=56     # = train_batch_size (single mini-batch per step)

cd "$(dirname "$0")"

python skill_skyrl_train.py \
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
  trainer.project_name=skill_games \
  trainer.run_name=${RUN_NAME} \
  trainer.resume_mode=null \
  trainer.ckpt_path=${CKPT_DIR} \
  "$@"
