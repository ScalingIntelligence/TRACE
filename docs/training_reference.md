# Training Reference

Complete CLI parameter reference for GRPO trainers with SFT+RL joint training.

## Quick Start

### RL-only training (default)
```bash
VLLM_BASE_URL=http://localhost:8000 python train_grpo.py \
    --game adversarial_policy --group-size 8 --groups-per-batch 16
```

### SFT+RL joint training
```bash
VLLM_BASE_URL=http://localhost:8000 python train_grpo.py \
    --game adversarial_policy --group-size 8 --groups-per-batch 16 \
    --sft-data "path/to/tau2_airline.json,path/to/tau2_retail.json" \
    --sft-coef 0.1 --sft-per-step 2
```

### Optimized trainer (multi-GPU)
```bash
VLLM_BASE_URL=http://localhost:8000 torchrun --nproc_per_node=4 \
    train_grpo_optimized.py --game adversarial_policy \
    --group-size 8 --groups-per-batch 16 \
    --sft-data "path/to/eval1.json,path/to/eval2.json" \
    --sft-coef 0.1 --sft-per-step 2
```

## SFT+RL Design

### Motivation
GRPO training on adversarial policy adherence teaches the model to follow policies but lacks exposure to full tau2-bench task format and correct multi-turn tool-calling flows. Joint SFT+RL training combines:
- **RL (GRPO)**: Policy adherence signal from adversarial micro-environment
- **SFT**: Correct multi-turn behavior from successful tau2-bench trajectories

### Data Source
Tau2-bench eval JSON files containing successful trajectories (reward=1.0). Each assistant turn in a successful trajectory becomes one SFT sample.

### Loss Design
Each training step computes RL loss + SFT loss in separate forward passes:
```
loss = policy_loss + kl_loss + sft_coef * sft_loss
```

SFT loss uses `normalize_by_len=True` (per-token NLL, typically 2-4), scaled by `sft_coef` (default 0.1) giving effective contribution of ~0.2-0.4, comparable to RL loss magnitude.

### Buffer Refresh
After each tau2-bench evaluation, newly generated successful trajectories are automatically added to the SFT buffer (deduped by file+task_id).

## CLI Parameters

### Common Parameters (both trainers)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--game` | `kuhn_poker` | Game environment from game_registry |
| `--model` | Config.MODEL_NAME | HuggingFace model name |
| `--group-size` | 16 | Rollouts per seed (inner loop) |
| `--groups-per-batch` | 4/8 | Different seeds per iteration (outer loop) |
| `--lora-rank` | 16 | LoRA rank |
| `--lora-alpha` | 16 | LoRA alpha scaling |
| `--lr` | 1e-5 | Learning rate |
| `--epochs` | 1 | Training epochs per iteration |
| `--mini-batch-size` | 4 | Mini-batch size for gradient updates |
| `--max-grad-norm` | 1.0 | Max gradient norm for clipping |
| `--use-clipping` | True (opt) / False (base) | PPO-style clipped surrogate |
| `--clip-eps` | 0.2 | Clipping epsilon |
| `--kl-coef` | 0.0 | KL penalty against base model |
| `--temperature` | 1.0 (opt) / 0.7 (base) | Sampling temperature |
| `--temperature-range` | None | Comma-separated temps for per-game variation |
| `--prefix-ratio` | 0.4 | Auto-play lookup prefix probability |
| `--compact-tools` | False | Strip descriptions from tool schemas |
| `--normalize-by-len` | False | Normalize logprobs by action length |
| `--resume` | None | Checkpoint directory to resume from |
| `--save-every` | 5 | Save checkpoint every N iterations |
| `--eval-every` | 100000 | Eval vs base every N iterations |
| `--eval-games` | 100000 | Games for eval vs base |
| `--math-eval-every` | 100000 | Math benchmark eval interval |
| `--tau2-eval-every` | 0 | Tau2-bench eval interval (0=disabled) |
| `--adversarial-ratio` | 0.6 | Adversarial vs cooperative scenario ratio |

### SFT Joint Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--sft-data` | None | Comma-separated paths to tau2-bench eval JSONs |
| `--sft-coef` | 0.1 | SFT loss weight |
| `--sft-per-step` | 2 | SFT samples per RL mini-batch step |

### User LLM Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--user-llm-url` | None | OpenAI-compatible base URL for user LLM |
| `--user-llm-model` | None | Model name for user LLM |
| `--user-llm-temperature` | 0.7 | Temperature for user LLM |
| `--user-llm-max-tokens` | 1024 | Max tokens for user LLM |

### Optimized Trainer Only

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--stats-chunk-size` | 4 | Chunk size for logprob computation |
| `--filter-info-turns` | True | Filter info-gathering tool calls |
| `--tool-result-max-chars` | 200 | Max chars for truncated old tool results |
| `--filter-constant-groups` | True | Remove groups with identical rewards |
| `--dist-lr-scale` | 1.0 | LR scale for distributed training |

## Logged Metrics

| Metric | Description |
|--------|-------------|
| `grpo/policy_loss` | Average RL policy loss |
| `grpo/sft_loss` | Average SFT loss (per-token NLL) |
| `grpo/sft_buffer_size` | Number of SFT samples in buffer |
| `grpo/kl_base_loss` | KL penalty against base model |
| `grpo/approx_kl_rollout` | Approx KL from rollout policy |
| `grpo/approx_kl_base` | Approx KL from base model |
| `grpo/clip_frac` | Fraction of clipped ratios |
| `grpo/ratio_mean` | Mean importance sampling ratio |
| `grpo/avg_reward` | Average reward |
| `grpo/avg_advantage` | Average advantage |
