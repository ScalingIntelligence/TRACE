# GRPO Trainer — Detailed Explanation

## What is GRPO?

GRPO (Group Relative Policy Optimization) is a reinforcement learning algorithm that trains language models without a learned value function. Instead of a value head estimating "how good is this state?", GRPO compares multiple attempts at the **same problem** and asks "which attempt was better?"

This is the algorithm used by the rl4rl team (ThinkingMachines) that showed strong transfer from RL training to downstream tasks. Our implementation (`train_grpo.py`) adapts their approach to work with our game environments.

## Architecture: Inner/Outer Loop

### Outer Loop — Diversity across problems
Each iteration samples `groups_per_batch` different game seeds (default: 16). Each seed defines a unique initial game state (e.g., a specific poker hand, a specific Liar's Dice configuration). The outer loop provides **task diversity** — the model can't just memorize one game state.

### Inner Loop — Comparison within a problem
For each seed, we play `group_size` independent games (default: 8) from that same initial state. Because we sample with temperature > 0, the model explores different strategies each time. Some attempts will succeed, others will fail. The key insight: we compute advantages **within each group**:

```
advantage_i = reward_i - mean(rewards_in_group)
```

This centering makes the signal **scale-invariant** — it doesn't matter if rewards are {-1, +1} or {0, 100}. All that matters is the relative ranking within the group.

### Concrete example (8 games, same poker seed)

| Game | Model's strategy | Outcome | Reward | Advantage |
|------|-----------------|---------|--------|-----------|
| 0 | Aggressive bluff | Win | +1 | +1 - 0.25 = +0.75 |
| 1 | Conservative fold | Lose | -1 | -1 - 0.25 = -1.25 |
| 2 | Calculated bet | Win | +1 | +1 - 0.25 = +0.75 |
| 3 | Early fold | Lose | -1 | -1 - 0.25 = -1.25 |
| 4 | Bluff + call | Win | +1 | +1 - 0.25 = +0.75 |
| 5 | Random play | Lose | -1 | -1 - 0.25 = -1.25 |
| 6 | Aggressive | Win | +1 | +1 - 0.25 = +0.75 |
| 7 | Passive check | Lose | -1 | -1 - 0.25 = -1.25 |

Mean reward = 0.25. The model gets positive gradient for strategies 0,2,4,6 (which worked) and negative gradient for 1,3,5,7 (which didn't). Crucially, this is for **this specific hand** — the model learns "for this hand, aggressive play is better than conservative play."

### Two-player games
For 2-player games (poker, dice), advantages are computed **per player** within each group. Player 0's actions are compared against other Player 0 outcomes in the same group, and similarly for Player 1. This prevents the zero-sum structure from washing out the signal.

## Key Differences from PPO Trainer

| Aspect | PPO (`train_ppo.py`) | GRPO (`train_grpo.py`) |
|--------|---------------------|----------------------|
| **Value function** | Learned value head (distorts backbone) | None — group-relative comparison |
| **Advantage** | return - value_prediction, globally normalized | reward - group_mean (within-group) |
| **Loss function** | PPO clip: min(ratio*adv, clip(ratio)*adv) | Importance sampling: -E[ratio*adv] (optionally clipped) |
| **Constrained decoding** | Removed (was available) | Never — model generates freely |
| **LoRA rank** | 16 (from Config) | 32 (default, configurable) |
| **Learning rate** | 1e-6 | 1e-5 (10x higher) |
| **Backbone distortion** | Value head forces game-state encoding | No value head — backbone stays general |
| **KL penalty** | None | Optional (--kl-coef, default 0.01) |
| **Gradient targets** | Action tokens (normalized by length) | All completion tokens (NOT normalized) |

## Loss Function Details

### Default: Pure Importance Sampling
```
loss = -E[ratio * advantage]
```
where `ratio = exp(new_logp - old_logp)`.

This is the simplest policy gradient estimator. It's unbiased but can have high variance when the ratio drifts far from 1. The KL penalty helps control this.

### Optional: Clipped Surrogate (--use-clipping)
```
loss = -E[min(ratio * advantage, clip(ratio, 1-eps, 1+eps) * advantage)]
```
Same as PPO's clipped objective. Use this if training is unstable with pure importance sampling.

### KL Penalty
```
kl_loss = kl_coef * E[new_logp - old_logp]
```
Penalizes the model for deviating too far from the rollout policy. This acts as a soft trust region, preventing the model from making too-large updates and forgetting base capabilities. Set `--kl-coef 0` to disable.

## Why No Value Head?

The PPO value head is a linear layer that maps the backbone's hidden state → scalar value estimate. Training this value head pushes the backbone to encode **game-state evaluation** — "am I winning at poker?" This is useful for poker but **hurts transfer** to other tasks (math, customer service, etc.).

GRPO doesn't need a value function because the group-relative comparison provides the baseline. The backbone representations stay general-purpose, which is why rl4rl-trained models transfer well.

## Why Not Normalize by Action Length?

The PPO trainer normalizes logprobs by action length: `logp = sum(logp_tokens) / num_tokens`. This makes a 1-token action `[bet]` and an 80-token reasoning chain have the same scale.

GRPO does NOT normalize by default (`--normalize-by-len` is off). This means longer, better reasoning chains get proportionally more gradient signal. The model is incentivized to generate useful reasoning, not just pick an action token. If you're training on games with very short actions, you can enable normalization, but for reasoning-heavy tasks it should stay off.

## vLLM Integration

The GRPO trainer uses the exact same vLLM infrastructure as PPO:

1. **Startup**: Connects to vLLM server(s) via `VLLM_BASE_URL` / `VLLM_BASE_URLS`
2. **Each iteration**:
   - Saves current LoRA adapter to disk
   - Hot-reloads the adapter into vLLM via `/v1/load_lora_adapter`
   - Collects rollouts using vLLM's batched generation (fast!)
   - Trains on collected data using the local model
3. **Evaluation**: Uses vLLM for fast batched eval (vs base, tau2-bench, math)

If no vLLM server is available, falls back to HF local generation (slower but works).

## File Structure

```
train_grpo.py           # Main GRPO trainer (this file)
├── collect_grpo_rollouts()  # Group-based rollout collection
├── compute_group_advantages()  # Within-group advantage centering
├── parse_grpo_args()    # CLI argument parsing
└── main()               # Training loop orchestration

# Reused from existing codebase:
config.py               # Shared config (model name, seq length, etc.)
game_registry.py        # GameEnv protocol, GameSpec, all registered games
inference.py            # vLLM backend, HF local backend
ppo.py                  # Shared utilities: JSONLLogger, pad_to_device,
                        #   build_prompt_plus_action, logprob_action_tokens
evaluation.py           # evaluate_vs_base, evaluate_math, evaluate_tau2_bench
```

## Training Commands

### Basic (with vLLM, recommended)
```bash
# Start vLLM server first (in separate terminal):
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --enable-lora --max-lora-rank 64 --gpu-memory-utilization 0.9

# Then run training:
VLLM_BASE_URL=http://localhost:8000 python train_grpo.py \
    --game adversarial_policy \
    --group-size 8 \
    --groups-per-batch 16 \
    --lr 1e-5 \
    --lora-rank 32 \
    --temperature 0.7 \
    --kl-coef 0.01
```

### With clipped objective
```bash
VLLM_BASE_URL=http://localhost:8000 python train_grpo.py \
    --game adversarial_policy \
    --group-size 8 \
    --groups-per-batch 16 \
    --use-clipping --clip-eps 0.2
```

### Conservative (smaller groups, more stability)
```bash
VLLM_BASE_URL=http://localhost:8000 python train_grpo.py \
    --game kuhn_poker \
    --group-size 4 \
    --groups-per-batch 8 \
    --lr 5e-6 \
    --lora-rank 16 \
    --kl-coef 0.05
```

### Multi-GPU (multiple vLLM servers)
```bash
VLLM_BASE_URLS=http://gpu1:8000,http://gpu2:8000 python train_grpo.py \
    --game progressive_service_agent \
    --group-size 8 \
    --groups-per-batch 32
```

### With evaluations enabled
```bash
VLLM_BASE_URL=http://localhost:8000 python train_grpo.py \
    --game adversarial_policy \
    --eval-every 25 \
    --math-eval-every 50 \
    --tau2-eval-every 50
```

### HF local (no vLLM, slower)
```bash
python train_grpo.py \
    --game kuhn_poker \
    --group-size 4 \
    --groups-per-batch 4 \
    --lr 1e-5
```

## Hyperparameter Guide

| Parameter | Default | What it controls | Tuning advice |
|-----------|---------|-----------------|---------------|
| `--group-size` | 8 | Rollouts per seed | Higher = cleaner signal but slower. 4-16 is typical. |
| `--groups-per-batch` | 16 | Seeds per iteration | Higher = more diversity. 8-32 is typical. |
| `--lr` | 1e-5 | Learning rate | Start here. Lower (1e-6) if unstable, higher (5e-5) if too slow. |
| `--lora-rank` | 32 | LoRA capacity | 32 is good. 16 for smaller models, 64 for larger. |
| `--kl-coef` | 0.01 | KL penalty strength | 0.01-0.1. Higher = more conservative. 0 to disable. |
| `--temperature` | 0.7 | Sampling diversity | Higher = more exploration in inner loop. 0.6-1.0 typical. |
| `--use-clipping` | off | PPO-style clip | Enable if training is unstable. |
| `--epochs` | 1 | Passes over data | 1 is usually fine. 2-3 if groups are small. |
| `--mini-batch-size` | 4 | GPU batch size | Increase if you have VRAM headroom. |

## What to Monitor

- `grpo/avg_abs_advantage`: Should be > 0 and stable. If it drops to 0, the group isn't differentiating (try higher temperature or more group_size).
- `grpo/approx_kl`: Should stay small (< 0.1). If it spikes, reduce LR or increase --kl-coef.
- `grpo/ratio_mean`: Should hover around 1.0. If it drifts far, the policy is changing too fast.
- `env/invalid_move_rate`: Should decrease over training (model learns valid action format).
- `grpo/avg_reward`: Should increase over training.
