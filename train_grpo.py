#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimization) trainer for game environments.

Implements the rl4rl-style GRPO training loop adapted for the games repo:

  Inner loop (per seed):  Sample `group_size` rollouts from the SAME game seed.
                          All rollouts start from an identical initial state;
                          diversity comes purely from temperature sampling.

  Outer loop (per batch): Iterate over `groups_per_batch` different seeds.

Key differences from the PPO trainer (train_ppo.py):
  - No value head:       Advantages come from group-relative reward centering,
                         not a learned value baseline. This keeps backbone
                         representations general instead of encoding game-state
                         evaluation.
  - Group-relative advantages: "Which of my N attempts at THIS problem was best?"
                         Scale-invariant, works across different reward ranges.
  - Full text training:  No constrained decoding. The model generates freely
                         (including chain-of-thought reasoning). Gradient flows
                         through ALL completion tokens, not just action tokens.
  - Importance sampling: Default loss is -E[ratio * advantage] (pure importance
                         sampling). Optional PPO-style clipping is available.
  - KL penalty:          Optional penalty to prevent catastrophic forgetting.
  - Higher capacity:     Default LoRA rank 32 (vs 4 in PPO), higher LR.

Works with all GameEnv-protocol games in the registry. Supports vLLM inference
backend exactly like the PPO trainer (hot LoRA reload each iteration).

Usage:
    # With vLLM server (recommended):
    VLLM_BASE_URL=http://localhost:8000 python train_grpo.py \\
        --game adversarial_policy --group-size 8 --groups-per-batch 16

    # With HF local generation:
    python train_grpo.py --game adversarial_policy --group-size 4 --groups-per-batch 8
"""
from unsloth import FastLanguageModel

from loguru import logger as _loguru_logger
_loguru_logger.remove()
_loguru_logger.disable("tau2")  # suppress tau2-bench internal debug logs (console-only, not seen by model)

import argparse
import json
import os
import random
import time
import torch
import torch.nn.functional as F
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from config import Config, setup_environment, autocast_ctx
from game_registry import get_game_spec, list_game_names, GameSpec
from inference import (
    InferenceBackend,
    HFLocalBackend,
    init_inference_backend,
    messages_for_game,
    tools_for_game,
    build_prompt_text,
    generate_completion,
)
from ppo import (
    JSONLLogger,
    pad_to_device,
    build_prompt_plus_action,
    logprob_action_tokens,
)
from evaluation import evaluate_tau2_bench
from sft_buffer import SFTBuffer

try:
    import wandb
except Exception:
    wandb = None

try:
    import trackio
except Exception:
    trackio = None


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class GRPOSample:
    """A single turn from a game rollout, tagged with group membership."""
    prompt_msgs: list          # Chat-template messages for this turn
    completion_text: str       # Full model output (thinking + action)
    player_id: int             # Which player produced this action
    reward: float              # Terminal reward for this player
    group_id: int              # Group index (same seed → same group)
    game_id: int               # Unique game identifier
    tools: Optional[list] = None  # Tool schemas for structured games (train→eval alignment)
    game_name: str = ""        # Name of the game that produced this sample


# ============================================================================
# Within-group difficulty variation for multistep_task
# ============================================================================

def multistep_difficulty_schedule(group_size: int) -> List[Optional[int]]:
    """Assign max_ops values for each game in a multistep_task group.

    Creates within-group difficulty variation:
      - 25% of games: max_ops=1 (easiest)
      - 25% of games: max_ops=2 (medium)
      - 50% of games: max_ops=None (full difficulty)

    This ensures non-constant rewards even when the model produces
    deterministic actions, because easier variants are more likely
    to succeed than harder ones.
    """
    schedule: List[Optional[int]] = []
    n_easy = max(1, group_size // 4)
    n_medium = max(1, group_size // 4)
    for i in range(group_size):
        if i < n_easy:
            schedule.append(1)
        elif i < n_easy + n_medium:
            schedule.append(2)
        else:
            schedule.append(None)
    return schedule


# ============================================================================
# Group-based rollout collection
# ============================================================================

def collect_grpo_rollouts(
    *,
    backend: InferenceBackend,
    tokenizer,
    game_spec: GameSpec,
    groups_per_batch: int,
    group_size: int,
    temperature: float,
    max_new_tokens: int,
    base_seed: int,
    logger: JSONLLogger,
    env_kwargs: Optional[Dict[str, Any]] = None,
    temperature_range: Optional[List[float]] = None,
    prefix_ratio: float = 0.0,
    compact_tools: bool = False,
    group_assignments: Optional[List[Tuple[GameSpec, Dict[str, Any]]]] = None,
    use_step_rewards: bool = True,
) -> Tuple[List[GRPOSample], Dict[str, float]]:
    """
    Collect rollouts with the GRPO inner/outer loop structure.

    Creates `groups_per_batch` groups, each containing `group_size` games
    played from the SAME random seed.  Within a group, games differ only
    because of temperature sampling — the model explores different strategies
    from an identical starting position.

    Args:
        temperature_range: If set, each game gets a random temperature from
            this list (e.g. [0.5, 0.7, 1.0]) instead of the single `temperature`.
        prefix_ratio: Probability (0-1) of auto-playing lookup prefix per group
            (adversarial_policy game only). 0.0 = never, 1.0 = always.
        compact_tools: If True, use compressed tool schemas (descriptions stripped)
            for both generation and training. Reduces prompt length by ~60%.
        group_assignments: Optional per-group (GameSpec, env_kwargs) overrides.
            When set, each group uses its own game/kwargs instead of the global
            game_spec/env_kwargs. Length must equal groups_per_batch.

    Returns:
        samples:  List of GRPOSample with group membership tagged.
        metrics:  Dict of collection statistics.
    """
    rng = random.Random(base_seed)
    env_kwargs = env_kwargs or {}

    # Build per-group assignments (backward compat: single game for all groups)
    if group_assignments is None:
        group_assignments = [(game_spec, env_kwargs)] * groups_per_batch
    assert len(group_assignments) == groups_per_batch

    # ---- Outer loop: one seed per group ----
    group_seeds = [rng.randint(0, 2**31 - 1) for _ in range(groups_per_batch)]
    total_games = groups_per_batch * group_size

    samples: List[GRPOSample] = []
    invalid_games = 0
    total_turns = 0
    extraction_failures = 0
    p0_reward_sum = 0.0
    response_lengths: List[int] = []

    if backend.supports_batch():
        # --- Batch mode (vLLM) ---
        envs = []
        env_game_specs: List[GameSpec] = []  # which GameSpec each env uses
        g_ids: List[int] = []        # group_id per env
        gm_ids: List[int] = []       # game_id per env
        # Each step tuple: (msgs, act, pid, completion, tools)
        episode_steps: List[List[Tuple[list, Optional[str], int, str, Optional[list]]]] = []

        difficulties: List[Optional[str]] = []
        game_temps: List[float] = []       # per-game temperature
        prefix_turns: List[int] = []       # per-game prefix count
        for g_idx, g_seed in enumerate(group_seeds):
            g_spec, g_kwargs = group_assignments[g_idx]

            # Per-group prefix decision (same for all rollouts in group)
            use_prefix = False
            if g_spec.name == "adversarial_policy" and prefix_ratio > 0:
                use_prefix = rng.random() < prefix_ratio

            # Compute multistep difficulty schedule once per group
            ms_schedule = None
            if g_spec.name in ("multistep_task", "multistep_task_revised"):
                ms_schedule = multistep_difficulty_schedule(group_size)

            for s_idx in range(group_size):
                env = g_spec.make_env(**g_kwargs)
                # Assign random difficulty for adversarial_policy game
                difficulty = None
                if g_spec.name == "adversarial_policy":
                    difficulty = rng.choice(["easy", "medium", "hard"])
                    env.reset(g_seed, user_difficulty=difficulty)
                elif ms_schedule is not None:
                    env.reset(g_seed, max_ops=ms_schedule[s_idx])
                else:
                    env.reset(g_seed)  # Same seed within group!

                # Auto-play prefix lookups if enabled for this group
                n_prefix = 0
                if use_prefix and hasattr(env, "auto_play_prefix"):
                    n_prefix = env.auto_play_prefix()

                envs.append(env)
                env_game_specs.append(g_spec)
                g_ids.append(g_idx)
                gm_ids.append(base_seed * 1_000_000 + g_idx * 1000 + s_idx)
                episode_steps.append([])
                difficulties.append(difficulty)

                # Assign per-game temperature
                if temperature_range:
                    game_temps.append(rng.choice(temperature_range))
                else:
                    game_temps.append(temperature)
                prefix_turns.append(n_prefix)

        # Play all games in parallel
        while True:
            active = [i for i, e in enumerate(envs) if not e.done]
            if not active:
                break

            prompts: List[str] = []
            meta: List[Tuple[int, int, str, List[str], list, Optional[list]]] = []

            for i in active:
                env = envs[i]
                pid = env.current_player
                obs = env.observe(pid)
                legal = env.legal_actions()
                msgs = messages_for_game(pid, obs, env_game_specs[i], env=env)
                tools = tools_for_game(env, compact=compact_tools)
                prompts.append(build_prompt_text(tokenizer, msgs, tools=tools))
                meta.append((i, pid, obs, legal, msgs, tools))

            t0 = time.time()

            # Group active games by (game_spec, temperature) for sub-batch generation
            # Different games have different stop_sequences and max_gen_tokens
            gen_groups: Dict[Tuple[str, float], List[Tuple[int, str, GameSpec]]] = {}
            for j, (i, pid, obs, legal, msgs, tools) in enumerate(meta):
                t = game_temps[i]
                key = (env_game_specs[i].name, t)
                if key not in gen_groups:
                    gen_groups[key] = []
                gen_groups[key].append((j, prompts[j], env_game_specs[i]))

            completions: List[Optional[str]] = [None] * len(prompts)
            for (gs_name, t), items in gen_groups.items():
                batch_prompts = [p for _, p, _ in items]
                gs = items[0][2]  # All items in group share same GameSpec
                results = backend.generate(
                    batch_prompts,
                    temperature=t,
                    max_new_tokens=gs.max_gen_tokens,
                    game_spec=gs,
                )
                for k, (j, _, _) in enumerate(items):
                    completions[j] = results[k]

            t1 = time.time()

            # Phase 1: Extract actions + bookkeeping (sequential)
            step_data = []
            for j, (i, pid, obs, legal, msgs, tools) in enumerate(meta):
                completion = completions[j]
                act = env_game_specs[i].extract_action(completion, legal)
                if act is None:
                    extraction_failures += 1

                response_lengths.append(len(completion))
                episode_steps[i].append((msgs, act, pid, completion, tools))
                total_turns += 1
                step_data.append((i, pid, legal, act, completion))

            # Phase 2: Step all environments in parallel (I/O-bound user LLM calls)
            with ThreadPoolExecutor(max_workers=len(step_data)) as executor:
                futures = [executor.submit(envs[i].step, act) for i, _, _, act, _ in step_data]
                for f in futures:
                    f.result()

            # Phase 3: Log results (sequential for ordered writes)
            for i, pid, legal, act, completion in step_data:
                logger.log({
                    "type": "step",
                    "game_id": gm_ids[i],
                    "group_id": g_ids[i],
                    "player_id": pid,
                    "legal_actions": legal,
                    "action": act,
                    "completion": completion[:500],
                    "illegal_move": act is None,
                    "timestamp": time.time(),
                })

        # Convert finished games to samples
        for i, env in enumerate(envs):
            invalid_games += 1 if env.invalid_player is not None else 0
            p0_reward_sum += float(env.rewards.get(0, 0.0))

            log_entry = {
                "type": "game_end",
                "game": env_game_specs[i].name,
                "game_id": gm_ids[i],
                "group_id": g_ids[i],
                "rewards": env.rewards,
                "invalid_player": env.invalid_player,
                "temperature": game_temps[i],
                "prefix_turns": prefix_turns[i],
                "timestamp": time.time(),
            }
            if difficulties[i] is not None:
                log_entry["user_difficulty"] = difficulties[i]
            logger.log(log_entry)

            # Use per-step rewards for credit assignment when available and enabled,
            # fall back to terminal reward for games that don't support it.
            step_rewards = getattr(env, '_step_rewards', None) if use_step_rewards else None
            terminal_reward = float(env.rewards.get(0, 0.0))

            for step_idx, (msgs, act, pid, completion, tools) in enumerate(episode_steps[i]):
                if step_rewards is not None and step_idx < len(step_rewards):
                    reward = step_rewards[step_idx]
                else:
                    reward = terminal_reward
                samples.append(GRPOSample(
                    prompt_msgs=msgs,
                    completion_text=completion,
                    player_id=pid,
                    reward=reward,
                    group_id=g_ids[i],
                    game_id=gm_ids[i],
                    tools=tools,
                    game_name=env_game_specs[i].name,
                ))

    else:
        # --- Sequential mode (HF local) ---
        for g_idx, g_seed in enumerate(group_seeds):
            g_spec, g_kwargs = group_assignments[g_idx]

            # Per-group prefix decision
            use_prefix = False
            if g_spec.name == "adversarial_policy" and prefix_ratio > 0:
                use_prefix = rng.random() < prefix_ratio

            # Compute multistep difficulty schedule once per group
            ms_schedule = None
            if g_spec.name in ("multistep_task", "multistep_task_revised"):
                ms_schedule = multistep_difficulty_schedule(group_size)

            for s_idx in range(group_size):
                game_id = base_seed * 1_000_000 + g_idx * 1000 + s_idx
                env = g_spec.make_env(**g_kwargs)
                # Assign random difficulty for adversarial_policy game
                if g_spec.name == "adversarial_policy":
                    difficulty = rng.choice(["easy", "medium", "hard"])
                    env.reset(g_seed, user_difficulty=difficulty)
                elif ms_schedule is not None:
                    env.reset(g_seed, max_ops=ms_schedule[s_idx])
                else:
                    env.reset(g_seed)

                # Auto-play prefix lookups if enabled
                n_prefix = 0
                if use_prefix and hasattr(env, "auto_play_prefix"):
                    n_prefix = env.auto_play_prefix()

                # Assign per-game temperature
                game_temp = rng.choice(temperature_range) if temperature_range else temperature

                ep_steps: List[Tuple[list, Optional[str], int, str, Optional[list]]] = []
                while not env.done:
                    pid = env.current_player
                    obs = env.observe(pid)
                    legal = env.legal_actions()

                    completion = generate_completion(
                        backend.model,
                        backend.tokenizer,
                        pid,
                        obs,
                        temperature=game_temp,
                        max_new_tokens=max_new_tokens,
                        device=backend.device,
                        game_spec=g_spec,
                        env=env,
                        compact_tools=compact_tools,
                    )

                    act = g_spec.extract_action(completion, legal)
                    if act is None:
                        extraction_failures += 1

                    tools = tools_for_game(env, compact=compact_tools)
                    response_lengths.append(len(completion))
                    ep_steps.append((messages_for_game(pid, obs, g_spec, env=env), act, pid, completion, tools))
                    env.step(act)
                    total_turns += 1

                invalid_games += 1 if env.invalid_player is not None else 0
                p0_reward_sum += float(env.rewards.get(0, 0.0))

                step_rewards = getattr(env, '_step_rewards', None) if use_step_rewards else None
                terminal_reward = float(env.rewards.get(0, 0.0))

                for step_idx, (msgs, act, pid, completion, tools) in enumerate(ep_steps):
                    if step_rewards is not None and step_idx < len(step_rewards):
                        reward = step_rewards[step_idx]
                    else:
                        reward = terminal_reward
                    samples.append(GRPOSample(
                        prompt_msgs=msgs,
                        completion_text=completion,
                        player_id=pid,
                        reward=reward,
                        group_id=g_idx,
                        game_id=game_id,
                        tools=tools,
                        game_name=g_spec.name,
                    ))

    # Metrics
    resp_len_mean = (sum(response_lengths) / len(response_lengths)) if response_lengths else 0.0

    metrics = {
        "env/total_games": total_games,
        "env/total_samples": len(samples),
        "env/invalid_move_rate": extraction_failures / max(1, total_turns),
        "env/turns_per_game_mean": total_turns / max(1, total_games),
        "env/avg_reward_p0": p0_reward_sum / max(1, total_games),
        "env/response_len_chars_mean": resp_len_mean,
    }
    return samples, metrics


# ============================================================================
# GRPO advantage computation
# ============================================================================

def compute_group_advantages(samples: List[GRPOSample]) -> torch.Tensor:
    """
    Compute group-relative advantages — the core GRPO mechanism.

    For each (group_id, player_id) pair, center rewards around the group mean:

        advantage_i = reward_i - mean(rewards in my group)

    The group mean is computed per-GAME (one reward per game_id, using the
    last turn's reward which equals the terminal reward). This ensures each
    game contributes equally regardless of turn count — preserving backward
    compatibility with terminal-only reward games.

    With per-step rewards, individual samples have different reward values
    than the per-game terminal reward, creating per-turn credit assignment:
    correct tool calls get positive advantage (step_reward > terminal_mean),
    wrong calls get negative advantage (step_reward < terminal_mean).

    For 2-player games, advantages are computed separately per player within
    each group, so player 0 and player 1 have independent baselines.
    """
    advantages = torch.zeros(len(samples), dtype=torch.float32)

    # Collect per-game rewards, grouped by (group_id, player_id).
    # Dict overwrites per game_id — the last turn's reward (terminal reward)
    # becomes the game's representative value for computing the group mean.
    group_game_rewards: Dict[Tuple[int, int], Dict[int, float]] = defaultdict(dict)
    for s in samples:
        key = (s.group_id, s.player_id)
        group_game_rewards[key][s.game_id] = s.reward

    # Compute mean reward per group (one value per game, equal weighting)
    group_means: Dict[Tuple[int, int], float] = {}
    for key, game_rewards in group_game_rewards.items():
        rewards = list(game_rewards.values())
        group_means[key] = sum(rewards) / len(rewards)

    # Assign centered advantages using each sample's own reward (per-step
    # when available, terminal otherwise) relative to the group mean.
    for i, s in enumerate(samples):
        key = (s.group_id, s.player_id)
        advantages[i] = s.reward - group_means[key]

    return advantages


def filter_constant_reward_groups(
    samples: List[GRPOSample],
    advantages: torch.Tensor,
) -> Tuple[List[GRPOSample], torch.Tensor, int]:
    """
    Remove samples from groups where all rollouts got the same reward.

    When all K rollouts in a (group, player) pair have identical rewards,
    advantages are all 0 — no useful gradient signal. Filtering these out:
      - Avoids wasting compute on zero-signal samples
      - Prevents them from contributing to KL penalty (which doesn't depend
        on advantage and would otherwise push the model in unhelpful directions)
      - Matches rl4rl's remove_constant_reward_groups() behavior

    Returns:
        filtered_samples, filtered_advantages, num_groups_removed
    """
    # Identify constant-reward (group, player) pairs
    group_rewards: Dict[Tuple[int, int], set] = defaultdict(set)
    for s in samples:
        group_rewards[(s.group_id, s.player_id)].add(s.reward)

    constant_groups = {k for k, v in group_rewards.items() if len(v) <= 1}

    if not constant_groups:
        return samples, advantages, 0

    # Keep samples NOT in constant groups
    keep_idx = [
        i for i, s in enumerate(samples)
        if (s.group_id, s.player_id) not in constant_groups
    ]

    if not keep_idx:
        # All groups are constant — keep everything to avoid empty batch
        return samples, advantages, len(constant_groups)

    filtered_samples = [samples[i] for i in keep_idx]
    filtered_advantages = advantages[torch.tensor(keep_idx, dtype=torch.long)]

    return filtered_samples, filtered_advantages, len(constant_groups)


# ============================================================================
# Argument parsing
# ============================================================================

def parse_grpo_args():
    parser = argparse.ArgumentParser(
        description="GRPO trainer for game environments (rl4rl-style)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- Game and model --
    parser.add_argument("--game", type=str, default="adversarial_policy",
        help="Game to train on (from game_registry)")
    parser.add_argument("--model", type=str, default=None,
        help="HuggingFace model name (default: Config.MODEL_NAME)")

    # -- GRPO structure (inner/outer loop) --
    parser.add_argument("--group-size", type=int, default=16,
        help="Inner loop: rollouts per seed. More = cleaner advantage signal.")
    parser.add_argument("--groups-per-batch", type=int, default=8,
        help="Outer loop: different seeds per iteration. More = more diversity.")

    # -- LoRA --
    parser.add_argument("--lora-rank", type=int, default=16,
        help="LoRA rank (higher = more capacity for reasoning changes)")
    parser.add_argument("--lora-alpha", type=int, default=16,
        help="LoRA alpha scaling factor")

    # -- Optimization --
    parser.add_argument("--lr", type=float, default=1e-5,
        help="Learning rate (higher than PPO for meaningful updates)")
    parser.add_argument("--epochs", type=int, default=1,
        help="Training epochs per iteration over collected samples")
    parser.add_argument("--mini-batch-size", type=int, default=4,
        help="Mini-batch size for gradient updates")
    parser.add_argument("--stats-chunk-size", type=int, default=2,
        help="Chunk size for computing old/base logprobs (lower = less VRAM)")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
        help="Max gradient norm for clipping")

    # -- GRPO loss function --
    parser.add_argument("--use-clipping", action="store_true",
        help="Use clipped surrogate (like PPO). Default: pure importance sampling.")
    parser.add_argument("--clip-eps", type=float, default=0.2,
        help="Clipping epsilon (only used with --use-clipping)")
    parser.add_argument("--kl-coef", type=float, default=0.0,
        help="KL penalty coefficient against base model (rl4rl style, 0 to disable)")
    parser.add_argument("--filter-constant-groups", action="store_true", default=True,
        help="Remove groups where all rollouts got the same reward (no signal)")
    parser.add_argument("--no-filter-constant-groups", dest="filter_constant_groups",
        action="store_false", help="Disable constant-reward group filtering")

    # -- Generation --
    parser.add_argument("--temperature", type=float, default=0.7,
        help="Sampling temperature for rollouts")
    parser.add_argument("--temperature-range", type=str, default=None,
        help="Comma-separated temperatures for per-game variation (e.g. '0.5,0.7,1.0'). "
             "If not set, uses single --temperature value.")
    parser.add_argument("--prefix-ratio", type=float, default=0.4,
        help="Probability of auto-playing lookup prefix per group (0.0-1.0, adversarial_policy only)")
    parser.add_argument("--compact-tools", action="store_true", default=False,
        help="Use compressed tool schemas for training (strips descriptions, ~60%% smaller prompts)")
    parser.add_argument("--normalize-by-len", action="store_true", default=False,
        help="Normalize logprobs by action length (NOT recommended per rl4rl analysis)")

    # -- Checkpointing and evaluation --
    parser.add_argument("--resume", type=str, default=None,
        help="Path to checkpoint directory to resume from")
    parser.add_argument("--rollout-log", type=str, default="rollouts_grpo.jsonl",
        help="Filename for rollout logs")
    parser.add_argument("--save-every", type=int, default=5,
        help="Save checkpoint every N iterations")
    parser.add_argument("--tau2-eval-every", type=int, default=0,
        help="Tau2-bench eval every N iterations (0 to disable)")

    # -- Adversarial policy game --
    parser.add_argument("--adversarial-ratio", type=float, default=0.6,
        help="Ratio of adversarial vs cooperative scenarios (0.0-1.0). "
             "At 0.6, 60%% adversarial (T1-T17) and 40%% cooperative (T18-T24).")

    # -- User LLM (for adversarial_policy game) --
    parser.add_argument("--user-llm-url", type=str, default=None,
        help="OpenAI-compatible base URL for user LLM (e.g. http://localhost:9000/v1)")
    parser.add_argument("--user-llm-model", type=str, default=None,
        help="Model name for user LLM server")
    parser.add_argument("--user-llm-temperature", type=float, default=0.7,
        help="Temperature for user LLM generation")
    parser.add_argument("--user-llm-max-tokens", type=int, default=1024,
        help="Max tokens for user LLM generation")

    # -- SFT joint training --
    parser.add_argument("--sft-data", type=str, default=None,
        help="Comma-separated paths to tau2-bench eval JSON files for SFT joint training")
    parser.add_argument("--sft-coef", type=float, default=0.1,
        help="SFT loss weight (scales per-token NLL before adding to RL loss)")
    parser.add_argument("--sft-per-step", type=int, default=2,
        help="SFT samples per RL mini-batch step")

    # -- Root directory (passed through to setup_environment) --
    parser.add_argument("--root", type=str, default=None,
        help="Root directory for cache and outputs")

    return parser.parse_args()


# ============================================================================
# Main training loop
# ============================================================================

def main():
    args = parse_grpo_args()

    # ---- Game setup ----
    game = args.game
    try:
        game_spec = get_game_spec(game)
    except KeyError as e:
        available = ", ".join(list_game_names())
        raise RuntimeError(f"{e}. Available games: {available}") from e

    max_gen_tokens = game_spec.max_gen_tokens
    env_kwargs: Dict[str, Any] = {}

    # Wire up user LLM for adversarial_policy game
    if args.user_llm_url and game_spec.name == "adversarial_policy":
        from adversarial_policy_game import UserLLMClient
        user_client = UserLLMClient(
            base_url=args.user_llm_url,
            model=args.user_llm_model or "default",
            max_tokens=args.user_llm_max_tokens,
            temperature=args.user_llm_temperature,
        )
        env_kwargs["user_client"] = user_client
        print(f"[User LLM] {args.user_llm_url} model={args.user_llm_model}")

    # Wire up adversarial_ratio for adversarial_policy game
    if game_spec.name == "adversarial_policy":
        env_kwargs["adversarial_ratio"] = args.adversarial_ratio

    # Set dynamic rollout log name
    args.rollout_log = f"rollouts_grpo_{game_spec.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

    # ---- Environment setup (reuse existing config infrastructure) ----
    env_config = setup_environment(args)
    device = env_config["device"]
    hf_hub = env_config["hf_hub"]
    output_dir_path = env_config["output_dir_path"]
    rollout_log_path = env_config["rollout_log_path"]

    # Parse temperature range
    temperature_range: Optional[List[float]] = None
    if args.temperature_range and args.temperature_range.lower() != "none":
        temperature_range = [float(t.strip()) for t in args.temperature_range.split(",")]

    total_games_per_iter = args.group_size * args.groups_per_batch

    print("=" * 60)
    print("GRPO TRAINER")
    print("=" * 60)
    print(f"  Game:                {game_spec.name}")
    print(f"  Group size (inner):  {args.group_size}")
    print(f"  Groups/batch (outer):{args.groups_per_batch}")
    print(f"  Total games/iter:    {total_games_per_iter}")
    print(f"  LoRA rank:           {args.lora_rank}")
    print(f"  LoRA alpha:          {args.lora_alpha}")
    print(f"  Learning rate:       {args.lr}")
    print(f"  KL coefficient:      {args.kl_coef} (vs base model)")
    print(f"  Filter const groups: {args.filter_constant_groups}")
    print(f"  Use clipping:        {args.use_clipping}")
    print(f"  Normalize by len:    {args.normalize_by_len}")
    print(f"  Temperature:         {args.temperature}")
    if temperature_range:
        print(f"  Temperature range:   {temperature_range}")
    if game_spec.name == "adversarial_policy":
        print(f"  Adversarial ratio:   {args.adversarial_ratio}")
        print(f"  Prefix ratio:        {args.prefix_ratio}")
    print(f"  Compact tools:       {args.compact_tools}")
    print(f"  Max gen tokens:      {max_gen_tokens}")
    print(f"  Device:              {device}")
    print(f"  Output dir:          {output_dir_path}")
    print("=" * 60)

    # ---- Load model + LoRA (higher rank than PPO) ----
    model_name = args.model or Config.MODEL_NAME
    print(f"[Model] Loading {model_name} with LoRA rank={args.lora_rank}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=Config.MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
        offload_embedding=True,
        cache_dir=str(hf_hub),
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    FastLanguageModel.for_training(model)

    # ---- Resume from checkpoint ----
    start_iter = 0
    if args.resume:
        import safetensors.torch
        from peft import set_peft_model_state_dict

        resume_path = Path(args.resume)
        print(f"[Resume] Loading from {resume_path}")

        adapter_path = resume_path / "adapter_model.safetensors"
        if adapter_path.exists():
            state = safetensors.torch.load_file(str(adapter_path))
            set_peft_model_state_dict(model, state)
            print(f"[Resume] Loaded adapter from {adapter_path}")
        else:
            raise FileNotFoundError(f"Adapter not found at {adapter_path}")

        # Extract iteration number from directory name
        try:
            start_iter = int(resume_path.name.split("_")[-2]) + 1
        except (ValueError, IndexError):
            start_iter = 0
        print(f"[Resume] Starting from iteration {start_iter}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = model.to(device)
    print(f"[Model] Loaded — NO value head (GRPO style)")

    # ---- Save initial adapter for eval baseline ----
    base_model_adapter_dir = output_dir_path / "base_model_adapter_grpo"
    if not args.resume:
        base_model_adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(base_model_adapter_dir))
        print(f"[Base] Saved initial adapter to {base_model_adapter_dir}")

    # ---- Initialize inference backend (vLLM or HF local) ----
    inference_backend = init_inference_backend(model, tokenizer, device)
    vllm_adapter_dir = output_dir_path / "vllm_adapter_latest_grpo"
    if hasattr(inference_backend, "sync_base_adapter"):
        inference_backend.sync_base_adapter(base_model_adapter_dir)

    # ---- wandb ----
    if wandb:
        if not os.getenv("WANDB_NAME"):
            os.environ["WANDB_NAME"] = f"grpo-{game}-{int(time.time())}"
        wandb.login(key=os.getenv("WANDB_API_KEY", ""), relogin=True)
        wandb.init(entity="forge_scaling_intelligence_lab", project="games")
        wandb.config.update({
            "trainer": "grpo",
            "game": game,
            "group_size": args.group_size,
            "groups_per_batch": args.groups_per_batch,
            "lora_rank": args.lora_rank,
            "lr": args.lr,
            "kl_coef": args.kl_coef,
            "use_clipping": args.use_clipping,
            "temperature": args.temperature,
            "temperature_range": temperature_range,
            "prefix_ratio": args.prefix_ratio,
            "compact_tools": args.compact_tools,
        })
        print("[wandb] Initialized")

    # ---- Optimizer (LoRA params only, NO value head) ----
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable_params, lr=args.lr)
    rollout_logger = JSONLLogger(rollout_log_path)

    print(f"[GRPO] Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    print(f"[GRPO] Starting training loop...")

    global_step = start_iter * 25 if args.resume else 0
    # ---- SFT buffer initialization ----
    sft_buffer = None
    if args.sft_data:
        sft_paths = [p.strip() for p in args.sft_data.split(",") if p.strip()]
        if sft_paths:
            sft_buffer = SFTBuffer(sft_paths, tokenizer, compact_tools=args.compact_tools)
            print(f"[SFT] Initialized buffer: {len(sft_buffer)} samples from {len(sft_paths)} files")
            print(f"[SFT] coef={args.sft_coef}, per_step={args.sft_per_step}")

    # ======================================================================
    # Main GRPO loop
    # ======================================================================
    for it in tqdm(range(start_iter, 10_000), desc="GRPO iters", initial=start_iter, total=10_000):
        t_iter_start = time.time()

        # ---- 1. Sync policy to vLLM ----
        inference_backend.sync_policy(model, vllm_adapter_dir)
        if not inference_backend.is_enabled():
            inference_backend = HFLocalBackend(model, tokenizer, device)

        # ---- 2. Periodic tau2-bench evaluation ----
        if args.tau2_eval_every and it % args.tau2_eval_every == 0 and it > 0:
            model.eval()
            inference_backend.sync_policy(model, vllm_adapter_dir)
            if not inference_backend.is_enabled():
                inference_backend = HFLocalBackend(model, tokenizer, device)

            print(f"\n[eval {it}] Starting tau2-bench evaluation...")
            t_tau2_0 = time.time()
            tau2_logs, tau2_eval_files = evaluate_tau2_bench(
                backend=inference_backend,
                output_dir=output_dir_path,
                iteration=it,
                domains=list(Config.TAU2_EVAL_DOMAINS),
                num_tasks=Config.TAU2_EVAL_NUM_TASKS,
                num_trials=Config.TAU2_EVAL_NUM_TRIALS,
                max_concurrency_per_shard=Config.TAU2_EVAL_MAX_CONCURRENCY_PER_SHARD,
                seed=Config.TAU2_EVAL_SEED,
            )
            t_tau2_1 = time.time()
            if tau2_logs:
                tau2_logs["eval_tau2/time_total_sec"] = t_tau2_1 - t_tau2_0
                print(f"[eval {it}] tau2-bench done ({t_tau2_1 - t_tau2_0:.1f}s)")
                if wandb:
                    wandb.log(tau2_logs, step=global_step)

            # Add newly successful tasks to SFT buffer
            if sft_buffer is not None and tau2_eval_files:
                sft_buffer.refresh(tau2_eval_files)
                print(f"[SFT] Refreshed: {len(sft_buffer)} samples")

        # ---- 3. Collect group-based rollouts ----
        t_collect0 = time.time()
        samples, env_metrics = collect_grpo_rollouts(
            backend=inference_backend,
            tokenizer=tokenizer,
            game_spec=game_spec,
            groups_per_batch=args.groups_per_batch,
            group_size=args.group_size,
            temperature=args.temperature,
            max_new_tokens=max_gen_tokens,
            base_seed=it,
            logger=rollout_logger,
            env_kwargs=env_kwargs,
            temperature_range=temperature_range,
            prefix_ratio=args.prefix_ratio,
            compact_tools=args.compact_tools,
        )
        t_collect1 = time.time()

        if not samples:
            print(f"[iter {it}] No samples collected, skipping")
            continue

        # ---- 4. Compute group-relative advantages ----
        advantages = compute_group_advantages(samples)

        # ---- 4b. Filter constant-reward groups (rl4rl style) ----
        num_filtered = 0
        if args.filter_constant_groups:
            pre_filter_count = len(samples)
            samples, advantages, num_filtered = filter_constant_reward_groups(
                samples, advantages,
            )
            if num_filtered > 0:
                print(
                    f"[iter {it}] Filtered {num_filtered} constant-reward groups "
                    f"({pre_filter_count} -> {len(samples)} samples)"
                )

        # ---- 4c. Early skip when all advantages are zero ----
        avg_abs_advantage = float(advantages.abs().mean().item())
        if avg_abs_advantage == 0.0:
            t_train1 = time.time()
            avg_reward = float(torch.tensor([s.reward for s in samples]).mean().item())
            N = len(samples)

            logs = {
                "iter": it,
                "global_step": global_step,
                "grpo/policy_loss": 0.0,
                "grpo/kl_base_loss": 0.0,
                "grpo/approx_kl_rollout": 0.0,
                "grpo/approx_kl_base": 0.0,
                "grpo/clip_frac": 0.0,
                "grpo/ratio_mean": 1.0,
                "grpo/avg_reward": avg_reward,
                "grpo/avg_advantage": 0.0,
                "grpo/avg_abs_advantage": 0.0,
                "grpo/num_samples": N,
                "grpo/constant_groups_filtered": num_filtered,
                "grpo/skipped_zero_adv": 1,
                "time/collect_sec": t_collect1 - t_collect0,
                "time/train_sec": 0.0,
                "time/iter_sec": t_train1 - t_iter_start,
                **env_metrics,
            }

            print(
                f"[iter {it}] SKIPPED (all advantages zero) "
                f"reward={avg_reward:.3f} avg_r={env_metrics['env/avg_reward_p0']:.3f} "
                f"samples={N} filtered={num_filtered} "
                f"collect={logs['time/collect_sec']:.1f}s"
            )

            if wandb:
                wandb.log(logs, step=global_step)

            # Still save checkpoints on skip iterations (1st, 2nd, then every save_every)
            if it <= 1 or it % args.save_every == 0:
                ckpt_dir = output_dir_path / f"grpo_ckpt_iter_{it}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(ckpt_dir))
                tokenizer.save_pretrained(str(ckpt_dir))
                print(f"[checkpoint] Saved to {ckpt_dir}")

            continue

        # ---- 5. Pre-tokenize all samples ----
        seqs_cpu: List[torch.Tensor] = []
        prompt_lens: List[int] = []
        action_lens: List[int] = []
        kept_indices: List[int] = []
        max_seq_len = Config.MAX_SEQ_LENGTH
        num_skipped_long = 0

        for i, s in enumerate(samples):
            ids, pL, aL = build_prompt_plus_action(tokenizer, s.prompt_msgs, s.completion_text, tools=s.tools)
            if ids.shape[0] > max_seq_len:
                num_skipped_long += 1
                continue
            kept_indices.append(i)
            seqs_cpu.append(ids.cpu())
            prompt_lens.append(pL)
            action_lens.append(aL)

        if num_skipped_long > 0:
            print(f"  Skipped {num_skipped_long}/{len(samples)} samples exceeding {max_seq_len} tokens")
            advantages = advantages[torch.tensor(kept_indices, dtype=torch.long)]

        N = len(seqs_cpu)
        if N == 0:
            print(f"  [iter {it}] All samples exceeded max_seq_len — skipping training step")
            continue

        # ---- 6. Compute old_logp and base_logp in a single merged loop ----
        # Both passes reuse the same padded tensors on GPU to avoid
        # redundant data prep and CPU→GPU transfers.
        model.eval()
        old_logp_cpu = torch.empty(N, dtype=torch.float32)
        compute_base = args.kl_coef > 0
        base_logp_cpu = torch.empty(N, dtype=torch.float32) if compute_base else None

        with torch.no_grad():
            for start in range(0, N, args.stats_chunk_size):
                end = min(N, start + args.stats_chunk_size)
                mb_idx = list(range(start, end))
                mb_seqs = [seqs_cpu[j] for j in mb_idx]
                mb_pl = [prompt_lens[j] for j in mb_idx]
                mb_al = [action_lens[j] for j in mb_idx]

                mb_ids, mb_attn = pad_to_device(mb_seqs, tokenizer.pad_token_id, device)

                # -- old_logp (adapter ON) --
                with autocast_ctx(device):
                    outputs = model(
                        input_ids=mb_ids,
                        attention_mask=mb_attn,
                        use_cache=False,
                    )
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

                mb_old_logp = logprob_action_tokens(
                    logits, mb_ids, mb_pl, mb_al,
                    normalize_by_len=args.normalize_by_len,
                )
                old_logp_cpu[mb_idx] = mb_old_logp.detach().cpu()
                del logits, outputs, mb_old_logp

                # -- base_logp (adapter OFF, reuse same mb_ids/mb_attn) --
                if compute_base:
                    model.disable_adapter_layers()
                    try:
                        with autocast_ctx(device):
                            outputs = model(
                                input_ids=mb_ids,
                                attention_mask=mb_attn,
                                use_cache=False,
                            )
                        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

                        mb_base_logp = logprob_action_tokens(
                            logits, mb_ids, mb_pl, mb_al,
                            normalize_by_len=args.normalize_by_len,
                        )
                        base_logp_cpu[mb_idx] = mb_base_logp.detach().cpu()
                        del logits, outputs, mb_base_logp
                    finally:
                        model.enable_adapter_layers()

                del mb_ids, mb_attn

        # ---- 7. GRPO training ----
        model.train()
        idxs = list(range(N))

        policy_loss_acc = 0.0
        sft_loss_acc = 0.0
        kl_loss_acc = 0.0
        approx_kl_acc = 0.0
        kl_base_acc = 0.0
        clip_frac_acc = 0.0
        ratio_mean_acc = 0.0
        updates = 0

        for _epoch in range(args.epochs):
            random.shuffle(idxs)
            for start in range(0, N, args.mini_batch_size):
                mb = idxs[start : start + args.mini_batch_size]
                if not mb:
                    continue

                mb_seqs = [seqs_cpu[i] for i in mb]
                mb_pl = [prompt_lens[i] for i in mb]
                mb_al = [action_lens[i] for i in mb]

                mb_ids, mb_attn = pad_to_device(mb_seqs, tokenizer.pad_token_id, device)
                mb_old_logp = old_logp_cpu[mb].to(device)
                mb_adv = advantages[mb].to(device)
                mb_base_logp = base_logp_cpu[mb].to(device) if base_logp_cpu is not None else None

                # Forward pass
                outputs = model(
                    input_ids=mb_ids,
                    attention_mask=mb_attn,
                    use_cache=False,
                )
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

                new_logp = logprob_action_tokens(
                    logits, mb_ids, mb_pl, mb_al,
                    normalize_by_len=args.normalize_by_len,
                )

                # Importance sampling ratio
                ratio = torch.exp(new_logp - mb_old_logp)

                # Policy loss
                if args.use_clipping:
                    # PPO-style clipped surrogate
                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(
                        ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps
                    ) * mb_adv
                    policy_loss = -torch.mean(torch.min(surr1, surr2))
                else:
                    # Pure importance sampling (rl4rl default)
                    policy_loss = -torch.mean(ratio * mb_adv)

                # KL penalty against base model (rl4rl style — prevents forgetting)
                kl_loss = torch.tensor(0.0, device=device)
                if args.kl_coef > 0 and mb_base_logp is not None:
                    # KL(π_new || π_base) ≈ E[new_logp - base_logp]
                    kl_loss = args.kl_coef * torch.mean(new_logp - mb_base_logp)

                loss = policy_loss + kl_loss

                optim.zero_grad(set_to_none=True)
                loss.backward()

                # ---- SFT forward + backward (separate graph to avoid OOM) ----
                # Backward SFT independently so RL and SFT activation graphs
                # never coexist in memory. Gradients still accumulate correctly.
                sft_loss_val = torch.tensor(0.0, device=device)
                if sft_buffer is not None and len(sft_buffer) > 0:
                    sft_ids, sft_attn, sft_pl, sft_al = sft_buffer.sample_batch(
                        args.sft_per_step, device
                    )
                    sft_out = model(
                        input_ids=sft_ids, attention_mask=sft_attn, use_cache=False
                    )
                    sft_logits = sft_out.logits if hasattr(sft_out, "logits") else sft_out[0]
                    sft_logp = logprob_action_tokens(
                        sft_logits, sft_ids, sft_pl, sft_al, normalize_by_len=True
                    )
                    sft_loss_val = -sft_logp.mean()
                    sft_scaled = args.sft_coef * sft_loss_val
                    sft_scaled.backward()
                    del sft_ids, sft_attn, sft_logits, sft_out, sft_logp, sft_scaled

                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optim.step()

                # Tracking (no grad)
                with torch.no_grad():
                    approx_kl = torch.mean(mb_old_logp - new_logp)
                    clip_frac = torch.mean(
                        (torch.abs(ratio - 1.0) > args.clip_eps).float()
                    )
                    ratio_mean_val = torch.mean(ratio)
                    if mb_base_logp is not None:
                        approx_kl_base = torch.mean(new_logp - mb_base_logp)
                    else:
                        approx_kl_base = torch.tensor(0.0)

                global_step += 1
                updates += 1
                policy_loss_acc += float(policy_loss.item())
                sft_loss_acc += float(sft_loss_val.item())
                kl_loss_acc += float(kl_loss.item())
                approx_kl_acc += float(approx_kl.item())
                kl_base_acc += float(approx_kl_base.item())
                clip_frac_acc += float(clip_frac.item())
                ratio_mean_acc += float(ratio_mean_val.item())

                del mb_ids, mb_attn, logits, outputs, new_logp, ratio, loss

        t_train1 = time.time()

        # ---- 8. Logging ----
        avg_advantage = float(advantages.mean().item())
        all_rewards = torch.tensor([s.reward for s in samples], dtype=torch.float32)
        avg_reward = float(all_rewards.mean().item())

        logs = {
            "iter": it,
            "global_step": global_step,
            "grpo/policy_loss": policy_loss_acc / max(1, updates),
            "grpo/sft_loss": sft_loss_acc / max(1, updates),
            "grpo/sft_buffer_size": len(sft_buffer) if sft_buffer is not None else 0,
            "grpo/kl_base_loss": kl_loss_acc / max(1, updates),
            "grpo/approx_kl_rollout": approx_kl_acc / max(1, updates),
            "grpo/approx_kl_base": kl_base_acc / max(1, updates),
            "grpo/clip_frac": clip_frac_acc / max(1, updates),
            "grpo/ratio_mean": ratio_mean_acc / max(1, updates),
            "grpo/avg_reward": avg_reward,
            "grpo/avg_advantage": avg_advantage,
            "grpo/avg_abs_advantage": avg_abs_advantage,
            "grpo/num_samples": N,
            "grpo/constant_groups_filtered": num_filtered,
            "grpo/skipped_zero_adv": 0,
            "time/collect_sec": t_collect1 - t_collect0,
            "time/train_sec": t_train1 - t_collect1,
            "time/iter_sec": t_train1 - t_iter_start,
            **env_metrics,
        }

        print(
            f"[iter {it}] step={global_step} "
            f"reward={avg_reward:.3f} avg_r={env_metrics['env/avg_reward_p0']:.3f} "
            f"abs_adv={avg_abs_advantage:.3f} "
            f"invalid={env_metrics['env/invalid_move_rate']:.3f} "
            f"samples={N} filtered={num_filtered} "
            f"KL_rollout={logs['grpo/approx_kl_rollout']:.4f} "
            f"KL_base={logs['grpo/approx_kl_base']:.4f} "
            f"collect={logs['time/collect_sec']:.1f}s train={logs['time/train_sec']:.1f}s"
        )

        if wandb:
            wandb.log(logs, step=global_step)

        # ---- 9. Save checkpoint: 1st iter, 2nd iter, then every save_every ----
        if it <= 1 or it % args.save_every == 0:
            ckpt_dir = output_dir_path / f"grpo_ckpt_iter_{it}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ckpt_dir))
            tokenizer.save_pretrained(str(ckpt_dir))
            print(f"[checkpoint] Saved to {ckpt_dir}")

    # ---- Final save ----
    final_dir = output_dir_path / f"grpo_{game}_final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"[GRPO] Training complete! Saved to {final_dir}")


if __name__ == "__main__":
    main()
