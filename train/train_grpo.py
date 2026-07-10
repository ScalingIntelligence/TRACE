#!/usr/bin/env python3
import argparse
import json
import os
import random
import time

try:
    from loguru import logger as _loguru_logger

    _loguru_logger.remove()
except ImportError:
    pass

import requests
import torch
import torch.nn.functional as F
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from .config import Config, setup_environment, autocast_ctx
from game_registry import get_game_spec, list_game_names, GameSpec, GameMix, GameMixEntry
from concurrent.futures import ThreadPoolExecutor
from .inference import (
    InferenceBackend,
    HFLocalBackend,
    init_inference_backend,
    messages_for_game,
    tools_for_game,
    build_prompt_text,
    generate_completion,
)
from .ppo import (
    JSONLLogger,
    build_prompt_plus_action,
    logprob_action_tokens,
)

from .dist_utils import (
    dist_init, dist_pre_init, dist_nccl_init,
    dist_cleanup, is_main_rank, broadcast_objects,
    shard_batches, allreduce_coalesced_grads, allreduce_scalars,
    barrier, suppress_print,
)

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
    prompt_msgs: list          # Chat-template messages for this turn
    completion_text: str       # Full model output (thinking + action)
    player_id: int             # Which player produced this action
    reward: float              # Terminal reward for this player
    group_id: int              # Group index (same seed → same group)
    game_id: int               # Unique game identifier
    tools: Optional[list] = None  # Tool schemas for structured games (train→eval alignment)
    game_name: str = ""        # Name of the game that produced this sample


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
    compact_tools: bool = False,
    group_assignments: Optional[List[Tuple[GameSpec, Dict[str, Any]]]] = None,
    use_step_rewards: bool = True,
) -> Tuple[List[GRPOSample], Dict[str, float]]:
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

        game_temps: List[float] = []       # per-game temperature
        for g_idx, g_seed in enumerate(group_seeds):
            g_spec, g_kwargs = group_assignments[g_idx]

            for s_idx in range(group_size):
                env = g_spec.make_env(**g_kwargs)
                env.reset(g_seed)  # Same seed within group!

                envs.append(env)
                env_game_specs.append(g_spec)
                g_ids.append(g_idx)
                gm_ids.append(base_seed * 1_000_000 + g_idx * 1000 + s_idx)
                episode_steps.append([])

                # Assign per-game temperature
                if temperature_range:
                    game_temps.append(rng.choice(temperature_range))
                else:
                    game_temps.append(temperature)

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
                "timestamp": time.time(),
            }
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

            for s_idx in range(group_size):
                game_id = base_seed * 1_000_000 + g_idx * 1000 + s_idx
                env = g_spec.make_env(**g_kwargs)
                env.reset(g_seed)

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
# Hint-swap for training
# ============================================================================

def _swap_hint_prompts(samples: List) -> None:
    swapped = 0
    for s in samples:
        if not s.game_name:
            continue
        try:
            spec = get_game_spec(s.game_name)
        except KeyError:
            continue
        if spec.hint_prompt is None or spec.base_prompt is None:
            continue
        if (s.prompt_msgs
                and s.prompt_msgs[0].get("role") == "system"
                and s.prompt_msgs[0].get("content") == spec.hint_prompt):
            s.prompt_msgs[0]["content"] = spec.base_prompt
            swapped += 1
    if swapped > 0:
        print(f"  [hint-swap] Replaced hint→base prompt in {swapped}/{len(samples)} samples")


# ============================================================================
# Optimization helpers
# ============================================================================

def make_sorted_batches(seq_lens: List[int], batch_size: int) -> List[List[int]]:
    sorted_idx = sorted(range(len(seq_lens)), key=lambda i: seq_lens[i])
    return [sorted_idx[i:i + batch_size] for i in range(0, len(sorted_idx), batch_size)]


def pad_and_pin(seqs: List[torch.Tensor], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(s.shape[0] for s in seqs)
    bsz = len(seqs)
    ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((bsz, max_len), dtype=torch.long)
    for i, s in enumerate(seqs):
        L = s.shape[0]
        ids[i, :L] = s
        attn[i, :L] = 1
    if torch.cuda.is_available():
        ids = ids.pin_memory()
        attn = attn.pin_memory()
    return ids, attn


def compute_padding_efficiency(seq_lens: List[int], batches: List[List[int]]) -> float:
    total_real = sum(seq_lens)
    total_padded = sum(
        len(batch) * max(seq_lens[i] for i in batch)
        for batch in batches
    )
    return total_real / max(1, total_padded)


def pad_batch(
    batch_idx: List[int],
    seqs_cpu: List[torch.Tensor],
    prompt_lens: List[int],
    action_lens: List[int],
    pad_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
    mb_seqs = [seqs_cpu[j] for j in batch_idx]
    mb_ids, mb_attn = pad_and_pin(mb_seqs, pad_id)
    mb_pl = [prompt_lens[j] for j in batch_idx]
    mb_al = [action_lens[j] for j in batch_idx]
    return mb_ids, mb_attn, mb_pl, mb_al




def _extract_tool_name(completion_text: str) -> Optional[str]:
    depth = 0
    start = -1
    for i, c in enumerate(completion_text):
        if c == '{':
            if depth == 0:
                start = i
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    obj = json.loads(completion_text[start:i + 1])
                    if "name" in obj:
                        return obj["name"]
                except (json.JSONDecodeError, ValueError):
                    pass
                start = -1
    return None


def filter_info_gathering_turns(
    samples: List["GRPOSample"],
    advantages: torch.Tensor,
) -> Tuple[List["GRPOSample"], torch.Tensor, int]:
    # Pass 1: mark info-only turns for removal
    keep_mask = []
    for s in samples:
        tool_name = _extract_tool_name(s.completion_text)
        if tool_name is None:
            keep_mask.append(True)
        else:
            keep_mask.append(False)

    # Pass 2: guarantee at least one sample per game_id (keep last turn)
    last_idx_per_game: Dict[int, int] = {}
    for i, s in enumerate(samples):
        last_idx_per_game[s.game_id] = i
    for last_idx in last_idx_per_game.values():
        keep_mask[last_idx] = True

    # Pass 3: build filtered lists
    keep_indices = [i for i, keep in enumerate(keep_mask) if keep]
    num_filtered_out = len(samples) - len(keep_indices)

    if num_filtered_out == 0:
        return samples, advantages, 0

    filtered_samples = [samples[i] for i in keep_indices]
    filtered_advantages = advantages[torch.tensor(keep_indices, dtype=torch.long)]
    return filtered_samples, filtered_advantages, num_filtered_out


def compress_tool_results(
    prompt_msgs: List[Dict[str, Any]],
    max_tool_result_chars: int = 200,
) -> Tuple[List[Dict[str, Any]], int]:
    tool_indices = [
        i for i, msg in enumerate(prompt_msgs)
        if msg.get("role") == "tool"
    ]

    if len(tool_indices) <= 1:
        return prompt_msgs, 0

    truncate_set = set(tool_indices[:-1])
    num_truncated = 0

    compressed = []
    for i, msg in enumerate(prompt_msgs):
        if i in truncate_set:
            content = msg.get("content", "")
            if len(content) > max_tool_result_chars:
                new_msg = dict(msg)
                new_msg["content"] = content[:max_tool_result_chars] + "... [truncated]"
                compressed.append(new_msg)
                num_truncated += 1
            else:
                compressed.append(msg)
        else:
            compressed.append(msg)

    return compressed, num_truncated


# ============================================================================
# Argument parsing (extends train_grpo's parser with optimization defaults)
# ============================================================================

def parse_grpo_args_optimized():
    parser = argparse.ArgumentParser(
        description="Optimized GRPO trainer (same algorithm, faster training step)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- Game and model --
    parser.add_argument("--game", type=str, default=None,
        help="Game to train on (from game_registry). Required unless --games is set.")
    parser.add_argument("--games", type=str, default=None,
        help="Multi-game mix as 'game1:weight1,game2:weight2,...'. "
             "Weights are normalized to sum to 1. Overrides --game when set.")
    parser.add_argument("--model", type=str, default=None,
        help="HuggingFace model name (default: Config.MODEL_NAME)")

    # -- GRPO structure --
    parser.add_argument("--group-size", type=int, default=16,
        help="Inner loop: rollouts per seed.")
    parser.add_argument("--groups-per-batch", type=int, default=4,
        help="Outer loop: different seeds per iteration.")

    # -- LoRA --
    parser.add_argument("--lora-rank", type=int, default=16,
        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16,
        help="LoRA alpha scaling factor")

    # -- Optimization --
    parser.add_argument("--lr", type=float, default=1e-5,
        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1,
        help="Training epochs per iteration")
    parser.add_argument("--mini-batch-size", type=int, default=4,
        help="Mini-batch size for gradient updates")
    parser.add_argument("--stats-chunk-size", type=int, default=4,
        help="Chunk size for computing old/base logprobs (matches mini-batch-size for unified caching)")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
        help="Max gradient norm for clipping")

    # -- GRPO loss function --
    parser.add_argument("--use-clipping", action="store_true", default=True,
        help="Use clipped surrogate (like PPO).")
    parser.add_argument("--no-clipping", dest="use_clipping", action="store_false",
        help="Disable clipping and use pure importance sampling.")
    parser.add_argument("--clip-eps", type=float, default=0.28,
        help="Clipping epsilon (only used with --use-clipping). Default 0.28 per DAPO for better exploration.")
    parser.add_argument("--kl-coef", type=float, default=0.0,
        help="KL penalty coefficient against base model (0 to disable, saves ~50%% of step 6 time)")
    parser.add_argument("--filter-constant-groups", action="store_true", default=True,
        help="Remove groups where all rollouts got the same reward")
    parser.add_argument("--no-filter-constant-groups", dest="filter_constant_groups",
        action="store_false")

    # -- Generation --
    parser.add_argument("--temperature", type=float, default=1.0,
        help="Sampling temperature for rollouts")
    parser.add_argument("--temperature-range", type=str, default=None,
        help="Comma-separated temperatures for per-game variation (e.g. '0.5,0.7,1.0'). "
             "If not set, uses single --temperature value.")
    parser.add_argument("--compact-tools", action="store_true", default=False,
        help="Use compressed tool schemas for training (strips descriptions, ~60%% smaller prompts)")
    parser.add_argument("--normalize-by-len", action="store_true",
        help="Normalize logprobs by action length")

    # -- Checkpointing and evaluation --
    parser.add_argument("--resume", type=str, default=None,
        help="Path to checkpoint directory to resume from")
    parser.add_argument("--rollout-log", type=str, default="selfplay_rollouts_grpo.jsonl",
        help="Filename for rollout logs")
    parser.add_argument("--save-every", type=int, default=5,
        help="Save checkpoint every N iterations")

    # -- User LLM (for games with needs_user_llm=True) --
    parser.add_argument("--user-llm-url", type=str, default=None,
        help="OpenAI-compatible base URL for user LLM")
    parser.add_argument("--user-llm-model", type=str, default=None,
        help="Model name for user LLM server")
    parser.add_argument("--user-llm-temperature", type=float, default=0.7,
        help="Temperature for user LLM generation")
    parser.add_argument("--user-llm-max-tokens", type=int, default=1024,
        help="Max tokens for user LLM generation")

    # -- Dynamic sampling (DAPO-style filter) --
    parser.add_argument("--dynamic-sampling-max-batches", type=int, default=3,
        help="Max re-collection attempts when too many groups are constant-reward. "
             "0 to disable. Each attempt collects a full batch of rollouts with new "
             "seeds and keeps only informative groups. Stops when enough informative "
             "samples are accumulated or limit is reached.")
    parser.add_argument("--dynamic-sampling-min-groups", type=int, default=0,
        help="Minimum number of informative groups required before stopping re-collection. "
             "0 = auto (half of groups_per_batch). Set explicitly to override.")

    # -- Training sample optimizations --
    parser.add_argument("--filter-info-turns", action="store_true", default=True,
        help="Filter out info-gathering tool calls from training samples")
    parser.add_argument("--no-filter-info-turns", dest="filter_info_turns",
        action="store_false", help="Disable info-gathering turn filtering")
    parser.add_argument("--step-rewards", action="store_true", default=True,
        help="Use per-step rewards for credit assignment when the env exposes "
             "`_step_rewards`; falls back to terminal reward otherwise.")
    parser.add_argument("--no-step-rewards", dest="step_rewards",
        action="store_false", help="Disable per-step rewards (all turns get terminal reward)")
    parser.add_argument("--tool-result-max-chars", type=int, default=200,
        help="Max chars for truncated old tool results (0 to disable)")

    # -- Distributed training --
    parser.add_argument("--dist-lr-scale", type=float, default=1.0,
        help="Scale learning rate for distributed training (try 2-4x with many GPUs)")

    # -- Root directory --
    parser.add_argument("--root", type=str, default=None,
        help="Root directory for cache and outputs")

    return parser.parse_args()


# ============================================================================
# User LLM client (for games with needs_user_llm=True)
# ============================================================================

class UserLLMClient:

    def __init__(
        self,
        base_url: str,
        model: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.session = requests.Session()
        # Increase pool size to handle parallel rollouts (default 10 is too small)
        adapter = requests.adapters.HTTPAdapter(pool_maxsize=128, pool_connections=128)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def generate(self, system_prompt: str, messages: List[Dict[str, str]]) -> str:
        all_messages = [{"role": "system", "content": system_prompt}] + messages
        payload = {
            "model": self.model,
            "messages": all_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        resp = self.session.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"] or ""


# ============================================================================
# Multi-game helpers
# ============================================================================

def build_env_kwargs(game_spec: GameSpec, args) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    if args.user_llm_url and game_spec.needs_user_llm:
        kwargs["user_client"] = UserLLMClient(
            base_url=args.user_llm_url,
            model=args.user_llm_model or "default",
            max_tokens=args.user_llm_max_tokens,
            temperature=args.user_llm_temperature,
        )
    return kwargs


def parse_game_mix(games_str: str, args) -> GameMix:
    entries = []
    total_weight = 0.0
    for part in games_str.split(","):
        part = part.strip()
        if ":" in part:
            name, w = part.rsplit(":", 1)
            weight = float(w)
        else:
            name = part
            weight = 1.0
        spec = get_game_spec(name.strip())
        total_weight += weight
        entries.append((spec, weight))

    # Normalize weights
    mix_entries = []
    for spec, weight in entries:
        normed = weight / total_weight
        kwargs = build_env_kwargs(spec, args)
        mix_entries.append(GameMixEntry(game_spec=spec, weight=normed, env_kwargs=kwargs))

    return GameMix(entries=mix_entries)


# ============================================================================
# Main training loop (optimized)
# ============================================================================

def main():
    # Import Unsloth BEFORE NCCL init: Unsloth's module-level code
    # (vLLM logger suppression, torch.compile LoRA patching) deadlocks
    # when NCCL is already initialized.
    from unsloth import FastLanguageModel

    # ---- Distributed init (Phase 1: CUDA device only, NO NCCL yet) ----
    # Defer NCCL initialization until after model loading because
    # from_pretrained / accelerate detect is_initialized() and trigger
    # internal collective operations that differ across ranks.
    rank, world_size, local_rank = dist_pre_init()
    if rank != 0:
        suppress_print()

    args = parse_grpo_args_optimized()

    # Apply distributed LR scaling
    if world_size > 1 and args.dist_lr_scale != 1.0:
        args.lr *= args.dist_lr_scale

    # Optimization flags via env vars
    use_compile = os.getenv("GRPO_COMPILE", "0") == "1"
    use_train_autocast = os.getenv("GRPO_TRAIN_AUTOCAST", "0") == "1"

    # ---- Game setup ----
    if not args.games and not args.game:
        raise SystemExit("error: one of --game or --games is required")

    if args.games:
        # Multi-game mix mode
        game_mix = parse_game_mix(args.games, args)
        # Primary game_spec used for backward-compat paths (logging, wandb config)
        game_spec = game_mix.entries[0].game_spec
        game = "+".join(e.game_spec.name for e in game_mix.entries)
        max_gen_tokens = game_mix.max_gen_tokens
        env_kwargs: Dict[str, Any] = game_mix.entries[0].env_kwargs
        print(f"[GameMix] {len(game_mix.entries)} games: "
              + ", ".join(f"{e.game_spec.name}:{e.weight:.2f}" for e in game_mix.entries))
    else:
        # Single-game mode (backward compatible)
        game_mix = None
        game = args.game
        try:
            game_spec = get_game_spec(game)
        except KeyError as e:
            available = ", ".join(list_game_names())
            raise RuntimeError(f"{e}. Available games: {available}") from e

        max_gen_tokens = game_spec.max_gen_tokens
        env_kwargs = build_env_kwargs(game_spec, args)
        if "user_client" in env_kwargs:
            print(f"[User LLM] {args.user_llm_url} model={args.user_llm_model}")

    # Set dynamic rollout log name
    args.rollout_log = f"rollouts_grpo_{game}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    # Make `args.game` always a string so setup_environment can use it for the
    # output directory name (in --games mode it would otherwise still be None).
    args.game = game

    # ---- Environment setup ----
    env_config = setup_environment(args)
    device = f"cuda:{local_rank}" if world_size > 1 else env_config["device"]
    hf_hub = env_config["hf_hub"]
    output_dir_path = env_config["output_dir_path"]
    rollout_log_path = env_config["rollout_log_path"]

    # Parse temperature range
    temperature_range: Optional[List[float]] = None
    if args.temperature_range and args.temperature_range.lower() != "none":
        temperature_range = [float(t.strip()) for t in args.temperature_range.split(",")]

    total_games_per_iter = args.group_size * args.groups_per_batch

    print("=" * 60)
    print("GRPO TRAINER (OPTIMIZED)")
    print("=" * 60)
    print(f"  Game:                {game}")
    if game_mix:
        for e in game_mix.entries:
            print(f"    - {e.game_spec.name}: weight={e.weight:.2f}, max_gen={e.game_spec.max_gen_tokens}")
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
    print(f"  Compact tools:       {args.compact_tools}")
    print(f"  Max gen tokens:      {max_gen_tokens}")
    print(f"  Device:              {device}")
    print(f"  Output dir:          {output_dir_path}")
    print(f"  [OPT] length sorting: ON")
    print(f"  [OPT] pinned memory:  ON")
    print(f"  [OPT] torch.compile:  {use_compile}")
    print(f"  [OPT] train autocast: ALWAYS ON (matches step 6)")
    if args.kl_coef == 0:
        print(f"  [OPT] base_logp:      SKIPPED (kl_coef=0)")
    print(f"  [OPT] filter info turns: {args.filter_info_turns}")
    print(f"  [OPT] step rewards:      {args.step_rewards}")
    print(f"  [OPT] tool result trunc: {args.tool_result_max_chars} chars (0=off)")
    if world_size > 1:
        print(f"  [DIST] world_size:    {world_size}")
        print(f"  [DIST] rank:          {rank}")
        print(f"  [DIST] lr_scale:      {args.dist_lr_scale}")
    print("=" * 60)

    # ---- Load model + LoRA ----
    # All ranks load simultaneously BEFORE NCCL is initialized.  Each rank
    # targets its own GPU via CUDA_VISIBLE_DEVICES + torch.cuda.set_device().
    # NCCL must NOT be initialized yet: from_pretrained / accelerate /
    # transformers detect is_initialized() and trigger internal collective
    # operations that differ across ranks, causing asymmetric NCCL hangs.
    model_name = args.model or Config.MODEL_NAME
    print(f"[Model] Rank {rank}: Loading {model_name} with LoRA rank={args.lora_rank}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=Config.MAX_SEQ_LENGTH,
        dtype=None,
        use_exact_model_name=True,
        cache_dir=str(hf_hub),
        device_map={"": local_rank},
    )
    print(f"[Model] Rank {rank}: Model loaded successfully", flush=True)

    # ---- Distributed init (Phase 2: NOW init NCCL) ----
    # Model is loaded on each rank independently. Safe to init NCCL now
    # since no more from_pretrained calls that could trigger rogue collectives.
    print(f"[NCCL] Rank {rank}: Initializing NCCL process group...", flush=True)
    dist_nccl_init()
    print(f"[NCCL] Rank {rank}: NCCL initialized, entering barrier...", flush=True)
    barrier()
    print(f"[NCCL] Rank {rank}: Barrier passed, setting up LoRA...", flush=True)
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
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

        try:
            start_iter = int(resume_path.name.split("_")[-2]) + 1
        except (ValueError, IndexError):
            start_iter = 0
        print(f"[Resume] Starting from iteration {start_iter}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = model.to(device)
    print(f"[Model] Loaded — NO value head (GRPO style)")

    # ---- Optional: torch.compile ----
    if use_compile:
        print("[OPT] Compiling model with torch.compile...")
        model = torch.compile(model)
        print("[OPT] Compilation complete")

    # ---- Save initial adapter for eval baseline (rank 0 only) ----
    base_model_adapter_dir = output_dir_path / "base_model_adapter_grpo"
    if not args.resume and is_main_rank():
        base_model_adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(base_model_adapter_dir))
        print(f"[Base] Saved initial adapter to {base_model_adapter_dir}")
    barrier()

    # ---- Initialize inference backend (rank 0 only for vLLM interaction) ----
    if is_main_rank():
        inference_backend = init_inference_backend(model, tokenizer, device)
        vllm_adapter_dir = output_dir_path / "vllm_adapter_latest_grpo"
        if hasattr(inference_backend, "sync_base_adapter"):
            inference_backend.sync_base_adapter(base_model_adapter_dir)
    else:
        inference_backend = None
        vllm_adapter_dir = None

    # ---- wandb (rank 0 only) ----
    if wandb and is_main_rank():
        if not os.getenv("WANDB_NAME"):
            os.environ["WANDB_NAME"] = f"grpo-opt-{game}-{int(time.time())}"
        wandb.login(key=os.getenv("WANDB_API_KEY", ""), relogin=True)
        wandb.init(entity="forge_scaling_intelligence_lab", project="games")
        wandb.config.update({
            "trainer": "grpo_optimized",
            "game": game,
            "games": args.games or args.game,
            "group_size": args.group_size,
            "groups_per_batch": args.groups_per_batch,
            "lora_rank": args.lora_rank,
            "lr": args.lr,
            "kl_coef": args.kl_coef,
            "use_clipping": args.use_clipping,
            "temperature": args.temperature,
            "opt_compile": use_compile,
            "opt_train_autocast": use_train_autocast,
            "world_size": world_size,
        })
        print("[wandb] Initialized")

    # ---- Optimizer (LoRA params only, NO value head) ----
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable_params, lr=args.lr)
    rollout_logger = JSONLLogger(rollout_log_path) if is_main_rank() else None

    print(f"[GRPO] Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    print(f"[GRPO] Starting optimized training loop...")

    global_step = start_iter * 25 if args.resume else 0

    # ======================================================================
    # Main GRPO loop (optimized)
    # ======================================================================
    for it in tqdm(range(start_iter, 10_000), desc="GRPO iters", initial=start_iter, total=10_000):
        t_iter_start = time.time()

        # ---- 1. Sync policy to vLLM (rank 0 only) ----
        if is_main_rank():
            inference_backend.sync_policy(model, vllm_adapter_dir)
            if not inference_backend.is_enabled():
                inference_backend = HFLocalBackend(model, tokenizer, device)
        barrier()

        # ---- 2. Collect group-based rollouts with dynamic sampling (rank 0) ----
        # Dynamic sampling (DAPO-style filter): if too many groups are constant-
        # reward, re-collect with new seeds and accumulate informative samples
        # until we have enough signal or hit the max-batch limit.
        ds_max = args.dynamic_sampling_max_batches
        ds_min_groups = args.dynamic_sampling_min_groups
        if ds_min_groups <= 0:
            ds_min_groups = max(1, args.groups_per_batch // 2)

        if is_main_rank():
            accumulated_samples: List[GRPOSample] = []
            accumulated_metrics: Dict[str, float] = {}
            t_collect0 = time.time()
            ds_attempt = 0
            ds_total_groups_collected = 0
            ds_informative_groups = 0

            while True:
                ds_attempt += 1
                # Seed: first attempt uses `it` (backward-compatible with old code).
                # Re-collection attempts use large offset to avoid collisions
                # with other iterations (it*100 could collide: iter 0 attempt 2
                # = seed 1 would collide with iter 1 attempt 1 = seed 1).
                batch_seed = it if ds_attempt == 1 else it + ds_attempt * 1_000_000

                if game_mix:
                    iter_group_assignments = game_mix.assign_groups(args.groups_per_batch)
                else:
                    iter_group_assignments = None

                batch_samples, batch_metrics = collect_grpo_rollouts(
                    backend=inference_backend,
                    tokenizer=tokenizer,
                    game_spec=game_spec,
                    groups_per_batch=args.groups_per_batch,
                    group_size=args.group_size,
                    temperature=args.temperature,
                    max_new_tokens=max_gen_tokens,
                    base_seed=batch_seed,
                    logger=rollout_logger,
                    env_kwargs=env_kwargs,
                    temperature_range=temperature_range,
                    compact_tools=args.compact_tools,
                    group_assignments=iter_group_assignments,
                    use_step_rewards=args.step_rewards,
                )

                if not batch_samples:
                    break

                # ---- Hint-swap: replace hint system prompt with base prompt ----
                # For games that opt in via GameSpec.hint_prompt / .base_prompt,
                # swap any sample whose system prompt matches the hint with the
                # base version, so the gradient reinforces behavior conditioned
                # on the base (eval-time) prompt rather than the hint.
                _swap_hint_prompts(batch_samples)

                # Count informative groups in this batch (groups with reward variance)
                batch_group_rewards: Dict[Tuple[int, int], set] = defaultdict(set)
                for s in batch_samples:
                    batch_group_rewards[(s.group_id, s.player_id)].add(s.reward)
                batch_total_groups = len(batch_group_rewards)
                batch_constant = sum(1 for v in batch_group_rewards.values() if len(v) <= 1)
                batch_informative = batch_total_groups - batch_constant

                ds_total_groups_collected += batch_total_groups

                if args.filter_constant_groups:
                    # Keep only samples from informative groups
                    constant_keys = {k for k, v in batch_group_rewards.items() if len(v) <= 1}
                    kept_samples = [s for s in batch_samples if (s.group_id, s.player_id) not in constant_keys]
                    ds_informative_groups += batch_informative

                    # Remap group_ids to avoid collisions across attempts.
                    # Offset by (attempt-1) * groups_per_batch so group_ids are unique.
                    gid_offset = (ds_attempt - 1) * args.groups_per_batch
                    for s in kept_samples:
                        s.group_id = s.group_id + gid_offset
                        s.game_id = s.game_id + (ds_attempt - 1) * 10_000_000

                    accumulated_samples.extend(kept_samples)
                else:
                    # No filtering — just take the whole batch
                    ds_informative_groups += batch_informative
                    accumulated_samples.extend(batch_samples)

                # Update metrics (keep first batch's metrics, overwrite timing)
                if not accumulated_metrics:
                    accumulated_metrics = batch_metrics

                print(
                    f"[iter {it}] DS attempt {ds_attempt}: "
                    f"{batch_informative}/{batch_total_groups} informative groups, "
                    f"accumulated {ds_informative_groups} informative, "
                    f"{len(accumulated_samples)} total samples"
                )

                # Stop conditions
                if ds_max <= 1:
                    break  # dynamic sampling disabled — single collection only
                if ds_informative_groups >= ds_min_groups:
                    break  # enough signal
                if ds_attempt >= ds_max:
                    print(f"[iter {it}] DS: hit max attempts ({ds_max}), proceeding with {ds_informative_groups} informative groups")
                    break

            t_collect1 = time.time()
            samples = accumulated_samples
            env_metrics = accumulated_metrics
        else:
            t_collect0 = t_collect1 = time.time()
            samples, env_metrics = [], {}

        # ---- 4. Broadcast raw samples, compute advantages on all ranks ----
        if is_main_rank():
            if not samples:
                print(f"[iter {it}] No samples collected, skipping")
                broadcast_data = [True, None, None, 0.0, 0.0, 0, 0]
            else:
                broadcast_data = [False, samples, env_metrics, t_collect0, t_collect1,
                                  ds_informative_groups, ds_total_groups_collected]
        else:
            broadcast_data = [None] * 7

        broadcast_data = broadcast_objects(broadcast_data)
        skip_flag = broadcast_data[0]

        if skip_flag:
            continue

        _, samples, env_metrics, t_collect0, t_collect1, ds_informative_groups, ds_total_groups_collected = broadcast_data

        # All ranks compute advantages in parallel (deterministic, identical results)
        advantages = compute_group_advantages(samples)

        # 4b. Filter constant-reward groups (all ranks, deterministic)
        # Note: when dynamic sampling is active, constant groups were already
        # filtered during collection (rank 0). This second pass catches any
        # remaining constant groups and handles the single-attempt case (ds_max=0).
        num_filtered = 0
        if args.filter_constant_groups:
            pre_filter_count = len(samples)
            samples, advantages, num_filtered = filter_constant_reward_groups(
                samples, advantages,
            )
            if num_filtered > 0 and is_main_rank():
                print(
                    f"[iter {it}] Filtered {num_filtered} constant-reward groups "
                    f"({pre_filter_count} -> {len(samples)} samples)"
                )

        # 4b2. Filter info-gathering turns (all ranks, deterministic)
        dt_filtered = 0
        samples_before_dt_filter = len(samples)
        if args.filter_info_turns:
            samples, advantages, dt_filtered = filter_info_gathering_turns(
                samples, advantages,
            )
            if dt_filtered > 0 and is_main_rank():
                print(
                    f"[iter {it}] Filtered {dt_filtered} info-gathering turns "
                    f"({samples_before_dt_filter} -> {len(samples)} samples)"
                )

        # 4c. Normalize advantages PER GAME + reweight for equal gradient (all ranks, deterministic)
        # Two-step normalization to prevent cross-game gradient interference:
        #   Step 1: Standardize each game independently (mean=0, std=1).
        #   Step 2: Scale by 1/N_game so total gradient contribution per game is equal,
        #           regardless of how many samples survived filtering.
        # Without step 1, games with higher raw reward variance dominate.
        # Without step 2, games with more informative groups dominate.
        game_names = set(s.game_name for s in samples if s.game_name)
        if game_names:
            # Count samples per game (for reweighting)
            game_counts = {}
            for gn in game_names:
                game_counts[gn] = sum(1 for s in samples if s.game_name == gn)
            n_games_present = len(game_counts)

            for gn in game_names:
                mask = torch.tensor([s.game_name == gn for s in samples], dtype=torch.bool)
                if mask.any():
                    game_adv = advantages[mask]
                    game_std = game_adv.std()
                    if game_std > 1e-8:
                        # Step 1: standardize to mean=0, std=1
                        normed = (game_adv - game_adv.mean()) / (game_std + 1e-8)
                        # Step 2: scale by 1/N_game so total |gradient| per game is equal.
                        # After step 1, E[|adv|] ≈ 0.8 per sample (std-normal).
                        # Total |gradient| ∝ N_game * E[|adv|]. To equalize across games,
                        # divide by N_game (and multiply by mean_N to keep magnitude stable).
                        mean_n = sum(game_counts.values()) / n_games_present
                        advantages[mask] = normed * (mean_n / game_counts[gn])
                    else:
                        advantages[mask] = 0.0
        else:
            # Fallback: single-game mode, normalize globally
            adv_std = advantages.std()
            if adv_std > 1e-8:
                advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

        avg_abs_advantage = float(advantages.abs().mean().item())

        # ---- 4d. Early skip when all advantages are zero ----
        if avg_abs_advantage == 0.0:
            if is_main_rank():
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
                    "opt/info_turns_filtered": dt_filtered,
                    "opt/tool_results_truncated": 0,
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

                if it <= 1 or it % args.save_every == 0:
                    ckpt_dir = output_dir_path / f"grpo_ckpt_iter_{it}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(str(ckpt_dir))
                    tokenizer.save_pretrained(str(ckpt_dir))
                    print(f"[checkpoint] Saved to {ckpt_dir}")
            barrier()
            continue

        # ================================================================
        # OPTIMIZED STEPS 5-7
        # ================================================================

        # ---- 5. Pre-tokenize all samples ----
        t_tok0 = time.time()

        seqs_cpu: List[torch.Tensor] = []
        prompt_lens: List[int] = []
        action_lens: List[int] = []
        kept_indices: List[int] = []
        total_tool_results_truncated = 0
        max_seq_len = Config.MAX_SEQ_LENGTH
        num_skipped_long = 0

        for i, s in enumerate(samples):
            if args.tool_result_max_chars > 0:
                msgs, n_trunc = compress_tool_results(s.prompt_msgs, args.tool_result_max_chars)
                total_tool_results_truncated += n_trunc
            else:
                msgs = s.prompt_msgs
            ids, pL, aL = build_prompt_plus_action(tokenizer, msgs, s.completion_text, tools=s.tools)
            if ids.shape[0] > max_seq_len:
                num_skipped_long += 1
                continue
            kept_indices.append(i)
            seqs_cpu.append(ids.cpu())
            prompt_lens.append(pL)
            action_lens.append(aL)

        if num_skipped_long > 0:
            print(f"  [OPT] Skipped {num_skipped_long}/{len(samples)} samples exceeding {max_seq_len} tokens")
            advantages = advantages[torch.tensor(kept_indices, dtype=torch.long)]
            samples = [samples[i] for i in kept_indices]

        N = len(seqs_cpu)
        if N == 0:
            print(f"  [iter {it}] All samples exceeded max_seq_len — skipping training step")
            barrier()
            continue
        seq_lens = [s.shape[0] for s in seqs_cpu]

        t_tok1 = time.time()

        # ---- 5b. Create length-sorted batches ----
        # When stats_chunk_size == mini_batch_size (default), we use a SINGLE
        # set of batches for both steps 6 and 7 — avoids duplicate padding and
        # halves CPU memory usage (benchmarked: 4.7x faster caching, 50% less mem).
        unified_batches = args.stats_chunk_size == args.mini_batch_size
        train_batches = make_sorted_batches(seq_lens, args.mini_batch_size)

        if unified_batches:
            stats_batches = train_batches
        else:
            stats_batches = make_sorted_batches(seq_lens, args.stats_chunk_size)

        # ---- 5c. Batch lists ready (padding done on-the-fly per batch) ----
        # Each rank only materializes padded tensors for batches it processes,
        # ensuring multi-GPU memory per rank matches single-GPU memory.
        t_pad = time.time()

        # ---- 6. Compute old_logp and (optionally) base_logp [DISTRIBUTED] ----
        model.eval()
        old_logp_cpu = torch.zeros(N, dtype=torch.float32)  # zeros for all-reduce SUM trick
        compute_base = args.kl_coef > 0
        base_logp_cpu = torch.zeros(N, dtype=torch.float32) if compute_base else None

        # Shard stats batches across ranks
        all_stats_indices = list(range(len(stats_batches)))
        my_stats_indices, _ = shard_batches(all_stats_indices, rank, world_size)

        with torch.no_grad():
            # -- Pass 1: old_logp (adapter ON) — each rank does its shard --
            for bi in my_stats_indices:
                mb_idx = stats_batches[bi]
                mb_ids_cpu, mb_attn_cpu, mb_pl, mb_al = pad_batch(
                    mb_idx, seqs_cpu, prompt_lens, action_lens, tokenizer.pad_token_id
                )
                mb_ids = mb_ids_cpu.to(device, non_blocking=True)
                mb_attn = mb_attn_cpu.to(device, non_blocking=True)

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
                del logits, outputs, mb_old_logp, mb_ids, mb_attn, mb_ids_cpu, mb_attn_cpu

            # All-reduce: each rank wrote to disjoint indices (rest are 0), SUM assembles full vector
            if world_size > 1:
                old_logp_gpu = old_logp_cpu.to(device)
                torch.distributed.all_reduce(old_logp_gpu, op=torch.distributed.ReduceOp.SUM)
                old_logp_cpu = old_logp_gpu.cpu()
                del old_logp_gpu

            # -- Pass 2: base_logp (adapter OFF) — distributed --
            if compute_base:
                model.disable_adapter_layers()
                try:
                    for bi in my_stats_indices:
                        mb_idx = stats_batches[bi]
                        mb_ids_cpu, mb_attn_cpu, mb_pl, mb_al = pad_batch(
                            mb_idx, seqs_cpu, prompt_lens, action_lens, tokenizer.pad_token_id
                        )
                        mb_ids = mb_ids_cpu.to(device, non_blocking=True)
                        mb_attn = mb_attn_cpu.to(device, non_blocking=True)

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
                        del logits, outputs, mb_base_logp, mb_ids, mb_attn, mb_ids_cpu, mb_attn_cpu

                    if world_size > 1:
                        base_logp_gpu = base_logp_cpu.to(device)
                        torch.distributed.all_reduce(base_logp_gpu, op=torch.distributed.ReduceOp.SUM)
                        base_logp_cpu = base_logp_gpu.cpu()
                        del base_logp_gpu
                finally:
                    model.enable_adapter_layers()

        t_stats = time.time()

        # ---- 7. GRPO training [DISTRIBUTED: gradient accumulation + all-reduce] ----
        model.train()

        policy_loss_acc = 0.0
        kl_loss_acc = 0.0
        approx_kl_acc = 0.0
        kl_base_acc = 0.0
        clip_frac_acc = 0.0
        ratio_mean_acc = 0.0
        local_updates = 0

        t_minibatch_start = time.time()

        for _epoch in range(args.epochs):
            # Shuffle the ORDER of mini-batches — same seed on all ranks for
            # deterministic sharding (each rank gets a disjoint subset).
            batch_order = list(range(len(train_batches)))
            rng = random.Random(it * 1000 + _epoch)
            rng.shuffle(batch_order)

            # Shard mini-batches across ranks
            my_batch_order, n_total_batches = shard_batches(batch_order, rank, world_size)

            # Zero gradients once — accumulate across all local mini-batches
            optim.zero_grad(set_to_none=True)

            for bi in my_batch_order:
                mb_idx = train_batches[bi]
                mb_ids_cpu, mb_attn_cpu, mb_pl, mb_al = pad_batch(
                    mb_idx, seqs_cpu, prompt_lens, action_lens, tokenizer.pad_token_id
                )

                mb_ids = mb_ids_cpu.to(device, non_blocking=True)
                mb_attn = mb_attn_cpu.to(device, non_blocking=True)
                mb_old_logp = old_logp_cpu[mb_idx].to(device)
                mb_adv = advantages[mb_idx].to(device)
                mb_base_logp = base_logp_cpu[mb_idx].to(device) if base_logp_cpu is not None else None

                # Forward pass (always autocast to match step 6 precision)
                with autocast_ctx(device):
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
                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(
                        ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps
                    ) * mb_adv
                    policy_loss = -torch.mean(torch.min(surr1, surr2))
                else:
                    policy_loss = -torch.mean(ratio * mb_adv)

                # KL penalty against base model
                kl_loss = torch.tensor(0.0, device=device)
                if args.kl_coef > 0 and mb_base_logp is not None:
                    kl_loss = args.kl_coef * torch.mean(new_logp - mb_base_logp)

                # Scale loss for gradient accumulation: divide by LOCAL batch count
                # so accumulated gradient = local mean.  allreduce_coalesced_grads
                # then averages across ranks (SUM / world_size) → correct global mean.
                # Using n_total_batches here would under-scale by 1/world_size because
                # the allreduce already divides by world_size.
                loss = (policy_loss + kl_loss) / len(my_batch_order)
                loss.backward()  # Gradients accumulate (no zero_grad per mini-batch)

                # Tracking (no grad, unscaled losses for logging)
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

                local_updates += 1
                policy_loss_acc += float(policy_loss.item())
                kl_loss_acc += float(kl_loss.item())
                approx_kl_acc += float(approx_kl.item())
                kl_base_acc += float(approx_kl_base.item())
                clip_frac_acc += float(clip_frac.item())
                ratio_mean_acc += float(ratio_mean_val.item())

                del mb_ids, mb_attn, mb_ids_cpu, mb_attn_cpu, logits, outputs, new_logp, ratio, loss

            # Per-rank timing for diagnosing NCCL timeout issues
            t_minibatch_end = time.time()
            t_local_compute = t_minibatch_end - t_minibatch_start
            if t_local_compute > 1800:  # log if compute > 30 min (anomaly)
                print(
                    f"[SLOW rank {rank}] iter {it} mini-batch compute took "
                    f"{t_local_compute:.1f}s ({len(my_batch_order)} batches, "
                    f"{t_local_compute/max(1,len(my_batch_order)):.1f}s/batch)",
                    flush=True,
                )

            # All-reduce accumulated LoRA gradients across ranks
            t_ar_start = time.time()
            allreduce_coalesced_grads(trainable_params)
            t_ar_end = time.time()
            if t_ar_end - t_ar_start > 600:  # log if allreduce wait > 10 min
                print(
                    f"[SLOW allreduce rank {rank}] iter {it} allreduce_grads waited "
                    f"{t_ar_end - t_ar_start:.1f}s (local compute was {t_local_compute:.1f}s)",
                    flush=True,
                )

            # Single gradient clip + optimizer step (identical on all ranks)
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optim.step()

        # Aggregate metrics across ranks for correct averaging
        if world_size > 1:
            local_metrics = {
                "approx_kl": approx_kl_acc,
                "clip_frac": clip_frac_acc,
                "kl_base": kl_base_acc,
                "kl_loss": kl_loss_acc,
                "n_updates": float(local_updates),
                "policy_loss": policy_loss_acc,
                "ratio_mean": ratio_mean_acc,
            }
            agg = allreduce_scalars(local_metrics)
            total_updates = int(agg["n_updates"])
            policy_loss_acc = agg["policy_loss"]
            kl_loss_acc = agg["kl_loss"]
            approx_kl_acc = agg["approx_kl"]
            kl_base_acc = agg["kl_base"]
            clip_frac_acc = agg["clip_frac"]
            ratio_mean_acc = agg["ratio_mean"]
        else:
            total_updates = local_updates

        global_step += n_total_batches  # preserve approximate scale with original
        updates = total_updates

        t_train1 = time.time()

        # Free tokenized sequences (padded tensors are freed per-batch)
        del seqs_cpu

        # ---- 8. Logging (rank 0 only) ----
        if is_main_rank():
            avg_advantage = float(advantages.mean().item())
            all_rewards = torch.tensor([s.reward for s in samples], dtype=torch.float32)
            avg_reward = float(all_rewards.mean().item())

            pad_eff = compute_padding_efficiency(seq_lens, train_batches)

            logs = {
                "iter": it,
                "global_step": global_step,
                "grpo/policy_loss": policy_loss_acc / max(1, updates),
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
                "time/tokenize_sec": t_tok1 - t_tok0,
                "time/pad_cache_sec": t_pad - t_tok1,
                "time/stats_sec": t_stats - t_pad,
                "time/grad_sec": t_train1 - t_stats,
                "opt/padding_efficiency": pad_eff,
                "opt/info_turns_filtered": dt_filtered,
                "opt/tool_results_truncated": total_tool_results_truncated,
                "ds/informative_groups": ds_informative_groups,
                "ds/total_groups_collected": ds_total_groups_collected,
                "dist/world_size": world_size,
                **env_metrics,
            }

            # Per-game metrics (only meaningful in multi-game mode)
            if game_mix:
                per_game_rewards: Dict[str, List[float]] = defaultdict(list)
                per_game_samples: Dict[str, int] = defaultdict(int)
                for s in samples:
                    gn = s.game_name or "unknown"
                    per_game_rewards[gn].append(s.reward)
                    per_game_samples[gn] += 1
                for gn, rewards in per_game_rewards.items():
                    logs[f"env/{gn}/avg_reward"] = sum(rewards) / len(rewards)
                    logs[f"env/{gn}/num_samples"] = per_game_samples[gn]

            print(
                f"[iter {it}] step={global_step} "
                f"reward={avg_reward:.3f} avg_r={env_metrics['env/avg_reward_p0']:.3f} "
                f"abs_adv={avg_abs_advantage:.3f} "
                f"invalid={env_metrics['env/invalid_move_rate']:.3f} "
                f"samples={N} info_filt={dt_filtered} const_filt={num_filtered} "
                f"trunc={total_tool_results_truncated} "
                f"KL_rollout={logs['grpo/approx_kl_rollout']:.4f} "
                f"KL_base={logs['grpo/approx_kl_base']:.4f} "
                f"collect={logs['time/collect_sec']:.1f}s train={logs['time/train_sec']:.1f}s "
                f"(tok={t_tok1 - t_tok0:.1f}s pad={t_pad - t_tok1:.1f}s "
                f"stats={t_stats - t_pad:.1f}s grad={t_train1 - t_stats:.1f}s "
                f"pad_eff={pad_eff:.1%})"
            )
            if game_mix:
                parts = [f"{gn}={sum(r)/len(r):.3f}({len(r)})" for gn, r in per_game_rewards.items()]
                print(f"  [mix] " + " | ".join(parts))

            if wandb:
                wandb.log(logs, step=global_step)

        # ---- 9. Save checkpoint: 1st iter, 2nd iter, then every save_every (rank 0 only) ----
        if it % args.save_every == 0:
            if is_main_rank():
                ckpt_dir = output_dir_path / f"grpo_ckpt_iter_{it}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(ckpt_dir))
                tokenizer.save_pretrained(str(ckpt_dir))
                print(f"[checkpoint] Saved to {ckpt_dir}")
            barrier()

    # ---- Final save (rank 0 only) ----
    if is_main_rank():
        final_dir = output_dir_path / f"grpo_{game}_final"
        final_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        print(f"[GRPO] Training complete! Saved to {final_dir}")

    dist_cleanup()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        dist_cleanup()
