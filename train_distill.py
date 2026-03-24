#!/usr/bin/env python3
"""
On-policy cross-stage distillation trainer (GLM-5 §3.5 / SkyRL-style).

Supports two loss functions (--loss-type):

  reverse_kl (default):
    L = E_{y~student}[ sum_t (log π_student(y_t|...) - log π_teacher(y_t|...))^2 ]
    Token-level logprob MSE. Only 2 forward passes (student + teacher).
    Pushes student up where teacher is more confident, down where less.
    Converges when student per-token logprobs match teacher's.

  ppo_surrogate:
    Per-token advantages from teacher–student logprob gap with PPO clipping.
    Requires 3 forward passes (old student + teacher + new student).
    This is what SkyRL/GLM-5 actually implement (KL gap as PPO reward).

Teacher modes:
  - --teacher-url + --teacher-model: vLLM teacher(s). Both accept per-skill mappings.
    Single server:  --teacher-url http://localhost:9000 --teacher-model model_name
    Multi-server:   --teacher-url "skill1=http://host1:9000,skill2=http://host2:9001"
                    --teacher-model "skill1=model_a,skill2=model_b"
  - --teacher-url + --teacher-adapters: Per-skill LoRA routing on one multi-LoRA server.
  - --teacher-adapter /path: Local LoRA adapter (single teacher, no vLLM).
  - No teacher flag: base model is teacher.

Usage:
    # Multiple teacher models on separate vLLM servers:
    torchrun --nproc_per_node=5 train_distill.py \\
        --teacher-url "structured_data_reasoning=http://localhost:9000,tau_tool_calling=http://localhost:9001" \\
        --teacher-model "structured_data_reasoning=sdr-teacher,tau_tool_calling=tc-teacher" \\
        --loss-type reverse_kl \\
        --games "structured_data_reasoning:0.5,tau_tool_calling:0.5" \\
        --groups-per-batch 64 --lr 1e-5

    # Single teacher (full model on separate vLLM):
    torchrun --nproc_per_node=5 train_distill.py \\
        --teacher-url http://localhost:9000 \\
        --teacher-model tarsur909/Qwen3-30B-A3B-Instruct-2507-structured-10 \\
        --games "structured_data_reasoning:1.0" \\
        --groups-per-batch 64 --lr 1e-5

    # Per-skill LoRAs on one multi-LoRA vLLM server:
    torchrun --nproc_per_node=5 train_distill.py \\
        --teacher-url http://localhost:9000 \\
        --teacher-model Qwen/Qwen3-30B-A3B-Instruct-2507 \\
        --teacher-adapters "structured_data_reasoning=sdr_lora,tau_tool_calling=tc_lora" \\
        --games "structured_data_reasoning:0.5,tau_tool_calling:0.5" \\
        --groups-per-batch 64 --lr 1e-5
"""
import argparse
import concurrent.futures
import json
import os
import random
import requests
import time

from loguru import logger as _loguru_logger
_loguru_logger.remove()
_loguru_logger.disable("tau2")

import torch
import torch.nn.functional as F
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from config import Config, setup_environment, autocast_ctx
from game_registry import get_game_spec, list_game_names, GameSpec, GameMix, GameMixEntry
from inference import (
    InferenceBackend,
    HFLocalBackend,
    init_inference_backend,
)
from ppo import (
    JSONLLogger,
    build_prompt_plus_action,
    logprob_action_tokens,
    per_token_action_logprobs,
)

from train_grpo import (
    GRPOSample,
    collect_grpo_rollouts,
)
from train_grpo_optimized import (
    make_sorted_batches,
    pad_batch,
    pad_and_pin,
    compute_padding_efficiency,
    compress_tool_results,
    filter_info_gathering_turns,
    parse_game_mix,
    build_env_kwargs,
)
from sft_buffer import SFTBuffer
from dist_utils import (
    dist_pre_init, dist_nccl_init,
    dist_cleanup, is_main_rank, broadcast_objects,
    shard_batches, allreduce_coalesced_grads, allreduce_scalars,
    barrier, suppress_print,
)

try:
    import wandb
except Exception:
    wandb = None


# ============================================================================
# Argument parsing
# ============================================================================

def parse_distill_args():
    parser = argparse.ArgumentParser(
        description="On-policy cross-stage distillation (GLM-5 §3.5)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- Teacher --
    parser.add_argument("--teacher-adapter", type=str, default=None,
        help="Path to teacher LoRA checkpoint (loaded locally). Single teacher only.")
    parser.add_argument("--teacher-url", type=str, default=None,
        help="vLLM server URL(s). Single URL: 'http://localhost:9000'. "
             "Per-skill: 'skill1=http://host1:9000,skill2=http://host2:9001'. "
             "Supports multi-LoRA per-skill routing with --teacher-adapters.")
    parser.add_argument("--teacher-model", type=str, default=None,
        help="Teacher model name(s). Single: 'model_name'. "
             "Per-skill: 'skill1=model_a,skill2=model_b'. "
             "Used as the 'model' field in the vLLM API request.")
    parser.add_argument("--teacher-adapters", type=str, default=None,
        help="Per-skill teacher LoRA mapping for vLLM multi-LoRA server. "
             "Format: 'skill=lora_name,...' or 'single_name' for all skills. "
             "Example: 'structured_data_reasoning=sdr_lora,tau_tool_calling=tc_lora'. "
             "LoRA names must match --lora-modules on the teacher vLLM server.")
    parser.add_argument("--teacher-concurrency", type=int, default=16,
        help="Max concurrent requests to teacher vLLM server")
    parser.add_argument("--teacher-max-loras", type=int, default=None,
        help="Max LoRA adapters loaded simultaneously on the teacher server. "
             "When set, adapters are hot-swapped between query batches. "
             "Requires --teacher-adapter-paths.")
    parser.add_argument("--teacher-adapter-paths", type=str, default=None,
        help="LoRA adapter disk paths for hot-swapping, format: 'lora_name=/path,...'. "
             "Required when --teacher-max-loras is set. "
             "Names must match those in --teacher-adapters.")

    # -- Loss function --
    parser.add_argument("--loss-type", type=str, default="reverse_kl",
        choices=["reverse_kl", "ppo_surrogate"],
        help="Loss function. 'reverse_kl': per-token KL(student||teacher), 2 fwd passes. "
             "'ppo_surrogate': clipped surrogate with teacher gap as advantage, 3 fwd passes.")

    # -- Game and model --
    parser.add_argument("--game", type=str, default="distill_game",
        help="Game to train on (from game_registry)")
    parser.add_argument("--games", type=str, default=None,
        help="Multi-game mix (e.g. 'structured_data_reasoning:0.5,multistep_task:0.25,"
             "adversarial_policy:0.25'). Overrides --game when set.")
    parser.add_argument("--model", type=str, default=None,
        help="HuggingFace model name (default: Config.MODEL_NAME)")

    # -- Rollout structure (group_size hardcoded to 1) --
    parser.add_argument("--groups-per-batch", type=int, default=64,
        help="Number of rollouts per iteration (each group_size=1)")

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
    parser.add_argument("--mini-batch-size", type=int, default=2,
        help="Mini-batch size for gradient updates")
    parser.add_argument("--stats-chunk-size", type=int, default=2,
        help="Chunk size for computing logprobs in stats phase")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
        help="Max gradient norm for clipping")

    # -- Loss function --
    parser.add_argument("--use-clipping", action="store_true", default=True,
        help="Use clipped surrogate (like PPO)")
    parser.add_argument("--no-clipping", dest="use_clipping", action="store_false",
        help="Disable clipping and use pure importance sampling")
    parser.add_argument("--clip-eps", type=float, default=0.2,
        help="Clipping epsilon")

    # -- Generation --
    parser.add_argument("--temperature", type=float, default=1.0,
        help="Sampling temperature for rollouts")
    parser.add_argument("--temperature-range", type=str, default=None,
        help="Comma-separated temperatures for per-game variation")
    parser.add_argument("--compact-tools", action="store_true", default=False,
        help="Use compressed tool schemas for training")

    # -- Training sample optimizations --
    parser.add_argument("--filter-info-turns", action="store_true", default=True,
        help="Filter out info-gathering tool calls from training samples")
    parser.add_argument("--no-filter-info-turns", dest="filter_info_turns",
        action="store_false")
    parser.add_argument("--tool-result-max-chars", type=int, default=200,
        help="Max chars for truncated old tool results (0 to disable)")

    # -- SFT joint training --
    parser.add_argument("--sft-data", type=str, default=None,
        help="Comma-separated paths to tau2-bench eval JSON files for SFT joint training")
    parser.add_argument("--sft-coef", type=float, default=0.1,
        help="SFT loss weight")
    parser.add_argument("--sft-per-step", type=int, default=2,
        help="SFT samples per mini-batch step")

    # -- Checkpointing --
    parser.add_argument("--save-every", type=int, default=5,
        help="Save checkpoint every N iterations")
    parser.add_argument("--resume", type=str, default=None,
        help="Path to checkpoint directory to resume from")

    # -- User LLM (for adversarial_policy game) --
    parser.add_argument("--user-llm-url", type=str, default=None,
        help="OpenAI-compatible base URL for user LLM")
    parser.add_argument("--user-llm-model", type=str, default=None,
        help="Model name for user LLM server")
    parser.add_argument("--user-llm-temperature", type=float, default=0.7,
        help="Temperature for user LLM generation")
    parser.add_argument("--user-llm-max-tokens", type=int, default=1024,
        help="Max tokens for user LLM generation")

    # -- Adversarial policy game --
    parser.add_argument("--adversarial-ratio", type=float, default=1.0,
        help="Ratio of adversarial vs cooperative scenarios")
    parser.add_argument("--prefix-ratio", type=float, default=0.4,
        help="Probability of auto-playing lookup prefix per group")

    # -- Tau tool-calling game --
    parser.add_argument("--tau-domain", type=str, default=None,
        choices=["airline", "retail"],
        help="Domain filter for tau_tool_calling game")

    # -- Distributed training --
    parser.add_argument("--dist-lr-scale", type=float, default=1.0,
        help="Scale learning rate for distributed training")

    # -- Root directory --
    parser.add_argument("--root", type=str, default=None,
        help="Root directory for cache and outputs")

    return parser.parse_args()


# ============================================================================
# Per-skill teacher routing helpers
# ============================================================================

def parse_skill_mapping(spec: str) -> Dict[str, str]:
    """Parse a skill-to-value mapping specification.

    Supports two formats:
      Single value:  "http://localhost:9000"  → {"__default__": "http://localhost:9000"}
      Per-skill:     "skill1=val1,skill2=val2" → {"skill1": "val1", "skill2": "val2"}

    A value without '=' in per-skill format sets the __default__.
    """
    if not spec:
        return {}
    if "=" not in spec:
        return {"__default__": spec.strip()}
    mapping = {}
    for entry in spec.split(","):
        entry = entry.strip()
        if "=" in entry:
            key, val = entry.split("=", 1)
            mapping[key.strip()] = val.strip()
        else:
            mapping["__default__"] = entry.strip()
    return mapping


def get_skill_value(
    skill_name: str,
    mapping: Dict[str, str],
    fallback: str = "",
) -> str:
    """Look up a value by skill name, falling back to __default__ then fallback."""
    if skill_name in mapping:
        return mapping[skill_name]
    if "__default__" in mapping:
        return mapping["__default__"]
    return fallback


# ============================================================================
# vLLM teacher logprob query (per-skill routing)
# ============================================================================

def _swap_teacher_adapters(
    teacher_url: str,
    to_load: Dict[str, str],
    to_unload: List[str],
    timeout: int = 60,
    max_retries: int = 3,
) -> None:
    """Hot-swap LoRA adapters on the teacher vLLM server.

    Unloads old adapters first, then loads new ones.  Each load is preceded
    by a best-effort unload to handle adapters already present on the server
    (e.g. pre-loaded via --lora-modules at server startup).
    """
    for name in to_unload:
        try:
            resp = requests.post(
                f"{teacher_url}/v1/unload_lora_adapter",
                json={"lora_name": name},
                timeout=timeout,
            )
            if resp.status_code not in (200, 404):
                print(f"[Teacher swap] WARNING: unload '{name}' returned "
                      f"{resp.status_code}: {resp.text[:300]}")
        except Exception as e:
            print(f"[Teacher swap] WARNING: unload '{name}' exception: {e}")

    # Small delay after unloading to let vLLM release resources
    if to_unload:
        time.sleep(1.0)

    for name, path in to_load.items():
        # Best-effort unload first (handles pre-loaded adapters)
        try:
            requests.post(
                f"{teacher_url}/v1/unload_lora_adapter",
                json={"lora_name": name},
                timeout=timeout,
            )
        except Exception:
            pass

        last_err = None
        for attempt in range(max_retries):
            if attempt > 0:
                delay = 2.0 * attempt
                print(f"[Teacher swap] Retrying load '{name}' in {delay:.0f}s "
                      f"(attempt {attempt + 1}/{max_retries})...")
                time.sleep(delay)
            resp = requests.post(
                f"{teacher_url}/v1/load_lora_adapter",
                json={"lora_name": name, "lora_path": path},
                timeout=timeout,
            )
            if resp.status_code == 200:
                last_err = None
                break
            last_err = (f"load '{name}' from {path} returned "
                        f"{resp.status_code}: {resp.text[:500]}")
            print(f"[Teacher swap] WARNING: {last_err}")

        if last_err is not None:
            raise RuntimeError(f"[Teacher swap] Failed after {max_retries} "
                               f"attempts: {last_err}")


def _fetch_logprobs_one(
    teacher_url: str,
    model_name: str,
    token_ids: List[int],
    prompt_len: int,
    action_len: int,
    timeout: int = 120,
) -> torch.Tensor:
    """Fetch per-token action logprobs for a single sample from vLLM."""
    resp = requests.post(
        f"{teacher_url}/v1/completions",
        json={
            "model": model_name,
            "prompt": token_ids,
            "max_tokens": 1,
            "echo": True,
            "logprobs": 1,
            "temperature": 1.0,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()

    if "choices" not in data or not data["choices"]:
        raise RuntimeError(
            f"vLLM response missing 'choices' (model={model_name}): "
            f"{str(data)[:500]}")

    token_logprobs = data["choices"][0]["logprobs"]["token_logprobs"]

    action_lps = []
    for t in range(prompt_len, prompt_len + action_len):
        if t < len(token_logprobs) and token_logprobs[t] is not None:
            action_lps.append(token_logprobs[t])
        else:
            action_lps.append(0.0)

    return torch.tensor(action_lps, dtype=torch.float32)


def query_teacher_logprobs_vllm(
    teacher_urls: Dict[str, str],
    teacher_models: Dict[str, str],
    teacher_adapters: Dict[str, str],
    seqs_cpu: List[torch.Tensor],
    prompt_lens: List[int],
    action_lens: List[int],
    sample_indices: List[int],
    max_al: int,
    game_names: List[str],
    concurrency: int = 16,
    timeout: int = 120,
    teacher_max_loras: Optional[int] = None,
    teacher_adapter_paths: Optional[Dict[str, str]] = None,
    rank: int = 0,
    world_size: int = 1,
    dist_barrier: Optional[Any] = None,
) -> torch.Tensor:
    """Query vLLM teacher(s) for per-token action logprobs with per-skill routing.

    Resolves per-skill URL and model name for each sample:
      - teacher_urls: per-skill URL mapping (or single __default__)
      - teacher_adapters (LoRA name) takes priority over teacher_models for model name
      - teacher_models: per-skill model name mapping (or single __default__)

    When teacher_max_loras is set, hot-swaps LoRA adapters on the teacher
    server in chunks to stay within the server's --max-loras limit.

    Only queries samples at `sample_indices` (this rank's shard).
    Returns padded tensor [N, max_al] with zeros for non-shard samples
    (compatible with all-reduce SUM assembly).
    """
    N = len(seqs_cpu)
    result = torch.zeros(N, max_al, dtype=torch.float32)

    def fetch_one(idx: int) -> Tuple[int, torch.Tensor]:
        skill = game_names[idx]
        url = get_skill_value(skill, teacher_urls)
        # LoRA adapter name takes priority over base model name
        model_name = get_skill_value(skill, teacher_adapters, "")
        if not model_name:
            model_name = get_skill_value(skill, teacher_models)
        lps = _fetch_logprobs_one(
            url, model_name,
            seqs_cpu[idx].tolist(), prompt_lens[idx], action_lens[idx],
            timeout=timeout,
        )
        return idx, lps

    def _query_batch(indices: List[int]) -> None:
        """Fire concurrent queries for a batch of sample indices."""
        if not indices:
            return
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(fetch_one, i): i for i in indices}
            for future in concurrent.futures.as_completed(futures):
                try:
                    idx, lps = future.result()
                    al = action_lens[idx]
                    result[idx, :al] = lps[:al]
                except Exception as e:
                    orig_idx = futures[future]
                    print(f"[Teacher vLLM] Error for sample {orig_idx} "
                          f"(game={game_names[orig_idx]}): {e}")

    # --- No hot-swap: original path (all adapters pre-loaded on server) ---
    if teacher_max_loras is None or not teacher_adapter_paths:
        _query_batch(sample_indices)
        return result

    # --- Hot-swap path: group by adapter, process in chunks ---
    my_indices_set = set(sample_indices)

    # Group ALL samples (not just this rank's) by adapter for deterministic
    # chunking — all ranks must agree on the same chunks for barrier sync.
    all_lora_groups: Dict[str, List[int]] = defaultdict(list)
    non_lora_indices: List[int] = []
    for idx in range(N):
        skill = game_names[idx]
        adapter_name = get_skill_value(skill, teacher_adapters, "")
        if adapter_name and adapter_name in teacher_adapter_paths:
            all_lora_groups[adapter_name].append(idx)
        else:
            non_lora_indices.append(idx)

    # Query non-LoRA samples immediately (base model, no swap needed)
    _query_batch([i for i in non_lora_indices if i in my_indices_set])

    unique_adapters = sorted(all_lora_groups.keys())
    if not unique_adapters:
        return result

    # Resolve teacher base URL for swap API calls
    swap_url = (teacher_urls.get("__default__")
                or next(iter(teacher_urls.values())))

    # Chunk adapters into groups of teacher_max_loras
    adapter_chunks = [
        unique_adapters[i:i + teacher_max_loras]
        for i in range(0, len(unique_adapters), teacher_max_loras)
    ]

    loaded: set = set()

    for ci, chunk in enumerate(adapter_chunks):
        chunk_set = set(chunk)
        to_unload = [n for n in loaded if n not in chunk_set]
        to_load = {n: teacher_adapter_paths[n]
                   for n in chunk if n not in loaded}

        if to_unload or to_load:
            # Only rank 0 performs the swap; all ranks wait for completion
            if rank == 0:
                t0 = time.time()
                _swap_teacher_adapters(swap_url, to_load, to_unload)
                print(f"[Teacher swap] chunk {ci+1}/{len(adapter_chunks)}: "
                      f"-{to_unload} +{list(to_load)} ({time.time() - t0:.1f}s)")
            if dist_barrier and world_size > 1:
                dist_barrier()
            loaded = (loaded - set(to_unload)) | set(to_load)

        # Collect this rank's indices for this chunk's adapters
        chunk_all_indices = []
        for adapter_name in chunk:
            chunk_all_indices.extend(all_lora_groups[adapter_name])
        _query_batch([i for i in chunk_all_indices if i in my_indices_set])

        # Barrier: ensure all ranks finish querying before next swap
        if dist_barrier and world_size > 1:
            dist_barrier()

    return result


# ============================================================================
# Main training loop
# ============================================================================

def main():
    from unsloth import FastLanguageModel

    # ---- Distributed init (Phase 1: CUDA device only, NO NCCL yet) ----
    rank, world_size, local_rank = dist_pre_init()
    if rank != 0:
        suppress_print()

    args = parse_distill_args()

    # Parse teacher mappings (all support single value or per-skill)
    teacher_urls = parse_skill_mapping(args.teacher_url or "")
    teacher_models = parse_skill_mapping(args.teacher_model or "")
    teacher_adapters = parse_skill_mapping(args.teacher_adapters or "")
    teacher_adapter_paths = parse_skill_mapping(args.teacher_adapter_paths or "")
    has_teacher_url = bool(teacher_urls)

    # Validate teacher args
    if args.teacher_adapter and has_teacher_url:
        raise ValueError("--teacher-adapter and --teacher-url are mutually exclusive")
    if has_teacher_url and not teacher_models and not teacher_adapters:
        raise ValueError("--teacher-model or --teacher-adapters required with --teacher-url")
    if teacher_adapters and not has_teacher_url:
        raise ValueError("--teacher-adapters requires --teacher-url")
    if args.teacher_max_loras is not None and not teacher_adapter_paths:
        raise ValueError("--teacher-max-loras requires --teacher-adapter-paths")
    if args.teacher_max_loras is not None:
        for skill, adapter_name in teacher_adapters.items():
            if adapter_name not in teacher_adapter_paths:
                raise ValueError(
                    f"--teacher-adapter-paths missing path for adapter '{adapter_name}' "
                    f"(skill '{skill}'). Have: {list(teacher_adapter_paths.keys())}")

    # Apply distributed LR scaling
    if world_size > 1 and args.dist_lr_scale != 1.0:
        args.lr *= args.dist_lr_scale

    # Fixed: group_size=1 for distillation (teacher provides dense signal)
    group_size = 1

    # ---- Game setup ----
    if args.games:
        game_mix = parse_game_mix(args.games, args)
        game_spec = game_mix.entries[0].game_spec
        game = "+".join(e.game_spec.name for e in game_mix.entries)
        max_gen_tokens = game_mix.max_gen_tokens
        env_kwargs: Dict[str, Any] = game_mix.entries[0].env_kwargs
        print(f"[GameMix] {len(game_mix.entries)} games: "
              + ", ".join(f"{e.game_spec.name}:{e.weight:.2f}" for e in game_mix.entries))
    else:
        game_mix = None
        game = args.game
        try:
            game_spec = get_game_spec(game)
        except KeyError as e:
            available = ", ".join(list_game_names())
            raise RuntimeError(f"{e}. Available games: {available}") from e
        max_gen_tokens = game_spec.max_gen_tokens
        env_kwargs = build_env_kwargs(game_spec, args)

    # Dynamic rollout log name
    args.rollout_log = f"rollouts_distill_{game}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

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

    total_rollouts_per_iter = args.groups_per_batch  # group_size=1

    # Determine teacher mode
    has_multi_url = has_teacher_url and "__default__" not in teacher_urls
    has_multi_model = bool(teacher_models) and "__default__" not in teacher_models
    if has_teacher_url and (teacher_adapters or has_multi_url or has_multi_model):
        teacher_mode = "vllm_multiskill"
    elif has_teacher_url:
        teacher_mode = "vllm"
    elif args.teacher_adapter:
        teacher_mode = "adapter"
    else:
        teacher_mode = "base_model"

    print("=" * 60)
    print("ON-POLICY DISTILLATION TRAINER (GLM-5 §3.5 / SkyRL)")
    print("=" * 60)
    print(f"  Game:                {game}")
    if game_mix:
        for e in game_mix.entries:
            print(f"    - {e.game_spec.name}: weight={e.weight:.2f}, max_gen={e.game_spec.max_gen_tokens}")
    print(f"  Teacher mode:        {teacher_mode}")
    print(f"  Loss type:           {args.loss_type}")
    if args.teacher_adapter:
        print(f"  Teacher adapter:     {args.teacher_adapter}")
    if has_teacher_url:
        if len(teacher_urls) == 1 and "__default__" in teacher_urls:
            print(f"  Teacher URL:         {teacher_urls['__default__']}")
        else:
            print(f"  Teacher URLs:")
            for skill, url in teacher_urls.items():
                print(f"    {skill} -> {url}")
        if len(teacher_models) == 1 and "__default__" in teacher_models:
            print(f"  Teacher model:       {teacher_models['__default__']}")
        elif teacher_models:
            print(f"  Teacher models:")
            for skill, m in teacher_models.items():
                print(f"    {skill} -> {m}")
        print(f"  Teacher concurrency: {args.teacher_concurrency}")
    if teacher_adapters:
        print(f"  Teacher adapters:")
        for skill, lora in teacher_adapters.items():
            print(f"    {skill} -> {lora}")
    if args.teacher_max_loras:
        print(f"  Teacher max LoRAs:   {args.teacher_max_loras} (hot-swap enabled)")
        print(f"  Adapter paths:")
        for name, path in teacher_adapter_paths.items():
            print(f"    {name} -> {path}")
    print(f"  Rollouts/iter:       {total_rollouts_per_iter} (group_size=1)")
    print(f"  LoRA rank:           {args.lora_rank}")
    print(f"  LoRA alpha:          {args.lora_alpha}")
    print(f"  Learning rate:       {args.lr}")
    print(f"  Use clipping:        {args.use_clipping}")
    print(f"  Temperature:         {args.temperature}")
    if temperature_range:
        print(f"  Temperature range:   {temperature_range}")
    print(f"  Compact tools:       {args.compact_tools}")
    print(f"  Max gen tokens:      {max_gen_tokens}")
    print(f"  Device:              {device}")
    print(f"  Output dir:          {output_dir_path}")
    print(f"  Filter info turns:   {args.filter_info_turns}")
    print(f"  Tool result trunc:   {args.tool_result_max_chars} chars")
    if world_size > 1:
        print(f"  [DIST] world_size:   {world_size}")
        print(f"  [DIST] rank:         {rank}")
        print(f"  [DIST] lr_scale:     {args.dist_lr_scale}")
    print("=" * 60)

    # ---- Load model + LoRA ----
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

    # ---- Distributed init (Phase 2: NCCL) ----
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
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    FastLanguageModel.for_training(model)

    # ---- Load teacher adapter (if provided) ----
    has_teacher_adapter = False
    if args.teacher_adapter:
        teacher_path = Path(args.teacher_adapter)
        print(f"[Teacher] Loading adapter from {teacher_path}...")
        model.load_adapter(str(teacher_path), adapter_name="teacher")
        # Freeze teacher parameters
        model.set_adapter("teacher")
        for p in model.parameters():
            if p.requires_grad:
                p.requires_grad = False
        # Switch back to student (default) adapter
        model.set_adapter("default")
        has_teacher_adapter = True
        print(f"[Teacher] Adapter loaded and frozen")

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
    print(f"[Model] Loaded — distillation mode (teacher={teacher_mode})")

    # ---- Save initial adapter for eval baseline (rank 0 only) ----
    base_model_adapter_dir = output_dir_path / "base_model_adapter_distill"
    if not args.resume and is_main_rank():
        base_model_adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(base_model_adapter_dir))
        print(f"[Base] Saved initial adapter to {base_model_adapter_dir}")
    barrier()

    # ---- Initialize inference backend (rank 0 only) ----
    if is_main_rank():
        inference_backend = init_inference_backend(model, tokenizer, device)
        vllm_adapter_dir = output_dir_path / "vllm_adapter_latest_distill"
        if hasattr(inference_backend, "sync_base_adapter"):
            inference_backend.sync_base_adapter(base_model_adapter_dir)
    else:
        inference_backend = None
        vllm_adapter_dir = None

    # ---- wandb (rank 0 only) ----
    if wandb and is_main_rank():
        if not os.getenv("WANDB_NAME"):
            os.environ["WANDB_NAME"] = f"distill-{game}-{int(time.time())}"
        wandb.login(key=os.getenv("WANDB_API_KEY", ""), relogin=True)
        wandb.init(entity="forge_scaling_intelligence_lab", project=os.getenv("WANDB_PROJECT", "games"))
        wandb.config.update({
            "trainer": "distill",
            "teacher_mode": teacher_mode,
            "teacher_adapter": args.teacher_adapter,
            "teacher_urls": dict(teacher_urls),
            "teacher_models": dict(teacher_models),
            "teacher_adapters": dict(teacher_adapters),
            "loss_type": args.loss_type,
            "game": game,
            "games": args.games or args.game,
            "groups_per_batch": args.groups_per_batch,
            "group_size": 1,
            "lora_rank": args.lora_rank,
            "lr": args.lr,
            "use_clipping": args.use_clipping,
            "temperature": args.temperature,
            "world_size": world_size,
        })
        print("[wandb] Initialized")

    # ---- Optimizer (student LoRA params only) ----
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable_params, lr=args.lr)
    rollout_logger = JSONLLogger(rollout_log_path) if is_main_rank() else None

    print(f"[Distill] Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    print(f"[Distill] Starting training loop...")

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
    # Main distillation loop
    # ======================================================================
    for it in tqdm(range(start_iter, 10_000), desc="Distill iters", initial=start_iter, total=10_000):
        t_iter_start = time.time()

        # ---- 1. Sync student policy to vLLM (rank 0 only) ----
        if is_main_rank():
            inference_backend.sync_policy(model, vllm_adapter_dir)
            if not inference_backend.is_enabled():
                inference_backend = HFLocalBackend(model, tokenizer, device)
        barrier()

        # ---- 2. Collect rollouts with group_size=1 (rank 0) + broadcast ----
        if is_main_rank():
            if game_mix:
                iter_group_assignments = game_mix.assign_groups(args.groups_per_batch)
            else:
                iter_group_assignments = None

            t_collect0 = time.time()
            samples, env_metrics = collect_grpo_rollouts(
                backend=inference_backend,
                tokenizer=tokenizer,
                game_spec=game_spec,
                groups_per_batch=args.groups_per_batch,
                group_size=group_size,  # =1
                temperature=args.temperature,
                max_new_tokens=max_gen_tokens,
                base_seed=it,
                logger=rollout_logger,
                env_kwargs=env_kwargs,
                temperature_range=temperature_range,
                prefix_ratio=args.prefix_ratio,
                compact_tools=args.compact_tools,
                group_assignments=iter_group_assignments,
            )
            t_collect1 = time.time()
        else:
            t_collect0 = t_collect1 = time.time()
            samples, env_metrics = [], {}

        # ---- 3. Broadcast raw samples ----
        if is_main_rank():
            if not samples:
                print(f"[iter {it}] No samples collected, skipping")
                broadcast_data = [True, None, None, 0.0, 0.0]
            else:
                broadcast_data = [False, samples, env_metrics, t_collect0, t_collect1]
        else:
            broadcast_data = [None] * 5

        broadcast_data = broadcast_objects(broadcast_data)
        skip_flag = broadcast_data[0]

        if skip_flag:
            continue

        _, samples, env_metrics, t_collect0, t_collect1 = broadcast_data

        # ---- 3b. Optional: filter info-gathering turns ----
        # For distillation we create dummy zero advantages for the filter function
        dt_filtered = 0
        samples_before_dt_filter = len(samples)
        if args.filter_info_turns:
            dummy_adv = torch.zeros(len(samples))
            samples, dummy_adv, dt_filtered = filter_info_gathering_turns(
                samples, dummy_adv,
            )
            if dt_filtered > 0 and is_main_rank():
                print(
                    f"[iter {it}] Filtered {dt_filtered} info-gathering turns "
                    f"({samples_before_dt_filter} -> {len(samples)} samples)"
                )

        # ---- 4. Tokenize samples ----
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
            samples = [samples[i] for i in kept_indices]

        N = len(seqs_cpu)
        if N == 0:
            print(f"  [iter {it}] All samples exceeded max_seq_len — skipping training step")
            barrier()
            continue
        seq_lens = [s.shape[0] for s in seqs_cpu]

        t_tok1 = time.time()

        # ---- 4b. Create length-sorted batches ----
        unified_batches = args.stats_chunk_size == args.mini_batch_size
        train_batches = make_sorted_batches(seq_lens, args.mini_batch_size)

        if unified_batches:
            stats_batches = train_batches
        else:
            stats_batches = make_sorted_batches(seq_lens, args.stats_chunk_size)

        t_pad = time.time()

        # ================================================================
        # 5. Stats phase: compute per-token teacher (& old student) logprobs
        # ================================================================
        model.eval()

        max_al = max(action_lens)
        teacher_logp_padded = torch.zeros(N, max_al, dtype=torch.float32)  # CPU

        # Per-sample game names for per-skill teacher routing
        game_names = [s.game_name or "unknown" for s in samples]

        # Old student logprobs only needed for ppo_surrogate loss
        need_old_student = (args.loss_type == "ppo_surrogate")
        student_logp_padded = torch.zeros(N, max_al, dtype=torch.float32) if need_old_student else None

        # Shard stats batches across ranks
        all_stats_indices = list(range(len(stats_batches)))
        my_stats_indices, _ = shard_batches(all_stats_indices, rank, world_size)

        with torch.no_grad():
            # -- Pass 1 (ppo_surrogate only): old student per-token logprobs --
            if need_old_student:
                if has_teacher_adapter:
                    model.set_adapter("default")

                for bi in my_stats_indices:
                    mb_idx = stats_batches[bi]
                    mb_ids_cpu, mb_attn_cpu, mb_pl, mb_al = pad_batch(
                        mb_idx, seqs_cpu, prompt_lens, action_lens, tokenizer.pad_token_id
                    )
                    mb_ids = mb_ids_cpu.to(device, non_blocking=True)
                    mb_attn = mb_attn_cpu.to(device, non_blocking=True)

                    with autocast_ctx(device):
                        outputs = model(
                            input_ids=mb_ids, attention_mask=mb_attn, use_cache=False,
                        )
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                    mb_per_token = per_token_action_logprobs(logits, mb_ids, mb_pl, mb_al)
                    for local_j, global_j in enumerate(mb_idx):
                        al = action_lens[global_j]
                        student_logp_padded[global_j, :al] = mb_per_token[local_j][:al].cpu()
                    del logits, outputs, mb_per_token, mb_ids, mb_attn, mb_ids_cpu, mb_attn_cpu

                if world_size > 1:
                    student_gpu = student_logp_padded.to(device)
                    torch.distributed.all_reduce(student_gpu, op=torch.distributed.ReduceOp.SUM)
                    student_logp_padded = student_gpu.cpu()
                    del student_gpu

            # -- Pass 2: teacher per-token logprobs --
            if has_teacher_url:
                # --- vLLM teacher (single or multi-server/skill routing) ---
                my_sample_indices = list(range(rank, N, world_size))
                teacher_logp_padded = query_teacher_logprobs_vllm(
                    teacher_urls=teacher_urls,
                    teacher_models=teacher_models,
                    teacher_adapters=teacher_adapters,
                    seqs_cpu=seqs_cpu,
                    prompt_lens=prompt_lens,
                    action_lens=action_lens,
                    sample_indices=my_sample_indices,
                    max_al=max_al,
                    game_names=game_names,
                    concurrency=args.teacher_concurrency,
                    teacher_max_loras=args.teacher_max_loras,
                    teacher_adapter_paths=teacher_adapter_paths if args.teacher_max_loras else None,
                    rank=rank,
                    world_size=world_size,
                    dist_barrier=barrier,
                )
            else:
                # --- Local teacher: adapter switch or base model ---
                if has_teacher_adapter:
                    model.set_adapter("teacher")
                else:
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
                                input_ids=mb_ids, attention_mask=mb_attn, use_cache=False,
                            )
                        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                        mb_per_token = per_token_action_logprobs(logits, mb_ids, mb_pl, mb_al)
                        for local_j, global_j in enumerate(mb_idx):
                            al = action_lens[global_j]
                            teacher_logp_padded[global_j, :al] = mb_per_token[local_j][:al].cpu()
                        del logits, outputs, mb_per_token, mb_ids, mb_attn, mb_ids_cpu, mb_attn_cpu
                finally:
                    if has_teacher_adapter:
                        model.set_adapter("default")
                    else:
                        model.enable_adapter_layers()

            # All-reduce teacher logprobs
            if world_size > 1:
                teacher_gpu = teacher_logp_padded.to(device)
                torch.distributed.all_reduce(teacher_gpu, op=torch.distributed.ReduceOp.SUM)
                teacher_logp_padded = teacher_gpu.cpu()
                del teacher_gpu

        t_stats = time.time()

        # ================================================================
        # 6. Training phase
        # ================================================================
        model.train()

        policy_loss_acc = 0.0
        sft_loss_acc = 0.0
        clip_frac_acc = 0.0
        ratio_mean_acc = 0.0
        approx_kl_acc = 0.0
        teacher_gap_acc = 0.0
        local_updates = 0

        # Per-game gap/loss tracking
        per_game_gap: Dict[str, List[float]] = defaultdict(list)
        per_game_loss: Dict[str, List[float]] = defaultdict(list)

        eps = args.clip_eps
        use_reverse_kl = (args.loss_type == "reverse_kl")

        for _epoch in range(args.epochs):
            batch_order = list(range(len(train_batches)))
            rng = random.Random(it * 1000 + _epoch)
            rng.shuffle(batch_order)

            my_batch_order, n_total_batches = shard_batches(batch_order, rank, world_size)

            optim.zero_grad(set_to_none=True)

            for bi in my_batch_order:
                mb_idx = train_batches[bi]
                mb_ids_cpu, mb_attn_cpu, mb_pl, mb_al = pad_batch(
                    mb_idx, seqs_cpu, prompt_lens, action_lens, tokenizer.pad_token_id
                )
                mb_ids = mb_ids_cpu.to(device, non_blocking=True)
                mb_attn = mb_attn_cpu.to(device, non_blocking=True)

                # Forward pass — compute new per-token logprobs
                with autocast_ctx(device):
                    outputs = model(
                        input_ids=mb_ids, attention_mask=mb_attn, use_cache=False,
                    )
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                    new_logp_list = per_token_action_logprobs(logits, mb_ids, mb_pl, mb_al)

                total_loss = torch.tensor(0.0, device=device)
                mb_clip_frac = 0.0
                mb_ratio_mean = 0.0
                mb_approx_kl = 0.0
                mb_gap = 0.0

                for local_j, global_j in enumerate(mb_idx):
                    al = action_lens[global_j]
                    new_lp = new_logp_list[local_j][:al]                       # [al] — has grad
                    teacher_lp = teacher_logp_padded[global_j, :al].to(device) # [al] — detached

                    if use_reverse_kl:
                        # ---- Logprob MSE: L = mean((student_lp - teacher_lp)^2) ----
                        # Gradient: 2*(student_lp - teacher_lp) * ∂student_lp/∂θ
                        #   student > teacher → pushes student DOWN
                        #   student < teacher → pushes student UP
                        # Converges when student = teacher. Only 2 forward passes.
                        sample_loss = ((new_lp - teacher_lp) ** 2).mean()

                        with torch.no_grad():
                            mb_gap += (teacher_lp - new_lp.detach()).mean().item()
                            mb_approx_kl += (new_lp.detach() - teacher_lp).mean().item()
                    else:
                        # ---- PPO surrogate with teacher gap as advantage ----
                        old_lp = student_logp_padded[global_j, :al].to(device) # [al] — detached
                        ratio = torch.exp(new_lp - old_lp)
                        advantage = (teacher_lp - old_lp).detach()

                        if args.use_clipping:
                            surr1 = ratio * advantage
                            surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * advantage
                            sample_loss = -torch.min(surr1, surr2).mean()
                        else:
                            sample_loss = -(ratio * advantage).mean()

                        with torch.no_grad():
                            mb_clip_frac += (torch.abs(ratio - 1.0) > eps).float().mean().item()
                            mb_ratio_mean += ratio.mean().item()
                            mb_approx_kl += (old_lp - new_lp.detach()).mean().item()
                            mb_gap += (teacher_lp - old_lp).mean().item()

                    total_loss = total_loss + sample_loss

                    # Per-game gap/loss tracking
                    gn = game_names[global_j]
                    per_game_loss[gn].append(float(sample_loss.item()))
                    with torch.no_grad():
                        per_game_gap[gn].append((teacher_lp - new_lp.detach()).mean().item())

                n_mb = len(mb_idx)
                loss = total_loss / n_mb / len(my_batch_order)
                loss.backward()

                # ---- SFT forward + backward (separate graph) ----
                sft_loss_val = torch.tensor(0.0, device=device)
                if sft_buffer is not None and len(sft_buffer) > 0:
                    sft_ids, sft_attn, sft_pl, sft_al = sft_buffer.sample_batch(
                        args.sft_per_step, device
                    )
                    with autocast_ctx(device):
                        sft_out = model(
                            input_ids=sft_ids, attention_mask=sft_attn, use_cache=False
                        )
                        sft_logits = sft_out.logits if hasattr(sft_out, "logits") else sft_out[0]
                        sft_logp = logprob_action_tokens(
                            sft_logits, sft_ids, sft_pl, sft_al, normalize_by_len=True
                        )
                    sft_loss_val = -sft_logp.mean()
                    sft_scaled = (args.sft_coef * sft_loss_val) / len(my_batch_order)
                    sft_scaled.backward()
                    del sft_ids, sft_attn, sft_logits, sft_out, sft_logp, sft_scaled

                local_updates += 1
                policy_loss_acc += float(total_loss.item()) / n_mb
                sft_loss_acc += float(sft_loss_val.item())
                clip_frac_acc += mb_clip_frac / n_mb
                ratio_mean_acc += mb_ratio_mean / n_mb
                approx_kl_acc += mb_approx_kl / n_mb
                teacher_gap_acc += mb_gap / n_mb

                del mb_ids, mb_attn, mb_ids_cpu, mb_attn_cpu, logits, outputs, new_logp_list, loss, total_loss

            # All-reduce accumulated LoRA gradients across ranks
            allreduce_coalesced_grads(trainable_params)

            # Single gradient clip + optimizer step
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optim.step()

        # Aggregate metrics across ranks
        if world_size > 1:
            local_metrics = {
                "approx_kl": approx_kl_acc,
                "clip_frac": clip_frac_acc,
                "n_updates": float(local_updates),
                "policy_loss": policy_loss_acc,
                "sft_loss": sft_loss_acc,
                "ratio_mean": ratio_mean_acc,
                "teacher_gap": teacher_gap_acc,
            }
            agg = allreduce_scalars(local_metrics)
            total_updates = int(agg["n_updates"])
            policy_loss_acc = agg["policy_loss"]
            sft_loss_acc = agg["sft_loss"]
            approx_kl_acc = agg["approx_kl"]
            clip_frac_acc = agg["clip_frac"]
            ratio_mean_acc = agg["ratio_mean"]
            teacher_gap_acc = agg["teacher_gap"]
        else:
            total_updates = local_updates

        global_step += n_total_batches
        updates = total_updates

        t_train1 = time.time()

        del seqs_cpu

        # ---- 7. Logging (rank 0 only) ----
        if is_main_rank():
            all_rewards = torch.tensor([s.reward for s in samples], dtype=torch.float32)
            avg_reward = float(all_rewards.mean().item())

            pad_eff = compute_padding_efficiency(seq_lens, train_batches)

            logs = {
                "iter": it,
                "global_step": global_step,
                "distill/policy_loss": policy_loss_acc / max(1, updates),
                "distill/sft_loss": sft_loss_acc / max(1, updates),
                "distill/sft_buffer_size": len(sft_buffer) if sft_buffer is not None else 0,
                "distill/clip_frac": clip_frac_acc / max(1, updates),
                "distill/ratio_mean": ratio_mean_acc / max(1, updates),
                "distill/teacher_student_gap": teacher_gap_acc / max(1, updates),
                "distill/approx_kl": approx_kl_acc / max(1, updates),
                "distill/avg_reward": avg_reward,
                "distill/num_samples": N,
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
                "dist/world_size": world_size,
                **env_metrics,
            }

            # Per-game metrics
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

            # Per-game gap and loss
            for gn, gaps in per_game_gap.items():
                logs[f"distill/{gn}/gap"] = sum(gaps) / len(gaps)
                logs[f"distill/{gn}/loss"] = sum(per_game_loss[gn]) / len(per_game_loss[gn])
                logs[f"distill/{gn}/n_train_samples"] = len(gaps)

            print(
                f"[iter {it}] step={global_step} "
                f"reward={avg_reward:.3f} "
                f"gap={logs['distill/teacher_student_gap']:.4f} "
                f"loss={logs['distill/policy_loss']:.4f} "
                f"ratio={logs['distill/ratio_mean']:.4f} "
                f"clip={logs['distill/clip_frac']:.3f} "
                f"KL={logs['distill/approx_kl']:.4f} "
                f"samples={N} info_filt={dt_filtered} "
                f"trunc={total_tool_results_truncated} "
                f"collect={logs['time/collect_sec']:.1f}s train={logs['time/train_sec']:.1f}s "
                f"(tok={t_tok1 - t_tok0:.1f}s stats={t_stats - t_pad:.1f}s "
                f"grad={t_train1 - t_stats:.1f}s pad_eff={pad_eff:.1%})"
            )
            if game_mix:
                parts = [f"{gn}={sum(r)/len(r):.3f}({len(r)})" for gn, r in per_game_rewards.items()]
                print(f"  [mix] " + " | ".join(parts))
            if per_game_gap:
                gap_parts = [f"{gn}: gap={sum(g)/len(g):.4f} loss={sum(per_game_loss[gn])/len(per_game_loss[gn]):.4f}({len(g)})"
                             for gn, g in per_game_gap.items()]
                print(f"  [gap] " + " | ".join(gap_parts))

            if wandb:
                wandb.log(logs, step=global_step)

        # ---- 8. Save checkpoint ----
        if it % args.save_every == 0:
            if is_main_rank():
                ckpt_dir = output_dir_path / f"distill_ckpt_iter_{it}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(ckpt_dir))
                tokenizer.save_pretrained(str(ckpt_dir))
                print(f"[checkpoint] Saved to {ckpt_dir}")
            barrier()

    # ---- Final save (rank 0 only) ----
    if is_main_rank():
        final_dir = output_dir_path / f"distill_{game}_final"
        final_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        print(f"[Distill] Training complete! Saved to {final_dir}")

    dist_cleanup()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        dist_cleanup()
