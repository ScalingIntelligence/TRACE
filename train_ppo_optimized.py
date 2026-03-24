#!/usr/bin/env python3
"""
Optimized PPO trainer — based on train_ppo.py with performance improvements
ported from train_grpo_optimized.py.

PPO algorithm (value head baseline, clipped surrogate + value loss) combined
with the GRPO-optimized infrastructure (length-sorted batching, pinned memory,
distributed training, gradient accumulation, autocast, detailed timing).

Optimizations over train_ppo.py (each benchmarked for significant impact in
the GRPO-optimized trainer and ported here):
  1. Length-sorted batching: sequences sorted by length to minimize padding waste
     in forward passes. Padding efficiency goes from ~55% to ~99%.
  2. Unified batch caching: stats_chunk_size == mini_batch_size so one set of
     pre-padded batches serves both the stats pass (old_logp/old_v) and the
     training pass.
  3. Pinned-memory CPU tensors for faster async GPU transfers.
  4. KL regularization against base model (optional, default OFF). When kl_coef=0,
     skips base_logp pass entirely.
  5. Single adapter toggle: computes ALL old_logp first (adapter ON), then ALL
     base_logp (adapter OFF), toggling adapter layers only once.
  6. Training-time autocast (bfloat16) via PolicyWithValueHead.
  7. Detailed sub-step timing breakdown (tokenize, pad, stats, grad, pad_eff).
  8. Multi-GPU distributed training via torchrun (gradient accumulation +
     all-reduce).
  9. Optional info-gathering turn filtering and tool result compression
     for adversarial_policy-style games.

Algorithmic differences from train_ppo.py (beyond pure performance):
  - Gradient accumulation: gradients are accumulated across all mini-batches in
    an epoch, then a single optimizer step is taken (required for distributed
    all-reduce). train_ppo.py takes one optimizer step per mini-batch.
  - global_step increments by n_total_batches per epoch (preserves approximate
    scale with the original trainer).

Usage (drop-in replacement for train_ppo.py):
    # Single GPU:
    VLLM_BASE_URL=http://localhost:8000 python train_ppo_optimized.py \\
        --game adversarial_policy --games-per-iter 256

    # Multi-GPU (N GPUs for ~Nx speedup on training step):
    VLLM_BASE_URL=http://localhost:8000 torchrun --nproc_per_node=4 \\
        train_ppo_optimized.py --game adversarial_policy --games-per-iter 256
"""
import argparse
import json
import os
import random
import time
import torch
import torch.nn.functional as F
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from config import Config, setup_environment, autocast_ctx
from game_registry import get_game_spec, list_game_names, GameSpec
from inference import (
    InferenceBackend,
    HFLocalBackend,
    init_inference_backend,
)
from model import PolicyWithValueHead
from ppo import (
    JSONLLogger,
    EMA,
    StepSample,
    collect_games,
    build_prompt_plus_action,
    logprob_action_tokens,
    values_from_hidden,
)
from evaluation import evaluate_tau2_bench
from dist_utils import (
    dist_pre_init, dist_nccl_init, dist_cleanup, is_main_rank, broadcast_objects,
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
# Optimization helpers (ported from train_grpo_optimized.py)
# ============================================================================

def make_sorted_batches(seq_lens: List[int], batch_size: int) -> List[List[int]]:
    """
    Create batches of indices sorted by sequence length.

    Groups similar-length sequences together to minimize padding waste.
    Returns list of index-lists, each of size <= batch_size.
    """
    sorted_idx = sorted(range(len(seq_lens)), key=lambda i: seq_lens[i])
    return [sorted_idx[i:i + batch_size] for i in range(0, len(sorted_idx), batch_size)]


def pad_and_pin(seqs: List[torch.Tensor], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad sequences to max length in the batch and pin memory for fast GPU transfer.

    Returns (input_ids, attention_mask) on CPU with pinned memory.
    """
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


def pad_batch(
    batch_idx: List[int],
    seqs_cpu: List[torch.Tensor],
    prompt_lens: List[int],
    action_lens: List[int],
    pad_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
    """Pad a single mini-batch on-the-fly (no pre-caching).

    Returns (input_ids, attention_mask, prompt_lens, action_lens).
    Input tensors are on CPU with pinned memory for fast GPU transfer.
    Used instead of pre-caching all batches so each rank only allocates
    memory for batches it actually processes (multi-GPU memory parity).
    """
    mb_seqs = [seqs_cpu[j] for j in batch_idx]
    mb_ids, mb_attn = pad_and_pin(mb_seqs, pad_id)
    mb_pl = [prompt_lens[j] for j in batch_idx]
    mb_al = [action_lens[j] for j in batch_idx]
    return mb_ids, mb_attn, mb_pl, mb_al


def compute_padding_efficiency(seq_lens: List[int], batches: List[List[int]]) -> float:
    """Compute ratio of real tokens to padded tokens (1.0 = perfect, no waste)."""
    total_real = sum(seq_lens)
    total_padded = sum(
        len(batch) * max(seq_lens[i] for i in batch)
        for batch in batches
    )
    return total_real / max(1, total_padded)


# ============================================================================
# PPO-specific info-turn filtering (adapted from train_grpo_optimized.py)
# ============================================================================

# Read-only tool names — info-gathering, no policy decision.
_INFO_ONLY_TOOLS = frozenset({
    "get_user_details", "get_reservation_details",
    "search_direct_flight", "search_onestop_flight",
    "list_all_airports", "get_flight_status",
    "get_order_details", "get_product_details",
    "find_user_id_by_email", "find_user_id_by_name_zip",
    "list_all_product_types",
    "calculate",
})


def _extract_tool_name(completion_text: str) -> Optional[str]:
    """Extract tool name from a model's raw completion text."""
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


def filter_info_gathering_samples(
    samples: List[StepSample],
) -> Tuple[List[StepSample], List[int], int]:
    """Remove samples where the agent only made a read-only tool call.

    Unlike train_grpo_optimized.py's filter_info_gathering_turns(), this
    version does NOT require an advantages tensor (PPO advantages are computed
    after the stats pass, so they don't exist at filter time).

    Guarantees at least one sample per game_id is kept (the last turn)
    so the game's reward signal is preserved.

    Returns:
        (filtered_samples, keep_indices, num_filtered_out)
    """
    keep_mask = []
    for s in samples:
        tool_name = _extract_tool_name(s.completion_text)
        if tool_name is None or tool_name not in _INFO_ONLY_TOOLS:
            keep_mask.append(True)
        else:
            keep_mask.append(False)

    # Guarantee at least one sample per game_id (keep last turn)
    last_idx_per_game: Dict[int, int] = {}
    for i, s in enumerate(samples):
        last_idx_per_game[s.game_id] = i
    for last_idx in last_idx_per_game.values():
        keep_mask[last_idx] = True

    keep_indices = [i for i, keep in enumerate(keep_mask) if keep]
    num_filtered_out = len(samples) - len(keep_indices)

    if num_filtered_out == 0:
        return samples, list(range(len(samples))), 0

    filtered_samples = [samples[i] for i in keep_indices]
    return filtered_samples, keep_indices, num_filtered_out


def compress_tool_results(
    prompt_msgs: List[Dict[str, Any]],
    max_tool_result_chars: int = 200,
) -> Tuple[List[Dict[str, Any]], int]:
    """Truncate old tool results to reduce prompt length.

    Keeps the LAST tool result in full (needed for current decision).
    Truncates all earlier tool results to max_tool_result_chars.
    """
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
# Argument parsing
# ============================================================================

def parse_ppo_args_optimized():
    parser = argparse.ArgumentParser(
        description="Optimized PPO trainer (same algorithm, faster training step)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- Game and model --
    parser.add_argument("--game", type=str, default="adversarial_policy",
        help="Game to train on (from game_registry)")
    parser.add_argument("--model", type=str, default=None,
        help="HuggingFace model name (default: Config.MODEL_NAME)")

    # -- PPO structure --
    parser.add_argument("--games-per-iter", type=int, default=256,
        help="Number of self-play games per iteration")

    # -- LoRA --
    parser.add_argument("--lora-rank", type=int, default=16,
        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16,
        help="LoRA alpha scaling factor")

    # -- Optimization --
    parser.add_argument("--lr", type=float, default=1e-6,
        help="Learning rate")
    parser.add_argument("--ppo-epochs", type=int, default=1,
        help="PPO training epochs per iteration")
    parser.add_argument("--mini-batch-size", type=int, default=4,
        help="Mini-batch size for gradient updates")
    parser.add_argument("--stats-chunk-size", type=int, default=4,
        help="Chunk size for computing old logprobs/values "
             "(set equal to mini-batch-size for unified caching)")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
        help="Max gradient norm for clipping")

    # -- PPO loss function --
    parser.add_argument("--clip-eps", type=float, default=0.2,
        help="PPO clipping epsilon")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="Value function loss coefficient")
    parser.add_argument("--kl-coef", type=float, default=0.0,
        help="KL penalty against base model (0 to disable)")

    # -- Generation --
    parser.add_argument("--temperature", type=float, default=0.7,
        help="Sampling temperature for rollouts")

    # -- Role baseline --
    parser.add_argument("--use-role-baseline", action="store_true", default=True,
        help="Use EMA role baseline for advantage estimation")
    parser.add_argument("--no-role-baseline", dest="use_role_baseline",
        action="store_false", help="Disable role baseline")
    parser.add_argument("--role-baseline-gamma", type=float, default=0.95,
        help="EMA gamma for role baseline")

    # -- Checkpointing and evaluation --
    parser.add_argument("--resume", type=str, default=None,
        help="Path to checkpoint directory to resume from")
    parser.add_argument("--rollout-log", type=str, default="selfplay_rollouts_ppo.jsonl",
        help="Filename for rollout logs")
    parser.add_argument("--save-every", type=int, default=10,
        help="Save checkpoint every N iterations")
    parser.add_argument("--tau2-eval-every", type=int, default=0,
        help="Tau2-bench eval every N iterations (0 to disable)")

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
    parser.add_argument("--adversarial-ratio", type=float, default=0.6,
        help="Ratio of adversarial vs cooperative scenarios (0.0-1.0)")

    # -- Training sample optimizations --
    parser.add_argument("--filter-info-turns", action="store_true", default=True,
        help="Filter out info-gathering tool calls from training samples")
    parser.add_argument("--no-filter-info-turns", dest="filter_info_turns",
        action="store_false", help="Disable info-gathering turn filtering")
    parser.add_argument("--tool-result-max-chars", type=int, default=200,
        help="Max chars for truncated old tool results (0 to disable)")

    # -- Distributed training --
    parser.add_argument("--dist-lr-scale", type=float, default=1.0,
        help="Scale learning rate for distributed training")

    # -- Root directory --
    parser.add_argument("--root", type=str, default=None,
        help="Root directory for cache and outputs")

    return parser.parse_args()


# ============================================================================
# Main training loop (optimized)
# ============================================================================

def main():
    # Import Unsloth BEFORE dist_init() to avoid NCCL deadlock
    # (see train_grpo_optimized.py for details)
    from unsloth import FastLanguageModel

    # ---- Distributed init (Phase 1: CUDA device only, no NCCL yet) ----
    # NCCL is deferred until after model loading to prevent from_pretrained /
    # accelerate from triggering rogue collectives that cause NCCL deadlocks.
    rank, world_size, local_rank = dist_pre_init()
    if not is_main_rank():
        suppress_print()

    args = parse_ppo_args_optimized()

    # Apply distributed LR scaling
    if world_size > 1 and args.dist_lr_scale != 1.0:
        args.lr *= args.dist_lr_scale

    # Optimization flags via env vars
    use_compile = os.getenv("PPO_COMPILE", "0") == "1"

    # ---- Game setup ----
    game = args.game
    try:
        game_spec = get_game_spec(game)
    except KeyError as e:
        available = ", ".join(list_game_names())
        raise RuntimeError(f"{e}. Available games: {available}") from e

    max_gen_tokens = game_spec.max_gen_tokens
    env_kwargs: Dict[str, Any] = {}
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

    if game_spec.name == "adversarial_policy":
        env_kwargs["adversarial_ratio"] = args.adversarial_ratio

    # Set dynamic rollout log name
    args.rollout_log = f"rollouts_ppo_{game_spec.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

    # ---- Environment setup ----
    env_config = setup_environment(args)
    device = f"cuda:{local_rank}" if world_size > 1 else env_config["device"]
    hf_hub = env_config["hf_hub"]
    output_dir_path = env_config["output_dir_path"]
    rollout_log_path = env_config["rollout_log_path"]

    print("=" * 60)
    print("PPO TRAINER (OPTIMIZED)")
    print("=" * 60)
    print(f"  Game:                {game_spec.name}")
    print(f"  Games per iter:      {args.games_per_iter}")
    print(f"  LoRA rank:           {args.lora_rank}")
    print(f"  LoRA alpha:          {args.lora_alpha}")
    print(f"  Learning rate:       {args.lr}")
    print(f"  PPO epochs:          {args.ppo_epochs}")
    print(f"  Mini-batch size:     {args.mini_batch_size}")
    print(f"  Clip epsilon:        {args.clip_eps}")
    print(f"  VF coefficient:      {args.vf_coef}")
    print(f"  KL coefficient:      {args.kl_coef}")
    print(f"  Temperature:         {args.temperature}")
    print(f"  Role baseline:       {args.use_role_baseline}")
    if game_spec.name == "adversarial_policy":
        print(f"  Adversarial ratio:   {args.adversarial_ratio}")
    print(f"  Max gen tokens:      {max_gen_tokens}")
    print(f"  Device:              {device}")
    print(f"  Output dir:          {output_dir_path}")
    print(f"  [OPT] length sorting: ON")
    print(f"  [OPT] pinned memory:  ON")
    print(f"  [OPT] torch.compile:  {use_compile}")
    print(f"  [OPT] train autocast: ON (PolicyWithValueHead)")
    if args.kl_coef == 0:
        print(f"  [OPT] base_logp:      SKIPPED (kl_coef=0)")
    print(f"  [OPT] filter info turns: {args.filter_info_turns}")
    print(f"  [OPT] tool result trunc: {args.tool_result_max_chars} chars (0=off)")
    if world_size > 1:
        print(f"  [DIST] world_size:    {world_size}")
        print(f"  [DIST] rank:          {rank}")
        print(f"  [DIST] lr_scale:      {args.dist_lr_scale}")
    print("=" * 60)

    # ---- Load model + LoRA ----
    model_name = args.model or Config.MODEL_NAME
    print(f"[Model] Rank {rank}: Loading {model_name} with LoRA rank={args.lora_rank}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=Config.MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
        offload_embedding=True,
        use_exact_model_name=True,
        cache_dir=str(hf_hub),
        device_map={"": local_rank},
    )
    print(f"[Model] Rank {rank}: Model loaded successfully", flush=True)

    # ---- Distributed init (Phase 2: NCCL after model loading) ----
    # Now safe to initialize NCCL — no more from_pretrained calls that could
    # trigger rogue collectives causing asymmetric NCCL hangs.
    print(f"[NCCL] Rank {rank}: Initializing NCCL process group...", flush=True)
    dist_nccl_init()
    print(f"[NCCL] Rank {rank}: NCCL initialized", flush=True)
    barrier()

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

    # ---- Resume from checkpoint (LoRA adapter) ----
    start_iter = 0
    if args.resume:
        import safetensors.torch
        from peft import set_peft_model_state_dict

        resume_path = Path(args.resume)
        print(f"[Resume] Loading from {resume_path}")

        adapter_path = resume_path / "policy" / "adapter_model.safetensors"
        if adapter_path.exists():
            state = safetensors.torch.load_file(str(adapter_path))
            set_peft_model_state_dict(model, state)
            print(f"[Resume] Loaded adapter from {adapter_path}")
        else:
            raise FileNotFoundError(f"Adapter not found at {adapter_path}")

        try:
            start_iter = int(resume_path.name.split("_")[-1]) + 1
        except (ValueError, IndexError):
            start_iter = 0
        print(f"[Resume] Starting from iteration {start_iter}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Wrap with value head (PPO-specific) ----
    ac = PolicyWithValueHead(model, device=device).to(device)

    # ---- Resume value head ----
    if args.resume:
        value_head_path = Path(args.resume) / "value_head.pt"
        if value_head_path.exists():
            ac.value_head.load_state_dict(
                torch.load(value_head_path, map_location=device)
            )
            print(f"[Resume] Loaded value head from {value_head_path}")
        else:
            print(f"[Resume] Warning: value_head.pt not found, starting fresh value head")

    # ---- Sync value head across ranks (Bug 2 fix) ----
    # nn.Linear random init differs across ranks; broadcast from rank 0
    if world_size > 1:
        vhead_state = [ac.value_head.state_dict()] if is_main_rank() else [None]
        vhead_state = broadcast_objects(vhead_state)
        if not is_main_rank():
            ac.value_head.load_state_dict(vhead_state[0])
        barrier()
        print(f"[DIST] Value head synced across {world_size} ranks")

    print(f"[Model] Loaded with value head (PPO style)")

    # ---- Optional: torch.compile ----
    if use_compile:
        print("[OPT] Compiling model with torch.compile...")
        ac.lm = torch.compile(ac.lm)
        print("[OPT] Compilation complete")

    # ---- Save initial adapter for eval baseline (rank 0 only) ----
    base_model_adapter_dir = output_dir_path / "base_model_adapter_ppo"
    if not args.resume and is_main_rank():
        base_model_adapter_dir.mkdir(parents=True, exist_ok=True)
        ac.lm.save_pretrained(str(base_model_adapter_dir))
        print(f"[Base] Saved initial adapter to {base_model_adapter_dir}")
    elif args.resume:
        print(f"[Resume] Using existing base model adapter from {base_model_adapter_dir}")
    barrier()

    # ---- Initialize inference backend (rank 0 only for vLLM interaction) ----
    if is_main_rank():
        inference_backend = init_inference_backend(ac.lm, tokenizer, device)
        vllm_adapter_dir = output_dir_path / "vllm_adapter_latest_ppo"
        if hasattr(inference_backend, "sync_base_adapter"):
            inference_backend.sync_base_adapter(base_model_adapter_dir)
    else:
        inference_backend = None
        vllm_adapter_dir = None

    # ---- wandb (rank 0 only) ----
    if wandb and is_main_rank():
        if not os.getenv("WANDB_NAME"):
            os.environ["WANDB_NAME"] = f"ppo-opt-{game}-{int(time.time())}"
        wandb.login(key=os.getenv("WANDB_API_KEY", ""), relogin=True)
        wandb.init(entity="forge_scaling_intelligence_lab", project="games")
        wandb.config.update({
            "trainer": "ppo_optimized",
            "game": game,
            "games_per_iter": args.games_per_iter,
            "lora_rank": args.lora_rank,
            "lr": args.lr,
            "ppo_epochs": args.ppo_epochs,
            "clip_eps": args.clip_eps,
            "vf_coef": args.vf_coef,
            "kl_coef": args.kl_coef,
            "temperature": args.temperature,
            "opt_compile": use_compile,
            "world_size": world_size,
        })
        print("[wandb] Initialized")

    # ---- Optimizer (LoRA + value head) ----
    trainable_params = [p for p in ac.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable_params, lr=args.lr)
    rollout_logger = JSONLLogger(rollout_log_path) if is_main_rank() else None

    print(f"[PPO] Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    print(f"[PPO] Starting optimized training loop...")

    global_step = start_iter * 25 if args.resume else 0
    # Role baseline EMA (PPO-specific)
    role_baseline_ema = None
    if args.use_role_baseline:
        role_baseline_ema = {
            0: EMA(args.role_baseline_gamma),
            1: EMA(args.role_baseline_gamma),
        }

    # ======================================================================
    # Main PPO loop (optimized)
    # ======================================================================
    for it in tqdm(range(start_iter, 10_000), desc="PPO iters", initial=start_iter, total=10_000):
        t_iter_start = time.time()

        # ---- 1. Sync policy to vLLM (rank 0 only) ----
        if is_main_rank():
            inference_backend.sync_policy(ac.lm, vllm_adapter_dir)
            if not inference_backend.is_enabled():
                inference_backend = HFLocalBackend(ac.lm, tokenizer, device)
        barrier()

        # ---- 2. Periodic tau2-bench evaluation (rank 0 only) ----
        if args.tau2_eval_every and it % args.tau2_eval_every == 0 and it > 0:
            if is_main_rank():
                ac.eval()
                inference_backend.sync_policy(ac.lm, vllm_adapter_dir)
                if not inference_backend.is_enabled():
                    inference_backend = HFLocalBackend(ac.lm, tokenizer, device)

                print(f"\n[eval {it}] Starting tau2-bench evaluation...")
                t_tau2_0 = time.time()
                tau2_logs, _ = evaluate_tau2_bench(
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
            barrier()

        # ---- 3. Collect self-play rollouts (rank 0) + broadcast ----
        if is_main_rank():
            t_collect0 = time.time()
            batch, env_metrics = collect_games(
                model=ac.lm,
                tokenizer=tokenizer,
                backend=inference_backend,
                num_games=args.games_per_iter,
                temperature=args.temperature,
                max_new_tokens=max_gen_tokens,
                seed=it,
                logger=rollout_logger,
                device=device,
                role_baseline_ema=role_baseline_ema,
                game_spec=game_spec,
                env_kwargs=env_kwargs,
            )
            t_collect1 = time.time()
        else:
            t_collect0 = t_collect1 = time.time()
            batch, env_metrics = [], {}

        # ---- 3b. Filter info-gathering turns (rank 0, before broadcast) ----
        dt_filtered = 0
        if is_main_rank() and args.filter_info_turns and batch:
            samples_before = len(batch)
            batch, _keep_idx, dt_filtered = filter_info_gathering_samples(batch)
            if dt_filtered > 0:
                print(
                    f"[iter {it}] Filtered {dt_filtered} info-gathering turns "
                    f"({samples_before} -> {len(batch)} samples)"
                )

        # Broadcast collected data to all ranks
        if is_main_rank():
            if not batch:
                broadcast_data = [True, None, None, 0, t_collect0, t_collect1]
                print(f"[iter {it}] No samples collected, skipping")
            else:
                random.shuffle(batch)
                broadcast_data = [False, batch, env_metrics, dt_filtered,
                                  t_collect0, t_collect1]
        else:
            broadcast_data = [None] * 6

        broadcast_data = broadcast_objects(broadcast_data)
        skip_flag = broadcast_data[0]

        if skip_flag:
            continue

        _, batch, env_metrics, dt_filtered, t_collect0, t_collect1 = broadcast_data

        # ================================================================
        # OPTIMIZED STEPS 4-7
        # ================================================================

        # ---- 4. Pre-tokenize all samples ----
        t_tok0 = time.time()

        seqs_cpu: List[torch.Tensor] = []
        prompt_lens: List[int] = []
        action_lens: List[int] = []
        returns_cpu_list: List[float] = []
        total_tool_results_truncated = 0
        max_seq_len = Config.MAX_SEQ_LENGTH
        num_skipped_long = 0

        for i, s in enumerate(batch):
            if args.tool_result_max_chars > 0:
                msgs, n_trunc = compress_tool_results(
                    s.prompt_msgs, args.tool_result_max_chars
                )
                total_tool_results_truncated += n_trunc
            else:
                msgs = s.prompt_msgs
            ids, pL, aL = build_prompt_plus_action(tokenizer, msgs, s.action_str)
            if ids.shape[0] > max_seq_len:
                num_skipped_long += 1
                continue
            seqs_cpu.append(ids.cpu())
            prompt_lens.append(pL)
            action_lens.append(aL)
            returns_cpu_list.append(float(s.ret))

        if num_skipped_long > 0:
            print(f"  [OPT] Skipped {num_skipped_long}/{len(batch)} samples exceeding {max_seq_len} tokens")

        N = len(seqs_cpu)
        if N == 0:
            print(f"  [iter {it}] All samples exceeded max_seq_len — skipping training step")
            continue

        returns_cpu = torch.tensor(returns_cpu_list, dtype=torch.float32)
        seq_lens = [s.shape[0] for s in seqs_cpu]

        t_tok1 = time.time()

        # ---- 4b. Create length-sorted batches ----
        unified_batches = args.stats_chunk_size == args.mini_batch_size
        train_batches = make_sorted_batches(seq_lens, args.mini_batch_size)

        if unified_batches:
            stats_batches = train_batches
        else:
            stats_batches = make_sorted_batches(seq_lens, args.stats_chunk_size)

        # ---- 4c. Batch lists ready (padding done on-the-fly per batch) ----
        # Each rank only materializes padded tensors for batches it processes,
        # ensuring multi-GPU memory per rank matches single-GPU memory.
        t_pad = time.time()

        # ---- 5. Compute old_logp and old_v [DISTRIBUTED, ON GPU] ----
        ac.eval()
        # Zeros for all-reduce SUM trick: each rank writes disjoint indices.
        # Tensors live on the training device (GPU) — avoids CPU↔GPU roundtrips
        # during stats, advantage computation, and training mini-batch indexing.
        old_logp_dev = torch.zeros(N, dtype=torch.float32, device=device)
        old_v_dev = torch.zeros(N, dtype=torch.float32, device=device)
        compute_base = args.kl_coef > 0
        base_logp_dev = torch.zeros(N, dtype=torch.float32, device=device) if compute_base else None

        # Shard stats batches across ranks
        all_stats_indices = list(range(len(stats_batches)))
        my_stats_indices, _ = shard_batches(all_stats_indices, rank, world_size)

        with torch.no_grad():
            # -- Pass 1: old_logp + old_v (adapter ON) --
            for bi in my_stats_indices:
                mb_idx = stats_batches[bi]
                mb_ids_cpu, mb_attn_cpu, mb_pl, mb_al = pad_batch(
                    mb_idx, seqs_cpu, prompt_lens, action_lens, tokenizer.pad_token_id
                )
                mb_ids = mb_ids_cpu.to(device, non_blocking=True)
                mb_attn = mb_attn_cpu.to(device, non_blocking=True)

                logits, last_h = ac(input_ids=mb_ids, attention_mask=mb_attn)

                mb_old_logp = logprob_action_tokens(
                    logits, mb_ids, mb_pl, mb_al, normalize_by_len=True,
                )
                mb_old_v = values_from_hidden(last_h, ac.value_head, mb_pl)

                old_logp_dev[mb_idx] = mb_old_logp.detach()
                old_v_dev[mb_idx] = mb_old_v.detach()
                del logits, last_h, mb_old_logp, mb_old_v, mb_ids, mb_attn
                del mb_ids_cpu, mb_attn_cpu

            # All-reduce: each rank wrote to disjoint indices, SUM assembles full vector.
            # Tensors already on GPU — all-reduce in-place, no CPU↔GPU transfer.
            if world_size > 1:
                torch.distributed.all_reduce(old_logp_dev, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(old_v_dev, op=torch.distributed.ReduceOp.SUM)

            # -- Pass 2: base_logp (adapter OFF) — distributed --
            if compute_base:
                ac.lm.disable_adapter_layers()
                try:
                    for bi in my_stats_indices:
                        mb_idx = stats_batches[bi]
                        mb_ids_cpu, mb_attn_cpu, mb_pl, mb_al = pad_batch(
                            mb_idx, seqs_cpu, prompt_lens, action_lens, tokenizer.pad_token_id
                        )
                        mb_ids = mb_ids_cpu.to(device, non_blocking=True)
                        mb_attn = mb_attn_cpu.to(device, non_blocking=True)

                        logits, _ = ac(input_ids=mb_ids, attention_mask=mb_attn)

                        mb_base_logp = logprob_action_tokens(
                            logits, mb_ids, mb_pl, mb_al, normalize_by_len=True,
                        )
                        base_logp_dev[mb_idx] = mb_base_logp.detach()
                        del logits, mb_base_logp, mb_ids, mb_attn, _
                        del mb_ids_cpu, mb_attn_cpu

                    if world_size > 1:
                        torch.distributed.all_reduce(base_logp_dev, op=torch.distributed.ReduceOp.SUM)
                finally:
                    ac.lm.enable_adapter_layers()

        t_stats = time.time()

        # ---- 6. Compute advantages on GPU [DISTRIBUTED: all ranks in parallel] ----
        # After all-reduce, every rank has identical old_v_dev → identical advantages.
        returns_dev = returns_cpu.to(device)
        adv_dev = returns_dev - old_v_dev
        adv_std = adv_dev.std(unbiased=False)
        if adv_std > 1e-8:
            adv_dev = (adv_dev - adv_dev.mean()) / (adv_std + 1e-8)
        del old_v_dev  # No longer needed after advantage computation

        # ---- 7. PPO training [DISTRIBUTED: gradient accumulation + all-reduce] ----
        ac.train()

        policy_loss_acc = 0.0
        value_loss_acc = 0.0
        kl_loss_acc = 0.0
        approx_kl_acc = 0.0
        clip_frac_acc = 0.0
        ratio_mean_acc = 0.0
        local_updates = 0

        for _epoch in range(args.ppo_epochs):
            # Shuffle the ORDER of mini-batches — same seed on all ranks for
            # deterministic sharding
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
                mb_idx_dev = torch.tensor(mb_idx, dtype=torch.long, device=device)
                mb_old_logp = old_logp_dev[mb_idx_dev]
                mb_returns = returns_dev[mb_idx_dev]
                mb_adv = adv_dev[mb_idx_dev]
                mb_base_logp = base_logp_dev[mb_idx_dev] if base_logp_dev is not None else None

                # Forward pass through actor-critic
                logits, last_h = ac(input_ids=mb_ids, attention_mask=mb_attn)

                new_logp = logprob_action_tokens(
                    logits, mb_ids, mb_pl, mb_al, normalize_by_len=True,
                )
                new_v = values_from_hidden(last_h, ac.value_head, mb_pl)

                # PPO clipped surrogate
                ratio = torch.exp(new_logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(
                    ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps
                ) * mb_adv
                policy_loss = -torch.mean(torch.min(surr1, surr2))

                # Value loss
                value_loss = 0.5 * F.mse_loss(new_v, mb_returns)

                # Optional KL penalty against base model
                kl_loss = torch.tensor(0.0, device=device)
                if args.kl_coef > 0 and mb_base_logp is not None:
                    kl_loss = args.kl_coef * torch.mean(new_logp - mb_base_logp)

                # Scale loss for gradient accumulation: use LOCAL batch count.
                # allreduce_coalesced_grads averages across ranks (SUM / W).
                loss = (policy_loss + args.vf_coef * value_loss + kl_loss) / len(my_batch_order)
                loss.backward()

                # Tracking (no grad, unscaled losses)
                with torch.no_grad():
                    approx_kl = torch.mean(mb_old_logp - new_logp)
                    clip_frac = torch.mean(
                        (torch.abs(ratio - 1.0) > args.clip_eps).float()
                    )
                    ratio_mean_val = torch.mean(ratio)

                local_updates += 1
                policy_loss_acc += float(policy_loss.item())
                value_loss_acc += float(value_loss.item())
                kl_loss_acc += float(kl_loss.item())
                approx_kl_acc += float(approx_kl.item())
                clip_frac_acc += float(clip_frac.item())
                ratio_mean_acc += float(ratio_mean_val.item())

                del mb_ids, mb_attn, logits, last_h, new_logp, new_v, ratio, loss
                del mb_old_logp, mb_returns, mb_adv, mb_base_logp, mb_idx_dev
                del mb_ids_cpu, mb_attn_cpu

            # All-reduce accumulated gradients across ranks
            allreduce_coalesced_grads(trainable_params)

            # Single gradient clip + optimizer step
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optim.step()

        # Aggregate metrics across ranks for correct averaging
        if world_size > 1:
            local_metrics = {
                "approx_kl": approx_kl_acc,
                "clip_frac": clip_frac_acc,
                "kl_loss": kl_loss_acc,
                "n_updates": float(local_updates),
                "policy_loss": policy_loss_acc,
                "ratio_mean": ratio_mean_acc,
                "value_loss": value_loss_acc,
            }
            agg = allreduce_scalars(local_metrics)
            total_updates = int(agg["n_updates"])
            policy_loss_acc = agg["policy_loss"]
            value_loss_acc = agg["value_loss"]
            kl_loss_acc = agg["kl_loss"]
            approx_kl_acc = agg["approx_kl"]
            clip_frac_acc = agg["clip_frac"]
            ratio_mean_acc = agg["ratio_mean"]
        else:
            total_updates = local_updates

        global_step += n_total_batches
        updates = total_updates

        t_train1 = time.time()

        # Free GPU stat tensors and CPU sequence cache
        del old_logp_dev, adv_dev, returns_dev, seqs_cpu
        base_logp_dev = None  # Free GPU memory if allocated

        # ---- 8. Logging (rank 0 only) ----
        if is_main_rank():
            avg_return = float(returns_cpu.mean().item())
            abs_return = float(returns_cpu.abs().mean().item())

            pad_eff = compute_padding_efficiency(seq_lens, train_batches)

            logs = {
                "iter": it,
                "global_step": global_step,
                "ppo/policy_loss": policy_loss_acc / max(1, updates),
                "ppo/value_loss": value_loss_acc / max(1, updates),
                "ppo/kl_loss": kl_loss_acc / max(1, updates),
                "ppo/avg_return": avg_return,
                "ppo/avg_abs_return": abs_return,
                "ppo/approx_kl": approx_kl_acc / max(1, updates),
                "ppo/clip_frac": clip_frac_acc / max(1, updates),
                "ppo/ratio_mean": ratio_mean_acc / max(1, updates),
                "ppo/num_samples": N,
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

            print(
                f"[iter {it}] step={global_step} "
                f"avg_return={avg_return:.3f} abs_return={abs_return:.3f} "
                f"win_p0={env_metrics.get('env/win_rate_p0', 0):.3f} "
                f"invalid={env_metrics.get('env/invalid_move_rate', 0):.3f} "
                f"samples={N} info_filt={dt_filtered} "
                f"trunc={total_tool_results_truncated} "
                f"KL={logs['ppo/approx_kl']:.4f} "
                f"clip={logs['ppo/clip_frac']:.3f} "
                f"collect={logs['time/collect_sec']:.1f}s train={logs['time/train_sec']:.1f}s "
                f"(tok={t_tok1 - t_tok0:.1f}s pad={t_pad - t_tok1:.1f}s "
                f"stats={t_stats - t_pad:.1f}s grad={t_train1 - t_stats:.1f}s "
                f"pad_eff={pad_eff:.1%})"
            )

            if wandb:
                wandb.log(logs, step=global_step)

        # ---- 9. Save checkpoint: 1st iter, 2nd iter, then every save_every (rank 0 only) ----
        if it <= 1 or it % args.save_every == 0:
            if is_main_rank():
                ckpt_dir = output_dir_path / f"ppo_ckpt_iter_{it}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                ac.lm.save_pretrained(str(ckpt_dir / "policy"))
                tokenizer.save_pretrained(str(ckpt_dir / "policy"))
                torch.save(ac.value_head.state_dict(), str(ckpt_dir / "value_head.pt"))
                print(f"[checkpoint] Saved to {ckpt_dir}")
            barrier()

    # ---- Final save (rank 0 only) ----
    if is_main_rank():
        final_dir = output_dir_path / f"ppo_{game}_final"
        final_dir.mkdir(parents=True, exist_ok=True)
        ac.lm.save_pretrained(str(final_dir / "policy"))
        tokenizer.save_pretrained(str(final_dir / "policy"))
        torch.save(ac.value_head.state_dict(), str(final_dir / "value_head.pt"))
        print(f"[PPO] Training complete! Saved to {final_dir}")

    dist_cleanup()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        dist_cleanup()
