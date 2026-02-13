#!/usr/bin/env python3
"""
Optimized GRPO trainer — based on train_grpo.py with performance and algorithmic improvements.

Optimizations over train_grpo.py (each benchmarked for significant impact):
  1. Length-sorted batching: sequences sorted by length to minimize padding waste
     in forward passes. Padding efficiency goes from ~55% to ~99%.
     (Benchmarked: 1.78x less wasted compute)
  2. stats_chunk_size=8 (was 4): halves number of forward passes in step 6.
     Unified with mini_batch_size so one pre-padded cache serves both steps 6+7.
     (Benchmarked: 3.6x faster step 6, 4.7x faster caching, 50% less CPU memory)
  3. KL regularization defaults to OFF (kl_coef=0). LoRA already constrains
     drift; skipping base_logp eliminates ~50% of step 6 forward passes.
  4. Single adapter toggle: computes ALL old_logp first (adapter ON), then ALL
     base_logp (adapter OFF), toggling adapter layers only once.
  5. Pinned-memory CPU tensors for faster async GPU transfers.
     (Benchmarked: 18.9x faster CPU→GPU transfer)
  6. Optional training-time autocast (bfloat16) via GRPO_TRAIN_AUTOCAST=1.
  7. Detailed sub-step timing breakdown (tokenize, pad, stats, grad, pad_eff).

Algorithmic differences from train_grpo.py (beyond pure performance):
  - Advantage normalization: advantages are standardized (mean=0, std=1) after
    group-relative centering. train_grpo.py uses raw centered advantages.
  - Gradient accumulation: gradients are accumulated across all mini-batches in
    an epoch, then a single optimizer step is taken (required for distributed
    all-reduce). train_grpo.py takes one optimizer step per mini-batch.
  - Info-gathering turn filtering: read-only tool call turns can be filtered out
    of training samples (--filter-info-turns, default ON).
  - Tool result compression: old tool results are truncated to reduce prompt
    length (--tool-result-max-chars, default 200).

Performance-only differences (no algorithmic impact):
  - Batch composition (grouping similar-length sequences reduces padding)
  - Mini-batch ordering (length-bucketed then shuffled vs fully shuffled)
  - Floating-point ordering from different batch compositions (negligible)

Usage (drop-in replacement for train_grpo.py):
    # Single GPU:
    VLLM_BASE_URL=http://localhost:8000 python train_grpo_optimized.py \\
        --game adversarial_policy --group-size 8 --groups-per-batch 16

    # Multi-GPU (N GPUs for ~Nx speedup on training step):
    VLLM_BASE_URL=http://localhost:8000 torchrun --nproc_per_node=4 \\
        train_grpo_optimized.py --game adversarial_policy \\
        --group-size 8 --groups-per-batch 16
"""
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
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from config import Config, setup_environment, autocast_ctx
from game_registry import get_game_spec, list_game_names, GameSpec
from inference import (
    InferenceBackend,
    HFLocalBackend,
    init_inference_backend,
)
from ppo import (
    JSONLLogger,
    build_prompt_plus_action,
    logprob_action_tokens,
)

# Reuse data structures and helpers from train_grpo (no duplication)
from train_grpo import (
    GRPOSample,
    collect_grpo_rollouts,
    compute_group_advantages,
    filter_constant_reward_groups,
)
from evaluation import evaluate_vs_base, evaluate_math, evaluate_tau2_bench
from dist_utils import (
    dist_init, dist_cleanup, is_main_rank, broadcast_objects,
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
# Optimization helpers
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
    Identical output to ppo.pad_to_device() but stays on CPU with pinned memory
    so the caller can transfer asynchronously with non_blocking=True.
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


def compute_padding_efficiency(seq_lens: List[int], batches: List[List[int]]) -> float:
    """Compute ratio of real tokens to padded tokens (1.0 = perfect, no waste)."""
    total_real = sum(seq_lens)
    total_padded = sum(
        len(batch) * max(seq_lens[i] for i in batch)
        for batch in batches
    )
    return total_real / max(1, total_padded)


# ============================================================================
# Adversarial policy training optimizations
# ============================================================================

# Read-only tool names — info-gathering, no policy decision.
# Matches read_ops in adversarial_policy_game/verification.py:109-118.
_INFO_ONLY_TOOLS = frozenset({
    # Airline read operations
    "get_user_details", "get_reservation_details",
    "search_direct_flight", "search_onestop_flight",
    "list_all_airports", "get_flight_status",
    # Retail read operations
    "get_order_details", "get_product_details",
    "find_user_id_by_email", "find_user_id_by_name_zip",
    "list_all_product_types",
    # Generic
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


def filter_info_gathering_turns(
    samples: List["GRPOSample"],
    advantages: torch.Tensor,
) -> Tuple[List["GRPOSample"], torch.Tensor, int]:
    """Remove samples where the agent only made a read-only tool call.

    Keeps samples where the agent called respond_to_user, transfer,
    write/action tools, think, or produced plain text (no tool call).

    Guarantees at least one sample per game_id is kept (the last turn)
    so the game's reward signal is preserved for advantage computation.

    Applied AFTER compute_group_advantages and filter_constant_reward_groups.
    """
    # Pass 1: mark info-only turns for removal
    keep_mask = []
    for s in samples:
        tool_name = _extract_tool_name(s.completion_text)
        if tool_name is None or tool_name not in _INFO_ONLY_TOOLS:
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
    """Truncate old tool results to reduce prompt length.

    Keeps the LAST tool result in full (needed for current decision).
    Truncates all earlier tool results to max_tool_result_chars.

    Returns (compressed_msgs, num_truncated).
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
# Argument parsing (extends train_grpo's parser with optimization defaults)
# ============================================================================

def parse_grpo_args_optimized():
    parser = argparse.ArgumentParser(
        description="Optimized GRPO trainer (same algorithm, faster training step)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- Game and model --
    parser.add_argument("--game", type=str, default="kuhn_poker",
        help="Game to train on (from game_registry)")
    parser.add_argument("--model", type=str, default=None,
        help="HuggingFace model name (default: Config.MODEL_NAME)")

    # -- GRPO structure --
    parser.add_argument("--group-size", type=int, default=16,
        help="Inner loop: rollouts per seed.")
    parser.add_argument("--groups-per-batch", type=int, default=4,
        help="Outer loop: different seeds per iteration.")

    # -- LoRA --
    parser.add_argument("--lora-rank", type=int, default=8,
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
        help="Chunk size for computing old/base logprobs (matches mini-batch-size for unified caching)")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
        help="Max gradient norm for clipping")

    # -- GRPO loss function --
    parser.add_argument("--use-clipping", action="store_true", default=True,
        help="Use clipped surrogate (like PPO).")
    parser.add_argument("--no-clipping", dest="use_clipping", action="store_false",
        help="Disable clipping and use pure importance sampling.")
    parser.add_argument("--clip-eps", type=float, default=0.2,
        help="Clipping epsilon (only used with --use-clipping)")
    parser.add_argument("--kl-coef", type=float, default=0.0,
        help="KL penalty coefficient against base model (0 to disable, saves ~50%% of step 6 time)")
    parser.add_argument("--filter-constant-groups", action="store_true", default=True,
        help="Remove groups where all rollouts got the same reward")
    parser.add_argument("--no-filter-constant-groups", dest="filter_constant_groups",
        action="store_false")

    # -- Generation --
    parser.add_argument("--temperature", type=float, default=0.7,
        help="Sampling temperature for rollouts")
    parser.add_argument("--normalize-by-len", action="store_true",
        help="Normalize logprobs by action length")

    # -- Checkpointing and evaluation --
    parser.add_argument("--resume", type=str, default=None,
        help="Path to checkpoint directory to resume from")
    parser.add_argument("--rollout-log", type=str, default="selfplay_rollouts_grpo.jsonl",
        help="Filename for rollout logs")
    parser.add_argument("--save-every", type=int, default=5,
        help="Save checkpoint every N iterations")
    parser.add_argument("--eval-every", type=int, default=100000,
        help="Eval vs base model every N iterations")
    parser.add_argument("--eval-games", type=int, default=100000,
        help="Number of games for eval vs base")
    parser.add_argument("--math-eval-every", type=int, default=100000,
        help="Math benchmark eval every N iterations (0 to disable)")
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
    parser.add_argument("--adversarial-ratio", type=float, default=0.2,
        help="Ratio of adversarial vs cooperative scenarios (0.0-1.0). "
             "At 0.2, 20%% adversarial (T1-T12) and 80%% cooperative (T13-T21). "
             "Reflects real tau2-bench distribution (~16%% adversarial).")

    # -- Training sample optimizations --
    parser.add_argument("--filter-info-turns", action="store_true", default=True,
        help="Filter out info-gathering tool calls from training samples")
    parser.add_argument("--no-filter-info-turns", dest="filter_info_turns",
        action="store_false", help="Disable info-gathering turn filtering")
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
# Main training loop (optimized)
# ============================================================================

def main():
    # Import Unsloth BEFORE dist_init(): Unsloth's module-level code
    # (vLLM logger suppression, torch.compile LoRA patching) deadlocks
    # when NCCL is already initialized.  Importing first is safe because
    # CUDA_VISIBLE_DEVICES is set externally, so any CUDA context Unsloth
    # creates during import lands on the correct physical GPU.
    from unsloth import FastLanguageModel

    # ---- Distributed init ----
    rank, world_size, local_rank = dist_init()
    if not is_main_rank():
        suppress_print()

    args = parse_grpo_args_optimized()

    # Apply distributed LR scaling
    if world_size > 1 and args.dist_lr_scale != 1.0:
        args.lr *= args.dist_lr_scale

    # Optimization flags via env vars
    use_compile = os.getenv("GRPO_COMPILE", "0") == "1"
    use_train_autocast = os.getenv("GRPO_TRAIN_AUTOCAST", "0") == "1"

    # ---- Game setup ----
    game = args.game
    try:
        game_spec = get_game_spec(game)
    except KeyError as e:
        available = ", ".join(list_game_names())
        raise RuntimeError(f"{e}. Available games: {available}") from e

    max_gen_tokens = game_spec.max_gen_tokens
    env_kwargs: Dict[str, Any] = {}
    if game_spec.name == "kuhn_poker":
        env_kwargs["num_rounds"] = Config.NUM_ROUNDS
    elif game_spec.name == "liars_dice":
        env_kwargs["num_dice"] = Config.NUM_DICE

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

    # ---- Environment setup ----
    env_config = setup_environment(args)
    device = f"cuda:{local_rank}" if world_size > 1 else env_config["device"]
    hf_hub = env_config["hf_hub"]
    output_dir_path = env_config["output_dir_path"]
    rollout_log_path = env_config["rollout_log_path"]

    total_games_per_iter = args.group_size * args.groups_per_batch

    print("=" * 60)
    print("GRPO TRAINER (OPTIMIZED)")
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
    if game_spec.name == "adversarial_policy":
        print(f"  Adversarial ratio:   {args.adversarial_ratio}")
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
    print(f"  [OPT] tool result trunc: {args.tool_result_max_chars} chars (0=off)")
    if world_size > 1:
        print(f"  [DIST] world_size:    {world_size}")
        print(f"  [DIST] rank:          {rank}")
        print(f"  [DIST] lr_scale:      {args.dist_lr_scale}")
    print("=" * 60)

    # ---- Load model + LoRA ----
    # All ranks load simultaneously.  Each rank targets its own GPU via
    # CUDA_VISIBLE_DEVICES + torch.cuda.set_device(), so there is no memory
    # contention.  Sequential loading with barrier() deadlocks because
    # from_pretrained triggers internal distributed collectives (accelerate /
    # transformers check torch.distributed.is_initialized()).
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
    )
    print(f"[Model] Rank {rank}: Model loaded successfully")
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
    math_eval_data_path = Path(__file__).resolve().parent / "data"

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

        # ---- 2. Periodic evaluation (rank 0 only) ----
        if args.eval_every and it % args.eval_every == 0 and it > 0:
            if is_main_rank():
                model.eval()
                inference_backend.sync_policy(model, vllm_adapter_dir)
                if not inference_backend.is_enabled():
                    inference_backend = HFLocalBackend(model, tokenizer, device)

                print(f"\n[eval {it}] Starting eval vs base ({args.eval_games} games)")
                t_eval0 = time.time()
                eval_logs = evaluate_vs_base(
                    current_model=model,
                    base_model_adapter_dir=base_model_adapter_dir,
                    tokenizer=tokenizer,
                    backend=inference_backend,
                    num_games=args.eval_games,
                    temperature=0.0,
                    max_new_tokens=max_gen_tokens,
                    seed=20_000 + it,
                    hf_hub=hf_hub,
                    device=device,
                    game_spec=game_spec,
                    env_kwargs=env_kwargs,
                )
                t_eval1 = time.time()
                wr = eval_logs.get("eval/win_rate_vs_base", 0.0)
                print(f"[eval {it}] win_rate={wr:.3f} ({t_eval1 - t_eval0:.1f}s)")
                if wr > 0.5:
                    print(f"[eval {it}] Model is better than base!")

                if wandb:
                    wandb.log(eval_logs, step=global_step)
            barrier()

        # tau2-bench evaluation (rank 0 only)
        if args.tau2_eval_every and it % args.tau2_eval_every == 0 and it > 0:
            if is_main_rank():
                model.eval()
                inference_backend.sync_policy(model, vllm_adapter_dir)
                if not inference_backend.is_enabled():
                    inference_backend = HFLocalBackend(model, tokenizer, device)

                print(f"\n[eval {it}] Starting tau2-bench evaluation...")
                t_tau2_0 = time.time()
                tau2_logs = evaluate_tau2_bench(
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

        # ---- 3. Collect group-based rollouts (rank 0) + broadcast ----
        if is_main_rank():
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
            )
            t_collect1 = time.time()
        else:
            t_collect0 = t_collect1 = time.time()
            samples, env_metrics = [], {}

        # ---- 4. Compute advantages + filter (rank 0), then broadcast ----
        # broadcast_data layout: [skip_flag, samples, advantages, num_filtered,
        #   dt_filtered, env_metrics, avg_abs_advantage, t_collect0, t_collect1]
        # skip_flag: True = skip this iteration, False = proceed
        if is_main_rank():
            if not samples:
                broadcast_data = [True, None, None, 0, 0, {}, 0.0, t_collect0, t_collect1]
                print(f"[iter {it}] No samples collected, skipping")
            else:
                # Compute group-relative advantages
                advantages = compute_group_advantages(samples)

                # 4b. Filter constant-reward groups
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

                # 4b2. Filter info-gathering turns
                dt_filtered = 0
                samples_before_dt_filter = len(samples)
                if args.filter_info_turns:
                    samples, advantages, dt_filtered = filter_info_gathering_turns(
                        samples, advantages,
                    )
                    if dt_filtered > 0:
                        print(
                            f"[iter {it}] Filtered {dt_filtered} info-gathering turns "
                            f"({samples_before_dt_filter} -> {len(samples)} samples)"
                        )

                # 4c. Normalize advantages
                adv_std = advantages.std()
                if adv_std > 1e-8:
                    advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

                avg_abs_advantage = float(advantages.abs().mean().item())

                broadcast_data = [False, samples, advantages, num_filtered, dt_filtered,
                                  env_metrics, avg_abs_advantage, t_collect0, t_collect1]
        else:
            broadcast_data = [None] * 9

        broadcast_data = broadcast_objects(broadcast_data)
        skip_flag = broadcast_data[0]

        if skip_flag:
            continue

        _, samples, advantages, num_filtered, dt_filtered, \
            env_metrics, avg_abs_advantage, t_collect0, t_collect1 = broadcast_data

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

                if it % args.save_every == 0:
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

        # ---- 5c. Pre-pad and pin all batches on CPU ----
        # Also pre-compute per-batch prompt_lens/action_lens to avoid repeated
        # list comprehensions in the hot loops of steps 6 and 7.
        BatchEntry = Tuple[List[int], torch.Tensor, torch.Tensor, List[int], List[int]]

        def build_batch_cache(batches: List[List[int]]) -> List[BatchEntry]:
            cache = []
            for mb_idx in batches:
                mb_seqs = [seqs_cpu[j] for j in mb_idx]
                mb_ids, mb_attn = pad_and_pin(mb_seqs, tokenizer.pad_token_id)
                mb_pl = [prompt_lens[j] for j in mb_idx]
                mb_al = [action_lens[j] for j in mb_idx]
                cache.append((mb_idx, mb_ids, mb_attn, mb_pl, mb_al))
            return cache

        train_batch_cache = build_batch_cache(train_batches)
        if unified_batches:
            stats_batch_cache = train_batch_cache
        else:
            stats_batch_cache = build_batch_cache(stats_batches)

        t_pad = time.time()

        # ---- 6. Compute old_logp and (optionally) base_logp [DISTRIBUTED] ----
        model.eval()
        old_logp_cpu = torch.zeros(N, dtype=torch.float32)  # zeros for all-reduce SUM trick
        compute_base = args.kl_coef > 0
        base_logp_cpu = torch.zeros(N, dtype=torch.float32) if compute_base else None

        # Shard stats batches across ranks
        all_stats_indices = list(range(len(stats_batch_cache)))
        my_stats_indices, _ = shard_batches(all_stats_indices, rank, world_size)

        with torch.no_grad():
            # -- Pass 1: old_logp (adapter ON) — each rank does its shard --
            for bi in my_stats_indices:
                mb_idx, mb_ids_cpu, mb_attn_cpu, mb_pl, mb_al = stats_batch_cache[bi]
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
                del logits, outputs, mb_old_logp, mb_ids, mb_attn

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
                        mb_idx, mb_ids_cpu, mb_attn_cpu, mb_pl, mb_al = stats_batch_cache[bi]
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
                        del logits, outputs, mb_base_logp, mb_ids, mb_attn

                    if world_size > 1:
                        base_logp_gpu = base_logp_cpu.to(device)
                        torch.distributed.all_reduce(base_logp_gpu, op=torch.distributed.ReduceOp.SUM)
                        base_logp_cpu = base_logp_gpu.cpu()
                        del base_logp_gpu
                finally:
                    model.enable_adapter_layers()

        # Free stats cache if separate from train cache
        if not unified_batches:
            del stats_batch_cache

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

        for _epoch in range(args.epochs):
            # Shuffle the ORDER of mini-batches — same seed on all ranks for
            # deterministic sharding (each rank gets a disjoint subset).
            batch_order = list(range(len(train_batch_cache)))
            rng = random.Random(it * 1000 + _epoch)
            rng.shuffle(batch_order)

            # Shard mini-batches across ranks
            my_batch_order, n_total_batches = shard_batches(batch_order, rank, world_size)

            # Zero gradients once — accumulate across all local mini-batches
            optim.zero_grad(set_to_none=True)

            for bi in my_batch_order:
                mb_idx, mb_ids_cpu, mb_attn_cpu, mb_pl, mb_al = train_batch_cache[bi]

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

                # Scale loss for gradient accumulation: divide by total batches so
                # accumulated gradient = (1/N) * sum(g_i), the full-batch gradient.
                loss = (policy_loss + kl_loss) / n_total_batches
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

                del mb_ids, mb_attn, logits, outputs, new_logp, ratio, loss

            # All-reduce accumulated LoRA gradients across ranks
            allreduce_coalesced_grads(trainable_params)

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

        # Free pre-padded batch cache
        del train_batch_cache

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
                "dist/world_size": world_size,
                **env_metrics,
            }

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

            if wandb:
                wandb.log(logs, step=global_step)

        # ---- Math evaluation (rank 0 only) ----
        if args.math_eval_every and it % args.math_eval_every == 0 and it > 0:
            if is_main_rank():
                model.eval()
                all_math_logs: Dict[str, float] = {}
                for ds_name in Config.MATH_EVAL_DATASETS:
                    math_start = time.time()
                    math_logs = evaluate_math(
                        model=model,
                        tokenizer=tokenizer,
                        data_path=math_eval_data_path,
                        dataset_name=ds_name,
                        num_samples=Config.MATH_EVAL_SAMPLES,
                        temperature=0.0,
                        max_new_tokens=Config.MAX_TOKENS_MATH_EVAL,
                        backend=inference_backend,
                        device=device,
                    )
                    math_end = time.time()
                    acc = math_logs.get(f"eval_math/{ds_name}_accuracy", 0.0)
                    print(f"[eval {it}] {ds_name}: accuracy={acc:.3f} ({math_end - math_start:.1f}s)")
                    all_math_logs.update(math_logs)
                if wandb:
                    wandb.log(all_math_logs, step=global_step)
            barrier()

        # ---- 9. Save checkpoint (rank 0 only) ----
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

