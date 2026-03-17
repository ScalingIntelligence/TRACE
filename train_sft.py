#!/usr/bin/env python3
"""
Standalone SFT trainer for tau2-bench trajectory data.

Trains a LoRA adapter on successful (reward=1.0) assistant turns extracted
from tau2-bench simulation JSONs.  Uses the same exact loss computation as
the SFT term in train_grpo_optimized.py:

    loss = -mean(logprob_action_tokens(logits, ids, prompt_lens, action_lens,
                                        normalize_by_len=True))

Supports:
  - torchrun multi-GPU (gradient all-reduce, no FSDP)
  - LoRA via unsloth FastLanguageModel
  - SFTBuffer for data loading and tokenization
  - Exact per-token loss computation via logprob_action_tokens
  - Gradient accumulation across mini-batches
  - Sequence packing for reduced padding waste
  - torch.compile for kernel fusion speedup
  - Checkpoint saving and resuming
  - wandb logging

Usage:
    # Single GPU:
    python train_sft.py --sft-data file1.json,file2.json

    # Multi-GPU:
    torchrun --nproc_per_node=4 train_sft.py --sft-data file1.json,file2.json

    # With packing + compile:
    python train_sft.py --sft-data file1.json --pack-sequences --compile
"""
import argparse
import json
import math
import os
import random
import time


os.environ.setdefault("NCCL_P2P_DISABLE", "1")

from loguru import logger as _loguru_logger
_loguru_logger.remove()
_loguru_logger.disable("tau2")

import torch
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

from config import Config, setup_environment, autocast_ctx
from ppo import logprob_action_tokens, pad_to_device, build_prompt_plus_action
from sft_buffer import SFTBuffer
from gumbel_topk import gumbel_kl_loss
from dist_utils import (
    dist_pre_init, dist_nccl_init,
    dist_cleanup, is_main_rank, barrier,
    shard_batches, allreduce_coalesced_grads, allreduce_scalars,
    suppress_print,
)

try:
    import wandb
except Exception:
    wandb = None


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="SFT trainer for tau2-bench trajectory data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- Data --
    parser.add_argument("--sft-data", type=str, required=True,
        help="Comma-separated paths to tau2-bench simulation JSON files")

    # -- Model --
    parser.add_argument("--model", type=str, default=None,
        help="HuggingFace model name (default: Config.MODEL_NAME)")

    # -- LoRA --
    parser.add_argument("--lora-rank", type=int, default=16,
        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16,
        help="LoRA alpha scaling factor")

    # -- Optimization --
    parser.add_argument("--lr", type=float, default=1e-5,
        help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=3,
        help="Number of full passes over the SFT dataset")
    parser.add_argument("--mini-batch-size", type=int, default=4,
        help="Mini-batch size for gradient updates")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
        help="Number of mini-batches to accumulate before each optimizer step")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
        help="Max gradient norm for clipping")
    parser.add_argument("--warmup-ratio", type=float, default=0.05,
        help="Fraction of total steps for linear warmup")
    parser.add_argument("--weight-decay", type=float, default=0.01,
        help="Weight decay for AdamW")

    # -- Performance --
    parser.add_argument("--compile", action="store_true", default=False,
        help="Use torch.compile for kernel fusion (may conflict with some unsloth versions)")
    parser.add_argument("--pack-sequences", action="store_true", default=False,
        help="Pack multiple sequences per row to reduce padding waste. "
             "Requires flash attention 2 + transformers >= 4.44 for correct "
             "cross-sequence attention masking via position_ids.")

    # -- Data processing --
    parser.add_argument("--compact-tools", action="store_true", default=False,
        help="Use compressed tool schemas (strips descriptions)")
    parser.add_argument("--max-seq-length", type=int, default=None,
        help="Max sequence length (default: Config.MAX_SEQ_LENGTH)")
    parser.add_argument("--max-samples", type=int, default=None,
        help="Train on only the first N samples (after seq-length filtering). "
             "Useful for quick debugging or ablation runs.")
    parser.add_argument("--max-samples-per-file", type=str, default=None,
        help="Limit each input file to at most N samples before merging. "
             "A single int applies to all files; a comma-separated list "
             "sets per-file limits matching --sft-data order. "
             "E.g., --max-samples-per-file 100 or --max-samples-per-file 50,200,100")
    parser.add_argument("--shuffle-seed", type=int, default=42,
        help="Random seed for data shuffling")

    # -- Checkpointing --
    parser.add_argument("--save-every", type=int, default=1,
        help="Save checkpoint every N epochs")
    parser.add_argument("--resume", type=str, default=None,
        help="Path to checkpoint directory to resume from")
    parser.add_argument("--output-dir", type=str, default=None,
        help="Output directory for checkpoints (default: auto)")

    # -- Gumbel distillation regularization --
    parser.add_argument("--kl-coef", type=float, default=0.0,
        help="Gumbel distillation loss weight (0 = disabled). "
             "Requires --gumbel-data from precompute_gumbel.py")
    parser.add_argument("--gumbel-data", type=str, default=None,
        help="Path to precomputed Gumbel top-k .pt file from precompute_gumbel.py")
    parser.add_argument("--gumbel-k", type=int, default=None,
        help="Override k for Gumbel soft targets (default: use all k from file)")

    # -- Logging --
    parser.add_argument("--log-every", type=int, default=10,
        help="Log metrics every N optimizer steps")
    parser.add_argument("--wandb-project", type=str, default="games",
        help="wandb project name")

    # -- Root directory --
    parser.add_argument("--root", type=str, default=None,
        help="Root directory for cache and outputs")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data preparation (unpacked mode)
# ---------------------------------------------------------------------------

def prepare_batches(
    seqs: List[torch.Tensor],
    mini_batch_size: int,
) -> List[List[int]]:
    """Sort sequences by length and create mini-batches.

    Length-sorted batching minimizes padding waste in forward passes.
    Returns list of index lists, each of length <= mini_batch_size.
    """
    indices = sorted(range(len(seqs)), key=lambda i: seqs[i].shape[0])
    batches = []
    for start in range(0, len(indices), mini_batch_size):
        batches.append(indices[start:start + mini_batch_size])
    return batches


def pad_batch(
    batch_indices: List[int],
    seqs: List[torch.Tensor],
    prompt_lens: List[int],
    action_lens: List[int],
    pad_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
    """Pad a mini-batch of sequences and return (ids, attn, pls, als)."""
    batch_seqs = [seqs[i] for i in batch_indices]
    pls = [prompt_lens[i] for i in batch_indices]
    als = [action_lens[i] for i in batch_indices]
    ids, attn = pad_to_device(batch_seqs, pad_id, "cpu")
    return ids, attn, pls, als


# ---------------------------------------------------------------------------
# Sequence packing
# ---------------------------------------------------------------------------

def pack_into_rows(
    seqs: List[torch.Tensor],
    max_pack_len: int,
) -> List[List[int]]:
    """Greedily pack sequences into rows of at most max_pack_len tokens.

    Sorts sequences shortest-first then fills rows sequentially.
    Returns list of lists of indices into `seqs`.
    """
    indices = sorted(range(len(seqs)), key=lambda i: seqs[i].shape[0])

    rows: List[List[int]] = []
    current_row: List[int] = []
    current_len = 0

    for idx in indices:
        seq_len = seqs[idx].shape[0]
        if seq_len > max_pack_len:
            continue
        if current_len + seq_len > max_pack_len and current_row:
            rows.append(current_row)
            current_row = []
            current_len = 0
        current_row.append(idx)
        current_len += seq_len

    if current_row:
        rows.append(current_row)

    return rows


def build_packed_batch(
    row_indices_list: List[List[int]],
    seqs: List[torch.Tensor],
    prompt_lens: List[int],
    action_lens: List[int],
    pad_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[Tuple[int, int, int]]]]:
    """Build padded tensors for a batch of packed rows.

    Returns:
        ids:         [num_rows, max_row_len]
        position_ids:[num_rows, max_row_len]
        attn_mask:   [num_rows, max_row_len]
        pack_info:   list of lists of (offset, prompt_len, action_len) per row
    """
    batch_ids: List[torch.Tensor] = []
    batch_pos: List[torch.Tensor] = []
    pack_info: List[List[Tuple[int, int, int]]] = []

    for row_indices in row_indices_list:
        tokens_list: List[torch.Tensor] = []
        pos_list: List[torch.Tensor] = []
        segments: List[Tuple[int, int, int]] = []
        offset = 0

        for idx in row_indices:
            seq = seqs[idx]
            seq_len = seq.shape[0]
            tokens_list.append(seq)
            pos_list.append(torch.arange(seq_len, dtype=torch.long))
            segments.append((offset, prompt_lens[idx], action_lens[idx]))
            offset += seq_len

        batch_ids.append(torch.cat(tokens_list))
        batch_pos.append(torch.cat(pos_list))
        pack_info.append(segments)

    # Pad to max row length in this batch
    max_len = max(t.shape[0] for t in batch_ids)
    B = len(batch_ids)

    ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    pos = torch.zeros((B, max_len), dtype=torch.long)
    attn = torch.zeros((B, max_len), dtype=torch.long)

    for i in range(B):
        L = batch_ids[i].shape[0]
        ids[i, :L] = batch_ids[i]
        pos[i, :L] = batch_pos[i]
        attn[i, :L] = 1

    return ids, pos, attn, pack_info


def logprob_packed(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    pack_info: List[List[Tuple[int, int, int]]],
    normalize_by_len: bool = True,
) -> torch.Tensor:
    """Compute per-sequence normalized log-probs for packed rows.

    Args:
        logits:    [num_rows, max_packed_len, V]
        input_ids: [num_rows, max_packed_len]
        pack_info: list of lists of (offset, prompt_len, action_len)
        normalize_by_len: if True, divide total log-prob by action_len

    Returns:
        Tensor of per-sequence log-probs [N] (one per original sequence)
    """
    results: List[torch.Tensor] = []

    for row_idx, segments in enumerate(pack_info):
        for offset, pl, al in segments:
            if al <= 0:
                continue
            # Logits at position p predict token at position p+1.
            # Action tokens occupy positions [offset+pl, offset+pl+al).
            # So predicting logits are at   [offset+pl-1, offset+pl+al-1).
            l_start = offset + pl - 1
            t_start = offset + pl
            t_end = offset + pl + al

            # Clamp: if pl=0 at offset=0, l_start=-1 — skip the first
            # action token that has no preceding logit to predict it.
            if l_start < 0:
                skip = -l_start
                l_start = 0
                t_start += skip

            l_end = l_start + (t_end - t_start)
            eff_al = t_end - t_start
            if eff_al <= 0:
                continue

            seg_logits = logits[row_idx, l_start:l_end, :].float()   # [eff_al, V]
            seg_targets = input_ids[row_idx, t_start:t_end]           # [eff_al]

            log_p = F.log_softmax(seg_logits, dim=-1)
            tok_lp = log_p.gather(1, seg_targets.unsqueeze(1)).squeeze(1)  # [eff_al]

            total = tok_lp.sum()
            results.append(total / eff_al if normalize_by_len else total)

    if not results:
        return torch.zeros(1, device=logits.device, dtype=torch.float32)

    return torch.stack(results)


# ---------------------------------------------------------------------------
# Gumbel data gathering for batches
# ---------------------------------------------------------------------------

def gather_gumbel_for_unpacked(
    batch_indices: List[int],
    sample_order: List[int],
    gumbel_data: list,
    k: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather and concatenate Gumbel data for action positions in an unpacked batch.

    Returns:
        tokens:     [N_total, k] int64
        log_probs:  [N_total, k] float32
        thresholds: [N_total, k] float32
    where N_total = sum of action position counts across batch items.
    """
    all_tok, all_lp, all_thr = [], [], []
    for idx in batch_indices:
        orig_idx = sample_order[idx]
        gd = gumbel_data[orig_idx]
        n_pos = gd["tokens"].shape[0]
        if n_pos > 0:
            all_tok.append(gd["tokens"][:, :k].to(torch.int64))
            all_lp.append(gd["log_probs"][:, :k].float())
            all_thr.append(gd["thresholds"][:, :k].float())

    if not all_tok:
        empty = torch.zeros(0, k, device=device)
        return empty.long(), empty, empty

    return (
        torch.cat(all_tok).to(device),
        torch.cat(all_lp).to(device),
        torch.cat(all_thr).to(device),
    )


def gather_policy_logits_for_unpacked(
    logits: torch.Tensor,
    prompt_lens: List[int],
    action_lens: List[int],
) -> torch.Tensor:
    """Extract policy logits at action-predicting positions for an unpacked batch.

    Returns:
        policy_logits: [N_total, V] where N_total = sum of action position counts.
    """
    parts = []
    for i in range(logits.size(0)):
        pl, al = prompt_lens[i], action_lens[i]
        if al <= 0:
            continue
        l_start = max(0, pl - 1)
        l_end = pl + al - 1
        if l_end > l_start:
            parts.append(logits[i, l_start:l_end, :])
    if not parts:
        return torch.zeros(0, logits.size(-1), device=logits.device)
    return torch.cat(parts, dim=0)


def gather_gumbel_for_packed(
    pack_info: List[List[Tuple[int, int, int]]],
    row_indices_list: List[List[int]],
    sample_order: List[int],
    gumbel_data: list,
    k: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather Gumbel data for action positions in a packed batch."""
    all_tok, all_lp, all_thr = [], [], []
    for row_indices, segments in zip(row_indices_list, pack_info):
        for seq_local_idx, (_, _, al) in zip(row_indices, segments):
            orig_idx = sample_order[seq_local_idx]
            gd = gumbel_data[orig_idx]
            n_pos = gd["tokens"].shape[0]
            if n_pos > 0 and al > 0:
                all_tok.append(gd["tokens"][:, :k].to(torch.int64))
                all_lp.append(gd["log_probs"][:, :k].float())
                all_thr.append(gd["thresholds"][:, :k].float())

    if not all_tok:
        empty = torch.zeros(0, k, device=device)
        return empty.long(), empty, empty

    return (
        torch.cat(all_tok).to(device),
        torch.cat(all_lp).to(device),
        torch.cat(all_thr).to(device),
    )


def gather_policy_logits_for_packed(
    logits: torch.Tensor,
    pack_info: List[List[Tuple[int, int, int]]],
) -> torch.Tensor:
    """Extract policy logits at action-predicting positions for a packed batch."""
    parts = []
    for row_idx, segments in enumerate(pack_info):
        for offset, pl, al in segments:
            if al <= 0:
                continue
            l_start = max(0, offset + pl - 1)
            l_end = offset + pl + al - 1
            if l_end > l_start:
                parts.append(logits[row_idx, l_start:l_end, :])
    if not parts:
        return torch.zeros(0, logits.size(-1), device=logits.device)
    return torch.cat(parts, dim=0)


# ---------------------------------------------------------------------------
# Learning rate scheduler
# ---------------------------------------------------------------------------

def get_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    """Linear warmup then cosine decay."""
    if step < warmup_steps:
        return base_lr * (step + 1) / (warmup_steps + 1)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def set_lr(optimizer, lr: float):
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---- Distributed init (Phase 1: env vars + CUDA device, NO NCCL) ----
    rank, world_size, local_rank = dist_pre_init()
    if world_size > 1 and rank != 0:
        suppress_print()

    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    # ---- Environment setup ----
    env = setup_environment(args)
    hf_hub = env["hf_hub"]

    max_seq_len = args.max_seq_length or Config.MAX_SEQ_LENGTH

    # ---- Output directory ----
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = env["output_dir_path"] / "sft"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load model + LoRA ----
    from unsloth import FastLanguageModel

    model_name = args.model or Config.MODEL_NAME
    print(f"[Model] Rank {rank}: Loading {model_name} with LoRA rank={args.lora_rank}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_len,
        dtype=None,
        use_exact_model_name=True,
        cache_dir=str(hf_hub),
        device_map={"": local_rank},
    )
    print(f"[Model] Rank {rank}: Model loaded successfully", flush=True)

    # ---- Distributed init (Phase 2: NCCL) ----
    print(f"[NCCL] Rank {rank}: Initializing NCCL process group...", flush=True)
    dist_nccl_init()
    print(f"[NCCL] Rank {rank}: NCCL initialized", flush=True)
    barrier()

    # ---- LoRA ----
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

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = model.to(device)
    print(f"[Model] Loaded with LoRA — SFT mode")

    # ---- torch.compile (optional) ----
    # Keep a reference to the unwrapped model for save_pretrained, since
    # torch.compile wraps it in OptimizedModule which lacks that method.
    unwrapped_model = model
    if args.compile:
        print("[Compile] Compiling model with torch.compile...")
        model = torch.compile(model)
        print("[Compile] Done (first forward pass will trigger compilation)")

    # ---- Resume from checkpoint ----
    start_epoch = 0
    global_step = 0
    if args.resume:
        import safetensors.torch
        from peft import set_peft_model_state_dict

        resume_path = Path(args.resume)
        print(f"[Resume] Loading from {resume_path}")

        adapter_path = resume_path / "adapter_model.safetensors"
        if adapter_path.exists():
            state = safetensors.torch.load_file(str(adapter_path))
            set_peft_model_state_dict(unwrapped_model, state)
            print(f"[Resume] Loaded adapter from {adapter_path}")
        else:
            raise FileNotFoundError(f"Adapter not found at {adapter_path}")

        meta_path = resume_path / "sft_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            start_epoch = meta.get("epoch", 0) + 1
            global_step = meta.get("global_step", 0)
            print(f"[Resume] Starting from epoch {start_epoch}, step {global_step}")

    # ---- Load SFT data ----
    sft_paths = [p.strip() for p in args.sft_data.split(",") if p.strip()]

    # Parse --max-samples-per-file: single int or comma-separated per-file limits
    max_per_file = None
    if args.max_samples_per_file is not None:
        parts = [x.strip() for x in args.max_samples_per_file.split(",")]
        if len(parts) == 1:
            max_per_file = [int(parts[0])] * len(sft_paths)
        else:
            if len(parts) != len(sft_paths):
                raise ValueError(
                    f"--max-samples-per-file has {len(parts)} values but "
                    f"--sft-data has {len(sft_paths)} files"
                )
            max_per_file = [int(x) for x in parts]

    print(f"[SFT] Loading data from {len(sft_paths)} files...")
    sft_buffer = SFTBuffer(sft_paths, tokenizer, compact_tools=args.compact_tools,
                           max_samples_per_file=max_per_file,
                           max_seq_len=max_seq_len)
    total_samples = len(sft_buffer)
    print(f"[SFT] Loaded {total_samples} samples from {len(sft_paths)} files")

    if total_samples == 0:
        print("[SFT] ERROR: No samples loaded. Check that JSON files contain reward=1.0 simulations.")
        return

    # ---- Truncate to first N samples if requested ----
    if args.max_samples is not None and args.max_samples < total_samples:
        sft_buffer._seqs_cpu = sft_buffer._seqs_cpu[:args.max_samples]
        sft_buffer._prompt_lens = sft_buffer._prompt_lens[:args.max_samples]
        sft_buffer._action_lens = sft_buffer._action_lens[:args.max_samples]
        total_samples = len(sft_buffer)
        print(f"[SFT] Truncated to first {args.max_samples} samples (--max-samples)")

    # ---- Load precomputed Gumbel data (required for --kl-coef > 0) ----
    gumbel_data = None
    gumbel_k = None
    if args.kl_coef > 0 and not args.gumbel_data:
        raise ValueError(
            "--kl-coef > 0 requires --gumbel-data. Run precompute_gumbel.py first."
        )
    if args.gumbel_data:
        gumbel_file = torch.load(args.gumbel_data, map_location="cpu", weights_only=False)
        gumbel_data = gumbel_file["data"]
        gumbel_k = args.gumbel_k or gumbel_file["gumbel_k"]
        if len(gumbel_data) != total_samples:
            raise ValueError(
                f"Gumbel data has {len(gumbel_data)} samples but SFT data has "
                f"{total_samples}. They must match (same --sft-data and --max-samples)."
            )
        print(f"[Gumbel] Loaded precomputed data: k={gumbel_k}, "
              f"{len(gumbel_data)} samples from {args.gumbel_data}")
        if args.kl_coef <= 0:
            print("[Gumbel] WARNING: --gumbel-data provided but --kl-coef is 0. "
                  "Set --kl-coef > 0 to enable Gumbel distillation loss.")

    # ---- Optimizer ----
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    print(f"[SFT] Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    # ---- Compute schedule ----
    accum_steps = args.gradient_accumulation_steps
    seqs = sft_buffer._seqs_cpu
    prompt_lens = sft_buffer._prompt_lens
    action_lens = sft_buffer._action_lens

    if args.pack_sequences:
        # Trial pack to estimate row count (packing is re-done each epoch)
        trial_rows = pack_into_rows(seqs, max_seq_len)
        num_packed_rows = len(trial_rows)
        batches_per_epoch_total = math.ceil(num_packed_rows / args.mini_batch_size)
        packing_ratio = total_samples / max(1, num_packed_rows)
        del trial_rows
    else:
        num_packed_rows = None
        packing_ratio = 1.0
        batches_per_epoch_total = math.ceil(total_samples / args.mini_batch_size)

    batches_per_epoch_per_rank = math.ceil(batches_per_epoch_total / world_size)
    steps_per_epoch = math.ceil(batches_per_epoch_per_rank / accum_steps)
    total_steps = steps_per_epoch * args.num_epochs
    warmup_steps = int(args.warmup_ratio * total_steps)

    print(f"\n{'=' * 60}")
    print(f"  SFT Training Configuration")
    print(f"{'=' * 60}")
    print(f"  Model:              {model_name}")
    print(f"  LoRA rank:          {args.lora_rank}")
    print(f"  LoRA alpha:         {args.lora_alpha}")
    print(f"  Learning rate:      {args.lr}")
    print(f"  Weight decay:       {args.weight_decay}")
    print(f"  Epochs:             {args.num_epochs}")
    print(f"  Total samples:      {total_samples}")
    print(f"  Mini-batch size:    {args.mini_batch_size}")
    print(f"  Grad accum steps:   {accum_steps}")
    print(f"  Effective batch:    {args.mini_batch_size * accum_steps * world_size}")
    print(f"  Pack sequences:     {args.pack_sequences}")
    if args.pack_sequences:
        print(f"  Packed rows:        ~{num_packed_rows} ({packing_ratio:.1f}x packing)")
    print(f"  torch.compile:      {args.compile}")
    print(f"  Batches/epoch:      {batches_per_epoch_total}")
    print(f"  Steps/epoch:        {steps_per_epoch}")
    print(f"  Total optim steps:  {total_steps}")
    print(f"  Warmup steps:       {warmup_steps}")
    print(f"  Max seq length:     {max_seq_len}")
    print(f"  Max grad norm:      {args.max_grad_norm}")
    print(f"  Compact tools:      {args.compact_tools}")
    if args.kl_coef > 0:
        print(f"  KL coef:            {args.kl_coef}")
        print(f"  Gumbel k:           {gumbel_k}")
    print(f"  Device:             {device}")
    print(f"  Output dir:         {output_dir}")
    if world_size > 1:
        print(f"  World size:         {world_size}")
        print(f"  Rank:               {rank}")
    print(f"{'=' * 60}\n")

    # ---- wandb (rank 0 only) ----
    if wandb and is_main_rank():
        if not os.getenv("WANDB_NAME"):
            os.environ["WANDB_NAME"] = f"sft-{int(time.time())}"
        wandb.login(key=os.getenv("WANDB_API_KEY", ""), relogin=True)
        wandb.init(entity="forge_scaling_intelligence_lab", project=args.wandb_project)
        wandb.config.update({
            "trainer": "sft",
            "model": model_name,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lr": args.lr,
            "num_epochs": args.num_epochs,
            "total_samples": total_samples,
            "mini_batch_size": args.mini_batch_size,
            "gradient_accumulation_steps": accum_steps,
            "max_seq_length": max_seq_len,
            "compact_tools": args.compact_tools,
            "pack_sequences": args.pack_sequences,
            "torch_compile": args.compile,
            "world_size": world_size,
            "kl_coef": args.kl_coef,
            "gumbel_k": gumbel_k,
        })
        print("[wandb] Initialized")

    # ---- Sequence length stats ----
    seq_lens = [seqs[i].shape[0] for i in range(total_samples)]
    print(f"[SFT] Sequence length stats: "
          f"min={min(seq_lens)}, max={max(seq_lens)}, "
          f"mean={sum(seq_lens)/len(seq_lens):.0f}, "
          f"median={sorted(seq_lens)[len(seq_lens)//2]}")

    # ====================================================================
    # Training loop
    # ====================================================================
    lr = args.lr  # initialize for logging safety (overwritten by schedule)

    for epoch in range(start_epoch, args.num_epochs):
        t_epoch_start = time.time()
        model.train()

        # Shuffle sample order (deterministic across ranks for sharding)
        rng = random.Random(args.shuffle_seed + epoch)
        sample_order = list(range(total_samples))
        rng.shuffle(sample_order)

        # Reorder references via shuffled indices
        epoch_seqs = [seqs[i] for i in sample_order]
        epoch_pls = [prompt_lens[i] for i in sample_order]
        epoch_als = [action_lens[i] for i in sample_order]

        if args.pack_sequences:
            # Pack shuffled sequences into rows
            packed_rows = pack_into_rows(epoch_seqs, max_seq_len)

            # Create batches of packed rows, sorted by row length
            row_lens = [sum(epoch_seqs[i].shape[0] for i in row) for row in packed_rows]
            row_order = sorted(range(len(packed_rows)), key=lambda r: row_lens[r])
            batches = []
            for start in range(0, len(row_order), args.mini_batch_size):
                batches.append([packed_rows[row_order[j]]
                                for j in range(start, min(start + args.mini_batch_size, len(row_order)))])
        else:
            # Standard length-sorted mini-batches
            batches = prepare_batches(epoch_seqs, args.mini_batch_size)

        # Shuffle batch order (same seed on all ranks)
        batch_order = list(range(len(batches)))
        rng.shuffle(batch_order)

        # Shard batches across ranks
        my_batch_order, _ = shard_batches(batch_order, rank, world_size)

        # Pad so all ranks have equal batch count (prevents allreduce deadlock
        # when ranks get different counts from round-robin sharding).
        real_batch_count = len(my_batch_order)
        target_len = math.ceil(len(batch_order) / max(1, world_size))
        while len(my_batch_order) < target_len:
            # Use last real batch index, or 0 if this rank got no batches
            my_batch_order.append(my_batch_order[-1] if my_batch_order else 0)

        # Initialize gradient accumulation.
        # Allocate zero-grad tensors so allreduce_coalesced_grads always
        # finds non-empty grads, even on padding-only accumulation windows
        # where no backward() runs.  set_to_none=False keeps them after step.
        for p in trainable_params:
            if p.grad is None:
                p.grad = torch.zeros_like(p.data)
        optim.zero_grad(set_to_none=False)

        epoch_loss_acc = 0.0
        epoch_kl_acc = 0.0
        epoch_tokens_acc = 0
        epoch_seqs_acc = 0
        local_fwd_count = 0  # forward passes (for averaging loss)

        for step_in_epoch, bi in enumerate(my_batch_order):
            is_padding = step_in_epoch >= real_batch_count

            # LR schedule: update at each optimizer step boundary
            if step_in_epoch % accum_steps == 0:
                lr = get_lr(global_step, total_steps, warmup_steps, args.lr)
                set_lr(optim, lr)

            # Skip forward/backward on padding batches (they exist only to
            # keep allreduce call counts in sync across ranks).
            if is_padding:
                pass  # no forward/backward, just fall through to optimizer step
            elif args.pack_sequences:
                batch_rows = batches[bi]
                mb_ids_cpu, mb_pos_cpu, mb_attn_cpu, pack_info = build_packed_batch(
                    batch_rows, epoch_seqs, epoch_pls, epoch_als,
                    tokenizer.pad_token_id,
                )

                try:
                    mb_ids = mb_ids_cpu.to(device, non_blocking=True)
                    mb_pos = mb_pos_cpu.to(device, non_blocking=True)
                    mb_attn = mb_attn_cpu.to(device, non_blocking=True)

                    with autocast_ctx(device):
                        outputs = model(
                            input_ids=mb_ids, attention_mask=mb_attn,
                            position_ids=mb_pos, use_cache=False,
                        )
                        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                        logp = logprob_packed(logits, mb_ids, pack_info, normalize_by_len=True)

                    sft_loss = -logp.mean()

                    # Gumbel distillation regularization
                    kl_loss = torch.tensor(0.0, device=device)
                    if args.kl_coef > 0:
                        pol_logits_flat = gather_policy_logits_for_packed(logits, pack_info)
                        g_tok, g_lp, g_thr = gather_gumbel_for_packed(
                            pack_info, batch_rows, sample_order,
                            gumbel_data, gumbel_k, device,
                        )
                        kl_loss = gumbel_kl_loss(
                            pol_logits_flat, g_tok, g_lp, g_thr, gumbel_k,
                        )
                        del pol_logits_flat, g_tok, g_lp, g_thr

                    total_loss = sft_loss + args.kl_coef * kl_loss
                    scaled_loss = total_loss / accum_steps
                    scaled_loss.backward()

                    with torch.no_grad():
                        batch_action_tokens = sum(al for segs in pack_info for _, _, al in segs)
                        batch_seq_count = sum(len(segs) for segs in pack_info)
                        epoch_loss_acc += float(sft_loss.item())
                        epoch_kl_acc += float(kl_loss.item())
                        epoch_tokens_acc += batch_action_tokens
                        epoch_seqs_acc += batch_seq_count
                        local_fwd_count += 1

                    del mb_ids, mb_pos, mb_attn, logits, outputs, logp, scaled_loss, sft_loss, kl_loss, total_loss

                except torch.cuda.OutOfMemoryError:
                    for v in ("mb_ids", "mb_pos", "mb_attn", "logits", "outputs",
                              "logp", "scaled_loss", "sft_loss", "kl_loss",
                              "total_loss"):
                        locals().pop(v, None)
                    torch.cuda.empty_cache()
                    if is_main_rank():
                        print(f"[OOM] packed batch {step_in_epoch+1} — retrying rows individually (distill skipped)")

                    # Retry: process each packed row one at a time
                    n_rows = len(batch_rows)
                    batch_loss_acc = 0.0
                    batch_tokens_acc = 0
                    batch_seq_count = 0
                    for ri in range(n_rows):
                        s_ids, s_pos, s_attn, s_info = build_packed_batch(
                            [batch_rows[ri]], epoch_seqs, epoch_pls, epoch_als,
                            tokenizer.pad_token_id,
                        )
                        s_ids = s_ids.to(device, non_blocking=True)
                        s_pos = s_pos.to(device, non_blocking=True)
                        s_attn = s_attn.to(device, non_blocking=True)
                        with autocast_ctx(device):
                            out = model(input_ids=s_ids, attention_mask=s_attn,
                                        position_ids=s_pos, use_cache=False)
                            lg = out.logits if hasattr(out, "logits") else out[0]
                            lp = logprob_packed(lg, s_ids, s_info, normalize_by_len=True)
                        loss_i = -lp.mean() / (accum_steps * n_rows)
                        loss_i.backward()
                        with torch.no_grad():
                            batch_loss_acc += float((-lp.mean()).item())
                            batch_tokens_acc += sum(al for segs in s_info for _, _, al in segs)
                            batch_seq_count += sum(len(segs) for segs in s_info)
                        del s_ids, s_pos, s_attn, out, lg, lp, loss_i
                        torch.cuda.empty_cache()
                    with torch.no_grad():
                        epoch_loss_acc += batch_loss_acc / n_rows
                        epoch_tokens_acc += batch_tokens_acc
                        epoch_seqs_acc += batch_seq_count
                    local_fwd_count += 1

                del mb_ids_cpu, mb_pos_cpu, mb_attn_cpu

            else:
                # ---- Standard (unpacked) forward pass ----
                mb_idx = batches[bi]
                mb_ids_cpu, mb_attn_cpu, mb_pl, mb_al = pad_batch(
                    mb_idx, epoch_seqs, epoch_pls, epoch_als, tokenizer.pad_token_id
                )

                try:
                    mb_ids = mb_ids_cpu.to(device, non_blocking=True)
                    mb_attn = mb_attn_cpu.to(device, non_blocking=True)

                    with autocast_ctx(device):
                        outputs = model(
                            input_ids=mb_ids, attention_mask=mb_attn, use_cache=False
                        )
                        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                        logp = logprob_action_tokens(
                            logits, mb_ids, mb_pl, mb_al, normalize_by_len=True
                        )

                    sft_loss = -logp.mean()

                    # Gumbel distillation regularization
                    kl_loss = torch.tensor(0.0, device=device)
                    if args.kl_coef > 0:
                        pol_logits_flat = gather_policy_logits_for_unpacked(
                            logits, mb_pl, mb_al,
                        )
                        g_tok, g_lp, g_thr = gather_gumbel_for_unpacked(
                            mb_idx, sample_order, gumbel_data,
                            gumbel_k, device,
                        )
                        kl_loss = gumbel_kl_loss(
                            pol_logits_flat, g_tok, g_lp, g_thr, gumbel_k,
                        )
                        del pol_logits_flat, g_tok, g_lp, g_thr

                    total_loss = sft_loss + args.kl_coef * kl_loss
                    scaled_loss = total_loss / accum_steps
                    scaled_loss.backward()

                    with torch.no_grad():
                        batch_action_tokens = sum(mb_al)
                        epoch_loss_acc += float(sft_loss.item())
                        epoch_kl_acc += float(kl_loss.item())
                        epoch_tokens_acc += batch_action_tokens
                        epoch_seqs_acc += len(mb_al)
                        local_fwd_count += 1

                    del mb_ids, mb_attn, logits, outputs, logp, scaled_loss, sft_loss, kl_loss, total_loss

                except torch.cuda.OutOfMemoryError:
                    for v in ("mb_ids", "mb_attn", "logits", "outputs",
                              "logp", "scaled_loss", "sft_loss", "kl_loss",
                              "total_loss"):
                        locals().pop(v, None)
                    torch.cuda.empty_cache()
                    if is_main_rank():
                        print(f"[OOM] batch {step_in_epoch+1} — retrying {len(mb_idx)} samples individually (distill skipped)")

                    n_samples = len(mb_idx)
                    batch_loss_acc = 0.0
                    batch_tokens_acc = 0
                    for si in range(n_samples):
                        s_ids, s_attn, s_pl, s_al = pad_batch(
                            [mb_idx[si]], epoch_seqs, epoch_pls, epoch_als,
                            tokenizer.pad_token_id,
                        )
                        s_ids = s_ids.to(device, non_blocking=True)
                        s_attn = s_attn.to(device, non_blocking=True)
                        with autocast_ctx(device):
                            out = model(input_ids=s_ids, attention_mask=s_attn, use_cache=False)
                            lg = out.logits if hasattr(out, "logits") else out[0]
                            lp = logprob_action_tokens(lg, s_ids, s_pl, s_al, normalize_by_len=True)
                        loss_i = -lp.mean() / (accum_steps * n_samples)
                        loss_i.backward()
                        with torch.no_grad():
                            batch_loss_acc += float((-lp.mean()).item())
                            batch_tokens_acc += s_al[0]
                        del s_ids, s_attn, out, lg, lp, loss_i
                        torch.cuda.empty_cache()
                    with torch.no_grad():
                        epoch_loss_acc += batch_loss_acc / n_samples
                        epoch_tokens_acc += batch_tokens_acc
                        epoch_seqs_acc += n_samples
                    local_fwd_count += 1

                del mb_ids_cpu, mb_attn_cpu

            # ---- Optimizer step at accumulation boundaries ----
            is_accum_boundary = (step_in_epoch + 1) % accum_steps == 0
            is_last_step = (step_in_epoch + 1) == len(my_batch_order)

            if is_accum_boundary or is_last_step:
                allreduce_coalesced_grads(trainable_params)
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                torch.cuda.empty_cache()
                optim.step()
                optim.zero_grad(set_to_none=False)
                global_step += 1

                # Log at optimizer step boundaries
                if is_main_rank() and global_step % args.log_every == 0:
                    avg_loss = epoch_loss_acc / max(1, local_fwd_count)
                    kl_str = ""
                    if args.kl_coef > 0:
                        avg_kl = epoch_kl_acc / max(1, local_fwd_count)
                        kl_str = f" kl={avg_kl:.4f}"
                    print(
                        f"[epoch {epoch}] step={global_step}/{total_steps} "
                        f"loss={avg_loss:.4f}{kl_str} lr={lr:.2e} "
                        f"fwd={step_in_epoch + 1}/{len(my_batch_order)} "
                        f"seqs={epoch_seqs_acc}"
                    )

        # ---- Epoch metrics ----
        t_epoch_end = time.time()

        if world_size > 1:
            local_metrics = {
                "loss": epoch_loss_acc,
                "kl": epoch_kl_acc,
                "tokens": float(epoch_tokens_acc),
                "seqs": float(epoch_seqs_acc),
                "n_fwd": float(local_fwd_count),
            }
            agg = allreduce_scalars(local_metrics)
            total_fwd = int(agg["n_fwd"])
            epoch_loss_acc = agg["loss"]
            epoch_kl_acc = agg["kl"]
            epoch_tokens_acc = int(agg["tokens"])
            epoch_seqs_acc = int(agg["seqs"])
        else:
            total_fwd = local_fwd_count

        avg_epoch_loss = epoch_loss_acc / max(1, total_fwd)
        avg_epoch_kl = epoch_kl_acc / max(1, total_fwd)
        epoch_time = t_epoch_end - t_epoch_start

        if is_main_rank():
            kl_str = ""
            if args.kl_coef > 0:
                avg_total = avg_epoch_loss + args.kl_coef * avg_epoch_kl
                kl_str = f" kl={avg_epoch_kl:.4f} total={avg_total:.4f}"
            print(
                f"\n[Epoch {epoch}/{args.num_epochs}] "
                f"avg_loss={avg_epoch_loss:.4f}{kl_str} "
                f"seqs={epoch_seqs_acc:,} "
                f"tokens={epoch_tokens_acc:,} "
                f"fwd_passes={total_fwd} "
                f"optim_steps={global_step} "
                f"time={epoch_time:.1f}s"
            )

            if wandb:
                log_dict = {
                    "sft/epoch": epoch,
                    "sft/loss": avg_epoch_loss,
                    "sft/action_tokens": epoch_tokens_acc,
                    "sft/sequences": epoch_seqs_acc,
                    "sft/lr": lr,
                    "sft/epoch_time_sec": epoch_time,
                    "sft/fwd_passes": total_fwd,
                    "sft/optim_steps": global_step,
                }
                if args.kl_coef > 0:
                    log_dict["sft/kl_loss"] = avg_epoch_kl
                    log_dict["sft/total_loss"] = avg_epoch_loss + args.kl_coef * avg_epoch_kl
                wandb.log(log_dict, step=global_step)

        # ---- Save checkpoint ----
        if (epoch + 1) % args.save_every == 0 or epoch == args.num_epochs - 1:
            if is_main_rank():
                ckpt_dir = output_dir / f"sft_ckpt_epoch_{epoch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                unwrapped_model.save_pretrained(str(ckpt_dir))
                tokenizer.save_pretrained(str(ckpt_dir))

                # Save metadata for resuming
                meta = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "avg_loss": avg_epoch_loss,
                    "total_samples": total_samples,
                    "model_name": model_name,
                }
                (ckpt_dir / "sft_meta.json").write_text(json.dumps(meta, indent=2))
                print(f"[Checkpoint] Saved to {ckpt_dir}")
            barrier()

    # ---- Cleanup ----
    if is_main_rank():
        print(f"\n[SFT] Training complete! {args.num_epochs} epochs, {global_step} optim steps")
    dist_cleanup()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Ensure the traceback is visible even when builtins.print is
        # suppressed on non-main ranks.
        import traceback, sys
        rank = os.environ.get("RANK", "?")
        sys.__stderr__.write(f"[Rank {rank}] FATAL:\n{traceback.format_exc()}\n")
        sys.__stderr__.flush()
        raise
