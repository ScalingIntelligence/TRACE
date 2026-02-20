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
  - Checkpoint saving and resuming
  - wandb logging

Usage:
    # Single GPU:
    python train_sft.py --sft-data file1.json,file2.json

    # Multi-GPU:
    torchrun --nproc_per_node=4 train_sft.py --sft-data file1.json,file2.json
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
from typing import Dict, List, Tuple

from tqdm import tqdm

from config import Config, setup_environment, autocast_ctx
from ppo import logprob_action_tokens, pad_to_device, build_prompt_plus_action
from sft_buffer import SFTBuffer
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
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
        help="Max gradient norm for clipping")
    parser.add_argument("--warmup-ratio", type=float, default=0.05,
        help="Fraction of total steps for linear warmup")
    parser.add_argument("--weight-decay", type=float, default=0.01,
        help="Weight decay for AdamW")

    # -- Data processing --
    parser.add_argument("--compact-tools", action="store_true", default=False,
        help="Use compressed tool schemas (strips descriptions)")
    parser.add_argument("--max-seq-length", type=int, default=None,
        help="Max sequence length (default: Config.MAX_SEQ_LENGTH)")
    parser.add_argument("--shuffle-seed", type=int, default=42,
        help="Random seed for data shuffling")

    # -- Checkpointing --
    parser.add_argument("--save-every", type=int, default=1,
        help="Save checkpoint every N epochs")
    parser.add_argument("--resume", type=str, default=None,
        help="Path to checkpoint directory to resume from")
    parser.add_argument("--output-dir", type=str, default=None,
        help="Output directory for checkpoints (default: auto)")

    # -- Logging --
    parser.add_argument("--log-every", type=int, default=10,
        help="Log metrics every N steps")
    parser.add_argument("--wandb-project", type=str, default="games",
        help="wandb project name")

    # -- Root directory --
    parser.add_argument("--root", type=str, default=None,
        help="Root directory for cache and outputs")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data preparation
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
            set_peft_model_state_dict(model, state)
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
    print(f"[SFT] Loading data from {len(sft_paths)} files...")
    sft_buffer = SFTBuffer(sft_paths, tokenizer, compact_tools=args.compact_tools)
    total_samples = len(sft_buffer)
    print(f"[SFT] Loaded {total_samples} samples from {len(sft_paths)} files")

    if total_samples == 0:
        print("[SFT] ERROR: No samples loaded. Check that JSON files contain reward=1.0 simulations.")
        return

    # Filter sequences that exceed max_seq_length
    valid_indices = []
    for i in range(total_samples):
        seq_len = sft_buffer._seqs_cpu[i].shape[0]
        if seq_len <= max_seq_len:
            valid_indices.append(i)

    if len(valid_indices) < total_samples:
        skipped = total_samples - len(valid_indices)
        sft_buffer._seqs_cpu = [sft_buffer._seqs_cpu[i] for i in valid_indices]
        sft_buffer._prompt_lens = [sft_buffer._prompt_lens[i] for i in valid_indices]
        sft_buffer._action_lens = [sft_buffer._action_lens[i] for i in valid_indices]
        total_samples = len(sft_buffer)
        print(f"[SFT] Filtered {skipped} sequences > {max_seq_len} tokens. Remaining: {total_samples}")

    # ---- Optimizer ----
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    print(f"[SFT] Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    # ---- Compute schedule ----
    batches_per_epoch_total = math.ceil(total_samples / args.mini_batch_size)
    batches_per_epoch_per_rank = math.ceil(batches_per_epoch_total / world_size)
    total_steps = batches_per_epoch_per_rank * args.num_epochs
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
    print(f"  Batches/epoch:      {batches_per_epoch_total}")
    print(f"  Total steps:        {total_steps}")
    print(f"  Warmup steps:       {warmup_steps}")
    print(f"  Max seq length:     {max_seq_len}")
    print(f"  Max grad norm:      {args.max_grad_norm}")
    print(f"  Compact tools:      {args.compact_tools}")
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
            "max_seq_length": max_seq_len,
            "compact_tools": args.compact_tools,
            "world_size": world_size,
        })
        print("[wandb] Initialized")

    # ---- Sequence length stats ----
    seq_lens = [sft_buffer._seqs_cpu[i].shape[0] for i in range(total_samples)]
    print(f"[SFT] Sequence length stats: "
          f"min={min(seq_lens)}, max={max(seq_lens)}, "
          f"mean={sum(seq_lens)/len(seq_lens):.0f}, "
          f"median={sorted(seq_lens)[len(seq_lens)//2]}")

    # ====================================================================
    # Training loop
    # ====================================================================
    seqs = sft_buffer._seqs_cpu
    prompt_lens = sft_buffer._prompt_lens
    action_lens = sft_buffer._action_lens

    for epoch in range(start_epoch, args.num_epochs):
        t_epoch_start = time.time()
        model.train()

        # Shuffle sample order (deterministic across ranks for sharding)
        rng = random.Random(args.shuffle_seed + epoch)
        sample_order = list(range(total_samples))
        rng.shuffle(sample_order)

        # Reorder data by shuffled indices
        epoch_seqs = [seqs[i] for i in sample_order]
        epoch_pls = [prompt_lens[i] for i in sample_order]
        epoch_als = [action_lens[i] for i in sample_order]

        # Create length-sorted mini-batches
        batches = prepare_batches(epoch_seqs, args.mini_batch_size)

        # Shuffle batch order (same seed on all ranks)
        batch_order = list(range(len(batches)))
        rng.shuffle(batch_order)

        # Shard batches across ranks
        my_batch_order, n_total_batches = shard_batches(batch_order, rank, world_size)

        # Zero gradients once — accumulate across all mini-batches in epoch
        optim.zero_grad(set_to_none=True)

        epoch_loss_acc = 0.0
        epoch_tokens_acc = 0
        local_updates = 0

        for step_in_epoch, bi in enumerate(my_batch_order):
            # LR schedule
            lr = get_lr(global_step, total_steps, warmup_steps, args.lr)
            set_lr(optim, lr)

            mb_idx = batches[bi]
            mb_ids_cpu, mb_attn_cpu, mb_pl, mb_al = pad_batch(
                mb_idx, epoch_seqs, epoch_pls, epoch_als, tokenizer.pad_token_id
            )

            # Forward + backward with OOM retry: if the full mini-batch
            # OOMs, clear the CUDA cache and re-process samples one by
            # one so that no batch is silently dropped.
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
                scaled_loss = sft_loss / n_total_batches
                scaled_loss.backward()

                with torch.no_grad():
                    batch_action_tokens = sum(mb_al)
                    epoch_loss_acc += float(sft_loss.item())
                    epoch_tokens_acc += batch_action_tokens
                    local_updates += 1

                del mb_ids, mb_attn, logits, outputs, logp, scaled_loss, sft_loss

            except torch.cuda.OutOfMemoryError:
                # Clean up partially-allocated tensors and retry one sample
                # at a time to avoid dropping any data.
                for v in ("mb_ids", "mb_attn", "logits", "outputs",
                          "logp", "scaled_loss", "sft_loss"):
                    locals().pop(v, None)
                torch.cuda.empty_cache()
                if is_main_rank():
                    print(f"[OOM] batch {step_in_epoch+1} — retrying {len(mb_idx)} samples individually")

                for si in range(len(mb_idx)):
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
                    loss_i = -lp.mean() / n_total_batches
                    loss_i.backward()
                    with torch.no_grad():
                        epoch_loss_acc += float((-lp.mean()).item())
                        epoch_tokens_acc += s_al[0]
                    del s_ids, s_attn, out, lg, lp, loss_i
                    torch.cuda.empty_cache()
                local_updates += 1

            del mb_ids_cpu, mb_attn_cpu

            # Log periodically
            if is_main_rank() and (global_step + 1) % args.log_every == 0:
                avg_loss = epoch_loss_acc / max(1, local_updates)
                print(
                    f"[epoch {epoch}] step={global_step + 1}/{total_steps} "
                    f"loss={avg_loss:.4f} lr={lr:.2e} "
                    f"batch={step_in_epoch + 1}/{len(my_batch_order)}"
                )

            global_step += 1

        # All-reduce gradients across ranks
        allreduce_coalesced_grads(trainable_params)

        # Gradient clip + optimizer step
        torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
        optim.step()

        # ---- Epoch metrics ----
        t_epoch_end = time.time()

        if world_size > 1:
            local_metrics = {
                "loss": epoch_loss_acc,
                "tokens": float(epoch_tokens_acc),
                "n_updates": float(local_updates),
            }
            agg = allreduce_scalars(local_metrics)
            total_updates = int(agg["n_updates"])
            epoch_loss_acc = agg["loss"]
            epoch_tokens_acc = int(agg["tokens"])
        else:
            total_updates = local_updates

        avg_epoch_loss = epoch_loss_acc / max(1, total_updates)
        epoch_time = t_epoch_end - t_epoch_start

        if is_main_rank():
            print(
                f"\n[Epoch {epoch}/{args.num_epochs}] "
                f"avg_loss={avg_epoch_loss:.4f} "
                f"tokens={epoch_tokens_acc:,} "
                f"updates={total_updates} "
                f"time={epoch_time:.1f}s"
            )

            if wandb:
                wandb.log({
                    "sft/epoch": epoch,
                    "sft/loss": avg_epoch_loss,
                    "sft/action_tokens": epoch_tokens_acc,
                    "sft/lr": lr,
                    "sft/epoch_time_sec": epoch_time,
                    "sft/updates": total_updates,
                }, step=global_step)

        # ---- Save checkpoint ----
        if (epoch + 1) % args.save_every == 0 or epoch == args.num_epochs - 1:
            if is_main_rank():
                ckpt_dir = output_dir / f"sft_ckpt_epoch_{epoch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(ckpt_dir))
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
        print(f"\n[SFT] Training complete! {args.num_epochs} epochs, {global_step} steps")
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
