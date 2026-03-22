#!/usr/bin/env python3
"""
SFT trainer for ADP sharegpt-format data, using the same infrastructure as
train_sft.py (unsloth 4-bit QLoRA, manual gradient all-reduce, gradient
checkpointing).

This avoids DDP entirely — each GPU loads the 4-bit model independently and
only LoRA gradients are all-reduced. This is what makes multi-GPU QLoRA work.

Input data format (JSONL, one per line):
    {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}],
     "system": "..."}

Usage:
    # Single GPU:
    python train_sft_adp.py --data data/adp_sft_train.jsonl

    # Multi-GPU:
    CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 torchrun --nproc_per_node=6 train_sft_adp.py \
        --data data/adp_sft_train.jsonl
"""
import argparse
import json
import math
import os
import random
import time

os.environ.setdefault("NCCL_P2P_DISABLE", "1")

# CRITICAL: Isolate GPU BEFORE importing torch to prevent each rank from
# creating CUDA contexts on all visible GPUs (~518 MiB each).
from dist_utils import isolate_gpu_per_rank
isolate_gpu_per_rank()

import torch
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

from config import Config, setup_environment, autocast_ctx
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
        description="SFT trainer for ADP sharegpt-format data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- Data --
    parser.add_argument("--data", type=str, required=True,
        help="Path to JSONL file in sharegpt format")

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
    parser.add_argument("--num-epochs", type=int, default=1,
        help="Number of full passes over the dataset")
    parser.add_argument("--mini-batch-size", type=int, default=1,
        help="Mini-batch size per forward pass")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16,
        help="Number of mini-batches to accumulate before each optimizer step")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
        help="Max gradient norm for clipping")
    parser.add_argument("--warmup-ratio", type=float, default=0.1,
        help="Fraction of total steps for linear warmup")
    parser.add_argument("--weight-decay", type=float, default=0.01,
        help="Weight decay for AdamW")

    # -- Data processing --
    parser.add_argument("--max-seq-length", type=int, default=8192,
        help="Max sequence length (sequences longer are truncated)")
    parser.add_argument("--max-samples", type=int, default=None,
        help="Train on only the first N samples (for debugging)")
    parser.add_argument("--shuffle-seed", type=int, default=42,
        help="Random seed for data shuffling")

    # -- Checkpointing --
    parser.add_argument("--save-every", type=int, default=1,
        help="Save checkpoint every N epochs")
    parser.add_argument("--save-steps", type=int, default=500,
        help="Save checkpoint every N optimizer steps (0 to disable)")
    parser.add_argument("--resume", type=str, default=None,
        help="Path to checkpoint directory to resume from")
    parser.add_argument("--output-dir", type=str,
        default="/home/ubuntu/.cache/huggingface/adp_baseline/sft_lora",
        help="Output directory for checkpoints")

    # -- Logging --
    parser.add_argument("--log-every", type=int, default=10,
        help="Log metrics every N optimizer steps")
    parser.add_argument("--wandb-project", type=str, default="games",
        help="wandb project name")
    parser.add_argument("--run-name", type=str, default="adp-baseline-lora-r16",
        help="wandb run name")

    # -- Root directory --
    parser.add_argument("--root", type=str, default=None,
        help="Root directory for cache and outputs")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading: sharegpt JSONL → tokenized sequences
# ---------------------------------------------------------------------------

def _process_line(line: str, tokenizer, max_seq_length: int):
    """Process a single JSONL line. Returns (full_ids, p_len, a_len) or None."""
    item = json.loads(line)
    convs = item.get("conversations", [])
    system = (item.get("system") or "").strip()

    if len(convs) < 2:
        return None

    # Merge consecutive same-role turns
    merged = []
    for turn in convs:
        role = turn.get("from", "")
        value = (turn.get("value") or "").strip()
        if not value or not role:
            continue
        if merged and merged[-1]["from"] == role:
            merged[-1]["value"] += "\n" + value
        else:
            merged.append({"from": role, "value": value})

    if len(merged) < 2 or merged[0]["from"] != "human":
        return None

    while merged and merged[-1]["from"] != "gpt":
        merged.pop()
    if len(merged) < 2:
        return None

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    for turn in merged:
        role = "user" if turn["from"] == "human" else "assistant"
        messages.append({"role": role, "content": turn["value"]})

    try:
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    except Exception:
        return None

    prompt_messages = messages[:-1]
    try:
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    except Exception:
        return None

    p_len = len(prompt_ids)
    a_len = len(full_ids) - p_len

    if a_len <= 0:
        return None

    if len(full_ids) > max_seq_length:
        full_ids = full_ids[:max_seq_length]
        a_len = max_seq_length - p_len
        if a_len <= 0:
            return None

    return (full_ids, p_len, a_len)


def _worker_init(tokenizer_name, max_seq_len):
    """Initialize tokenizer in each worker process."""
    global _worker_tokenizer, _worker_max_seq
    from transformers import AutoTokenizer
    _worker_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    _worker_max_seq = max_seq_len


def _worker_process(line):
    """Process one line using the worker's tokenizer."""
    return _process_line(line, _worker_tokenizer, _worker_max_seq)


def load_sharegpt_jsonl(path: str, tokenizer, max_seq_length: int,
                        max_samples: int = None,
                        num_workers: int = 16) -> Tuple[List[torch.Tensor], List[int], List[int]]:
    """Load sharegpt JSONL and tokenize into (seqs, prompt_lens, action_lens).

    Uses multiprocessing for fast tokenization.
    """
    import multiprocessing as mp

    print(f"[Data] Loading {path}...")
    with open(path) as f:
        lines = f.readlines()

    if max_samples:
        lines = lines[:max_samples]

    print(f"[Data] Processing {len(lines)} examples with {num_workers} workers...")

    # Get tokenizer name for workers
    tokenizer_name = tokenizer.name_or_path

    seqs = []
    prompt_lens = []
    action_lens = []
    skipped = 0

    with mp.Pool(num_workers, initializer=_worker_init,
                 initargs=(tokenizer_name, max_seq_length)) as pool:
        results = pool.imap(_worker_process, lines, chunksize=256)
        for result in tqdm(results, total=len(lines), desc="Tokenizing",
                           disable=not is_main_rank()):
            if result is None:
                skipped += 1
                continue
            full_ids, p_len, a_len = result
            seqs.append(torch.tensor(full_ids, dtype=torch.long))
            prompt_lens.append(p_len)
            action_lens.append(a_len)

    print(f"[Data] Loaded {len(seqs)} samples, skipped {skipped}")
    return seqs, prompt_lens, action_lens


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

def prepare_batches(seqs: List[torch.Tensor], mini_batch_size: int) -> List[List[int]]:
    """Sort sequences by length and create mini-batches."""
    indices = sorted(range(len(seqs)), key=lambda i: seqs[i].shape[0])
    batches = []
    for start in range(0, len(indices), mini_batch_size):
        batches.append(indices[start:start + mini_batch_size])
    return batches


def pad_batch(batch_indices, seqs, prompt_lens, action_lens, pad_id):
    """Pad a mini-batch and return (ids, attn, pls, als)."""
    batch_seqs = [seqs[i] for i in batch_indices]
    pls = [prompt_lens[i] for i in batch_indices]
    als = [action_lens[i] for i in batch_indices]

    max_len = max(s.shape[0] for s in batch_seqs)
    B = len(batch_seqs)
    ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((B, max_len), dtype=torch.long)
    for i, s in enumerate(batch_seqs):
        ids[i, :s.shape[0]] = s
        attn[i, :s.shape[0]] = 1

    return ids, attn, pls, als


# ---------------------------------------------------------------------------
# Loss computation (same as train_sft.py)
# ---------------------------------------------------------------------------

def logprob_action_tokens(logits, input_ids, prompt_lens, action_lens,
                          normalize_by_len=True):
    """Compute per-sequence log-probs over action tokens only."""
    results = []
    for i in range(logits.size(0)):
        pl, al = prompt_lens[i], action_lens[i]
        if al <= 0:
            continue
        l_start = max(0, pl - 1)
        l_end = pl + al - 1
        if l_end <= l_start:
            continue

        seg_logits = logits[i, l_start:l_end, :].float()
        seg_targets = input_ids[i, pl:pl + al]

        if seg_logits.shape[0] != seg_targets.shape[0]:
            min_len = min(seg_logits.shape[0], seg_targets.shape[0])
            seg_logits = seg_logits[:min_len]
            seg_targets = seg_targets[:min_len]

        log_p = F.log_softmax(seg_logits, dim=-1)
        tok_lp = log_p.gather(1, seg_targets.unsqueeze(1)).squeeze(1)
        total = tok_lp.sum()
        results.append(total / al if normalize_by_len else total)

    if not results:
        return torch.zeros(1, device=logits.device, dtype=torch.float32)
    return torch.stack(results)


# ---------------------------------------------------------------------------
# LR scheduler
# ---------------------------------------------------------------------------

def get_lr(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * (step + 1) / (warmup_steps + 1)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def set_lr(optimizer, lr):
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

    max_seq_len = args.max_seq_length

    # ---- Output directory ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load model + LoRA (unsloth 4-bit) ----
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
        attn_implementation="sdpa",
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

    unwrapped_model = model

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

        meta_path = resume_path / "sft_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            start_epoch = meta.get("epoch", 0) + 1
            global_step = meta.get("global_step", 0)
            print(f"[Resume] Starting from epoch {start_epoch}, step {global_step}")

    # ---- Load data (rank 0 tokenizes, others wait then load cache) ----
    cache_path = Path(args.data).with_suffix(f".tokenized_ms{max_seq_len}.pt")
    need_tokenize = False

    if cache_path.exists():
        cached = torch.load(str(cache_path), map_location="cpu", weights_only=False)
        seqs = cached["seqs"]
        prompt_lens = cached["prompt_lens"]
        action_lens = cached["action_lens"]
        print(f"[Data] Loaded {len(seqs)} samples from cache")
        if args.max_samples and args.max_samples < len(seqs):
            seqs = seqs[:args.max_samples]
            prompt_lens = prompt_lens[:args.max_samples]
            action_lens = action_lens[:args.max_samples]
    else:
        if is_main_rank():
            seqs, prompt_lens, action_lens = load_sharegpt_jsonl(
                args.data, tokenizer, max_seq_len, max_samples=args.max_samples,
            )
            print(f"[Data] Saving tokenized cache to {cache_path}...")
            torch.save({
                "seqs": seqs, "prompt_lens": prompt_lens,
                "action_lens": action_lens,
            }, str(cache_path))
            print(f"[Data] Cache saved")
        barrier()
        if not is_main_rank():
            cached = torch.load(str(cache_path), map_location="cpu", weights_only=False)
            seqs = cached["seqs"]
            prompt_lens = cached["prompt_lens"]
            action_lens = cached["action_lens"]
            if args.max_samples and args.max_samples < len(seqs):
                seqs = seqs[:args.max_samples]
                prompt_lens = prompt_lens[:args.max_samples]
                action_lens = action_lens[:args.max_samples]
            print(f"[Data] Rank {rank}: Loaded {len(seqs)} samples from cache")

    total_samples = len(seqs)

    if total_samples == 0:
        print("[Data] ERROR: No samples loaded.")
        return

    # ---- Optimizer ----
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay,
    )
    print(f"[SFT] Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    # ---- Schedule ----
    accum_steps = args.gradient_accumulation_steps
    batches_per_epoch_total = math.ceil(total_samples / args.mini_batch_size)
    batches_per_epoch_per_rank = math.ceil(batches_per_epoch_total / world_size)
    steps_per_epoch = math.ceil(batches_per_epoch_per_rank / accum_steps)
    total_steps = steps_per_epoch * args.num_epochs
    warmup_steps = int(args.warmup_ratio * total_steps)

    print(f"\n{'=' * 60}")
    print(f"  ADP SFT Training Configuration")
    print(f"{'=' * 60}")
    print(f"  Model:              {model_name}")
    print(f"  LoRA rank:          {args.lora_rank}")
    print(f"  LoRA alpha:         {args.lora_alpha}")
    print(f"  Learning rate:      {args.lr}")
    print(f"  Epochs:             {args.num_epochs}")
    print(f"  Total samples:      {total_samples}")
    print(f"  Mini-batch size:    {args.mini_batch_size}")
    print(f"  Grad accum steps:   {accum_steps}")
    print(f"  Effective batch:    {args.mini_batch_size * accum_steps * world_size}")
    print(f"  Steps/epoch:        {steps_per_epoch}")
    print(f"  Total optim steps:  {total_steps}")
    print(f"  Warmup steps:       {warmup_steps}")
    print(f"  Max seq length:     {max_seq_len}")
    print(f"  Device:             {device}")
    print(f"  World size:         {world_size}")
    print(f"  Output dir:         {output_dir}")
    print(f"{'=' * 60}\n")

    # ---- wandb ----
    if wandb and is_main_rank():
        wandb.login(key=os.getenv("WANDB_API_KEY", ""), relogin=True)
        wandb.init(
            entity="MARL_collab", project=args.wandb_project,
            name=args.run_name,
        )
        wandb.config.update(vars(args))
        print("[wandb] Initialized")

    # ---- Sequence length stats ----
    seq_lens = [seqs[i].shape[0] for i in range(total_samples)]
    print(f"[Data] Sequence length stats: "
          f"min={min(seq_lens)}, max={max(seq_lens)}, "
          f"mean={sum(seq_lens)/len(seq_lens):.0f}, "
          f"median={sorted(seq_lens)[len(seq_lens)//2]}")

    # ====================================================================
    # Training loop
    # ====================================================================
    lr = args.lr
    t_last_step = None

    for epoch in range(start_epoch, args.num_epochs):
        t_epoch_start = time.time()
        t_last_step = time.time()
        model.train()

        # Shuffle (deterministic across ranks)
        rng = random.Random(args.shuffle_seed + epoch)
        sample_order = list(range(total_samples))
        rng.shuffle(sample_order)

        epoch_seqs = [seqs[i] for i in sample_order]
        epoch_pls = [prompt_lens[i] for i in sample_order]
        epoch_als = [action_lens[i] for i in sample_order]

        # Length-sorted mini-batches
        batches = prepare_batches(epoch_seqs, args.mini_batch_size)

        # Shuffle batch order
        batch_order = list(range(len(batches)))
        rng.shuffle(batch_order)

        # Shard across ranks
        my_batch_order, _ = shard_batches(batch_order, rank, world_size)
        real_batch_count = len(my_batch_order)
        target_len = math.ceil(len(batch_order) / max(1, world_size))
        while len(my_batch_order) < target_len:
            my_batch_order.append(my_batch_order[-1] if my_batch_order else 0)

        # Initialize gradients
        for p in trainable_params:
            if p.grad is None:
                p.grad = torch.zeros_like(p.data)
        optim.zero_grad(set_to_none=False)

        epoch_loss_acc = 0.0
        epoch_tokens_acc = 0
        epoch_seqs_acc = 0
        local_fwd_count = 0

        for step_in_epoch, bi in enumerate(my_batch_order):
            is_padding = step_in_epoch >= real_batch_count

            # LR schedule
            if step_in_epoch % accum_steps == 0:
                lr = get_lr(global_step, total_steps, warmup_steps, args.lr)
                set_lr(optim, lr)

            if is_padding:
                pass
            else:
                mb_idx = batches[bi]
                mb_ids_cpu, mb_attn_cpu, mb_pl, mb_al = pad_batch(
                    mb_idx, epoch_seqs, epoch_pls, epoch_als,
                    tokenizer.pad_token_id,
                )

                try:
                    mb_ids = mb_ids_cpu.to(device, non_blocking=True)
                    mb_attn = mb_attn_cpu.to(device, non_blocking=True)

                    with autocast_ctx(device):
                        outputs = model(
                            input_ids=mb_ids, attention_mask=mb_attn,
                            use_cache=False,
                        )
                        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                        logp = logprob_action_tokens(
                            logits, mb_ids, mb_pl, mb_al, normalize_by_len=True,
                        )

                    sft_loss = -logp.mean()
                    scaled_loss = sft_loss / accum_steps
                    scaled_loss.backward()

                    with torch.no_grad():
                        epoch_loss_acc += float(sft_loss.item())
                        epoch_tokens_acc += sum(mb_al)
                        epoch_seqs_acc += len(mb_al)
                        local_fwd_count += 1

                    del mb_ids, mb_attn, logits, outputs, logp, scaled_loss, sft_loss

                except torch.cuda.OutOfMemoryError:
                    for v in ("mb_ids", "mb_attn", "logits", "outputs",
                              "logp", "scaled_loss", "sft_loss"):
                        locals().pop(v, None)
                    torch.cuda.empty_cache()
                    if is_main_rank():
                        print(f"[OOM] batch {step_in_epoch+1} — skipping")

                del mb_ids_cpu, mb_attn_cpu

            # ---- Optimizer step ----
            is_accum_boundary = (step_in_epoch + 1) % accum_steps == 0
            is_last_step = (step_in_epoch + 1) == len(my_batch_order)

            if is_accum_boundary or is_last_step:
                allreduce_coalesced_grads(trainable_params)
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                torch.cuda.empty_cache()
                optim.step()
                optim.zero_grad(set_to_none=False)
                global_step += 1

                # Log
                if is_main_rank() and global_step % args.log_every == 0:
                    avg_loss = epoch_loss_acc / max(1, local_fwd_count)
                    t_now = time.time()
                    step_time = t_now - t_last_step
                    elapsed = t_now - t_epoch_start
                    remaining = step_time / max(1, args.log_every) * (total_steps - global_step)
                    t_last_step = t_now
                    print(
                        f"[epoch {epoch}] step={global_step}/{total_steps} "
                        f"loss={avg_loss:.4f} lr={lr:.2e} "
                        f"seqs={epoch_seqs_acc} "
                        f"step_time={step_time:.1f}s "
                        f"elapsed={elapsed/60:.1f}m "
                        f"eta={remaining/60:.1f}m"
                    )

                    if wandb:
                        wandb.log({
                            "sft/loss": avg_loss,
                            "sft/lr": lr,
                            "sft/sequences": epoch_seqs_acc,
                            "sft/action_tokens": epoch_tokens_acc,
                        }, step=global_step)

                # Mid-epoch checkpoint
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    if is_main_rank():
                        ckpt_dir = output_dir / f"checkpoint-{global_step}"
                        ckpt_dir.mkdir(parents=True, exist_ok=True)
                        unwrapped_model.save_pretrained(str(ckpt_dir))
                        tokenizer.save_pretrained(str(ckpt_dir))
                        meta = {
                            "epoch": epoch, "global_step": global_step,
                            "avg_loss": epoch_loss_acc / max(1, local_fwd_count),
                        }
                        (ckpt_dir / "sft_meta.json").write_text(json.dumps(meta, indent=2))
                        print(f"[Checkpoint] Saved to {ckpt_dir}")
                    barrier()

        # ---- Epoch summary ----
        t_epoch_end = time.time()

        if world_size > 1:
            agg = allreduce_scalars({
                "loss": epoch_loss_acc,
                "tokens": float(epoch_tokens_acc),
                "seqs": float(epoch_seqs_acc),
                "n_fwd": float(local_fwd_count),
            })
            total_fwd = int(agg["n_fwd"])
            epoch_loss_acc = agg["loss"]
            epoch_tokens_acc = int(agg["tokens"])
            epoch_seqs_acc = int(agg["seqs"])
        else:
            total_fwd = local_fwd_count

        avg_epoch_loss = epoch_loss_acc / max(1, total_fwd)

        if is_main_rank():
            print(
                f"\n[Epoch {epoch}/{args.num_epochs}] "
                f"avg_loss={avg_epoch_loss:.4f} "
                f"seqs={epoch_seqs_acc:,} "
                f"tokens={epoch_tokens_acc:,} "
                f"time={t_epoch_end - t_epoch_start:.1f}s"
            )

        # ---- End-of-epoch checkpoint ----
        if (epoch + 1) % args.save_every == 0 or epoch == args.num_epochs - 1:
            if is_main_rank():
                ckpt_dir = output_dir / f"sft_ckpt_epoch_{epoch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                unwrapped_model.save_pretrained(str(ckpt_dir))
                tokenizer.save_pretrained(str(ckpt_dir))
                meta = {
                    "epoch": epoch, "global_step": global_step,
                    "avg_loss": avg_epoch_loss, "total_samples": total_samples,
                    "model_name": model_name,
                }
                (ckpt_dir / "sft_meta.json").write_text(json.dumps(meta, indent=2))
                print(f"[Checkpoint] Saved to {ckpt_dir}")
            barrier()

    # ---- Cleanup ----
    if is_main_rank():
        print(f"\n[Done] Training complete. Final checkpoint in {output_dir}")
    if wandb and is_main_rank():
        wandb.finish()
    dist_cleanup()


if __name__ == "__main__":
    main()
