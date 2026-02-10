#!/usr/bin/env python3
"""
End-to-end profiling of GRPO training step on Qwen3-30B-A3B-Instruct-2507.

Loads the real model with LoRA, creates synthetic tokenized data matching
production length distributions, and profiles both ORIGINAL and OPTIMIZED
training steps side-by-side.

Profiles steps 5-7:
  5. Pre-tokenize (skipped — uses synthetic pre-tokenized sequences)
  5c. Batch formation + padding/caching
  6. old_logp computation (reference model forward passes under no_grad)
  7. Gradient updates (training forward + backward + optimizer step)

Usage:
    conda activate games && CUDA_VISIBLE_DEVICES=0 python profile_grpo_training.py
    conda activate games && CUDA_VISIBLE_DEVICES=0 python profile_grpo_training.py --n-samples 300
"""
from unsloth import FastLanguageModel

import argparse
import copy
import gc
import os
import random
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from config import Config, autocast_ctx
from ppo import pad_to_device, logprob_action_tokens
from train_grpo_optimized import make_sorted_batches, pad_and_pin, compute_padding_efficiency


def parse_args():
    p = argparse.ArgumentParser(description="Profile GRPO training step on real model")
    p.add_argument("--n-samples", type=int, default=500,
                   help="Number of synthetic samples (production range: 378-713)")
    p.add_argument("--mini-batch-size", type=int, default=8,
                   help="Mini-batch size for gradient updates")
    p.add_argument("--stats-chunk-size-orig", type=int, default=4,
                   help="Stats chunk size for original approach")
    p.add_argument("--stats-chunk-size-opt", type=int, default=8,
                   help="Stats chunk size for optimized approach")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--min-seq-len", type=int, default=200,
                   help="Minimum synthetic sequence length (tokens)")
    p.add_argument("--max-seq-len", type=int, default=6000,
                   help="Maximum synthetic sequence length (tokens)")
    p.add_argument("--skip-original", action="store_true",
                   help="Skip original approach (only profile optimized)")
    return p.parse_args()


def setup_device():
    """Setup CUDA device and TF32."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return device


def load_model(device, lora_rank=16, lora_alpha=16):
    """Load Qwen3-30B-A3B with LoRA, exactly as in training."""
    local_home = Path.home()
    hf_hub = local_home / ".cache" / "huggingface" / "hub"
    hf_hub.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    model_name = Config.MODEL_NAME
    print(f"[Model] Loading {model_name} with LoRA rank={lora_rank}...")
    t0 = time.time()

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
        r=lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_alpha,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    FastLanguageModel.for_training(model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = model.to(device)
    t1 = time.time()
    print(f"[Model] Loaded in {t1-t0:.1f}s")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"[Model] Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    return model, tokenizer, trainable_params


def generate_synthetic_data(N, min_len, max_len, vocab_size, pad_token_id):
    """
    Generate synthetic tokenized sequences with a log-normal length distribution.
    Simulates real rollout data where most sequences are medium-length with
    some very long ones (thinking + action).
    """
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    # Log-normal distribution: bulk around 500-2000, tail up to max_len
    mu = np.log(800)  # median around 800 tokens
    sigma = 0.8
    raw_lens = np.random.lognormal(mu, sigma, N)
    seq_lens = np.clip(raw_lens, min_len, max_len).astype(int).tolist()

    seqs = []
    prompt_lens = []
    action_lens = []

    for L in seq_lens:
        # Random tokens (avoid pad_token_id=0)
        seq = torch.randint(1, vocab_size, (L,))
        # Prompt is ~60-80% of the sequence (realistic for thinking + action)
        pL = max(1, int(L * random.uniform(0.6, 0.8)))
        aL = L - pL
        seqs.append(seq)
        prompt_lens.append(pL)
        action_lens.append(aL)

    # Synthetic advantages (non-zero, realistic range)
    advantages = torch.randn(N) * 0.5

    print(f"[Data] Generated {N} sequences")
    print(f"  Length range: {min(seq_lens)}-{max(seq_lens)} tokens")
    print(f"  Mean length: {np.mean(seq_lens):.0f} tokens")
    print(f"  Median length: {np.median(seq_lens):.0f} tokens")
    print(f"  Total tokens: {sum(seq_lens):,}")

    return seqs, prompt_lens, action_lens, advantages, seq_lens


def save_lora_state(model):
    """Save LoRA adapter state for restoration between profiling runs."""
    state = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            state[name] = param.data.clone()
    return state


def restore_lora_state(model, state):
    """Restore LoRA adapter state."""
    for name, param in model.named_parameters():
        if name in state:
            param.data.copy_(state[name])


# ============================================================================
# Original training step (from train_grpo.py)
# ============================================================================

def profile_original(
    model, tokenizer, trainable_params, device,
    seqs, prompt_lens, action_lens, advantages, seq_lens,
    mini_batch_size, stats_chunk_size, max_grad_norm,
):
    """Profile the original (unoptimized) training step."""
    N = len(seqs)
    timings = {}

    # -- Step 5c: no pre-caching in original (padding done inline) --
    timings["cache_sec"] = 0.0

    # -- Step 6: old_logp with random batching, pad_to_device per chunk --
    model.eval()
    old_logp_cpu = torch.empty(N, dtype=torch.float32)

    torch.cuda.synchronize()
    t6_start = time.time()

    with torch.no_grad():
        for start in range(0, N, stats_chunk_size):
            end = min(N, start + stats_chunk_size)
            mb_idx = list(range(start, end))
            mb_seqs = [seqs[j] for j in mb_idx]
            mb_pl = [prompt_lens[j] for j in mb_idx]
            mb_al = [action_lens[j] for j in mb_idx]

            mb_ids, mb_attn = pad_to_device(mb_seqs, tokenizer.pad_token_id, device)

            with autocast_ctx(device):
                outputs = model(
                    input_ids=mb_ids,
                    attention_mask=mb_attn,
                    use_cache=False,
                )
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            mb_old_logp = logprob_action_tokens(
                logits, mb_ids, mb_pl, mb_al, normalize_by_len=False,
            )
            old_logp_cpu[mb_idx] = mb_old_logp.detach().cpu()
            del logits, outputs, mb_old_logp, mb_ids, mb_attn

    torch.cuda.synchronize()
    timings["step6_sec"] = time.time() - t6_start

    # -- Step 7: gradient updates with random batching --
    model.train()
    optim = torch.optim.AdamW(trainable_params, lr=1e-5)
    idxs = list(range(N))

    torch.cuda.synchronize()
    t7_start = time.time()

    random.shuffle(idxs)
    updates = 0
    for start in range(0, N, mini_batch_size):
        mb = idxs[start:start + mini_batch_size]
        if not mb:
            continue

        mb_seqs = [seqs[i] for i in mb]
        mb_pl = [prompt_lens[i] for i in mb]
        mb_al = [action_lens[i] for i in mb]

        mb_ids, mb_attn = pad_to_device(mb_seqs, tokenizer.pad_token_id, device)
        mb_old_logp = old_logp_cpu[mb].to(device)
        mb_adv = advantages[mb].to(device)

        outputs = model(
            input_ids=mb_ids,
            attention_mask=mb_attn,
            use_cache=False,
        )
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        new_logp = logprob_action_tokens(
            logits, mb_ids, mb_pl, mb_al, normalize_by_len=False,
        )

        ratio = torch.exp(new_logp - mb_old_logp)
        policy_loss = -torch.mean(ratio * mb_adv)
        loss = policy_loss

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
        optim.step()
        updates += 1

        del mb_ids, mb_attn, logits, outputs, new_logp, ratio, loss

    torch.cuda.synchronize()
    timings["step7_sec"] = time.time() - t7_start
    timings["updates"] = updates

    # Compute padding efficiency for random batching
    random_idx = list(range(N))
    random.shuffle(random_idx)
    random_batches = [random_idx[i:i+mini_batch_size] for i in range(0, N, mini_batch_size)]
    timings["pad_eff"] = compute_padding_efficiency(seq_lens, random_batches)
    timings["total_sec"] = timings["cache_sec"] + timings["step6_sec"] + timings["step7_sec"]

    del optim, old_logp_cpu
    torch.cuda.empty_cache()
    gc.collect()

    return timings


# ============================================================================
# Optimized training step (from train_grpo_optimized.py)
# ============================================================================

def profile_optimized(
    model, tokenizer, trainable_params, device,
    seqs, prompt_lens, action_lens, advantages, seq_lens,
    mini_batch_size, stats_chunk_size, max_grad_norm,
):
    """Profile the optimized training step."""
    N = len(seqs)
    timings = {}
    unified_batches = stats_chunk_size == mini_batch_size

    # -- Step 5c: sorted batching + pre-padded + pinned memory caching --
    torch.cuda.synchronize()
    tc_start = time.time()

    train_batches = make_sorted_batches(seq_lens, mini_batch_size)
    if unified_batches:
        stats_batches = train_batches
    else:
        stats_batches = make_sorted_batches(seq_lens, stats_chunk_size)

    # Build batch cache with pre-computed metadata
    def build_cache(batches):
        cache = []
        for mb_idx in batches:
            mb_seqs = [seqs[j] for j in mb_idx]
            mb_ids, mb_attn = pad_and_pin(mb_seqs, tokenizer.pad_token_id)
            mb_pl = [prompt_lens[j] for j in mb_idx]
            mb_al = [action_lens[j] for j in mb_idx]
            cache.append((mb_idx, mb_ids, mb_attn, mb_pl, mb_al))
        return cache

    train_batch_cache = build_cache(train_batches)
    if unified_batches:
        stats_batch_cache = train_batch_cache
    else:
        stats_batch_cache = build_cache(stats_batches)

    timings["cache_sec"] = time.time() - tc_start

    # -- Step 6: old_logp with sorted batching, pre-padded tensors --
    model.eval()
    old_logp_cpu = torch.empty(N, dtype=torch.float32)

    torch.cuda.synchronize()
    t6_start = time.time()

    with torch.no_grad():
        for mb_idx, mb_ids_cpu, mb_attn_cpu, mb_pl, mb_al in stats_batch_cache:
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
                logits, mb_ids, mb_pl, mb_al, normalize_by_len=False,
            )
            old_logp_cpu[mb_idx] = mb_old_logp.detach().cpu()
            del logits, outputs, mb_old_logp, mb_ids, mb_attn

    torch.cuda.synchronize()
    timings["step6_sec"] = time.time() - t6_start

    if not unified_batches:
        del stats_batch_cache

    # -- Step 7: gradient updates with sorted batch order shuffling --
    model.train()
    optim = torch.optim.AdamW(trainable_params, lr=1e-5)

    torch.cuda.synchronize()
    t7_start = time.time()

    batch_order = list(range(len(train_batch_cache)))
    random.shuffle(batch_order)
    updates = 0

    for bi in batch_order:
        mb_idx, mb_ids_cpu, mb_attn_cpu, mb_pl, mb_al = train_batch_cache[bi]

        mb_ids = mb_ids_cpu.to(device, non_blocking=True)
        mb_attn = mb_attn_cpu.to(device, non_blocking=True)
        mb_old_logp = old_logp_cpu[mb_idx].to(device)
        mb_adv = advantages[mb_idx].to(device)

        outputs = model(
            input_ids=mb_ids,
            attention_mask=mb_attn,
            use_cache=False,
        )
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        new_logp = logprob_action_tokens(
            logits, mb_ids, mb_pl, mb_al, normalize_by_len=False,
        )

        ratio = torch.exp(new_logp - mb_old_logp)
        policy_loss = -torch.mean(ratio * mb_adv)
        loss = policy_loss

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
        optim.step()
        updates += 1

        del mb_ids, mb_attn, logits, outputs, new_logp, ratio, loss

    torch.cuda.synchronize()
    timings["step7_sec"] = time.time() - t7_start
    timings["updates"] = updates
    timings["pad_eff"] = compute_padding_efficiency(seq_lens, train_batches)
    timings["total_sec"] = timings["cache_sec"] + timings["step6_sec"] + timings["step7_sec"]

    del optim, old_logp_cpu, train_batch_cache
    torch.cuda.empty_cache()
    gc.collect()

    return timings


def print_results(name, timings):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Cache/pad:     {timings['cache_sec']:8.1f}s")
    print(f"  Step 6 (logp): {timings['step6_sec']:8.1f}s")
    print(f"  Step 7 (grad): {timings['step7_sec']:8.1f}s")
    print(f"  TOTAL:         {timings['total_sec']:8.1f}s")
    print(f"  Pad efficiency:{timings['pad_eff']:8.1%}")
    print(f"  Grad updates:  {timings['updates']}")
    print(f"{'='*60}")


def main():
    args = parse_args()
    device = setup_device()

    print("=" * 60)
    print("GRPO TRAINING STEP PROFILER")
    print(f"  Model: {Config.MODEL_NAME}")
    print(f"  N samples: {args.n_samples}")
    print(f"  Seq lengths: {args.min_seq_len}-{args.max_seq_len}")
    print(f"  Mini-batch size: {args.mini_batch_size}")
    print(f"  Stats chunk (orig): {args.stats_chunk_size_orig}")
    print(f"  Stats chunk (opt):  {args.stats_chunk_size_opt}")
    print(f"  Device: {device}")
    print("=" * 60)

    # Load real model
    model, tokenizer, trainable_params = load_model(
        device, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
    )
    vocab_size = tokenizer.vocab_size

    # Generate synthetic data
    seqs, prompt_lens, action_lens, advantages, seq_lens = generate_synthetic_data(
        N=args.n_samples,
        min_len=args.min_seq_len,
        max_len=args.max_seq_len,
        vocab_size=vocab_size,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Save initial LoRA state for fair comparison
    print("\n[Profile] Saving initial LoRA state...")
    initial_state = save_lora_state(model)

    # --- Warmup: one forward pass to prime CUDA caches ---
    print("[Profile] Warming up CUDA...")
    warmup_ids = seqs[0][:200].unsqueeze(0).to(device)
    warmup_attn = torch.ones_like(warmup_ids)
    with torch.no_grad():
        with autocast_ctx(device):
            _ = model(input_ids=warmup_ids, attention_mask=warmup_attn, use_cache=False)
    torch.cuda.synchronize()
    del warmup_ids, warmup_attn
    torch.cuda.empty_cache()

    # ---- Profile OPTIMIZED approach ----
    print("\n[Profile] Running OPTIMIZED training step...")
    restore_lora_state(model, initial_state)
    torch.cuda.empty_cache()
    gc.collect()

    opt_timings = profile_optimized(
        model, tokenizer, trainable_params, device,
        seqs, prompt_lens, action_lens, advantages, seq_lens,
        mini_batch_size=args.mini_batch_size,
        stats_chunk_size=args.stats_chunk_size_opt,
        max_grad_norm=args.max_grad_norm,
    )
    print_results("OPTIMIZED (sorted, pinned, chunk=8)", opt_timings)

    # ---- Profile ORIGINAL approach ----
    if not args.skip_original:
        print("\n[Profile] Running ORIGINAL training step...")
        restore_lora_state(model, initial_state)
        torch.cuda.empty_cache()
        gc.collect()

        orig_timings = profile_original(
            model, tokenizer, trainable_params, device,
            seqs, prompt_lens, action_lens, advantages, seq_lens,
            mini_batch_size=args.mini_batch_size,
            stats_chunk_size=args.stats_chunk_size_orig,
            max_grad_norm=args.max_grad_norm,
        )
        print_results("ORIGINAL (random, pad_to_device, chunk=4)", orig_timings)

        # ---- Comparison ----
        print(f"\n{'='*60}")
        print("  COMPARISON")
        print(f"{'='*60}")
        speedup_6 = orig_timings["step6_sec"] / max(opt_timings["step6_sec"], 0.01)
        speedup_7 = orig_timings["step7_sec"] / max(opt_timings["step7_sec"], 0.01)
        speedup_total = orig_timings["total_sec"] / max(opt_timings["total_sec"], 0.01)

        print(f"  Step 6 speedup:  {speedup_6:.2f}x  ({orig_timings['step6_sec']:.1f}s → {opt_timings['step6_sec']:.1f}s)")
        print(f"  Step 7 speedup:  {speedup_7:.2f}x  ({orig_timings['step7_sec']:.1f}s → {opt_timings['step7_sec']:.1f}s)")
        print(f"  TOTAL speedup:   {speedup_total:.2f}x  ({orig_timings['total_sec']:.1f}s → {opt_timings['total_sec']:.1f}s)")
        print(f"  Pad eff change:  {orig_timings['pad_eff']:.1%} → {opt_timings['pad_eff']:.1%}")
        print(f"{'='*60}")

        # Memory report
        if torch.cuda.is_available():
            print(f"\n  GPU memory (peak): {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
            print(f"  GPU memory (now):  {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    else:
        if torch.cuda.is_available():
            print(f"\n  GPU memory (peak): {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
            print(f"  GPU memory (now):  {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    print("\nDONE")


if __name__ == "__main__":
    main()
