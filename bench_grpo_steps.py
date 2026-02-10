#!/usr/bin/env python3
"""
Micro-benchmark for GRPO training steps 5-7.
Profiles each sub-step to identify remaining bottlenecks.
Uses TinyCausalLM to isolate overhead from model compute.

Usage:
    conda activate games && CUDA_VISIBLE_DEVICES=0 python bench_grpo_steps.py
"""
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from ppo import logprob_action_tokens
from train_grpo_optimized import make_sorted_batches, pad_and_pin


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size=256, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(4096, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            batch_first=True, dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        B, T = input_ids.shape
        device = input_ids.device
        pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        x = self.embed(input_ids) + self.pos(pos_ids)
        causal_mask = torch.triu(torch.full((T, T), float('-inf'), device=device), diagonal=1)
        src_key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        x = self.encoder(x, mask=causal_mask, src_key_padding_mask=src_key_padding_mask)
        logits = self.head(x)
        return type('Out', (), {'logits': logits})()


def bench_item_calls(device):
    """Benchmark .item() CUDA sync overhead."""
    print("\n--- .item() sync overhead ---")
    n_calls = 200
    t = torch.tensor(1.0, device=device)

    # Warm up
    for _ in range(10):
        _ = float(t.item())

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_calls):
        _ = float(t.item())
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    print(f"  {n_calls} .item() calls: {elapsed*1000:.1f}ms ({elapsed/n_calls*1000:.3f}ms each)")

    # Compare: accumulate on GPU, single .item() at end
    accum = torch.tensor(0.0, device=device)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_calls):
        accum = accum + t.detach()
    val = float(accum.item())
    torch.cuda.synchronize()
    elapsed2 = time.time() - t0
    print(f"  GPU accum + 1 .item(): {elapsed2*1000:.1f}ms (speedup: {elapsed/max(elapsed2, 1e-9):.1f}x)")


def bench_stats_chunk_size(model, device, N=200, vocab_size=256):
    """Benchmark step 6 with different stats_chunk_sizes."""
    print(f"\n--- Step 6: stats_chunk_size comparison (N={N}) ---")
    torch.manual_seed(42)
    random.seed(42)

    seqs = [torch.randint(1, vocab_size, (random.randint(50, 400),)) for _ in range(N)]
    prompt_lens = [max(1, int(s.shape[0] * 0.6)) for s in seqs]
    action_lens = [s.shape[0] - pl for s, pl in zip(seqs, prompt_lens)]
    seq_lens = [s.shape[0] for s in seqs]

    for chunk_size in [4, 8, 16]:
        batches = make_sorted_batches(seq_lens, chunk_size)

        # Pre-pad
        batch_cache = []
        for mb_idx in batches:
            mb_seqs = [seqs[j] for j in mb_idx]
            mb_ids, mb_attn = pad_and_pin(mb_seqs, 0)
            batch_cache.append((mb_idx, mb_ids, mb_attn))

        # Warm up
        model.eval()
        with torch.no_grad():
            mb_ids = batch_cache[0][1].to(device)
            mb_attn = batch_cache[0][2].to(device)
            _ = model(input_ids=mb_ids, attention_mask=mb_attn)
        torch.cuda.synchronize()

        old_logp = torch.empty(N, dtype=torch.float32)

        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            for mb_idx, mb_ids_cpu, mb_attn_cpu in batch_cache:
                mb_pl = [prompt_lens[j] for j in mb_idx]
                mb_al = [action_lens[j] for j in mb_idx]
                mb_ids = mb_ids_cpu.to(device, non_blocking=True)
                mb_attn = mb_attn_cpu.to(device, non_blocking=True)
                outputs = model(input_ids=mb_ids, attention_mask=mb_attn)
                logits = outputs.logits
                mb_logp = logprob_action_tokens(logits, mb_ids, mb_pl, mb_al, normalize_by_len=False)
                old_logp[mb_idx] = mb_logp.cpu()
                del logits, outputs, mb_logp, mb_ids, mb_attn
        torch.cuda.synchronize()
        elapsed = time.time() - t0

        n_batches = len(batches)
        print(f"  chunk_size={chunk_size:2d}: {n_batches:3d} batches, {elapsed*1000:.0f}ms ({elapsed/n_batches*1000:.1f}ms/batch)")


def bench_tracking_overhead(model, device, N=200, vocab_size=256):
    """Benchmark tracking metric computation: per-step .item() vs GPU accumulation."""
    print(f"\n--- Step 7: tracking overhead (N={N}) ---")
    torch.manual_seed(42)
    random.seed(42)

    seqs = [torch.randint(1, vocab_size, (random.randint(50, 400),)) for _ in range(N)]
    prompt_lens = [max(1, int(s.shape[0] * 0.6)) for s in seqs]
    action_lens = [s.shape[0] - pl for s, pl in zip(seqs, prompt_lens)]
    seq_lens = [s.shape[0] for s in seqs]
    advantages = torch.randn(N)

    mb_size = 8
    batches = make_sorted_batches(seq_lens, mb_size)

    batch_cache = []
    for mb_idx in batches:
        mb_seqs = [seqs[j] for j in mb_idx]
        mb_ids, mb_attn = pad_and_pin(mb_seqs, 0)
        batch_cache.append((mb_idx, mb_ids, mb_attn))

    old_logp = torch.randn(N)

    # Approach 1: per-step .item() (original)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    torch.cuda.synchronize()
    t0 = time.time()

    policy_loss_acc = 0.0
    ratio_mean_acc = 0.0
    updates = 0

    for mb_idx, mb_ids_cpu, mb_attn_cpu in batch_cache:
        mb_pl = [prompt_lens[j] for j in mb_idx]
        mb_al = [action_lens[j] for j in mb_idx]
        mb_ids = mb_ids_cpu.to(device, non_blocking=True)
        mb_attn = mb_attn_cpu.to(device, non_blocking=True)
        mb_old_logp = old_logp[mb_idx].to(device)
        mb_adv = advantages[mb_idx].to(device)

        outputs = model(input_ids=mb_ids, attention_mask=mb_attn)
        logits = outputs.logits
        new_logp = logprob_action_tokens(logits, mb_ids, mb_pl, mb_al, normalize_by_len=False)
        ratio = torch.exp(new_logp - mb_old_logp)
        policy_loss = -torch.mean(ratio * mb_adv)

        optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        optimizer.step()

        # Per-step .item() tracking
        with torch.no_grad():
            ratio_mean_val = torch.mean(ratio)
        policy_loss_acc += float(policy_loss.item())
        ratio_mean_acc += float(ratio_mean_val.item())
        updates += 1

        del mb_ids, mb_attn, logits, outputs, new_logp, ratio, policy_loss

    torch.cuda.synchronize()
    t_item = time.time() - t0

    # Approach 2: GPU accumulation (optimized)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    torch.cuda.synchronize()
    t0 = time.time()

    policy_loss_sum = torch.tensor(0.0, device=device)
    ratio_mean_sum = torch.tensor(0.0, device=device)
    updates2 = 0

    for mb_idx, mb_ids_cpu, mb_attn_cpu in batch_cache:
        mb_pl = [prompt_lens[j] for j in mb_idx]
        mb_al = [action_lens[j] for j in mb_idx]
        mb_ids = mb_ids_cpu.to(device, non_blocking=True)
        mb_attn = mb_attn_cpu.to(device, non_blocking=True)
        mb_old_logp = old_logp[mb_idx].to(device)
        mb_adv = advantages[mb_idx].to(device)

        outputs = model(input_ids=mb_ids, attention_mask=mb_attn)
        logits = outputs.logits
        new_logp = logprob_action_tokens(logits, mb_ids, mb_pl, mb_al, normalize_by_len=False)
        ratio = torch.exp(new_logp - mb_old_logp)
        policy_loss = -torch.mean(ratio * mb_adv)

        optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        optimizer.step()

        # GPU accumulation tracking
        with torch.no_grad():
            policy_loss_sum += policy_loss.detach()
            ratio_mean_sum += ratio.mean()
        updates2 += 1

        del mb_ids, mb_attn, logits, outputs, new_logp, ratio, policy_loss

    # Single sync at end
    final_loss = float(policy_loss_sum.item()) / updates2
    final_ratio = float(ratio_mean_sum.item()) / updates2
    torch.cuda.synchronize()
    t_accum = time.time() - t0

    print(f"  per-step .item():    {t_item*1000:.0f}ms ({updates} batches)")
    print(f"  GPU accumulation:    {t_accum*1000:.0f}ms ({updates2} batches)")
    print(f"  speedup: {t_item/max(t_accum, 1e-9):.2f}x")


def bench_unified_vs_separate_cache(N=500, vocab_size=256):
    """Benchmark single vs separate batch caching."""
    print(f"\n--- Batch caching: unified vs separate (N={N}) ---")
    random.seed(42)

    seqs = [torch.randint(1, vocab_size, (random.randint(100, 2000),)) for _ in range(N)]
    seq_lens = [s.shape[0] for s in seqs]

    # Separate caching (stats_chunk_size=4, mini_batch_size=8)
    t0 = time.time()
    stats_batches = make_sorted_batches(seq_lens, 4)
    train_batches = make_sorted_batches(seq_lens, 8)
    stats_cache = []
    for mb_idx in stats_batches:
        mb_seqs = [seqs[j] for j in mb_idx]
        ids, attn = pad_and_pin(mb_seqs, 0)
        stats_cache.append((mb_idx, ids, attn))
    train_cache = []
    for mb_idx in train_batches:
        mb_seqs = [seqs[j] for j in mb_idx]
        ids, attn = pad_and_pin(mb_seqs, 0)
        train_cache.append((mb_idx, ids, attn))
    t_separate = time.time() - t0

    total_mem_separate = sum(c[1].nelement() + c[2].nelement() for c in stats_cache)
    total_mem_separate += sum(c[1].nelement() + c[2].nelement() for c in train_cache)

    del stats_cache, train_cache

    # Unified caching (both use batch_size=8)
    t0 = time.time()
    unified_batches = make_sorted_batches(seq_lens, 8)
    unified_cache = []
    for mb_idx in unified_batches:
        mb_seqs = [seqs[j] for j in mb_idx]
        ids, attn = pad_and_pin(mb_seqs, 0)
        unified_cache.append((mb_idx, ids, attn))
    t_unified = time.time() - t0

    total_mem_unified = sum(c[1].nelement() + c[2].nelement() for c in unified_cache)

    del unified_cache

    print(f"  separate (4+8): {t_separate*1000:.0f}ms, {len(stats_batches)+len(train_batches)} batches, {total_mem_separate*8/1e6:.1f}MB")
    print(f"  unified (8):    {t_unified*1000:.0f}ms, {len(unified_batches)} batches, {total_mem_unified*8/1e6:.1f}MB")
    print(f"  time speedup:   {t_separate/max(t_unified, 1e-9):.2f}x")
    print(f"  memory savings:  {(1 - total_mem_unified/total_mem_separate)*100:.0f}%")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    torch.manual_seed(42)
    model = TinyCausalLM(vocab_size=256, d_model=128, nhead=4, num_layers=4).to(device)

    bench_item_calls(device)
    bench_stats_chunk_size(model, device, N=200)
    bench_tracking_overhead(model, device, N=200)
    bench_unified_vs_separate_cache(N=500)

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
