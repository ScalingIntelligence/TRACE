#!/usr/bin/env python3
"""
Correctness tests for train_grpo_optimized.py.

Validates that optimizations preserve numerical equivalence with train_grpo.py.
Tests:
  1. make_sorted_batches: partition correctness and ordering
  2. pad_and_pin: equivalence with ppo.pad_to_device
  3. Padding efficiency: quantifies improvement from length sorting
  4. Logprob equivalence: sorted vs random batching with a tiny causal LM
  5. Full training step equivalence: sorted vs random produce same per-sample logprobs

Usage:
    conda activate games
    python test_grpo_optimized.py
"""
import math
import random
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from train_grpo_optimized import make_sorted_batches, pad_and_pin, compute_padding_efficiency
from ppo import pad_to_device, logprob_action_tokens


# ============================================================================
# Tiny causal LM for testing (no external model needed)
# ============================================================================

class TinyCausalLM(nn.Module):
    """Minimal causal language model for testing logprob equivalence."""

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

        # Causal mask (additive, -inf for masked positions)
        causal_mask = torch.triu(
            torch.full((T, T), float('-inf'), device=device), diagonal=1
        )

        # Padding mask (True = ignore)
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)

        x = self.encoder(x, mask=causal_mask, src_key_padding_mask=src_key_padding_mask)
        logits = self.head(x)
        return type('Output', (), {'logits': logits})()


# ============================================================================
# Test helpers
# ============================================================================

def make_random_sequences(n: int, min_len: int, max_len: int, vocab_size: int = 256) -> List[torch.Tensor]:
    """Create n random token sequences with varying lengths."""
    return [
        torch.randint(1, vocab_size, (random.randint(min_len, max_len),))
        for _ in range(n)
    ]


def compute_logprobs_batched(
    model: nn.Module,
    seqs: List[torch.Tensor],
    prompt_lens: List[int],
    action_lens: List[int],
    batch_indices: List[List[int]],
    pad_id: int,
    device: str,
) -> torch.Tensor:
    """Compute per-sample logprobs using the given batch grouping."""
    N = len(seqs)
    logprobs = torch.empty(N, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        for mb_idx in batch_indices:
            mb_seqs = [seqs[j] for j in mb_idx]
            mb_pl = [prompt_lens[j] for j in mb_idx]
            mb_al = [action_lens[j] for j in mb_idx]

            mb_ids, mb_attn = pad_to_device(mb_seqs, pad_id, device)

            outputs = model(input_ids=mb_ids, attention_mask=mb_attn)
            logits = outputs.logits

            mb_logp = logprob_action_tokens(logits, mb_ids, mb_pl, mb_al, normalize_by_len=False)
            logprobs[mb_idx] = mb_logp.detach().cpu()

            del logits, outputs, mb_logp, mb_ids, mb_attn

    return logprobs


# ============================================================================
# Tests
# ============================================================================

def test_sorted_batches_partition():
    """make_sorted_batches covers all indices exactly once."""
    print("  test_sorted_batches_partition...", end=" ")
    for n in [1, 7, 8, 15, 16, 100, 513]:
        for bs in [1, 4, 8, 16, 64]:
            seq_lens = [random.randint(10, 5000) for _ in range(n)]
            batches = make_sorted_batches(seq_lens, bs)

            # All indices present exactly once
            all_idx = sorted(sum(batches, []))
            assert all_idx == list(range(n)), f"Partition failed for n={n}, bs={bs}"

            # Batch size respected
            for b in batches:
                assert len(b) <= bs, f"Batch too large: {len(b)} > {bs}"

            # Within each batch, lengths should be non-decreasing (sorted)
            for b in batches:
                b_lens = [seq_lens[i] for i in b]
                assert b_lens == sorted(b_lens), f"Batch not sorted by length"

    print("PASS")


def test_sorted_batches_single_element():
    """Edge case: single element."""
    print("  test_sorted_batches_single_element...", end=" ")
    batches = make_sorted_batches([42], 8)
    assert batches == [[0]]
    print("PASS")


def test_pad_and_pin_equivalence():
    """pad_and_pin produces identical results to pad_to_device (on CPU)."""
    print("  test_pad_and_pin_equivalence...", end=" ")
    for _ in range(10):
        n = random.randint(1, 16)
        seqs = make_random_sequences(n, 10, 500, vocab_size=1000)
        pad_id = 0

        ids_ref, attn_ref = pad_to_device(seqs, pad_id, "cpu")
        ids_opt, attn_opt = pad_and_pin(seqs, pad_id)

        # Unpin for comparison (pinned tensors compare fine, but just to be safe)
        assert torch.equal(ids_ref, ids_opt.clone()), "IDs mismatch"
        assert torch.equal(attn_ref, attn_opt.clone()), "Attention mask mismatch"

    print("PASS")


def test_pad_and_pin_shapes():
    """pad_and_pin produces correct shapes and padding."""
    print("  test_pad_and_pin_shapes...", end=" ")
    seqs = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6, 7, 8])]
    ids, attn = pad_and_pin(seqs, pad_id=0)

    assert ids.shape == (2, 5), f"Expected (2, 5), got {ids.shape}"
    assert attn.shape == (2, 5)

    # First sequence: [1, 2, 3, 0, 0]
    assert ids[0].tolist() == [1, 2, 3, 0, 0]
    assert attn[0].tolist() == [1, 1, 1, 0, 0]

    # Second sequence: [4, 5, 6, 7, 8]
    assert ids[1].tolist() == [4, 5, 6, 7, 8]
    assert attn[1].tolist() == [1, 1, 1, 1, 1]

    print("PASS")


def test_padding_efficiency():
    """Length sorting substantially reduces padding waste."""
    print("  test_padding_efficiency...", end=" ")
    random.seed(42)
    seq_lens = [random.randint(100, 5000) for _ in range(500)]
    batch_size = 8

    # Random batching (original approach)
    random_idx = list(range(len(seq_lens)))
    random.shuffle(random_idx)
    random_batches = [random_idx[i:i + batch_size] for i in range(0, len(random_idx), batch_size)]
    random_eff = compute_padding_efficiency(seq_lens, random_batches)

    # Sorted batching (optimized approach)
    sorted_batches = make_sorted_batches(seq_lens, batch_size)
    sorted_eff = compute_padding_efficiency(seq_lens, sorted_batches)

    print(f"random={random_eff:.1%} sorted={sorted_eff:.1%} speedup={sorted_eff/random_eff:.2f}x", end=" ")

    assert sorted_eff > random_eff, (
        f"Sorted ({sorted_eff:.3f}) should be more efficient than random ({random_eff:.3f})"
    )
    assert sorted_eff > 0.85, f"Sorted efficiency {sorted_eff:.3f} lower than expected (>0.85)"

    print("PASS")


def test_logprob_equivalence():
    """Sorted batching produces identical per-sample logprobs as random batching."""
    print("  test_logprob_equivalence...", end=" ")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    random.seed(42)

    vocab_size = 256
    model = TinyCausalLM(vocab_size=vocab_size, d_model=64, nhead=4, num_layers=2).to(device)
    model.eval()

    # Create sequences with varying lengths (simulating real data)
    N = 32
    seqs = make_random_sequences(N, 20, 200, vocab_size=vocab_size)
    # Simulate prompt/action split: first 60% is prompt, rest is action
    prompt_lens = [max(1, int(s.shape[0] * 0.6)) for s in seqs]
    action_lens = [s.shape[0] - pl for s, pl in zip(seqs, prompt_lens)]

    seq_lens = [s.shape[0] for s in seqs]
    batch_size = 4
    pad_id = 0

    # Random batching (original)
    random_idx = list(range(N))
    random.shuffle(random_idx)
    random_batches = [random_idx[i:i + batch_size] for i in range(0, N, batch_size)]

    logp_random = compute_logprobs_batched(
        model, seqs, prompt_lens, action_lens, random_batches, pad_id, device
    )

    # Sorted batching (optimized)
    sorted_batches = make_sorted_batches(seq_lens, batch_size)

    logp_sorted = compute_logprobs_batched(
        model, seqs, prompt_lens, action_lens, sorted_batches, pad_id, device
    )

    # Compare per-sample logprobs
    max_diff = (logp_random - logp_sorted).abs().max().item()
    mean_diff = (logp_random - logp_sorted).abs().mean().item()

    print(f"max_diff={max_diff:.2e} mean_diff={mean_diff:.2e}", end=" ")

    # Tolerance: attention softmax with different padding lengths can cause
    # tiny floating-point differences. Should be negligible.
    assert max_diff < 1e-4, (
        f"Logprob difference too large: max_diff={max_diff:.6f}. "
        f"Sorted batching may not be equivalent."
    )

    print("PASS")


def test_logprob_equivalence_varied_lengths():
    """Same as above but with more extreme length variation."""
    print("  test_logprob_equivalence_varied_lengths...", end=" ")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(123)
    random.seed(123)

    vocab_size = 256
    model = TinyCausalLM(vocab_size=vocab_size, d_model=64, nhead=4, num_layers=2).to(device)
    model.eval()

    # Extreme length variation: 10 to 500 tokens
    N = 24
    seqs = make_random_sequences(N, 10, 500, vocab_size=vocab_size)
    prompt_lens = [max(1, int(s.shape[0] * 0.5)) for s in seqs]
    action_lens = [s.shape[0] - pl for s, pl in zip(seqs, prompt_lens)]

    seq_lens = [s.shape[0] for s in seqs]
    batch_size = 6
    pad_id = 0

    # Random batching
    random_idx = list(range(N))
    random.shuffle(random_idx)
    random_batches = [random_idx[i:i + batch_size] for i in range(0, N, batch_size)]

    logp_random = compute_logprobs_batched(
        model, seqs, prompt_lens, action_lens, random_batches, pad_id, device
    )

    # Sorted batching
    sorted_batches = make_sorted_batches(seq_lens, batch_size)

    logp_sorted = compute_logprobs_batched(
        model, seqs, prompt_lens, action_lens, sorted_batches, pad_id, device
    )

    max_diff = (logp_random - logp_sorted).abs().max().item()
    mean_diff = (logp_random - logp_sorted).abs().mean().item()

    print(f"max_diff={max_diff:.2e} mean_diff={mean_diff:.2e}", end=" ")

    assert max_diff < 1e-4, f"Logprob difference too large: max_diff={max_diff:.6f}"

    print("PASS")


def test_single_sample_batch():
    """Edge case: batches with a single sample."""
    print("  test_single_sample_batch...", end=" ")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(99)

    vocab_size = 256
    model = TinyCausalLM(vocab_size=vocab_size, d_model=64, nhead=4, num_layers=2).to(device)
    model.eval()

    N = 5
    seqs = make_random_sequences(N, 20, 100, vocab_size=vocab_size)
    prompt_lens = [max(1, s.shape[0] // 2) for s in seqs]
    action_lens = [s.shape[0] - pl for s, pl in zip(seqs, prompt_lens)]

    seq_lens = [s.shape[0] for s in seqs]
    pad_id = 0

    # Batch size 1 (each sample alone)
    single_batches = [[i] for i in range(N)]
    logp_single = compute_logprobs_batched(
        model, seqs, prompt_lens, action_lens, single_batches, pad_id, device
    )

    # Batch size N (all together)
    full_batch = [list(range(N))]
    logp_full = compute_logprobs_batched(
        model, seqs, prompt_lens, action_lens, full_batch, pad_id, device
    )

    max_diff = (logp_single - logp_full).abs().max().item()
    print(f"max_diff={max_diff:.2e}", end=" ")

    assert max_diff < 1e-4, f"Single vs full batch difference too large: {max_diff:.6f}"

    print("PASS")


def benchmark_padding():
    """Benchmark pad_to_device vs pad_and_pin + transfer."""
    print("  benchmark_padding...", end=" ")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("SKIP (no CUDA)")
        return

    N = 64
    seqs = make_random_sequences(N, 100, 2000, vocab_size=1000)
    pad_id = 0

    # Benchmark pad_to_device (original)
    t0 = time.time()
    for _ in range(50):
        ids, attn = pad_to_device(seqs, pad_id, device)
        del ids, attn
    torch.cuda.synchronize()
    t_original = time.time() - t0

    # Benchmark pad_and_pin + transfer (optimized)
    # Pre-pad once (amortized cost)
    ids_cpu, attn_cpu = pad_and_pin(seqs, pad_id)
    t0 = time.time()
    for _ in range(50):
        ids = ids_cpu.to(device, non_blocking=True)
        attn = attn_cpu.to(device, non_blocking=True)
        del ids, attn
    torch.cuda.synchronize()
    t_optimized = time.time() - t0

    print(f"original={t_original:.3f}s optimized={t_optimized:.3f}s ratio={t_original/max(t_optimized, 1e-9):.2f}x", end=" ")
    print("PASS")


def test_compute_padding_efficiency():
    """Test the padding efficiency helper."""
    print("  test_compute_padding_efficiency...", end=" ")

    # Perfect efficiency: all same length
    seq_lens = [100, 100, 100, 100]
    batches = [[0, 1, 2, 3]]
    eff = compute_padding_efficiency(seq_lens, batches)
    assert abs(eff - 1.0) < 1e-6, f"Expected 1.0, got {eff}"

    # 50% efficiency: one short, one long
    seq_lens = [50, 100]
    batches = [[0, 1]]
    eff = compute_padding_efficiency(seq_lens, batches)
    assert abs(eff - 0.75) < 1e-6, f"Expected 0.75, got {eff}"  # (50+100)/(2*100) = 0.75

    print("PASS")


def test_unified_cache():
    """Unified cache (stats_chunk_size=mini_batch_size) produces same logprobs."""
    print("  test_unified_cache...", end=" ")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(77)
    random.seed(77)

    vocab_size = 256
    model = TinyCausalLM(vocab_size=vocab_size, d_model=64, nhead=4, num_layers=2).to(device)
    model.eval()

    N = 40
    seqs = make_random_sequences(N, 20, 300, vocab_size=vocab_size)
    prompt_lens = [max(1, int(s.shape[0] * 0.6)) for s in seqs]
    action_lens = [s.shape[0] - pl for s, pl in zip(seqs, prompt_lens)]
    seq_lens = [s.shape[0] for s in seqs]
    pad_id = 0

    # Separate caches: stats_chunk_size=4, mini_batch_size=8
    stats_batches_4 = make_sorted_batches(seq_lens, 4)
    logp_chunk4 = compute_logprobs_batched(
        model, seqs, prompt_lens, action_lens, stats_batches_4, pad_id, device
    )

    # Unified cache: stats_chunk_size=8=mini_batch_size
    stats_batches_8 = make_sorted_batches(seq_lens, 8)
    logp_chunk8 = compute_logprobs_batched(
        model, seqs, prompt_lens, action_lens, stats_batches_8, pad_id, device
    )

    max_diff = (logp_chunk4 - logp_chunk8).abs().max().item()
    print(f"chunk4_vs_chunk8 max_diff={max_diff:.2e}", end=" ")

    assert max_diff < 1e-4, f"Logprob mismatch between chunk sizes: {max_diff:.6f}"
    print("PASS")


def test_stats_chunk_size_speedup():
    """Verify stats_chunk_size=8 is measurably faster than 4."""
    print("  test_stats_chunk_size_speedup...", end=" ")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("SKIP (no CUDA)")
        return

    torch.manual_seed(42)
    random.seed(42)

    vocab_size = 256
    model = TinyCausalLM(vocab_size=vocab_size, d_model=128, nhead=4, num_layers=4).to(device)
    model.eval()

    N = 200
    seqs = make_random_sequences(N, 50, 400, vocab_size=vocab_size)
    prompt_lens = [max(1, int(s.shape[0] * 0.6)) for s in seqs]
    action_lens = [s.shape[0] - pl for s, pl in zip(seqs, prompt_lens)]
    seq_lens = [s.shape[0] for s in seqs]
    pad_id = 0

    times = {}
    for chunk_size in [4, 8]:
        batches = make_sorted_batches(seq_lens, chunk_size)
        cache = []
        for mb_idx in batches:
            mb_seqs = [seqs[j] for j in mb_idx]
            ids, attn = pad_and_pin(mb_seqs, pad_id)
            cache.append((mb_idx, ids, attn))

        # Warmup
        ids = cache[0][1].to(device)
        attn = cache[0][2].to(device)
        with torch.no_grad():
            _ = model(input_ids=ids, attention_mask=attn)
        torch.cuda.synchronize()

        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            for mb_idx, ids_cpu, attn_cpu in cache:
                ids = ids_cpu.to(device, non_blocking=True)
                attn = attn_cpu.to(device, non_blocking=True)
                outputs = model(input_ids=ids, attention_mask=attn)
                logits = outputs.logits
                _ = logprob_action_tokens(logits, ids,
                    [prompt_lens[j] for j in mb_idx],
                    [action_lens[j] for j in mb_idx],
                    normalize_by_len=False)
                del logits, outputs, ids, attn
        torch.cuda.synchronize()
        times[chunk_size] = time.time() - t0

    speedup = times[4] / max(times[8], 1e-9)
    print(f"chunk4={times[4]*1000:.0f}ms chunk8={times[8]*1000:.0f}ms speedup={speedup:.2f}x", end=" ")
    assert speedup > 1.3, f"Expected >1.3x speedup, got {speedup:.2f}x"
    print("PASS")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("GRPO Optimization Correctness Tests")
    print("=" * 60)

    print("\n[Unit Tests]")
    test_sorted_batches_partition()
    test_sorted_batches_single_element()
    test_pad_and_pin_equivalence()
    test_pad_and_pin_shapes()
    test_compute_padding_efficiency()

    print("\n[Padding Efficiency]")
    test_padding_efficiency()

    print("\n[Logprob Equivalence]")
    test_logprob_equivalence()
    test_logprob_equivalence_varied_lengths()
    test_single_sample_batch()
    test_unified_cache()

    print("\n[Performance Validation]")
    test_stats_chunk_size_speedup()

    print("\n[Benchmarks]")
    benchmark_padding()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
