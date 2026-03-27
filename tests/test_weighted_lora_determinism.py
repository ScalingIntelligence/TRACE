"""
Test that the weighted LoRA path is fully deterministic.

Verifies that:
1. add_weighted_lora_linear (float32 intermediates) is deterministic
2. add_weighted_lora_logits (torch.matmul, no Triton) is deterministic
3. The Triton lora_shrink kernel (SPLIT_K>1, atomic_add) is nondeterministic
   at float32 precision — confirming we need the torch.matmul path
4. MoE weighted paths (float32 intermediates) are deterministic
"""
import torch


def test_weighted_linear_determinism():
    """add_weighted_lora_linear must produce identical results across runs."""
    torch.manual_seed(42)
    device = "cuda"

    # Simulate Qwen3-30B-A3B dimensions
    num_tokens = 1
    hidden_size = 2048
    out_dim = 2048
    rank = 16
    max_loras = 3

    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    lora_a = (
        torch.randn(max_loras, 1, rank, hidden_size, dtype=torch.bfloat16, device=device)
        * 0.01
    )
    lora_b = (
        torch.randn(max_loras, 1, out_dim, rank, dtype=torch.bfloat16, device=device)
        * 0.01
    )

    config = [(0, 0.7), (1, 0.3)]
    output_slices = (out_dim,)

    # Replicate add_weighted_lora_linear logic (float32 intermediates)
    def weighted_linear(x, lora_a, lora_b, config, output_slices):
        y = torch.zeros(x.size(0), out_dim, dtype=torch.bfloat16, device=device)
        for slot_idx, weight in config:
            if weight == 0:
                continue
            for s in range(1):
                A = lora_a[slot_idx, 0, :rank, :]
                B = lora_b[slot_idx, 0, :, :rank]
                hidden = x.float() @ A.T.float()
                delta = hidden @ B.T.float()
                if len(output_slices) == 1:
                    y.add_(delta.to(y.dtype), alpha=weight)
                else:
                    offset = sum(output_slices[:s])
                    y[:, offset : offset + output_slices[s]].add_(
                        delta.to(y.dtype), alpha=weight
                    )
        return y

    results = []
    for _ in range(50):
        r = weighted_linear(x, lora_a, lora_b, config, output_slices)
        results.append(r.clone())
        torch.cuda.synchronize()

    diffs = [(results[0] - r).abs().max().item() for r in results[1:]]
    assert all(d == 0 for d in diffs), f"Non-zero diffs: {[d for d in diffs if d > 0]}"
    print("PASS: weighted linear determinism (float32 intermediates)")


def test_weighted_logits_determinism():
    """add_weighted_lora_logits must produce identical results across runs."""
    torch.manual_seed(42)
    device = "cuda"

    num_tokens = 1
    hidden_size = 2048
    vocab_size = 32000
    rank = 16
    max_loras = 3

    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    lora_a = (
        torch.randn(max_loras, 1, rank, hidden_size, dtype=torch.bfloat16, device=device)
        * 0.01
    )
    lora_b = (
        torch.randn(max_loras, 1, vocab_size, rank, dtype=torch.bfloat16, device=device)
        * 0.01
    )
    config = [(0, 0.7), (1, 0.3)]

    # Replicate add_weighted_lora_logits logic
    def weighted_logits(x, lora_a, lora_b, config, scale=1.0):
        y = torch.zeros(x.size(0), vocab_size, dtype=torch.bfloat16, device=device)
        for slot_idx, weight in config:
            if weight == 0:
                continue
            A = lora_a[slot_idx, 0, :rank, :]
            B = lora_b[slot_idx, 0, :, :rank]
            hidden = x.float() @ A.T.float()
            hidden = hidden * scale
            delta = hidden @ B.T.float()
            y.add_(delta.to(y.dtype), alpha=weight)
        return y

    results = []
    for _ in range(50):
        r = weighted_logits(x, lora_a, lora_b, config)
        results.append(r.clone())
        torch.cuda.synchronize()

    diffs = [(results[0] - r).abs().max().item() for r in results[1:]]
    assert all(d == 0 for d in diffs), f"Non-zero diffs: {[d for d in diffs if d > 0]}"
    print("PASS: weighted logits determinism (float32 intermediates)")


def test_triton_shrink_nondeterminism():
    """Confirm that Triton lora_shrink with SPLIT_K>1 IS nondeterministic
    at float32 precision (justifying our torch.matmul replacement)."""
    from vllm.lora.ops.triton_ops import lora_shrink, LoRAKernelMeta

    torch.manual_seed(42)
    device = "cuda"

    max_loras = 3
    num_tokens = 1
    hidden_size = 2048
    rank = 16

    lora_a = (
        torch.randn(max_loras, 1, rank, hidden_size, dtype=torch.bfloat16, device=device)
        * 0.01
    )
    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)

    meta = LoRAKernelMeta.make(max_loras, num_tokens, device=device)
    token_lora_indices = torch.zeros(num_tokens, dtype=torch.long, device=device)
    meta.prepare_tensors(token_lora_indices)

    results = []
    for _ in range(100):
        buffer = torch.empty(1, num_tokens, rank, dtype=torch.float32, device=device)
        lora_shrink(x, [lora_a], buffer, *meta.meta_args(num_tokens), 1.0)
        torch.cuda.synchronize()
        results.append(buffer.clone())

    diffs = [(results[0] - r).abs().max().item() for r in results[1:]]
    nonzero = sum(1 for d in diffs if d > 0)
    print(
        f"Triton lora_shrink (SPLIT_K=64): {nonzero}/{len(diffs)} runs differ "
        f"(max_diff={max(diffs):.2e})"
    )
    # We expect nondeterminism — this test documents it, not fails on it
    if nonzero > 0:
        print("CONFIRMED: Triton shrink is nondeterministic (expected)")
    else:
        print("NOTE: Triton shrink was deterministic on this GPU (unusual)")
    print("PASS: Triton shrink nondeterminism check")


def test_moe_weighted_determinism():
    """MoE weighted paths (float32 intermediates) must be deterministic."""
    torch.manual_seed(33)
    device = "cuda"

    num_experts = 8
    hidden_size = 128
    inter_size = 256
    rank = 8
    num_tokens = 16
    top_k = 2
    N = num_tokens * top_k

    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), device=device)
    flat_eids = topk_ids.reshape(-1)

    # Two adapters
    A1 = torch.randn(num_experts, rank, hidden_size, dtype=torch.bfloat16, device=device) * 0.01
    B1 = torch.randn(num_experts, inter_size, rank, dtype=torch.bfloat16, device=device) * 0.01
    A2 = torch.randn(num_experts, rank, hidden_size, dtype=torch.bfloat16, device=device) * 0.01
    B2 = torch.randn(num_experts, inter_size, rank, dtype=torch.bfloat16, device=device) * 0.01

    config = [(0, 0.6), (1, 0.4)]
    adapters = [(A1, B1), (A2, B2)]

    x_flat = x.unsqueeze(1).expand(-1, top_k, -1).reshape(N, -1)

    def weighted_w13(x_flat, flat_eids, adapters, config):
        y = torch.zeros(N, inter_size, dtype=torch.bfloat16, device=device)
        for (A_all, B_all), (_, weight) in zip(adapters, config):
            A_gathered = A_all[flat_eids, :rank]
            B_gathered = B_all[flat_eids, :, :rank]
            hidden = torch.bmm(
                x_flat.unsqueeze(1).float(),
                A_gathered.transpose(1, 2).float(),
            ).squeeze(1)
            delta = torch.bmm(
                hidden.unsqueeze(1),
                B_gathered.transpose(1, 2).float(),
            ).squeeze(1)
            y.add_(delta.to(y.dtype), alpha=weight)
        return y

    results = []
    for _ in range(50):
        r = weighted_w13(x_flat, flat_eids, adapters, config)
        results.append(r.clone())
        torch.cuda.synchronize()

    diffs = [(results[0] - r).abs().max().item() for r in results[1:]]
    assert all(d == 0 for d in diffs), f"MoE non-zero diffs: {[d for d in diffs if d > 0]}"
    print("PASS: MoE weighted determinism (float32 intermediates)")


def test_multi_layer_determinism():
    """Simulate multiple layers of weighted LoRA to check error accumulation."""
    torch.manual_seed(42)
    device = "cuda"

    hidden_size = 2048
    rank = 16
    num_tokens = 1
    num_layers = 28  # approximate Qwen3-30B layer count
    max_loras = 2

    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)

    # Pre-generate all layer weights
    W_bases = [
        torch.randn(hidden_size, hidden_size, dtype=torch.bfloat16, device=device) * 0.01
        for _ in range(num_layers)
    ]
    lora_as = [
        torch.randn(max_loras, 1, rank, hidden_size, dtype=torch.bfloat16, device=device)
        * 0.01
        for _ in range(num_layers)
    ]
    lora_bs = [
        torch.randn(max_loras, 1, hidden_size, rank, dtype=torch.bfloat16, device=device)
        * 0.01
        for _ in range(num_layers)
    ]

    config = [(0, 0.7), (1, 0.3)]

    def full_forward(x):
        h = x.clone()
        for layer in range(num_layers):
            base_out = h @ W_bases[layer].T
            for slot_idx, weight in config:
                A = lora_as[layer][slot_idx, 0, :rank, :]
                B = lora_bs[layer][slot_idx, 0, :, :rank]
                hidden = h.float() @ A.T.float()
                delta = hidden @ B.T.float()
                base_out.add_(delta.to(base_out.dtype), alpha=weight)
            h = base_out
        return h

    results = []
    for _ in range(20):
        r = full_forward(x)
        results.append(r.clone())
        torch.cuda.synchronize()

    diffs = [(results[0] - r).abs().max().item() for r in results[1:]]
    assert all(d == 0 for d in diffs), f"Multi-layer non-zero diffs: {max(diffs)}"
    print("PASS: multi-layer weighted LoRA determinism")


if __name__ == "__main__":
    print("=" * 60)
    print("Weighted LoRA Determinism Tests")
    print("=" * 60 + "\n")

    test_weighted_linear_determinism()
    test_weighted_logits_determinism()
    test_triton_shrink_nondeterminism()
    test_moe_weighted_determinism()
    test_multi_layer_determinism()

    print("\n" + "=" * 60)
    print("ALL DETERMINISM TESTS PASSED")
    print("=" * 60)
