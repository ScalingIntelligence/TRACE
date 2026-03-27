"""
Test that the chunked weighted LoRA accumulation produces results identical
to the concat-merge approach and the PEFT merge_and_unload delta approach.

Three methods tested:
  1. PEFT-style delta merge (merge_and_unload approach from merge_and_push_RL.py)
     output = base @ x + sum_i (w_i / sum_w) * (alpha_i/r_i) * B_i @ A_i @ x

  2. Concat merge (old vllm approach)
     A_cat = cat([A_1, ..., A_K], dim=0)         # [K*r, in]
     B_cat = cat([w_1*B_1, ..., w_K*B_K], dim=1) # [out, K*r]
     output = base @ x + B_cat @ A_cat @ x

  3. Chunked accumulation (new vllm weighted LoRA)
     output = base @ x + sum_i w_i * B_i @ A_i @ x

Methods 2 and 3 should be EXACTLY equal (both use raw weights, no normalization).
Method 1 normalizes by total_weight, so it differs unless total_weight == 1.

All three should match when weights are normalized to sum to 1.
"""
import torch
import torch.nn.functional as F


def peft_delta_merge(x, W_base, adapters, weights, normalize=True):
    """Simulate merge_and_push_RL.py: weighted average of deltas.

    Each adapter's delta = (alpha/r) * B @ A  (scaling already in B).
    With normalize=True, weights are divided by sum(weights).
    """
    output = x @ W_base.T  # base output
    total_w = sum(weights) if normalize else 1.0
    for (A, B), w in zip(adapters, weights):
        # B already has scaling baked in (like vLLM's optimize())
        delta = (w / total_w) * (x @ A.T @ B.T)
        output = output + delta
    return output


def concat_merge(x, W_base, adapters, weights):
    """Old vLLM approach: concatenate into one big adapter.

    A_cat = cat([A_1, ..., A_K], dim=0)       # [K*r, in_dim]
    B_cat = cat([w1*B1, ..., wK*BK], dim=1)   # [out_dim, K*r]
    output = base @ x + B_cat @ A_cat @ x
    """
    A_blocks = [A for A, B in adapters]
    B_blocks = [w * B for (A, B), w in zip(adapters, weights)]
    A_cat = torch.cat(A_blocks, dim=0)   # [K*r, in_dim]
    B_cat = torch.cat(B_blocks, dim=1)   # [out_dim, K*r]

    output = x @ W_base.T
    lora_out = x @ A_cat.T @ B_cat.T  # [tokens, out_dim]
    output = output + lora_out
    return output


def chunked_accumulation(x, W_base, adapters, weights):
    """New vLLM weighted LoRA: loop over adapters, accumulate.

    output = base @ x + sum_i w_i * B_i @ A_i @ x
    """
    output = x @ W_base.T
    for (A, B), w in zip(adapters, weights):
        hidden = x @ A.T       # [tokens, r]
        delta = hidden @ B.T   # [tokens, out_dim]
        output = output + w * delta
    return output


def chunked_accumulation_inplace(x, W_base, adapters, weights):
    """Exactly mirrors add_weighted_lora_linear: in-place add_."""
    output = x @ W_base.T
    for (A, B), w in zip(adapters, weights):
        hidden = x @ A.T
        delta = hidden @ B.T
        output.add_(delta, alpha=w)
    return output


def test_equivalence_basic():
    """Test all methods with simple 2-adapter case."""
    torch.manual_seed(42)
    in_dim, out_dim, rank = 128, 256, 8
    num_tokens = 16

    x = torch.randn(num_tokens, in_dim, dtype=torch.float32)
    W_base = torch.randn(out_dim, in_dim, dtype=torch.float32)

    # Create 2 adapters with scaling baked into B (like vLLM optimize())
    adapters = []
    for i in range(2):
        A = torch.randn(rank, in_dim, dtype=torch.float32) * 0.01
        B_raw = torch.randn(out_dim, rank, dtype=torch.float32) * 0.01
        alpha, r = 16, rank
        B_scaled = B_raw * (alpha / r)  # bake scaling into B
        adapters.append((A, B_scaled))

    weights = [0.7, 0.3]

    out_concat = concat_merge(x, W_base, adapters, weights)
    out_chunked = chunked_accumulation(x, W_base, adapters, weights)
    out_chunked_ip = chunked_accumulation_inplace(x, W_base, adapters, weights)

    # Concat and chunked must be EXACTLY equal (same math, different order)
    diff_concat_chunked = (out_concat - out_chunked).abs().max().item()
    diff_chunked_ip = (out_chunked - out_chunked_ip).abs().max().item()

    print(f"concat vs chunked:       max_diff = {diff_concat_chunked:.2e}")
    print(f"chunked vs chunked_ip:   max_diff = {diff_chunked_ip:.2e}")

    assert diff_concat_chunked < 1e-5, f"concat != chunked: {diff_concat_chunked}"
    assert diff_chunked_ip < 1e-5, f"chunked != inplace: {diff_chunked_ip}"

    # PEFT delta merge with normalized weights should also match
    norm_weights = [w / sum(weights) for w in weights]
    out_peft_norm = peft_delta_merge(x, W_base, adapters, norm_weights, normalize=False)
    # When we pass already-normalized weights with normalize=False,
    # this is equivalent to chunked with those normalized weights
    out_chunked_norm = chunked_accumulation(x, W_base, adapters, norm_weights)
    diff_peft_chunked = (out_peft_norm - out_chunked_norm).abs().max().item()
    print(f"peft(norm) vs chunked(norm): max_diff = {diff_peft_chunked:.2e}")
    assert diff_peft_chunked < 1e-5

    # PEFT with normalize=True and raw weights should match chunked with normalized weights
    out_peft_raw = peft_delta_merge(x, W_base, adapters, weights, normalize=True)
    diff_peft_raw = (out_peft_raw - out_chunked_norm).abs().max().item()
    print(f"peft(raw,normalize) vs chunked(norm): max_diff = {diff_peft_raw:.2e}")
    assert diff_peft_raw < 1e-5

    print("PASS: basic equivalence\n")


def test_equivalence_bfloat16():
    """Test in bfloat16 (actual training/inference dtype)."""
    torch.manual_seed(123)
    in_dim, out_dim, rank = 256, 512, 16
    num_tokens = 32

    x = torch.randn(num_tokens, in_dim, dtype=torch.bfloat16)
    W_base = torch.randn(out_dim, in_dim, dtype=torch.bfloat16)

    adapters = []
    for _ in range(3):
        A = torch.randn(rank, in_dim, dtype=torch.bfloat16) * 0.01
        B = torch.randn(out_dim, rank, dtype=torch.bfloat16) * 0.01
        adapters.append((A, B))

    weights = [0.5, 0.3, 0.2]

    out_concat = concat_merge(x, W_base, adapters, weights)
    out_chunked = chunked_accumulation(x, W_base, adapters, weights)
    out_chunked_ip = chunked_accumulation_inplace(x, W_base, adapters, weights)

    diff_cc = (out_concat - out_chunked).abs().max().item()
    diff_ip = (out_chunked - out_chunked_ip).abs().max().item()

    print(f"[bf16] concat vs chunked:     max_diff = {diff_cc:.2e}")
    print(f"[bf16] chunked vs chunked_ip: max_diff = {diff_ip:.2e}")

    # bf16 has ~3 decimal digits of precision; concat vs chunked accumulate
    # in different order, so rounding differs. 0.05 is generous but safe.
    assert diff_cc < 0.05, f"bf16 concat != chunked: {diff_cc}"
    assert diff_ip < 0.05, f"bf16 chunked != inplace: {diff_ip}"
    print("PASS: bfloat16 equivalence\n")


def test_single_adapter_weight1():
    """Single adapter with weight=1.0 should match stock LoRA path."""
    torch.manual_seed(7)
    in_dim, out_dim, rank = 128, 256, 8
    num_tokens = 10

    x = torch.randn(num_tokens, in_dim)
    W_base = torch.randn(out_dim, in_dim)
    A = torch.randn(rank, in_dim) * 0.01
    B = torch.randn(out_dim, rank) * 0.01

    adapters = [(A, B)]
    weights = [1.0]

    # Stock path: output = base + B @ A @ x  (scale=1.0, already baked)
    stock_output = x @ W_base.T + (x @ A.T) @ B.T

    out_chunked = chunked_accumulation(x, W_base, adapters, weights)
    diff = (stock_output - out_chunked).abs().max().item()
    print(f"stock vs chunked (w=1.0): max_diff = {diff:.2e}")
    assert diff < 1e-6, f"single adapter mismatch: {diff}"
    print("PASS: single adapter weight=1.0\n")


def test_weight_zero_skipped():
    """Adapter with weight=0 should not contribute."""
    torch.manual_seed(99)
    in_dim, out_dim, rank = 64, 128, 4
    num_tokens = 8

    x = torch.randn(num_tokens, in_dim)
    W_base = torch.randn(out_dim, in_dim)
    A1 = torch.randn(rank, in_dim) * 0.01
    B1 = torch.randn(out_dim, rank) * 0.01
    A2 = torch.randn(rank, in_dim) * 0.01
    B2 = torch.randn(out_dim, rank) * 0.01

    # Only adapter 1 active
    out_one = chunked_accumulation(x, W_base, [(A1, B1)], [0.6])
    # Both adapters but second has weight 0
    out_two = chunked_accumulation(x, W_base, [(A1, B1), (A2, B2)], [0.6, 0.0])

    diff = (out_one - out_two).abs().max().item()
    print(f"w=[0.6] vs w=[0.6, 0.0]: max_diff = {diff:.2e}")
    assert diff == 0.0, f"weight=0 should be no-op: {diff}"
    print("PASS: weight=0 skipped\n")


def test_merge_script_normalization_warning():
    """Demonstrate that merge_and_push_RL.py normalizes but vllm weighted doesn't.

    This is NOT a bug — it's a design difference. The caller must pass
    the right weights. This test documents the relationship.
    """
    torch.manual_seed(55)
    in_dim, out_dim, rank = 64, 128, 4
    num_tokens = 8

    x = torch.randn(num_tokens, in_dim)
    W_base = torch.randn(out_dim, in_dim)
    A1 = torch.randn(rank, in_dim) * 0.01
    B1 = torch.randn(out_dim, rank) * 0.01
    A2 = torch.randn(rank, in_dim) * 0.01
    B2 = torch.randn(out_dim, rank) * 0.01

    adapters = [(A1, B1), (A2, B2)]
    raw_weights = [1.0, 1.0]  # as in merge_and_push_RL.py

    # merge_and_push_RL.py: normalizes by total_weight (2.0)
    # so each adapter gets effective weight 0.5
    out_peft = peft_delta_merge(x, W_base, adapters, raw_weights, normalize=True)

    # vllm weighted LoRA: uses raw weights (1.0, 1.0)
    out_vllm_raw = chunked_accumulation(x, W_base, adapters, raw_weights)

    # vllm weighted LoRA with normalized weights: should match PEFT
    norm_weights = [w / sum(raw_weights) for w in raw_weights]
    out_vllm_norm = chunked_accumulation(x, W_base, adapters, norm_weights)

    diff_raw = (out_peft - out_vllm_raw).abs().max().item()
    diff_norm = (out_peft - out_vllm_norm).abs().max().item()

    print(f"peft(norm) vs vllm(raw [1,1]):  max_diff = {diff_raw:.2e}  (expected DIFFERENT)")
    print(f"peft(norm) vs vllm(norm [.5,.5]): max_diff = {diff_norm:.2e}  (expected SAME)")

    assert diff_raw > 1e-4, "Should differ with un-normalized weights"
    assert diff_norm < 1e-6, "Should match with normalized weights"
    print("PASS: normalization difference documented\n")


def test_sliced_output():
    """Test the output_slices logic (QKV-style fused linear layers).

    vllm fuses Q/K/V projections into one output tensor with slices.
    The weighted path must accumulate into the correct slice offsets.
    """
    torch.manual_seed(77)
    in_dim = 128
    q_dim, k_dim, v_dim = 256, 64, 64  # typical: Q larger than K,V
    out_dim = q_dim + k_dim + v_dim
    rank = 8
    num_tokens = 10

    x = torch.randn(num_tokens, in_dim)
    W_base = torch.randn(out_dim, in_dim)

    # Separate adapters for each slice (like vllm stores them)
    output_slices = (q_dim, k_dim, v_dim)
    adapters_per_slot = []  # list of (slot_adapters_A, slot_adapters_B) per adapter
    num_adapters = 2

    for _ in range(num_adapters):
        As = []
        Bs = []
        for slice_dim in output_slices:
            A = torch.randn(rank, in_dim) * 0.01
            B = torch.randn(slice_dim, rank) * 0.01
            As.append(A)
            Bs.append(B)
        adapters_per_slot.append((As, Bs))

    weights = [0.7, 0.3]

    # Method 1: Compute full output by doing each slice separately
    output_ref = x @ W_base.T
    for (As, Bs), w in zip(adapters_per_slot, weights):
        offset = 0
        for s, (A, B) in enumerate(zip(As, Bs)):
            hidden = x @ A.T
            delta = hidden @ B.T
            output_ref[:, offset:offset + output_slices[s]] += w * delta
            offset += output_slices[s]

    # Method 2: Mimic add_weighted_lora_linear with output_slices
    output_test = x @ W_base.T
    for adapter_idx, ((As, Bs), w) in enumerate(zip(adapters_per_slot, weights)):
        for s in range(len(As)):
            A = As[s]
            B = Bs[s]
            hidden = x @ A.T
            delta = hidden @ B.T
            if len(output_slices) == 1:
                output_test.add_(delta, alpha=w)
            else:
                offset = sum(output_slices[:s])
                output_test[:, offset:offset + output_slices[s]].add_(delta, alpha=w)

    diff = (output_ref - output_test).abs().max().item()
    print(f"sliced ref vs sliced impl: max_diff = {diff:.2e}")
    assert diff < 1e-6, f"sliced output mismatch: {diff}"
    print("PASS: sliced output equivalence\n")


def test_moe_weighted_w13():
    """Test MoE w13 (gate+up) deterministic LoRA via torch.bmm.

    Simulates _weighted_w13_lora logic: per-expert A/B gathered by token's
    assigned expert, then batched matmul.
    """
    torch.manual_seed(33)
    num_experts = 8
    hidden_size = 128
    inter_size = 256
    rank = 8
    num_tokens = 16
    top_k = 2

    x = torch.randn(num_tokens, hidden_size)
    topk_ids = torch.randint(0, num_experts, (num_tokens, top_k))

    # Two adapters
    adapters = []
    for _ in range(2):
        A = torch.randn(num_experts, rank, hidden_size) * 0.01
        B = torch.randn(num_experts, inter_size, rank) * 0.01
        adapters.append((A, B))

    weights = [0.6, 0.4]

    # Reference: loop per token, per expert
    N = num_tokens * top_k
    y_ref = torch.zeros(N, inter_size)
    flat_eids = topk_ids.reshape(-1)
    x_flat = x.unsqueeze(1).expand(-1, top_k, -1).reshape(N, -1)

    for (A_all, B_all), w in zip(adapters, weights):
        for n in range(N):
            eid = flat_eids[n]
            A_e = A_all[eid, :rank]     # [r, hidden]
            B_e = B_all[eid, :, :rank]  # [inter, r]
            delta = (x_flat[n] @ A_e.T) @ B_e.T  # [inter]
            y_ref[n] += w * delta

    # Batched: mirrors _weighted_w13_lora
    y_batched = torch.zeros(N, inter_size)
    for (A_all, B_all), w in zip(adapters, weights):
        A_gathered = A_all[flat_eids, :rank]    # [N, r, hidden]
        B_gathered = B_all[flat_eids, :, :rank] # [N, inter, r]
        hidden = torch.bmm(
            x_flat.unsqueeze(1), A_gathered.transpose(1, 2)
        ).squeeze(1)
        delta = torch.bmm(
            hidden.unsqueeze(1), B_gathered.transpose(1, 2)
        ).squeeze(1)
        y_batched.add_(delta, alpha=w)

    diff = (y_ref - y_batched).abs().max().item()
    print(f"MoE w13 loop vs bmm: max_diff = {diff:.2e}")
    assert diff < 1e-5, f"MoE w13 mismatch: {diff}"
    print("PASS: MoE w13 equivalence\n")


def test_moe_weighted_w2():
    """Test MoE w2 (down_proj) deterministic LoRA via torch.bmm.

    Simulates _weighted_w2_lora logic with 3D output [tokens, top_k, hidden].
    """
    torch.manual_seed(44)
    num_experts = 8
    hidden_size = 128
    inter_size = 256
    rank = 8
    num_tokens = 16
    top_k = 2

    x = torch.randn(num_tokens * top_k, inter_size)  # post-activation
    topk_ids = torch.randint(0, num_experts, (num_tokens, top_k))

    adapters = []
    for _ in range(2):
        A = torch.randn(num_experts, rank, inter_size) * 0.01
        B = torch.randn(num_experts, hidden_size, rank) * 0.01
        adapters.append((A, B))

    weights = [0.6, 0.4]
    N = num_tokens * top_k
    flat_eids = topk_ids.reshape(-1)

    # Reference: per-token loop
    y_ref = torch.zeros(num_tokens, top_k, hidden_size)
    for (A_all, B_all), w in zip(adapters, weights):
        for n in range(N):
            tok = n // top_k
            k = n % top_k
            eid = flat_eids[n]
            A_e = A_all[eid, :rank]
            B_e = B_all[eid, :, :rank]
            delta = (x[n] @ A_e.T) @ B_e.T
            y_ref[tok, k] += w * delta

    # Batched: mirrors _weighted_w2_lora (3D path)
    y_batched = torch.zeros(num_tokens, top_k, hidden_size)
    for (A_all, B_all), w in zip(adapters, weights):
        A_gathered = A_all[flat_eids, :rank]
        B_gathered = B_all[flat_eids, :, :rank]
        hidden = torch.bmm(
            x.unsqueeze(1), A_gathered.transpose(1, 2)
        ).squeeze(1)
        delta = torch.bmm(
            hidden.unsqueeze(1), B_gathered.transpose(1, 2)
        ).squeeze(1)
        y_batched.add_(delta.view(num_tokens, top_k, -1), alpha=w)

    diff = (y_ref - y_batched).abs().max().item()
    print(f"MoE w2 loop vs bmm (3D): max_diff = {diff:.2e}")
    assert diff < 1e-5, f"MoE w2 mismatch: {diff}"
    print("PASS: MoE w2 equivalence\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Weighted LoRA Equivalence Tests")
    print("=" * 60 + "\n")

    test_equivalence_basic()
    test_equivalence_bfloat16()
    test_single_adapter_weight1()
    test_weight_zero_skipped()
    test_merge_script_normalization_warning()
    test_sliced_output()
    test_moe_weighted_w13()
    test_moe_weighted_w2()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
