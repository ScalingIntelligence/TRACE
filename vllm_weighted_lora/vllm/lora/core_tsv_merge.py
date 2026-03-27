# SPDX-License-Identifier: Apache-2.0
"""Core/TSV LoRA merge: align subspaces via SVD, merge in core space, output PEFT adapter.

No vLLM imports — pure torch + safetensors.  Can be imported by both vLLM
serving code and the offline merge_and_push_RL.py script.
"""

import gc
import json
import os

import torch
from safetensors.torch import load_file, save_file


# ---------------------------------------------------------------------------
# LoRA I/O helpers
# ---------------------------------------------------------------------------

def load_lora_ab_pairs(adapter_path: str) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Load LoRA A/B matrix pairs from a PEFT adapter's safetensors.

    Returns dict: {layer_base_name: (A, B)} where A is (r, in_dim), B is (out_dim, r),
    and the LoRA scaling (alpha/r) is pre-applied to A so that delta_W = B @ A.
    """
    with open(os.path.join(adapter_path, "adapter_config.json")) as f:
        cfg = json.load(f)
    scaling = cfg["lora_alpha"] / cfg["r"]

    state = load_file(os.path.join(adapter_path, "adapter_model.safetensors"))
    ab_pairs = {}
    for key in state:
        if "lora_A" not in key:
            continue
        b_key = key.replace("lora_A", "lora_B")
        if b_key not in state:
            continue
        base_name = key.replace(".lora_A.weight", "")
        A = state[key] * scaling
        B = state[b_key]
        ab_pairs[base_name] = (A, B)
    return ab_pairs


def _peft_key_to_model_key(peft_base_name: str) -> str:
    """Convert PEFT adapter key to model state_dict key.

    e.g. 'base_model.model.model.layers.0.mlp.experts.0.down_proj'
      -> 'model.layers.0.mlp.experts.0.down_proj.weight'
    """
    key = peft_base_name
    if key.startswith("base_model.model."):
        key = key[len("base_model.model."):]
    return key + ".weight"


def _model_key_to_peft_key(model_key: str) -> str:
    """Inverse of _peft_key_to_model_key.

    e.g. 'model.layers.0.mlp.experts.0.down_proj.weight'
      -> 'base_model.model.model.layers.0.mlp.experts.0.down_proj'
    """
    key = model_key
    if key.endswith(".weight"):
        key = key[:-len(".weight")]
    return "base_model.model." + key


# ---------------------------------------------------------------------------
# Core space TSV merge
# ---------------------------------------------------------------------------

def _merge_core_matrices_tsv(
    M_list: list[torch.Tensor],
    weights: list[float],
) -> torch.Tensor:
    """Task Singular Vector merge: allocate each adapter a proportional share
    of singular directions, then re-orthogonalize.

    Uses batched SVD for GPU efficiency.
    """
    T = len(M_list)
    dim = M_list[0].shape[0]  # T*r
    k = max(1, int(dim / T))

    w_tensor = torch.tensor(weights[:T], dtype=M_list[0].dtype, device=M_list[0].device)
    stacked = torch.stack(M_list) * w_tensor.view(T, 1, 1)

    U_all, S_all, Vh_all = torch.linalg.svd(stacked, full_matrices=False)

    sum_u = torch.zeros(dim, dim, dtype=stacked.dtype, device=stacked.device)
    sum_s = torch.zeros(dim, dtype=stacked.dtype, device=stacked.device)
    sum_v = torch.zeros(dim, dim, dtype=stacked.dtype, device=stacked.device)

    for i in range(T):
        sl = slice(i * k, (i + 1) * k)
        sum_u[:, sl] = U_all[i, :, :k]
        sum_s[sl] = S_all[i, :k]
        sum_v[sl, :] = Vh_all[i, :k, :]

    u_u, _, v_u = torch.linalg.svd(sum_u, full_matrices=False)
    u_v, _, v_v = torch.linalg.svd(sum_v, full_matrices=False)

    left = u_u @ v_u
    right = u_v @ v_v
    return (left * sum_s.unsqueeze(0)) @ right


def combine_core_tsv(
    adapter_paths: list[str],
    weights: list[float],
    force_cpu: bool = False,
) -> dict[str, torch.Tensor]:
    """Core Space Merging with TSV sub-method and isotropize=True.

    Aligns LoRA subspaces via SVD reference bases and merges in the aligned
    low-rank core space.  Works directly on LoRA A/B matrices.

    Returns dict: {peft_base_name: delta_W_tensor (bfloat16, cpu)}
    keyed by PEFT base name (not model state_dict key) for easier
    re-factorization.
    """
    if force_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = len(adapter_paths)

    print(f"\n  [core_tsv] Loading LoRA A/B matrices from {T} adapters...")
    all_ab = []
    for i, path in enumerate(adapter_paths):
        print(f"    Adapter {i}: {os.path.basename(path)} (weight={weights[i]:.4f})")
        ab = load_lora_ab_pairs(path)
        all_ab.append(ab)
        print(f"      {len(ab)} LoRA layer pairs loaded")

    print(f"  [core_tsv] Moving all LoRA matrices to {device} (float64)...")
    for ab in all_ab:
        for key in ab:
            A, B = ab[key]
            ab[key] = (A.to(device=device, dtype=torch.float64),
                       B.to(device=device, dtype=torch.float64))

    layer_keys = list(all_ab[0].keys())
    print(f"  [core_tsv] {len(layer_keys)} LoRA layers to merge (TSV + isotropize)")

    combined_delta = {}

    for layer_idx, layer_key in enumerate(layer_keys):
        if layer_idx % 2000 == 0:
            print(f"    Processing layer {layer_idx}/{len(layer_keys)}...")

        A_list = []
        B_list = []
        for ab in all_ab:
            if layer_key not in ab:
                continue
            A, B = ab[layer_key]
            A_list.append(A)
            B_list.append(B)

        if len(A_list) < 2:
            if len(A_list) == 1:
                idx = next(j for j, ab in enumerate(all_ab) if layer_key in ab)
                A, B = all_ab[idx][layer_key]
                delta = (B @ A).to(torch.bfloat16) * weights[idx]
                combined_delta[layer_key] = delta.cpu()
            continue

        A_stack = torch.cat(A_list, dim=0)
        B_stack = torch.cat(B_list, dim=1)

        Vh_A_ref = torch.linalg.svd(A_stack, full_matrices=False)[2]
        U_B_ref = torch.linalg.svd(B_stack, full_matrices=False)[0]

        M_list = []
        for A, B in zip(A_list, B_list):
            M = U_B_ref.T @ B @ A @ Vh_A_ref.T
            M_list.append(M)

        M_merged = _merge_core_matrices_tsv(M_list, weights)

        # Isotropize: equalize singular values
        U_m, S_m, Vh_m = torch.linalg.svd(M_merged, full_matrices=False)
        M_merged = S_m.mean() * (U_m @ Vh_m)

        delta_W = U_B_ref @ M_merged @ Vh_A_ref

        combined_delta[layer_key] = delta_W.to(dtype=torch.bfloat16, device="cpu")

        del A_stack, B_stack, Vh_A_ref, U_B_ref, M_list, M_merged, delta_W
        if layer_idx % 500 == 0 and device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"  [core_tsv] Merge complete: {len(combined_delta)} layers")

    del all_ab
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return combined_delta


# ---------------------------------------------------------------------------
# Re-factorize full-rank delta → low-rank (A, B) and save as PEFT adapter
# ---------------------------------------------------------------------------

def refactorize_delta(
    delta_W: torch.Tensor,
    target_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Truncated SVD of delta_W → (A, B) such that B @ A ≈ delta_W.

    A: (target_rank, in_dim), B: (out_dim, target_rank).
    Scaling is pre-applied (lora_alpha = target_rank → scaling = 1.0).
    """
    # SVD in float32 for numerical stability
    U, S, Vh = torch.linalg.svd(delta_W.float(), full_matrices=False)
    sqrt_S = S[:target_rank].sqrt()
    B = (U[:, :target_rank] * sqrt_S.unsqueeze(0)).to(delta_W.dtype)   # (out, r)
    A = (sqrt_S.unsqueeze(1) * Vh[:target_rank, :]).to(delta_W.dtype)  # (r, in)
    return A, B


def core_tsv_merge(
    adapter_paths: list[str],
    weights: list[float],
    output_path: str,
    target_rank: int | None = None,
    force_cpu: bool = False,
) -> None:
    """End-to-end: core/TSV merge multiple adapters → single PEFT adapter on disk.

    1. Load A/B pairs from each adapter
    2. Core space merge with TSV + isotropize
    3. Re-factorize each layer's delta_W to (A, B) at target_rank
    4. Save as PEFT-format adapter (safetensors + adapter_config.json)
    """
    # Read source adapter config for rank and target modules
    with open(os.path.join(adapter_paths[0], "adapter_config.json")) as f:
        src_cfg = json.load(f)

    if target_rank is None:
        target_rank = src_cfg["r"]

    print(f"  [core_tsv_merge] Target rank: {target_rank}")

    # Step 1-2: Core/TSV merge → dict of {peft_base_name: delta_W}
    combined_delta = combine_core_tsv(adapter_paths, weights, force_cpu=force_cpu)

    # Step 3: Batched re-factorize — group layers by shape, batch SVD
    print(f"  [core_tsv_merge] Re-factorizing {len(combined_delta)} layers to rank {target_rank}...")
    adapter_state = {}

    # Group by (out_dim, in_dim) for batched SVD
    from collections import defaultdict
    shape_groups: dict[tuple[int, int], list[tuple[str, torch.Tensor]]] = defaultdict(list)
    for layer_key, delta_W in combined_delta.items():
        shape_groups[(delta_W.shape[0], delta_W.shape[1])].append((layer_key, delta_W))

    print(f"    {len(shape_groups)} unique shapes, batching SVDs...")
    for (out_dim, in_dim), group in shape_groups.items():
        keys = [k for k, _ in group]
        # Stack all deltas of this shape: (batch, out_dim, in_dim)
        stacked = torch.stack([d.float() for _, d in group])
        batch_size = stacked.shape[0]

        # Process in chunks to avoid excessive memory use
        chunk_size = min(512, batch_size)
        for chunk_start in range(0, batch_size, chunk_size):
            chunk_end = min(chunk_start + chunk_size, batch_size)
            chunk = stacked[chunk_start:chunk_end]
            chunk_keys = keys[chunk_start:chunk_end]

            U, S, Vh = torch.linalg.svd(chunk, full_matrices=False)
            sqrt_S = S[:, :target_rank].sqrt()

            for i, layer_key in enumerate(chunk_keys):
                orig_dtype = group[chunk_start + i][1].dtype
                B_i = (U[i, :, :target_rank] * sqrt_S[i].unsqueeze(0)).to(orig_dtype)
                A_i = (sqrt_S[i].unsqueeze(1) * Vh[i, :target_rank, :]).to(orig_dtype)
                adapter_state[layer_key + ".lora_A.weight"] = A_i
                adapter_state[layer_key + ".lora_B.weight"] = B_i

        del stacked
    print(f"    Batched re-factorization complete")

    del combined_delta, shape_groups
    gc.collect()

    # Step 4: Save as PEFT adapter
    os.makedirs(output_path, exist_ok=True)

    save_file(adapter_state, os.path.join(output_path, "adapter_model.safetensors"))

    # Write adapter_config.json: alpha = r so scaling = 1.0 (pre-applied in A)
    out_cfg = dict(src_cfg)
    out_cfg["r"] = target_rank
    out_cfg["lora_alpha"] = target_rank  # scaling = alpha/r = 1.0
    with open(os.path.join(output_path, "adapter_config.json"), "w") as f:
        json.dump(out_cfg, f, indent=2)

    print(f"  [core_tsv_merge] Saved merged adapter to {output_path}")
    print(f"    {len(adapter_state)} tensors, rank={target_rank}, alpha={target_rank}")
