#!/usr/bin/env python3
"""Offline core/TSV merge: combine multiple LoRA adapters into a single adapter.

Aligns LoRA subspaces via SVD reference bases (Core Space Merging, NeurIPS 2025),
merges using Task Singular Vectors (TSV) with isotropization, then re-factorizes
back to rank-r and saves as a standard PEFT adapter.

Optimized for MoE models: batches all same-shape layers into single SVD calls,
and refactorizes directly in core space (no full delta_W materialization).

Usage:
    python core_tsv_merge_adapters.py \
        --adapters /path/to/adapter1:0.6 /path/to/adapter2:0.4 \
        --output /path/to/merged_adapter

    # Equal weights (default 1.0 each):
    python core_tsv_merge_adapters.py \
        --adapters /path/to/adapter1 /path/to/adapter2 /path/to/adapter3 \
        --output /path/to/merged_adapter
"""

import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict

import torch
from safetensors.torch import load_file, save_file


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# LoRA I/O
# ---------------------------------------------------------------------------

def load_lora_ab_pairs(adapter_path):
    """Load LoRA A/B matrix pairs with scaling pre-applied to A."""
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
        ab_pairs[base_name] = (state[key] * scaling, state[b_key])
    return ab_pairs


# ---------------------------------------------------------------------------
# Core/TSV merge with fused core-space refactorization
# ---------------------------------------------------------------------------

def merge_core_tsv(adapter_paths, weights, target_rank, device):
    """Core Space Merging with TSV + isotropize, with fused refactorization.

    Key optimizations vs naive approach:
    1. float32 instead of float64 (sufficient for bfloat16 source weights)
    2. Refactorizes directly from the small (Tr x Tr) core matrix instead of
       materializing full (m x n) delta_W — mathematically exact because
       orthogonal reference bases preserve singular values.
    3. Single-adapter layers keep original A/B (no redundant SVD round-trip).
    """
    T = len(adapter_paths)
    log(f"\n  Loading LoRA A/B matrices from {T} adapters...")
    all_ab = []
    for i, path in enumerate(adapter_paths):
        log(f"    Adapter {i}: {os.path.basename(path)} (weight={weights[i]:.4f})")
        ab = load_lora_ab_pairs(path)
        all_ab.append(ab)
        log(f"      {len(ab)} LoRA layer pairs loaded")

    # Use union of all adapter keys (not just adapter[0])
    all_keys = set()
    for ab in all_ab:
        all_keys.update(ab.keys())

    multi_adapter_layers = []
    single_adapter_layers = []
    for layer_key in sorted(all_keys):
        present_count = sum(1 for ab in all_ab if layer_key in ab)
        if present_count >= 2:
            multi_adapter_layers.append(layer_key)
        elif present_count == 1:
            single_adapter_layers.append(layer_key)

    log(f"  {len(multi_adapter_layers)} multi-adapter layers, "
        f"{len(single_adapter_layers)} single-adapter layers")

    adapter_state = {}

    # Single-adapter layers: keep original A/B scaled by weight
    for layer_key in single_adapter_layers:
        idx = next(j for j, ab in enumerate(all_ab) if layer_key in ab)
        A, B = all_ab[idx][layer_key]
        w = weights[idx]
        adapter_state[layer_key + ".lora_A.weight"] = (A * w).to(torch.bfloat16).contiguous()
        adapter_state[layer_key + ".lora_B.weight"] = B.to(torch.bfloat16).contiguous()

    if not multi_adapter_layers:
        log(f"  No multi-adapter layers to merge")
        return adapter_state

    # Group by shape for batched SVD
    shape_groups = defaultdict(list)
    for layer_key in multi_adapter_layers:
        for ab in all_ab:
            if layer_key in ab:
                A0, B0 = ab[layer_key]
                sig = (A0.shape[0], A0.shape[1], B0.shape[0], B0.shape[1])
                shape_groups[sig].append(layer_key)
                break

    log(f"  {len(shape_groups)} unique shape groups for batched SVD")

    for sig, group_keys in shape_groups.items():
        r, n, m, r2 = sig
        N = len(group_keys)
        Tr = T * r
        effective_rank = min(target_rank, Tr)
        log(f"    Shape ({m},{r})x({r},{n}): {N} layers...")

        # Stack A and B for all adapters (zero-fill if adapter lacks a key)
        A_all = []
        B_all = []
        for ab in all_ab:
            A_batch = torch.stack([
                ab[k][0] if k in ab else torch.zeros(r, n)
                for k in group_keys
            ]).to(device=device, dtype=torch.float32)
            B_batch = torch.stack([
                ab[k][1] if k in ab else torch.zeros(m, r2)
                for k in group_keys
            ]).to(device=device, dtype=torch.float32)
            A_all.append(A_batch)
            B_all.append(B_batch)

        # Reference bases via batched SVD
        A_stack = torch.cat(A_all, dim=1)  # (N, T*r, n)
        B_stack = torch.cat(B_all, dim=2)  # (N, m, T*r)

        log(f"      Computing reference bases (batched SVD)...")
        Vh_A_ref = torch.linalg.svd(A_stack, full_matrices=False)[2]  # (N, Tr, n)
        U_B_ref = torch.linalg.svd(B_stack, full_matrices=False)[0]   # (N, m, Tr)

        del A_stack, B_stack

        log(f"      Core matrices → TSV merge → refactorize...")
        w_tensor = torch.tensor(weights[:T], dtype=torch.float32, device=device)

        chunk_size = 512
        for c_start in range(0, N, chunk_size):
            c_end = min(c_start + chunk_size, N)
            cn = c_end - c_start

            U_chunk = U_B_ref[c_start:c_end]   # (cn, m, Tr)
            Vh_chunk = Vh_A_ref[c_start:c_end]  # (cn, Tr, n)

            # Core matrices: M_i = U_B_ref^T @ B_i @ A_i @ Vh_A_ref^T
            M_stacked = torch.zeros(T, cn, Tr, Tr, dtype=torch.float32, device=device)
            for t_idx in range(T):
                A_t = A_all[t_idx][c_start:c_end]
                B_t = B_all[t_idx][c_start:c_end]
                M_stacked[t_idx] = torch.bmm(
                    U_chunk.transpose(1, 2),
                    torch.bmm(B_t, torch.bmm(A_t, Vh_chunk.transpose(1, 2)))
                )

            # TSV: weighted SVD per adapter, assemble top-k directions
            k = max(1, int(Tr / T))
            M_weighted = M_stacked * w_tensor.view(T, 1, 1, 1)
            M_flat = M_weighted.reshape(T * cn, Tr, Tr)

            U_m_all, S_m_all, Vh_m_all = torch.linalg.svd(M_flat, full_matrices=False)
            U_m_all = U_m_all.reshape(T, cn, Tr, Tr)
            S_m_all = S_m_all.reshape(T, cn, Tr)
            Vh_m_all = Vh_m_all.reshape(T, cn, Tr, Tr)

            del M_stacked, M_weighted, M_flat

            sum_u = torch.zeros(cn, Tr, Tr, dtype=torch.float32, device=device)
            sum_s = torch.zeros(cn, Tr, dtype=torch.float32, device=device)
            sum_v = torch.zeros(cn, Tr, Tr, dtype=torch.float32, device=device)

            for t_idx in range(T):
                sl = slice(t_idx * k, (t_idx + 1) * k)
                sum_u[:, :, sl] = U_m_all[t_idx, :, :, :k]
                sum_s[:, sl] = S_m_all[t_idx, :, :k]
                sum_v[:, sl, :] = Vh_m_all[t_idx, :, :k, :]

            del U_m_all, S_m_all, Vh_m_all

            # Re-orthogonalize (polar decomposition via SVD)
            u_u, _, v_u = torch.linalg.svd(sum_u, full_matrices=False)
            u_v, _, v_v = torch.linalg.svd(sum_v, full_matrices=False)
            left = torch.bmm(u_u, v_u)
            right = torch.bmm(u_v, v_v)
            M_merged = torch.bmm(left * sum_s.unsqueeze(1), right)

            del sum_u, sum_s, sum_v, left, right, u_u, v_u, u_v, v_v

            # Isotropize: equalize singular values
            U_iso, S_iso, Vh_iso = torch.linalg.svd(M_merged, full_matrices=False)
            S_mean = S_iso.mean(dim=1, keepdim=True).unsqueeze(1)  # (cn, 1, 1)
            M_merged = S_mean * torch.bmm(U_iso, Vh_iso)

            del U_iso, S_iso, Vh_iso

            # --- Fused core-space refactorization ---
            # SVD of M_merged (cn, Tr, Tr) — small matrices, trivial cost.
            # Truncate to effective_rank, then multiply through reference bases
            # to get LoRA A/B directly. Skips materializing full (m, n) deltas.
            U_M, S_M, Vh_M = torch.linalg.svd(M_merged, full_matrices=False)
            sqrt_S = S_M[:, :effective_rank].sqrt()  # (cn, er)

            # B_new = U_B_ref @ U_M[:,:er] * sqrt(S)  — shape (cn, m, er)
            B_new = torch.bmm(U_chunk, U_M[:, :, :effective_rank]) * sqrt_S.unsqueeze(1)
            # A_new = sqrt(S) * Vh_M[:er,:] @ Vh_A_ref — shape (cn, er, n)
            A_new = sqrt_S.unsqueeze(2) * torch.bmm(Vh_M[:, :effective_rank, :], Vh_chunk)

            del M_merged, U_M, S_M, Vh_M, sqrt_S

            for i, layer_key in enumerate(group_keys[c_start:c_end]):
                adapter_state[layer_key + ".lora_A.weight"] = \
                    A_new[i].to(torch.bfloat16).cpu().contiguous()
                adapter_state[layer_key + ".lora_B.weight"] = \
                    B_new[i].to(torch.bfloat16).cpu().contiguous()

            del B_new, A_new

        del A_all, B_all, U_B_ref, Vh_A_ref
        if device.type == "cuda":
            torch.cuda.empty_cache()

    log(f"  Merge complete: {len(adapter_state) // 2} LoRA layers")
    del all_ab
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return adapter_state


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_adapter_spec(spec):
    """Parse 'path:weight' or 'path' (default weight=1.0)."""
    if ":" in spec:
        parts = spec.rsplit(":", 1)
        try:
            weight = float(parts[1])
            return parts[0], weight
        except ValueError:
            pass
    return spec, 1.0


def main():
    parser = argparse.ArgumentParser(
        description="Core/TSV merge multiple LoRA adapters into one"
    )
    parser.add_argument(
        "--adapters", nargs="+", required=True,
        help="Adapter paths with optional weights: /path:weight (default weight=1.0)"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for merged PEFT adapter"
    )
    parser.add_argument(
        "--target-rank", type=int, default=None,
        help="Target rank for merged adapter (default: same as source)"
    )
    args = parser.parse_args()

    adapter_paths = []
    weights = []
    for spec in args.adapters:
        path, weight = parse_adapter_spec(spec)
        if not os.path.isdir(path):
            log(f"ERROR: Adapter path not found: {path}")
            sys.exit(1)
        adapter_paths.append(path)
        weights.append(weight)

    if len(adapter_paths) < 2:
        log("ERROR: Need at least 2 adapters to merge")
        sys.exit(1)

    with open(os.path.join(adapter_paths[0], "adapter_config.json")) as f:
        src_cfg = json.load(f)
    target_rank = args.target_rank or src_cfg["r"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log(f"Core/TSV Merge")
    log(f"  Adapters: {len(adapter_paths)}")
    for p, w in zip(adapter_paths, weights):
        log(f"    {os.path.basename(p)} (weight={w})")
    log(f"  Target rank: {target_rank}")
    log(f"  Device: {device}")
    log(f"  Output: {args.output}")

    t0 = time.time()
    adapter_state = merge_core_tsv(adapter_paths, weights, target_rank, device)
    t1 = time.time()
    log(f"  Merge + refactorize took {t1 - t0:.1f}s")

    os.makedirs(args.output, exist_ok=True)
    save_file(adapter_state, os.path.join(args.output, "adapter_model.safetensors"))

    out_cfg = dict(src_cfg)
    out_cfg["r"] = target_rank
    out_cfg["lora_alpha"] = target_rank
    with open(os.path.join(args.output, "adapter_config.json"), "w") as f:
        json.dump(out_cfg, f, indent=2)

    t2 = time.time()
    n_a = sum(1 for k in adapter_state if "lora_A" in k)
    log(f"\n  Saved merged adapter to {args.output}")
    log(f"    {n_a} LoRA layers, rank={target_rank}, alpha={target_rank}")
    log(f"    Total time: {t2 - t0:.1f}s")


if __name__ == "__main__":
    main()
