import os
os.environ["HF_HOME"] = "/home/ubuntu/.cache/huggingface"
os.environ["TMPDIR"] = "/home/ubuntu/tmp"
os.makedirs("/home/ubuntu/tmp", exist_ok=True)

from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi
import torch
import gc
import json
import shutil
from safetensors.torch import load_file

# --- Configuration ---

HF_TOKEN = "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"

# Each entry: (target_repo, adapters_list, optional_config_dict)
#
# adapters_list: [(adapter_path, weight), ...]
#
# config dict options:
#   method: "linear"     - (default) weighted sum of full-model deltas
#           "ties_dare"  - TIES-DARE merge on full-model deltas
#           "stack"      - concatenate LoRA matrices via PEFT (preserves each adapter's
#                          rank-r subspace, effective rank = num_adapters * r),
#                          then merge into base for full-model output
#           "core"       - Core Space Merging (NeurIPS 2025): aligns LoRA subspaces via
#                          SVD reference bases, merges in aligned low-rank core space.
#                          No full-model loading for delta computation — works directly
#                          on LoRA A/B matrices. Supports heterogeneous ranks.
#
#   For ties_dare:
#     density: float 0-1 (default 0.7) - DARE keep probability & TIES trim threshold
#     majority_sign_method: "total" or "frequency" (default "total")
#     seed: int (default 42) - random seed for DARE reproducibility
#
#   For core:
#     core_merge: "sum"       - (default) Task Arithmetic: weighted sum of aligned core matrices
#                 "ties"      - TIES merge in core space (prune + sign election + disjoint merge)
#                 "dare_ties" - DARE + TIES in core space
#                 "tsv"       - Task Singular Vector: allocates each adapter a proportional
#                               share of singular directions, avoids inter-task interference
#     isotropize: bool (default True) - equalize singular values of merged core matrix per layer
#     density: float 0-1 (default 0.7) - for ties/dare_ties: TIES trim fraction
#     dare_density: float 0-1 (default 0.7) - for dare_ties: DARE keep probability
#     seed: int (default 42) - random seed for DARE
MERGE_JOBS = [
    # (
    #     "tarsur909/merged-tc-sd-ms-pre-linear",
    #     [
    #         ("/home/ubuntu/.cache/huggingface/structured_data_reasoning/grpo_ckpt_iter_40", 1.0),
    #         ("/home/ubuntu/.cache/huggingface/multistep_task/grpo_ckpt_iter_10_20260318_110546", 1.0),
    #         ("/home/ubuntu/.cache/huggingface/tau_tool_calling/grpo_ckpt_iter_40", 1.0),
    #         ("/home/ubuntu/.cache/huggingface/precondition_check/grpo_ckpt_iter_40_20260319_035848", 1.0),
    #     ],
    # ),
    # (
    #     "tarsur909/merged-tc-sd-ms-pre-stac-30mt",
    #     [
    #         ("/home/ubuntu/.cache/huggingface/structured_data_reasoning/grpo_ckpt_iter_40", 1.0),
    #         ("/home/ubuntu/.cache/huggingface/multistep_task/grpo_ckpt_iter_10_20260318_110546", 1.0),
    #         ("/home/ubuntu/.cache/huggingface/tau_tool_calling/grpo_ckpt_iter_40", 1.0),
    #         ("/home/ubuntu/.cache/huggingface/precondition_check/grpo_ckpt_iter_40_20260319_035848", 1.0),
    #     ],
    #     {"method": "stack"},
    # ),
    # (
    #     "tarsur909/merged-tc-sd-ms-pre-ties-dare",
    #     [
    #         ("/home/ubuntu/.cache/huggingface/structured_data_reasoning/grpo_ckpt_iter_40", 1.0),
    #         ("/home/ubuntu/.cache/huggingface/multistep_task/grpo_ckpt_iter_10_20260318_110546", 1.0),
    #         ("/home/ubuntu/.cache/huggingface/tau_tool_calling/grpo_ckpt_iter_40", 1.0),
    #         ("/home/ubuntu/.cache/huggingface/precondition_check/grpo_ckpt_iter_40_20260319_035848", 1.0),
    #     ],
    #     {"method": "ties_dare"},
    # ),
    (
        "tarsur909/sft-adp-v1",
        [
            ("/home/ubuntu/.cache/huggingface/adp_baseline/sft_lora/checkpoint-120", 1.0)
        ]
    ),
]


# --- Merge helpers ---

def compute_deltas(base_model_name, base_state, adapters):
    """Compute full-model weight deltas for each adapter relative to base."""
    deltas = []
    for i, (path, weight) in enumerate(adapters):
        print(f"\n  Computing delta for adapter {i}: {os.path.basename(path)} (weight={weight})")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        peft_model = PeftModel.from_pretrained(model, path)
        merged = peft_model.merge_and_unload()
        merged_state = merged.state_dict()

        delta = {}
        for k in merged_state:
            d = merged_state[k].cpu() - base_state[k]
            if d.any():
                delta[k] = d

        deltas.append(delta)
        print(f"    {len(delta)} parameters changed")

        del model, peft_model, merged, merged_state
        gc.collect()
        torch.cuda.empty_cache()

    return deltas


def combine_linear(deltas, weights):
    """Weighted sum of deltas."""
    combined = {}
    for delta, w in zip(deltas, weights):
        for k, v in delta.items():
            if k not in combined:
                combined[k] = v * w
            else:
                combined[k] = combined[k] + v * w
    return combined


def combine_ties_dare(deltas, weights, density=0.7, majority_sign_method="total", seed=42):
    """
    GPU-vectorized TIES-DARE merge on full-model deltas.

    1. DARE: randomly drop (1-density) fraction of each delta, rescale by 1/density
    2. TIES Trim: keep only top `density` fraction by magnitude per delta
    3. TIES Elect sign: per-position majority vote weighted by magnitude
    4. TIES Disjoint merge: keep only values matching elected sign, weighted sum

    All operations are batched across adapters on GPU using stacked [N, *shape] tensors.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

    all_keys = set()
    for d in deltas:
        all_keys.update(d.keys())

    combined = {}
    for k in all_keys:
        indices = [i for i, d in enumerate(deltas) if k in d]
        if not indices:
            continue

        orig_dtype = deltas[indices[0]][k].dtype
        # Stack all adapter deltas for this key: [N, *shape] on GPU as float32
        stacked = torch.stack([deltas[i][k] for i in indices]).to(device=device, dtype=torch.float32)
        N = stacked.shape[0]
        w = torch.tensor([weights[i] for i in indices], dtype=torch.float32, device=device)
        # Broadcast-ready weights: [N, 1, 1, ...]
        w_broad = w.view(N, *([1] * (stacked.dim() - 1)))

        # Step 1: DARE - vectorized drop and rescale across all adapters at once
        if density < 1.0:
            dare_mask = torch.bernoulli(torch.full_like(stacked, density))
            stacked = stacked * dare_mask / density

        # Step 2: TIES Trim - batched kthvalue threshold per adapter
        # Uses kthvalue instead of quantile to handle large tensors (>2^24 elements)
        if density < 1.0:
            flat = stacked.abs().flatten(start_dim=1)  # [N, numel]
            numel = flat.shape[1]
            if numel > 0:
                kth = max(1, int((1.0 - density) * numel))
                threshold, _ = torch.kthvalue(flat, kth, dim=1, keepdim=True)  # [N, 1]
                threshold = threshold.view(N, *([1] * (stacked.dim() - 1)))
                stacked = stacked * (stacked.abs() >= threshold)

        # Step 3: TIES Elect sign - vectorized weighted vote
        signs = torch.sign(stacked)  # [N, *shape]
        if majority_sign_method == "total":
            sign_vote = (signs * stacked.abs() * w_broad).sum(dim=0)
        else:  # frequency
            sign_vote = (signs * w_broad).sum(dim=0)
        elected_sign = torch.sign(sign_vote)  # [*shape]

        # Step 4: TIES Disjoint merge - vectorized masked weighted sum
        agree = (signs == elected_sign.unsqueeze(0))  # [N, *shape]
        result = (stacked * agree * w_broad).sum(dim=0)  # [*shape]

        combined[k] = result.to(dtype=orig_dtype, device="cpu")

        del stacked, w, w_broad, signs, sign_vote, elected_sign, agree, result
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return combined


def load_lora_ab_pairs(adapter_path):
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
        # Strip PEFT prefix and .lora_A.weight suffix to get the model weight key
        base_name = key.replace(".lora_A.weight", "")
        A = state[key] * scaling  # pre-apply LoRA scaling so delta_W = B @ A
        B = state[b_key]
        ab_pairs[base_name] = (A, B)
    return ab_pairs


def _peft_key_to_model_key(peft_base_name):
    """Convert PEFT adapter key to model state_dict key.

    e.g. 'base_model.model.model.layers.0.mlp.experts.0.down_proj'
      -> 'model.layers.0.mlp.experts.0.down_proj.weight'
    """
    # Strip 'base_model.model.' prefix added by PEFT
    key = peft_base_name
    if key.startswith("base_model.model."):
        key = key[len("base_model.model."):]
    return key + ".weight"


def _merge_core_matrices_sum(M_list, weights):
    """Task Arithmetic: weighted sum."""
    result = torch.zeros_like(M_list[0])
    for M, w in zip(M_list, weights):
        result += M * w
    return result


def _merge_core_matrices_ties(M_list, weights, density=0.7):
    """TIES merge on flattened core matrices: trim, elect sign, disjoint merge."""
    stacked = torch.stack(M_list)  # (T, Tr, Tr)
    T = stacked.shape[0]
    orig_shape = stacked.shape[1:]
    flat = stacked.reshape(T, -1)  # (T, Tr*Tr)

    # Apply weights
    w = torch.tensor(weights, dtype=flat.dtype, device=flat.device).view(T, 1)
    flat_weighted = flat * w

    # Step 1: Trim — keep top `density` fraction by magnitude per adapter
    if density < 1.0:
        numel = flat.shape[1]
        kth = max(1, int((1.0 - density) * numel))
        threshold, _ = flat.abs().kthvalue(kth, dim=1, keepdim=True)
        flat = flat * (flat.abs() >= threshold)
        flat_weighted = flat * w

    # Step 2: Elect sign — majority vote weighted by magnitude
    signs = torch.sign(flat_weighted)
    sign_vote = (signs * flat_weighted.abs()).sum(dim=0)
    elected_sign = torch.sign(sign_vote)

    # Step 3: Disjoint merge — keep only values matching elected sign, sum
    agree = (torch.sign(flat) == elected_sign.unsqueeze(0))
    result = (flat_weighted * agree).sum(dim=0)

    return result.reshape(orig_shape)


def _merge_core_matrices_dare_ties(M_list, weights, density=0.7, dare_density=0.7, seed=42):
    """DARE + TIES merge on core matrices."""
    torch.manual_seed(seed)
    stacked = torch.stack(M_list)  # (T, Tr, Tr)

    # DARE: randomly drop and rescale
    if dare_density < 1.0:
        dare_mask = torch.bernoulli(torch.full_like(stacked, dare_density))
        stacked = stacked * dare_mask / dare_density

    M_list_dare = list(stacked)
    return _merge_core_matrices_ties(M_list_dare, weights, density=density)


def _merge_core_matrices_tsv(M_list, weights):
    """Task Singular Vector merge: allocate each adapter a proportional share of
    singular directions, then re-orthogonalize.

    Each adapter's core matrix M_i is SVD'd. The top (1/T) fraction of singular
    vectors from each adapter are placed into dedicated slots, preserving each
    task's most important directions without interference. Final reconstruction
    re-orthogonalizes U and V via SVD to produce a valid matrix.

    Uses batched SVD for GPU efficiency — all per-adapter SVDs run in a single
    kernel launch instead of T sequential calls.
    """
    T = len(M_list)
    dim = M_list[0].shape[0]  # T*r
    k = max(1, int(dim / T))  # singular directions per adapter

    # Apply weights and batch all adapter matrices for a single batched SVD
    w_tensor = torch.tensor(weights[:T], dtype=M_list[0].dtype, device=M_list[0].device)
    stacked = torch.stack(M_list) * w_tensor.view(T, 1, 1)  # (T, dim, dim)

    # Batched SVD: one kernel launch for all adapters
    U_all, S_all, Vh_all = torch.linalg.svd(stacked, full_matrices=False)

    # Scatter top-k singular components from each adapter into assembled matrices
    sum_u = torch.zeros(dim, dim, dtype=stacked.dtype, device=stacked.device)
    sum_s = torch.zeros(dim, dtype=stacked.dtype, device=stacked.device)
    sum_v = torch.zeros(dim, dim, dtype=stacked.dtype, device=stacked.device)

    for i in range(T):
        sl = slice(i * k, (i + 1) * k)
        sum_u[:, sl] = U_all[i, :, :k]
        sum_s[sl] = S_all[i, :k]
        sum_v[sl, :] = Vh_all[i, :k, :]

    # Re-orthogonalize U and V via SVD
    u_u, _, v_u = torch.linalg.svd(sum_u, full_matrices=False)
    u_v, _, v_v = torch.linalg.svd(sum_v, full_matrices=False)

    # Efficient diagonal scaling: (A @ diag(s) @ B) = (A * s[None,:]) @ B
    # Avoids materializing a full (dim, dim) diagonal matrix
    left = u_u @ v_u   # (dim, dim)
    right = u_v @ v_v   # (dim, dim)
    return (left * sum_s.unsqueeze(0)) @ right


def combine_core_space(adapters, core_merge="sum", isotropize=True, density=0.7,
                       dare_density=0.7, seed=42):
    """
    Core Space Merging (Panariello et al., NeurIPS 2025).

    Aligns LoRA subspaces via SVD reference bases and merges in the aligned
    low-rank core space. Works directly on LoRA A/B matrices — no full-model
    delta computation needed.

    For each LoRA layer across T adapters:
      1. Stack A matrices → SVD → Vh_A_ref (input space reference basis)
      2. Stack B matrices → SVD → U_B_ref (output space reference basis)
      3. Project each adapter: M_i = U_B_ref^T @ B_i @ A_i @ Vh_A_ref^T  (T*r × T*r)
      4. Merge M matrices using sub-method (sum/ties/dare_ties)
      5. Optionally isotropize (equalize singular values)
      6. Reconstruct: delta_W = U_B_ref @ M_merged @ Vh_A_ref

    Returns dict: {model_state_dict_key: delta_tensor (bfloat16, cpu)}
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter_weights = [w for _, w in adapters]
    T = len(adapters)

    # Load all adapter A/B pairs
    print(f"\n  Loading LoRA A/B matrices from {T} adapters...")
    all_ab = []
    for i, (path, weight) in enumerate(adapters):
        print(f"    Adapter {i}: {os.path.basename(path)} (weight={weight})")
        ab = load_lora_ab_pairs(path)
        all_ab.append(ab)
        print(f"      {len(ab)} LoRA layer pairs loaded")

    # Pre-move all LoRA matrices to target device in float64 (avoids per-layer transfers)
    print(f"  Moving all LoRA matrices to {device} (float64)...")
    for ab in all_ab:
        for key in ab:
            A, B = ab[key]
            ab[key] = (A.to(device=device, dtype=torch.float64),
                       B.to(device=device, dtype=torch.float64))

    # Use first adapter's keys as reference (all adapters should have same layers)
    layer_keys = list(all_ab[0].keys())
    print(f"  {len(layer_keys)} LoRA layers to merge in core space")
    print(f"  Sub-merge method: {core_merge}, isotropize: {isotropize}")

    combined_delta = {}
    rank_stats = []

    for layer_idx, layer_key in enumerate(layer_keys):
        if layer_idx % 2000 == 0:
            print(f"    Processing layer {layer_idx}/{len(layer_keys)}...")

        # Gather A and B matrices for this layer from all adapters
        A_list = []
        B_list = []
        for ab in all_ab:
            if layer_key not in ab:
                continue
            A, B = ab[layer_key]
            A_list.append(A)
            B_list.append(B)

        if len(A_list) < 2:
            # Fewer than 2 adapters have this layer — just use weighted single delta
            if len(A_list) == 1:
                idx = next(j for j, ab in enumerate(all_ab) if layer_key in ab)
                A, B = all_ab[idx][layer_key]
                delta = (B @ A).to(torch.bfloat16) * adapter_weights[idx]
                model_key = _peft_key_to_model_key(layer_key)
                combined_delta[model_key] = delta.cpu()
            continue

        # Compute reference bases via SVD of stacked matrices
        # A_stack: (T*r, n), B_stack: (m, T*r)
        A_stack = torch.cat(A_list, dim=0)  # already on device, float64
        B_stack = torch.cat(B_list, dim=1)

        Vh_A_ref = torch.linalg.svd(A_stack, full_matrices=False)[2]  # (T*r, n)
        U_B_ref = torch.linalg.svd(B_stack, full_matrices=False)[0]   # (m, T*r)

        # Compute aligned core matrices: M_i = U_B_ref^T @ B_i @ A_i @ Vh_A_ref^T
        M_list = []
        for A, B in zip(A_list, B_list):
            M = U_B_ref.T @ B @ A @ Vh_A_ref.T  # (T*r, T*r)
            M_list.append(M)

        # Merge core matrices using chosen sub-method
        if core_merge == "ties":
            M_merged = _merge_core_matrices_ties(M_list, adapter_weights, density=density)
        elif core_merge == "dare_ties":
            M_merged = _merge_core_matrices_dare_ties(
                M_list, adapter_weights, density=density,
                dare_density=dare_density, seed=seed,
            )
        elif core_merge == "tsv":
            M_merged = _merge_core_matrices_tsv(M_list, adapter_weights)
        else:  # "sum" (Task Arithmetic)
            M_merged = _merge_core_matrices_sum(M_list, adapter_weights)

        # Optionally isotropize: equalize singular values of merged core matrix
        if isotropize:
            U_m, S_m, Vh_m = torch.linalg.svd(M_merged, full_matrices=False)
            # All singular values become their mean, so U @ diag(c*I) @ Vh = c * (U @ Vh)
            M_merged = S_m.mean() * (U_m @ Vh_m)

        # Reconstruct full delta: delta_W = U_B_ref @ M_merged @ Vh_A_ref
        delta_W = U_B_ref @ M_merged @ Vh_A_ref  # (m, n) in float64

        model_key = _peft_key_to_model_key(layer_key)
        combined_delta[model_key] = delta_W.to(dtype=torch.bfloat16, device="cpu")

        if layer_idx % 500 == 0:
            rank = torch.linalg.matrix_rank(delta_W.float()).item()
            rank_stats.append(rank)

        # Free GPU memory periodically
        del A_stack, B_stack, Vh_A_ref, U_B_ref, M_list, M_merged, delta_W
        if layer_idx % 500 == 0 and device.type == "cuda":
            torch.cuda.empty_cache()

    if rank_stats:
        print(f"  Average rank of delta_W: {sum(rank_stats)/len(rank_stats):.1f}")
    print(f"  Core space merge complete: {len(combined_delta)} parameters changed")

    # Free adapter data
    del all_ab
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return combined_delta


def stack_via_peft(base_model_name, adapters):
    """
    Stack adapters by concatenating LoRA matrices (replicates PEFT combination_type="cat").
    Manually concatenates lora_A/lora_B from each adapter's safetensors to bypass PEFT's
    limitation of one adapter with target_parameters per model (affects MoE models).
    Preserves each adapter's rank-r subspace; effective rank = N * r.
    Returns a full merged model (merge_and_unload after concatenation).
    """
    import json
    import tempfile
    from safetensors.torch import load_file, save_file

    # Read config from first adapter to get r, alpha, etc.
    with open(os.path.join(adapters[0][0], "adapter_config.json")) as f:
        config = json.load(f)
    original_r = config["r"]
    original_alpha = config["lora_alpha"]
    scaling = original_alpha / original_r  # per-adapter scaling factor

    # Load all adapter state dicts from safetensors
    print("\n  Loading adapter state dicts for manual LoRA concatenation...")
    adapter_states = []
    for i, (path, weight) in enumerate(adapters):
        print(f"  Loading adapter {i}: {os.path.basename(path)} (weight={weight})")
        state = load_file(os.path.join(path, "adapter_model.safetensors"))
        adapter_states.append((state, weight))

    # Concatenate LoRA matrices (replicates PEFT _cat logic):
    #   lora_A: scaled by (weight * alpha/r), cat along dim 0  [r, in] -> [N*r, in]
    #   lora_B: unscaled, cat along dim 1                      [out, r] -> [out, N*r]
    # Combined adapter uses lora_alpha = new_r so merge scaling = 1.0
    first_state = adapter_states[0][0]
    concatenated = {}
    for key in first_state:
        if "lora_A" in key:
            concatenated[key] = torch.cat(
                [s[key] * w * scaling for s, w in adapter_states], dim=0
            )
        elif "lora_B" in key:
            concatenated[key] = torch.cat(
                [s[key] for s, _ in adapter_states], dim=1
            )
        else:
            concatenated[key] = first_state[key]

    new_r = original_r * len(adapters)
    config["r"] = new_r
    config["lora_alpha"] = new_r  # scaling = new_r/new_r = 1.0 (matches PEFT cat behavior)

    print(f"  Concatenated {len(adapters)} adapters -> rank {new_r} (from {len(adapters)} × {original_r})")

    # Save concatenated adapter to temp dir and load as single adapter
    with tempfile.TemporaryDirectory(dir="/home/ubuntu/tmp") as tmp:
        save_file(concatenated, os.path.join(tmp, "adapter_model.safetensors"))
        with open(os.path.join(tmp, "adapter_config.json"), "w") as f:
            json.dump(config, f)

        print("  Loading base model for LoRA stacking...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, tmp)

    stacked_rank = model.peft_config["default"].r
    print(f"  Stacked adapter rank: {stacked_rank}")

    print("  Merging stacked adapter into base model weights...")
    model = model.merge_and_unload()

    return model


# --- Run merge jobs ---

for job_idx, job in enumerate(MERGE_JOBS):
    if len(job) == 3:
        target_repo, adapters, config = job
    else:
        target_repo, adapters = job
        config = {}

    method = config.get("method", "linear")

    print(f"\n{'='*60}")
    print(f"Job {job_idx + 1}/{len(MERGE_JOBS)}: {target_repo}")
    print(f"Method: {method}")
    print(f"{'='*60}")

    adapter_paths = [a[0] for a in adapters]
    adapter_weights = [a[1] for a in adapters]

    # Auto-detect base model from first adapter
    cfg = PeftConfig.from_pretrained(adapter_paths[0])
    base_model = cfg.base_model_name_or_path
    print(f"Base model: {base_model}")

    for path, weight in adapters:
        print(f"  - {os.path.basename(path)} (weight={weight})")

    # Load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    save_dir = f"/home/ubuntu/merged_upload/{target_repo.split('/')[-1]}"
    os.makedirs(save_dir, exist_ok=True)

    if len(adapter_paths) == 1:
        # Single adapter: just load and merge
        print("\nLoading base model...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        print(f"Loading adapter: {os.path.basename(adapter_paths[0])}")
        model = PeftModel.from_pretrained(model, adapter_paths[0])
        print("Merging adapter into base model weights...")
        model = model.merge_and_unload()

    elif method == "stack":
        # Stack: concatenate LoRA matrices, then merge into base
        model = stack_via_peft(base_model, adapters)

    elif method == "core":
        # Core Space Merging: align LoRA subspaces via SVD, merge in core space
        core_merge = config.get("core_merge", "sum")
        isotropize = config.get("isotropize", True)
        density = config.get("density", 0.7)
        dare_density = config.get("dare_density", 0.7)
        seed = config.get("seed", 42)
        print(f"\nCore Space Merging (sub-method={core_merge}, isotropize={isotropize})")

        combined_delta = combine_core_space(
            adapters, core_merge=core_merge, isotropize=isotropize,
            density=density, dare_density=dare_density, seed=seed,
        )

        # Load base model and apply deltas
        print(f"Loading base model and applying {len(combined_delta)} core-space deltas...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model_state = model.state_dict()
        applied = 0
        for k, delta in combined_delta.items():
            if k in model_state:
                model_state[k] = model_state[k] + delta.to(model_state[k].device)
                applied += 1
            else:
                print(f"  WARNING: key {k} not found in model state dict, skipping")
        model.load_state_dict(model_state)
        print(f"  Applied {applied}/{len(combined_delta)} deltas to base model")
        del combined_delta, model_state
        gc.collect()
        torch.cuda.empty_cache()

    else:
        # Multi-adapter delta-based merge (linear or ties_dare)
        print("\nLoading base model for delta computation...")
        base_model_obj = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        base_state = {k: v.cpu() for k, v in base_model_obj.state_dict().items()}
        del base_model_obj
        gc.collect()
        torch.cuda.empty_cache()

        # Compute per-adapter deltas
        deltas = compute_deltas(base_model, base_state, adapters)

        # Combine deltas
        if method == "ties_dare":
            density = config.get("density", 0.7)
            msm = config.get("majority_sign_method", "total")
            seed = config.get("seed", 42)
            print(f"\nApplying TIES-DARE (density={density}, sign_method={msm}, seed={seed})...")
            combined_delta = combine_ties_dare(deltas, adapter_weights, density, msm, seed)
        else:
            print("\nApplying linear merge...")
            combined_delta = combine_linear(deltas, adapter_weights)

        del deltas
        gc.collect()

        # Reload base and apply combined deltas
        print(f"Applying combined deltas ({len(combined_delta)} parameters changed)...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model_state = model.state_dict()
        for k, delta in combined_delta.items():
            model_state[k] = model_state[k] + delta.to(model_state[k].device)
        model.load_state_dict(model_state)
        del base_state, combined_delta, model_state
        gc.collect()
        torch.cuda.empty_cache()

    # Save merged model
    print(f"Saving merged model to {save_dir}...")
    model.save_pretrained(save_dir, safe_serialization=True)
    tokenizer.save_pretrained(save_dir)

    # Upload to HuggingFace
    print(f"Uploading to {target_repo}...")
    api = HfApi()
    api.create_repo(target_repo, token=HF_TOKEN, exist_ok=True)
    api.upload_folder(folder_path=save_dir, repo_id=target_repo, token=HF_TOKEN)
    print(f"Done! {target_repo} pushed to HuggingFace.")

    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Deleting merged folder {save_dir}...")
    shutil.rmtree(save_dir)
    print(f"Deleted {save_dir}.")

print(f"\nAll {len(MERGE_JOBS)} jobs complete.")
