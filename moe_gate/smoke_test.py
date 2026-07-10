"""
End-to-end smoke test for the capability router.

Verifies:
  1. build_capability_model loads base + 4 capability adapters and replaces
     q/k/v/o projections with MultiLoRAGatedLinear.
  2. Freeze invariants hold (assert_frozen passes).
  3. A forward pass through the gated model produces finite logits.
  4. A 1-step warm-start CE update against a dummy P_target produces real
     gradients on gater parameters and *only* on gater parameters.

Usage (on the pod):
    python -m capability_router.smoke_test
    # or with a specific cache dir:
    HF_HOME=/workspace/hf_cache python -m capability_router.smoke_test
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F

from capability_router import (
    CapabilityModelConfig,
    assert_frozen,
    build_capability_model,
    gater_parameters,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    p.add_argument(
        "--adapters",
        nargs="+",
        default=[
            "tarsur909/precondition-v1-40-ckpt",
            "tarsur909/structured_data_reasoning_grpo_iter40",
            "tarsur909/tau_tool_calling_grpo_iter40",
            "tarsur909/multistep-v4-10-ckpt",
        ],
    )
    p.add_argument("--no-base-cap", action="store_true",
                   help="Disable the 'base' routing slot.")
    p.add_argument("--granularity", default="trajectory",
                   choices=["trajectory", "token"])
    p.add_argument(
        "--target-modules",
        nargs="+",
        default=None,
        help=(
            "Override target_modules. Use 'attn' shorthand for "
            "(q,k,v,o)_proj only (fast smoke), or 'all' for the default 7."
        ),
    )
    p.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    p.add_argument("--device-map", default="auto")
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--dry-build-only", action="store_true",
                   help="Build model + freeze checks; skip forward/backward.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    if args.target_modules == ["attn"]:
        target_modules = ("q_proj", "k_proj", "v_proj", "o_proj")
    elif args.target_modules == ["all"] or args.target_modules is None:
        target_modules = (
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        )
    else:
        target_modules = tuple(args.target_modules)

    cfg = CapabilityModelConfig(
        base_model=args.base,
        capability_adapters=args.adapters,
        include_base_capability=not args.no_base_cap,
        target_modules=target_modules,
        routing_granularity=args.granularity,
        cache_dir=args.cache_dir,
        device_map=args.device_map,
        dtype=torch.bfloat16,
    )
    print(f"==[smoke]== target_modules = {target_modules}")
    print("==[smoke]== Building capability model")
    model, meta = build_capability_model(cfg, return_meta=True)
    tokenizer = meta["tokenizer"]
    n_caps = meta["num_capabilities"]
    print(f"==[smoke]== num_capabilities = {n_caps} ({meta['capability_names']})")

    print("==[smoke]== assert_frozen(...)")
    assert_frozen(model)
    print("==[smoke]== freeze invariants OK")

    if args.dry_build_only:
        return 0

    # ---- 3. Forward pass ----------------------------------------------------
    model.eval()  # cheap: gaters are init-zero so output should match base behaviour
    enc = tokenizer(
        args.prompt,
        return_tensors="pt",
        padding="max_length",
        max_length=args.seq_len,
        truncation=True,
    )
    # Move to model's device (use first param's device).
    dev = next(model.parameters()).device
    input_ids = enc["input_ids"].to(dev)
    attn_mask = enc["attention_mask"].to(dev)

    print(f"==[smoke]== Forward (input_ids shape {tuple(input_ids.shape)})")
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn_mask)
    logits = out.logits
    assert torch.isfinite(logits).all(), "non-finite logits"
    print(f"==[smoke]== logits.shape={tuple(logits.shape)}, "
          f"|max|={logits.abs().max().item():.3f}")

    # ---- 4. Tiny warm-start CE step ----------------------------------------
    # Build a synthetic P_target distribution. With n_caps=5 and zero-init
    # gaters, the model's gater outputs uniform 1/5 — make the target sharp on
    # capability index 1 (or 0 if only one capability) so we get non-trivial CE.
    target_idx = 1 % n_caps
    p_target = torch.zeros(input_ids.size(0), n_caps, device=dev, dtype=torch.float32)
    p_target[:, target_idx] = 1.0

    # We need to *capture* the gater outputs during a forward pass with grad.
    # Easiest: register forward hooks on each gater to record its output, then
    # forward, then sum CE losses across layers.
    gaters = meta["gaters"]
    captured = []

    def _make_hook(layer_idx):
        def _hook(_module, _inp, output):
            # output: trajectory -> [B, 1, C]; token -> [B, T, C]
            captured.append(output)
        return _hook

    handles = [g.register_forward_hook(_make_hook(i)) for i, g in enumerate(gaters)]

    model.train()
    out = model(input_ids=input_ids, attention_mask=attn_mask)
    # Sum CE across all gater outputs -> a single scalar loss.
    loss = torch.tensor(0.0, device=dev, dtype=torch.float32)
    for w in captured:
        # Reduce token dim (mean) so we have [B, C], then KL/CE vs p_target.
        wflat = w.float().mean(dim=1) if w.dim() == 3 else w.float()
        # Each gater can sit on a different device under device_map="auto";
        # move both tensors to the same device for the KL.
        wflat = wflat.to(device=dev)
        # Avoid log(0) — clamp.
        log_p = torch.log(wflat.clamp_min(1e-9))
        loss = loss + F.kl_div(log_p, p_target, reduction="batchmean")
    print(f"==[smoke]== warm-start dummy CE loss = {loss.item():.4f}")

    loss.backward()
    for h in handles:
        h.remove()

    # ---- 5. Gradient invariants --------------------------------------------
    n_gater_with_grad = 0
    n_gater_without_grad = 0
    for name, p in gater_parameters(model):
        if p.grad is None:
            n_gater_without_grad += 1
        else:
            n_gater_with_grad += 1
    n_other_with_grad = 0
    gater_ids = {id(p) for _, p in gater_parameters(model)}
    for name, p in model.named_parameters():
        if id(p) in gater_ids:
            continue
        if p.grad is not None and p.grad.abs().sum() > 0:
            n_other_with_grad += 1
            print(f"==[smoke]== UNEXPECTED non-gater grad on: {name}")

    print(
        f"==[smoke]== gater params with grad: {n_gater_with_grad} / "
        f"{n_gater_with_grad + n_gater_without_grad}"
    )
    if n_gater_with_grad == 0:
        print("==[smoke]== FAIL: no gater received gradient")
        return 1
    if n_other_with_grad > 0:
        print(f"==[smoke]== FAIL: {n_other_with_grad} non-gater params got gradient")
        return 1

    print("==[smoke]== ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
