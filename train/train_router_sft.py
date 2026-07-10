#!/usr/bin/env python3
"""Distributed router-only SFT trainer for the routed Qwen3MoE model.

Trains ``cap_gate`` (492K params, ~5.9 MB optimizer state) on successful
trajectories with a Switch-style top-1 MoE objective:

    L = L_CE(response_tokens) + lambda_LB * L_LB

Everything except cap_gate is frozen. Forward uses hard top-1 cap routing
with multiplicative gating, identical to inference.

Data format (parquet, one row per assistant turn or full rollout):
  - reward: float (filtered by --reward-threshold)
  - messages_full: JSON List[Dict] with role={system,user,assistant,tool},
    where the FINAL assistant turn(s) are what we train on
  - tools: JSON tool schemas (optional)
  - capability: source capability (kept for diagnostics)

Usage:
    # Single-GPU smoke test
    python train_router_sft.py \\
        --rollouts /workspace/games_outputs/rollouts_full.parquet \\
        --routed-extras /workspace/loras/routed_extras.safetensors \\
        --output-dir /workspace/games_outputs/router_sft

    # 4-GPU
    torchrun --nproc_per_node=4 train_router_sft.py [...]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F
from torch.optim import AdamW
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from dist_utils import (
    dist_pre_init,
    dist_nccl_init,
    dist_cleanup,
    is_main_rank,
    barrier,
    shard_batches,
    allreduce_coalesced_grads,
    allreduce_scalars,
    suppress_print,
)
from vllm_routed_qwen3.hf_modeling import (
    load_routed_qwen3_hf,
    freeze_except_cap_gate,
    cap_gate_parameters,
    cap_gate_state_dict,
    load_balancing_loss,
    CapLogitsCollector,
    set_forced_routing,
    clear_forced_routing,
    get_routed_blocks,
    register_block_input_capture,
    release_block_input_capture,
)

try:
    import wandb
except Exception:
    wandb = None


# Maps step2's `capability` column to the cap_id used by the routed model's
# LoRA bank slots (matches CAPS in pack_routed_extras.py:
#   cap_id=0: base, 1: multistep, 2: precondition, 3: sdr, 4: tau_tool).
CAPABILITY_TO_CAP_ID = {
    "base": 0,
    "tarsur909/multistep-v4-10-ckpt": 1,
    "tarsur909/precondition-v1-40-ckpt": 2,
    "tarsur909/structured_data_reasoning_grpo_iter40": 3,
    "tarsur909/tau_tool_calling_grpo_iter40": 4,
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Router-only SFT for routed Qwen3MoE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument("--rollouts", type=str, required=True,
                   help="Parquet with messages_full + reward + (optional) tools")
    p.add_argument("--reward-threshold", type=float, default=1.0)
    p.add_argument("--max-seq-length", type=int, default=8192)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--shuffle-seed", type=int, default=42)
    p.add_argument("--turns-per-row", type=str, default="last",
                   choices=["last", "all"],
                   help="'last': train on the final assistant turn per rollout; "
                        "'all': train on every assistant turn (more data, more "
                        "redundancy across turns)")

    # Model
    p.add_argument("--base-model", type=str,
                   default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    p.add_argument("--routed-extras", type=str, required=True)
    p.add_argument("--num-capabilities", type=int, default=5)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--cap-gate-base-bias", type=float, default=None,
                   help="Override cap_gate.bias[0] post-load. Default uses the "
                        "value packed in routed_extras (typically 5.0). Lower "
                        "values (e.g. 1.0) give the router more headroom to "
                        "diversify but degrade init CE.")

    # Optim
    p.add_argument("--lr", type=float, default=5e-4,
                   help="cap_gate LR (small head -> can use higher LR)")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--num-epochs", type=int, default=3)
    p.add_argument("--mini-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction, default=True,
        help="Enable gradient checkpointing. Disable with "
             "--no-gradient-checkpointing when only cap_gate is trainable "
             "(activations needed for backward are minimal).",
    )

    # Aux losses
    p.add_argument("--lb-coef", type=float, default=0.001)
    p.add_argument("--label-coef", type=float, default=0.0,
                   help="Cap-label cross-entropy coefficient. Forces the "
                        "router to predict each trajectory's winning cap on "
                        "response tokens at every layer. Defaults to 0 "
                        "(disabled). Recommended ~0.5 to drive routing "
                        "diversification.")

    # Checkpointing & logging
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--wandb-project", type=str, default="routed-router-sft")
    p.add_argument("--no-wandb", action="store_true", default=False)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading and tokenization
# ---------------------------------------------------------------------------

def _build_one_sample(
    tokenizer,
    prompt_msgs: list,
    response_msg: dict,
    tools=None,
    enable_thinking: bool = False,
) -> Tuple[torch.Tensor, int, int]:
    """Tokenize prompt + assistant response. Returns (full_ids, prompt_len, action_len).

    Uses the chat template with ``add_generation_prompt=True`` for the prompt
    portion to match how the assistant turn was generated, then concatenates
    with the response message tokenized via the same template (no
    generation prompt). Action length = full - prompt.
    """
    kwargs_p = dict(
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )
    kwargs_full = dict(
        add_generation_prompt=False,
        tokenize=True,
        return_tensors="pt",
    )
    if tools:
        kwargs_p["tools"] = tools
        kwargs_full["tools"] = tools

    prompt_ids = tokenizer.apply_chat_template(prompt_msgs, **kwargs_p)[0]
    full_ids = tokenizer.apply_chat_template(
        prompt_msgs + [response_msg], **kwargs_full,
    )[0]

    pl = int(prompt_ids.shape[0])
    al = int(full_ids.shape[0]) - pl
    return full_ids, pl, al


def load_trajectories(
    parquet_path: str,
    reward_threshold: float,
    tokenizer,
    max_seq_length: int,
    max_samples: int = None,
    shuffle_seed: int = 42,
    turns_per_row: str = "last",
) -> List[Tuple[torch.Tensor, int, int, str]]:
    """Returns list of (ids, prompt_len, action_len, capability_str, cap_id).

    cap_id is the int label used for the routed model's LoRA bank slot
    (0=base, 1=multistep, 2=precondition, 3=sdr, 4=tau_tool). Trajectories
    whose ``capability`` field is missing, NaN, or not in
    ``CAPABILITY_TO_CAP_ID`` are kept with ``cap_id = -1`` (sentinel for
    "unlabeled"); they still contribute to CE + load-balancing losses
    but are masked out of the optional label-CE term in
    ``build_label_targets``.
    """
    df = pd.read_parquet(parquet_path)
    if "reward" not in df.columns:
        raise ValueError(f"Parquet missing 'reward' column: {df.columns.tolist()}")
    df = df[df["reward"] >= reward_threshold].reset_index(drop=True)

    rng = random.Random(shuffle_seed)
    indices = list(range(len(df)))
    rng.shuffle(indices)

    samples = []
    n_unlabeled = 0
    has_cap_col = "capability" in df.columns
    for i in indices:
        row = df.iloc[i]
        cap_raw = row.get("capability", None) if has_cap_col else None
        if cap_raw is None or (
            isinstance(cap_raw, float) and pd.isna(cap_raw)
        ):
            cap = "unlabeled"
        else:
            cap = str(cap_raw)
        if cap in CAPABILITY_TO_CAP_ID:
            cap_id = CAPABILITY_TO_CAP_ID[cap]
        else:
            cap_id = -1
            n_unlabeled += 1
        msg_field = row.get("messages_full", None)
        if msg_field is None or (isinstance(msg_field, float) and pd.isna(msg_field)):
            continue
        try:
            messages = json.loads(msg_field)
        except Exception:
            continue
        tools = None
        if "tools" in df.columns:
            tools_field = row.get("tools", None)
            if tools_field is not None and not (
                isinstance(tools_field, float) and pd.isna(tools_field)
            ):
                try:
                    tools = json.loads(tools_field)
                except Exception:
                    tools = None

        # Find assistant turn indices
        a_indices = [
            j for j, m in enumerate(messages)
            if m.get("role") == "assistant"
        ]
        if not a_indices:
            continue
        chosen = a_indices if turns_per_row == "all" else [a_indices[-1]]
        for last_a in chosen:
            prompt_msgs = messages[:last_a]
            response_msg = messages[last_a]
            try:
                ids, pl, al = _build_one_sample(
                    tokenizer, prompt_msgs, response_msg, tools=tools,
                )
            except Exception:
                continue
            if al <= 0 or ids.shape[0] > max_seq_length:
                continue
            samples.append((ids, pl, al, cap, cap_id))

        if max_samples is not None and len(samples) >= max_samples:
            break

    if max_samples is not None:
        samples = samples[:max_samples]
    return samples


def prepare_batches(
    seqs: List[torch.Tensor],
    mini_batch_size: int,
) -> List[List[int]]:
    indices = sorted(range(len(seqs)), key=lambda i: seqs[i].shape[0])
    return [
        indices[i:i + mini_batch_size]
        for i in range(0, len(indices), mini_batch_size)
    ]


def pad_batch(
    indices: List[int],
    seqs: List[torch.Tensor],
    pad_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    seqs_b = [seqs[i] for i in indices]
    max_len = max(s.shape[0] for s in seqs_b)
    B = len(seqs_b)
    ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((B, max_len), dtype=torch.long)
    for i, s in enumerate(seqs_b):
        L = s.shape[0]
        ids[i, :L] = s
        attn[i, :L] = 1
    return ids, attn


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def ce_loss_on_response(
    hidden_states: torch.Tensor, # [B, S, hidden]
    lm_head: torch.nn.Module,    # nn.Linear(hidden, vocab); applied lazily
    input_ids: torch.Tensor,     # [B, S]
    prompt_lens: List[int],
    action_lens: List[int],
) -> Tuple[torch.Tensor, int]:
    """Token-mean CE on response positions only. Returns (loss, n_tokens).

    Memory-efficient: receives hidden states (NOT logits) and applies
    ``lm_head`` per-sample only to the response-token slice. Peak logits
    tensor is ``[al, V]`` (al ≤ ~1700) instead of ``[B, S, V]`` (S can be
    32K). At V=152K bf16 that's a ~9 GB difference at S=32K — the
    difference between fitting on an 80 GB H100 and OOM.
    """
    B, S, H = hidden_states.shape
    if S < 2:
        return torch.tensor(
            0.0, device=hidden_states.device, requires_grad=True,
        ), 0

    # Targets are the next token; we predict tokens [1, S) from positions
    # [0, S-1). Per-sample response slice is [pl-1, pl-1+al) on logits
    # (= positions that predict tokens [pl, pl+al)).
    total_ce = torch.tensor(
        0.0, device=hidden_states.device, dtype=torch.float32,
    )
    n_tokens = 0
    for b in range(B):
        pl = prompt_lens[b]
        al = action_lens[b]
        start = pl - 1
        end = pl - 1 + al
        if end > S - 1 or start < 0:
            continue
        # Slice hidden states FIRST, then apply lm_head only here.
        # F.cross_entropy is a fused (log_softmax + nll) so it only
        # allocates an [al, V] fp32 tensor. Both .float()s above and
        # the matmul below stay bounded at al × V.
        resp_hidden = hidden_states[b, start:end]          # [al, H]
        resp_logits = lm_head(resp_hidden).float()         # [al, V]
        resp_targets = input_ids[b, start + 1:end + 1]     # [al]
        ce = F.cross_entropy(resp_logits, resp_targets, reduction="sum")
        total_ce = total_ce + ce
        n_tokens += al
    if n_tokens == 0:
        return torch.tensor(
            0.0, device=hidden_states.device, requires_grad=True,
        ), 0
    return total_ce / n_tokens, n_tokens


def label_loss_for_batch(
    cap_logits_list: List[torch.Tensor],
    cap_id_per_token: torch.Tensor,    # [B*S], int label per token
    response_mask: torch.Tensor,        # [B*S], bool, True for response tokens
) -> torch.Tensor:
    """Cross-entropy on cap_logits vs winning cap_id, masked to response tokens.

    Averaged across response tokens, then mean over routed layers.
    """
    if not cap_logits_list:
        return torch.tensor(0.0, device=cap_id_per_token.device)
    n_response = response_mask.float().sum().clamp(min=1.0)
    losses = []
    for cap_logits in cap_logits_list:
        per_token = F.cross_entropy(
            cap_logits.float(), cap_id_per_token, reduction="none",
        )
        masked = per_token * response_mask.to(per_token.dtype)
        losses.append(masked.sum() / n_response)
    return torch.stack(losses).mean()


def build_label_targets(
    batch_indices: List[int],
    cap_ids: List[int],
    prompt_lens: List[int],
    action_lens: List[int],
    pad_to_len: int,
    device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct ``cap_id_per_token`` and ``response_mask`` for a batch.

    Both tensors are [B*pad_to_len] flat shape, matching how the routed
    block sees hidden_states ([B, S, d] -> [B*S, d]).

    Samples with ``cap_ids[sample_idx] == -1`` are treated as "unlabeled":
    their target is set to 0 (placeholder, never read) and their entries
    in ``response_mask`` stay False so the label-CE term in
    ``label_loss_for_batch`` skips them. Other losses (CE, LB) are
    unaffected — they don't use ``response_mask``.
    """
    B = len(batch_indices)
    cap_id_pt = torch.zeros(B, pad_to_len, dtype=torch.long, device=device)
    resp_mask = torch.zeros(B, pad_to_len, dtype=torch.bool, device=device)
    for b_local, sample_idx in enumerate(batch_indices):
        cid = cap_ids[sample_idx]
        # cap_id=-1 is the "unlabeled" sentinel: leave target as 0 (placeholder)
        # and leave resp_mask False so this sample's response tokens don't
        # contribute to the label-CE loss.
        if cid >= 0:
            cap_id_pt[b_local].fill_(cid)
            pl = prompt_lens[sample_idx]
            al = action_lens[sample_idx]
            end = min(pl + al, pad_to_len)
            if pl < end:
                resp_mask[b_local, pl:end] = True
    return cap_id_pt.view(-1), resp_mask.view(-1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    rank, world_size, local_rank = dist_pre_init()
    if not is_main_rank():
        suppress_print()

    print(f"[rank {rank}/{world_size}] starting; local_rank={local_rank}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Data
    print(f"[rank {rank}] loading rollouts from {args.rollouts}")
    samples = load_trajectories(
        args.rollouts,
        args.reward_threshold,
        tokenizer,
        args.max_seq_length,
        args.max_samples,
        args.shuffle_seed,
        args.turns_per_row,
    )
    n_samples = len(samples)
    if n_samples == 0:
        raise RuntimeError(
            f"No samples after filtering (reward >= {args.reward_threshold}). "
            f"Check the parquet path and reward distribution."
        )
    cap_counts = {}
    for _, _, _, cap, _ in samples:
        cap_counts[cap] = cap_counts.get(cap, 0) + 1
    print(f"[rank {rank}] {n_samples} samples; cap counts:")
    for cap, n in sorted(cap_counts.items()):
        print(f"  {cap}: {n}")

    seqs = [s[0] for s in samples]
    prompt_lens = [s[1] for s in samples]
    action_lens = [s[2] for s in samples]
    cap_ids = [s[4] for s in samples]

    # Model
    print(f"[rank {rank}] loading routed model from {args.base_model}")
    model, n_swapped = load_routed_qwen3_hf(
        base_model=args.base_model,
        routed_extras_path=args.routed_extras,
        num_capabilities=args.num_capabilities,
        lora_rank=args.lora_rank,
        dtype=torch.bfloat16,
        attn_impl="sdpa",
        device=f"cuda:{local_rank}",
    )
    print(f"[rank {rank}] swapped {n_swapped} MoE blocks to routed")

    n_train, n_total = freeze_except_cap_gate(model)
    print(f"[rank {rank}] trainable: {n_train:,} / {n_total:,} "
          f"({100*n_train/n_total:.4f}%)")

    # Override cap_gate.bias[0] if requested. After load_routed_extras the
    # bias was set from the packed safetensors (typically 5.0 → cap_gate_w
    # ≈ 0.974). A lower bias (e.g. 1.0 → 0.404) reduces the base privilege
    # so the router has more flexibility to diversify, at the cost of
    # higher initial CE.
    if args.cap_gate_base_bias is not None:
        from vllm_routed_qwen3.hf_modeling import CapabilityRoutedMoeBlockHF
        n_overridden = 0
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, CapabilityRoutedMoeBlockHF):
                    module.cap_gate.bias.zero_()
                    module.cap_gate.bias[0] = args.cap_gate_base_bias
                    n_overridden += 1
        print(f"[rank {rank}] override cap_gate.bias[0] = "
              f"{args.cap_gate_base_bias} on {n_overridden} blocks")

    # Model is already on cuda:{local_rank} via device_map in load_routed_qwen3_hf.
    # Gradient checkpointing is intentionally NOT used in the default
    # training path: with banks + base frozen and cap_gate applied
    # externally on captured block inputs (see training loop below), the
    # model.model() forward runs in ``torch.no_grad()`` and saves zero
    # autograd activations. ckpt would be a no-op there. The flag is kept
    # for backward compatibility / experiments with the unforced (Switch-
    # style) regime.
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        model.enable_input_require_grads()
    model.train()

    # Cache the routed blocks once. Used both to register input-capture
    # pre-hooks and to apply cap_gate externally on those captured inputs.
    routed_blocks = get_routed_blocks(model)
    if is_main_rank():
        print(f"[rank {rank}] {len(routed_blocks)} routed MoE blocks")

    # Phase 2 distributed init
    dist_nccl_init()

    # Optimizer (cap_gate only)
    trainable_params = cap_gate_parameters(model)
    optim = AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Scheduler
    batches_per_epoch = math.ceil(n_samples / args.mini_batch_size)
    optim_steps_per_epoch = math.ceil(
        batches_per_epoch / args.gradient_accumulation_steps,
    )
    total_optim_steps = optim_steps_per_epoch * args.num_epochs
    warmup_steps = max(1, int(args.warmup_ratio * total_optim_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_optim_steps - warmup_steps)
        return max(0.0, 1.0 - progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    collector = CapLogitsCollector(model)

    # WandB
    if is_main_rank() and wandb is not None and not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=datetime.now().strftime("router-sft-%Y%m%d-%H%M%S"),
        )

    # Output dir
    if args.output_dir is None:
        args.output_dir = f"./router_sft_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if is_main_rank():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(args.output_dir) / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2)
    barrier()

    print(f"[rank {rank}] starting training: {args.num_epochs} epochs, "
          f"{batches_per_epoch} batches/epoch, "
          f"{optim_steps_per_epoch} optim steps/epoch, "
          f"warmup={warmup_steps}, total_optim_steps={total_optim_steps}")

    global_step = 0
    t_start = time.time()

    for epoch in range(args.num_epochs):
        all_batches = prepare_batches(seqs, args.mini_batch_size)
        # Stable shard: each rank gets every world_size-th batch.
        my_batches, total = shard_batches(all_batches, rank, world_size)

        epoch_loss = 0.0
        epoch_ce = 0.0
        epoch_lb = 0.0
        epoch_lab = 0.0
        epoch_n_tokens = 0
        epoch_steps = 0

        accum_count = 0

        pbar = tqdm(
            my_batches,
            desc=f"epoch {epoch}",
            disable=not is_main_rank(),
        )
        for batch_idx, batch_indices in enumerate(pbar):
            ids, attn = pad_batch(
                batch_indices, seqs, tokenizer.pad_token_id,
            )
            ids = ids.to(f"cuda:{local_rank}")
            attn = attn.to(f"cuda:{local_rank}")
            pls = [prompt_lens[i] for i in batch_indices]
            als = [action_lens[i] for i in batch_indices]

            collector.reset()

            # Per-token cap label + response mask for label supervision.
            # ``cap_id_per_token`` and ``response_mask`` are flat ([B*S])
            # to match how routed blocks see hidden_states after view.
            B_local, S_pad = ids.shape
            cap_id_per_token, response_mask = build_label_targets(
                batch_indices, cap_ids, prompt_lens, action_lens,
                pad_to_len=S_pad, device=ids.device,
            )

            # ---- Memory-efficient forced-routing forward ----
            #
            # Banks + base are frozen; the only trainable parameter is
            # cap_gate (~492K params). We exploit this to run the LM
            # forward in ``torch.no_grad()`` (no autograd graph saved)
            # while still training cap_gate via L_router:
            #
            #   1. Pre-hooks on each routed block capture the block's
            #      pre-MoE input ``hidden_states`` (post-attention
            #      LayerNorm output). These tensors are bf16, no grad_fn.
            #   2. ``set_forced_routing`` makes the block use the per-
            #      token label as its routing decision (instead of
            #      cap_gate's argmax) — train forward shape == eval
            #      forward shape.
            #   3. ``model.model()`` runs in ``torch.no_grad()``: each
            #      block's output is correct (forced routing produces the
            #      right cap-z bank application; weights are frozen) but
            #      no activations are saved.
            #   4. ``ce_loss_on_response`` is also in no_grad — it's a
            #      diagnostic, not a training signal (cap_gate isn't in
            #      its gradient path with forced routing + no
            #      multiplicative gate).
            #   5. AFTER model.model() returns, cap_gate is applied
            #      *externally* (with grad enabled) to each captured
            #      hidden_states. The captured tensor acts as a
            #      grad-leaf; cap_gate's weights get the L_router
            #      gradient cleanly with no checkpointing involved.
            #
            # This decouples cap_gate's training compute from the LM
            # forward's autograd graph entirely, avoiding the reentrant-
            # ckpt-drops-attribute-tensor failure and dramatically
            # reducing peak memory.
            captured, capture_handles = register_block_input_capture(
                routed_blocks,
            )
            set_forced_routing(model, cap_id_per_token)

            try:
                with torch.no_grad(), torch.autocast(
                    "cuda", dtype=torch.bfloat16,
                ):
                    inner_out = model.model(
                        input_ids=ids, attention_mask=attn,
                    )
                    hidden_states = inner_out.last_hidden_state  # [B, S, H]
                    ce, n_tokens = ce_loss_on_response(
                        hidden_states, model.lm_head, ids, pls, als,
                    )
            finally:
                clear_forced_routing(model)
                release_block_input_capture(capture_handles)

            # External cap_gate forward (with grad). Each captured
            # tensor has shape [B, S, H]; reshape to [B*S, H] to match
            # the per-token cap_id_per_token / response_mask layout.
            with torch.autocast("cuda", dtype=torch.bfloat16):
                cap_logits_list = []
                for blk, h_in in zip(routed_blocks, captured):
                    h2d = h_in.view(-1, h_in.shape[-1])
                    cl = blk.cap_gate(
                        h2d.to(blk.cap_gate.weight.dtype),
                    )  # [B*S, C]
                    cap_logits_list.append(cl)

                lb = load_balancing_loss(cap_logits_list)
                if args.label_coef > 0:
                    lab = label_loss_for_batch(
                        cap_logits_list,
                        cap_id_per_token, response_mask,
                    )
                else:
                    lab = torch.tensor(0.0, device=ids.device)

                # ce was computed in no_grad — already detached. Loss
                # gradient flows only through ``lab`` (and ``lb`` if
                # lb_coef > 0). cap_gate is the only trainable param
                # along these paths.
                loss = (
                    ce.detach()
                    + args.lb_coef * lb
                    + args.label_coef * lab
                )

            (loss / args.gradient_accumulation_steps).backward()

            # Drop references to captured tensors so memory frees ASAP.
            captured.clear()

            epoch_loss += float(loss.detach().item())
            epoch_ce += float(ce.detach().item())
            epoch_lb += float(lb.detach().item()) if isinstance(lb, torch.Tensor) else 0.0
            epoch_lab += float(lab.detach().item()) if isinstance(lab, torch.Tensor) else 0.0
            epoch_n_tokens += int(n_tokens)
            epoch_steps += 1
            accum_count += 1

            # Optimizer step
            do_step = (
                accum_count == args.gradient_accumulation_steps
                or batch_idx == len(my_batches) - 1
            )
            if do_step:
                allreduce_coalesced_grads(trainable_params)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_params, args.max_grad_norm,
                )
                optim.step()
                scheduler.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1
                accum_count = 0

                # Free fragmented allocator memory periodically. Correctness-
                # neutral; costs a small bit of throughput. Helps when the
                # 80 GB GPU sits at 99% — without it, allocator fragmentation
                # can OOM mid-backward on a slightly longer sequence.
                if global_step % 50 == 0:
                    torch.cuda.empty_cache()

                if is_main_rank() and global_step % args.log_every == 0:
                    avg_loss = epoch_loss / max(1, epoch_steps)
                    log = {
                        "step": global_step,
                        "epoch": epoch,
                        "loss": avg_loss,
                        "ce_loss": epoch_ce / max(1, epoch_steps),
                        "lb_loss": epoch_lb / max(1, epoch_steps),
                        "label_loss": epoch_lab / max(1, epoch_steps),
                        "lr": float(scheduler.get_last_lr()[0]),
                        "grad_norm": float(grad_norm.item())
                                       if isinstance(grad_norm, torch.Tensor)
                                       else float(grad_norm),
                        "tokens": epoch_n_tokens,
                        "elapsed_s": time.time() - t_start,
                    }
                    print(f"[step {global_step}] " +
                          " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                   for k, v in log.items()))
                    if wandb is not None and not args.no_wandb:
                        wandb.log(log)

            pbar.set_postfix(
                loss=f"{epoch_loss/max(1,epoch_steps):.4f}",
                ce=f"{epoch_ce/max(1,epoch_steps):.4f}",
            )

        # End-of-epoch reduce + save
        red = allreduce_scalars({
            "epoch_loss_sum": epoch_loss,
            "epoch_steps": float(epoch_steps),
            "epoch_n_tokens": float(epoch_n_tokens),
        })
        if is_main_rank():
            avg = red["epoch_loss_sum"] / max(1, red["epoch_steps"])
            print(f"[epoch {epoch} done] avg_loss={avg:.4f} "
                  f"total_tokens={int(red['epoch_n_tokens'])} "
                  f"elapsed={time.time()-t_start:.0f}s")

        if is_main_rank() and (epoch + 1) % args.save_every == 0:
            ckpt = {
                "epoch": epoch,
                "step": global_step,
                "args": vars(args),
                "state": cap_gate_state_dict(model),
            }
            ckpt_path = Path(args.output_dir) / f"cap_gate_epoch_{epoch}.pt"
            torch.save(ckpt, ckpt_path)
            print(f"saved {ckpt_path}")
        barrier()

    # Final save
    if is_main_rank():
        ckpt = {
            "epoch": args.num_epochs - 1,
            "step": global_step,
            "args": vars(args),
            "state": cap_gate_state_dict(model),
        }
        torch.save(ckpt, Path(args.output_dir) / "cap_gate_final.pt")
        print(f"saved final cap_gate at {args.output_dir}/cap_gate_final.pt")
        if wandb is not None and not args.no_wandb:
            wandb.finish()

    collector.remove()
    dist_cleanup()


if __name__ == "__main__":
    main()
