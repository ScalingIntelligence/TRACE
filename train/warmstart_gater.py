#!/usr/bin/env python3
"""Step 5: warm-start the capability router via supervised loss.

Two scopes (selected via ``--routing-scope``):

  * ``per_layer`` (default): per-block softmax routers
    (``LayerCapabilityGater``), trained against Step 4's soft target via
    per-layer KL divergence. Saves ``gater_state.pt``.

  * ``global``: a single ``TrajectoryGlobalRouter`` classifier head over
    last-token-pooled features of the FROZEN base model, trained against
    ``argmax(rewards)`` via cross-entropy. At inference the argmax is
    broadcast as a one-hot routing decision to every block. Saves
    ``router_state.pt`` (under the same ``--ckpt-out`` path).

Both scopes consume the routing targets parquet/jsonl produced by
``step4_make_routing_targets.py`` (one row per seed with fields
``capability_names``, ``rewards``, ``p_target``).

Usage:

    # Per-block (existing):
    python step5_warmstart_gater.py \\
        --targets /tmp/games/routing_targets.parquet \\
        --ckpt-out /tmp/games/gater_warmstart.pt \\
        --batch-size 2 --epochs 3 --lr 1e-3

    # Global (new):
    python step5_warmstart_gater.py --routing-scope global \\
        --targets /tmp/games/routing_targets.parquet \\
        --ckpt-out /tmp/games/global_router_warmstart.pt \\
        --batch-size 2 --epochs 3 --lr 1e-3
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from capability_router import (
    CapabilityModelConfig,
    TrajectoryGlobalRouter,
    assert_frozen,
    build_capability_model,
    gater_parameters,
)
from tau2_bench_game import Tau2BenchGame
from tau2_bench_synthetic_game import Tau2BenchSyntheticGame


# ---------------------------------------------------------------------------
# Shared data utilities
# ---------------------------------------------------------------------------

def _load_targets(path: Path) -> List[Dict]:
    if path.suffix == ".jsonl":
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    import pandas as pd  # type: ignore
    return pd.read_parquet(path).to_dict(orient="records")


def _build_prompt_for_seed(
    seed: int, domain: Optional[str]
) -> Tuple[List[Dict[str, str]], List[Dict]]:
    """Synthetic-game prompt builder (existing per-layer pipeline).

    Returns ``(messages, tools)``. Tools are returned so the caller can pass
    them through ``apply_chat_template(..., tools=...)``.
    """
    game = Tau2BenchSyntheticGame(max_steps=1, user_client=None, domain=domain)
    game.reset(seed)
    messages = [
        {"role": "system", "content": game.get_system_prompt()},
        *game.get_messages(),
    ]
    if hasattr(game, "get_tool_schemas_compact"):
        tools = game.get_tool_schemas_compact()
    else:
        tools = game.get_tool_schemas()
    return messages, tools


def _build_prompt_tau2bench(
    domain: str,
    first_messages: List[Dict],
) -> Tuple[List[Dict[str, str]], List[Dict]]:
    """tau2bench prompt builder — byte-identical to tau2-bench's actual eval.

    Reproduces the EXACT prompt the agent saw on its first turn during
    Step 2 rollouts: same system prompt (= tau2-bench's LLMAgent.system_prompt
    for the resolved domain) + same FULL openai-format tool schemas (=
    tau2-bench's env.get_tools()'s openai_schema, NOT the compact stripped
    variant) prepended to the captured non-system messages.

    No user_client / vLLM needed at training time — the captured
    first-turn messages already include both the agent's fixed
    DEFAULT_FIRST_AGENT_MESSAGE greeting and the user simulator's
    response, persisted by Step 2 at rollout time.

    ``domain`` must be 'airline' or 'retail' (NOT mixed/None) — Step 4
    propagates the per-task domain that Tau2BenchGame.reset() resolved
    during rollout, so Step 5 sees the resolved value.
    """
    if domain not in ("airline", "retail"):
        raise ValueError(
            f"tau2bench prompt builder requires resolved domain "
            f"('airline' or 'retail'); got {domain!r}"
        )
    # Construct only to access the static get_system_prompt /
    # get_tool_schemas helpers — never call reset(), so no user_client
    # is needed. Tau2BenchGame.__init__ now mirrors the domain filter
    # into self._domain so these helpers return the right values
    # without an explicit reset.
    game = Tau2BenchGame(max_steps=1, user_client=None, domain=domain)
    messages = [
        {"role": "system", "content": game.get_system_prompt()},
        *first_messages,
    ]
    tools = game.get_tool_schemas()  # FULL openai schemas — matches tau2-bench eval
    return messages, tools


def _apply_chat_template(tokenizer, msgs, tools) -> str:
    try:
        return tokenizer.apply_chat_template(
            msgs, tools=tools, tokenize=False, add_generation_prompt=True,
        )
    except TypeError:
        # Older tokenizers without `tools=` support.
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )


def _tokenize_prompts(
    seeds: Sequence[int],
    domains: Sequence[Optional[str]],
    tokenizer,
    max_length: int,
    *,
    game: str = "synthetic",
    first_messages_per_seed: Optional[Dict[int, List[Dict]]] = None,
    domain_per_seed: Optional[Dict[int, str]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tokenize a batch of prompts.

    ``game='synthetic'``: uses ``_build_prompt_for_seed`` (Tau2BenchSyntheticGame).
    ``game='tau2bench'``: uses ``_build_prompt_tau2bench`` with the captured
        first-turn messages from Step 2 (passed via ``first_messages_per_seed``)
        and the resolved per-task domain (``domain_per_seed``).
    """
    chats: List[str] = []
    for seed, dom in zip(seeds, domains):
        if game == "tau2bench":
            if first_messages_per_seed is None or domain_per_seed is None:
                raise ValueError(
                    "tau2bench mode requires first_messages_per_seed and "
                    "domain_per_seed to be provided."
                )
            resolved_domain = domain_per_seed[int(seed)]
            fm = first_messages_per_seed[int(seed)]
            msgs, tools = _build_prompt_tau2bench(resolved_domain, fm)
        else:
            msgs, tools = _build_prompt_for_seed(int(seed), dom)
        chats.append(_apply_chat_template(tokenizer, msgs, tools))
    enc = tokenizer(
        chats, return_tensors="pt", padding=True, truncation=True, max_length=max_length,
    )
    return enc["input_ids"], enc["attention_mask"]


def _hard_label_from_target_row(row: Dict) -> int:
    """argmax of empirical mean rewards (preferred); fall back to p_target."""
    rewards = row.get("rewards")
    if rewards is not None:
        return int(max(range(len(rewards)), key=lambda i: rewards[i]))
    p = row["p_target"]
    return int(max(range(len(p)), key=lambda i: p[i]))


# ---------------------------------------------------------------------------
# Per-layer training (existing): per-block KL against Step 4's soft target
# ---------------------------------------------------------------------------

def _train_per_layer(
    args: argparse.Namespace,
    targets: List[Dict],
    canonical: List[str],
    num_capabilities: int,
    *,
    first_messages_per_seed: Optional[Dict[int, List[Dict]]] = None,
    domain_per_seed: Optional[Dict[int, str]] = None,
) -> int:
    capability_adapters = [c for c in canonical if c != "base"]
    include_base = ("base" in canonical)

    if args.target_modules == "all":
        target_modules = (
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        )
    else:
        target_modules = ("q_proj", "k_proj", "v_proj", "o_proj")

    cfg = CapabilityModelConfig(
        base_model=args.base_model,
        capability_adapters=capability_adapters,
        include_base_capability=include_base,
        target_modules=target_modules,
        routing_granularity=args.routing_granularity,
        routing_scope="per_layer",
        cache_dir=args.cache_dir,
        device_map=args.device_map,
        dtype=torch.bfloat16,
    )
    model, meta = build_capability_model(cfg, return_meta=True)
    tokenizer = meta["tokenizer"]
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("[step5] assert_frozen...")
    assert_frozen(model)
    print("[step5] freeze invariants OK")

    # ---- Forward hooks to capture gater outputs --------------------------
    captured: List[torch.Tensor] = []

    def _hook(_module, _inputs, output):
        captured.append(output)

    handles = []
    for g in meta["gaters"]:
        handles.append(g.register_forward_hook(_hook))

    # ---- Optimizer over gater params only --------------------------------
    gater_params = [p for _, p in gater_parameters(model)]
    n_params = sum(p.numel() for p in gater_params)
    print(f"[step5] optimizing {n_params:,} gater params with lr={args.lr}")
    optim = torch.optim.AdamW(
        gater_params, lr=args.lr, weight_decay=args.weight_decay
    )

    # Where to put the loss tensor / target tensor (same as first gater).
    loss_dev = next(meta["gaters"][0].parameters()).device

    domain_arg = None if args.domain == "mixed" else args.domain
    n_train = len(targets)
    indices = list(range(n_train))

    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        random.Random(args.seed + epoch).shuffle(indices)
        for batch_start in range(0, n_train, args.batch_size):
            batch_idx = indices[batch_start : batch_start + args.batch_size]
            seeds = [targets[i]["seed"] for i in batch_idx]
            p_targets = [targets[i]["p_target"] for i in batch_idx]
            domains = [domain_arg] * len(seeds)

            input_ids, attn_mask = _tokenize_prompts(
                seeds, domains, tokenizer, args.max_length,
                game=args.game,
                first_messages_per_seed=first_messages_per_seed,
                domain_per_seed=domain_per_seed,
            )
            p_target = torch.tensor(p_targets, dtype=torch.float32)

            captured.clear()

            embed_dev = next(model.parameters()).device
            input_ids = input_ids.to(embed_dev)
            attn_mask = attn_mask.to(embed_dev)
            p_target = p_target.to(loss_dev)

            optim.zero_grad(set_to_none=True)
            _ = model(input_ids=input_ids, attention_mask=attn_mask)

            # Per-layer KL(g_l || p_target). Average across response tokens
            # for token-mode (g shape [B, T, C]); trajectory mode is [B, 1, C]
            # already.
            loss = torch.tensor(0.0, device=loss_dev, dtype=torch.float32)
            for w in captured:
                wflat = w.float().mean(dim=1) if w.dim() == 3 else w.float()
                wflat = wflat.to(device=loss_dev)
                log_p = torch.log(wflat.clamp_min(1e-9))
                loss = loss + F.kl_div(log_p, p_target, reduction="batchmean")
            loss = loss / max(1, len(captured))

            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(gater_params, args.max_grad_norm)
            optim.step()

            step += 1
            if step % args.log_every == 0 or step == 1:
                pieces = [
                    (w.float().mean(dim=1) if w.dim() == 3 else w.float()).to(loss_dev)
                    for w in captured
                ]
                avg = torch.stack(pieces).mean(dim=0)
                avg_per_cap = avg.mean(dim=0).tolist()
                tgt_per_cap = p_target.mean(dim=0).tolist()
                kl_str = ", ".join(f"{a:.2f}/{t:.2f}" for a, t in zip(avg_per_cap, tgt_per_cap))
                print(
                    f"[step5] epoch={epoch} step={step} loss={loss.item():.4f} "
                    f"avg_g/p_target=[{kl_str}] dt={time.time()-t0:.1f}s",
                    flush=True,
                )

    for h in handles:
        h.remove()

    # ---- Save gater state -------------------------------------------------
    state: Dict[str, torch.Tensor] = {}
    for layer_idx, g in enumerate(meta["gaters"]):
        for k, v in g.state_dict().items():
            state[f"layer{layer_idx}.{k}"] = v.detach().cpu()
    out = Path(args.ckpt_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "gater_state": state,
            "capability_names": canonical,
            "num_capabilities": num_capabilities,
            "routing_granularity": args.routing_granularity,
            "routing_scope": "per_layer",
            "base_model": args.base_model,
            "args": vars(args),
        },
        out,
    )
    print(f"[step5] saved warm-started gaters -> {out}")

    # Optional HF export (per-layer).
    if args.export_dir or args.push_to_hub:
        from capability_router import (
            save_capability_routed_checkpoint,
            push_capability_routed_to_hub,
        )
        export_dir = Path(args.export_dir or (out.parent / f"{out.stem}_hf_export"))
        save_capability_routed_checkpoint(
            export_dir,
            base_model=args.base_model,
            capability_adapters=capability_adapters,
            include_base_capability=include_base,
            routing_granularity=args.routing_granularity,
            routing_scope="per_layer",
            gater_state_path=out,
        )
        print(f"[step5] HF checkpoint exported -> {export_dir}")
        if args.push_to_hub:
            url = push_capability_routed_to_hub(
                export_dir, args.push_to_hub,
                private=args.hub_private,
                commit_message="Step 5: warm-started capability gates",
            )
            print(f"[step5] pushed to {url}")
    return 0


# ---------------------------------------------------------------------------
# Global training (new): CE on last-token features of the frozen base
# ---------------------------------------------------------------------------

def _last_token_pool(
    base_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Last-token pool of last hidden state under no_grad. Returns [B, H]."""
    with torch.no_grad():
        out = base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    hidden = out.hidden_states[-1]  # [B, T, H], post-final-norm
    last_pos = (attention_mask.sum(dim=1) - 1).clamp(min=0)
    batch_idx = torch.arange(hidden.size(0), device=hidden.device)
    return hidden[batch_idx, last_pos.to(hidden.device), :]


def _train_global(
    args: argparse.Namespace,
    targets: List[Dict],
    canonical: List[str],
    num_capabilities: int,
    *,
    first_messages_per_seed: Optional[Dict[int, List[Dict]]] = None,
    domain_per_seed: Optional[Dict[int, str]] = None,
) -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[step5b] loading FROZEN base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map=args.device_map,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, cache_dir=args.cache_dir, trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    hidden_size = int(base_model.config.hidden_size)
    embed_dev = base_model.get_input_embeddings().weight.device

    router = TrajectoryGlobalRouter(
        hidden_size=hidden_size, num_capabilities=num_capabilities,
    ).to(dtype=torch.float32, device=embed_dev)
    n_params = sum(p.numel() for p in router.parameters())
    print(f"[step5b] router: {n_params:,} trainable params on {embed_dev}")

    optim = torch.optim.AdamW(router.parameters(), lr=args.lr)

    # ---- Train/val split --------------------------------------------------
    domain_arg = None if args.domain == "mixed" else args.domain
    rng = random.Random(args.seed)
    perm = list(range(len(targets)))
    rng.shuffle(perm)
    n_val = (
        max(1, int(round(len(targets) * args.val_fraction)))
        if args.val_fraction > 0 else 0
    )
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    print(f"[step5b] {len(train_idx)} train / {len(val_idx)} val rows")

    # ---- Pre-extract features ONCE (frozen base => features are constant
    #      across epochs; caching turns Step 5b from O(epochs * forwards)
    #      into O(forwards) + O(epochs * tiny linear). For 164 prompts
    #      at ~3s/forward this is ~8 min vs ~hours for 60 epochs uncached).
    print(f"[step5b] extracting last-token features for {len(targets)} prompts "
          f"(one frozen-base forward each)...")
    t_feat = time.time()
    features_per_idx: List[torch.Tensor] = [None] * len(targets)
    for batch_start in range(0, len(targets), args.batch_size):
        batch = list(range(batch_start, min(batch_start + args.batch_size, len(targets))))
        seeds = [targets[i]["seed"] for i in batch]
        domains = [domain_arg] * len(seeds)
        input_ids, attn_mask = _tokenize_prompts(
            seeds, domains, tokenizer, args.max_length,
            game=args.game,
            first_messages_per_seed=first_messages_per_seed,
            domain_per_seed=domain_per_seed,
        )
        input_ids = input_ids.to(embed_dev)
        attn_mask = attn_mask.to(embed_dev)
        feats = _last_token_pool(base_model, input_ids, attn_mask)
        feats = feats.detach().to(dtype=torch.float32, device="cpu")
        for j, i in enumerate(batch):
            features_per_idx[i] = feats[j]
        if (batch_start // args.batch_size) % 10 == 0:
            print(f"[step5b]   features {batch_start + len(batch)}/{len(targets)} "
                  f"dt={time.time()-t_feat:.1f}s", flush=True)
    cache = torch.stack(features_per_idx, dim=0)  # [N, H], fp32 on CPU
    labels_per_idx = torch.tensor(
        [_hard_label_from_target_row(t) for t in targets], dtype=torch.long,
    )
    print(f"[step5b] feature cache: shape={tuple(cache.shape)}, "
          f"dtype={cache.dtype}, total feature time={time.time()-t_feat:.1f}s")

    # Free the base model — features are cached, training never touches it.
    del base_model
    torch.cuda.empty_cache()

    # Move cache + labels to a single GPU; the linear router lives there.
    train_dev = next(router.parameters()).device
    cache_gpu = cache.to(train_dev)
    labels_gpu = labels_per_idx.to(train_dev)

    train_X = cache_gpu[train_idx]
    train_y = labels_gpu[train_idx]
    val_X = cache_gpu[val_idx] if val_idx else None
    val_y = labels_gpu[val_idx] if val_idx else None

    # ---- Train (full-batch CE on cached features — fast linear) ----------
    print(f"[step5b] training for {args.epochs} epochs (full-batch CE on cache)")
    t0 = time.time()
    for epoch in range(args.epochs):
        optim.zero_grad(set_to_none=True)
        logits = router(train_X)
        loss = F.cross_entropy(logits, train_y)
        loss.backward()
        optim.step()
        if epoch % args.log_every == 0 or epoch == args.epochs - 1:
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                acc = (pred == train_y).float().mean().item()
                if val_X is not None:
                    val_logits = router(val_X)
                    val_pred = val_logits.argmax(dim=-1)
                    val_acc = (val_pred == val_y).float().mean().item()
                    val_str = f" val_acc={val_acc:.3f}"
                else:
                    val_str = ""
            print(
                f"[step5b] epoch={epoch} loss={loss.item():.4f} "
                f"train_acc={acc:.3f}{val_str} dt={time.time()-t0:.1f}s",
                flush=True,
            )

    # ---- Validation diagnostics ------------------------------------------
    val_metrics: Dict[str, float] = {}
    pred_hist: List[int] = [0] * num_capabilities
    if val_idx:
        with torch.no_grad():
            val_logits = router(val_X)
        pred = val_logits.argmax(dim=-1).cpu().tolist()
        truth = val_y.cpu().tolist()
        for p_i in pred:
            pred_hist[int(p_i)] += 1
        n_correct = sum(int(p == t) for p, t in zip(pred, truth))
        n_total = len(truth)
        if n_total > 0:
            val_metrics["val_top1_acc"] = n_correct / n_total
            print(f"[step5b] val top-1 accuracy: "
                  f"{val_metrics['val_top1_acc']:.3f} ({n_correct}/{n_total})")
        n_pred_total = sum(pred_hist) or 1
        print("[step5b] val argmax distribution:")
        for cap, n in zip(canonical, pred_hist):
            print(f"  {cap}: {n / n_pred_total:.3f}  (n={n})")

    # ---- Save -------------------------------------------------------------
    out = Path(args.ckpt_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "router_state": {
                k: v.detach().cpu() for k, v in router.state_dict().items()
            },
            "capability_names": canonical,
            "num_capabilities": num_capabilities,
            "hidden_size": hidden_size,
            "routing_scope": "global",
            "routing_granularity": "trajectory",
            "base_model": args.base_model,
            "args": vars(args),
            "val_metrics": val_metrics,
            "val_pred_hist": pred_hist,
        },
        out,
    )
    print(f"[step5b] saved global router -> {out}")

    # ---- Optional HF export ----------------------------------------------
    if args.export_dir or args.push_to_hub:
        from capability_router import (
            save_capability_routed_checkpoint,
            push_capability_routed_to_hub,
        )
        export_dir = Path(args.export_dir or (out.parent / f"{out.stem}_hf_export"))
        capability_adapters = [c for c in canonical if c != "base"]
        save_capability_routed_checkpoint(
            export_dir,
            base_model=args.base_model,
            capability_adapters=capability_adapters,
            include_base_capability=("base" in canonical),
            routing_granularity="trajectory",
            routing_scope="global",
            global_router_state_path=out,
        )
        print(f"[step5b] HF checkpoint exported -> {export_dir}")
        if args.push_to_hub:
            url = push_capability_routed_to_hub(
                export_dir, args.push_to_hub,
                private=args.hub_private,
                commit_message="Step 5b: global router warm-start (CE)",
            )
            print(f"[step5b] pushed to {url}")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    # Common
    ap.add_argument("--targets", required=True)
    ap.add_argument("--ckpt-out", required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--routing-scope", choices=["per_layer", "global"], default="per_layer",
                    help="'per_layer' trains 48 block-local soft routers via KL; "
                         "'global' trains a single classifier head via CE.")
    ap.add_argument("--game", choices=["synthetic", "tau2bench"], default="synthetic",
                    help="'synthetic' = Tau2BenchSyntheticGame prompt builder "
                         "(matches step5's prior behavior); 'tau2bench' = use the "
                         "captured first-turn messages persisted by Step 2 + "
                         "Step 4 to reproduce the EXACT prompt the agent saw "
                         "during rollouts. tau2bench mode requires the targets "
                         "parquet to carry 'first_messages_json' and 'domain' "
                         "columns (Step 4 produces these from a Step 2 run that "
                         "also used --game tau2bench).")
    ap.add_argument("--routing-granularity", default="trajectory",
                    choices=["trajectory", "token"],
                    help="Per-layer mode only. Global mode is always trajectory.")
    ap.add_argument("--target-modules", default="attn",
                    choices=["attn", "all"],
                    help="Per-layer mode only. 'attn' = q/k/v/o; 'all' = + each "
                         "Qwen3 expert's gate/up/down. 'all' requires "
                         "transformers<5 (unfused experts).")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="Per-layer mode only.")
    ap.add_argument("--max-length", type=int, default=8192,
                    help="Prompt-token cap. tau2-bench full-schema prompts measure "
                         "4784-5140 tokens; synthetic-env compact-schema prompts ~3000.")
    ap.add_argument("--max-grad-norm", type=float, default=1.0,
                    help="Per-layer mode only.")
    ap.add_argument("--val-fraction", type=float, default=0.1,
                    help="Global mode only. Held-out fraction for top-1 accuracy.")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--domain", choices=["airline", "retail", "mixed"], default="mixed")
    ap.add_argument("--export-dir", default=None,
                    help="If set, also export a self-contained HF checkpoint here.")
    ap.add_argument("--push-to-hub", default=None,
                    help="If set (e.g. 'user/repo'), push the export to the HF Hub.")
    ap.add_argument("--hub-private", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    targets = _load_targets(Path(args.targets))
    if not targets:
        print(f"[step5] empty targets file: {args.targets}")
        return 1

    canonical = list(targets[0]["capability_names"])
    num_capabilities = len(canonical)
    print(f"[step5] capability ordering: {canonical}")
    print(f"[step5] routing_scope={args.routing_scope}, game={args.game}")

    # In tau2bench mode, decode the captured per-seed first messages + domain
    # once. They flow through _tokenize_prompts to produce prompts byte-
    # identical to what the agent saw on its first turn during Step 2.
    first_messages_per_seed: Optional[Dict[int, List[Dict]]] = None
    domain_per_seed: Optional[Dict[int, str]] = None
    if args.game == "tau2bench":
        first_messages_per_seed = {}
        domain_per_seed = {}
        missing = 0
        for row in targets:
            seed = int(row["seed"])
            fm_raw = row.get("first_messages_json", "")
            dom = row.get("domain", "")
            if not fm_raw or not dom:
                missing += 1
                continue
            try:
                fm = json.loads(fm_raw)
            except Exception as e:
                print(f"[step5] warning: bad first_messages_json for seed={seed}: {e}")
                missing += 1
                continue
            first_messages_per_seed[seed] = list(fm)
            domain_per_seed[seed] = str(dom)
        if missing:
            print(f"[step5] WARNING: {missing}/{len(targets)} target rows lack "
                  f"first_messages_json/domain. Run Step 2 with --game tau2bench "
                  f"and Step 4 against that parquet to populate them.")
        if not first_messages_per_seed:
            print(f"[step5] FATAL: no captured first_messages — refusing to "
                  f"silently fall back to synthetic prompt builder. "
                  f"Re-run Steps 2+4 with --game tau2bench.")
            return 2
        print(f"[step5] tau2bench: loaded captured first_messages for "
              f"{len(first_messages_per_seed)} seeds")

    if args.routing_scope == "global":
        return _train_global(
            args, targets, canonical, num_capabilities,
            first_messages_per_seed=first_messages_per_seed,
            domain_per_seed=domain_per_seed,
        )
    return _train_per_layer(
        args, targets, canonical, num_capabilities,
        first_messages_per_seed=first_messages_per_seed,
        domain_per_seed=domain_per_seed,
    )


if __name__ == "__main__":
    sys.exit(main())
