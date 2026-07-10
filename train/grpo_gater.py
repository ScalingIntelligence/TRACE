#!/usr/bin/env python3
"""Step 6: GRPO with gating-only updates.

Loads the warm-started capability model (frozen base + frozen LoRAs +
warm-started per-block gaters) and runs a minimal GRPO loop where ONLY the
gater Linears are updated.

Compared to the full ``train_grpo_optimized.py``, this trainer is intentionally
small. It is designed to verify end-to-end correctness on the cluster — once
the loss decreases on a small mix of synthetic tau2-bench tasks, the
production trainer can be adapted to call ``build_capability_model`` instead
of ``FastLanguageModel`` and reuse the same rollout / advantage code.

Generation correctness note:
  * Under ``routing_granularity="trajectory"`` the routing should be computed
    once on the prompt and held constant during generation. We do this via
    a "lock-after-prefill" pattern: forward the prompt, ``gater.lock()``
    every block's gater, generate, then ``gater.unlock()``.
  * Under ``"token"``, locking is skipped — each new token gets a fresh g.

Smoke-test mode:
    python step6_grpo_gater.py --smoke --num-tasks 2 --rollouts-per 2
This builds the model, asserts freezes, runs a small rollout batch, and
takes ONE optimizer step on the gaters.

Real run (with vLLM rollout server for the user-LLM only):
    USER_LLM_BASE_URL=http://localhost:9000/v1 \\
    USER_LLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 \\
    python step6_grpo_gater.py \\
        --gater-init /tmp/games/gater_warmstart.pt \\
        --num-iters 50 --num-tasks 8 --rollouts-per 4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from capability_router import (
    CapabilityModelConfig,
    assert_frozen,
    build_capability_model,
    gater_parameters,
)
from capability_router.model_builder import DEFAULT_CAPABILITY_REPOS
from capability_router.gater import (
    LayerCapabilityGater,
    lock_gaters_after_first_forward,
)
from tau2_bench_synthetic_game import Tau2BenchSyntheticGame


# ---------------------------------------------------------------------------
# Generation under trajectory locking
# ---------------------------------------------------------------------------

def _lock_all_gaters(gaters: List[LayerCapabilityGater]) -> None:
    for g in gaters:
        g.lock()


def _unlock_all_gaters(gaters: List[LayerCapabilityGater]) -> None:
    for g in gaters:
        g.unlock()


def _build_chat_prompt(tokenizer, game, compact_tools: bool = True) -> str:
    """Build the agent-side chat prompt for a tau-bench step.

    Includes ``tools=`` so the chat template formats tool schemas inline
    (Qwen3's template wraps them in ``<tools>`` XML). Uses compact schemas
    by default — the full schemas alone are ~3.5K tokens.
    """
    messages = [
        {"role": "system", "content": game.get_system_prompt()},
        *game.get_messages(),
    ]
    if compact_tools and hasattr(game, "get_tool_schemas_compact"):
        tools = game.get_tool_schemas_compact()
    else:
        tools = game.get_tool_schemas()
    try:
        return tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )
    except TypeError:
        # Older tokenizers without `tools=` support.
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


@torch.no_grad()
def _gated_generate(
    model,
    tokenizer,
    gaters: List[LayerCapabilityGater],
    prompt_text: str,
    max_new_tokens: int,
    temperature: float,
    granularity: str,
) -> Tuple[str, torch.Tensor, torch.Tensor]:
    """Generate under correct gater locking. In trajectory mode a one-shot
    hook locks each gater right after generate's internal prefill, so
    subsequent T=1 forwards return the prompt-only routing.
    """
    enc = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=8192)
    embed_dev = next(model.parameters()).device
    input_ids = enc["input_ids"].to(embed_dev)
    attn_mask = enc["attention_mask"].to(embed_dev)

    handle = None
    if granularity == "trajectory":
        handle = lock_gaters_after_first_forward(model, gaters)

    try:
        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    finally:
        if handle is not None:
            handle.remove()
        if granularity == "trajectory":
            _unlock_all_gaters(gaters)

    completion_ids = gen[0, input_ids.size(1):]
    completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    return completion_text, input_ids[0], completion_ids


# ---------------------------------------------------------------------------
# Logprob computation (used both for "old logp" at rollout time and "new logp"
# at update time)
# ---------------------------------------------------------------------------

def _full_sequence_logprobs(
    model,
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor,
    gaters: List[LayerCapabilityGater],
    granularity: str,
    lock_mode: str,
) -> torch.Tensor:
    """Per-completion-token logprobs under a chosen gater-locking strategy.

    ``lock_mode`` (trajectory granularity only):
      * ``"detached"`` — no_grad prefill on prompt, lock without grad.
        Use for ``old_lp``: matches what generation produced at rollout.
      * ``"grad"`` — prefill on prompt with grad, capture each gater's
        output, lock to that grad-attached tensor. Use for ``new_lp``:
        gradient flows through the gater Linear while the routing input
        still matches rollout (prompt-only, not prompt+completion), so
        the PPO ratio reflects only the Linear update.
      * ``"none"`` — no locking. Required for token granularity, where
        per-position routing is decoding-mode-invariant.
    """
    if lock_mode not in ("detached", "grad", "none"):
        raise ValueError(f"lock_mode must be 'detached'|'grad'|'none', got {lock_mode!r}")

    embed_dev = next(model.parameters()).device
    full = torch.cat([prompt_ids, completion_ids], dim=0).unsqueeze(0).to(embed_dev)
    attn = torch.ones_like(full)

    apply_lock = granularity == "trajectory" and lock_mode in ("detached", "grad")

    if apply_lock:
        prompt_in = prompt_ids.unsqueeze(0).to(embed_dev)
        prompt_attn = torch.ones_like(prompt_in)

    if apply_lock and lock_mode == "detached":
        with torch.no_grad():
            _ = model(input_ids=prompt_in, attention_mask=prompt_attn, use_cache=False)
        _lock_all_gaters(gaters)
    elif apply_lock and lock_mode == "grad":
        captured: List[Optional[torch.Tensor]] = [None] * len(gaters)
        handles: List[Any] = []

        def _make_hook(idx: int):
            def _hook(_module, _inputs, output):
                captured[idx] = output
            return _hook

        for i, g in enumerate(gaters):
            handles.append(g.register_forward_hook(_make_hook(i)))
        try:
            _ = model(input_ids=prompt_in, attention_mask=prompt_attn, use_cache=False)
        finally:
            for h in handles:
                h.remove()
        for i, g in enumerate(gaters):
            if captured[i] is None:
                raise RuntimeError(
                    f"Did not capture gater output for layer {i} during prefill."
                )
            g.lock(weights=captured[i], detach=False)

    try:
        out = model(input_ids=full, attention_mask=attn, use_cache=False)
    finally:
        if apply_lock:
            _unlock_all_gaters(gaters)

    logits = out.logits[0, :-1, :]
    targets = full[0, 1:]
    log_probs = F.log_softmax(logits.float(), dim=-1)
    per_token = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return per_token[-completion_ids.size(0):]


# ---------------------------------------------------------------------------
# Rollout container
# ---------------------------------------------------------------------------

@dataclass
class Rollout:
    seed: int
    domain: str
    prompt_ids: torch.Tensor   # [T_prompt]
    completion_ids: torch.Tensor  # [T_completion]
    old_logp: torch.Tensor     # [T_completion]
    reward: float
    group_id: int


def _grpo_advantages(group: List[Rollout], norm_by_std: bool = True) -> List[float]:
    """Standardized group-relative advantage (one value per rollout)."""
    rs = [r.reward for r in group]
    mu = sum(rs) / len(rs)
    centered = [r - mu for r in rs]
    if norm_by_std and len(rs) > 1:
        var = sum(c * c for c in centered) / max(1, len(rs) - 1)
        std = math.sqrt(var) + 1e-8
        return [c / std for c in centered]
    return centered


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--routing-granularity", default="trajectory",
                    choices=["trajectory", "token"])
    ap.add_argument(
        "--target-modules", default="attn",
        choices=["attn", "all"],
        help=("'attn' = q/k/v/o; 'all' = + each Qwen3 expert's gate/up/down. "
              "'all' requires transformers<5 (unfused experts); on >=5 the "
              "expert paths fail to resolve and 'all' degrades to 'attn'."),
    )
    ap.add_argument("--gater-init", default=None,
                    help="Path to warm-started gater state dict (Step 5 output).")
    ap.add_argument("--user-llm-url", default=os.environ.get("USER_LLM_BASE_URL"))
    ap.add_argument("--user-llm-model", default=os.environ.get("USER_LLM_MODEL"))
    ap.add_argument("--num-iters", type=int, default=50)
    ap.add_argument("--num-tasks", type=int, default=4,
                    help="Distinct task seeds per iteration (= GRPO groups).")
    ap.add_argument("--rollouts-per", type=int, default=4,
                    help="K = rollouts per group.")
    ap.add_argument("--hint-fraction", type=float, default=0.5,
                    help=(
                        "Fraction of rollouts in each (task) group that get "
                        "the gold-action hint scaffolded into the system "
                        "prompt. The hint is what guarantees within-group "
                        "reward variance — without it, K=2/4/8 rollouts "
                        "from a strong+strict-reward task all collapse to "
                        "the same outcome and GRPO advantage is 0. The hint "
                        "is left in the prompt for both rollout AND policy "
                        "update; the gater learns to route well *given the "
                        "hint*, which is fine because the prompt's hidden "
                        "state is what it conditions on."
                    ))
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=15)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--seed-offset", type=int, default=10_000)
    ap.add_argument("--ckpt-out", default="/tmp/games/gater_grpo.pt")
    ap.add_argument("--smoke", action="store_true",
                    help="Build, do 1 iter on tiny synthetic data, exit.")
    ap.add_argument("--export-dir", default=None,
                    help="If set, also export a self-contained HF checkpoint here.")
    ap.add_argument("--push-to-hub", default=None,
                    help="If set (e.g. 'user/repo'), push the export to the HF Hub.")
    ap.add_argument("--hub-private", action="store_true")
    args = ap.parse_args()

    rng = random.Random(0)

    # ---- 1. Build model ---------------------------------------------------
    if args.target_modules == "all":
        target_modules = (
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        )
    else:
        target_modules = ("q_proj", "k_proj", "v_proj", "o_proj")

    # Pre-load checkpoint so we can build cfg with the SAME capability ordering
    # the gater Linears were trained against in Step 5. Step 4 sorts capabilities
    # alphabetically when constructing canonical, which differs from
    # DEFAULT_CAPABILITY_REPOS' declaration order — without this, the loaded
    # state_dict's columns would be misinterpreted.
    gater_state_blob = None
    loaded_canonical: Optional[List[str]] = None
    if args.gater_init:
        gater_state_blob = torch.load(
            args.gater_init, map_location="cpu", weights_only=False,
        )
        cn = gater_state_blob.get("capability_names")
        loaded_canonical = list(cn) if cn is not None else None

    if loaded_canonical:
        capability_adapters = [c for c in loaded_canonical if c != "base"]
        include_base = ("base" in loaded_canonical)
    else:
        capability_adapters = list(DEFAULT_CAPABILITY_REPOS)
        include_base = True

    cfg = CapabilityModelConfig(
        base_model=args.base_model,
        capability_adapters=capability_adapters,
        include_base_capability=include_base,
        target_modules=target_modules,
        routing_granularity=args.routing_granularity,
        cache_dir=args.cache_dir,
        device_map=args.device_map,
        dtype=torch.bfloat16,
    )
    model, meta = build_capability_model(cfg, return_meta=True)
    tokenizer = meta["tokenizer"]
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    gaters: List[LayerCapabilityGater] = meta["gaters"]
    canonical = meta["capability_names"]

    # ---- 2. Load warm-started gaters --------------------------------------
    if gater_state_blob is not None:
        if loaded_canonical is not None and canonical != loaded_canonical:
            raise RuntimeError(
                "Gater capability ordering mismatch:\n"
                f"  checkpoint: {loaded_canonical}\n"
                f"  built:      {canonical}"
            )
        gs = gater_state_blob["gater_state"]
        for layer_idx, g in enumerate(gaters):
            sub = {k.split(".", 1)[1]: v
                   for k, v in gs.items() if k.startswith(f"layer{layer_idx}.")}
            g.load_state_dict(sub, strict=True)
        print(f"[step6] loaded warm-started gaters from {args.gater_init} "
              f"(canonical: {canonical})")

    # ---- 3. Freeze invariants ---------------------------------------------
    print("[step6] assert_frozen...")
    assert_frozen(model)
    print("[step6] freeze invariants OK")

    gater_params = [p for _, p in gater_parameters(model)]
    optim = torch.optim.AdamW(gater_params, lr=args.lr, weight_decay=args.weight_decay)
    print(f"[step6] {sum(p.numel() for p in gater_params):,} trainable gater params")

    # ---- 4. User LLM client (optional in --smoke) -------------------------
    user_client = None
    if args.user_llm_url and args.user_llm_model:
        from adversarial_policy_game.llm_user import UserLLMClient  # type: ignore
        user_client = UserLLMClient(base_url=args.user_llm_url, model=args.user_llm_model)
    elif not args.smoke:
        print("[step6] WARNING: no user LLM configured. Episodes will terminate "
              "immediately (the synthetic env emits ###STOP### without one).")

    # ---- 5. Rollout + GRPO update loop ------------------------------------
    n_iters = 1 if args.smoke else args.num_iters
    n_tasks = 2 if args.smoke else args.num_tasks
    rollouts_per = 2 if args.smoke else args.rollouts_per

    # Hint stratification within each group (matches step2's pattern).
    # hint_fraction <= 0 fully disables hints (matches Step 5's no-hint training
    # distribution). hint_fraction > 0 floors to at least one hinted rollout per
    # group so binary tau-bench rewards still produce within-group variance.
    if args.hint_fraction <= 0:
        n_hinted_per_group = 0
    else:
        n_hinted_per_group = max(1, int(round(rollouts_per * args.hint_fraction)))
        n_hinted_per_group = min(n_hinted_per_group, rollouts_per)
    hint_schedule = [True] * n_hinted_per_group + [False] * (rollouts_per - n_hinted_per_group)
    print(f"[step6] rollouts_per={rollouts_per}, hint_schedule={hint_schedule} "
          f"({sum(hint_schedule)} hinted / {rollouts_per})")

    for it in range(n_iters):
        t0 = time.time()
        all_rollouts: List[Rollout] = []
        for grp in range(n_tasks):
            seed = args.seed_offset + it * 1000 + grp
            for k in range(rollouts_per):
                game = Tau2BenchSyntheticGame(
                    max_steps=args.max_steps, user_client=user_client, domain=None,
                )
                game.reset(seed, use_hint=hint_schedule[k])

                # Multi-turn loop: agent plays until ``game.done``. We record
                # ONLY the first (prompt_ids, completion_ids) pair for GRPO —
                # the gater's gradient comes from "given this initial prompt,
                # producing this initial response led to terminal reward R".
                # Continued turns affect the reward but not the (prompt,
                # completion) used for the loss.
                first_prompt_ids: torch.Tensor | None = None
                first_completion_ids: torch.Tensor | None = None
                steps_played = 0
                while not game.done and steps_played < args.max_steps:
                    steps_played += 1
                    prompt_text = _build_chat_prompt(tokenizer, game, compact_tools=True)
                    completion_text, p_ids, c_ids = _gated_generate(
                        model, tokenizer, gaters, prompt_text,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        granularity=args.routing_granularity,
                    )
                    if first_prompt_ids is None:
                        first_prompt_ids = p_ids
                        first_completion_ids = c_ids
                    # Tau-bench expects raw JSON tool-call text; if the model
                    # returned plain text (no JSON), wrap it as respond_to_user
                    # so the conversation continues instead of terminating.
                    import re as _re
                    candidate = completion_text.strip()
                    j = _re.search(r"\{[\s\S]*\}", candidate)
                    if j:
                        try:
                            json.loads(j.group())
                            action_text = j.group()
                        except Exception:
                            action_text = json.dumps({
                                "name": "respond_to_user",
                                "arguments": {"message": candidate},
                            })
                    elif candidate:
                        action_text = json.dumps({
                            "name": "respond_to_user",
                            "arguments": {"message": candidate},
                        })
                    else:
                        action_text = json.dumps({
                            "name": "respond_to_user",
                            "arguments": {"message": ""},
                        })
                    game.step(action_text)

                if first_prompt_ids is None:
                    # Game terminated before the agent ever spoke — skip.
                    continue
                prompt_ids = first_prompt_ids
                completion_ids = first_completion_ids
                reward = float(game.rewards.get(0, 0.0))

                # Old logprobs (frozen reference). Match generation's routing.
                old_mode = "detached" if args.routing_granularity == "trajectory" else "none"
                with torch.no_grad():
                    old_lp = _full_sequence_logprobs(
                        model, prompt_ids, completion_ids,
                        gaters, args.routing_granularity, lock_mode=old_mode,
                    ).detach().cpu()

                all_rollouts.append(Rollout(
                    seed=seed, domain=game._domain or "?",
                    prompt_ids=prompt_ids.cpu(),
                    completion_ids=completion_ids.cpu(),
                    old_logp=old_lp,
                    reward=reward, group_id=grp,
                ))

        # ---- Group-relative advantages
        groups: Dict[int, List[Rollout]] = {}
        for r in all_rollouts:
            groups.setdefault(r.group_id, []).append(r)
        advantages: Dict[int, List[float]] = {}
        for gid, group in groups.items():
            advantages[gid] = _grpo_advantages(group, norm_by_std=True)

        # ---- Skip iter if all groups have zero variance
        flat = [a for adv_list in advantages.values() for a in adv_list]
        var = sum(a * a for a in flat) / max(1, len(flat))
        if var < 1e-8:
            print(f"[step6] iter {it}: zero-variance batch, skipping update")
            continue

        # ---- Compute clipped surrogate loss
        optim.zero_grad(set_to_none=True)
        total_loss = torch.tensor(0.0, device=next(gaters[0].parameters()).device,
                                  dtype=torch.float32)
        n_seq = 0
        new_mode = "grad" if args.routing_granularity == "trajectory" else "none"
        for gid, group in groups.items():
            for adv, r in zip(advantages[gid], group):
                new_lp = _full_sequence_logprobs(
                    model, r.prompt_ids, r.completion_ids,
                    gaters, args.routing_granularity, lock_mode=new_mode,
                )
                old_lp = r.old_logp.to(new_lp.device, dtype=new_lp.dtype)
                ratio = torch.exp(new_lp - old_lp)  # [T_completion]
                a_t = torch.tensor(adv, device=new_lp.device, dtype=new_lp.dtype)
                unclipped = ratio * a_t
                clipped = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * a_t
                # Negative (we minimize -E[min(...)]).
                seq_loss = -torch.minimum(unclipped, clipped).mean()
                total_loss = total_loss + seq_loss
                n_seq += 1
        total_loss = total_loss / max(1, n_seq)
        total_loss.backward()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(gater_params, args.max_grad_norm)
        optim.step()

        avg_r = sum(r.reward for r in all_rollouts) / max(1, len(all_rollouts))
        print(
            f"[step6] iter {it}/{n_iters} loss={total_loss.item():.4f} "
            f"avg_reward={avg_r:.3f} n_rollouts={len(all_rollouts)} "
            f"dt={time.time() - t0:.1f}s",
            flush=True,
        )

    # ---- 6. Save final gater state -----------------------------------------
    state: Dict[str, torch.Tensor] = {}
    for layer_idx, g in enumerate(gaters):
        for k, v in g.state_dict().items():
            state[f"layer{layer_idx}.{k}"] = v.detach().cpu()
    out = Path(args.ckpt_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "gater_state": state,
            "capability_names": canonical,
            "routing_granularity": args.routing_granularity,
            "base_model": args.base_model,
            "args": vars(args),
        },
        out,
    )
    print(f"[step6] saved gaters -> {out}")

    # Optional: export a self-contained HF checkpoint and push.
    if args.export_dir or args.push_to_hub:
        from capability_router import (
            save_capability_routed_checkpoint,
            push_capability_routed_to_hub,
        )
        export_dir = Path(args.export_dir or (out.parent / f"{out.stem}_hf_export"))
        save_capability_routed_checkpoint(
            export_dir,
            base_model=args.base_model,
            capability_adapters=cfg.capability_adapters,
            include_base_capability=cfg.include_base_capability,
            routing_granularity=cfg.routing_granularity,
            gater_state_path=out,
        )
        print(f"[step6] HF checkpoint exported -> {export_dir}")
        if args.push_to_hub:
            url = push_capability_routed_to_hub(
                export_dir, args.push_to_hub,
                private=args.hub_private,
                commit_message="Step 6: GRPO-trained capability gates",
            )
            print(f"[step6] pushed to {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
