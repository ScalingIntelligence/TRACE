"""
Build a frozen base model + N frozen capability LoRAs + per-block routers.

This is the only place that ties everything together:

  1. Load the base HF model.
  2. For each capability adapter (HF repo or local path), download its
     ``adapter_config.json`` + ``adapter_model.safetensors`` and parse the
     A/B matrices keyed by ``(layer_idx, proj_name)``.
  3. For each transformer block, attach a ``LayerCapabilityGater`` and a
     forward wrapper that fills a per-block ``GaterContext`` before delegating
     to the original block forward.
  4. Replace each LoRA-adapted ``nn.Linear`` (q_proj/k_proj/v_proj/o_proj by
     default) with a ``MultiLoRAGatedLinear`` that holds (A_i, B_i) for every
     capability and reads the block's weights from the ``GaterContext``.
  5. Freeze every parameter except the gater Linears.

We deliberately do NOT route through ``peft.PeftModel`` for the forward pass:
PEFT's stock ``LoraLayer`` doesn't support the per-token / per-trajectory
weighted mixture we need. Loading the safetensors directly is simpler and
keeps full control over dtype and freeze invariants.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from safetensors.torch import load_file as load_safetensors

from .freeze_utils import freeze_all_except_gaters
from .gater import LayerCapabilityGater, TrajectoryGlobalRouter
from .multi_lora import GaterContext, MultiLoRAGatedLinear


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_CAPABILITY_REPOS: List[str] = [
    "tarsur909/precondition-v1-40-ckpt",
    "tarsur909/structured_data_reasoning_grpo_iter40",
    "tarsur909/tau_tool_calling_grpo_iter40",
    "tarsur909/multistep-v4-10-ckpt",
]

# Each default adapter contains 18,624 LoRA pairs per layer-set:
#   192    attention      (q/k/v/o × 48 layers)
#   18,432 Qwen3 expert FFN  (128 experts × {gate,up,down} × 48 layers)
# `--target-modules all` wraps all 7 leaves; `attn` only the first 4.
# Wrapping the expert FFN paths requires Qwen3's experts as an unfused
# ModuleList — true on transformers<5; transformers>=5 packs them into
# Qwen3MoeExperts and the expert paths silently fail to resolve at build.
DEFAULT_TARGET_MODULES: Tuple[str, ...] = (
    "q_proj", "k_proj", "v_proj", "o_proj",
)
ALL_TARGET_MODULES: Tuple[str, ...] = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


@dataclass
class CapabilityModelConfig:
    base_model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    capability_adapters: List[str] = field(
        default_factory=lambda: list(DEFAULT_CAPABILITY_REPOS)
    )
    include_base_capability: bool = True
    """If True, prepend a 'base' slot (no LoRA) to the routing simplex."""
    target_modules: Tuple[str, ...] = DEFAULT_TARGET_MODULES
    routing_granularity: str = "trajectory"  # "trajectory" or "token"
    routing_scope: str = "per_layer"  # "per_layer" or "global"
    """``per_layer``: 48 independent block-local soft routers (existing).
    ``global``: one ``TrajectoryGlobalRouter`` over the prompt; its decision
    is broadcast to every block. Per-block gaters are passive (no Linear).
    Requires ``routing_granularity='trajectory'``."""
    dtype: torch.dtype = torch.bfloat16
    device_map: Optional[str] = "auto"
    cache_dir: Optional[str] = None
    """Local HF cache (defaults to env HF_HOME / ~/.cache/huggingface)."""

    @property
    def num_capabilities(self) -> int:
        return len(self.capability_adapters) + (1 if self.include_base_capability else 0)

    @property
    def capability_names(self) -> List[str]:
        names = list(self.capability_adapters)
        if self.include_base_capability:
            names = ["base"] + names
        return names


# ---------------------------------------------------------------------------
# Adapter loading
# ---------------------------------------------------------------------------

# Match keys like:
#   base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
#   base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight
#   base_model.model.model.layers.0.mlp.experts.96.gate_proj.lora_B.weight
# We capture the full sub-path under the decoder block (``self_attn.q_proj``,
# ``mlp.experts.96.gate_proj``) so that we can recover the exact module
# location later. The trailing ``.<adapter_name>`` segment (when present in
# multi-adapter checkpoints) is consumed by the optional group.
_LORA_KEY_RE = re.compile(
    r"^(?:base_model\.model\.)?model\.layers\.(?P<layer>\d+)"
    r"\.(?P<path>.+?)\.(?P<which>lora_A|lora_B)"
    r"(?:\.\w+)?\.weight$"
)


def _find_nested_adapter(root: Path) -> Optional[Path]:
    """Return the directory inside ``root`` containing adapter_config.json."""
    if (root / "adapter_config.json").exists():
        return root
    for sub in sorted(root.iterdir()):
        if sub.is_dir() and (sub / "adapter_config.json").exists():
            return sub
    return None


def _extract_zip_adapter(zip_path: Path, dest: Path) -> Path:
    """Extract a zipped PEFT adapter checkpoint and return its dir."""
    import zipfile

    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(str(dest))
    found = _find_nested_adapter(dest)
    if found is None:
        raise RuntimeError(f"Extracted {zip_path} but no adapter_config.json found under {dest}")
    return found


def _resolve_adapter_dir(repo_or_path: str, cache_dir: Optional[str]) -> Path:
    """Return a local directory containing ``adapter_config.json``.

    Accepts either a local directory or a Hugging Face repo id. Two layouts
    are supported on the HF side:

      1. Standard PEFT layout: repo root contains ``adapter_config.json`` +
         ``adapter_model.safetensors`` directly.
      2. Zipped checkpoint: repo contains a single ``*.zip`` whose top-level
         directory holds the PEFT files (auto-extracted to a sibling dir).
    """
    p = Path(repo_or_path)
    if p.exists() and (p / "adapter_config.json").exists():
        return p

    from huggingface_hub import HfApi, hf_hub_download, snapshot_download  # lazy import

    api = HfApi()
    files = api.list_repo_files(repo_or_path)

    if "adapter_config.json" in files:
        local = snapshot_download(
            repo_id=repo_or_path,
            cache_dir=cache_dir,
            allow_patterns=["adapter_config.json", "adapter_model.safetensors", "*.bin"],
        )
        return Path(local)

    # Zipped layout: pick the largest .zip in the repo and extract it.
    zips = [f for f in files if f.endswith(".zip")]
    if not zips:
        raise FileNotFoundError(
            f"Repo {repo_or_path} has no adapter_config.json and no .zip; cannot resolve."
        )
    zip_name = zips[0]
    zip_path = Path(
        hf_hub_download(repo_id=repo_or_path, filename=zip_name, cache_dir=cache_dir)
    )
    # Extract into a sibling "extracted" directory so we don't repeat work.
    dest = zip_path.parent / (zip_path.stem + "__extracted")
    if not (dest.exists() and _find_nested_adapter(dest) is not None):
        dest_resolved = _extract_zip_adapter(zip_path, dest)
    else:
        dest_resolved = _find_nested_adapter(dest)
        assert dest_resolved is not None
    return dest_resolved


def _load_adapter(
    adapter_dir: Path,
) -> Tuple[Dict[Tuple[int, str], Tuple[torch.Tensor, torch.Tensor]], float, int]:
    """Load (A, B) pairs from a PEFT adapter directory.

    Returns:
        weights: dict keyed by ``(layer_idx, sub_path)`` -> ``(A_weight, B_weight)``.
            ``sub_path`` is the full dotted path under the decoder block
            (e.g. ``self_attn.q_proj``, ``mlp.experts.96.gate_proj``).
        scaling: ``lora_alpha / r`` (or read from config if explicit scaling).
        rank: lora rank ``r``.
    """
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No adapter_config.json in {adapter_dir}")
    with cfg_path.open() as f:
        cfg = json.load(f)
    rank = int(cfg.get("r", cfg.get("lora_r", 0)))
    alpha = float(cfg.get("lora_alpha", rank))
    if rank <= 0:
        raise ValueError(f"Bad LoRA rank {rank} in {cfg_path}")
    scaling = alpha / rank

    st_path = adapter_dir / "adapter_model.safetensors"
    if not st_path.exists():
        raise FileNotFoundError(
            f"adapter_model.safetensors missing in {adapter_dir} "
            "(only safetensors format is supported)."
        )
    raw = load_safetensors(str(st_path))

    pairs: Dict[Tuple[int, str], Dict[str, torch.Tensor]] = {}
    for k, v in raw.items():
        m = _LORA_KEY_RE.match(k)
        if not m:
            continue
        layer = int(m.group("layer"))
        sub_path = m.group("path")  # full dotted path under the block
        which = m.group("which")  # "lora_A" or "lora_B"
        slot = pairs.setdefault((layer, sub_path), {})
        slot[which] = v

    out: Dict[Tuple[int, str], Tuple[torch.Tensor, torch.Tensor]] = {}
    for key, slot in pairs.items():
        if "lora_A" in slot and "lora_B" in slot:
            out[key] = (slot["lora_A"], slot["lora_B"])
    if not out:
        raise RuntimeError(
            f"No LoRA (A, B) pairs found in {st_path}. Keys looked at: "
            f"{list(raw.keys())[:3]}..."
        )
    return out, scaling, rank


# ---------------------------------------------------------------------------
# Model surgery
# ---------------------------------------------------------------------------

def _find_decoder_blocks(model: nn.Module) -> List[nn.Module]:
    """Return the list of decoder blocks (e.g. ``model.model.layers``)."""
    base = getattr(model, "model", model)
    layers = getattr(base, "layers", None)
    if layers is None:
        # Some HF wrappers expose it deeper.
        inner = getattr(base, "model", None)
        if inner is not None:
            layers = getattr(inner, "layers", None)
    if layers is None:
        raise RuntimeError(
            "Could not locate decoder layers list. Model class: "
            f"{type(model).__name__}"
        )
    return list(layers)


def _get_hidden_size(model: nn.Module) -> int:
    cfg = model.config
    for k in ("hidden_size", "d_model", "n_embd"):
        if hasattr(cfg, k):
            return int(getattr(cfg, k))
    raise RuntimeError(f"Cannot determine hidden_size from config {cfg}")


def _replace_module(parent: nn.Module, child_name: str, new_child: nn.Module) -> None:
    setattr(parent, child_name, new_child)


def _resolve_sub_path(
    block: nn.Module, sub_path: str
) -> Optional[Tuple[nn.Module, str, nn.Linear]]:
    """Walk a dotted sub-path under ``block`` and return ``(parent, leaf_name, linear)``.

    ``sub_path`` examples:
        ``self_attn.q_proj``
        ``mlp.experts.96.gate_proj``
    Numeric segments index into ``nn.ModuleList``/``nn.Sequential``; otherwise
    we ``getattr``.  Returns ``None`` if the path doesn't resolve to an
    ``nn.Linear``.
    """
    parts = sub_path.split(".")
    if not parts:
        return None
    current = block
    for seg in parts[:-1]:
        if seg.isdigit():
            try:
                current = current[int(seg)]
            except (TypeError, IndexError):
                return None
        else:
            current = getattr(current, seg, None)
            if current is None:
                return None
    leaf = parts[-1]
    target = getattr(current, leaf, None) if not leaf.isdigit() else (
        current[int(leaf)] if hasattr(current, "__getitem__") else None
    )
    if not isinstance(target, nn.Linear):
        return None
    return current, leaf, target


def _to_2d_attention_mask(
    attention_mask: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Coerce a possibly-4D HF attention mask down to ``[B, T]`` ``{0, 1}``.

    Returns None if no mask is provided.
    """
    if attention_mask is None:
        return None
    am = attention_mask
    if am.dim() == 2:
        return am
    if am.dim() == 4:
        # HF attention mask is [B, 1, Tq, Tk] with -inf on padded keys, 0 elsewhere.
        # The "key" axis tells us which key positions are real.
        # Take any row (e.g., the last query) and check finite-ness.
        last = am[:, 0, -1, :]  # [B, Tk]
        return (last > -1e4).to(dtype=hidden_states.dtype).to(am.device)
    # Fallback: assume all-real
    return None


def _attach_block_router(
    block: nn.Module,
    gater: LayerCapabilityGater,
    ctx: GaterContext,
) -> None:
    """Install hooks so the gater fires on the block's input and ``ctx.weights``
    is populated before each forward and cleared after.

    Uses ``register_forward_pre_hook(with_kwargs=True)`` rather than
    monkey-patching ``block.forward`` so the wiring composes with gradient
    checkpointing: ``torch.utils.checkpoint`` re-runs ``block.__call__`` on
    backward, which re-fires the pre-hook and recomputes the gater output
    with grad enabled.
    """
    def _pre_hook(_module, args, kwargs):
        hidden_states = args[0] if args else kwargs.get("hidden_states")
        if hidden_states is None:
            return None
        # Passive gaters have no parameters; their forward returns a locked
        # tensor that gates handle moving to the right device on its own.
        first_param = next(gater.parameters(), None)
        if first_param is not None and first_param.device != hidden_states.device:
            gater.to(device=hidden_states.device)
        am2d = _to_2d_attention_mask(kwargs.get("attention_mask"), hidden_states)
        ctx.weights = gater(hidden_states, attention_mask=am2d)
        return None

    def _post_hook(_module, _args, output):
        ctx.weights = None
        return output

    block.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    block.register_forward_hook(_post_hook)


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_capability_model(
    cfg: CapabilityModelConfig,
    *,
    tokenizer: Optional[object] = None,
    return_meta: bool = False,
):
    """Build the routed mixture model.

    Args:
        cfg: ``CapabilityModelConfig`` (base model + adapter list + flags).
        tokenizer: optional, returned unchanged for caller convenience.
        return_meta: if True, also return a dict with diagnostic info.

    Returns:
        ``model`` or ``(model, meta)``.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy import

    if cfg.routing_scope not in ("per_layer", "global"):
        raise ValueError(
            f"routing_scope must be 'per_layer' or 'global', got {cfg.routing_scope!r}"
        )
    if cfg.routing_scope == "global" and cfg.routing_granularity != "trajectory":
        raise ValueError(
            "routing_scope='global' requires routing_granularity='trajectory' "
            f"(got {cfg.routing_granularity!r}). Token-level global routing is "
            "not supported: a single decision per token across all layers "
            "would require per-token expert assignments to be plumbed through "
            "the MoE expert path, which the current MultiLoRAGatedLinear does "
            "not do."
        )

    # ---- 1. Load base model -------------------------------------------------
    print(f"[capability_router] Loading base model: {cfg.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=cfg.dtype,
        device_map=cfg.device_map,
        cache_dir=cfg.cache_dir,
        trust_remote_code=True,
    )
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.base_model, cache_dir=cfg.cache_dir, trust_remote_code=True
        )

    hidden_size = _get_hidden_size(model)
    blocks = _find_decoder_blocks(model)
    n_layers = len(blocks)
    print(f"[capability_router] hidden_size={hidden_size}, n_layers={n_layers}")

    # ---- 2. Load each adapter into a per-(layer, proj) dict -----------------
    print(f"[capability_router] Loading {len(cfg.capability_adapters)} capability adapters")
    per_adapter: List[Dict[Tuple[int, str], Tuple[torch.Tensor, torch.Tensor]]] = []
    per_adapter_scaling: List[float] = []
    for repo in cfg.capability_adapters:
        adir = _resolve_adapter_dir(repo, cfg.cache_dir)
        weights, scaling, rank = _load_adapter(adir)
        per_adapter.append(weights)
        per_adapter_scaling.append(scaling)
        print(
            f"[capability_router]   {repo}: rank={rank}, scaling={scaling:.3f}, "
            f"#layers_covered={len({l for (l, _) in weights})}"
        )

    # ---- 3. Allocate per-block GaterContexts + LayerCapabilityGaters --------
    num_capabilities = cfg.num_capabilities
    is_global = cfg.routing_scope == "global"
    gaters: List[LayerCapabilityGater] = []
    contexts: List[GaterContext] = []
    for layer_idx, block in enumerate(blocks):
        gater = LayerCapabilityGater(
            hidden_size=hidden_size,
            num_capabilities=num_capabilities,
            granularity=cfg.routing_granularity,
            passive=is_global,
        ).to(dtype=torch.float32)  # gater stays in fp32 for stable softmax / training
        # Pin the gater to the same device as the block's first parameter.
        try:
            block_device = next(block.parameters()).device
            gater = gater.to(device=block_device)
        except StopIteration:
            pass
        # Attach as a submodule so it is captured by named_modules / state_dict.
        block.add_module("capability_gater", gater)
        ctx = GaterContext()
        gaters.append(gater)
        contexts.append(ctx)

    # ---- 4. Replace LoRA-adapted projections with MultiLoRAGatedLinear -----
    # Build the union of (layer, sub_path) keys across all adapters whose
    # leaf name is in ``cfg.target_modules``. Each adapter contributes a slot
    # in the per-capability list (None if it doesn't cover this sub_path).
    target_leaves = set(cfg.target_modules)
    n_replaced = 0
    n_skipped_unresolved = 0
    coverage: Dict[str, int] = {repo: 0 for repo in cfg.capability_adapters}
    for layer_idx, block in enumerate(blocks):
        ctx = contexts[layer_idx]

        # Union of sub-paths this layer's adapters touch (filtered by leaf).
        layer_sub_paths: List[str] = []
        seen: set[str] = set()
        for weights in per_adapter:
            for (l, sp) in weights.keys():
                if l != layer_idx:
                    continue
                if sp in seen:
                    continue
                if sp.split(".")[-1] not in target_leaves:
                    continue
                seen.add(sp)
                layer_sub_paths.append(sp)

        for sub_path in layer_sub_paths:
            located = _resolve_sub_path(block, sub_path)
            if located is None:
                # Adapter referenced a path that doesn't exist on this model
                # (e.g. fused Qwen3MoeExperts in transformers >= 5.x where
                # ``mlp.experts.{i}.gate_proj`` no longer exists as a Linear).
                n_skipped_unresolved += 1
                continue
            parent, child_name, base_linear = located

            lora_pairs: List[Optional[Tuple[torch.Tensor, torch.Tensor, float]]] = []
            if cfg.include_base_capability:
                lora_pairs.append(None)  # base slot — no LoRA contribution
            for adapter_idx, weights in enumerate(per_adapter):
                pair = weights.get((layer_idx, sub_path))
                scaling = per_adapter_scaling[adapter_idx]
                if pair is None:
                    lora_pairs.append(None)
                else:
                    A, B = pair
                    A = A.to(dtype=cfg.dtype, device=base_linear.weight.device)
                    B = B.to(dtype=cfg.dtype, device=base_linear.weight.device)
                    lora_pairs.append((A, B, scaling))
                    coverage[cfg.capability_adapters[adapter_idx]] += 1

            wrapped = MultiLoRAGatedLinear(
                base_linear=base_linear,
                lora_pairs=lora_pairs,
                gater_ctx=ctx,
            )
            _replace_module(parent, child_name, wrapped)
            n_replaced += 1

        _attach_block_router(block, gaters[layer_idx], ctx)

    print(
        f"[capability_router] Replaced {n_replaced} projections with "
        f"MultiLoRAGatedLinear; gaters: {n_layers}, "
        f"capabilities: {num_capabilities} ({cfg.capability_names})"
    )
    if n_skipped_unresolved:
        print(
            f"[capability_router] WARNING: {n_skipped_unresolved} LoRA paths "
            "did not resolve to standalone nn.Linear modules — typically "
            "mlp.experts.{i}.{gate,up,down}_proj under transformers>=5 "
            "(fused Qwen3MoeExperts). Capability routing falls back to "
            "attention only, discarding ~99% of each adapter. "
            "Pin transformers<5 for full coverage."
        )
    for repo, c in coverage.items():
        print(f"[capability_router]   adapter coverage: {repo} -> {c} projections")

    # ---- 5. Freeze everything except gaters --------------------------------
    freeze_all_except_gaters(model)

    # ---- 6. (Global mode only) build TrajectoryGlobalRouter ----------------
    global_router: Optional[TrajectoryGlobalRouter] = None
    if is_global:
        global_router = TrajectoryGlobalRouter(
            hidden_size=hidden_size, num_capabilities=num_capabilities,
        ).to(dtype=torch.float32)
        # Pin to the embed-table's device so feature pooling stays on-shard.
        try:
            embed_device = model.get_input_embeddings().weight.device
            global_router = global_router.to(device=embed_device)
        except (AttributeError, StopIteration):
            pass
        # Router params are trainable; the model graph itself is frozen.
        for p in global_router.parameters():
            p.requires_grad = True
        print(
            f"[capability_router] routing_scope=global: built "
            f"TrajectoryGlobalRouter(hidden_size={hidden_size}, "
            f"num_capabilities={num_capabilities}); "
            f"per-block gaters are passive."
        )

    # Quick sanity: report the trainable surface for the active scope.
    n_train_model = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total_model = sum(p.numel() for p in model.parameters())
    n_train_router = (
        sum(p.numel() for p in global_router.parameters() if p.requires_grad)
        if global_router is not None else 0
    )
    print(
        f"[capability_router] Trainable params: model={n_train_model:,} / "
        f"{n_total_model:,}, router={n_train_router:,}"
    )

    if return_meta:
        meta = {
            "tokenizer": tokenizer,
            "hidden_size": hidden_size,
            "n_layers": n_layers,
            "num_capabilities": num_capabilities,
            "capability_names": cfg.capability_names,
            "gaters": gaters,
            "contexts": contexts,
            "coverage": coverage,
            "global_router": global_router,
        }
        return model, meta
    return model
