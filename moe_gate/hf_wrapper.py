"""
Hugging Face wrapper for the capability-routed model.

Provides:
  * ``CapabilityRoutedConfig``: a ``PretrainedConfig`` carrying base model id,
    capability adapter list, target modules, routing granularity.
  * ``CapabilityRoutedForCausalLM``: a thin ``PreTrainedModel`` wrapper that
    constructs the gated mixture via ``build_capability_model`` and dispatches
    forward/generate to the inner HF model.
  * ``save_capability_routed_checkpoint``: writes a self-contained directory
    that loads via ``AutoModelForCausalLM.from_pretrained(... trust_remote_code=True)``.
  * ``push_capability_routed_to_hub``: convenience wrapper.

The exported directory contains:
  - config.json                    (model_type + auto_map + capability config)
  - gater_state.pt                 (per-block LayerCapabilityGater weights)
  - modeling_capability_routed.py  (this wrapper, copied verbatim)
  - capability_router/             (gater / multi_lora / model_builder / freeze_utils)
  - README.md                      (usage snippet)

The base Qwen3-30B and the 4 LoRA adapters are NOT serialized — they are
fetched from the Hub at load time using the ids stored in config. This keeps
the published checkpoint small (~2 MB) and avoids re-uploading pretrained
weights.
"""
from __future__ import annotations

import json
import os
import shutil
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel

from .gater import (
    LayerCapabilityGater,
    TrajectoryGlobalRouter,
    base_only_one_hot,
    bind_global_routing,
    lock_gaters_after_first_forward,
    pool_prompt_features,
    unlock_all,
)
from .model_builder import (
    ALL_TARGET_MODULES,
    DEFAULT_CAPABILITY_REPOS,
    DEFAULT_TARGET_MODULES,
    CapabilityModelConfig,
    build_capability_model,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class CapabilityRoutedConfig(PretrainedConfig):
    model_type = "capability_routed_qwen3"

    def __init__(
        self,
        base_model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
        capability_adapters: Optional[List[str]] = None,
        include_base_capability: bool = True,
        target_modules: Optional[List[str]] = None,
        routing_granularity: str = "trajectory",
        routing_scope: str = "per_layer",
        torch_dtype: str = "bfloat16",
        **kwargs,
    ) -> None:
        self.base_model = base_model
        self.capability_adapters = list(capability_adapters or DEFAULT_CAPABILITY_REPOS)
        self.include_base_capability = bool(include_base_capability)
        self.target_modules = list(target_modules or DEFAULT_TARGET_MODULES)
        self.routing_granularity = routing_granularity
        self.routing_scope = routing_scope
        self.torch_dtype = torch_dtype
        super().__init__(**kwargs)

    def to_capability_model_config(
        self,
        cache_dir: Optional[str] = None,
        device_map: Optional[str] = "auto",
    ) -> CapabilityModelConfig:
        dtype_map = {
            "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
            "float16": torch.float16, "fp16": torch.float16,
            "float32": torch.float32, "fp32": torch.float32,
        }
        return CapabilityModelConfig(
            base_model=self.base_model,
            capability_adapters=self.capability_adapters,
            include_base_capability=self.include_base_capability,
            target_modules=tuple(self.target_modules),
            routing_granularity=self.routing_granularity,
            routing_scope=self.routing_scope,
            dtype=dtype_map.get(self.torch_dtype, torch.bfloat16),
            cache_dir=cache_dir,
            device_map=device_map,
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CapabilityRoutedForCausalLM(PreTrainedModel):
    """Self-contained wrapper that rebuilds the gated mixture on load."""

    config_class = CapabilityRoutedConfig
    base_model_prefix = "inner"
    supports_gradient_checkpointing = True

    def __init__(self, config: CapabilityRoutedConfig) -> None:
        super().__init__(config)
        cfg = config.to_capability_model_config(
            cache_dir=os.environ.get("HF_HOME"),
            device_map=os.environ.get("CAP_ROUTER_DEVICE_MAP", "auto"),
        )
        inner, meta = build_capability_model(cfg, return_meta=True)
        self.inner = inner
        self._capability_gaters: List[LayerCapabilityGater] = meta["gaters"]
        self.tokenizer = meta["tokenizer"]
        # In global mode this is the single classifier head; in per-layer mode
        # it's None and the per-block gater Linears carry the trainable state.
        self.global_router: Optional[TrajectoryGlobalRouter] = meta.get("global_router")

    def forward(self, *args, **kwargs):
        return self.inner(*args, **kwargs)

    def generate(self, *args, **kwargs):
        if self.config.routing_scope == "global":
            return self._global_generate(*args, **kwargs)
        # Per-layer trajectory mode: lock all gaters to prompt-only routing
        # right after generate's internal prefill, so per-token forwards don't
        # re-pool a single new token in place of the original prompt.
        if self.config.routing_granularity != "trajectory":
            return self.inner.generate(*args, **kwargs)
        handle = lock_gaters_after_first_forward(self.inner, self._capability_gaters)
        try:
            return self.inner.generate(*args, **kwargs)
        finally:
            handle.remove()
            for g in self._capability_gaters:
                g.unlock()

    def _global_generate(self, *args, **kwargs):
        """Two-phase generate:

        Phase A (no_grad): lock all gaters to base-only one-hot, prefill
        the prompt to extract last-token features, then ask the global
        router for the routing decision (argmax-one-hot).

        Phase B: re-lock to that decision and call inner.generate as
        usual. The decode-step forwards see the locked decision via the
        same pre-hook plumbing per-layer mode uses.
        """
        if self.global_router is None:
            raise RuntimeError(
                "routing_scope='global' but no TrajectoryGlobalRouter is "
                "attached. Did from_pretrained skip loading global_router_state.pt?"
            )
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is None:
            raise ValueError(
                "global generate requires input_ids (positional or keyword)."
            )
        attention_mask = kwargs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        embed_dev = self.inner.get_input_embeddings().weight.device
        input_ids_dev = input_ids.to(embed_dev)
        attention_mask_dev = attention_mask.to(embed_dev)

        num_caps = self._capability_gaters[0].num_capabilities

        with torch.no_grad():
            base_lock = base_only_one_hot(
                num_caps, device=embed_dev, dtype=torch.float32,
            )
            bind_global_routing(
                base_lock, self._capability_gaters, detach=True,
            )
            features = pool_prompt_features(
                self.inner, input_ids_dev, attention_mask_dev,
            )
            router_dev = next(self.global_router.parameters()).device
            decision = self.global_router.route(
                features.to(router_dev), mode="hard",
            )

        bind_global_routing(decision, self._capability_gaters, detach=True)
        try:
            return self.inner.generate(*args, **kwargs)
        finally:
            unlock_all(self._capability_gaters)

    # ---- State serialization ------------------------------------------------
    # Per-layer mode: gater_state.pt holds the per-block Linear weights.
    # Global mode:    global_router_state.pt holds the classifier head.
    # In both modes config.json carries routing_scope so loaders pick correctly.

    def gater_state_dict(self) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for layer_idx, g in enumerate(self._capability_gaters):
            for k, v in g.state_dict().items():
                out[f"layer{layer_idx}.{k}"] = v.detach().cpu()
        return out

    def load_gater_state(self, state: Dict[str, torch.Tensor]) -> None:
        for layer_idx, g in enumerate(self._capability_gaters):
            sub = {
                k.split(".", 1)[1]: v
                for k, v in state.items()
                if k.startswith(f"layer{layer_idx}.")
            }
            if sub:
                g.load_state_dict(sub, strict=True)

    def global_router_state_dict(self) -> Dict[str, torch.Tensor]:
        if self.global_router is None:
            return {}
        return {k: v.detach().cpu() for k, v in self.global_router.state_dict().items()}

    def load_global_router_state(self, state: Dict[str, torch.Tensor]) -> None:
        if self.global_router is None:
            raise RuntimeError(
                "Cannot load global router state: routing_scope is "
                f"{self.config.routing_scope!r}, no TrajectoryGlobalRouter on this wrapper."
            )
        self.global_router.load_state_dict(state, strict=True)

    def save_pretrained(self, save_directory, **kwargs):
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        cfg_dict = self.config.to_dict()
        cfg_dict["auto_map"] = {
            "AutoConfig": "modeling_capability_routed.CapabilityRoutedConfig",
            "AutoModelForCausalLM": "modeling_capability_routed.CapabilityRoutedForCausalLM",
        }
        with (save_directory / "config.json").open("w") as f:
            json.dump(cfg_dict, f, indent=2)
        if self.config.routing_scope == "global":
            torch.save(
                self.global_router_state_dict(),
                save_directory / "global_router_state.pt",
            )
        else:
            torch.save(self.gater_state_dict(), save_directory / "gater_state.pt")
        _copy_modeling_code(save_directory)
        _write_readme(save_directory, self.config)
        return str(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        config = kwargs.pop("config", None) or CapabilityRoutedConfig.from_pretrained(
            pretrained_model_name_or_path,
        )
        model = cls(config)

        from huggingface_hub import hf_hub_download

        if config.routing_scope == "global":
            fname = "global_router_state.pt"
            local = Path(pretrained_model_name_or_path) / fname
            if local.exists():
                state_path = local
            else:
                state_path = Path(hf_hub_download(
                    repo_id=str(pretrained_model_name_or_path),
                    filename=fname,
                    cache_dir=os.environ.get("HF_HOME"),
                ))
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            model.load_global_router_state(state)
            return model

        gater_path = Path(pretrained_model_name_or_path) / "gater_state.pt"
        if not gater_path.exists():
            gater_path = Path(
                hf_hub_download(
                    repo_id=pretrained_model_name_or_path,
                    filename="gater_state.pt",
                    cache_dir=os.environ.get("HF_HOME"),
                )
            )
        state = torch.load(gater_path, map_location="cpu", weights_only=False)
        model.load_gater_state(state)
        return model


# ---------------------------------------------------------------------------
# Save / push helpers
# ---------------------------------------------------------------------------

def save_capability_routed_checkpoint(
    save_directory: os.PathLike,
    *,
    base_model: str,
    capability_adapters: List[str],
    routing_granularity: str,
    routing_scope: str = "per_layer",
    target_modules: Optional[List[str]] = None,
    include_base_capability: bool = True,
    gater_state: Optional[Dict[str, torch.Tensor]] = None,
    gater_state_path: Optional[os.PathLike] = None,
    global_router_state: Optional[Dict[str, torch.Tensor]] = None,
    global_router_state_path: Optional[os.PathLike] = None,
) -> Path:
    """Write a self-contained HF checkpoint to ``save_directory``.

    Per-layer mode: provide ``gater_state`` or ``gater_state_path``.
    Global mode:    provide ``global_router_state`` or ``global_router_state_path``.
    """
    save_directory = Path(save_directory)
    save_directory.mkdir(parents=True, exist_ok=True)

    if routing_scope not in ("per_layer", "global"):
        raise ValueError(
            f"routing_scope must be 'per_layer' or 'global', got {routing_scope!r}"
        )

    config = CapabilityRoutedConfig(
        base_model=base_model,
        capability_adapters=capability_adapters,
        include_base_capability=include_base_capability,
        target_modules=list(target_modules or DEFAULT_TARGET_MODULES),
        routing_granularity=routing_granularity,
        routing_scope=routing_scope,
    )
    cfg_dict = config.to_dict()
    cfg_dict["auto_map"] = {
        "AutoConfig": "modeling_capability_routed.CapabilityRoutedConfig",
        "AutoModelForCausalLM": "modeling_capability_routed.CapabilityRoutedForCausalLM",
    }
    with (save_directory / "config.json").open("w") as f:
        json.dump(cfg_dict, f, indent=2)

    if routing_scope == "global":
        if global_router_state is None:
            if global_router_state_path is None:
                raise ValueError(
                    "routing_scope='global' requires global_router_state or "
                    "global_router_state_path."
                )
            loaded = torch.load(
                global_router_state_path, map_location="cpu", weights_only=False,
            )
            # step5_warmstart_global saves under "router_state".
            global_router_state = loaded.get("router_state", loaded)
        torch.save(global_router_state, save_directory / "global_router_state.pt")
    else:
        if gater_state is None:
            if gater_state_path is None:
                raise ValueError(
                    "routing_scope='per_layer' requires gater_state or gater_state_path."
                )
            loaded = torch.load(gater_state_path, map_location="cpu", weights_only=False)
            # step5/step6 save a dict with a "gater_state" sub-key.
            gater_state = loaded.get("gater_state", loaded)
        torch.save(gater_state, save_directory / "gater_state.pt")

    _copy_modeling_code(save_directory)
    _write_readme(save_directory, config)
    return save_directory


def push_capability_routed_to_hub(
    save_directory: os.PathLike,
    repo_id: str,
    *,
    private: bool = False,
    commit_message: str = "Upload capability-routed checkpoint",
    token: Optional[str] = None,
) -> str:
    from huggingface_hub import HfApi  # lazy import

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(save_directory),
        repo_id=repo_id,
        commit_message=commit_message,
        token=token,
    )
    return f"https://huggingface.co/{repo_id}"


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _copy_modeling_code(save_directory: Path) -> None:
    """Write a single self-contained ``modeling_capability_routed.py``.

    The file inlines every class/function the runtime needs (gater,
    MultiLoRAGatedLinear, build_capability_model, the wrapper) so that
    ``trust_remote_code=True`` can load it without our package on the path.
    Only ``torch``, ``transformers``, ``safetensors``, and ``huggingface_hub``
    are required at load time — all standard pip-installables.
    """
    pkg_root = Path(__file__).resolve().parent

    # Concatenate the bodies of the source files (with their docstrings).
    # We strip ``from .X import Y`` lines because they reference siblings.
    # Order matters: each file may reference earlier ones.
    parts = []
    parts.append(_INLINED_HEADER)
    for fname in (
        "gater.py",
        "multi_lora.py",
        "freeze_utils.py",
        "model_builder.py",
    ):
        src = (pkg_root / fname).read_text()
        parts.append(_strip_relative_imports(src, banner=f"# === inlined from {fname} ==="))
    parts.append(_INLINED_WRAPPER)

    out = "\n\n".join(parts)
    (save_directory / "modeling_capability_routed.py").write_text(out)


def _strip_relative_imports(src: str, banner: str) -> str:
    """Inline-cleanup for concatenation:

    - Drop the leading module docstring (we already have one in the header).
    - Drop ``from __future__`` imports (must appear only at the very top).
    - Drop ``from .X`` and ``from capability_router...`` imports (siblings
      that are inlined in the same file just above this section).
    """
    lines = src.splitlines()
    # State machine: SEEK_DOC -> IN_DOC -> AFTER_DOC.
    state = "SEEK_DOC"
    out_lines: List[str] = [banner]
    open_quote: Optional[str] = None
    for line in lines:
        stripped = line.lstrip()

        if state == "SEEK_DOC":
            if stripped == "" or stripped.startswith("#"):
                continue
            if stripped.startswith('"""') or stripped.startswith("'''"):
                quote = stripped[:3]
                # Single-line docstring (e.g. """one liner""") -> close immediately.
                rest = stripped[3:]
                if rest.endswith(quote):
                    state = "AFTER_DOC"
                else:
                    state = "IN_DOC"
                    open_quote = quote
                continue
            # No module docstring at all.
            state = "AFTER_DOC"
            # fall through to emit this line

        if state == "IN_DOC":
            assert open_quote is not None
            if open_quote in stripped:
                state = "AFTER_DOC"
                open_quote = None
            continue

        # AFTER_DOC: filter imports
        if stripped.startswith("from __future__"):
            continue
        if stripped.startswith("from .") or stripped.startswith("from capability_router"):
            continue
        out_lines.append(line)

    return "\n".join(out_lines)


# Top-of-file imports the inlined module needs.
_INLINED_HEADER = textwrap.dedent('''\
    # Auto-generated by capability_router.hf_wrapper. Self-contained — the only
    # dependencies are torch, transformers, safetensors, and huggingface_hub.
    """Capability-routed Qwen3 single-file modeling code.

    Loaded via AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True).
    """
    from __future__ import annotations

    import json
    import os
    import re
    import threading
    import zipfile
    from dataclasses import dataclass, field
    from pathlib import Path
    from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from safetensors.torch import load_file as load_safetensors
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import PretrainedConfig, PreTrainedModel
''')


# Wrapper class definitions that wire CapabilityRoutedConfig +
# CapabilityRoutedForCausalLM to the inlined helpers above. Mirrors
# ``CapabilityRoutedConfig`` / ``CapabilityRoutedForCausalLM`` in this file
# but without the relative imports.
_INLINED_WRAPPER = textwrap.dedent('''\
    # === inlined wrapper ============================================================

    class CapabilityRoutedConfig(PretrainedConfig):
        model_type = "capability_routed_qwen3"

        def __init__(
            self,
            base_model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
            capability_adapters: Optional[List[str]] = None,
            include_base_capability: bool = True,
            target_modules: Optional[List[str]] = None,
            routing_granularity: str = "trajectory",
            routing_scope: str = "per_layer",
            torch_dtype: str = "bfloat16",
            **kwargs,
        ) -> None:
            self.base_model = base_model
            self.capability_adapters = list(
                capability_adapters or list(DEFAULT_CAPABILITY_REPOS)
            )
            self.include_base_capability = bool(include_base_capability)
            self.target_modules = list(target_modules or list(DEFAULT_TARGET_MODULES))
            self.routing_granularity = routing_granularity
            self.routing_scope = routing_scope
            self.torch_dtype = torch_dtype
            super().__init__(**kwargs)


    class CapabilityRoutedForCausalLM(PreTrainedModel):
        config_class = CapabilityRoutedConfig
        base_model_prefix = "inner"
        supports_gradient_checkpointing = True

        def __init__(self, config: CapabilityRoutedConfig) -> None:
            super().__init__(config)
            dtype_map = {
                "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
                "float16": torch.float16, "fp16": torch.float16,
                "float32": torch.float32, "fp32": torch.float32,
            }
            cfg = CapabilityModelConfig(
                base_model=config.base_model,
                capability_adapters=list(config.capability_adapters),
                include_base_capability=config.include_base_capability,
                target_modules=tuple(config.target_modules),
                routing_granularity=config.routing_granularity,
                routing_scope=config.routing_scope,
                dtype=dtype_map.get(config.torch_dtype, torch.bfloat16),
                cache_dir=os.environ.get("HF_HOME"),
                device_map=os.environ.get("CAP_ROUTER_DEVICE_MAP", "auto"),
            )
            inner, meta = build_capability_model(cfg, return_meta=True)
            self.inner = inner
            self._capability_gaters: List[LayerCapabilityGater] = meta["gaters"]
            self.tokenizer = meta["tokenizer"]
            self.global_router: Optional[TrajectoryGlobalRouter] = meta.get("global_router")

        def forward(self, *args, **kwargs):
            return self.inner(*args, **kwargs)

        def generate(self, *args, **kwargs):
            if self.config.routing_scope == "global":
                return self._global_generate(*args, **kwargs)
            if self.config.routing_granularity != "trajectory":
                return self.inner.generate(*args, **kwargs)
            handle = lock_gaters_after_first_forward(
                self.inner, self._capability_gaters
            )
            try:
                return self.inner.generate(*args, **kwargs)
            finally:
                handle.remove()
                for g in self._capability_gaters:
                    g.unlock()

        def _global_generate(self, *args, **kwargs):
            if self.global_router is None:
                raise RuntimeError(
                    "routing_scope='global' but no TrajectoryGlobalRouter is "
                    "attached. Did from_pretrained skip global_router_state.pt?"
                )
            input_ids = kwargs.get("input_ids")
            if input_ids is None and args:
                input_ids = args[0]
            if input_ids is None:
                raise ValueError("global generate requires input_ids.")
            attention_mask = kwargs.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)

            embed_dev = self.inner.get_input_embeddings().weight.device
            input_ids_dev = input_ids.to(embed_dev)
            attention_mask_dev = attention_mask.to(embed_dev)

            num_caps = self._capability_gaters[0].num_capabilities

            with torch.no_grad():
                base_lock = base_only_one_hot(
                    num_caps, device=embed_dev, dtype=torch.float32,
                )
                bind_global_routing(
                    base_lock, self._capability_gaters, detach=True,
                )
                features = pool_prompt_features(
                    self.inner, input_ids_dev, attention_mask_dev,
                )
                router_dev = next(self.global_router.parameters()).device
                decision = self.global_router.route(
                    features.to(router_dev), mode="hard",
                )

            bind_global_routing(decision, self._capability_gaters, detach=True)
            try:
                return self.inner.generate(*args, **kwargs)
            finally:
                unlock_all(self._capability_gaters)

        def gater_state_dict(self) -> Dict[str, torch.Tensor]:
            out: Dict[str, torch.Tensor] = {}
            for layer_idx, g in enumerate(self._capability_gaters):
                for k, v in g.state_dict().items():
                    out[f"layer{layer_idx}.{k}"] = v.detach().cpu()
            return out

        def load_gater_state(self, state: Dict[str, torch.Tensor]) -> None:
            for layer_idx, g in enumerate(self._capability_gaters):
                sub = {
                    k.split(".", 1)[1]: v for k, v in state.items()
                    if k.startswith(f"layer{layer_idx}.")
                }
                if sub:
                    g.load_state_dict(sub, strict=True)

        def global_router_state_dict(self) -> Dict[str, torch.Tensor]:
            if self.global_router is None:
                return {}
            return {
                k: v.detach().cpu() for k, v in self.global_router.state_dict().items()
            }

        def load_global_router_state(self, state: Dict[str, torch.Tensor]) -> None:
            if self.global_router is None:
                raise RuntimeError(
                    "Cannot load global router state: no TrajectoryGlobalRouter "
                    "on this wrapper (config.routing_scope="
                    f"{self.config.routing_scope!r})."
                )
            self.global_router.load_state_dict(state, strict=True)

        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
            from huggingface_hub import hf_hub_download

            config = kwargs.pop("config", None) or CapabilityRoutedConfig.from_pretrained(
                pretrained_model_name_or_path,
            )
            model = cls(config)

            local = Path(pretrained_model_name_or_path)

            if config.routing_scope == "global":
                fname = "global_router_state.pt"
                if local.exists() and (local / fname).exists():
                    state_path = local / fname
                else:
                    state_path = Path(hf_hub_download(
                        repo_id=str(pretrained_model_name_or_path),
                        filename=fname,
                        cache_dir=os.environ.get("HF_HOME"),
                    ))
                state = torch.load(state_path, map_location="cpu", weights_only=False)
                model.load_global_router_state(state)
                return model

            if local.exists() and (local / "gater_state.pt").exists():
                gater_path = local / "gater_state.pt"
            else:
                gater_path = Path(hf_hub_download(
                    repo_id=str(pretrained_model_name_or_path),
                    filename="gater_state.pt",
                    cache_dir=os.environ.get("HF_HOME"),
                ))
            state = torch.load(gater_path, map_location="cpu", weights_only=False)
            model.load_gater_state(state)
            return model
''')


def _write_readme(save_directory: Path, config: CapabilityRoutedConfig) -> None:
    if config.routing_scope == "global":
        scope_blurb = (
            "**Single classifier head over the prompt** (`TrajectoryGlobalRouter`): "
            "one capability is selected for the entire trajectory, applied "
            "identically at every decoder block. The classifier reads the "
            "frozen base model's last-hidden-state at the last prompt position. "
            "The selected slot's LoRA is the only one active for the whole "
            "trajectory; per-block gaters are passive lock holders only."
        )
    else:
        scope_blurb = (
            "**Per-block soft router** that mixes a small set of capability-"
            "specific LoRA adapters. One gate per decoder block, shared across "
            "every adapted projection in that block."
        )

    md = textwrap.dedent(f"""\
    ---
    library_name: transformers
    base_model: {config.base_model}
    tags:
      - capability-routing
      - lora-moe
      - tau-bench
    ---

    # Capability-Routed {config.base_model.split('/')[-1]}

    {scope_blurb}

    ## Routing
    - **Slots:** `{config.capability_adapters}`
    - **Base ("no adapter") slot included:** `{config.include_base_capability}`  \
      (slot 0 contributes 0 to the LoRA delta — fully selecting it is
      equivalent to running the base model unmodified)
    - **Target modules:** `{config.target_modules}`
    - **Granularity:** `{config.routing_granularity}`
    - **Scope:** `{config.routing_scope}`

    Each capability adapter carries ~18,624 LoRA pairs spanning attention
    (`q/k/v/o`) and every Qwen3 expert FFN (`{{gate,up,down}}_proj` × 128
    experts × 48 layers). Wrapping the expert FFN paths requires the
    unfused-experts layout (`transformers>=4.51,<5`); on `transformers>=5`
    those paths fail to resolve and only attention is routed (~1% of each
    adapter's capacity) — the build log surfaces this as
    `n_skipped_unresolved`.

    ```bash
    pip install "transformers>=4.51,<5"
    ```

    ## Forward
    For each adapted projection at block `l`:

        y = W·x + Σ_i  g_i(x_l) · scaling_i · B_i(A_i x)

    `g(x_l) = softmax(Linear(x_l.detach()))` is the capability gate.
    Base model and all LoRA adapters are frozen; only the gates train.

    ## Usage

    ```python
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        "<this-repo-id>", trust_remote_code=True, torch_dtype="bfloat16",
    )
    tok = AutoTokenizer.from_pretrained("{config.base_model}")

    out = model.generate(
        **tok("Hello", return_tensors="pt").to(model.device),
        max_new_tokens=64,
    )
    print(tok.decode(out[0], skip_special_tokens=True))
    ```

    The base model and LoRA adapters are downloaded from the Hub on first
    load; only the gate weights live in this repo.
    """)
    (save_directory / "README.md").write_text(md)
