"""
Freeze-verification utilities for capability routing.

The spec (constraint #3) requires an explicit assertion that the base model
and every LoRA capability adapter are frozen before Step 5 (warm start) and
Step 6 (GRPO). Only routing parameters may have ``requires_grad == True``:

* ``routing_scope='per_layer'``: per-block ``LayerCapabilityGater.linear``.
* ``routing_scope='global'``:    ``TrajectoryGlobalRouter.head`` (gaters
                                 are passive — no Linear).

``additional_trainable`` lets callers (e.g. global mode) extend the set of
allowed trainable modules without weakening the leak check.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import torch.nn as nn

from .gater import LayerCapabilityGater


def gater_parameters(model: nn.Module) -> Iterable[Tuple[str, nn.Parameter]]:
    """Yield ``(qualified_name, parameter)`` for every gater parameter."""
    for module_name, module in model.named_modules():
        if isinstance(module, LayerCapabilityGater):
            for pname, p in module.named_parameters(recurse=True):
                yield f"{module_name}.{pname}" if module_name else pname, p


def assert_frozen(
    model: nn.Module,
    *,
    additional_trainable: Optional[Iterable[nn.Module]] = None,
) -> None:
    """Raise if any param outside (gaters ∪ ``additional_trainable``) has
    ``requires_grad=True``, or if no allowed param is trainable at all.

    Call this immediately before Step 5 / Step 5b / Step 6 to catch
    accidental thaw of base / capability LoRA params. Pass the global
    router via ``additional_trainable=[global_router]`` in global mode.
    """
    allowed_param_ids: set[int] = set()
    for _, p in gater_parameters(model):
        allowed_param_ids.add(id(p))
    additional_modules = list(additional_trainable) if additional_trainable else []
    for module in additional_modules:
        for p in module.parameters():
            allowed_param_ids.add(id(p))

    leaks: List[str] = []
    for pname, p in model.named_parameters():
        if id(p) in allowed_param_ids:
            continue
        if p.requires_grad:
            leaks.append(pname)

    if leaks:
        head = "\n  ".join(leaks[:20])
        more = "" if len(leaks) <= 20 else f"\n  ... and {len(leaks) - 20} more"
        raise RuntimeError(
            "[capability_router] Found non-routing parameters with "
            f"requires_grad=True ({len(leaks)} total). The base model and all "
            "capability LoRA adapters must be frozen before warm-start / GRPO."
            f"\n  {head}{more}"
        )

    n_gater_trainable = sum(1 for _, p in gater_parameters(model) if p.requires_grad)
    n_additional_trainable = sum(
        1 for module in additional_modules for p in module.parameters() if p.requires_grad
    )
    if n_gater_trainable + n_additional_trainable == 0:
        raise RuntimeError(
            "[capability_router] No trainable routing parameters found in "
            "either gaters or additional_trainable modules. Per-layer mode "
            "expects gater Linears with requires_grad=True; global mode "
            "expects a TrajectoryGlobalRouter passed via additional_trainable."
        )


def freeze_all_except_gaters(model: nn.Module) -> None:
    """Set ``requires_grad=False`` on every param except gaters'.

    In global mode all gaters are passive (no params) and this is a pure
    freeze — the trainable surface lives on the global router, which is
    not a child of ``model`` and so is unaffected.
    """
    gater_param_ids: set[int] = set()
    for _, p in gater_parameters(model):
        gater_param_ids.add(id(p))
        p.requires_grad = True
    for _, p in model.named_parameters():
        if id(p) not in gater_param_ids:
            p.requires_grad = False
