"""
Routing-decision modules and helpers.

Two scopes are supported:

* ``routing_scope='per_layer'`` (default): one ``LayerCapabilityGater`` per
  decoder block. Each gater computes a softmax over the N capability slots
  from its block-local pooled hidden state:

      g(x_l) = Softmax(Linear(x_l.detach()))     # zero-init -> uniform start

  Detached input by spec — the gate routes on existing hidden states,
  never pushes gradient back into the base model.

* ``routing_scope='global'``: one ``TrajectoryGlobalRouter`` over the
  prompt produces a single decision broadcast to every block. Per-block
  gaters are constructed in passive mode (``passive=True``) — no Linear
  is allocated; they only serve as per-block lock holders. The global
  router's output is bound into every gater via
  ``bind_global_routing(...)``, then read back by the existing pre-hook
  plumbing as if it were a per-block decision.

Routing decisions in both modes flow through ``LayerCapabilityGater.lock``
into ``GaterContext`` and are consumed by ``MultiLoRAGatedLinear`` in the
adapted projections.
"""
from __future__ import annotations

import threading
from typing import Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Per-block gater
# ---------------------------------------------------------------------------

class LayerCapabilityGater(nn.Module):
    """One soft router per transformer block.

    ``passive=True`` skips the Linear allocation; the gater becomes a
    per-block lock holder only (used by ``routing_scope='global'``).
    """

    def __init__(
        self,
        hidden_size: int,
        num_capabilities: int,
        granularity: str = "trajectory",
        *,
        passive: bool = False,
    ) -> None:
        super().__init__()
        if granularity not in ("token", "trajectory"):
            raise ValueError(
                f"granularity must be 'token' or 'trajectory', got {granularity!r}"
            )
        self.hidden_size = int(hidden_size)
        self.num_capabilities = int(num_capabilities)
        self.granularity = granularity
        self.passive = bool(passive)

        if self.passive:
            # No Linear: routing is produced externally and bound via lock().
            self.linear = None
        else:
            self.linear = nn.Linear(hidden_size, num_capabilities, bias=True)
            nn.init.zeros_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

        # Lock state is per-thread so concurrent forwards (eval-during-training,
        # threaded inference, DataParallel replicas) don't race on a shared slot.
        self._tls = threading.local()

    @property
    def _locked(self) -> bool:
        return getattr(self._tls, "locked", False)

    @_locked.setter
    def _locked(self, value: bool) -> None:
        self._tls.locked = bool(value)

    @property
    def _locked_weights(self) -> Optional[torch.Tensor]:
        return getattr(self._tls, "locked_weights", None)

    @_locked_weights.setter
    def _locked_weights(self, value: Optional[torch.Tensor]) -> None:
        self._tls.locked_weights = value

    def lock(
        self,
        weights: Optional[torch.Tensor] = None,
        *,
        detach: bool = True,
    ) -> None:
        """Freeze the gater's output to ``weights``.

        ``detach=True`` (default) is the inference path: locked routing carries
        no gradient. ``detach=False`` requires an explicit ``weights=`` tensor
        captured with grad attached — the cached output from a prior forward
        was detached at stash time and cannot be re-attached.
        """
        if weights is None:
            if self._locked_weights is None:
                raise RuntimeError(
                    "Cannot lock without weights and without a prior forward."
                )
            if not detach:
                raise RuntimeError(
                    "lock(weights=None, detach=False) is invalid: the cached "
                    "_locked_weights tensor was detached during forward. Pass "
                    "a grad-attached `weights=` explicitly."
                )
        else:
            self._locked_weights = weights.detach() if detach else weights
        self._locked = True

    def unlock(self) -> None:
        self._locked = False

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self._locked and self._locked_weights is not None:
            return self._locked_weights.to(device=x.device)

        if self.passive:
            raise RuntimeError(
                "Passive LayerCapabilityGater forward called without a prior "
                "lock(). In routing_scope='global', external code must bind "
                "routing weights via lock(weights=...) before any model forward "
                "(see TrajectoryGlobalRouter / bind_global_routing)."
            )

        param_dtype = self.linear.weight.dtype
        x_in = x.detach().to(dtype=param_dtype)

        if self.granularity == "token":
            out = torch.softmax(self.linear(x_in), dim=-1)
            self._locked_weights = out.detach()
            return out

        if attention_mask is not None:
            mask = attention_mask.to(dtype=x_in.dtype, device=x_in.device).unsqueeze(-1)
            pooled = (x_in * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            pooled = x_in.mean(dim=1)

        weights = torch.softmax(self.linear(pooled), dim=-1).unsqueeze(1)
        self._locked_weights = weights.detach()
        return weights


def lock_gaters_after_first_forward(
    model: nn.Module,
    gaters: List["LayerCapabilityGater"],
) -> Any:
    """Install a one-shot ``forward`` hook on ``model`` that locks every gater
    after the model's first forward returns, then removes itself.

    Used to lock trajectory routing immediately after generate's internal
    prefill, avoiding a redundant manual prefill on the prompt.
    """
    fired = False
    handle: Any = None

    def _hook(_module, _args, _output):
        nonlocal fired, handle
        if fired:
            return
        fired = True
        for g in gaters:
            g.lock()
        if handle is not None:
            h, handle = handle, None
            h.remove()

    handle = model.register_forward_hook(_hook)
    return handle


# ---------------------------------------------------------------------------
# Global router (one decision per trajectory, broadcast to every block)
# ---------------------------------------------------------------------------

class TrajectoryGlobalRouter(nn.Module):
    """Single classifier head: pooled prompt features -> capability logits.

    ``forward`` returns logits (no softmax/argmax). Use ``route()`` for the
    broadcast-shape ``[B, 1, num_capabilities]`` tensor that gets fed into
    ``LayerCapabilityGater.lock`` via ``bind_global_routing``.
    """

    def __init__(
        self,
        hidden_size: int,
        num_capabilities: int,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_capabilities = int(num_capabilities)
        # Linear classifier head. Zero-init matches per-layer gater
        # convention (uniform softmax at start of training).
        self.head = nn.Linear(hidden_size, num_capabilities, bias=True)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, pooled_features: torch.Tensor) -> torch.Tensor:
        """Return logits ``[B, num_capabilities]``."""
        param_dtype = self.head.weight.dtype
        return self.head(pooled_features.to(dtype=param_dtype))

    def route(
        self,
        pooled_features: torch.Tensor,
        *,
        mode: str = "soft",
        use_ste: bool = False,
    ) -> torch.Tensor:
        """Return the broadcast-shape ``[B, 1, num_capabilities]`` tensor
        suitable for ``LayerCapabilityGater.lock(weights=...)``.

        ``mode='soft'``: returns softmax (training).
        ``mode='hard'``: returns one-hot of argmax (inference).
        ``use_ste=True``: pass-through gradient via straight-through estimator
            (only meaningful with ``mode='hard'``; default False).
        """
        if mode not in ("soft", "hard"):
            raise ValueError(f"mode must be 'soft' or 'hard', got {mode!r}")
        logits = self.forward(pooled_features)
        soft = torch.softmax(logits, dim=-1)
        if mode == "soft":
            return soft.unsqueeze(1)
        idx = soft.argmax(dim=-1)
        hard = F.one_hot(idx, num_classes=self.num_capabilities).to(soft.dtype)
        if use_ste:
            hard = hard + (soft - soft.detach())
        return hard.unsqueeze(1)


def pool_prompt_features(
    inner_model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Last-token pool of the last hidden state from a frozen causal LM.

    For a right-padded batch, reads the hidden state at the last non-pad
    position per row. Always runs under ``torch.no_grad()`` — features are
    extracted from a frozen base model.

    Returns ``[B, hidden_size]`` in the model's compute dtype.
    """
    if attention_mask is None:
        raise ValueError("pool_prompt_features requires an attention_mask.")
    with torch.no_grad():
        out = inner_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    hidden = out.hidden_states[-1]  # [B, T, H]
    last_pos = (attention_mask.sum(dim=1) - 1).clamp(min=0)  # [B]
    batch_idx = torch.arange(hidden.size(0), device=hidden.device)
    return hidden[batch_idx, last_pos.to(hidden.device), :]


def bind_global_routing(
    routing_weights: torch.Tensor,
    gaters: List["LayerCapabilityGater"],
    *,
    detach: bool,
) -> None:
    """Lock every gater to ``routing_weights``.

    ``routing_weights`` shape: ``[B, 1, num_capabilities]``. The same tensor
    object is referenced by every gater (no storage duplication).

    ``detach=True``: inference path (no grad through the locked weights).
    ``detach=False``: training path (grad flows through the routing weights
    into the producer — the global router head).
    """
    for g in gaters:
        g.lock(weights=routing_weights, detach=detach)


def base_only_one_hot(
    num_capabilities: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Routing weights that select slot 0 (the 'base'/no-LoRA capability).

    Used to disable the LoRA mixture during the feature-extraction prefill:
    with this lock active, every ``MultiLoRAGatedLinear`` outputs exactly
    ``base_linear(x)`` and the model behaves as the unmodified base.
    """
    w = torch.zeros(1, 1, num_capabilities, device=device, dtype=dtype)
    w[..., 0] = 1.0
    return w


def unlock_all(gaters: List["LayerCapabilityGater"]) -> None:
    for g in gaters:
        g.unlock()
