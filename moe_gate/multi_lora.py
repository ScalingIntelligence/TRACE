"""
MultiLoRAGatedLinear: a frozen base Linear weighted-mixed with N frozen
LoRA pairs by the block-level capability gate.

    y = W·x + Σ_i  w[i] · scaling_i · B_i(A_i x)

The gate output `w` (shape ``[B, T or 1, N]``) is computed once per block
by ``LayerCapabilityGater`` and shared across every MultiLoRAGatedLinear
in that block via ``GaterContext``. Slot 0 (the "base"/"no adapter" slot)
has no LoRA — it consumes a column in ``w`` but contributes 0 to the sum.
Only the gate Linear is trainable; base and (A_i, B_i) are frozen.
"""
from __future__ import annotations

import threading
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class GaterContext:
    """Per-block context filled by the block hook before each forward.

    Storage is per-thread so concurrent forwards on the same model don't
    race on a shared ``weights`` slot.
    """

    def __init__(self) -> None:
        self._tls = threading.local()

    @property
    def weights(self) -> Optional[torch.Tensor]:
        return getattr(self._tls, "weights", None)

    @weights.setter
    def weights(self, value: Optional[torch.Tensor]) -> None:
        self._tls.weights = value


class MultiLoRAGatedLinear(nn.Module):
    """Wraps a frozen base Linear with N frozen LoRA pairs and a shared gate.

    ``lora_pairs[i]`` is either ``(A, B, scaling)`` or ``None`` (a slot that
    consumes a column in ``capability_weights`` but contributes 0 to the
    LoRA delta — used for the "base" capability).
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        lora_pairs: List[Optional[Tuple[torch.Tensor, torch.Tensor, float]]],
        gater_ctx: GaterContext,
    ) -> None:
        super().__init__()
        self.base_linear = base_linear
        for p in self.base_linear.parameters():
            p.requires_grad = False

        self.num_capabilities = len(lora_pairs)
        self._has_lora: List[bool] = []
        self._scalings: List[float] = []
        for i, pair in enumerate(lora_pairs):
            if pair is None:
                self._has_lora.append(False)
                self._scalings.append(0.0)
                self.register_buffer(f"lora_A_{i}", torch.empty(0), persistent=False)
                self.register_buffer(f"lora_B_{i}", torch.empty(0), persistent=False)
                continue
            A_w, B_w, scaling = pair
            self._has_lora.append(True)
            self._scalings.append(float(scaling))
            self.register_buffer(f"lora_A_{i}", A_w.detach().contiguous(), persistent=False)
            self.register_buffer(f"lora_B_{i}", B_w.detach().contiguous(), persistent=False)

        self._gater_ctx = gater_ctx

    def _A(self, i: int) -> torch.Tensor:
        return getattr(self, f"lora_A_{i}")

    def _B(self, i: int) -> torch.Tensor:
        return getattr(self, f"lora_B_{i}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)

        ctx_weights = self._gater_ctx.weights
        if ctx_weights is None:
            return base_out

        ctx_weights = ctx_weights.to(dtype=base_out.dtype, device=base_out.device)

        if x.dim() == 2:
            # Qwen3 expert FFN path (post top-k gather). Only live when
            # individual expert projections resolve as Linears (unfused
            # experts, transformers<5). Trajectory + B=1 is exact (squeeze);
            # other shapes mean-pool — token mode here is approximate
            # because per-token expert assignments aren't plumbed through.
            if ctx_weights.size(0) != 1 or ctx_weights.size(1) != 1:
                w = ctx_weights.mean(dim=(0, 1))
            else:
                w = ctx_weights.view(-1)
            for i in range(self.num_capabilities):
                if not self._has_lora[i]:
                    continue
                lora_out = torch.nn.functional.linear(
                    torch.nn.functional.linear(x, self._A(i)), self._B(i)
                )
                base_out = base_out + lora_out * (w[i] * self._scalings[i])
            return base_out

        for i in range(self.num_capabilities):
            if not self._has_lora[i]:
                continue
            lora_out = torch.nn.functional.linear(
                torch.nn.functional.linear(x, self._A(i)), self._B(i)
            )
            w_i = ctx_weights[..., i : i + 1]
            base_out = base_out + lora_out * (w_i * self._scalings[i])
        return base_out
