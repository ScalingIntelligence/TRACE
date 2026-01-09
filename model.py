#!/usr/bin/env python3
"""
Model architecture for PPO training with value head.
"""
import torch
import torch.nn as nn

from config import autocast_ctx


# =========================
# Model helpers
# =========================
def _unwrap_backbone(causal_lm: nn.Module) -> nn.Module:
    """Unwrap the transformer backbone from the causal LM wrapper."""
    m = causal_lm
    for _ in range(3):
        if hasattr(m, "model") and isinstance(getattr(m, "model"), nn.Module):
            m = m.model
        else:
            break
    return m


def _get_lm_head(causal_lm: nn.Module) -> nn.Module:
    """Get the language model head from the causal LM."""
    if hasattr(causal_lm, "lm_head") and isinstance(causal_lm.lm_head, nn.Module):
        return causal_lm.lm_head
    head = causal_lm.get_output_embeddings()
    if head is None:
        raise ValueError("Could not find lm_head / output embeddings.")
    return head


# =========================
# PPO actor-critic wrapper (value head)
# =========================
class PolicyWithValueHead(nn.Module):
    """
    Memory-friendly:
    - backbone forward returns last hidden
    - logits = lm_head(last_hidden)
    - value head runs in fp32 on pooled hidden state
    """
    def __init__(self, causal_lm: nn.Module, device: str = "cuda"):
        super().__init__()
        self.lm = causal_lm
        self.backbone = _unwrap_backbone(causal_lm)
        self.lm_head = _get_lm_head(causal_lm)
        self.device = device

        cfg = getattr(self.lm, "config", None)
        hidden_size = None
        for attr in ("hidden_size", "n_embd", "d_model"):
            if cfg is not None and hasattr(cfg, attr):
                hidden_size = int(getattr(cfg, attr))
                break
        if hidden_size is None:
            raise ValueError("Could not infer hidden size from model.config")

        self.value_head = nn.Linear(hidden_size, 1).to(dtype=torch.float32)

    def forward(self, input_ids, attention_mask=None):
        with (autocast_ctx(self.device) if self.device == "cuda" else torch.no_grad()):
            out = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )

        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            last_hidden = out.last_hidden_state
        elif isinstance(out, (tuple, list)) and len(out) > 0:
            last_hidden = out[0]
        else:
            raise RuntimeError(f"Backbone output has no last hidden state. Type={type(out)}")

        logits = self.lm_head(last_hidden)
        return logits, last_hidden

