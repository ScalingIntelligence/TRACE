"""
Capability routing: gating across frozen LoRA capability adapters.

Two scopes:
    routing_scope='per_layer' (default): 48 independent block-local soft routers.
    routing_scope='global':              one TrajectoryGlobalRouter over the
                                         prompt; broadcast to every block.

Public API:
    build_capability_model:    load base + frozen LoRAs + (per-block gaters
                               or global router, depending on scope).
    LayerCapabilityGater:      per-block lock-state holder (active in
                               per_layer mode; passive in global mode).
    TrajectoryGlobalRouter:    single classifier head used in global mode.
    pool_prompt_features:      last-token pool of last hidden state.
    bind_global_routing:       lock all gaters to a routing decision.
    base_only_one_hot:         routing weights that select the base slot.
    MultiLoRAGatedLinear:      replaces a LoRA-adapted Linear with a weighted-mix forward.
    assert_frozen:             raise if any non-routing param has requires_grad=True.
"""
from .gater import (
    LayerCapabilityGater,
    TrajectoryGlobalRouter,
    base_only_one_hot,
    bind_global_routing,
    lock_gaters_after_first_forward,
    pool_prompt_features,
    unlock_all,
)
from .multi_lora import MultiLoRAGatedLinear
from .freeze_utils import assert_frozen, gater_parameters
from .model_builder import build_capability_model, CapabilityModelConfig
from .hf_wrapper import (
    CapabilityRoutedConfig,
    CapabilityRoutedForCausalLM,
    push_capability_routed_to_hub,
    save_capability_routed_checkpoint,
)

__all__ = [
    "LayerCapabilityGater",
    "lock_gaters_after_first_forward",
    "TrajectoryGlobalRouter",
    "pool_prompt_features",
    "bind_global_routing",
    "base_only_one_hot",
    "unlock_all",
    "MultiLoRAGatedLinear",
    "assert_frozen",
    "gater_parameters",
    "build_capability_model",
    "CapabilityModelConfig",
    "CapabilityRoutedConfig",
    "CapabilityRoutedForCausalLM",
    "save_capability_routed_checkpoint",
    "push_capability_routed_to_hub",
]
