# MoE Gate (Capability Routing)

This package implements the **inference-time routing** layer used by TRACE to
select the right capability-specific LoRA adapter for each task. It's the
"Select & Adapt" step from the TRACE paper (Section 3.4 / Step 4 of the
pipeline overview in the top-level README).

## What problem this solves

After running the capability selection + environment generation + GRPO
training steps, you end up with N LoRA adapters — one per discovered
capability:

```
adapters/
├── deep_call_graph_traversal.safetensors
├── semantic_logic_precision.safetensors
├── multi_site_consistency.safetensors
└── ...
```

At inference time, you don't know which capability the next task needs.
This package wraps the base model so it picks the right adapter (or a
soft mixture of adapters) automatically.

## Two routing scopes

```
routing_scope='per_layer'   # default — block-local soft routers
routing_scope='global'      # single classifier over the prompt,
                            # decision broadcast to every block
```

### `per_layer` (default)

Each transformer block gets a `LayerCapabilityGater` — a 2-layer MLP that
maps the block-local pooled hidden state to a softmax over the N capability
slots:

```python
g(x_l) = Softmax(Linear(x_l.detach()))   # zero-init → uniform at start
```

The router input is **detached** by design: the gate routes on existing
hidden states, never pushes gradient back into the base model.

Each block's routing decision is then consumed by `MultiLoRAGatedLinear`,
which replaces the LoRA-adapted `Linear` in attention / MLP projections
with a weighted mix:

```
y = base(x) + Σ_i g_i · LoRA_i(x)
```

The weights `g_i` come from the per-block gater.

### `global`

A single `TrajectoryGlobalRouter` reads the prompt's pooled features and
emits one decision. That decision is then broadcast to every block via
`bind_global_routing(...)`, so the per-block gaters become passive lock
holders (no Linear allocated).

Use `global` when you want a single interpretable "which capability did this
task use?" prediction. Use `per_layer` when you want layer-specific mixing
(empirically slightly better on heterogeneous tasks, but harder to
interpret).

## Public API

From `__init__.py`:

```python
from moe_gate import (
    build_capability_model,        # build base + frozen LoRAs + router
    LayerCapabilityGater,
    TrajectoryGlobalRouter,
    MultiLoRAGatedLinear,          # weighted-mix forward
    base_only_one_hot,             # decision that selects the base slot
    bind_global_routing,           # lock all gaters to a routing decision
    pool_prompt_features,          # last-token pool of last hidden state
    assert_frozen,                 # safety check: only routing params train
)
```

### Quick start

```python
from moe_gate import build_capability_model

model = build_capability_model(
    base_model_name="Qwen/Qwen3.6-27B",
    adapter_dir="adapters/",        # one .safetensors per capability
    routing_scope="per_layer",      # or "global"
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

# All adapter and base-model parameters are frozen at this point.
# The only trainable parameters are the gater Linears + the (optional)
# TrajectoryGlobalRouter head.
```

To train just the routing layer, see `train/train_router_sft.py` (SFT
warm-start on a labeled mix of capabilities) and the GRPO gater training
script.

## Files

| File | What |
|---|---|
| `__init__.py` | Public API exports. |
| `gater.py` | `LayerCapabilityGater`, `TrajectoryGlobalRouter`, locking helpers, `pool_prompt_features`. |
| `multi_lora.py` | `MultiLoRAGatedLinear` — replaces a LoRA-adapted `Linear` with a weighted-mix forward. |
| `model_builder.py` | `build_capability_model` — loads base + adapters, installs gaters. |
| `hf_wrapper.py` | HF integration: forward hooks, `transformers`-compatible wrapper class. |
| `freeze_utils.py` | `assert_frozen` and helpers to freeze non-routing params. |
| `smoke_test.py` | Loads a tiny model + 2 adapters + checks routing math; runnable as `python -m moe_gate.smoke_test`. |

## Sanity checks

After `build_capability_model(...)` returns, the only parameters with
`requires_grad=True` should be:

- Each block's `LayerCapabilityGater.linear.{weight,bias}` (per_layer mode), OR
- The single `TrajectoryGlobalRouter.head.{weight,bias}` (global mode).

`assert_frozen(model)` raises if any non-routing param has `requires_grad=True`.
Always call it right before starting training.

## Inference-time behavior

`MultiLoRAGatedLinear` reads the per-block routing decision from a
thread-local `GaterContext` set by the corresponding `LayerCapabilityGater`'s
forward hook. This means:

- One thread / one forward pass = one consistent routing decision per block.
- Decisions are **locked** after the first forward (by
  `lock_gaters_after_first_forward(...)`) so KV-cached subsequent tokens
  reuse the same adapter mix instead of recomputing routing every token.
- `unlock_all(model)` resets the locks between independent requests.

## Reference

This implementation matches the routing description in the TRACE paper.
The mixing weights `g_i` are softmax over capability slots; the LoRAs are
the per-capability adapters trained by GRPO in the
`environment_generation` step.
