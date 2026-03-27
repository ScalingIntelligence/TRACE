# Cached Core/TSV Merge in vLLM

**Date:** 2026-03-24
**Status:** Approved

## Problem

The current weighted LoRA path in vLLM performs linear addition of K adapters at forward time: `y += Σᵢ wᵢ Bᵢ Aᵢ x`. This is exact but loops K times per layer in Python. Core/TSV merging (from `merge_and_push_RL.py`) produces higher-quality merges by aligning LoRA subspaces via SVD before combining, but currently only runs offline.

## Solution

When `create_weighted_lora` is called, perform core/TSV merge offline, cache the result as a single PEFT adapter, load it into vLLM, and serve via the existing single-adapter deterministic path.

## Design Decisions

- **Synchronous:** Caller blocks until merge completes (~15-20s cold start)
- **Merge method:** Core space merging with TSV sub-method, isotropize=True (hardcoded for now)
- **Target rank:** Same as source adapters (no rank expansion)
- **Cache location:** `/tmp/vllm_core_tsv_cache/`
- **Cache eviction:** None (unbounded, manual cleanup)
- **Multi-process safety:** `fcntl.flock` per cache entry
- **Forward path:** Unchanged — merged adapter loaded as single LoRA, config set to `[(merged_id, 1.0)]`

## Architecture

```
create_weighted_lora(adapters=[A,B,C], weights=[0.6,0.3,0.1])
    │
    ▼
compute cache key = hash(sorted((path_A,0.6),(path_B,0.3),(path_C,0.1)))
    │
    ▼
cache hit? ──yes──► load merged adapter from disk
    │
    no
    │
    ▼
acquire flock on /tmp/vllm_core_tsv_cache/<key>/.lock
    │
    ▼
double-check (another process may have merged) ──hit──► release lock, load
    │
    no
    │
    ▼
run core_tsv_merge() in ThreadPoolExecutor:
  1. load_lora_ab_pairs() for each adapter
  2. combine_core_space(core_merge="tsv", isotropize=True)
  3. re-factorize ΔW → (A_merged, B_merged) via truncated SVD at rank r
  4. save as PEFT adapter to cache dir
    │
    ▼
release lock
    │
    ▼
load merged adapter via load_lora_adapter (name="core_tsv_<key>")
    │
    ▼
set weighted config to [(merged_lora_id, 1.0)]
```

## File Changes

| File | Change |
|------|--------|
| `vllm/lora/core_tsv_merge.py` | **New.** Shared module: `load_lora_ab_pairs()`, core/TSV merge, SVD re-factorization, save as PEFT |
| `vllm/lora/core_tsv_cache.py` | **New.** Cache key, flock locking, cache-or-merge orchestration |
| `vllm/entrypoints/openai/models/serving.py` | **Modify** `create_weighted_lora()` to use cache, load merged adapter, set single-adapter config |
| `merge_and_push_RL.py` | **Refactor** to import from `core_tsv_merge.py` |

No changes to forward path (punica_gpu.py, base_linear.py, fused_moe.py).

## Cache Key

```python
key_data = sorted((os.path.realpath(path), round(weight, 4)) for path, weight in zip(paths, weights))
cache_key = hashlib.sha256(json.dumps(key_data).encode()).hexdigest()[:16]
```

## Multi-Process Locking

```python
lock_path = os.path.join(cache_dir, cache_key, ".lock")
os.makedirs(os.path.dirname(lock_path), exist_ok=True)
with open(lock_path, "w") as lock_fd:
    fcntl.flock(lock_fd, fcntl.LOCK_EX)
    # double-check, merge if needed
    fcntl.flock(lock_fd, fcntl.LOCK_UN)
```

## Re-Factorization

After core/TSV merge produces full-rank ΔW per layer:

```python
U, S, Vh = torch.linalg.svd(delta_W, full_matrices=False)
A_merged = (torch.diag(S[:r].sqrt()) @ Vh[:r, :])   # [r, in_dim]
B_merged = (U[:, :r] @ torch.diag(S[:r].sqrt()))     # [out_dim, r]
# Verify: B_merged @ A_merged ≈ ΔW (truncated)
```

Scaling: `lora_alpha = r` so that PEFT scaling = `alpha/r = 1.0` (pre-applied).

## Testing

Compare output of cached merged adapter against `merge_and_push_RL.py` offline merge for identical adapter combinations and weights. Verify:
1. Cache hit returns identical adapter on second call
2. Parallel processes don't corrupt the cache
3. Merged adapter produces correct logits vs offline merge
