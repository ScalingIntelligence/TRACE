# Game Evaluation Randomness Analysis

This document explains the sources of non-determinism in the game evaluation system, including τ²-bench integration, game self-play, and model-vs-model experiments. This analysis identifies why evaluation results can vary between identical runs, even with `temperature=0.0` and fixed seeds.

---

## Summary

Even with `temperature=0.0` and `seed=42/300`, evaluation results are **NOT fully reproducible**. The primary causes are:
1. vLLM/LLM provider inference non-determinism
2. Concurrent execution ordering
3. Incomplete seed propagation (PyTorch/NumPy not seeded)
4. Random action fallback on extraction failures

---

## Sources of Randomness

### 1. vLLM Inference Non-Determinism (PRIMARY CAUSE)

**Impact: HIGH | Controllable: PARTIALLY**

Even with `temperature=0.0`, vLLM inference is **NOT guaranteed deterministic**:

- Floating-point rounding differences across GPU operations
- CUDA kernel non-determinism (especially with TensorFloat-32)
- Batching and parallelism in inference servers
- Different server instances may produce slightly different outputs

**Evidence in code** (`inference.py`, lines 170-177):

```python
payload = {
    "model": model_name,
    "prompt": prompt,
    "max_tokens": max_tokens,
    "temperature": temperature,
    "top_p": 1.0,  # Nucleus sampling disabled
    ...
}
```

**TF32 Configuration** (`config.py`, lines 115-116):

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

TensorFloat-32 operations are faster but sacrifice bit-exact reproducibility.

---

### 2. Concurrent Execution

**Impact: MEDIUM-HIGH | Controllable: YES**

#### τ²-bench Concurrency

In `evals/benchmarks/tau2_bench_eval/main.py` (line 381):

```bash
tau2 run --max-concurrency 10  # Default from config
```

When `max_concurrency > 1`:
- Simulations run in parallel with non-deterministic ordering
- Thread scheduling is OS-dependent
- Network latency variations affect completion order

#### Subprocess Parallelization

In `evaluation.py` (lines 609-657):

```python
for shard_idx, shard_ids in enumerate(shards):
    seed_for_shard = int(seed) + int(shard_idx)
    # Each shard runs as separate subprocess
```

While seeds are offset per-shard, subprocess scheduling is system-dependent.

**How to fix:**
```bash
# For τ²-bench
python main.py --max-concurrency 1

# For game evaluation
python evaluation.py --num-games 1  # Sequential games
```

---

### 3. Incomplete Seed Propagation

**Impact: MEDIUM | Controllable: YES**

#### What IS Seeded

In `evaluation.py` (line 199) and `eval_checkpoint.py` (line 95):

```python
rng = random.Random(int(seed))
```

Python's `random` module is seeded for:
- Per-game environment seed generation
- Random action fallback selection

#### What is NOT Seeded

| Component | Seeded? | Notes |
|-----------|---------|-------|
| Python `random` | ✅ Yes | Via `random.Random(seed)` |
| NumPy `np.random` | ❌ No | Used in scoring/metrics |
| PyTorch `torch` | ❌ No | Affects model inference |
| CUDA operations | ❌ No | Non-deterministic by default |
| vLLM server | ❌ No | External process, no seed control |

**Evidence - No global seeding** (`config.py`):

```python
# No torch.manual_seed() or np.random.seed() calls
# Only TF32 settings configured
```

**How to fix:**
```python
import random
import numpy as np
import torch

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

---

### 4. Random Action Fallback on Extraction Failures

**Impact: MEDIUM | Controllable: PARTIALLY**

When the model fails to produce a valid action, a random fallback is used.

In `evaluation.py` (lines 296-298):

```python
if act is None:
    extraction_failures += 1
    act = rng.choice(legal)  # Random fallback!
```

In `eval_checkpoint.py` (line 158):

```python
if act is None:
    extraction_failures += 1
    act = rng.choice(legal)
```

While the RNG is seeded, extraction failures themselves may vary due to LLM non-determinism, causing different fallback paths.

**How to fix:**
```python
# Option 1: Deterministic fallback (always pick first legal action)
if act is None:
    extraction_failures += 1
    act = sorted(legal)[0]  # Deterministic: alphabetically first

# Option 2: Retry with explicit prompting before fallback
```

---

### 5. τ²-bench User Simulator Randomness

**Impact: MEDIUM | Controllable: PARTIALLY**

The user simulator in τ²-bench makes its own LLM calls with potential non-determinism.

In `tau2-bench/src/tau2/utils/llm_utils.py`:

```python
response = completion(
    model=model,
    messages=litellm_messages,
    tools=tools,
    tool_choice=tool_choice,
    **kwargs,  # seed passed but not guaranteed
)
```

Even with `user_llm_args.temperature: 0.0`, the user simulator's responses may vary.

---

### 6. LiteLLM Provider Abstraction

**Impact: MEDIUM | Controllable: PARTIALLY**

LiteLLM routes requests to various providers, each with different reproducibility guarantees:

| Provider | Reproducibility |
|----------|----------------|
| vLLM (local) | Best (but not perfect) |
| Claude API | Not guaranteed with temp=0 |
| Qwen/DeepSeek API | Provider-dependent |
| Local HuggingFace | Best (with proper seeding) |

**Evidence** (`tau2-bench/src/tau2/config.py`):

```python
# Default temperatures set to 0.0
agent_llm_args:
  temperature: 0.0
user_llm_args:
  temperature: 0.0
```

---

### 7. Game Environment Initial State

**Impact: LOW | Controllable: YES**

Game environments are seeded per-game:

In `evaluation.py` (lines 202-203):

```python
game_seed = rng.randint(0, 2**31 - 1)
env.reset(game_seed)
```

This is properly seeded from the master RNG, ensuring consistent initial states if the master seed matches.

---

### 8. Multi-vLLM Load Balancing

**Impact: LOW | Controllable: YES**

When using multiple vLLM servers:

In `inference.py` (lines 453-456):

```python
class MultiVLLMBackend:
    def __init__(self, backends: list[VLLMServerBackend]):
        self.backends = backends
        self._counter = 0
```

Round-robin selection is deterministic, but different servers may produce slightly different outputs.

---

### 9. Timestamps and UUIDs

**Impact: LOW | Controllable: NOT EASILY**

τ²-bench uses timestamps and UUIDs in simulation metadata:

In `tau2-bench/src/tau2/orchestrator/orchestrator.py`:

```python
id=str(uuid.uuid4()),  # Random UUID
```

While not affecting LLM outputs directly, these can affect:
- Log ordering
- Cache key computation
- Downstream result processing

---

## Variance Cascade Effect

The variance compounds across multi-turn interactions:

```
Turn 1: vLLM outputs slightly different token
         ↓
Turn 2: Different prompt → different response
         ↓
Turn 3: Agent takes different action
         ↓
Turns 4-N: Conversation diverges completely
         ↓
Result: Game that was won now loses (or vice versa)
```

For a 10-turn game with 1% per-turn variance:
- Probability of identical trajectory: ~0.99^10 ≈ 90%
- ~10% of games may diverge

---

## Impact Ranking

| Source | Impact Level | Controllable? | Effort to Fix |
|--------|--------------|---------------|---------------|
| vLLM inference non-determinism | **HIGH** | Partially | Medium |
| Concurrent execution | **MEDIUM-HIGH** | Yes | Easy |
| Incomplete seeding (torch/numpy) | **MEDIUM** | Yes | Easy |
| Random action fallback | **MEDIUM** | Yes | Easy |
| User simulator LLM calls | **MEDIUM** | Partially | Medium |
| Multi-server load balancing | **LOW** | Yes | Easy |
| Timestamps/UUIDs | **LOW** | Difficult | Hard |

---

## Recommendations

### For Maximum Reproducibility

#### 1. Run Configuration

```bash
# τ²-bench evaluation
python main.py \
    --config config.yml \
    --max-concurrency 1 \
    --seed 42 \
    --num-trials 5

# Game evaluation
python evaluation.py \
    --seed 42 \
    --num-games 50 \
    --sequential  # If available
```

#### 2. Add Comprehensive Seeding

Add this at the start of evaluation scripts:

```python
def set_all_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # Enable deterministic operations
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Disable TF32 for bit-exact reproducibility
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        # Use deterministic algorithms (may be slower)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass

    # Set environment variable for CUDA determinism
    import os
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
```

Call this in:
- `evaluation.py` at script start
- `eval_checkpoint.py` at script start
- `config.py` initialization

#### 3. Deterministic Action Fallback

Replace random fallback with deterministic selection:

```python
# In evaluation.py and eval_checkpoint.py
if act is None:
    extraction_failures += 1
    # Deterministic: pick lexicographically first legal action
    act = sorted(legal)[0]
    # Or: pick action with lowest index if actions are numbered
    # act = min(legal, key=lambda x: legal.index(x))
```

#### 4. Disable TF32 for Critical Evaluations

In `config.py`, add a flag:

```python
def configure_deterministic_mode(deterministic: bool = False):
    if deterministic:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Default fast mode
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
```

#### 5. Single vLLM Server

Use a single vLLM server instance to avoid cross-server variance:

```bash
# Start single server
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --port 8080 \
    --seed 42  # vLLM server-side seed
```

#### 6. Run Multiple Trials and Report Statistics

```python
# Instead of single run:
results = [run_evaluation(seed=base_seed + i) for i in range(5)]

# Report:
mean_score = np.mean(results)
std_score = np.std(results)
print(f"Score: {mean_score:.2f} ± {std_score:.2f}")
```

#### 7. vLLM Deterministic Configuration

When starting vLLM server, add determinism flags:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --seed 42 \
    --dtype float16  # Avoid bfloat16 for better reproducibility
    # Note: --enforce-eager may help but reduces performance
```

---

## Acceptance of Variance

Given fundamental LLM inference non-determinism, some variance is **unavoidable**. Recommended approach:

| Use Case | Acceptable Variance | Recommendation |
|----------|---------------------|----------------|
| Development/debugging | ~5-10% | Use `max_concurrency=1` |
| Benchmarking | ~2-5% | Run 3-5 trials, report mean ± std |
| Paper/publication | ~1-2% | Run 5-10 trials, report CI |
| A/B comparisons | Any | Use paired tests on same seeds |

---

## Quick Checklist

- [ ] Set `temperature: 0.0` for all LLM calls
- [ ] Set `max_concurrency: 1` for reproducibility testing
- [ ] Add comprehensive seeding (Python, NumPy, PyTorch)
- [ ] Disable TF32 for critical evaluations
- [ ] Use deterministic action fallback
- [ ] Run multiple trials and report statistics
- [ ] Use single vLLM server instance
- [ ] Document seed values used in results

---

## Files to Modify

| File | Change |
|------|--------|
| `config.py` | Add `set_all_seeds()` and deterministic mode |
| `evaluation.py` | Call `set_all_seeds()` at start, fix action fallback |
| `eval_checkpoint.py` | Call `set_all_seeds()` at start, fix action fallback |
| `inference.py` | Pass seed to vLLM if supported |
| YAML configs | Set `max_concurrency: 1` for reproducibility |

---

*Document created: 2026-02-01*
