# On-Policy Distillation: Complete Code Guide

## Table of Contents

1. [Concept: What Is On-Policy Distillation?](#1-concept)
2. [Architecture: What Runs Where](#2-architecture)
3. [Code Walkthrough: Initialization](#3-initialization)
4. [Code Walkthrough: Main Loop Step by Step](#4-main-loop)
5. [The Two Loss Functions: Math + Code](#5-loss-functions)
6. [Teacher Modes: How Teacher Logprobs Are Obtained](#6-teacher-modes)
7. [Distributed Training: How 4 GPUs Coordinate](#7-distributed)
8. [Per-Skill Teacher Routing: Multi-Server Setup](#8-per-skill-routing)
9. [Key Helper Functions Reference](#9-helpers)
10. [Concrete Numerical Example](#10-example)
11. [Debugging Guide: What the Log Lines Mean](#11-debugging)
12. [Common Failure Modes](#12-failures)

---

## 1. Concept: What Is On-Policy Distillation? <a name="1-concept"></a>

Standard SFT distillation: teacher generates data offline, student trains on teacher's outputs.
Problem: distribution mismatch — the student is trained on trajectories it would never produce.

On-policy distillation flips this:
1. The **student** generates trajectories (plays games, makes tool calls)
2. The **teacher** scores each token the student produced: "I would have assigned probability X to that token"
3. The student's LoRA is updated to move its per-token probabilities closer to the teacher's

This means:
- The student is always trained on **its own** distribution (no mismatch)
- The teacher corrects the student's **actual** mistakes, not hypothetical ones
- As the student improves, it explores new parts of the state space, and the teacher guides it there too

### Key difference from GRPO

In GRPO, the training signal comes from **rewards** (did the agent solve the task? 0 or 1).
In distillation, the training signal comes from the **teacher's per-token logprobs** (dense, per-token signal).

GRPO needs `group_size > 1` to estimate advantages (which rollout was better than average).
Distillation uses `group_size = 1` because the teacher provides dense per-token signal — no need to compare rollouts.

---

## 2. Architecture: What Runs Where <a name="2-architecture"></a>

```
                    ┌─────────────────────────────────────────┐
                    │          vLLM Server (port 8080)         │
                    │  Qwen/Qwen3-30B-A3B-Instruct-2507       │
                    │  + student LoRA (hot-reloaded each iter) │
                    │  Role: generate rollouts (text)          │
                    │  GPU: e.g., GPU 0-2                      │
                    └──────────────────┬──────────────────────┘
                                       │ HTTP API
                    ┌──────────────────┴──────────────────────┐
                    │        train_distill.py (rank 0)         │
                    │  - Calls vLLM to generate game rollouts  │
                    │  - Broadcasts samples to all ranks       │
                    │  - Queries teacher vLLM for logprobs     │
                    │  - Runs forward/backward on GPU 5        │
                    └──────────────────┬──────────────────────┘
                                       │ NCCL all-reduce
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
    ┌─────────┴──────────┐  ┌─────────┴──────────┐  ┌─────────┴──────────┐
    │  rank 1 (GPU 6)    │  │  rank 2 (GPU 7)    │  │   ...              │
    │  Forward/backward  │  │  Forward/backward  │  │                    │
    │  on its shard      │  │  on its shard      │  │                    │
    └────────────────────┘  └────────────────────┘  └────────────────────┘

                    ┌─────────────────────────────────────────┐
                    │     Teacher vLLM Server(s)               │
                    │  port 9000: structured_data teacher      │
                    │  port 9001: tau_tool_calling teacher     │
                    │  port 9002: multistep_task teacher       │
                    │  Role: answer logprob queries only       │
                    │  (never generates text)                  │
                    └─────────────────────────────────────────┘
```

**Key point**: The student model exists in TWO places:
1. **On the vLLM server** (port 8080): for generating text during games. Updated each iteration via LoRA hot-reload.
2. **On the training GPUs** (one copy per rank): for computing gradients. Updated each iteration via optimizer step.

The teacher model(s) exist ONLY on separate vLLM servers. They are never loaded onto training GPUs (in the vLLM teacher mode).

---

## 3. Code Walkthrough: Initialization <a name="3-initialization"></a>

File: `train_distill.py`, lines 391-660

### 3.1 Distributed setup (lines 394-397)

```python
rank, world_size, local_rank = dist_pre_init()
```

`torchrun --nproc_per_node=3` launches 3 processes. Each gets:
- `rank`: global rank (0, 1, 2)
- `world_size`: total processes (3)
- `local_rank`: GPU index within this node (0, 1, 2 → maps to CUDA_VISIBLE_DEVICES 5, 6, 7)

Only rank 0 prints to stdout. Others have `suppress_print()`.

### 3.2 Parse teacher mappings (lines 401-413)

```python
teacher_urls = parse_skill_mapping(args.teacher_url or "")
teacher_models = parse_skill_mapping(args.teacher_model or "")
```

`parse_skill_mapping` handles two formats:
- Single value: `"http://localhost:9000"` → `{"__default__": "http://localhost:9000"}`
- Per-skill: `"sdr=http://localhost:9000,tc=http://localhost:9001"` → `{"sdr": "http://...:9000", "tc": "http://...:9001"}`

### 3.3 Game mix setup (lines 422-440)

```python
game_mix = parse_game_mix(args.games, args)
# e.g., "structured_data_reasoning:0.33,multistep_task:0.33,tau_tool_calling:0.34"
```

This creates a `GameMix` with 3 entries, each holding:
- `game_spec`: The game's environment factory, max_gen_tokens, etc.
- `weight`: Normalized proportion (0.33, 0.33, 0.34)
- `env_kwargs`: Per-game kwargs (e.g., `user_client` for tau_tool_calling)

### 3.4 Load model + LoRA (lines 520-551)

```python
model, tokenizer = FastLanguageModel.from_pretrained(model_name, ...)
model = FastLanguageModel.get_peft_model(model, r=16, ...)
```

1. Load the base Qwen3-30B-A3B model onto this rank's GPU
2. Wrap with a LoRA adapter (rank=16, targets all attention + MLP projections)
3. The LoRA starts as zero weights → student = base model initially

### 3.5 Initialize inference backend (lines 606-614)

```python
if is_main_rank():
    inference_backend = init_inference_backend(model, tokenizer, device)
```

Rank 0 only. Detects `VLLM_BASE_URLS` env var and creates a `VLLMBackend` that:
- Connects to vLLM at `http://localhost:8080`
- Can hot-reload LoRA weights via `sync_policy()`
- Generates text via the `/v1/chat/completions` API

### 3.6 Optimizer (lines 642-644)

```python
trainable_params = [p for p in model.parameters() if p.requires_grad]
optim = torch.optim.AdamW(trainable_params, lr=1e-5)
```

Only LoRA parameters are trainable. Base model is frozen.

---

## 4. Code Walkthrough: Main Loop Step by Step <a name="4-main-loop"></a>

File: `train_distill.py`, lines 664-1130

Each iteration has 8 steps:

### Step 1: Sync student LoRA to vLLM (line 668)

```python
inference_backend.sync_policy(model, vllm_adapter_dir)
```

Rank 0 only:
1. Saves current LoRA weights to `vllm_adapter_latest_distill/`
2. Sends HTTP request to vLLM server to hot-reload the adapter
3. After this, vLLM generates text using **base model + updated student LoRA**

All ranks wait at `barrier()`.

### Step 2: Collect rollouts (line 674)

Rank 0 only. Calls `collect_grpo_rollouts()` with `group_size=1`.

**What happens inside** (simplified):
```
For each of 64 groups (groups_per_batch=64):
    Pick a game based on weights (33% sdr, 33% mt, 34% tc)
    Create a game environment with a random seed
    Loop until game ends:
        env produces system prompt + user message + tools
        vLLM generates model response (temperature=1.0)
        env processes response (executes tool calls, etc.)
    Record reward (0 or 1)
    Create GRPOSample for each turn the model played
```

With `group_size=1`, each "group" is just one game. No repeated rollouts.

A single game may produce **multiple samples** (one per turn). Example:
- Turn 1: model calls `search_flights(...)` → GRPOSample with that turn's prompt + completion
- Turn 2: model calls `book_flight(...)` → another GRPOSample
- Turn 3: model says "Done, booked flight X" → another GRPOSample

All turns from the same game get the same `reward` (terminal reward).

Result: ~64 games → ~200-500 samples (depending on how many turns each game took).

Then `broadcast_objects(broadcast_data)` sends all samples to ranks 1, 2, ... via pickle.

### Step 3: Filter info-gathering turns (line 721)

```python
samples, dummy_adv, dt_filtered = filter_info_gathering_turns(samples, dummy_adv)
```

Removes turns where the model just called a lookup/search tool. These are "info-gathering" — the model isn't making a real decision, just fetching data. Training on these turns adds noise.

Typically filters ~300 out of ~500 samples → ~200 training samples.

### Step 4: Tokenize samples (line 736)

For each sample, converts chat messages + completion into token IDs:

```python
ids, pL, aL = build_prompt_plus_action(tokenizer, msgs, s.completion_text, tools=s.tools)
```

This calls `tokenizer.apply_chat_template(prompt_msgs, tools=tools)` for the prompt,
then `tokenizer(completion_text)` for the action, and concatenates:

```
Token sequence: [prompt tokens (pL tokens)] [action tokens (aL tokens)]
                 ^^^^^^^^^^^^^^^^^^^^^^^^    ^^^^^^^^^^^^^^^^^^^^^^^^^
                 System + user + tool msgs   Model's response text
                 (no gradients here)         (gradients computed here)
```

Samples exceeding `MAX_SEQ_LENGTH` are dropped.

Then `make_sorted_batches(seq_lens, mini_batch_size=2)` groups samples by length into mini-batches of 2, sorted to minimize padding waste.

### Step 5: Stats phase — compute teacher logprobs (line 786)

**Goal**: For every action token in every sample, get `log P_teacher(token | context)`.

This is the teacher scoring step — it tells us "how likely would the teacher have generated each token the student produced?"

```python
model.eval()
max_al = max(action_lens)
teacher_logp_padded = torch.zeros(N, max_al, dtype=torch.float32)
```

The result is a `[N, max_action_len]` tensor on CPU.

**For vLLM teacher mode** (lines 837-851):

Each rank queries its shard of samples. Rank 0 queries samples [0, 3, 6, ...], rank 1 queries [1, 4, 7, ...], etc.

For each sample, `query_teacher_logprobs_vllm` does:

1. Look up the game_name (e.g., `"structured_data_reasoning"`)
2. Resolve URL: `teacher_urls["structured_data_reasoning"]` → `http://localhost:9000`
3. Resolve model: `teacher_models["structured_data_reasoning"]` → `tarsur909/...-structured-10`
4. HTTP POST to `http://localhost:9000/v1/completions`:
   ```json
   {
     "model": "tarsur909/...-structured-10",
     "prompt": [token_id_0, token_id_1, ..., token_id_N],  // ALL tokens
     "max_tokens": 1,       // don't generate, just score
     "echo": true,          // return logprobs for input tokens too
     "logprobs": 1,
     "temperature": 1.0
   }
   ```
5. vLLM runs a forward pass and returns `token_logprobs` for every position
6. We extract the action region: `token_logprobs[prompt_len : prompt_len + action_len]`

This is done with 16 concurrent threads to saturate the teacher servers.

After all ranks finish their shards, `all_reduce(SUM)` assembles the full tensor — each rank only wrote to disjoint rows, others are zeros, so SUM gives the complete result on every rank.

**For ppo_surrogate loss only** (lines 806-834):

An additional pass computes **old student logprobs** (the student's log-probabilities BEFORE the gradient update). This uses the local model with the student LoRA adapter, running batched forward passes on the training GPUs.

### Step 6: Training phase — gradient updates (line 893)

```python
model.train()
```

For each epoch (default: 1):
1. Shuffle mini-batch order
2. Shard mini-batches across ranks
3. `optim.zero_grad()`

For each mini-batch (2 samples):

**a. Forward pass** (lines 926-932):
```python
outputs = model(input_ids=mb_ids, attention_mask=mb_attn)
logits = outputs.logits  # [2, T, vocab_size]
new_logp_list = per_token_action_logprobs(logits, mb_ids, mb_pl, mb_al)
# Returns: [tensor of shape [action_len_0], tensor of shape [action_len_1]]
```

**b. Compute loss** (lines 940-975):

For each sample in the mini-batch:
```python
new_lp = new_logp_list[j][:al]                    # student logprobs (HAS GRAD)
teacher_lp = teacher_logp_padded[j, :al].to(device) # teacher logprobs (DETACHED)
```

**reverse_kl** (logprob MSE):
```python
sample_loss = ((new_lp - teacher_lp) ** 2).mean()
```

**ppo_surrogate** (PPO with teacher gap as advantage):
```python
old_lp = student_logp_padded[j, :al].to(device)   # old student (DETACHED)
ratio = exp(new_lp - old_lp)                       # importance sampling ratio
advantage = (teacher_lp - old_lp).detach()          # per-token advantage
surr1 = ratio * advantage
surr2 = clamp(ratio, 1-eps, 1+eps) * advantage
sample_loss = -min(surr1, surr2).mean()
```

**c. Scale and backward** (lines 977-979):
```python
loss = total_loss / n_samples_in_mb / n_total_batches
loss.backward()
```

Division by `n_total_batches` means each rank accumulates a fraction of the total gradient. After all-reduce, the sum equals the gradient over all samples.

**d. All-reduce and step** (lines 1010-1015):
```python
allreduce_coalesced_grads(trainable_params)  # SUM gradients across ranks
clip_grad_norm_(trainable_params, 1.0)
optim.step()
```

After this, all ranks have identical updated LoRA weights.

### Step 7: Logging (line 1046)

Rank 0 logs to wandb. Key metrics:
- `distill/policy_loss`: Average loss value
- `distill/teacher_student_gap`: mean(teacher_lp - student_lp). Positive = teacher more confident. Should decrease as student matches teacher.
- `distill/avg_reward`: Game reward (NOT used for training, diagnostic only)
- `time/collect_sec`, `time/stats_sec`, `time/grad_sec`: Timing breakdown

### Step 8: Checkpoint (line 1112)

Every `save_every` iterations, rank 0 saves the student LoRA to disk.

---

## 5. The Two Loss Functions: Math + Code <a name="5-loss-functions"></a>

### 5.1 reverse_kl (Logprob MSE) — 2 forward passes

**Math:**
```
L = (1/N) * Σ_i (1/A_i) * Σ_t ( log π_student(y_{i,t} | ...) - log π_teacher(y_{i,t} | ...) )^2
```

**Code** (line 951):
```python
sample_loss = ((new_lp - teacher_lp) ** 2).mean()
```

**Gradient:**
```
∂L/∂θ = 2 * (log π_student - log π_teacher) * ∂ log π_student / ∂θ
```

- When student assigns MORE probability than teacher (`new_lp > teacher_lp`):
  gradient is positive → optimizer pushes `new_lp` DOWN
- When student assigns LESS probability than teacher (`new_lp < teacher_lp`):
  gradient is negative → optimizer pushes `new_lp` UP
- When they match: gradient = 0

**Forward passes needed:**
1. Teacher logprobs (via vLLM API)
2. Student logprobs (local forward pass with gradients)

### 5.2 ppo_surrogate (PPO with KL gap as reward) — 3 forward passes

**Math (per token):**
```
ratio_t = π_student_new(y_t) / π_student_old(y_t) = exp(new_lp_t - old_lp_t)
advantage_t = log π_teacher(y_t) - log π_student_old(y_t)

L = -min( ratio_t * advantage_t,  clip(ratio_t, 1-ε, 1+ε) * advantage_t )
```

**Code** (lines 958-967):
```python
old_lp = student_logp_padded[j, :al].to(device)
ratio = torch.exp(new_lp - old_lp)
advantage = (teacher_lp - old_lp).detach()

surr1 = ratio * advantage
surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * advantage
sample_loss = -torch.min(surr1, surr2).mean()
```

**Intuition:**
- `advantage_t > 0` means teacher likes this token more than old student → loss pushes student to increase probability
- `advantage_t < 0` means teacher likes this token less → loss pushes student to decrease probability
- The ratio + clipping prevents the new policy from diverging too far from the old policy in one step

**Forward passes needed:**
1. Old student logprobs (local forward pass, no gradients)
2. Teacher logprobs (via vLLM API)
3. New student logprobs (local forward pass with gradients)

This is what SkyRL and GLM-5 paper describe as "on-policy distillation."

### 5.3 Which to use?

| Property | reverse_kl (MSE) | ppo_surrogate |
|---|---|---|
| Forward passes | 2 | 3 |
| Stability | Good | Better (clipping) |
| Theory | Approximate | Principled (PPO) |
| Speed per iter | ~30% faster | Slower |

---

## 6. Teacher Modes: How Teacher Logprobs Are Obtained <a name="6-teacher-modes"></a>

### 6.1 vLLM teacher(s) — `--teacher-url` + `--teacher-model`

Teacher runs on separate vLLM server(s). Training GPUs never load the teacher.

```python
if has_teacher_url:
    my_sample_indices = list(range(rank, N, world_size))
    teacher_logp_padded = query_teacher_logprobs_vllm(...)
```

The vLLM `/v1/completions` endpoint with `echo=True` returns logprobs for input tokens.
We send ALL token IDs (prompt + action) and extract the action region.

**How `echo=True` works:**
```
Input tokens:  [sys][user says "book flight"][assistant calls search_flights][result][assistant says "I booked flight X"]
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                prompt region (P tokens)                                     action region (A tokens)

vLLM returns: token_logprobs = [None, -2.3, -1.1, ..., -0.5, -3.2, ...]
                                 ^                       ^^^^^^^^^^^^^^^^
                                 First token has no predecessor    We extract these A values
```

### 6.2 Local LoRA adapter — `--teacher-adapter /path`

Teacher is a LoRA adapter loaded into the same model. Shares base weights with student.

```python
model.load_adapter(str(teacher_path), adapter_name="teacher")
# During stats phase:
model.set_adapter("teacher")  # switch to teacher LoRA
# ... compute teacher logprobs via local forward pass ...
model.set_adapter("default")  # switch back to student LoRA
```

Pro: No extra server needed.
Con: Teacher must share the same base model. Uses training GPU memory for the forward pass.

### 6.3 Base model as teacher — no `--teacher-*` flags

The unmodified base model (no LoRA) is the teacher.

```python
model.disable_adapter_layers()  # forward pass with base weights only
# ... compute teacher logprobs ...
model.enable_adapter_layers()   # re-enable student LoRA
```

Useful for KL-regularized training: prevents the student from diverging too far from the base model.

---

## 7. Distributed Training: How 4 GPUs Coordinate <a name="7-distributed"></a>

### 7.1 Sharding strategy

Samples are NOT split across GPUs by data. Instead:

1. **ALL ranks have ALL samples** (broadcast from rank 0)
2. **Mini-batches are sharded** — if there are 100 mini-batches, rank 0 processes ~33, rank 1 ~33, rank 2 ~34
3. Each rank accumulates gradients from its mini-batches
4. **All-reduce SUM** combines gradients from all ranks
5. Every rank does the same optimizer step → identical weights

### 7.2 The all-reduce SUM trick for logprobs

For the stats phase (teacher/student logprobs), each rank computes logprobs for its **shard** of samples. The result tensor is `[N, max_action_len]` with zeros for samples this rank didn't process.

```
Rank 0:  [sample0_logprobs, 0, 0, sample3_logprobs, 0, 0, ...]
Rank 1:  [0, sample1_logprobs, 0, 0, sample4_logprobs, 0, ...]
Rank 2:  [0, 0, sample2_logprobs, 0, 0, sample5_logprobs, ...]
          ─────────────────── SUM ───────────────────────────
Result:  [sample0, sample1, sample2, sample3, sample4, sample5, ...]
```

This works because each rank writes to disjoint rows, and 0 + X = X.

### 7.3 Gradient accumulation pattern

```python
optim.zero_grad()
for bi in my_batch_order:               # each rank processes its shard
    loss = compute_loss(batch) / n_total_batches  # scale by TOTAL batches across all ranks
    loss.backward()                       # accumulates into .grad
allreduce_coalesced_grads(trainable_params)  # SUM across ranks
clip_grad_norm_(trainable_params, 1.0)
optim.step()                              # identical step on all ranks
```

The `/n_total_batches` scaling ensures that after SUM all-reduce, the gradient equals the mean over all samples.

---

## 8. Per-Skill Teacher Routing: Multi-Server Setup <a name="8-per-skill-routing"></a>

When you train on multiple games, each game can have a different teacher (trained specialist).

### 8.1 CLI specification

```bash
--teacher-url "structured_data_reasoning=http://localhost:9000,multistep_task=http://localhost:9002,tau_tool_calling=http://localhost:9001"
--teacher-model "structured_data_reasoning=tarsur909/...-structured-10,multistep_task=tarsur909/...-multistep-10,tau_tool_calling=tarsur909/...-toolcalling-40"
```

### 8.2 Routing logic

Inside `query_teacher_logprobs_vllm`, for each sample:

```python
def fetch_one(idx):
    skill = game_names[idx]  # e.g., "structured_data_reasoning"
    url = get_skill_value(skill, teacher_urls)    # → "http://localhost:9000"
    model_name = get_skill_value(skill, teacher_models)  # → "tarsur909/...-structured-10"
    return _fetch_logprobs_one(url, model_name, token_ids, ...)
```

Each sample is routed to the correct teacher server based on which game produced it.

### 8.3 How game_names flow through the code

```
collect_grpo_rollouts()
  → per group, picks game from game_mix based on weights
  → GRPOSample.game_name = game_spec.name  (e.g., "structured_data_reasoning")

main loop:
  game_names = [s.game_name for s in samples]  # ["sdr", "mt", "sdr", "tc", ...]

query_teacher_logprobs_vllm(..., game_names=game_names)
  → for sample i: lookup game_names[i] in teacher_urls dict
```

---

## 9. Key Helper Functions Reference <a name="9-helpers"></a>

### `build_prompt_plus_action(tokenizer, msgs, action_str, tools)` (ppo.py:81)

Converts chat messages + model completion into a single token sequence.

```python
prompt_ids = tokenizer.apply_chat_template(msgs, tools=tools, ...)  # [P]
action_ids = tokenizer(action_str, add_special_tokens=False)["input_ids"]  # [A]
return torch.cat([prompt_ids, action_ids]), P, A
```

Returns: `(token_ids [P+A], prompt_len P, action_len A)`

### `per_token_action_logprobs(logits, input_ids, prompt_lens, action_lens)` (ppo.py:188)

Given model output logits `[B, T, V]`, computes per-token log-probabilities for each action token.

```
logits at position t predict the token at position t+1.
So to get log P(token at position P) = log_softmax(logits[P-1])[token_P]

Action tokens are at positions [P, P+1, ..., P+A-1]
Their logprobs come from logits at positions [P-1, P, ..., P+A-2]
```

Returns: `List[Tensor]` — B tensors, each of shape `[action_lens[i]]`

Processes in chunks of 512 positions to avoid OOM.

### `logprob_action_tokens(logits, input_ids, prompt_lens, action_lens)` (ppo.py:99)

Same as above but returns a **scalar** per sample (sum or mean of per-token logprobs).
Used for the SFT loss component, not for distillation.

### `_fetch_logprobs_one(teacher_url, model_name, token_ids, prompt_len, action_len)` (train_distill.py:295)

Sends a single sample to a vLLM teacher server and extracts action logprobs.

```
POST {teacher_url}/v1/completions
  model: model_name
  prompt: [all token IDs]
  max_tokens: 1      ← don't generate
  echo: true          ← return input token logprobs
  logprobs: 1

Response: {choices: [{logprobs: {token_logprobs: [None, -2.3, -1.1, ..., -0.5, ...]}}]}

Extract: token_logprobs[prompt_len : prompt_len + action_len]
```

### `pad_batch(mb_idx, seqs_cpu, prompt_lens, action_lens, pad_token_id)` (train_grpo_optimized.py)

Pads a mini-batch of variable-length sequences to the same length for batched forward pass.
Left-pads with `pad_token_id`, adjusts prompt_lens accordingly.

### `make_sorted_batches(seq_lens, batch_size)` (train_grpo_optimized.py)

Groups samples by sequence length, creates mini-batches that minimize padding waste.
Returns list of lists: `[[sample_idx, sample_idx], [sample_idx, sample_idx], ...]`

### `parse_skill_mapping(spec)` (train_distill.py:254)

Parses `"single_value"` → `{"__default__": "single_value"}`
or `"skill1=val1,skill2=val2"` → `{"skill1": "val1", "skill2": "val2"}`

### `get_skill_value(skill_name, mapping, fallback)` (train_distill.py:278)

Lookup chain: `mapping[skill_name]` → `mapping["__default__"]` → `fallback`

---

## 10. Concrete Numerical Example <a name="10-example"></a>

Settings: 3 GPUs, 8 groups_per_batch, 3 games (sdr/mt/tc at 33/33/34%), mini_batch_size=2

### Iteration 5:

**Step 1: Sync LoRA** — rank 0 saves student LoRA (16 * 7 matrices, ~48MB), tells vLLM to reload.

**Step 2: Collect rollouts** — 8 games played:
- Games 0, 3, 6: structured_data_reasoning (3 games × ~3 turns each = 9 samples)
- Games 1, 4, 7: multistep_task (3 games × ~8 turns each = 24 samples)
- Games 2, 5: tau_tool_calling (2 games × ~5 turns each = 10 samples)
- Total: 43 raw samples

**Step 3: Filter** — 25 info-gathering turns removed → 18 training samples

**Step 4: Tokenize** — 18 samples, typical sequence lengths 500-3000 tokens.
Example sample:
```
sample[7]: game_name="multistep_task", seq_len=1200, prompt_len=900, action_len=300
```
Mini-batches (sorted by length): [[0,1], [2,3], [4,5], [6,7], [8,9], [10,11], [12,13], [14,15], [16,17]]
→ 9 mini-batches

**Step 5: Stats phase**

Teacher logprobs:
- Rank 0 queries samples [0, 3, 6, 9, 12, 15] → routes to correct teacher server
  - sample 0 (game=sdr) → POST http://localhost:9000/v1/completions
  - sample 3 (game=mt) → POST http://localhost:9002/v1/completions
  - sample 6 (game=tc) → POST http://localhost:9001/v1/completions
- Rank 1 queries samples [1, 4, 7, 10, 13, 16]
- Rank 2 queries samples [2, 5, 8, 11, 14, 17]

16 concurrent threads per rank → 48 total concurrent queries across all ranks.

All-reduce assembles complete `teacher_logp_padded [18, max_al]`.

**Step 6: Training**

9 mini-batches sharded: rank 0 gets 3, rank 1 gets 3, rank 2 gets 3.
`n_total_batches = 9`

Rank 0 processes mini-batch [0, 1]:
```python
# Forward pass through student model:
logits = model(input_ids=[seq0_padded, seq1_padded])  # [2, T, 151936]
new_logp = per_token_action_logprobs(logits, ...)
# new_logp[0] shape: [action_lens[0]], e.g., [150]
# new_logp[1] shape: [action_lens[1]], e.g., [200]

# For sample 0 (reverse_kl):
# new_lp = [-2.3, -1.1, -4.5, -0.8, ...]  (150 values, HAS GRAD)
# teacher_lp = [-1.9, -1.3, -3.8, -0.5, ...]  (150 values, detached)
# diff = [-0.4, 0.2, -0.7, -0.3, ...]
# loss = mean(diff^2) = mean([0.16, 0.04, 0.49, 0.09, ...]) = 0.195

loss = (0.195 + 0.230) / 2 / 9  # scale by mini-batch size and total batches
loss.backward()
```

After all 3 mini-batches on rank 0, same on rank 1 and 2:
```python
allreduce_coalesced_grads(params)  # SUM across 3 ranks
clip_grad_norm_(params, 1.0)
optim.step()  # identical update on all ranks
```

**Step 7: Logging**
```
[iter 5] step=45 reward=0.444 gap=0.082 loss=0.195 samples=18
  [mix] sdr=0.500(6) | mt=0.333(8) | tc=0.500(4)
```

---

## 11. Debugging Guide: What the Log Lines Mean <a name="11-debugging"></a>

```
[iter 21] step=2335 reward=0.362 gap=0.0871 loss=-0.0871 ratio=0.0000 clip=0.000 KL=-0.0871 samples=217 info_filt=301 trunc=0 collect=96.0s train=870.1s (tok=2.9s stats=71.0s grad=796.3s pad_eff=99.7%)
  [mix] structured_data_reasoning=0.490(32) | multistep_task=0.340(185)
```

| Field | Meaning | Healthy range |
|---|---|---|
| `reward` | Average game reward (diagnostic only) | 0.3-0.7 |
| `gap` | mean(teacher_lp - student_lp). Teacher more confident. | Should DECREASE over training |
| `loss` | Average loss value per sample | reverse_kl: should decrease; ppo_surrogate: near 0 |
| `ratio` | mean(exp(new_lp - old_lp)). Only for ppo_surrogate. | ~1.0 (0 for reverse_kl) |
| `clip` | Fraction of tokens clipped. Only for ppo_surrogate. | 5-20% |
| `KL` | mean(student_lp - teacher_lp) or mean(old_lp - new_lp) | Magnitude < 0.5 |
| `samples` | Training samples after filtering | 100-300 typical |
| `info_filt` | Info-gathering turns removed | 200-400 typical |
| `collect` | Time to generate rollouts via vLLM | 60-120s |
| `stats` | Time for teacher logprob queries | 30-120s |
| `grad` | Time for gradient computation | 200-800s |
| `pad_eff` | Padding efficiency (1.0 = no wasted compute) | >95% |
| `sdr=0.490(32)` | structured_data_reasoning: avg reward 0.49 from 32 samples | - |

### What to watch for:

**gap increasing**: Student is getting WORSE at matching teacher. Check loss function.

**gap stuck near 0**: Student already matches teacher (or teacher is too similar to base model).

**reward decreasing while gap decreasing**: Student matches teacher but teacher doesn't solve the tasks well. Need better teacher.

**collect time very high**: vLLM generation is slow. Check if vLLM server is overloaded.

**stats time very high**: Teacher vLLM queries are slow. Increase `--teacher-concurrency` or use faster teacher server.

**grad time dominating**: Normal — this is the forward/backward passes on training GPUs. Reduce `groups_per_batch` for faster iterations with less data.

---

## 12. Common Failure Modes <a name="12-failures"></a>

### 12.1 Gap increasing (original reverse_kl bug)

**Symptom**: `gap` grows every iteration, `loss` becomes more negative.

**Cause**: The old `reverse_kl` loss `(new_lp - teacher_lp).mean()` had gradient `∂new_lp/∂θ` which always pushes student logprobs DOWN regardless of teacher. Fixed to MSE: `((new_lp - teacher_lp)^2).mean()`.

### 12.2 BFloat16/Float32 dtype mismatch

**Error**: `RuntimeError: Index put requires the source and destination dtypes match, got Float for the destination and BFloat16 for the source`

**Cause**: `per_token_action_logprobs` computes logprobs in autocast (bfloat16) but stores in float32 result tensors.

**Fix**: Line 258 in ppo.py: `results[i][local_pos] = chunk_logp[i][mask].float()`

### 12.3 Teacher vLLM returns errors

**Symptom**: `[Teacher vLLM] Error for sample X (game=Y): ...`

**Common causes**:
- Teacher server not running or wrong port
- Model name doesn't match what's loaded on the server
- Sequence too long for teacher's max_model_len
- Teacher server OOM

**Debug**: Try manually:
```bash
curl http://localhost:9000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "tarsur909/...", "prompt": [1, 2, 3], "max_tokens": 1, "echo": true, "logprobs": 1}'
```

### 12.4 Student LoRA not syncing to vLLM

**Symptom**: Rollout quality doesn't change between iterations.

**Debug**: Check that `sync_policy` prints success. Check vLLM server logs for adapter reload messages.

### 12.5 All samples filtered

**Symptom**: `All samples exceeded max_seq_len — skipping training step`

**Cause**: Games produce very long conversations. Increase `MAX_SEQ_LENGTH` in config.py or use `--tool-result-max-chars` to truncate tool outputs.

### 12.6 Per-skill routing mismatch

**Symptom**: Samples from game X are sent to wrong teacher server.

**Debug**: Check that skill names in `--teacher-url` match game names exactly (e.g., `structured_data_reasoning` not `sdr`). Print `game_names` list to verify.

### 12.7 NaN loss

**Symptom**: `loss=nan`

**Possible causes**:
- Teacher returns `token_logprobs: null` for some tokens → 0.0 logprob used
- Extreme logprob values causing overflow in MSE or exp()
- Learning rate too high

### 12.8 Very slow iterations

**Breakdown of typical 16-minute iteration** (64 groups, 3 GPUs):
- collect: 90s (vLLM generation)
- stats: 70s (teacher queries + optional old student)
- grad: 750s (forward/backward on training GPUs)

If grad time dominates, this is expected — it's 200+ samples × forward + backward.
Reduce `groups_per_batch` or increase `mini_batch_size` for speed.
