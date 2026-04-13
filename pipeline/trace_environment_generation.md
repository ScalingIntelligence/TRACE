# TRACE Environment Generation Pipeline

This document describes how to create synthetic training environments that target specific
capabilities identified during the [capability selection step](./trace_capability_selection.md).

An LLM agent (Claude, Codex, etc.) generates the environment. The environment is then
validated by running rollouts on a vLLM server and checking the reward distribution.

---

## Configuration

Before starting, fill in these values. They are referenced as `{PLACEHOLDER}` throughout
this document — replace them with your actual values wherever you see them.

**Required:**

- **`{MODEL}`** — The model you are training / collecting rollouts with. This is the
  HuggingFace model ID or path that vLLM will serve.
  *Example:* `Qwen/Qwen3-30B-A3B-Instruct-2507`

**Optional (have sensible defaults):**

- **`{CAPABILITIES_FILE}`** — Path to `selected_capabilities.json` from the capability
  selection step. If not specified, defaults to `{OUTPUT_DIR}/selected_capabilities.json`
  (the output path from the capability selection pipeline).
  *Default:* `pipeline/selected_capabilities.json`

- **`{GPU_DEVICE}`** — Which GPU to use for the vLLM server. If not specified, the agent
  should run `nvidia-smi` to find a free GPU and use that one automatically.
  *Example:* `2`

- **`{PORT}`** — Port for the vLLM server. Pick any free port on your machine. *Default:* `5050`

- **`{GROUP_SIZE}`** — Number of rollouts per seed/group for GRPO. This determines how many
  parallel attempts the model gets per task. *Default:* `16`

- **`{NUM_SEEDS}`** — Number of different task seeds to collect rollouts for. More seeds =
  more diverse training data. *Default:* `100`

- **`{HINT_RATIO}`** — Fraction of rollouts in each group that get hint injection
  (e.g., 0.25 means 4 out of 16 get hints). *Default:* `0.25-0.5`

- **`{OUTPUT_DIR}`** — Directory for all outputs. *Default:* `pipeline/`

---

## Step 0: Environment Setup

Before running any of the steps below, make sure a conda environment named `trace`
exists with all required packages. The environment generation pipeline needs the
full ML stack (vLLM, PyTorch, transformers, etc.) plus the project's own dependencies.

The agent should run these checks/commands at the start:

```bash
# Check if the trace conda env already exists
conda env list | grep -E "^trace\s"

# If it does NOT exist, create it and install all required packages
if ! conda env list | grep -qE "^trace\s"; then
  conda create -n trace python=3.11 -y
  conda run -n trace pip install -r requirements.txt
  conda run -n trace pip install vllm
fi

# Activate it for subsequent commands
conda activate trace

# Verify the key packages are importable
python -c "import torch, transformers, vllm, requests, numpy; print('OK')"
```

If the env already exists but is missing packages, install them:

```bash
conda activate trace
pip install -r requirements.txt
pip install vllm
```

**Note:** Installing vLLM and PyTorch is heavy (multi-GB download, several minutes).
If the cluster already has a working `games` env from `environment.yml`, the agent
may use that as a fallback by activating `games` instead — both envs should work
since the project's `requirements.txt` is the same.

---

## Overview

This pipeline processes **exactly one PENDING capability per invocation**. The user
controls iteration by re-invoking the agent — each call picks the next pending
capability, generates its environment, validates it, marks it DONE, and stops.

On each invocation, the agent will:

1. Read `{CAPABILITIES_FILE}` and find all entries with
   `status: "PENDING"`, sorted by `mean_delta` descending (strongest weakness first).
2. If there are no PENDING capabilities, report "all capabilities done" and exit.
3. Otherwise, pick the **top one** (highest mean Δ) and process only that one
   through Steps 1-4.
4. After marking it DONE, **stop and report**. Do not move on to the next
   capability. The user will re-invoke the agent when they want the next
   environment generated.

---

## Step 1: Generate the Environment

> **Execution note:** Process exactly ONE capability per invocation — the top
> PENDING capability in `{CAPABILITIES_FILE}` (sorted by `mean_delta`
> descending). After completing Steps 1-4 for that one capability, stop and report.
> Do not loop to the next PENDING capability. The user controls iteration by
> re-running the pipeline.

Give the following prompt to an LLM agent (Claude, Codex, etc.) to generate the environment
for a specific capability. The agent will read the capability description and the existing
codebase, then produce a new environment targeting that capability.

### Agent Prompt

Copy the prompt below and fill in the `{PLACEHOLDERS}` from `{CAPABILITIES_FILE}`.

```
You are an environment designer for reinforcement learning. Your job is to create a
synthetic training environment that targets a specific capability that a model needs
to improve on.

## The Capability to Target

Name: {CAPABILITY_NAME}
Description: {CAPABILITY_DESCRIPTION}
Example failed task IDs: {EXAMPLE_FAILED_CASES}
Current success rate: {CAPABILITY_SUCCESS_RATE}

## The Model Being Trained

Model: {MODEL}

## What You Need to Do

Create a synthetic training environment (a Python game class) that:
1. Tests ONLY this specific capability — the reward must depend solely on whether the
   model demonstrates this capability correctly
2. Generates diverse scenarios from different random seeds (so the model can't memorize)
3. Has appropriate difficulty — not so easy the model always gets 1.0, not so hard it
   always gets 0.0

## Environment Interface

**Important:** `game_registry.py` declares a legacy `GameEnv` protocol with methods
like `observe()`, `legal_actions()`, etc. **IGNORE that protocol.** It is not what
the rollout script actually uses. The real contract your env must satisfy is the
one called by `collect_rollouts.py`'s `run_episode_toolcall` function, which is
designed for chat-based tool-calling agents.

Read `collect_rollouts.py` (specifically `run_episode_toolcall`) before writing
your env to confirm the exact interface. As of the current codebase, your game
class must expose:

**Methods:**
- `reset(seed: int) -> None` — initialize a fresh scenario from the seed
- `get_system_prompt() -> str` — the agent's system prompt for this episode
- `get_tool_schemas() -> list[dict]` — OpenAI-format tool schemas the agent can call
- `get_messages() -> list[dict]` — conversation history in OpenAI format
  (`[{"role": "user|assistant|tool", "content": ..., ...}, ...]`)
- `step(action: str) -> None` — process a JSON action string of the form
  `{"name": "tool_name", "arguments": {...}}` or
  `{"name": "respond_to_user", "arguments": {"message": "..."}}`
- `get_summary() -> dict` — final episode summary, must include at least
  `reward` and `steps`; can include `reason`, `tool_calls`, `domain`, etc.

**Attributes:**
- `done: bool` — True when the episode is over
- `_conversation: list` — raw conversation log (the rollout script reads this
  directly into the saved trajectory)

**Optional but recommended:**
- `max_steps: int` (or `_max_steps`) — episode timeout in steps
- `_finalize_with_verification()` — called if the episode hits max_steps without
  the agent terminating cleanly

You also need to register it as a `GameSpec` so `collect_rollouts.py` can find it:

```python
from game_registry import GameSpec, register_game

register_game(GameSpec(
    name="capability_<name>",
    make_env=lambda **kw: YourGameClass(**kw),
    extract_action=your_extract_action_fn,
    system_prompt="...",
    max_gen_tokens=1024,
))
```

If `collect_rollouts.py` ever calls a method you haven't implemented, the rollout
will crash with `AttributeError`. Re-read `run_episode_toolcall` until you've
covered every method/attribute it touches on the game object.

## Format Fidelity

The synthetic environment must preserve the target environment's format so that a
model trained on it transfers back to the original benchmark. Specifically, match:

- **Tool schemas** — exact function names, parameter names, parameter types, return types
- **State representation** — the same kind of database / observation structure
- **Observation format** — the same system prompt style, the same way tools and
  conversation history are presented to the model
- **Policy constraints** — any rules the original environment enforces

To learn these, read the eval result files (the same ones used in the capability
selection step). Look at successful trajectories to see what tools the agent called,
what arguments they took, what the user messages looked like, and what the system
prompt was. Your synthetic environment must produce trajectories that match this
format — otherwise the model trained on it won't transfer back to the original
environment.

## Anti-pattern: do not wrap the original simulator

If you find the original benchmark's source code in the repo (e.g., a tau2-bench
or other benchmark framework directory), **DO NOT** instantiate or wrap its
simulation runner. The point of a synthetic environment is to be MINIMAL and
TARGETED: a small, self-contained piece of code that exercises ONE capability
with a dense, attributable reward signal.

**What "preserve format" actually means:** match the SHAPES, not the
IMPLEMENTATION. Use the same tool function names and parameter signatures, the
same JSON shapes for observations, the same system prompt style — but with your
own from-scratch Python implementation that operates on a small synthetic state
you control.

Concrete example of the right approach:
- Read the original benchmark's tool definitions to learn that it has, say,
  `get_user_details(user_id)` returning `{"id": ..., "name": ..., "tier": ...}`
- In your env, implement a Python function with the same signature that returns
  the same JSON shape — but reading from a tiny in-memory dict you generate
  from a seed, not from the original DB
- Implement only the 3-5 tools the target capability actually needs, not all
  30 tools the original benchmark has

The original benchmark source code is a REFERENCE. Read it, learn from it,
match its format. Do not import it, instantiate it, or call it.

## Transfer Reasoning

The point of this synthetic environment is NOT to train a model that does well
on the synthetic environment itself. The point is to train a model that does
better on the ORIGINAL target environment — the one whose eval results were
used in capability selection.

Before finalizing the design, explicitly reason about transferability:

1. **What does success on this synthetic env teach the model?**
   Trace through: "if a rollout gets reward 1.0 here, what skill did the model
   demonstrate?" Is that skill the same one that distinguishes pass from fail
   trajectories in the original eval results?

2. **Does the synthetic env exercise the capability in the same way the original
   environment does?**
   - Same tool patterns (same kinds of tool calls in the same order)
   - Same kinds of constraints (policy rules, multi-turn dependencies)
   - Same kinds of distractors (irrelevant info the model has to filter out)
   - Same conversational dynamics if the original is multi-turn

3. **What would NOT transfer?**
   Be honest about the failure modes. Common pitfalls:
   - Simplifying so aggressively that the synthetic env is solvable by a
     pattern the original env doesn't reward
   - Adding novel tool names or argument formats not in the original
   - Making the reward depend on signals the original env doesn't expose
     (e.g., internal state the original benchmark doesn't check)
   - Overfitting to a single failure mode that's narrower than the capability
     description suggests

4. **Sanity check:** Pick 2-3 of the failed task IDs from the capability's
   `example_failed_cases` and ask: "If a model trained on my synthetic env
   was given THIS specific task from the original benchmark, would the
   skills it learned actually help it succeed?" If the answer is "not really,"
   redesign the env.

Write a short transfer rationale (3-5 sentences) at the top of your generated
game file as a docstring. This forces the reasoning to be explicit and gives
the user something to review.

## Reward Design

The reward must isolate the target capability — success should depend primarily on
whether the capability is exercised correctly, not on unrelated aspects of the task.
Look at the eval trajectories to understand:
- WHAT does success look like for this specific capability?
- WHAT can be checked automatically (state, output, action sequence)?
- WHAT are the partial-credit signals (if any)?

Then design a reward function that captures those signals.

**Critical constraint: within-group reward variance.** GRPO learns from variance
within each rollout group. If all rollouts in a group get the same reward, the
gradient is zero and the iteration is wasted. Your reward must produce variance —
some rollouts in each group should succeed, some should fail, and ideally there
should be partial credit between them.

**Use continuous / multi-level rewards** (e.g., 0.0, 0.2, 0.5, 0.8, 1.0) rather
than strict binary 0/1. Multi-level rewards distinguish "totally wrong" from
"close but not quite" and give GRPO a smoother gradient. A typical pattern is to
weight multiple sub-checks:

    reward = 0.6 * action_score + 0.4 * communication_score

where each sub-component is itself a value in [0, 1]. The exact decomposition
depends on the capability — design it from the trajectories, not from a template.

**Make the reward denser than the target environment's reward.** The point of a
synthetic environment is that success depends on the target capability alone, so
the reward signal should be more attributable than what the model gets from the
original benchmark.

## Hint Injection

The environment should support a hint injection mechanism. This is critical for GRPO
training: it ensures at least some rollouts in each group succeed, giving the
training algorithm both positive and negative examples to learn from.

How it works:
1. For a fraction of rollouts in each group (e.g., {HINT_RATIO} of {GROUP_SIZE}), append an
   `<expert_guidance>` block to the system prompt containing soft guidance:

   ```
   <expert_guidance>
   Guidance: <concrete advice about what to think about, what information matters,
   what order to approach things in — but NOT the literal tool calls or arguments.
   The model should still reason through the solution; the hint just nudges it in
   the right direction.>
   </expert_guidance>
   ```

   The hint is NOT the solution. It does not tell the model which tool to call or
   what arguments to pass. Instead, it's guidance that helps the model think about
   the task correctly — like a tutor pointing a student toward the relevant
   information without giving away the answer.

   **Examples of GOOD hints** (depending on the capability being trained):
   - "Pay close attention to the constraint the user mentioned about [X] before
     taking any action."
   - "This task requires multiple steps. Before responding, make sure you've
     gathered all the information you need from the user and the database."
   - "Re-read the user's request carefully — there's an important detail about [Y]
     that changes which approach is correct."
   - "Check whether all preconditions are satisfied before making any state changes."

   **Examples of BAD hints** (do not write hints like these):
   - "Call tool_X with arg=Y" (reveals the solution)
   - "STEP 1: do X. STEP 2: do Y." (reveals the solution)
   - "The answer is Z" (reveals the solution)

   The goal is to push the rollout success rate from ~0% (impossible without help)
   to ~60-80% (helped but not guaranteed), so that hint-injected rollouts provide
   positive examples in each group without trivializing the task. If you find that
   hint-injected rollouts always succeed, your hint is too revealing — make it more
   abstract.

2. Before the training update, strip the `<expert_guidance>` block so the model
   learns to solve the task WITHOUT the hint in the prompt. The correct trajectory
   actions are kept, but the hint tokens are removed so the gradient teaches the
   model the underlying behavior, not how to condition on hint tokens.

See `train/collect_rollouts.py` for the reference implementation
(`get_expert_guidance()` and `strip_expert_guidance()`). You must keep the
`<expert_guidance>` tag name so these existing helpers continue to work — only
the content inside the tag changes from a literal solution to soft guidance.

## Reference Files

Two files matter for this step:

1. **`game_registry.py`** — at the project root. Defines `GameSpec` and
   `register_game()`. **Note:** it also declares a legacy `GameEnv` Protocol with
   methods like `observe()` and `legal_actions()` — IGNORE that protocol. It is
   not what the rollout script actually calls (see "Environment Interface" below
   for the real contract).

2. **`collect_rollouts.py`** — the rollout script. It may live at the project
   root or under `train/`. Find it via `glob "**/collect_rollouts.py"`. Read its
   `run_episode_toolcall` function carefully — that function is the source of
   truth for what methods and attributes your game class must expose. Also read
   `get_expert_guidance()` and `strip_expert_guidance()` to learn the
   `<expert_guidance>` block format your env must support for hint injection.

Do not assume any other files exist. Write the environment from scratch against
the actual interface that `collect_rollouts.py` calls.

## What to Produce

1. A Python file (e.g., `capability_<name>_game.py`) implementing the environment class
2. The `GameSpec` registration at the bottom of the file
3. Scenario generation logic inside `reset()` that creates varied tasks from seeds
4. Reward computation that isolates the target capability
5. Hint injection support (can be toggled on/off)

## Where to put the new file

Place `capability_<name>_game.py` at the **project root**, next to
`game_registry.py`. This is where new game files belong so the
`from game_registry import GameSpec, register_game` import resolves naturally.

Make sure your `register_game(...)` call runs at module load time, AND that
something imports the new module before `collect_rollouts.py` queries the
registry. The simplest way is to add `import capability_<name>_game` at the top
of `collect_rollouts.py` (or wherever `get_game_spec` is first called) so the
module is loaded and registers itself.

If your new environment doesn't fit either existing game type registered in
`collect_rollouts.py` (e.g., `TOOL_CALLING_GAMES` or `OBSERVE_GAMES`), don't
force it into one — add a new game type entry alongside the existing ones and
make sure the rollout loop handles it correctly.
```

### What the agent produces

The agent will create a new Python file in the project root (e.g., `capability_<name>_game.py`)
that contains:
- A game class implementing `GameEnv`
- Scenario generation targeting the specific capability
- Binary reward based on whether the capability was correctly demonstrated
- A `GameSpec` registration so `collect_rollouts.py` can use it

---

## Step 2: Host vLLM Server and Collect Rollouts

> **Execution note:** Before launching, compute the VRAM requirement for
> `{MODEL}` yourself based on its parameter count, dtype, and the configured
> `--max-model-len` and `--gpu-memory-utilization`. Then run
> `nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv,noheader,nounits`
> and pick the first GPU whose `memory.free` exceeds your computed requirement
> (with a small safety margin). The GPU does NOT need to be idle — other small
> processes sharing it are fine as long as the free VRAM is enough.
>
> If no GPU has enough free VRAM, stop and report to the user.
>
> Launch the vLLM server with `run_in_background: true` and wait until you see
> `Uvicorn running on http://0.0.0.0:{PORT}` in the background output before
> sending requests. **Kill the server at the end of this invocation** (after
> Step 4) to free the GPU.

### 2a. Pick a GPU and launch the vLLM server

```bash
# Check which GPUs are available (if {GPU_DEVICE} was not specified, pick a free one)
nvidia-smi

# Launch vLLM server
conda activate trace
export HF_HOME=/workspace/.cache/huggingface
export VLLM_RPC_TIMEOUT=2000
export CUDA_VISIBLE_DEVICES={GPU_DEVICE}   # or whichever GPU is free from nvidia-smi
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
vllm serve {MODEL} \
  --host 0.0.0.0 \
  --port {PORT} \
  --dtype bfloat16 \
  --max-model-len 32000 \
  --enable-lora \
  --max-loras 2 \
  --gpu-memory-utilization 0.9 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

Wait until you see `Uvicorn running on http://0.0.0.0:{PORT}` before proceeding.

### 2b. Collect rollouts on the generated environment

`collect_rollouts.py` may live at the project root or under `train/`. Find it with
`glob "**/collect_rollouts.py"`. Run it from a working directory where
`game_registry.py` is importable (typically the project root, since that's where
both `game_registry.py` and your new `capability_<name>_game.py` live). If the
script is under `train/`, you may need `PYTHONPATH=. python train/collect_rollouts.py`
from the project root to make the import resolve.

```bash
# From the project root (assuming collect_rollouts.py is here or under train/)
python collect_rollouts.py \
  --env capability_<name> \
  --base-url http://localhost:{PORT}/v1 \
  --model {MODEL} \
  --num-seeds {NUM_SEEDS} \
  --num-samples {GROUP_SIZE} \
  --temperature 0.7 \
  --reward-threshold 0.0 \
  --output {OUTPUT_DIR}/rollouts_<capability_name>.json
```

**Key parameters:**

- `--env capability_<name>` — The name registered in `GameSpec` by the generated environment code
- `--model {MODEL}` — Must match the model served by vLLM
- `--num-seeds {NUM_SEEDS}` — Number of different task scenarios to test
- `--num-samples {GROUP_SIZE}` — Rollouts per seed (group size for GRPO)
- `--temperature 0.7` — Sampling temperature for diverse rollouts within each group
- `--reward-threshold 0.0` — Keep ALL trajectories (pass and fail) for reward analysis
- `--self-expert` — Add this flag to enable hint injection for a portion of rollouts

---

## Step 3: Validate Reward Distribution

After collecting rollouts, check the reward distribution to verify the environment difficulty
is appropriate for training.

### Why this matters

GRPO learns from the contrast between good and bad rollouts within each group. If all
rollouts in a group get the same reward (all 1.0 or all 0.0), there is no contrast and
no learning signal. The ideal environment produces a MIX of successes and failures.

### Quick check

```python
import json
import numpy as np

with open("{OUTPUT_DIR}/rollouts_<capability_name>.json") as f:
    data = json.load(f)

rewards = [sim["reward_info"]["reward"] for sim in data["simulations"]]
print(f"Mean reward:   {np.mean(rewards):.3f}")
print(f"Std reward:    {np.std(rewards):.3f}")
print(f"Success rate:  {np.mean([r == 1.0 for r in rewards]):.1%}")
print(f"Zero rate:     {np.mean([r == 0.0 for r in rewards]):.1%}")
print(f"Total rollouts: {len(rewards)}")

# Per-group analysis
group_size = {GROUP_SIZE}
for i in range(0, min(len(rewards), group_size * 10), group_size):  # show first 10 groups
    group = rewards[i:i+group_size]
    print(f"  Group {i//group_size}: mean={np.mean(group):.2f}, "
          f"pass={sum(r == 1.0 for r in group)}/{len(group)}")
```

### Acceptable ranges

- **Mean reward** — Too easy: > 0.8 / Good: 0.2 - 0.6 / Too hard: < 0.1
- **Success rate** — Too easy: > 80% / Good: 20-60% / Too hard: < 10%
- **Groups with all 1.0** — Too easy: > 50% of groups / Good: < 20% of groups
- **Groups with all 0.0** — Too hard: > 50% of groups / Good: < 20% of groups

### Transfer sanity check

A good reward distribution is necessary but NOT sufficient. After confirming
the distribution is in the "good" range, also reason about transferability:

1. **Sample 3-5 successful rollouts** from the generated trajectories. Read them
   end-to-end. Ask:
   - Is the model exercising the target capability, or is it succeeding via
     some shortcut the synthetic env accidentally allows?
   - Could a model trained on these successful trajectories generalize, or
     would it just memorize a narrow pattern specific to the synthetic env?

2. **Sample 3-5 failed rollouts** and ask:
   - Is the failure mode the same one the original eval results flagged for
     this capability?
   - Or is the model failing for some unrelated reason (e.g., the synthetic
     env has bugs, the prompt is confusing, the tools are too restrictive)?

3. **Compare against the eval result trajectories.** Pull up 2-3 of the failed
   task IDs from `example_failed_cases` in the capabilities file. Briefly
   compare them to your synthetic rollouts — do they look like they're
   testing the same thing?

If the answer to "would training on this transfer to the original environment?"
is uncertain or no, the reward distribution doesn't matter — regenerate the
environment with a tighter focus on the original capability.

Include a 2-3 sentence transfer assessment in the report you give the user at
the end of Step 4 ("This env trains X by Y, which should transfer because Z").

### If the distribution is bad

Go back to Step 1 and regenerate the environment with adjustments. **Retry up to 5
times** before giving up — environment design is hard to get right on the first try
and it's worth iterating.

- **Too easy** (most rollouts succeed): Add constraints, require more steps, make
  the task more ambiguous, or reduce hint injection ratio
- **Too hard** (most rollouts fail): Simplify scenarios, provide clearer instructions
  in the system prompt, increase hint injection ratio (e.g., 0.5-0.75), or break the
  capability into simpler sub-capabilities

If after 5 regeneration attempts the reward distribution still falls outside the
"good" range, **stop and report to the user**. Show the best-attempt distribution
and ask whether to skip this capability or try a fundamentally different approach.
Do NOT mark a capability DONE if its distribution is bad.

---

## Step 4: Mark Capability as Complete

Once the reward distribution looks good, update `{CAPABILITIES_FILE}` to
mark this capability as done:

```python
import json

with open("{CAPABILITIES_FILE}") as f:
    caps = json.load(f)

for cap in caps:
    if cap["skill"] == "<capability_name>":
        cap["status"] = "DONE"
        cap["environment_path"] = "{OUTPUT_DIR}/rollouts_<capability_name>.json"
        break

with open("{CAPABILITIES_FILE}", "w") as f:
    json.dump(caps, f, indent=2)
```

This ensures the next invocation of the pipeline skips already-completed capabilities
and picks up where this one left off.

**Stop here.** This invocation processed exactly one capability. After marking it
DONE, **kill the vLLM server** and report to the user:

- Which capability was processed
- The reward distribution stats from Step 3
- The path to the generated game file (`capability_<name>_game.py`)
- The path to the rollouts file (`{OUTPUT_DIR}/rollouts_<name>.json`)
- The number of remaining PENDING capabilities in `{CAPABILITIES_FILE}`

The user will re-invoke the pipeline (with the same instructions) when they want
the next capability processed. On the next invocation, the agent will naturally
pick up the next top PENDING capability because this one is now marked DONE.

---

## Pipeline Summary

```
{CAPABILITIES_FILE} (sorted by mean_delta, with PENDING capabilities)
        │
        │  Pick the TOP PENDING capability (highest mean Δ)
        ▼
┌───────────────────────────────────────┐
│ Step 1: Agent generates environment   │  LLM agent writes the GameEnv class
│  - Reads top PENDING capability       │  + scenarios + hint injection
│  - Produces GameEnv Python file       │
│  - Registers as GameSpec              │
└──────────┬────────────────────────────┘
           │
           ▼
┌───────────────────────────────────────┐
│ Step 2: Host vLLM + Collect Rollouts  │  vLLM serves {MODEL} on :{PORT}
│  - nvidia-smi → pick free GPU        │  collect_rollouts.py runs {NUM_SEEDS} seeds
│  - {GROUP_SIZE} rollouts per seed              │  × {GROUP_SIZE} rollouts each
└──────────┬────────────────────────────┘
           │
           ▼
┌───────────────────────────────────────┐
│ Step 3: Validate Reward Distribution  │  Check: mean reward in [0.2, 0.6]
│  - If bad: regenerate (up to 5 times) │  After 5 failures: stop, ask user
└──────────┬────────────────────────────┘
           │
           ▼
┌───────────────────────────────────────┐
│ Step 4: Mark DONE in capabilities     │  Set status = "DONE"
│  - Kill the vLLM server               │  Free the GPU
└──────────┬────────────────────────────┘
           │
           ▼
   STOP. Report to user.
   User re-invokes the pipeline to process the next PENDING capability.
```

---

## Related Files

- **`game_registry.py`** (project root) — `GameSpec` and `register_game()`. Note:
  also declares a legacy `GameEnv` protocol that is NOT what the rollout script
  actually uses — see "Environment Interface" above for the real contract.
- **`collect_rollouts.py`** (project root or `train/`, find via glob) — Rollout
  collection script. The `run_episode_toolcall` function defines the actual game
  interface. Also contains `get_expert_guidance()` and `strip_expert_guidance()`
  for hint injection.
- **`pipeline/trace_capability_selection.md`** — Upstream pipeline.
