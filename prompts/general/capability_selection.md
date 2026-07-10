# TRACE Capability Selection Pipeline

This document describes the contrastive capability selection process from the TRACE paper.
Given succeeded and failed agent trajectories, an LLM agent labels every (trajectory,
capability) pair as **NA**, **PRESENT**, or **LACKING**, then we apply the paper's
dual-threshold filter (`ρ` for coverage, `δ` for contrastive gap) and a cross-run
consistency check.

The process is **environment-agnostic** — it works with any evaluation results that contain
pass/fail trajectories, not just tau2-bench.

---

## Configuration

{CONFIG_SUMMARY}

---

## Overview

This pipeline follows the TRACE paper's two-phase design:

1. **Phase 1 — Discovery (1 run):** An LLM agent reads all trajectories and proposes up to
   {N_CANDIDATES} candidate capabilities, each with a name, description, and example
   trajectories. Discovery runs **once** to fix the candidate set.

2. **Phase 2 — Labeling ({N_RUNS} parallel subagents):** Given the fixed candidates, each
   subagent independently labels every (trajectory, capability) pair as **NA**, **PRESENT**,
   or **LACKING**. None of the subagents can see each other.

3. **Step 3 — Aggregation:** A small Python script computes coverage `Cov(c)` and contrastive
   gap `Δ(c)` per run, applies the dual-threshold filter `(Cov ≥ {RHO}, Δ ≥ {DELTA})`,
   and keeps only capabilities that pass in at least `{K_CONSISTENCY}` of `{N_RUNS}` runs.

**The agent does the labeling, not code.** The only script we run is the aggregation step.

### Why three-way labels (NA / PRESENT / LACKING) and not pass/fail

A trajectory can fail for many reasons unrelated to a given capability. The labeler must
judge — for each capability separately — whether the failure was specifically *due to lacking
that capability*, or whether the agent had the capability fine but failed for some other
reason. The three labels capture this:

- **NA** — capability is not relevant to the task
- **PRESENT** — capability is relevant AND the agent demonstrated it
- **LACKING** — capability is relevant AND the agent failed to demonstrate it

This 3-way distinction is what makes the contrastive gap `Δ(c)` meaningful: a capability with
high LACKING rate on failed trajectories AND high LACKING rate on successful trajectories
has a small `Δ` and gets filtered out — it's noise, not a true weakness.

---

## Phase 1: Discovery (1 run)

The agent reads through all pass/fail trajectories and proposes candidate capabilities.
This runs **once** to establish the fixed dictionary of capabilities that Phase 2 will label.

### Agent Prompt — Phase 1

Copy the prompt below, fill in the `{PLACEHOLDERS}`, and send it to an LLM agent
(Claude, Codex, etc.) along with the evaluation results file(s).

```
You are a capability discovery agent. You are given evaluation results from a model
called {MODEL_NAME} that was tested on a set of tasks. Each trajectory is either a
success (reward = 1.0) or a failure (reward < 1.0).

The evaluation results are located at: {EVAL_RESULTS}

Your job is to propose CANDIDATE CAPABILITIES that the model may consistently lack.
A "capability" is a general skill or competency (e.g., "multi-step planning",
"error recovery") — NOT a domain-specific rule or task-specific hint.

This is the DISCOVERY phase. A separate LABELING phase will use your candidates to
label every (trajectory, capability) pair, so the names you propose will be reused
verbatim. Make them clear, distinct, and well-described.

## Instructions

1. **Read through the evaluation results.** For each trajectory, examine:
   - What the task required the agent to do
   - What the agent actually did (actions taken, tool calls made, responses given)
   - Whether it succeeded or failed, and WHY it failed

2. **Propose up to {N_CANDIDATES} candidate capabilities.** Think about capabilities
   in terms of:
   - WHAT KIND of reasoning or planning is required?
   - WHAT PATTERN of actions is needed?
   - WHERE does the agent break down — understanding, planning, execution, or verification?

   Capabilities must be GENERAL, not domain-specific:
   ✓ "multi_step_state_modification", "search_then_act_pipeline", "numerical_verification"
   ✗ "flight_rebooking_procedure", "retail_return_policy_adherence" (too domain-specific)

   Think of capabilities as things that would improve ANY agent, not just one in a
   specific domain.

3. **For each candidate, provide:**
   - A short name (2-5 words, snake_case). This name will be reused verbatim in the
     labeling phase, so make it clear and unambiguous.
   - A 2-3 sentence description that defines the capability precisely enough that a
     different analyst could consistently decide whether a given trajectory needs it
     and whether the agent demonstrated it.
   - 2-4 example trajectory IDs (mix of failed and passed) that illustrate the
     capability, with a brief note explaining each example.

## Output Format

Return ONLY a JSON object with this exact structure (no other text):
{
  "candidates": [
    {
      "name": "capability_name_in_snake_case",
      "description": "2-3 sentence definition. Be specific enough that a different analyst could consistently classify trajectories against it.",
      "example_trajectories": [
        {"task_id": "T1", "reward": 0.0, "note": "brief reason this task demonstrates the capability gap"},
        {"task_id": "T5", "reward": 1.0, "note": "brief reason — model succeeded here"},
        ...
      ]
    },
    ...
  ]
}

## Important Guidelines

- Aim for 4-{N_CANDIDATES} meaningful, distinct capability categories
- Each capability must be clearly distinguishable from the others — avoid overlap
- Descriptions must support the LACKING vs PRESENT distinction: a labeler reading
  your description should be able to tell, on a failed trajectory, whether the failure
  was due to this capability specifically or some other reason
- Do NOT give hints about how to solve specific tasks — describe the CAPABILITY
- Do NOT reference specific domain rules, dollar amounts, or procedures
```

### Running Phase 1

1. Send the prompt + your evaluation results to the LLM agent
2. Save the output to `{OUTPUT_DIR}/candidate_capabilities.json`

Review the candidates to make sure they're distinct and well-described before proceeding
to Phase 2.

---

## Phase 2: Labeling ({N_RUNS} parallel subagents)

Given the fixed candidates from Phase 1, run {N_RUNS} independent labeling runs in
parallel using **subagents**. Each subagent gets a fresh, isolated context containing only
the Phase 2 prompt, the candidates from Phase 1, and access to the eval files.

Because labeling each (trajectory, capability) pair as NA / PRESENT / LACKING is a
genuinely difficult judgment call (especially the PRESENT vs LACKING distinction on failed
trajectories), independent subagents will disagree on borderline cases. That cross-run
disagreement is exactly what the consistency filter measures.

### Why subagents instead of sequential runs

- **True independence:** A single agent doing 10 sequential runs would see all prior runs
  in its context, biasing later runs. Subagents have zero shared context.
- **Parallelism:** All {N_RUNS} runs finish in roughly the time of one.
- **Single command:** The orchestrating agent fans out, collects results, and runs the
  aggregation script — the user only issues one instruction.

### Agent Prompt — Phase 2

Copy the prompt below, fill in the `{PLACEHOLDERS}`, and pass it to each subagent.

```
You are a contrastive capability labeling agent. You are given evaluation results from a
model called {MODEL_NAME} and a fixed set of candidate capabilities. Your job is to label
every (trajectory, capability) pair so we can compute coverage and contrastive gap metrics.

The evaluation results are located at: {EVAL_RESULTS}

The candidate capabilities (from a prior discovery phase) are:

{CANDIDATE_CAPABILITIES_JSON}

## Instructions

For each trajectory in the evaluation results AND each candidate capability, assign one of
three labels:

- **NA** — This capability is NOT APPLICABLE to this task. The task does not require this
  capability for successful completion.

- **PRESENT** — This capability IS APPLICABLE to this task, AND the agent successfully
  demonstrated it. The agent did exhibit the capability when it was needed.

- **LACKING** — This capability IS APPLICABLE to this task, AND the agent FAILED to
  demonstrate it. The agent did not exhibit the capability when it was needed.

The crucial distinction is between PRESENT and LACKING on FAILED trajectories: a trajectory
can fail for many reasons. For each capability, you must judge whether the failure was
specifically due to lacking THAT capability, or whether the agent had that capability fine
but failed for unrelated reasons.

Examples:
- A failed trajectory where the agent never needed numerical reasoning at all
  → numerical_reasoning is NA
- A failed trajectory where the agent did the math correctly but called the wrong tool
  → numerical_reasoning is PRESENT (the failure was due to a different capability)
- A failed trajectory where the agent computed the wrong total and that caused the failure
  → numerical_reasoning is LACKING

For SUCCESSFUL trajectories, the labels are still meaningful:
- A successful trajectory where the capability wasn't needed → NA
- A successful trajectory where the capability was needed and the agent did it → PRESENT
- A successful trajectory where the capability was needed but the agent did it wrong yet
  somehow succeeded anyway → LACKING (rare but possible)

## Output Format

Return ONLY a JSON object with this exact structure (no other text):
{
  "attempt": {ATTEMPT_NUMBER},
  "totals": {
    "total_failed": <int — total number of failed trajectories you analyzed>,
    "total_passed": <int — total number of successful trajectories you analyzed>
  },
  "labels": {
    "capability_name_1": {
      "lacking_failed": ["task_id_a", "task_id_b", ...],
      "present_failed": ["task_id_c", ...],
      "na_failed":      ["task_id_d", ...],
      "lacking_passed": ["task_id_e", ...],
      "present_passed": ["task_id_f", ...],
      "na_passed":      ["task_id_g", ...]
    },
    "capability_name_2": { ... },
    ...
  }
}

## Important Guidelines

- Use the EXACT capability names from the candidates — do not rename or invent new ones
- Every trajectory must be labeled for every capability — no skipping
- The six lists for each capability are a partition: each task ID appears in exactly ONE
  of the six lists (lacking_failed | present_failed | na_failed | lacking_passed |
  present_passed | na_passed)
- For each capability, the sum of (lacking_failed + present_failed + na_failed) must equal
  totals.total_failed, and the sum of the three "_passed" lists must equal totals.total_passed
- Be honest about borderline cases. If you genuinely cannot tell whether a capability is
  PRESENT or LACKING on a failed trajectory, lean toward PRESENT — false LACKING labels
  inflate Cov and Δ artificially
- Different subagents will disagree on borderline cases. That's expected and is exactly
  what the cross-run consistency check measures
```

### Running Phase 2 with subagents

The orchestrating agent should spawn {N_RUNS} subagents in parallel — all in a single
message so they execute concurrently — using the `Agent` tool.

For each subagent `i` from 1 to {N_RUNS}:

1. Substitute `{ATTEMPT_NUMBER}` with `i` in the Phase 2 prompt above
2. Substitute `{CANDIDATE_CAPABILITIES_JSON}` with the contents of
   `{OUTPUT_DIR}/candidate_capabilities.json` from Phase 1
3. Pass the substituted prompt to the subagent and instruct it to write its output JSON
   to `{OUTPUT_DIR}/run_{i:02d}.json`

Each subagent will:
- Read the eval result files from the paths in the prompt
- Independently label every (trajectory, capability) pair
- Write its result JSON to the specified path
- Return a brief confirmation to the orchestrator

---

## Step 3: Aggregation with Dual Thresholds and Consistency Filter

Once all {N_RUNS} subagents complete, run `aggregate_capabilities.py` to compute the
paper's metrics and apply the filter.

```bash
python pipeline/aggregate_capabilities.py \
  --runs {OUTPUT_DIR}/run_*.json \
  --candidates {OUTPUT_DIR}/candidate_capabilities.json \
  --rho {RHO} \
  --delta {DELTA} \
  --k {K_CONSISTENCY} \
  --export {OUTPUT_DIR}/selected_capabilities.json
```

### What this does

For each capability `c` and each run `r`, the script computes:

- **`Cov_r(c) = lacking_failed / total_failed`** — fraction of failed trajectories where the
  capability is LACKING. This is the paper's coverage metric.

- **`ER_minus = lacking_failed / (lacking_failed + present_failed)`** — rate of LACKING
  among failed trajectories where the capability is applicable (excluding NA).

- **`ER_plus = lacking_passed / (lacking_passed + present_passed)`** — same rate, but for
  successful trajectories.

- **`Δ_r(c) = ER_minus − ER_plus`** — the contrastive gap. Positive Δ means the capability
  is more often LACKING on failures than on successes.

A capability **passes a single run** iff:

```
Cov_r(c) ≥ ρ   AND   Δ_r(c) ≥ δ
```

A capability is **selected** iff it passes in at least `K_CONSISTENCY` of the `N_RUNS` runs.

### Defaults from the paper

- `ρ = 0.10` (capability must account for at least 10% of failures)
- `δ = 0.20` (lacking rate on failed must exceed lacking rate on successful by 20 points)
- `K = 8` out of `N = 10` (must pass in 8/10 runs)

---

## Output

The final output is `{OUTPUT_DIR}/selected_capabilities.json`:

```json
[
  {
    "skill": "multi_step_state_modification",
    "description": "Tasks requiring two or more sequential state-changing operations where later steps depend on earlier ones",
    "mean_cov": 0.18,
    "std_cov": 0.04,
    "mean_delta": 0.31,
    "std_delta": 0.05,
    "n_runs_passed": 9,
    "n_runs_total": 10,
    "passes_consistency": true,
    "example_failed_cases": ["T12", "T45", "T8", "T22"],
    "status": "PENDING"
  }
]
```

**Fields:**

- **`skill`** — Domain-agnostic capability name (from Phase 1 candidates)
- **`description`** — What this capability involves (from Phase 1)
- **`mean_cov` / `std_cov`** — Mean and std of coverage across runs
- **`mean_delta` / `std_delta`** — Mean and std of contrastive gap across runs
- **`n_runs_passed`** — Number of runs where this capability passed both thresholds
- **`passes_consistency`** — Whether `n_runs_passed ≥ K_CONSISTENCY`
- **`example_failed_cases`** — Representative task IDs labeled LACKING on failures (top 8)
- **`status`** — `"PENDING"` initially; set to `"DONE"` by the environment generation step

This file is the input to the next step: [Environment Generation](./trace_environment_generation.md).

---

## Pipeline Summary

```
{EVAL_RESULTS} (pass/fail trajectories for {MODEL_NAME})
        │
        ▼
┌──────────────────────────────────────┐
│  Phase 1: Discovery (1 run)          │  Agent reads trajectories, proposes
│  Produces candidate_capabilities.json│  up to {N_CANDIDATES} candidates with
│                                      │  names + descriptions + examples
└──────────┬───────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│  Phase 2: Labeling (×{N_RUNS})             │  Parallel subagents. Each labels
│  Three-way labels:                   │  every (trajectory, capability)
│  NA / PRESENT / LACKING              │  pair. Fresh isolated contexts.
│  Produces run_01..run_{N_RUNS}.json        │
└──────────┬───────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│  Step 3: aggregate_capabilities.py   │  Per-run: Cov(c), Δ(c)
│  Filter: Cov ≥ {RHO} AND Δ ≥ {DELTA}    │  Cross-run: must pass in
│  Cross-run consistency: ≥ {K_CONS}/  │  ≥ {K_CONSISTENCY} of {N_RUNS} runs
│  {N_RUNS}                            │
└──────────┬───────────────────────────┘
           │
           ▼
   selected_capabilities.json
   (with Cov, Δ, descriptions, examples, PENDING status)
           │
           ▼
   Next: trace_environment_generation.md
```

---

## Related Files

- **`pipeline/aggregate_capabilities.py`** — Computes Cov / Δ from per-run labeling outputs and applies the dual-threshold + consistency filter. The only script you run in this pipeline.