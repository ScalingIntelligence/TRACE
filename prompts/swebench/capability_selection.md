# TRACE Capability Selection Pipeline (SWE-bench Verified)

This document is the **SWE-bench variant** of the TRACE capability selection process.
It mirrors `../general/capability_selection.md` but is filled in for SWE-bench
Verified evaluation runs produced by `mini-swe-agent` against a vLLM-served model.

The discovery + labeling logic is identical to the general pipeline. The only
SWE-bench-specific parts are:

- **Trajectory format** — `<instance_id>/<instance_id>.traj.json` per instance,
  produced by `python -m minisweagent.run.benchmarks.swebench`.
- **Pass / fail signal** — comes from the `swebench.harness.run_evaluation`
  grader's per-instance `report.json` (`resolved: true|false`).
- **Granularity** — failures here are single-file Python bug fixes (~500
  instances on SWE-bench Verified), so capability candidates lean toward
  "edit pattern" rather than "long-horizon tool use".

If you've already read `../general/capability_selection.md`, skim this for the
filled-in placeholders and the SWE-bench-specific instructions, then jump to
**Phase 1: Discovery**.

---

## Configuration

All placeholders below have been filled in for the reference SWE-bench Verified
run that motivated this pipeline (Qwen3.6-27B, mini-swe-agent v2, Modal sandboxes).

- **Model:** `Qwen/Qwen3.6-27B`
- **Benchmark:** `princeton-nlp/SWE-bench_Verified` (500 instances)
- **Agent harness:** `mini-swe-agent` v2.3 (single-rollout, step_limit=150,
  cost_limit=$3, temperature=0.7, top_p=0.95, max_tokens=24000)
- **Trajectory directory:** `runs/baseline/` — each instance has a folder
  `<instance_id>/` containing `<instance_id>.traj.json` (the full multi-turn
  conversation) and (after grading) `report.json` with `resolved: true|false`.
- **Eval results:** `runs/baseline/preds.jsonl` (model patches) +
  `runs/baseline/modal-reports/` (grader output)
- **Number of Phase 2 labeling runs:** `10`
- **Candidates (Phase 1):** `5-8` (SWE-bench failures cluster tightly; <5 is
  too coarse, >8 invents distinctions that don't survive the consistency filter)
- **Coverage threshold (`ρ`):** `0.10`
- **Contrastive gap threshold (`δ`):** `0.20`
- **Cross-run consistency (`K`):** `8` of `10`
- **Output directory:** `pipeline/swebench/`

---

## Why this pipeline applies to SWE-bench

The TRACE method needs only two things from the target environment:

1. A set of trajectories labeled pass / fail.
2. Enough information in each trajectory to attribute the failure to a specific
   skill the agent did or did not exercise.

SWE-bench provides both. Each `<iid>.traj.json` contains:
- The model's full multi-turn reasoning (each `<think>` block from Qwen3.6)
- The shell commands the agent issued (`bash` tool calls inspecting the repo)
- The file diffs the agent emitted as its final submission
- The exit status (`Submitted` with a diff, `LimitsExceeded`, `ContextWindowExceeded`,
  etc.)

And the swebench grader's `report.json` tells you whether the patch actually
passed the held-out tests.

The capability candidates that emerged from the reference run are listed at the
bottom of this file — use them as anchors when judging whether your Phase 1
output is sane (your candidates should look broadly like these in **kind**, not
in literal name).

---

## Step 0: Environment Setup

The SWE-bench capability selection pipeline only needs `numpy`, `matplotlib`,
and the swebench harness (for reading the grader reports). It does **not** need
vLLM or PyTorch — those are only required for the environment-generation step.

```bash
conda env list | grep -E "^trace\s"
if ! conda env list | grep -qE "^trace\s"; then
  conda create -n trace python=3.11 -y
  conda run -n trace pip install numpy matplotlib swebench
fi
conda activate trace
python -c "import numpy, matplotlib, swebench" || pip install numpy matplotlib swebench
```

---

## Step 0.5: Compact the trajectories before sending to the agent

Raw `<iid>.traj.json` files for SWE-bench can be 50k-200k tokens each (the
agent inspects many files). Sending 500 raw trajectories to the discovery
agent is wasteful and will blow the context window.

Run the same `pipeline/_summarize.py` used for tau2-bench, but with the
SWE-bench-specific extractor that:

- Keeps the PR description (the input), the agent's first 3 `<think>` blocks,
  the list of bash commands issued (names + truncated args), the final diff,
  and the exit status.
- Drops verbose tool outputs (file reads, test stdout) beyond the first
  ~300 chars each.

```bash
python pipeline/_summarize.py \
  --input-dir runs/baseline/ \
  --output runs/baseline/summaries.jsonl \
  --benchmark swebench \
  --max-content-chars 400 \
  --max-tool-chars 200
```

Each line of `summaries.jsonl` is a single trajectory in compact form. The
discovery agent reads this file, not the raw `*.traj.json` files.

---

## Phase 1: Discovery (1 run)

The agent reads `summaries.jsonl` plus the grading report and proposes
candidate capabilities. This runs **once** to fix the candidate set.

### Agent Prompt — Phase 1

Copy the prompt below, fill in the `{PLACEHOLDERS}`, and send it to an LLM agent
(Claude, Codex, etc.) along with `summaries.jsonl` and the per-instance
`report.json` files.

```
You are a capability discovery agent. You are given evaluation results from a
model called {MODEL_NAME} that was tested on SWE-bench Verified — 500 single-
file Python bug-fix tasks from open-source projects. Each trajectory is
labeled "resolved" (target test passes after applying the model's patch) or
"unresolved" (it doesn't).

The compacted trajectories are at: {SUMMARIES_FILE}
The per-instance grading reports are under: {REPORTS_DIR}

Your job is to propose CANDIDATE CAPABILITIES that the model may consistently
lack when fixing bugs in real codebases. A "capability" is a general bug-fix
skill (e.g., "trace the root cause across the call graph", "preserve the
existing tests' invariants") — NOT a domain-specific rule or a particular
repository's idiom.

This is the DISCOVERY phase. A separate LABELING phase will use your
candidates to label every (trajectory, capability) pair, so the names you
propose will be reused verbatim. Make them clear, distinct, and well-described.

## What to look for in SWE-bench failures

Read the failed trajectories carefully. For each, ask:

- Did the model emit a parseable patch at all, or did it run out of turns /
  tokens / context?
- If it emitted a patch, what kind of edit was it? Did it:
  - Modify the same line the test exercises (symptom site) when the real bug
    is several functions away?
  - Get the right semantics but wrong placement (wrong file, wrong block)?
  - Fix the bug at one location but miss a parallel site that also needs the
    same fix?
  - Add new code instead of using existing helpers / constants the codebase
    already provides?
- What did the model say in its `<think>` blocks? Does it correctly identify
  the bug? Does it reason about which file to edit before editing? Does it
  hallucinate file contents?

These are the **kinds** of distinctions that tend to survive the consistency
filter. Capabilities that are too coarse ("understands Python") or too narrow
("knows django ORM") usually wash out.

## Instructions

1. **Read summaries.jsonl.** Bucket every trajectory by resolved / unresolved.
   For unresolved ones, also pull the corresponding report.json to see WHICH
   tests failed (FAIL_TO_PASS vs PASS_TO_PASS — `tests_status` in the report).

2. **Propose up to {N_CANDIDATES} candidate capabilities.** Examples of
   GENERAL bug-fix capabilities (these are illustrative — invent your own
   based on what you actually see):
   ✓ `deep_call_graph_traversal` — finding the root cause 3+ hops down
     the call chain instead of patching the symptom site
   ✓ `semantic_logic_precision` — implementing the correct algorithm vs.
     a plausible-looking but wrong one
   ✓ `multi_site_consistency` — fixing all parallel locations that share
     the bug, not just the one the failing test hits
   ✓ `diff_context_precision` — placing the edit at the structurally
     correct location vs. somewhere that compiles but doesn't fix the bug
   ✓ `structural_refactor_awareness` — using existing helpers / constants
     the codebase already exposes instead of inlining new logic

   Capabilities to AVOID:
   ✗ `understands_django_orm` (domain-specific)
   ✗ `python_syntax` (too coarse)
   ✗ `fixes_off_by_one_in_pagination` (a specific bug pattern, not a skill)

3. **For each candidate, provide:**
   - A short name (2-5 words, snake_case). This name will be reused verbatim
     in the labeling phase, so make it clear and unambiguous.
   - A 2-3 sentence description that defines the capability precisely enough
     that a different analyst could consistently decide whether a given
     trajectory needs it and whether the agent demonstrated it.
   - 2-4 example trajectory IDs (mix of resolved and unresolved) that
     illustrate the capability, with a brief note explaining each example.

## Output Format

Return ONLY a JSON object with this exact structure (no other text):
{
  "candidates": [
    {
      "name": "capability_name_in_snake_case",
      "description": "2-3 sentence definition. Be specific enough that a different analyst could consistently classify trajectories against it.",
      "example_trajectories": [
        {"task_id": "django__django-12273", "resolved": false, "note": "model patched serialize() but bug was in _normalize_field() called by 3 other public methods"},
        {"task_id": "astropy__astropy-12907", "resolved": true, "note": "model traced through compound_model into _separable correctly"},
        ...
      ]
    },
    ...
  ]
}

## Important Guidelines

- Aim for 5-{N_CANDIDATES} meaningful, distinct capability categories
- Each capability must be clearly distinguishable — avoid overlap
- Descriptions must support the LACKING vs PRESENT distinction: a labeler
  reading your description should be able to tell, on an unresolved
  trajectory, whether the failure was due to this capability specifically
  or some other reason
- Do NOT give hints about how to solve specific tasks — describe the SKILL
- Do NOT reference specific repos, file names, or function names that
  only appear in one trajectory
- A capability that explains 60%+ of unresolved trajectories is great — but
  also propose 1-2 less-frequent ones; the filter will drop them if they
  don't pass `δ`, but if they survive, they're high-value training targets
```

### Running Phase 1

1. Send the prompt + `summaries.jsonl` + the report directory to the LLM agent.
2. Save the output to `{OUTPUT_DIR}/candidate_capabilities.json`.

Review the candidates to make sure they're distinct and well-described before
proceeding to Phase 2.

---

## Phase 2: Labeling ({N_RUNS} parallel subagents)

Identical to the general pipeline (see `../general/capability_selection.md`
§Phase 2). Each subagent labels every (trajectory, capability) pair as **NA**,
**PRESENT**, or **LACKING**. The SWE-bench-specific notes:

- **NA** is more common in SWE-bench than in tau2-bench, because a capability
  like `multi_site_consistency` only applies to instances where the bug really
  does appear at multiple sites — that's maybe 10% of the dataset. Be honest
  about NA; don't force every capability to apply.

- **PRESENT vs LACKING on unresolved trajectories**: the patch may have a
  syntactically correct edit that the model THOUGHT addressed the bug. If
  the model's reasoning shows it understood the call graph correctly but
  the actual edit landed in the wrong place, label `deep_call_graph_traversal`
  as PRESENT and (say) `diff_context_precision` as LACKING. The skill being
  judged is the reasoning, not the outcome.

- Use the EXACT capability names from Phase 1. Do not rename or invent.

The prompt body, three-way label definitions, output schema, and consistency
requirements are all the same — see the general pipeline document.

---

## Step 3: Aggregation with Dual Thresholds and Consistency Filter

Identical to the general pipeline. Run `aggregate_capabilities.py`:

```bash
python pipeline/aggregate_capabilities.py \
  --runs {OUTPUT_DIR}/run_*.json \
  --candidates {OUTPUT_DIR}/candidate_capabilities.json \
  --rho 0.10 \
  --delta 0.20 \
  --k 8 \
  --export {OUTPUT_DIR}/selected_capabilities.json
```

### Defaults from the paper

- `ρ = 0.10` (capability accounts for ≥ 10% of unresolved trajectories)
- `δ = 0.20` (lacking rate on unresolved exceeds lacking rate on resolved by 20pp)
- `K = 8` out of `N = 10`

---

## Reference: capabilities discovered for Qwen3.6-27B on SWE-bench Verified

These are the actual categories that came out of the discovery + labeling
process on a Qwen3.6-27B baseline run (264 resolved, 123 unresolved, 73 empty
patches, 13 grader errors). They are listed here only as a **sanity check**
for your own Phase 1 output — your candidates should look broadly similar in
KIND. Don't copy the names; let the discovery agent invent its own and judge
later whether they overlap.

| Discovered category | Description | Coverage |
|---|---|---:|
| `semantic-logic-precision` | Implements a logically incorrect algorithm — misunderstands a domain protocol, math rule, or language semantic. Syntactically valid, solves the wrong problem. | 61.0% |
| `deep-call-graph-traversal` | Fails to trace through 3-5 call hops to find the root cause; patches the symptom site even though multiple callers share the buggy helper. | 16.3% |
| `diff-context-precision` | Edit is semantically correct but placed at a structurally wrong location (wrong block, wrong file, wrong indent). | 8.1% |
| `multi-site-consistency` | Fixes the bug at one location but misses a parallel site that also requires the same fix. | 8.1% |
| `structural-refactor-awareness` | Inlines new logic instead of leveraging existing helpers, constants, or patterns the codebase already exposes. | 6.5% |

The training environments described in `environment_generation.md` of this
folder were authored against these categories.

---

## Output

The final output is `{OUTPUT_DIR}/selected_capabilities.json` with the same
schema as the general pipeline (see general doc § Output).

This file is the input to the next step:
[`environment_generation.md`](./environment_generation.md).
