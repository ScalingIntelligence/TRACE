# TRACE Environment Generation Pipeline (SWE-bench Verified)

This document is the **SWE-bench variant** of the TRACE environment generation
process. The general version (`../general/environment_generation.md`) targets
tool-calling agents (tau2-bench, ToolSandBox). SWE-bench is different in three
ways that change the environment shape:

1. **Single-turn / short multi-turn.** SWE-bench predictions are typically a
   single `SEARCH/REPLACE` patch the agent emits after reading the file. The
   environment doesn't need an OpenAI tool-call protocol — it just needs to
   serve a "PR description + file content" prompt and grade the returned
   patch.
2. **Deterministic, verifiable reward.** Reward comes from `pytest` against
   a held-out target test. No reward hacking, no LLM judge — the test either
   passes or it doesn't.
3. **Scenarios are full Python projects.** Each scenario is a 300-600 LOC
   module + a `tests/test_<name>.py` file with 10+ pytest tests. The "bug"
   is baked into the module; the oracle is a `SEARCH/REPLACE` block that
   fixes it.

The general pipeline still applies for the high-level loop (one capability
per invocation, generate → self-test → smoke → mutate → accept). What
changes is the **schema of a scenario** and the **smoke / accept criteria**.

If you've read `../general/environment_generation.md`, skim that for the loop
shape and jump to "What a SWE-bench scenario looks like" below.

---

## Configuration

All placeholders below have been filled in for the reference SWE-bench Verified
run. Adjust per your own deployment.

- **Model:** `Qwen/Qwen3.6-27B`
- **Capabilities file:** `pipeline/swebench/selected_capabilities.json`
- **GPU device:** Auto-detect (the agent will run `nvidia-smi` and pick a free
  GPU pair for TP=2 — Qwen3.6-27B doesn't fit on a single H100 80GB at 32k+
  context with KV-cache headroom)
- **vLLM port:** `8000`
- **Number of candidate scenarios per invocation:** `20` (target acceptance:
  10 valid scenarios per capability)
- **Concurrency:** `4` parallel scenario-generation futures
- **Smoke test:** `N = 10` rollouts per scenario at `temperature = 1.0`,
  `top_p = 0.95`, `max_tokens = 4096`, `enable_thinking = false`
- **Accept threshold:** `rate ≤ 0.5` (model fails the target test on ≥ 50% of
  smoke rollouts, i.e., real difficulty)
- **Mutation budget:** `5` rounds per scenario before giving up
- **Output directory:** `env/swebench/<capability>/`

---

## Step 0: Environment Setup

The environment generation pipeline needs vLLM (to serve Qwen3.6 for both
synthesis and smoke testing) plus the standard ML stack. Same `trace` conda
env as the rest of the pipeline.

```bash
conda env list | grep -E "^trace\s"
if ! conda env list | grep -qE "^trace\s"; then
  conda create -n trace python=3.11 -y
  conda run -n trace pip install vllm transformers torch requests pytest
fi
conda activate trace
```

Verify vLLM imports cleanly:

```bash
python -c "import vllm; import requests; import pytest" || {
  echo "Missing deps. Install: pip install vllm requests pytest" >&2
  exit 1
}
```

---

## Overview

This pipeline processes **exactly one PENDING capability per invocation**.
The user controls iteration by re-invoking the agent — each call picks the
top PENDING capability (highest mean Δ), generates 10 scenarios for it,
marks the capability `DONE`, and stops.

On each invocation, the agent will:

1. Read `{CAPABILITIES_FILE}` and find all entries with `status: "PENDING"`,
   sorted by `mean_delta` descending.
2. If none are PENDING, report "all capabilities done" and exit.
3. Otherwise, pick the **top one** (highest mean Δ) and run the full
   Step 1 → Step 4 loop for it ONLY.
4. After Step 4 marks it DONE, **stop and report**. Do not move on.

---

## What a SWE-bench scenario looks like

A single scenario is a JSON object with these fields. The generator emits
this from an LLM call; the self-test + smoke test verify it.

```json
{
  "name": "satellite-pass-scheduler-deep-bug",
  "module_filename": "scheduler.py",
  "module_code": "import math\nimport collections\n...300-600 LOC of Python...",
  "test_filename": "tests/test_scheduler.py",
  "test_code": "import pytest\nfrom scheduler import ...\n...10+ pytest tests...",
  "target_test": "test_normalize_zero_vector_returns_zero",
  "pr_description": "When the scheduler is asked to compute a summary for ..., the result is off by a factor of ...\n",
  "oracle_search": "def _normalize(v):\n    norm = sum(x * x for x in v)\n    return [x / norm for x in v]",
  "oracle_replace": "def _normalize(v):\n    norm = math.sqrt(sum(x * x for x in v))\n    if norm == 0:\n        return [0.0 for _ in v]\n    return [x / norm for x in v]",
  "naive_fix_description": "A shallow solver might add a guard `if norm == 0: return 0` inside the top-level compute_summary() instead of fixing the underlying _normalize helper.",
  "hint_body": "The bug is somewhere in the normalization step that compute_summary, get_dispersion, and rank_passes all depend on."
}
```

**Critical invariants the self-test will check:**

- `module_code` parses (no SyntaxError).
- The test file is syntactically valid pytest.
- With the **unmodified** module + tests:
  - `target_test` **FAILS**.
  - At least 8 OTHER tests **PASS** (these are the "regression guard" tests).
- With `oracle_search` replaced by `oracle_replace` in the module:
  - `target_test` **PASSES**.
  - The regression guard tests **still pass** (oracle doesn't break anything).
- `oracle_search` appears **verbatim** in `module_code` (the SEARCH/REPLACE
  applies cleanly).
- No leaked open-source library identifiers (`django`, `sympy`, `sphinx`,
  `matplotlib`, `sklearn`, `astropy`, `scipy`, `pandas`, `flask`, `requests`,
  `pytest fixtures`, etc.) appear anywhere in the scenario. Stdlib only.

**Scale invariants the generator should hit:**

- `module_code`: 300-600 LOC. Below 300, the bug is too easy to spot; above
  600, generation timeouts dominate.
- `oracle_search` / `oracle_replace`: usually 1-30 lines. Multi-line oracles
  are fine; the model is good at multi-line SEARCH/REPLACE as long as the
  search appears verbatim somewhere.
- `test_code`: 10+ tests. The target test + 8 regression guards is the
  minimum that gives the self-test enough signal to reject broken scenarios.

---

## Step 1: Generate the Environment

> **Execution note:** Process exactly ONE capability per invocation. After
> completing Steps 1-4 for that one capability, stop and report. Do not loop.

### Agent Prompt

Copy the prompt below and fill in the `{PLACEHOLDERS}` from
`{CAPABILITIES_FILE}`. Send it to an LLM agent that has Python execution and
HTTP access to the vLLM server you start in Step 2.

```
You are a synthetic-environment author for a reinforcement-learning pipeline
that improves an LLM at fixing Python bugs in real codebases.

## The Capability to Target

Name:        {CAPABILITY_NAME}
Description: {CAPABILITY_DESCRIPTION}
Example failed task IDs: {EXAMPLE_FAILED_CASES}
Current LACKING rate on failures: {CAPABILITY_LACKING_RATE}

## The Model Being Trained

Model: {MODEL}

## What You Need to Produce

Ten synthetic Python bug-fix scenarios that ALL target this single capability.
Each scenario is a JSON object with the schema described in the document
you're reading (Step 0 of this section).

The scenarios must:
1. ISOLATE the capability. Reward (test pass/fail) must depend on whether the
   model exercises THIS capability, not some unrelated skill.
2. PRODUCE VARIANCE. A useful scenario is one where the base model fixes the
   bug on some of N=10 rollouts but not all. Saturation in either direction
   (always passes, always fails) is wasted training signal. Target rate 0.2-0.5.
3. PRESERVE FORMAT. The generated module + tests must look like real
   open-source Python: type hints, dataclasses, helper functions, docstrings.
   Use stdlib only (math, collections, dataclasses, typing, enum, itertools).
4. NO LEAKAGE. Do not reference django, sympy, sphinx, matplotlib, sklearn,
   astropy, scipy, pandas, numpy public APIs, requests, flask, fastapi,
   twisted, asyncio, pytest fixtures, pylint, or mypy. Pick a synthetic
   engineering domain instead: PID autotuner, smelter regulator, particle
   tracker, gantry crane controller, satellite pass scheduler, etc.

## Process per scenario

For each of the 10 scenarios:

1. **Pick a domain.** Use one of the synthetic domains from the list at
   the bottom of this prompt (or invent one of similar shape). Different
   scenarios should use different domains.

2. **Generate the scenario JSON via vLLM.** Call the synthesis prompt
   (Step 1 system prompt below) at `temperature=0.85`, `top_p=0.95`,
   `max_tokens=12000`. Use `chat_template_kwargs={"enable_thinking": false}` —
   thinking adds latency without improving structured XML output.

3. **Parse the XML response with the hybrid anchor parser** (Step 1 §"Parser"
   below). Long XML output frequently has missing or duplicated tags; the
   strict regex parsers fail on these. The hybrid parser uses `</close>` or
   the next field's `<open>` as anchors, whichever comes first.

4. **Self-test the scenario** (Step 2 below). If it fails self-test, send
   it back to the synthesis model with a `revise_for_selftest` prompt
   (up to 2 retries). If still failing after 2 retries, drop and move on.

5. **Smoke test the scenario** (Step 3 below). Run N=10 rollouts at
   `temperature=1.0` against the synthesis model. If success rate > 0.5,
   send back a `revise_for_mutation` prompt with one of the mutation
   strategies (up to 5 mutation rounds). The mutation strategies are
   designed to **preserve the bug** but **disguise the surface code** so
   the model can't pattern-match from the PR description.

6. **Accept** if smoke rate ≤ 0.5. Append to the output list.

7. Stop when you have 10 accepted scenarios OR you've burned through
   `N_CANDIDATES * 3 = 60` total generation attempts.

## Synthesis system prompt (capability-targeted)

Send this as the `system` message of every synthesis call (initial + revise).
Replace the `CAPABILITY_DEFINITION` block with the rendered description of
the capability you're targeting.

```text
You are designing synthetic Python bug-fix scenarios for RL training.

Target capability: {CAPABILITY_NAME_HUMAN}.

CAPABILITY DEFINITION:
  {CAPABILITY_DESCRIPTION_HUMAN}

REQUIRED SCENARIO STRUCTURE:
  (Capability-specific — fill in based on the target capability. For
   `deep-call-graph-traversal`, this would be: "Module exposes a high-level
    public API that calls a middle-layer helper that calls a low-level
    helper. The actual BUG is in the low-level helper. 3-5 other public
    methods also depend on the same low-level helper. The target test
    exercises a DIFFERENT public method than where a naive solver would
    look.")

STRICT RULES:
  - HARD NO-LEAKAGE: NO references to django, sympy, sphinx, matplotlib,
    sklearn, astropy, scipy, pandas, numpy public APIs, requests, flask,
    fastapi, twisted, pytest fixtures, pylint, mypy. Synthetic engineering
    domain only. Stdlib-only imports.
  - Module length: 300-600 LOC.
  - One module file + pytest tests (10+ tests).
  - Target test FAILS pre-fix, 8+ regression tests PASS pre-fix.
  - Oracle SEARCH/REPLACE targets the actual bug, not the surface symptom.
  - EMIT EVERY OPEN AND CLOSE TAG. Never write </tag> without <tag>.

Output exactly this XML (no fences, no preamble):

<scenario>
<name>kebab-id</name>
<module_filename>mod.py</module_filename>
<module_code>
# python (300-600 LOC, synthetic-domain)
</module_code>
<test_filename>tests/test_mod.py</test_filename>
<test_code>
# pytest (10+ tests)
</test_code>
<target_test>test_name</target_test>
<pr_description>1 para symptom only; no fix hint</pr_description>
<oracle_search>
# verbatim from module_code — the actual buggy line(s)
</oracle_search>
<oracle_replace>
# replacement
</oracle_replace>
<naive_fix_description>describes the WRONG fix a shallow solver would
attempt (this becomes a teaching signal during training)
</naive_fix_description>
<hint_body>one paragraph hint that NAMES the relevant function/class but
not the fix itself
</hint_body>
</scenario>
```

## Parser

Use the hybrid anchor parser. Pseudocode:

```python
def parse_scenario(raw: str):
    body = extract_between("<scenario>", "</scenario>", raw, default=raw)
    obj = {}
    cursor = 0
    for field in ORDERED_FIELDS:
        close_pos = find("</{field}>", body[cursor:])
        next_open_pos = first_of(["<{g}>" for g in REMAINING_FIELDS], body[cursor:])
        if close_pos is None and next_open_pos is None: continue
        end = min(close_pos, next_open_pos)  # treating None as +inf
        open_pos = find("<{field}>", body[cursor:end])
        start = open_pos.end if found else cursor
        value = body[start:end]
        # strip ```python fences if present
        obj[field] = value
        cursor = end + (len("</{field}>") if close_pos == end else 0)
    return obj if all_required_fields_present(obj) else None
```

Long XML outputs (>20k chars) often have:
- Spurious `</module_code>` tags emitted inside `test_code`
- Missing `<test_code>` opening tag (model jumps straight from filename to code)
- Tags appearing out of order

The hybrid parser tolerates all of these by treating `</close>` as the primary
anchor and the next field's `<open>` as a fallback.

## Mutation strategies (when smoke rate > 0.5)

Cycle through these strategies on each mutation round. The goal is to KEEP
THE BUG and DISGUISE THE SURFACE CODE so the model can't pattern-match.

Strategy A — **call graph extension**:
  (a) Add 2 more public methods that depend on the buggy helper.
  (b) Add a misleading comment on the buggy line ("INVARIANT: correct").
  (c) Inline-expand the middle-layer helper into its callers (extends the
      chain by 1 hop).
  (d) Rename the buggy helper to something generic (`_fn_a`).

Strategy B — **decoy + misdirection**:
  (a) Add a decoy helper with a similar name (e.g., `_normalize` vs
      `_normalize_value`) that is correct and called from a different path.
  (b) Insert a long misleading docstring on the high-level method pointing
      to a wrong subsystem as the cause.
  (c) Add 4 identical-looking lines elsewhere that look like the buggy line
      but are correct.
  (d) Rename the buggy variable to a name shared with an unrelated parameter.

Strategy C — **class hierarchy wrap**:
  (a) Wrap the call chain in a class hierarchy (Base + Mixin + Subclass);
      the bug lives in a subclass override.
  (b) Add `@cached_property` on the high-level method (changes apparent
      control flow).
  (c) Add 2 sibling subclasses with their own (correct) helpers — they
      should not be touched.
  (d) Update pr_description to point at the caching layer (misdirection).

Iterate through A, B, C, A, B for the 5 mutation rounds. After each
mutation, re-run self-test (in case the mutation broke the oracle); if
self-test fails, the mutation broke the scenario — drop it.

## Synthetic engineering domains (no library leakage)

Pick from these for variety. Each scenario should use a different domain.

PID autotuner, smelter draft regulator, particle tracker, chemical reactor,
voice allocator, satellite pass scheduler, RFID inventory drift, subway berth
alignment, telemetry budget allocator, sensor data aggregator, warehouse robot
router, network topology cache, elevator group dispatcher, gantry crane sway
damper, vibration spectrum classifier, magnetic-field declination corrector,
fuel cell purge controller, hydroponics nutrient ladder, tunnel ventilation
regulator, wind shear gust predictor.

## Output

Save accepted scenarios to:
{OUTPUT_DIR}/scenarios_parsed.json   (the canonical list)
{OUTPUT_DIR}/scenarios_raw.jsonl     (every model emission, for debugging)
{OUTPUT_DIR}/revision_log.jsonl       (per-scenario event log: parse, self-test,
                                        smoke rates, mutations, final outcome)
```

---

## Step 2: Self-test each generated scenario

For each candidate scenario, before spending compute on smoke testing,
verify it's internally consistent. The self-test catches ~30-40% of
generation failures cheaply.

```python
def self_test(sc) -> tuple[bool, str]:
    # 1. Leakage check
    for term in LEAK_TERMS:
        if term in sc["module_code"] + sc["test_code"] + sc["pr_description"]:
            return False, f"leakage:contains {term!r}"

    # 2. Pre-fix: target test FAILS, regression guards PASS
    wd = mkdtemp()
    write(wd / sc["module_filename"], sc["module_code"])
    write(wd / sc["test_filename"], sc["test_code"])
    pre = run_pytest(wd)
    if not pre:
        return False, "no_test_results_pre"
    if pre.get(sc["target_test"]) != "FAILED":
        return False, f"target_not_failing_pre:{pre.get(sc['target_test'])}"

    # 3. Apply oracle: oracle_search must exist verbatim
    code = (wd / sc["module_filename"]).read_text()
    if sc["oracle_search"] not in code:
        return False, "oracle_search_not_in_module"
    code = code.replace(sc["oracle_search"], sc["oracle_replace"], 1)
    (wd / sc["module_filename"]).write_text(code)

    # 4. Post-fix: target test PASSES, no regression broken
    post = run_pytest(wd)
    if post.get(sc["target_test"]) != "PASSED":
        return False, f"target_not_passing_post:{post.get(sc['target_test'])}"
    pre_passing = {t for t, s in pre.items() if s == "PASSED"}
    broken = [t for t in pre_passing if post.get(t) != "PASSED"]
    if broken:
        return False, f"broken_tests:{broken[:3]}"

    return True, "ok"
```

`run_pytest` should use a forgiving regex that handles class-based tests:
`r"(\S+::\S+)\s+(PASSED|FAILED|ERROR)"` and then `.split("::")[-1]` to extract
the bare method name. The strict `^test_[\w]+\s+(PASSED|FAILED)` form breaks
on `tests/test_x.py::TestClass::test_method` paths.

---

## Step 3: Smoke test (measure base-model success rate)

Once a scenario passes self-test, measure how often the base model can fix
the bug from scratch. The accept criterion is `rate ≤ 0.5` (model fails the
target test on ≥ 50% of rollouts).

```python
REPAIR_SYS = """
You are an expert software engineer. Fix the bug by emitting SEARCH/REPLACE
blocks. Format:

```python
### path/to/file.py
<<<<<<< SEARCH
(exact code)
=======
(replacement)
>>>>>>> REPLACE
```
"""

def rollout(sc):
    user = f"{sc['pr_description']}\n\nFile `{sc['module_filename']}`:\n```python\n{sc['module_code']}\n```\n\nWrite SEARCH/REPLACE block(s) to fix the bug."
    return vllm_chat(REPAIR_SYS, user, temperature=1.0, top_p=0.95, max_tokens=4096, enable_thinking=False)

def rollout_passes(sc, raw):
    blocks = parse_search_replace(raw)  # uses SR_RE
    if not blocks: return False
    wd = mkdtemp()
    write_files(wd, sc)
    code = (wd / sc["module_filename"]).read_text()
    for search, replace in blocks:
        if search in code:
            code = code.replace(search, replace, 1)
    (wd / sc["module_filename"]).write_text(code)
    return run_pytest(wd).get(sc["target_test"]) == "PASSED"

def smoke(sc, n=10):
    with ThreadPoolExecutor(max_workers=n) as ex:
        rollouts = [f.result() for f in [ex.submit(rollout, sc) for _ in range(n)]]
    successes = sum(1 for r in rollouts if rollout_passes(sc, r))
    return successes / n, next((r for r in rollouts if rollout_passes(sc, r)), None)
```

If the smoke rate is > 0.5, send the scenario back through `revise_for_mutation`
with the next mutation strategy. The mutation prompt includes the successful
rollout from smoke (so the model knows what the easy fix looks like and can
disguise the code accordingly).

After mutation, re-run self-test (mutations often break the oracle), then
re-smoke. Up to 5 mutation rounds.

---

## Step 4: Acceptance and persistence

Once a scenario hits `rate ≤ 0.5`:
- Append it to `{OUTPUT_DIR}/scenarios_parsed.json`.
- Append an event to `{OUTPUT_DIR}/revision_log.jsonl` with the per-mutation
  rate trajectory.

When you have 10 accepted scenarios for this capability:
- Mark the capability `status: "DONE"` in `{CAPABILITIES_FILE}`.
- Write a short summary at the top of `{OUTPUT_DIR}/<capability>/README.md`
  with: capability name, 10 scenario names, per-scenario base-model rates,
  total attempts spent.
- **Stop.** Do not move on to the next capability — the user re-invokes.

---

## Common failure modes (from prior runs)

When you encounter these, do NOT keep retrying without a fix:

1. **`target_not_failing_pre`** — the scenario's target test passes on the
   unmodified module. The bug isn't actually exercised. Almost always means
   the model invented a tangential edge-case test that doesn't exercise the
   real bug. **Fix:** ask the model to make the target test exercise the
   specific code path the bug is in (use `revise_for_selftest` with this
   reason).

2. **`target_not_passing_post:FAILED`** — applying the oracle doesn't fix
   the target test. Either the oracle is wrong, or the bug is in a different
   location than the oracle targets. **Fix:** revise with a stronger
   instruction that the oracle must be the EXACT minimal change that makes
   the target test go from FAIL → PASS.

3. **`leakage:contains 'numpy.'`** — the model snuck in an open-source
   library identifier. Reject; do not try to revise (the model will keep
   sneaking them in). Move on to the next domain.

4. **Saturated smoke (rate = 1.0)** — base model solves it trivially. Use
   the mutation strategies. If after 5 mutations the rate is still ≥ 0.5,
   abandon and move on. Common cause: the bug is described too literally in
   the PR description (e.g., "the formula should be `4a / πd`"). Mutations
   that disguise the surface help, but a sufficiently leaky PR description
   can't be saved.

5. **All-zero smoke (rate = 0.0)** — model never succeeds. This is actually
   GREAT for training (high room for improvement), but you need to verify
   at least the oracle works (self-test passed) and the test setup is sane.
   If self-test passed, ACCEPT — rate=0.0 means the capability gap is real.

6. **Mutation breaks oracle** — after applying a mutation, self-test fails
   on `oracle_search_not_in_module` or `target_not_passing_post`. Common
   cause: the mutation strategy renamed a variable that was inside
   `oracle_search`. **Fix:** instruct the mutation to update both the
   module AND `oracle_search` consistently, or just drop and re-try with
   the next mutation strategy.

7. **Parse failures at long output** — the model's XML has missing or
   duplicated tags. Use the hybrid anchor parser (Step 1 §Parser). If it
   STILL fails, the model likely emitted only the first half of the scenario
   before running out of `max_tokens=12000` budget. Bump the cap to 16000
   only as a last resort (slower per call).

---

## Pipeline Summary

```
{CAPABILITIES_FILE} (PENDING capabilities sorted by mean_delta desc)
       │
       ▼
┌─────────────────────────────────────────────────┐
│  Step 1: Generate scenario (vLLM synthesis)     │  per-attempt loop:
│  Step 2: Self-test                              │   generate → parse →
│  Step 3: Smoke test (N=10 rollouts at T=1.0)    │   self-test → smoke →
│  Step 4: Mutate if rate > 0.5 (up to 5 rounds)  │   accept or mutate
└────────────────────────┬────────────────────────┘
                         │
                         ▼ (when 10 accepted for this capability)
       Mark capability status="DONE"
       Write {OUTPUT_DIR}/<capability>/scenarios_parsed.json
       STOP. User re-invokes for the next capability.
```

---

## Reference Files

- **`pipeline/swebench/scenarios_parsed.json`** — accepted scenarios for the
  capability you just trained an env for.
- **`pipeline/swebench/revision_log.jsonl`** — per-scenario event log (parse,
  self-test, smoke, mutation history).
- **`train/`** — GRPO training scripts. Once you have scenarios, you load
  them in a `capability_<name>_game.py` (see general doc) and train the LoRA.
- **`moe_gate/`** — the gate that routes inference at deployment time across
  the LoRA adapters you train per capability.
