# SWE-bench pipeline scripts (Qwen3.6-27B reference run)

This directory contains the **concrete Python scripts** for the reference TRACE
run on Qwen3.6-27B / SWE-bench Verified.

The companion **prompts** that drive an LLM agent through the same pipeline
without these scripts are in
[`../../prompts/swebench/`](../../prompts/swebench/). Use the prompts if you
want to adapt the pipeline to a new benchmark; use these scripts if you want
to reproduce the SWE-bench numbers exactly.

---

## Files at a glance

| File | What |
|---|---|
| `launch_phase1.sh` | Sets env vars + launches phase 1 against a running vLLM. |
| `phase1_baseline_eval.py` | Runs `mini-swe-agent` over 500 SWE-bench Verified instances, then grades on Modal. Output goes to `runs/baseline/`. |
| `phase2_capability_mining.py` | 3-stage capability discovery + labeling + clustering on the unresolved trajectories from phase 1. Output: `selected_capabilities.json`. |
| `phase3_synth_semantic_logic_precision.py` | Synthetic-env generator for the **largest** discovered capability (61% of unresolved). v4 mutation strategies (rename / inline / decoy / docstring misdirection). |
| `phase3_synth_deep_call_graph_traversal.py` | Synthetic-env generator for the **second** capability (16% of unresolved). v6 strategies (call-graph extension / decoy helpers / class-hierarchy wrap). |
| `phase4_env_validation.py` | Self-test + smoke + acceptance filter for the generated scenarios (validates target test fails pre-fix and passes after oracle). |
| `phase5_training.py` | Pod orchestration sketch for GRPO training on accepted scenarios; see caveat below. |
| `train_grpo_trace_v4.py` | Trainer entry-point intended to import the capability game before calling the GRPO trainer. |
| `trace_v4_scenarios.py` | Loader for `scenarios_parsed.json`; exposes a `Scenario` dataclass. |
| `capability_semantic_logic_precision_game.py` | The game class registered with the trainer. Implements the rollout / reward interface. |
| `phase6_final_eval.py` | Re-runs `mini-swe-agent` on 500 instances with the trained LoRA hot-loaded into vLLM; compares to baseline. |
| `eval_iter_on_8gpu.sh` | Helper that boots vLLM (TP=4 × DP=2) on an 8-GPU node and grades a checkpoint. |
| `configs/swebench_qwen36_vllm.yaml` | The mini-swe-agent config — agent step limits, vLLM API base, sampling params. |

---

## Reproducing the reference run

Short version:

```bash
# Set up
export TRACE_PIPE_ROOT=/workspace/trace_pipeline
mkdir -p "$TRACE_PIPE_ROOT"/{runs,env,logs,training}

# Phase 1: baseline (24h, ~$150 Modal). Assumes vLLM is already serving.
bash pipeline/swebench/launch_phase1.sh
# Output:
#   $TRACE_PIPE_ROOT/runs/baseline/preds.jsonl
#   $TRACE_PIPE_ROOT/runs/baseline/modal-reports/

# Phase 2: capability mining (~30 min, uses an LLM to label trajectories)
python pipeline/swebench/phase2_capability_mining.py

# Phase 3: env synthesis — one script per capability target
#   ~100 min each, 4 vLLM calls / scenario, accept threshold rate <= 0.5
python pipeline/swebench/phase3_synth_semantic_logic_precision.py
python pipeline/swebench/phase3_synth_deep_call_graph_traversal.py

# Phase 4: validate the generated scenarios actually work (self-test + smoke)
python pipeline/swebench/phase4_env_validation.py

# Phase 5: GRPO training (~3.5h on 4×H100, TP=4 DDP)
python pipeline/swebench/phase5_training.py

# Phase 6: re-eval with the trained LoRA (~14h, $150 Modal)
python pipeline/swebench/phase6_final_eval.py --checkpoint iter_130
```

---

## Environment expectations

These scripts assume:

- **vLLM 0.22+** is running on `http://localhost:8000` with the args in
  `configs/swebench_qwen36_vllm.yaml`. The Qwen3.6-27B model needs
  `--tensor-parallel-size 4` to fit at 128k context with KV-cache headroom.
- `HF_TOKEN` is set (downloads the base model + uploads LoRA checkpoints).
- `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` are set (mini-swe-agent dispatches
  sandboxes to Modal, and the grader runs on Modal too).
- `mini-swe-agent` v2.3+ is installed (`pip install 'mini-swe-agent[swebench]'`)
  along with `swebench>=4.1`.

The training scripts (`phase5_training.py`, `train_grpo_trace_v4.py`)
additionally require:

- The `train/` package from this repo's root (importable as `from train.*`).
- 4× H100 (or equivalent) for TP=4 DDP — the LoRA adapter is rank 16, base
  weights are 4-bit quantized via `bitsandbytes`.

### Phase 5 caveat

`phase5_training.py` is not a turnkey public trainer launcher yet. It was written
for an internal OpenShift pod layout and currently emits a `TRAINER_LAUNCH_STUB`.
To make Phase 5 fully reproducible in this repo, the release needs:

- the shared `capability_regression_safe_edit_game.py` helper imported by
  `capability_semantic_logic_precision_game.py`;
- a generated accepted-scenario artifact (`scenarios_v4_all.json`) from Phase
  3/4, provided through `TRACE_V4_SCENARIOS_PATH` or copied to
  `/workspace/trace_pipeline/env/scenarios_v4_all.json`;
- a verified wrapper command that imports `train.train_grpo` from this repo and
  runs with the repository root plus this SWE-bench directory on `PYTHONPATH`;
- a local or documented cluster launch command that copies the repo, scenarios,
  and environment variables together before running `torchrun`.

Until those pieces are restored, use Phases 1-4 as the environment-generation
reference and treat Phase 5 as a description of the internal training handoff.

---

## How these relate to the prompts in `prompts/swebench/`

The prompts in `prompts/swebench/` describe the **same procedure** as a
markdown document an LLM coding agent can execute — which means the agent
will write something semantically equivalent to these scripts on the fly,
using vLLM + Python + pytest under the hood.

Use the scripts when:
- You want exact reproduction of the published numbers.
- You're running unattended (no agent in the loop).
- You need to extend a specific phase (e.g., add a new mutation strategy
  to phase 3).

Use the prompts when:
- You're adapting TRACE to a new benchmark (not SWE-bench).
- You want the agent to use new judgment in places where the script makes
  hard-coded choices (e.g., picking the next domain in phase 3).
- You're running interactively and want to see the agent's reasoning.

Both paths produce the same artifacts (`selected_capabilities.json`,
`scenarios_parsed.json`, per-capability LoRA checkpoints).
