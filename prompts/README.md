# TRACE Prompt Templates

This directory holds the prompts the TRACE pipeline gives to an LLM agent
(Claude, Codex, etc.) at each stage. The prompts are organized by **target
benchmark** so adding a new benchmark only requires writing one new folder.

```
prompts/
├── general/        Env-agnostic templates — work with any benchmark that
│                   provides pass/fail trajectories. {PLACEHOLDERS} are
│                   filled in by render_pipeline.py from a YAML config.
├── tau-bench/      Filled-in templates for tau2-bench (airline + retail).
│                   Same structure as `general/`, with values inlined.
└── swebench/       Filled-in templates for SWE-bench Verified.
                    Same structure as `general/`, with SWE-bench-specific
                    notes about trajectory format, hybrid XML parser for
                    long outputs, and capability mining over per-instance
                    grader reports.
```

## Each subdirectory contains exactly two files

- `capability_selection.md` — the prompt + procedure for **Step 1** of TRACE:
  discover candidate capabilities, run 10 parallel labeling subagents, and
  aggregate with the Cov / Δ dual-threshold filter.
- `environment_generation.md` — the prompt + procedure for **Step 2**: take
  one selected capability and synthesize a training environment for it
  (game class for tau-bench-style; scenario JSON + pytest harness for
  SWE-bench-style).

Both files in `general/` are templates with `{PLACEHOLDERS}` that get
rendered into per-experiment files by `render_pipeline.py`:

```bash
python render_pipeline.py configs/capability_selection.yaml --stage capability
python render_pipeline.py configs/environment_generation.yaml --stage environment
```

The benchmark-specific subdirectories (`tau-bench/`, `swebench/`) hold
**already-rendered, reviewed** versions you can cite directly in a PR or
paper.

## Adding a new benchmark

1. Decide what's different about the benchmark's trajectory format, reward
   structure, and scenario shape. Most of the TRACE pipeline is benchmark-
   agnostic — usually only ~20-30% of the prompt text changes.
2. Copy `prompts/general/capability_selection.md` and
   `prompts/general/environment_generation.md` into
   `prompts/<your-benchmark>/`.
3. Replace the env-agnostic descriptions with benchmark-specific ones:
   - Trajectory format (paths, JSON schema)
   - What "pass" and "fail" mean for this benchmark
   - Scenario / environment shape
   - Smoke test / acceptance criteria
4. Add a config under `configs/` if you want `render_pipeline.py` to
   produce a rendered version.
