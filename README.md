# TRACE: Capability-Targeted Agentic Training

[![Paper](https://img.shields.io/badge/Paper-ArXiv-red.svg)](https://arxiv.org/abs/2604.05336)
[![Blog](https://img.shields.io/badge/Blog-Post-blue.svg)](https://scalingintelligence.stanford.edu/blogs/trace/)

This repository contains the official implementation for **TRACE — Turning Recurrent
Agent failures into Capability-targeted training Environments**. TRACE is an
end-to-end system for environment-specific agent self-improvement.

Large Language Models deployed as agents in complex environments often fail because
they lack specific, underlying capabilities. Existing approaches typically rely on
generic synthetic data or direct RL on the target environment, which forces the
model to learn the missing capabilities implicitly and inefficiently. TRACE solves
this by:

1. **Automatically identifying** the exact capabilities the agent lacks,
2. **Synthesizing** targeted training environments that isolate each missing capability,
3. **Training** a separate LoRA adapter per capability via GRPO, and
4. **Routing** at inference time to the right adapter for the task at hand.

---

## How TRACE works

The pipeline has four automated steps. Each step is driven by an LLM coding agent
(Claude Code, Codex, etc.) following a markdown prompt under `prompts/`.

| # | Step | Prompt | Output |
|---|---|---|---|
| 1 | **Capability Selection** | `prompts/<benchmark>/capability_selection.md` | `selected_capabilities.json` — ranked list of missing capabilities |
| 2 | **Environment Generation** | `prompts/<benchmark>/environment_generation.md` | one synthetic training env per capability |
| 3 | **GRPO Training** | `train/train_grpo.py` | one LoRA adapter per capability |
| 4 | **Select & Adapt** | `moe_gate/` | gate that routes inference to the right adapter |

---

## Reference results

TRACE has been validated on three benchmarks across two model scales:

| Benchmark | Base model | Baseline | + TRACE | Δ | Notes |
|---|---|---:|---:|---:|---|
| τ²-Bench (Airline + Retail) | Qwen3-30B-a3b-instruct | 32.9% | **48.3%** | **+15.4 pp** | from paper |
| **SWE-bench Verified** | **Qwen3.6-27B** | **68.0%** | **73.2%** | **+5.2 pp** | best iter checkpoint; see `swebench/qwen3.6_self_trace/` |

The Qwen3.6 SWE-bench reference run lives in `swebench/qwen3.6_self_trace/`,
including the phase scripts, generated-environment wrapper, and reproduction notes.
It is the recommended starting point for adapting TRACE to a new benchmark.

---

## Repository layout

```
TRACE/
├── prompts/                        Prompts for the LLM-driven pipeline stages.
│   ├── general/                    Env-agnostic templates (use as starting point
│   │                               when adapting TRACE to a new benchmark).
│   ├── tau-bench/                  Filled-in version for τ²-Bench.
│   └── swebench/                   Filled-in version for SWE-bench Verified.
│
├── pipeline/                       Pipeline-stage scripts (benchmark-agnostic).
│   ├── aggregate_capabilities.py   Aggregation step for capability selection.
│   ├── calibrate_environment.py    Rollout statistics and pass@k threshold check.
│   └── _summarize.py               Trajectory compaction for capability selection.
│
├── swebench/                       SWE-bench reference runs (one folder per run).
│   └── qwen3.6_self_trace/         End-to-end Qwen3.6-27B run (phase1-phase6).
│
├── train/                          GRPO training + gate training.
│   ├── train_grpo.py               Train one capability-specific LoRA.
│   ├── train_router_sft.py         SFT warm-start of the MoE gate.
│   ├── warmstart_gater.py          Gate-only warm-start with frozen base + LoRAs.
│   ├── grpo_gater.py               GRPO fine-tune of the MoE gate.
│   ├── collect_rollouts.py         Rollout collection against vLLM.
│   └── ...                         model.py, ppo.py, inference.py, config.py
│
├── moe_gate/                       Inference-time routing across capability LoRAs.
│   ├── README.md                   How the gate works (per-layer / global modes).
│   ├── gater.py                    LayerCapabilityGater / TrajectoryGlobalRouter.
│   ├── multi_lora.py               MultiLoRAGatedLinear (weighted-mix forward).
│   ├── model_builder.py            build_capability_model(...).
│   ├── hf_wrapper.py / freeze_utils.py / smoke_test.py
│
├── configs/                        YAML configs for render_pipeline.py.
│   ├── capability_selection.yaml
│   ├── environment_generation.yaml
│   └── moe_gate.yaml
│
├── render_pipeline.py              Renders prompt templates with config values.
├── game_registry.py                Game/environment registry.
└── requirements.txt
```

---

## Getting started

### Prerequisites

- Python 3.11+
- CUDA-capable GPUs:
  - At least **1 GPU** for the vLLM inference server during capability selection.
  - **4+ GPUs** recommended for GRPO training (TP=4 keeps Qwen3.6-27B comfortably
    in memory at 128k context).
- [conda](https://docs.conda.io/en/latest/) or equivalent.

### Install

```bash
conda create -n trace python=3.11 -y
conda activate trace
pip install -r requirements.txt
```

---

## Quick start

TRACE is driven by handing per-stage markdown prompts to an LLM coding agent
(Claude Code, Codex, etc.). For each stage you have two choices:

1. Use a **filled-in** prompt from `prompts/<benchmark>/` directly — fastest if
   your benchmark already has a folder.
2. Render the **template** in `prompts/general/` from a YAML config — best when
   adapting to a new benchmark.

### Path A — use a benchmark-specific prompt

Both `tau-bench` and `swebench` ship with already-rendered, reviewed prompts:

```bash
# Capability selection (Step 1)
cat prompts/swebench/capability_selection.md          # read it first
# then hand it to Claude Code / Codex / etc.

# Environment generation (Step 2)
cat prompts/swebench/environment_generation.md
```

### Path B — render from a YAML config

Edit one of `configs/*.yaml` with your model + eval results, then render:

```bash
python render_pipeline.py configs/capability_selection.yaml --stage capability
python render_pipeline.py configs/environment_generation.yaml --stage environment
```

This writes rendered prompts into `prompts/rendered/<run_name>_<stage>.md`. Hand
those to your coding agent.

---

## Step-by-step

### Step 1 — Capability Selection

Run the **discovery + labeling + aggregation** flow described in
`prompts/<benchmark>/capability_selection.md`. The agent will:

1. Read your eval results (pass/fail trajectories).
2. Phase 1: Propose 5-10 candidate capabilities.
3. Phase 2: Spawn 10 parallel labeling subagents, each independently labeling every
   (trajectory, capability) pair as `NA`, `PRESENT`, or `LACKING`.
4. Phase 3: Run `pipeline/aggregate_capabilities.py` to compute coverage `Cov(c)`
   and contrastive gap `Δ(c)` per run, apply the dual-threshold filter
   (`Cov ≥ ρ`, `Δ ≥ δ`) and cross-run consistency (`K of N` runs).

**Output:** `pipeline/selected_capabilities.json` — each entry has a name,
description, `mean_cov`, `mean_delta`, example failed cases, and `status:
"PENDING"`.

### Step 2 — Environment Generation

Run `prompts/<benchmark>/environment_generation.md` **once per capability**. Each
invocation picks the top PENDING capability (highest mean Δ) and:

1. Generates a synthetic environment file (a game class for tool-use benchmarks; a
   pytest-graded scenario JSON set for SWE-bench).
2. Self-tests the environment (asserts the target test fails pre-fix and passes
   after applying the oracle).
3. Launches a vLLM server and collects validation rollouts at temperature 1.0.
4. **Calibrates difficulty by mutation**: if the base model's success rate is
   too high, the agent applies surface-disguising mutations (rename identifiers,
   inline helpers, add decoy edit sites) and re-tests, up to 5 rounds.
5. Marks the capability as `DONE` and stops. The user re-invokes for the next one.

**Output (tool-use benchmarks):** `capability_<name>_game.py` at the project root.

**Output (SWE-bench):** `pipeline/swebench/<capability>/scenarios_parsed.json`
(10 calibrated scenarios per capability).

### Step 3 — GRPO Training

Train one LoRA adapter per capability.

**3a. Launch the vLLM inference server**:

```bash
conda activate trace
export CUDA_VISIBLE_DEVICES=0,1,2,3            # GPUs for inference
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

vllm serve <YOUR_MODEL> \
  --host 0.0.0.0 --port 8000 \
  --dtype bfloat16 \
  --max-model-len 65536 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.85 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --enable-lora \
  --max-loras 2 \
  --max-lora-rank 32 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes        # use 'qwen3_xml' for Qwen3.6
```

**3b. Run GRPO training** on the remaining GPUs:

```bash
conda activate trace
export CUDA_VISIBLE_DEVICES=4,5,6,7
export VLLM_BASE_URLS=http://localhost:8000
export VLLM_MODEL=<YOUR_MODEL>
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

torchrun --nproc_per_node=<N_GPUS> --master-port=29501 -m train \
    --game capability_<NAME> \
    --model <YOUR_MODEL>
```

Repeat for each capability. The output is one LoRA adapter per capability
(`pipeline/adapters/<capability>.safetensors`).

### Step 4 — Select & Adapt (MoE Gate)

Once you have N capability-specific LoRAs, train the routing gate that picks the
right one at inference time. See `moe_gate/README.md` for the architecture and
`configs/moe_gate.yaml` for hyperparameters.

```bash
# SFT warm-start the gate on labeled (prompt, capability) pairs
python -m train.train_router_sft \
    --config configs/moe_gate.yaml

# Optional: GRPO fine-tune the gate end-to-end against the synthetic envs
python -m train.grpo_gater \
    --config configs/moe_gate.yaml
```

**At inference**, `build_capability_model(...)` from `moe_gate/` loads the base
model + all capability LoRAs + the trained gate. The gate routes each forward
through the right adapter (or a soft mixture) per layer.

---

## Adapting TRACE to a new benchmark

1. **Start from `prompts/general/`.** Copy both files to
   `prompts/<your-benchmark>/` and replace the env-agnostic descriptions with
   benchmark-specific ones (trajectory format, what "pass" means, scenario shape).
   The `prompts/swebench/` folder is a worked example of doing this for a
   single-turn code-edit benchmark; `prompts/tau-bench/` does the same for
   multi-turn tool-use.
2. **Compact your trajectories.** `pipeline/_summarize.py` provides a generic
   compaction; extend it for your benchmark's trajectory schema if needed.
3. **Pick thresholds.** Defaults (`ρ = 0.10`, `δ = 0.20`, `K = 8 of 10`) come
   from the paper and work well in practice. Tune only if your benchmark has
   unusual structure (very few failures, very long trajectories, etc.).
4. **Run.** The rest of the pipeline (capability selection → environment
   generation → GRPO → routing) is benchmark-agnostic.

---

## Citation

If you find this work helpful in your research, please consider citing our paper:

```bibtex
@misc{kang2026tracecapabilitytargetedagentictraining,
      title={TRACE: Capability-Targeted Agentic Training},
      author={Hangoo Kang and Tarun Suresh and Jon Saad-Falcon and Azalia Mirhoseini},
      year={2026},
      eprint={2604.05336},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2604.05336},
}
```
