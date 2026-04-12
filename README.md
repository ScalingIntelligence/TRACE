# TRACE: Capability-Targeted Agentic Training

[![Paper](https://img.shields.io/badge/Paper-ArXiv-red.svg)](#)
[![Status](https://img.shields.io/badge/Status-Code_Coming_Soon-blue.svg)](#)

This repository contains the official implementation for **TRACE (Turning Recurrent Agent failures into Capability-targeted training Environments)**. TRACE is an end-to-end system for environment-specific agent self-improvement. 

Large Language Models (LLMs) deployed in complex agentic environments often fail because they lack specific, underlying capabilities. Existing approaches typically rely on generic synthetic data or direct reinforcement learning (RL) on the target environment, which forces the model to learn these capabilities implicitly and inefficiently. TRACE solves this by automatically identifying the exact capabilities an agent lacks, synthesizing targeted training environments to isolate and train those capabilities, and routing to the appropriate adapter at inference time.

## How TRACE Works

The TRACE pipeline consists of four automated steps:

1. **Capability Selection:** An analysis agent contrasts successful and failed trajectories from the base model in the target environment. It identifies and ranks the specific capabilities that meaningfully distinguish successes from failures.
2. **Synthetic Environment Generation:** For each identified deficit, a generation agent constructs a capability-targeted synthetic training environment. This environment preserves the target environment's interface (tool schemas, interaction protocols) while isolating the missing capability for verifiable training.
3. **GRPO Training:** We train a separate Low-Rank Adaptation (LoRA) module for each capability-specific synthetic environment using Group Relative Policy Optimization (GRPO).
4. **Select & Adapt:** At inference, the base model identifies the most relevant capability for the task given natural language descriptions of each capability, and the corresponding LoRA adapter is activated for generation. 

## Key Results

TRACE demonstrates significant improvements and generalization across different complex environments:

* **$\tau^{2}$-Bench (Customer Service):** Improves upon the base agent by +14.1 points, achieving an overall pass rate of 47.0%. It scales more efficiently than baselines, outperforming GRPO and GEPA by +9.2 and +7.4 points given the same rollout budget.
* **ToolSandBox (Tool Use):** Achieves a mean similarity score of 0.552, improving over the base model by +0.141 points and +7 perfect scores.

---

## Getting Started

### Prerequisites

- Python 3.11+
- CUDA-capable GPUs (1 GPU for vLLM inference server, additional GPUs for training)
- [conda](https://docs.conda.io/en/latest/) or equivalent environment manager

### Installation

```bash
conda create -n trace python=3.11 -y
conda activate trace
pip install -r requirements.txt
```

---

## Running the TRACE Pipeline

TRACE is designed to be driven by an LLM coding agent (Claude Code, Codex, etc.). Each pipeline step is defined in a markdown file under `pipeline/` that you fill in with your configuration and then hand to the agent as instructions.

### Step 1: Capability Selection

Identify which capabilities your model lacks by analyzing its evaluation trajectories.

**Template:** [`pipeline/trace_capability_selection.md`](pipeline/trace_capability_selection.md)  
**Example (pre-filled):** [`pipeline/test_capability_selection.md`](pipeline/test_capability_selection.md)

Open the template and fill in the placeholders at the top:

| Placeholder | Description | Example |
|---|---|---|
| `{EVAL_RESULTS}` | Path to your evaluation results file(s) containing pass/fail trajectories | `results/eval_baseline.json` |
| `{MODEL_NAME}` | Name of the model being evaluated (for record-keeping) | `Qwen/Qwen3-30B-A3B-Instruct-2507` |
| `{N_RUNS}` | Number of independent labeling runs (default: `10`) | `10` |
| `{N_CANDIDATES}` | Max candidate capabilities to propose (default: `10`) | `10` |
| `{RHO}` | Coverage threshold (default: `0.10`) | `0.10` |
| `{DELTA}` | Contrastive gap threshold (default: `0.20`) | `0.20` |
| `{K_CONSISTENCY}` | Cross-run consistency threshold (default: `8`) | `8` |
| `{OUTPUT_DIR}` | Output directory (default: `pipeline/`) | `pipeline/` |

Then hand the filled-in document to your coding agent. It will:
1. **Phase 1 (Discovery):** Read trajectories and propose candidate capabilities → `pipeline/candidate_capabilities.json`
2. **Phase 2 (Labeling):** Run `{N_RUNS}` independent labeling passes → `pipeline/run_01.json` ... `pipeline/run_10.json`
3. **Phase 3 (Aggregation):** Run the filtering script → `pipeline/selected_capabilities.json`

### Step 2: Synthetic Environment Generation

Generate a targeted training environment for each identified capability deficit.

**Template:** [`pipeline/trace_environment_generation.md`](pipeline/trace_environment_generation.md)  
**Example (pre-filled):** [`pipeline/test_environment_generation.md`](pipeline/test_environment_generation.md)

Open the template and fill in the placeholders:

| Placeholder | Description | Example |
|---|---|---|
| `{MODEL}` | HuggingFace model ID to train / collect rollouts with | `Qwen/Qwen3-30B-A3B-Instruct-2507` |
| `{CAPABILITIES_FILE}` | Path to selected capabilities (default: `pipeline/selected_capabilities.json`) | `pipeline/selected_capabilities.json` |
| `{GPU_DEVICE}` | GPU index for vLLM server (auto-detected if omitted) | `2` |
| `{PORT}` | Port for the vLLM server (default: `5050`) | `5050` |
| `{GROUP_SIZE}` | Rollouts per seed for GRPO (default: `16`) | `16` |
| `{NUM_SEEDS}` | Number of task seeds to collect (default: `100`) | `100` |
| `{HINT_RATIO}` | Fraction of rollouts with hint injection (default: `0.25-0.5`) | `0.25` |

Then hand the document to your coding agent. **Each invocation processes one capability** (the highest-priority PENDING one). Re-invoke to process the next. For each capability, the agent will:
1. Generate a synthetic environment file (e.g., `capability_<name>_game.py`)
2. Launch a vLLM server and collect validation rollouts
3. Validate the reward distribution (target: 20-60% success rate)
4. Mark the capability as DONE in `selected_capabilities.json`

### Step 3: GRPO Training

Train a LoRA adapter for each generated environment.

#### 3a. Launch the vLLM Inference Server

The training loop collects rollouts from a running vLLM server. Start it on a dedicated GPU:

```bash
conda activate trace

export CUDA_VISIBLE_DEVICES=0        # GPU for inference
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

vllm serve <YOUR_MODEL> \
  --host 0.0.0.0 \
  --port 8080 \
  --dtype bfloat16 \
  --max-model-len 32000 \
  --enable-lora \
  --max-loras 2 \
  --gpu-memory-utilization 0.85 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

Replace `<YOUR_MODEL>` with your HuggingFace model ID (e.g., `Qwen/Qwen3-30B-A3B-Instruct-2507`).

#### 3b. Run GRPO Training

In a separate terminal, launch training on the remaining GPUs:

```bash
conda activate trace

export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7   # GPUs for training
export VLLM_BASE_URLS=http://localhost:8080
export VLLM_MODEL=<YOUR_MODEL>
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

torchrun --nproc_per_node=<N_GPUS> --master-port=29501 -m train \
    --game capability_<YOUR_CAPABILITY_NAME> \
    --model <YOUR_MODEL> \
    --group-size 8 \
    --groups-per-batch 8
```

| Argument | Description |
|---|---|
| `--game` | Registered game name (e.g., `capability_multi_step_transaction_completion`) |
| `--model` | HuggingFace model ID (must match what vLLM is serving) |
| `--group-size` | Number of rollouts per seed in each GRPO group |
| `--groups-per-batch` | Number of groups per training batch |
| `--resume` | Path to a checkpoint directory to resume from (optional) |

Repeat Steps 3a-3b for each capability-specific environment to produce a set of LoRA adapters.

---

## Project Structure

```
├── pipeline/
│   ├── trace_capability_selection.md      # Step 1 template
│   ├── trace_environment_generation.md    # Step 2 template
│   ├── test_capability_selection.md       # Pre-filled example (Step 1)
│   ├── test_environment_generation.md     # Pre-filled example (Step 2)
│   ├── aggregate_capabilities.py          # Aggregation script for Phase 3
│   ├── candidate_capabilities.json        # Phase 1 output
│   ├── selected_capabilities.json         # Phase 3 output
│   └── run_*.json                         # Phase 2 labeling outputs
├── train/
│   ├── __main__.py                        # Training entry point
│   ├── config.py                          # Hyperparameters (LoRA rank, LR, etc.)
│   ├── train_grpo.py                      # GRPO training loop
│   ├── collect_rollouts.py                # Rollout collection against vLLM
│   ├── inference.py                       # vLLM client & prompt building
│   ├── model.py                           # Model loading with LoRA
│   └── ppo.py                             # GRPO loss computation
├── game_registry.py                       # Central game/environment registry
├── capability_*_game.py                   # Generated synthetic environments
├── requirements.txt                       # Python dependencies
└── gameplay_rollouts/                     # Training rollout logs
```

## Citation

If you find this work helpful in your research, please consider citing our paper:

```bibtex
@article{kang2026trace,
  title={TRACE: Capability-Targeted Agentic Training},
  author={Kang, Hangoo and Suresh, Tarun and Saad-Falcon, Jon and Mirhoseini, Azalia},
  journal={arXiv preprint},
  year={2026}
}
```
