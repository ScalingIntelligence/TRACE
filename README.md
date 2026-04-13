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

TRACE is designed to be driven by an LLM coding agent (Claude Code, Codex, etc.). You write a short YAML config, render it into step-by-step instructions, and hand those instructions to the agent.

### Quick Start

**1. Create a config file** with your model and evaluation results:

```yaml
# pipeline/my_config.yaml
run_name: my_experiment
model: Qwen/Qwen3-30B-A3B-Instruct-2507

capability_selection:
  eval_results:
    - results/eval_baseline.json
```

All other parameters have sensible defaults (see [`pipeline/trace_config.yaml`](pipeline/trace_config.yaml) for the full list).

**2. Render the pipeline documents:**

```bash
python pipeline/render_pipeline.py pipeline/my_config.yaml
```

This produces two files in `pipeline/`:
- `my_experiment_capability_selection.md` — instructions for Step 1
- `my_experiment_environment_generation.md` — instructions for Step 2

**3. Hand each file to your coding agent** (Claude Code, Codex, etc.) to execute.

### Step 1: Capability Selection

Hand the rendered `*_capability_selection.md` to your coding agent. It will:
1. **Phase 1 (Discovery):** Read trajectories and propose candidate capabilities → `pipeline/candidate_capabilities.json`
2. **Phase 2 (Labeling):** Run N independent labeling passes → `pipeline/run_01.json` ... `pipeline/run_10.json`
3. **Phase 3 (Aggregation):** Run the filtering script → `pipeline/selected_capabilities.json`

### Step 2: Synthetic Environment Generation

Hand the rendered `*_environment_generation.md` to your coding agent. **Each invocation processes one capability** (the highest-priority PENDING one). Re-invoke to process the next. For each capability, the agent will:
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
│   ├── trace_config.yaml                  # YAML config template (copy & edit)
│   ├── render_pipeline.py                 # Renders templates from YAML config
│   ├── trace_capability_selection.md      # Step 1 template
│   ├── trace_environment_generation.md    # Step 2 template
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
