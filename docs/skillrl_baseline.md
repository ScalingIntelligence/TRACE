# SkillRL Baseline: Step-by-Step Guide

SkillRL (Recursive Skill-Augmented RL) baseline for tau-bench evaluation.
Adapts the approach from [arXiv:2602.08234](https://arxiv.org/pdf/2602.08234) to our Qwen3-30B-A3B infrastructure.

## Overview

SkillRL has 4 stages:
0. **Data Generation** — Generate synthetic tasks + SFT training data
1. **Cold-Start SFT** — Teach the model to use skills from the SkillBank
2. **GRPO with Skills** — RL training with skill-augmented system prompts
3. **Evaluation** — Evaluate on tau-bench

## Prerequisites

- Qwen3-30B-A3B-Instruct-2507 downloaded
- Working directory: `cd ~/hangook/games`

---

## Stage 0: Generate Synthetic Tasks + SFT Data

We generate novel training tasks (NOT tau-bench eval tasks) using the same DB,
then run simulations with skills in the prompt to collect SFT training data.

### Step 0a: Generate synthetic task definitions (instant, no GPU)

```bash
cd ~/hangook/games
python skillrl/generate_synthetic_tasks.py --generate-tasks
```

Output: `skillrl/data/synthetic_tasks.json` (~225 tasks)

### Step 0b: Launch vLLM servers for simulation

```bash
# Terminal 1 (tmux vllm_0) — Agent server, GPU 0-1
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --tensor-parallel-size 2 \
    --port 8080 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 16000

# Terminal 2 (tmux vllm_1) — User simulator, GPU 2-3
CUDA_VISIBLE_DEVICES=2,3 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --tensor-parallel-size 2 \
    --port 9000 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 16000
```

### Step 0c: Run simulations to generate SFT data

```bash
cd ~/hangook/games
python skillrl/generate_synthetic_tasks.py --run-sims \
    --agent-url http://localhost:8080/v1 \
    --user-url http://localhost:9000/v1 \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --num-trials 5 \
    --temperature 0.7 \
    --output skillrl/data/skillrl_sft_train.jsonl
```

For more data, increase `--num-trials 10` (target ~1K samples).

Output: `skillrl/data/skillrl_sft_train.jsonl`

---

## Stage 1: Cold-Start SFT

Teaches the model how to use skills from the SkillBank.

```bash
cd ~/hangook/games

CUDA_VISIBLE_DEVICES=6,7 \
WANDB_API_KEY=f4ef099e7073d103963e5c986e4f818f5a526ee8 \
torchrun --nproc_per_node=2 train_sft_adp.py \
    --data skillrl/data/skillrl_sft_train.jsonl \
    --max-seq-length 8192 \
    --num-epochs 3 \
    --lr 1e-4 \
    --mini-batch-size 1 \
    --gradient-accumulation-steps 8 \
    --save-steps 50 \
    --log-every 1 \
    --output-dir /home/ubuntu/.cache/huggingface/skillrl/sft_lora \
    --run-name skillrl-coldstart-sft
```

**Output:** LoRA adapter at `/home/ubuntu/.cache/huggingface/skillrl/sft_lora/`

---

## Stage 2: GRPO Training with Skills

### Step 2a: Restart vLLM with SFT adapter

Kill the vLLM servers from Stage 0, then relaunch with LoRA:

```bash
# Terminal 1 (tmux vllm_0) — Agent + rollout server, GPU 0-1
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --tensor-parallel-size 2 \
    --port 8080 \
    --enable-lora \
    --max-loras 2 \
    --lora-modules skillrl_sft=/home/ubuntu/.cache/huggingface/skillrl/sft_lora/sft_ckpt_epoch_2_* \
    --gpu-memory-utilization 0.85 \
    --max-model-len 16000 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes

# Terminal 2 (tmux vllm_1) — User simulator, GPU 2-3
CUDA_VISIBLE_DEVICES=2,3 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --tensor-parallel-size 2 \
    --port 9000 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 16000
```

### Step 2b: Launch GRPO training

```bash
# Terminal 3 (tmux train) — GRPO training, GPU 4-7
cd ~/hangook/games

CUDA_VISIBLE_DEVICES=4,5,6,7 \
NCCL_P2P_DISABLE=1 \
WANDB_API_KEY=f4ef099e7073d103963e5c986e4f818f5a526ee8 \
VLLM_BASE_URLS=http://localhost:8080 \
VLLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 \
VLLM_TIMEOUT_S=2000 \
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
VLLM_RPC_TIMEOUT=2000 \
PYTHONUNBUFFERED=1 \
torchrun --nproc_per_node=4 train_skillrl_grpo.py \
    --game adversarial_policy_skillrl \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --compact-tools \
    --temperature-range "0.7,0.9,1.0" \
    --groups-per-batch 8 \
    --group-size 16 \
    --skill-bank-dir skillrl/data \
    --skill-evolution \
    --evolution-teacher-url http://localhost:8080/v1 \
    --evolution-teacher-model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --evolution-threshold 0.4 \
    --evolution-max-new 3
```

### Skill Evolution (automatic during training)

When enabled, after each tau2-bench evaluation:
1. Check if domain success rate < threshold (0.4)
2. Analyze failed trajectories
3. Call teacher model (Qwen3-30B via vLLM) to generate 1-3 new skills
4. Update SkillBank JSON files in `skillrl/data/`
5. New skills automatically used in subsequent rollouts

---

## Stage 3: Evaluation

After GRPO training, evaluate the final adapter on tau-bench.

### Step 3a: Load trained adapter

```bash
curl -X POST http://localhost:8080/v1/load_lora_adapter \
    -H "Content-Type: application/json" \
    -d '{"lora_name": "skillrl_grpo", "lora_path": "/home/ubuntu/.cache/huggingface/adversarial_policy_skillrl/ppo_ckpt_latest"}'
```

### Step 3b: Run evaluation

```bash
cd ~/hangook/games/evals/benchmarks/tau2_bench_eval

# Airline
python main.py \
    --domain airline \
    --agent-llm vllm://skillrl_grpo \
    --user-llm vllm://Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --agent-api-base-urls http://localhost:8080/v1 \
    --user-api-base-urls http://localhost:9000/v1 \
    --num-trials 1 \
    --save-to skillrl-grpo-airline

# Retail
python main.py \
    --domain retail \
    --agent-llm vllm://skillrl_grpo \
    --user-llm vllm://Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --agent-api-base-urls http://localhost:8080/v1 \
    --user-api-base-urls http://localhost:9000/v1 \
    --num-trials 1 \
    --save-to skillrl-grpo-retail
```

---

## GPU Allocation Summary

| GPUs | Stage 0 | Stage 1 | Stage 2 | Stage 3 |
|------|---------|---------|---------|---------|
| 0-1  | vLLM agent (8080) | free | vLLM agent+LoRA (8080) | vLLM agent+LoRA (8080) |
| 2-3  | vLLM user (9000) | free | vLLM user (9000) | vLLM user (9000) |
| 4-5  | free | free | GRPO training | free |
| 6-7  | free | SFT training | GRPO training | free |

---

## File Structure

```
skillrl/
├── __init__.py                    # Package init
├── skillbank.py                   # SkillBank class: load, retrieve, format
├── skill_evolution.py             # Recursive evolution via vLLM teacher
├── prompts.py                     # Prompt templates
├── generate_synthetic_tasks.py    # Synthetic task + SFT data generation
└── data/
    ├── airline_skills.json        # 10 general + 18 category skills
    ├── retail_skills.json         # 8 general + 18 category skills
    ├── synthetic_tasks.json       # Generated synthetic tasks (Stage 0a)
    └── skillrl_sft_train.jsonl    # Generated SFT data (Stage 0c)

train_skillrl_grpo.py              # GRPO wrapper with skill injection + evolution
train_sft_adp.py                   # SFT trainer (reused for Stage 1)
```

---

## Matching Paper Parameters

| Paper Parameter | Our Setting | Notes |
|----------------|-------------|-------|
| Base model | Qwen3-30B-A3B-Instruct | Paper uses Qwen2.5-7B |
| SFT LR | 1e-4 | Same as paper |
| SFT epochs | 3 | Same as paper |
| SFT data | Synthetic tasks (not eval set) | Paper uses env trajectories |
| RL LR | 1e-6 | Same as paper |
| Group size | 16 | Paper uses 8 |
| KL coef | 0.01 | Same as paper |
| Top-K skills | 6 | Same as paper |
| Evolution threshold | 0.4 | Same as paper |
| Max new skills/step | 3 | Same as paper |
| Teacher model | Qwen3-30B (self) | Paper uses OpenAI o3 |
| Target benchmark | tau-bench | Paper uses ALFWorld/WebShop |
