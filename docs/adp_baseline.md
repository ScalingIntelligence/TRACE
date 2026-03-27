# Agent Data Protocol (ADP) Baseline

**Paper**: "Agent Data Protocol: Unifying Datasets for Diverse, Effective Fine-tuning of LLM Agents"
**ArXiv**: https://arxiv.org/abs/2510.24702
**Venue**: ICLR 2026 (Oral)
**Authors**: Yueqi Song, Ketan Ramaneti, Zaid Sheikh, Ziru Chen, Boyu Gou, Tianbao Xie, Yiheng Xu, Danyang Zhang, Apurva Gandhi, Fan Yang, Joseph Liu, Tianyue Ou, Zhihao Yuan, Frank Xu, Shuyan Zhou, Xingyao Wang, Xiang Yue, Tao Yu, Huan Sun, Yu Su, Graham Neubig
**Affiliations**: Carnegie Mellon University, The Ohio State University, University of Hong Kong, Duke University, Fujitsu Research, All Hands AI
**GitHub**: https://github.com/neulab/agent-data-protocol
**Website**: https://agentdataprotocol.com
**Dataset**: https://huggingface.co/collections/neulab/agent-data-protocol

## What is ADP?

ADP is a lightweight standardized representation (Pydantic schemas) that unifies fragmented agent training datasets into a common format. It defines:

- **Actions**: APIAction (tool/function calls), CodeAction (code execution), MessageAction (text communication)
- **Observations**: TextObservation, WebObservation

The key idea: convert diverse datasets into ADP format once, then convert ADP into any agent framework's SFT format (OpenHands, SWE-Agent, AgentLab). This is a data standardization protocol, not a training framework.

## ADP Dataset V1

1.3M training trajectories unified from 13 existing datasets:

| Dataset | Domain | Count | Source |
|---------|--------|-------|--------|
| AgentInstruct | C/T/W | 1.9K | synthetic |
| Code-Feedback | C | 66.4K | manual |
| CodeActInstruct | C | 7.1K | synthetic |
| Go-Browse | W | 9.5K | rollout |
| Mind2Web | W | 1.0K | manual |
| Nebius SWE-Agent | S | 13.4K | rollout |
| NNetNav-live | W | 5.0K | rollout |
| NNetNav-wa | W | 4.2K | rollout |
| OpenHands-feedback | C/T/W | 0.2K | rollout |
| Orca AgentInstruct | T | 1046.1K | synthetic |
| SWE-Gym | S | 0.5K | rollout |
| SWE-smith | S | 5.0K | manual |
| Synatra | W | 99.9K | rollout |

Domains: C=Coding, S=Software Engineering, T=API/Tool Use, W=Web Browsing

### Data Sampling

Large datasets are downsampled, small ones upsampled. Key multipliers:
- orca_agentinstruct: w=0.001 (heavily downsampled)
- synatra: w=0.01 (downsampled)
- swe-gym: w=3 (upsampled)
- agenttuning_*: w=2 (upsampled)
- Most others: w=1

## Training Setup

- **Base models**: Qwen2.5-7B-Instruct, Qwen3-8B
- **Training**: Full-parameter SFT via LLaMA-Factory
- **Method**: NOT LoRA — full fine-tuning
- **Specific hyperparameters**: Not published in paper or repo. Paper says "same SFT pipeline from LLaMA-Factory" for all models.

## Their Evaluation Benchmarks (NOT tau-bench)

ADP does NOT evaluate on tau-bench. Their benchmarks:

| Benchmark | Agent | Base (7B) | ADP (7B) | Base (14B) | ADP (14B) | Base (32B) | ADP (32B) |
|-----------|-------|-----------|----------|------------|-----------|------------|-----------|
| SWE-Bench Verified | SWE-Agent | 0.4% | 20.2% | 2.0% | 34.4% | 2.2% | 40.3% |
| SWE-Bench Verified | OpenHands | 2.8% | 20.4% | 5.8% | 30.6% | 10.6% | 36.8% |
| WebArena | AgentLab | 4.5% | 21.0% | 5.5% | 22.2% | 10.9% | 22.9% |
| AgentBench OS | OpenHands | 3.5% | 27.1% | 2.8% | 20.8% | 27.8% | 34.7% |
| GAIA | OpenHands | 7.3% | 9.1% | — | — | — | — |

Cross-task transfer result (Table 6): ADP mixed training outperforms task-specific-only tuning on every benchmark.

## Why Use ADP as a Baseline?

ADP represents the current best approach for "unified multi-source SFT" — training a single model on diverse agent data to get broad capabilities. This is conceptually similar to what our merged/distilled models try to achieve, but through data mixing rather than parameter merging.

The comparison is: **Can our skill-specialized LoRA experts (with orchestration or merging) outperform a single model trained on a massive, diverse dataset?**

## How to Reproduce as a Fair Baseline

### Step 1: Get the ADP training data

```bash
# Clone the repo
git clone https://github.com/neulab/agent-data-protocol.git
cd agent-data-protocol
pip install -r requirements.txt

# Download standardized ADP data from HuggingFace
# Dataset: neulab/agent-data-collection
python -c "from datasets import load_dataset; ds = load_dataset('neulab/agent-data-collection')"
```

### Step 2: Convert ADP data to SFT format

ADP provides converters for OpenHands, SWE-Agent, and AgentLab formats. For tau-bench evaluation, we need the **tool-calling / API-action format** since tau-bench is a tool-use benchmark.

The closest match would be using ADP's API Action format and converting to the chat + tool_call format that tau-bench expects (OpenAI-style function calling).

### Step 3: Train with LLaMA-Factory

The paper uses LLaMA-Factory for SFT. To match our setup as closely as possible:

```bash
# Install LLaMA-Factory
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
```

**Recommended training config** (matching paper's approach on our base model):

```yaml
# llama_factory_adp_config.yaml
model_name_or_path: Qwen/Qwen3-30B-A3B-Instruct-2507
stage: sft
do_train: true
finetuning_type: full  # Paper uses full SFT, not LoRA
dataset: adp_converted  # Point to converted ADP data
template: qwen3
output_dir: ./output/qwen3-30b-adp-sft

# Hyperparameters (estimate based on LLaMA-Factory defaults for 30B)
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 1e-5
num_train_epochs: 1
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
max_length: 32768

# DeepSpeed for 30B model
deepspeed: ds_config_zero3.json
```

**IMPORTANT**: The paper trains Qwen2.5-7B and Qwen3-8B. Our base is Qwen3-30B-A3B. For a fair comparison, we should either:
1. Train Qwen3-30B-A3B on ADP data (apples-to-apples with our experts), OR
2. Evaluate their released Qwen3-8B model on tau-bench (different base model, but shows ADP's approach)

### Step 4: Evaluate on tau-bench with our settings

Create a config matching our standard evaluation setup:

```yaml
# config-adp-airline.yml
domain: airline
agent_llm: vllm://qwen3-30b-adp-sft
agent: null

agent_llm_args:
  temperature: 0.0
  max_context_length: 32000
  tokenizer_model: openai/Qwen/Qwen3-30B-A3B-Instruct-2507

user_llm: vllm://Qwen/Qwen3-30B-A3B-Instruct-2507
user_llm_args:
  temperature: 0.0
  max_context_length: 32000
  tokenizer_model: openai/Qwen/Qwen3-30B-A3B-Instruct-2507

user: null
num_trials: 1
max_steps: 50
max_concurrency: 1
seed: 42
verbose: true
save_to: adp-sft-airline

vllm:
  base_url: http://localhost:8080/v1
```

Run command:
```bash
cd /home/ubuntu/hangook/games/evals/benchmarks/tau2_bench_eval
python main.py --config config-adp-airline.yml
python main.py --config config-adp-retail.yml
```

## Fair Comparison Settings

To ensure apples-to-apples comparison with our existing results:

| Setting | Our Setup | ADP Baseline |
|---------|-----------|-------------|
| Base model | Qwen3-30B-A3B-Instruct-2507 | Same (retrain on ADP data) |
| Temperature | 0.0 | 0.0 |
| Max context | 32000 | 32000 |
| Num trials | 1 | 1 |
| Max steps | 50 | 50 |
| Seed | 42 | 42 |
| User LLM | Qwen3-30B-A3B (same server) | Same |
| Domains | airline (50 tasks), retail (114 tasks) | Same |
| tau-bench version | tau2-bench (local fork) | Same |

## Key Differences from Our Approach

| Aspect | ADP | Our Approach |
|--------|-----|-------------|
| Training data | 1.3M diverse trajectories from 13 sources | Task-specific tau-bench rollouts |
| Training method | Full SFT on mixed data | Skill-specific GRPO LoRA adapters |
| Model count | 1 unified model | Multiple expert LoRA adapters |
| Inference | Single model | Orchestrator routing between experts |
| Skill specificity | General-purpose agent | Specialized per-skill |
| Data domain | Coding, SWE, browsing, tool use (general) | Customer service (tau-bench specific) |

## Caveats

1. **ADP's data is general-purpose** — it does NOT contain tau-bench-specific customer service data. Performance on tau-bench may be lower than on their reported benchmarks.
2. **No released 30B model** — we need to train our own. Their released models (if any) are 7-8B.
3. **Full SFT on 30B requires significant compute** — multiple GPUs with DeepSpeed ZeRO-3.
4. **Training hyperparameters not fully published** — we'll need to use reasonable defaults from LLaMA-Factory.
