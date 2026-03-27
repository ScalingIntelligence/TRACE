# Experimental Setup

## Benchmark

We evaluate on **tau2-bench** (Yao et al., 2025), a multi-turn customer service benchmark that tests an agent's ability to interact with simulated users through tool-calling conversations. The benchmark provides two domains:

- **Airline**: 50 tasks covering flight booking, modification, cancellation, and compensation handling. Tasks involve structured policy reasoning (e.g., basic economy restrictions, insurance eligibility), multi-step operations (e.g., cancel-then-rebook), and adversarial user interactions.
- **Retail**: 114 tasks covering order management, product exchanges, returns, and account modifications. Tasks emphasize multi-step execution across multiple orders, precise item selection from catalogs, and conditional logic.

Each task is evaluated against two criteria: (1) **DB correctness** — whether the agent's tool calls produced the correct database state, and (2) **Communication correctness** — whether the agent communicated required information to the user. A task is considered passed (reward = 1.0) only if both criteria are satisfied. We use a simulated user powered by the same base LLM, following the task scenario instructions.

### Evaluation Protocol

All evaluations use the following standardized settings unless otherwise noted:

| Parameter | Value |
|-----------|-------|
| Temperature | 0.0 (greedy decoding) |
| Max context length | 32,000 tokens |
| Max conversation steps | 50 |
| Number of trials | 1 |
| Random seed | 42 |
| User simulator | Same base model (Qwen3-30B-A3B) |

## Base Model

We use **Qwen3-30B-A3B-Instruct-2507** as our base model throughout all experiments. This is a Mixture-of-Experts (MoE) architecture with 30B total parameters and 3B active parameters per token, providing an efficient balance between model capacity and inference cost. The model supports native tool calling through its instruction-tuned chat template.

## Skill Identification

We identify skill gaps through **contrastive trajectory analysis**. Given baseline evaluation trajectories (38 airline failures, 72 retail failures out of 164 total tasks), we employ an LLM judge to independently analyze failed conversations and assign each failure to candidate skills from a predefined menu of 14 categories. This process is repeated 10 times with independent judge instances to measure selection robustness.

Our analysis identifies 5 skills that are consistently selected across runs:

| Skill | Selection Rate | Median Coverage |
|-------|---------------|-----------------|
| Structured data reasoning | 10/10 | 41 tasks |
| Multi-step task completion | 10/10 | 25 tasks |
| Precondition verification | 10/10 | 16 tasks |
| Tool calling precision | 8/10 | 20 tasks |
| Adversarial policy compliance | 4/10 | 14 tasks |

Distractor skills (language fluency, tone/empathy, format compliance, tool hallucination, proactive upselling) are selected 0/10 times, confirming the selection process does not over-fit to noise.

## Training

### Skill-Specific RL Training (GRPO)

We train separate LoRA adapters for each identified skill using **Group Relative Policy Optimization (GRPO)** (Shao et al., 2024). Each adapter is trained on a dedicated environment that isolates the target skill:

- **Structured data reasoning**: Multi-turn conversations requiring correct parsing of flight/product catalogs, attribute comparison, and selection under constraints.
- **Tool calling precision**: Single-turn and multi-turn tasks requiring exact tool argument construction from retrieved data.
- **Multi-step task completion**: Compound requests requiring sequential execution of 2-5 dependent operations.
- **Precondition verification**: Tasks testing policy eligibility checking before state-changing actions.
- **Adversarial policy compliance**: Conversations with adversarial users who apply emotional pressure, false claims, or social manipulation to override policy.

All environments use the same tools, database, and policy as tau2-bench, with reward computed by comparing the resulting database state against ground truth.

### LoRA Configuration

| Parameter | Value |
|-----------|-------|
| Rank (r) | 16 |
| Alpha | 16 |
| Dropout | 0.0 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| PEFT type | LoRA |

### GRPO Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 1e-5 |
| Sampling temperature | 1.0 |
| Mini-batch size | 4 |
| Groups per batch | 32--64 |
| Training iterations | 10--40 (per skill) |
| KL regularization | 0.0 (off; LoRA constrains drift) |
| Gradient checkpointing | Enabled (unsloth) |
| Optimizer | AdamW 8-bit |
| Max sequence length | 16,000 tokens |
| Thinking mode | Enabled |

Training is conducted on 4--8 NVIDIA A100-80GB GPUs using distributed data parallelism. Each skill adapter is trained independently, producing adapters of approximately 50M trainable parameters each (< 0.2% of total model parameters).

## Baselines

### Single-Model Baselines

- **Base model**: Qwen3-30B-A3B-Instruct-2507 with no fine-tuning.
- **On-policy GRPO (mixed)**: A single LoRA adapter trained on a uniform mixture of all skill environments simultaneously, testing whether joint training can capture multiple skills.
- **SFT distillation**: A single LoRA adapter trained via supervised fine-tuning on successful trajectories from the skill-specific experts, attempting to distill multiple experts into one model.

### Merging Baselines

We evaluate several approaches to combining skill-specific LoRA adapters into a single model:

- **Linear merge**: $W_{\text{final}} = W_{\text{base}} + \sum_i \alpha_i \cdot \Delta_i$ with equal weights.
- **Stacked merge**: Sequential application of adapter deltas, where each adapter's delta is computed on the already-modified weights from the previous adapter.
- **CORE-TSV merge**: Task-specific vector merging using TIES-inspired sign resolution.
- **Mixed-data training**: Training on combined rollout data from multiple skill environments at various mixing ratios.

### Skill-Augmented Prompting Baseline

We evaluate a prompt-only baseline where 12 general behavioral skills are appended to the system prompt without any model training. Skills describe broad agent capabilities (e.g., "ground every claim in tool output," "refuse clearly when policy prohibits") without domain-specific hints. This tests whether improved instructions alone can substitute for RL training.

### Multi-Model Orchestration

We evaluate an orchestrator that routes each conversation to the best-matched skill expert based on the user's initial request. The orchestrator uses a lightweight classifier to select among 3 expert LoRA adapters, with per-conversation routing (the selected expert handles the entire conversation).

## Evaluation Metrics

We report **pass rate** (fraction of tasks with reward = 1.0) as the primary metric, consistent with the tau2-bench evaluation protocol. We additionally report:

- **Union ceiling**: The fraction of tasks solved by at least one expert, representing the theoretical maximum achievable by perfect routing.
- **Per-task agreement**: For skill selection robustness experiments, we report the number of times each skill is selected across independent judge runs.
- **Coverage**: The number of failed tasks each identified skill addresses, measured as median with interquartile range (IQR) across independent analysis runs.

## Computational Resources

All experiments are conducted on a single node with 8 NVIDIA A100-SXM4-80GB GPUs (640 GB total GPU memory), AMD EPYC 7742 64-Core Processor, and 1.7 TB system memory. GRPO training for a single skill adapter takes approximately 2--4 hours. Evaluation of a single domain (50 or 114 tasks) takes approximately 30--60 minutes with a single vLLM inference server.
