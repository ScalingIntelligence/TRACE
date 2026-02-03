# PPO Training for Kuhn Poker

## Quick Start

### 1. Start as many vLLM servers as wanted for inference.

Remember to change port and cuda_visible_device for each server

```bash

export HF_HOME=/matx/u/$USER/.cache/huggingface #matx
export HF_HOME=/home/ubuntu/.cache/huggingface #pi
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export VLLM_RPC_TIMEOUT=2000
export CUDA_VISIBLE_DEVICES=0 # then 1, then 2
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --host 0.0.0.0 \
  --port 8082 \
  --dtype bfloat16 \
  --max-model-len 55000 \
  --enable-lora \
  --max-loras 2 \
  --gpu-memory-utilization 0.8 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes

```
#Qwen/Qwen3-30B-A3B-Instruct-2507


### 2. Start Training

```bash
# With vLLM server
export CUDA_VISIBLE_DEVICES=3
export VLLM_BASE_URLS="http://localhost:8080,http://localhost:8081,http://localhost:8082"
export VLLM_MODEL="Qwen/Qwen3-4B-Instruct-2507"
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export VLLM_TIMEOUT_S=2000
python train_ppo.py --root /matx/u/$USER --use_constrained_decoding True #matx

python train_ppo.py --game liars_dice --root /home/ubuntu/Alex/run35 
#pi liars dice


```

All model weights and outputs will be saved to `/matx/u/$USER/` (3T drive).

## Add a new OpenSpiel game (simple)
1) Open `openspiel_wrapper.py` and add a new `OpenSpielGameConfig` entry to `OPENSPIEL_GAME_CONFIGS` with your alias, the `openspiel_name` passed to `pyspiel.load_game`, and a short system prompt.  
2) (Optional) Provide a stable `action_map` or `allowed_action_ids` if the default `[action_N]` mapping is not what you want.  
3) Run training with `python train_ppo.py --game <your_alias> --use_constrained_decoding True --root /home/ubuntu` (adjust root/path as needed).

## tau2-bench evaluation during training
You can optionally run eval on tau2-bench at a fixed iteration interval.

1) Install tau2-bench (provides the `tau2` CLI) and its dependencies using `pip install -e evals/benchmarks/tau2_bench_eval/ from repo root`
2) Download tau2 data files:
   `python evals/benchmarks/tau2_bench_eval/setup_data.py`
3) Enable the periodic eval by setting `Config.TAU2_EVAL_EVERY_ITERS` in `config.py` to a positive integer.

Currently, the eval setup is:
- `agent_llm` is the currently-trained LoRA adapter served by vLLM (`ppo_policy` by default).
- `user_llm` is the base `Qwen/Qwen3-4B-Instruct-2507` model served by vLLM.

## OpenBMB ToolBench Evaluation

Evaluate models on [OpenBMB ToolBench](https://github.com/OpenBMB/ToolBench) benchmark (16,000+ real-world APIs).

### Setup

1) Clone OpenBMB ToolBench data (already included at `evals/toolbench/`):
   ```bash
   cd evals
   git clone https://github.com/OpenBMB/ToolBench.git toolbench
   ```

2) Download full test data from [Google Drive](https://drive.google.com/drive/folders/1TysbSWYpP8EioFu9xPJtpbJZMLLmwAmL) and place in `evals/toolbench/data/test_instruction/`

### Running Evaluation

Make sure a vLLM server is running (see Quick Start above), then:

```bash
# Evaluate on G1 test set (single-tool scenarios)
python3 evals/benchmarks/eval_openbmb_toolbench.py \
    --test-set G1 \
    --vllm-url http://localhost:8082 \
    --vllm-model Qwen/Qwen3-4B-Instruct-2507 \
    --toolbench-dir evals/toolbench \
    --num-samples 50 \
    --output-dir evals/toolbench_results

# Evaluate on G2 test set (intra-category multi-tool)
python3 evals/benchmarks/eval_openbmb_toolbench.py \
    --test-set G2 \
    --vllm-url http://localhost:8082 \
    --num-samples 50

# Evaluate on G3 test set (intra-collection multi-tool)
python3 evals/benchmarks/eval_openbmb_toolbench.py \
    --test-set G3 \
    --vllm-url http://localhost:8082 \
    --num-samples 50
```

### Test Sets
- **G1**: Single-tool scenarios (easiest)
- **G2**: Intra-category multi-tool scenarios
- **G3**: Intra-collection multi-tool scenarios (hardest)

### Output
Results are saved to `--output-dir` (default: `eval_results/openbmb_toolbench/`):
- `{test_set}_{model}_{timestamp}.json`: Aggregated metrics (precision, recall, F1, pass rate)
- `{test_set}_{model}_{timestamp}_responses.jsonl`: Detailed per-query responses for analysis
