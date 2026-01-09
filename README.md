# PPO Training for Kuhn Poker

## Quick Start

### 1. Start vLLM Server (Optional, for faster generation)

```bash
export HF_HOME=/matx/u/$USER/.cache/huggingface
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 768 \
  --enable-lora \
  --max-loras 1 \
  --gpu-memory-utilization 0.8
```

### 2. Start Training

```bash
# With vLLM server
export CUDA_VISIBLE_DEVICES=1,2,3
export VLLM_BASE_URL="http://localhost:8000"
export VLLM_MODEL="Qwen/Qwen3-4B-Instruct-2507"
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
python train_ppo.py --root /matx/u/$USER --use_constrained_decoding True

# Without vLLM (local generation)
python train_ppo.py --root /matx/u/$USER --use_constrained_decoding True
```

All model weights and outputs will be saved to `/matx/u/$USER/` (3T drive).

