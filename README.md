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
  --port 8036 \
  --dtype bfloat16 \
  --max-model-len 15000 \
  --enable-lora \
  --max-loras 2 \
  --gpu-memory-utilization 0.8
```

### 2. Start Training

```bash
# With vLLM server
export CUDA_VISIBLE_DEVICES=3
export VLLM_BASE_URLS="http://localhost:8034,http://localhost:8035,http://localhost:8036"
export VLLM_MODEL="Qwen/Qwen3-4B-Instruct-2507"
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export VLLM_TIMEOUT_S=2000
python train_ppo.py --root /matx/u/$USER --use_constrained_decoding True #matx

python train_ppo.py --game liars_dice --root /home/ubuntu/Alex/run3 
#pi liars dice


```

All model weights and outputs will be saved to `/matx/u/$USER/` (3T drive).

## Add a new OpenSpiel game (simple)
1) Open `openspiel_wrapper.py` and add a new `OpenSpielGameConfig` entry to `OPENSPIEL_GAME_CONFIGS` with your alias, the `openspiel_name` passed to `pyspiel.load_game`, and a short system prompt.  
2) (Optional) Provide a stable `action_map` or `allowed_action_ids` if the default `[action_N]` mapping is not what you want.  
3) Run training with `python train_ppo.py --game <your_alias> --use_constrained_decoding True --root /home/ubuntu` (adjust root/path as needed).
