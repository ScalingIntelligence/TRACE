


```bash
conda activate games
export HF_HOME=/workspace/.cache/huggingface
export VLLM_RPC_TIMEOUT=2000
export CUDA_VISIBLE_DEVICES=0
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --host 0.0.0.0 \
  --port 8080 \
  --dtype bfloat16 \
  --max-model-len 32000 \
  --enable-lora \
  --max-loras 2 \
  --gpu-memory-utilization 0.9 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --tensor-parallel-size 1 
```

conda activate games
export HF_HOME=/workspace/.cache/huggingface
export VLLM_RPC_TIMEOUT=2000
export CUDA_VISIBLE_DEVICES=1
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --host 0.0.0.0 \
  --port 9000 \
  --dtype bfloat16 \
  --max-model-len 32000 \
  --enable-lora \
  --max-loras 2 \
  --gpu-memory-utilization 0.9 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --tensor-parallel-size 1 



conda activate games
export HF_HOME=/workspace/.cache/huggingface
export VLLM_RPC_TIMEOUT=2000
export CUDA_VISIBLE_DEVICES=4
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507   --host 0.0.0.0   --port 8090   --dtype bfloat16   --max-model-len 32000   --enable-lora   --max-loras 2   --gpu-memory-utilization 0.85   --enable-auto-tool-choice   --tool-call-parser hermes 



conda activate games
export HF_HOME=/workspace/.cache/huggingface
export VLLM_RPC_TIMEOUT=2000
export CUDA_VISIBLE_DEVICES=5
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507   --host 0.0.0.0   --port 9010   --dtype bfloat16   --max-model-len 32000   --enable-lora   --max-loras 2   --gpu-memory-utilization 0.85   --enable-auto-tool-choice   --tool-call-parser hermes 





export VLLM_RPC_TIMEOUT=2000
export HF_HOME=/workspace/.cache/huggingface
export CUDA_VISIBLE_DEVICES=2
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507   --host 0.0.0.0   --port 9090   --dtype bfloat16   --max-model-len 32000   --enable-lora   --max-loras 2   --gpu-memory-utilization 0.85   --enable-auto-tool-choice   --tool-call-parser hermes 


conda activate games
export HF_HOME=/workspace/.cache/huggingface
export VLLM_RPC_TIMEOUT=2000
export CUDA_VISIBLE_DEVICES=6
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507   --host 0.0.0.0   --port 8082   --dtype bfloat16   --max-model-len 32000   --enable-lora   --max-loras 2   --gpu-memory-utilization 0.9   --enable-auto-tool-choice   --tool-call-parser hermes --no-enable-prefix-caching --max-num-seqs 1



conda activate games
export HF_HOME=/workspace/.cache/huggingface
export VLLM_RPC_TIMEOUT=2000
export CUDA_VISIBLE_DEVICES=2
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507   --host 0.0.0.0   --port 9092   --dtype bfloat16   --max-model-len 32000   --enable-lora   --max-loras 2   --gpu-memory-utilization 0.9   --enable-auto-tool-choice   --tool-call-parser hermes --no-enable-prefix-caching --max-num-seqs 1

0. 9000
1. 8080
2. 9090
3. 9092
4. 8090
5. 9010
6. 8070
7. 9070

vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --host 0.0.0.0 \
  --port 9000 \
  --dtype bfloat16 \
  --max-model-len 32000 \
  --enable-lora \
  --max-loras 2 \
  --gpu-memory-utilization 0.8 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --no-enable-prefix-caching \
  --tensor-parallel-size 1 \
  --max-num-seqs 1



  CUDA_VISIBLE_DEVICES=6,7 \
  NCCL_P2P_DISABLE=1 \
  WANDB_API_KEY=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 \
  VLLM_BASE_URLS=http://localhost:8090 \
  VLLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 \
  VLLM_TIMEOUT_S=2000 \
  VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
  VLLM_RPC_TIMEOUT=2000 \
  PYTHONUNBUFFERED=1 \
  torchrun --nproc_per_node=2 train_grpo_optimized.py \
    --game tau_tool_calling \
    --user-llm-url http://localhost:9010/v1 \
    --user-llm-model "Qwen/Qwen3-30B-A3B-Instruct-2507"

  export HF_HOME=/workspace/.cache/huggingface
  CUDA_VISIBLE_DEVICES=2,3 \
  NCCL_P2P_DISABLE=1 \
  WANDB_API_KEY=f4ef099e7073d103963e5c986e4f818f5a526ee8 \
  VLLM_BASE_URLS=http://localhost:8080 \
  VLLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 \
  VLLM_TIMEOUT_S=2000 \
  VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
  VLLM_RPC_TIMEOUT=2000 \
  PYTHONUNBUFFERED=1 \
  torchrun --nproc_per_node=2 train_grpo_optimized.py \
    --game adversarial_policy \
    --user-llm-url http://localhost:9000/v1 --temperature-range 0.5,0.7,1.0 \
    --user-llm-model "Qwen/Qwen3-30B-A3B-Instruct-2507"


  export HF_HOME=/workspace/.cache/huggingface
  CUDA_VISIBLE_DEVICES=2,3 \
  NCCL_P2P_DISABLE=1 \
  WANDB_API_KEY=f4ef099e7073d103963e5c986e4f818f5a526ee8 \
  VLLM_BASE_URLS=http://localhost:8080 \
  VLLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 \
  VLLM_TIMEOUT_S=2000 \
  VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
  VLLM_RPC_TIMEOUT=2000 \
  PYTHONUNBUFFERED=1 \
  torchrun --nproc_per_node=2 --master-port 29501 train_grpo_optimized.py \
    --game adversarial_policy \
    --user-llm-url http://localhost:9000/v1 --temperature-range 0.5,0.7,1.0 --compact-tools \
    --user-llm-model "Qwen/Qwen3-30B-A3B-Instruct-2507"