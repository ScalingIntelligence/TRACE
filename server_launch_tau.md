          
```bash
conda activate games
export HF_HOME=/dev/vda1/.cache/huggingface
export VLLM_RPC_TIMEOUT=2000
export CUDA_VISIBLE_DEVICES=7
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --host 0.0.0.0 \
  --port 2020 \
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

export HF_HOME=/lambda/nfs/lambda-stanford/tarun/.cache/huggingface                            
export VLLM_RPC_TIMEOUT=2000
export CUDA_VISIBLE_DEVICES=4
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507   --host 0.0.0.0   --port 8090   --dtype bfloat16   --max-model-len 32000   --enable-lora   --max-loras 2   --gpu-memory-utilization 0.9   --enable-auto-tool-choice   --tool-call-parser hermes 


export HF_HOME=/lambda/nfs/lambda-stanford/tarun/.cache/huggingface                            
export VLLM_RPC_TIMEOUT=2000
export CUDA_VISIBLE_DEVICES=5
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507   --host 0.0.0.0   --port 9090   --dtype bfloat16   --max-model-len 32000   --enable-lora   --max-loras 2   --gpu-memory-utilization 0.9   --enable-auto-tool-choice   --tool-call-parser hermes 


--no-enable-prefix-caching --max-num-seqs 1


conda activate games
export HF_HOME=/workspace/.cache/huggingface
export VLLM_RPC_TIMEOUT=2000
export CUDA_VISIBLE_DEVICES=0
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507   --host 0.0.0.0   --port 8080   --dtype bfloat16   --max-model-len 32000   --enable-lora   --max-loras 2   --gpu-memory-utilization 0.9   --enable-auto-tool-choice   --tool-call-parser hermes --no-enable-prefix-caching --max-num-seqs 1




conda activate games
export HF_HOME=/workspace/.cache/huggingface
export VLLM_RPC_TIMEOUT=2000
export CUDA_VISIBLE_DEVICES=0
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507   --host 0.0.0.0   --port 8080   --dtype bfloat16   --max-model-len 32000   --enable-lora   --max-loras 2   --gpu-memory-utilization 0.9   --enable-auto-tool-choice   --tool-call-parser hermes 

export VLLM_WEIGHTED_LORA_PIN_SLOT0=1
export VLLM_WEIGHTED_LORA_PORT=5051


conda activate games
export HF_HOME=/workspace/.cache/huggingface
export VLLM_RPC_TIMEOUT=2000
export CUDA_VISIBLE_DEVICES=0
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507   --host 0.0.0.0   --port 8080   --dtype bfloat16   --max-model-len 32000   --enable-lora   --max-loras 2   --gpu-memory-utilization 0.9   --enable-auto-tool-choice   --tool-call-parser hermes --no-enable-prefix-caching --max-num-seqs 1 


conda activate games
export HF_HOME=/workspace/.cache/huggingface
export VLLM_RPC_TIMEOUT=2000
export CUDA_VISIBLE_DEVICES=5
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507   --host 0.0.0.0   --port 9092   --dtype bfloat16   --max-model-len 32000   --enable-lora   --max-loras 2   --gpu-memory-utilization 0.9   --enable-auto-tool-choice   --tool-call-parser hermes --no-enable-prefix-caching --max-num-seqs 1


conda activate games
export HF_HOME=/workspace/.cache/huggingface
export VLLM_RPC_TIMEOUT=2000
export CUDA_VISIBLE_DEVICES=1
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
vllm serve tarsur909/Qwen3-30B-A3B-Instruct-2507-structured-10   --host 0.0.0.0   --port 8080   --dtype bfloat16   --max-model-len 32000   --enable-lora   --max-loras 2   --gpu-memory-utilization 0.9   --enable-auto-tool-choice   --tool-call-parser hermes --no-enable-prefix-caching --max-num-seqs 1



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


export HF_HOME=/workspace/.cache/huggingface
export VLLM_RPC_TIMEOUT=2000
export CUDA_VISIBLE_DEVICES=5
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True



conda activate games
export HF_HOME=~/.cache/huggingface
export VLLM_RPC_TIMEOUT=2000
export CUDA_VISIBLE_DEVICES=0
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507   --host 0.0.0.0   --port 8080   --dtype bfloat16   --max-model-len 32000   --enable-lora   --max-loras 2   --gpu-memory-utilization 0.9   --enable-auto-tool-choice   --tool-call-parser hermes --no-enable-prefix-caching --max-num-seqs 1


export HF_HOME=~/.cache/huggingface VLLM_RPC_TIMEOUT=2000 CUDA_VISIBLE_DEVICES=0 VLLM_ALLOW_RUNTIME_LORA_UPDATING=True && /home/ubuntu/miniconda3/envs/sky_games/bin/vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 --host 0.0.0.0 --port 8080 --dtype bfloat16 --max-model-len 32000 --enable-lora --max-loras 2 --gpu-memory-utilization 0.9 --enable-auto-tool-choice --tool-call-parser hermes

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
export CUDA_VISIBLE_DEVICES=7
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507   --host 0.0.0.0   --port 9003   --dtype bfloat16   --max-model-len 32000   --enable-lora   --max-loras 2   --gpu-memory-utilization 0.9   --enable-auto-tool-choice   --tool-call-parser hermes --no-enable-prefix-caching --max-num-seqs 1


8081
5. 9002
6. 8082
7. 9003[6:05 PM]

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
      --user-llm-url http://localhost:9000/v1 --temperature-range "0.5,0.7,1.0" \
      --compact-tools --user-llm-model "Qwen/Qwen3-30B-A3B-Instruct-2507" \
      --sft-data "/root/games/evals/benchmarks/tau2_bench_eval/data/simulations/qwen-3-30b-airline-qwen3-30b.json,/root/games/evals/benchmarks/tau2_bench_eval/data/simulations/qwen-3-30b-retail-qwen3-30b.json" \
      --sft-coef 0.5 --sft-per-step 2

CUDA_VISIBLE_DEVICES=2,3,6,7 \
  NCCL_P2P_DISABLE=1 \
  WANDB_API_KEY=f4ef099e7073d103963e5c986e4f818f5a526ee8 \
  VLLM_BASE_URLS=http://localhost:8080 \
  VLLM_MODEL=tarsur909/Qwen3-30B-A3B-Instruct-2507-tool-v2-35 \
  VLLM_TIMEOUT_S=2000 \
  VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
  VLLM_RPC_TIMEOUT=2000 \
  PYTHONUNBUFFERED=1 torchrun --nproc_per_node=2 --master-port 29501 train_grpo_optimized.py --game adversarial_policy --model tarsur909/Qwen3-30B-A3B-Instruct-2507-tool-v2-35 --compact-tools \
      --temperature-range "0.5,0.7,1.0" \
      --save-every 5 --user-llm-url http://localhost:9004/v1 --user-llm-model "Qwen/Qwen3-30B-A3B-Instruct-2507"


CUDA_VISIBLE_DEVICES=1,2,3 \
  NCCL_P2P_DISABLE=1 \
  WANDB_API_KEY=f4ef099e7073d103963e5c986e4f818f5a526ee8 \
  VLLM_BASE_URLS=http://localhost:8080 \
  VLLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 \
  VLLM_TIMEOUT_S=2000 \
  VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
  VLLM_RPC_TIMEOUT=2000 \
  PYTHONUNBUFFERED=1 torchrun --nproc_per_node=3 --master-port 29501 train_grpo_optimized.py --game structured_data_v2 --model Qwen/Qwen3-30B-A3B-Instruct-2507 --compact-tools \
      --temperature-range "0.5,0.7,1.0" \
      --save-every 5


CUDA_VISIBLE_DEVICES=5,6   NCCL_P2P_DISABLE=1   WANDB_API_KEY=f4ef099e7073d103963e5c986e4f818f5a526ee8   VLLM_BASE_URLS=http://localhost:9090   VLLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507   VLLM_TIMEOUT_S=2000   VLLM_ALLOW_RUNTIME_LORA_UPDATING=True   VLLM_RPC_TIMEOUT=2000   PYTHONUNBUFFERED=1 torchrun --nproc_per_node=2 --master-port 29500 train_grpo_optimized.py --game tau_tool_calling --model Qwen/Qwen3-30B-A3B-Instruct-2507 --compact-tools       --temperature-range "0.5,0.7,1.0"       --save-every 5 --group-size 32


export HF_HOME=/workspace/.cache/huggingface
export NCCL_P2P_DISABLE=1
export VLLM_RPC_TIMEOUT=2000
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=False
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507   --host 0.0.0.0   --port 8080   --dtype bfloat16   --max-model-len 32000    --gpu-memory-utilization 0.9   --enable-auto-tool-choice   --tool-call-parser hermes   --data-parallel-size 4 



export HF_HOME=/workspace/.cache/huggingface
export VLLM_RPC_TIMEOUT=2000
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=2,3
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=False
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507   --host 0.0.0.0   --port 9000   --dtype bfloat16   --max-model-len 32000    --gpu-memory-utilization 0.9   --enable-auto-tool-choice   --tool-call-parser hermes   --data-parallel-size 2 



CUDA_VISIBLE_DEVICES=1,2,3,4   NCCL_P2P_DISABLE=1   WANDB_API_KEY=f4ef099e7073d103963e5c986e4f818f5a526ee8   VLLM_BASE_URLS=http://localhost:8080   VLLM_MODEL=tarsur909/Qwen3-30B-A3B-Instruct-2507-structured-10   VLLM_TIMEOUT_S=2000   VLLM_ALLOW_RUNTIME_LORA_UPDATING=True   VLLM_RPC_TIMEOUT=2000   PYTHONUNBUFFERED=1   torchrun --nproc_per_node=4 --master-port 29501 train_grpo_optimized.py     --game multistep_task     --model tarsur909/Qwen3-30B-A3B-Instruct-2507-structured-10     --compact-tools     --temperature-range "0.7,0.9,1.0"     --save-every 5  --groups-per-batch 8 --group-size 16 



CUDA_VISIBLE_DEVICES=5 NCCL_P2P_DISABLE=1 WANDB_API_KEY=f4ef099e7073d103963e5c986e4f818f5a526ee8 python train_sft.py --sft-data /root/games/game_rollouts/structured_data_reasoning-d3-Qwen3-30B-A3B-Instruct-2507.json --model Qwen/Qwen3-30B-A3B-Instruct-2507

  CUDA_VISIBLE_DEVICES=1,2,3,4 NCCL_P2P_DISABLE=1 WANDB_API_KEY=f4ef099e7073d103963e5c986e4f818f5a526ee8 torchrun --nproc_per_node=4 train_sft.py \
  --sft-data /root/games/game_rollouts/structured_data_reasoning-d3-Qwen3-30B-A3B-Instruct-2507.json,/root/games/game_rollouts/multistep_task-Qwen3-30B-A3B-Instruct-2507-n100-s0.json \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 --max-samples-per-file 100

  CUDA_VISIBLE_DEVICES=3,4 NCCL_P2P_DISABLE=1 WANDB_API_KEY=f4ef099e7073d103963e5c986e4f818f5a526ee8 torchrun --nproc_per_node=2 train_sft.py \
  --sft-data /root/games/game_rollouts/multistep_task-Qwen3-30B-A3B-Instruct-2507-n100-s0.json \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 --max-samples-per-file 100

  CUDA_VISIBLE_DEVICES=1,2 NCCL_P2P_DISABLE=1 WANDB_API_KEY=f4ef099e7073d103963e5c986e4f818f5a526ee8 torchrun --nproc_per_node=2 --master-port 29501 train_sft.py \
  --sft-data /root/games/game_rollouts/tau_tool_calling-Qwen3-30B-A3B-Instruct-2507-Qwen3-30B-A3B-Instruct-2507-n400-s0_airline.json,/root/games/game_rollouts/tau_tool_calling-Qwen3-30B-A3B-Instruct-2507-Qwen3-30B-A3B-Instruct-2507-n400-s0_retail.json \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 --max-samples-per-file 50


CUDA_VISIBLE_DEVICES=6,7 NCCL_P2P_DISABLE=1 WANDB_API_KEY=f4ef099e7073d103963e5c986e4f818f5a526ee8 torchrun --nproc_per_node=2 train_sft.py \
  --sft-data "/home/ubuntu/hangook/games/data/sft_think_tagged/airline_llm_agent_qwen3-max-2026-01-23_user_simulator_gpt-4.1-2025-04-14.json,/home/ubuntu/hangook/games/data/sft_think_tagged/retail_llm_agent_qwen3-max-2026-01-23_user_simulator_gpt-4.1-2025-04-14.json" \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 




CUDA_VISIBLE_DEVICES=3 NCCL_P2P_DISABLE=1 WANDB_API_KEY=f4ef099e7073d103963e5c986e4f818f5a526ee8 python train_sft.py \
  --sft-data "/root/games/game_rollouts/structured_data_v2-Qwen3-30B-A3B-Instruct-2507-Qwen3-30B-A3B-Instruct-2507-n200-s0.json," \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 --max-samples-per-file 100 --num-epochs 1


python collect_rollouts.py --env structured_data_v2 --model Qwen/Qwen3-30B-A3B-Instruct-2507 --num-seeds 400 --base-url http://localhost:8080/v1 



CUDA_VISIBLE_DEVICES=6 NCCL_P2P_DISABLE=1 WANDB_API_KEY=f4ef099e7073d103963e5c986e4f818f5a526ee8 VLLM_BASE_URLS=http://localhost:9090   VLLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 VLLM_TIMEOUT_S=2000 VLLM_ALLOW_RUNTIME_LORA_UPDATING=True VLLM_RPC_TIMEOUT=2000   PYTHONUNBUFFERED=1 torchrun --nproc_per_node=2 --master-port 29500 train_grpo_optimized.py --game tau_tool_calling --model Qwen/Qwen3-30B-A3B-Instruct-2507 --compact-tools --temperature-range "0.5,0.7,1.0" --save-every 5 --group-size 32 --user-llm-url http://localhost:8080/v1 --user-llm-model "Qwen/Qwen3-30B-A3B-Instruct-2507




  CUDA_VISIBLE_DEVICES=4,5,6,7   NCCL_P2P_DISABLE=1   WANDB_API_KEY=f4ef099e7073d103963e5c986e4f818f5a526ee8   VLLM_BASE_URLS=http://localhost:8080   VLLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507   VLLM_TIMEOUT_S=2000   VLLM_ALLOW_RUNTIME_LORA_UPDATING=True   VLLM_RPC_TIMEOUT=2000   PYTHONUNBUFFERED=1 \
  torchrun --nproc_per_node=4 --master-port 29501 train_grpo_optimized.py     --games "structured_data_reasoning:0.5,multistep_task:0.5"     --model Qwen/Qwen3-30B-A3B-Instruct-2507     --compact-tools     --temperature-range "0.7,0.9,1.0"     --save-every 5  --groups-per-batch 8 --group-size 16 --user-llm-url http://localhost:9000/v1 --user-llm-model "Qwen/Qwen3-30B-A3B-Instruct-2507"





  CUDA_VISIBLE_DEVICES=0,1,3,4,6 \
  NCCL_P2P_DISABLE=1 \
  WANDB_API_KEY=f4ef099e7073d103963e5c986e4f818f5a526ee8 \
  WANDB_PROJECT=games \
  VLLM_BASE_URLS=http://localhost:9001 \
  VLLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 \
  VLLM_TIMEOUT_S=2000 \
  VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
  PYTHONUNBUFFERED=1 \
  torchrun --nproc_per_node=5 --master-port 29501 train_grpo_optimized.py \
      --game tec \
      --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
      --compact-tools \
      --groups-per-batch 16 \
      --temperature 1.0 \
      --lr 1e-5 \
      --mini-batch-size 2 \
      --stats-chunk-size 2 \
      --save-every 5

  CUDA_VISIBLE_DEVICES=0,1,3,4,6 \
  NCCL_P2P_DISABLE=1 \
  WANDB_API_KEY=f4ef099e7073d103963e5c986e4f818f5a526ee8 \
  WANDB_PROJECT=games \
  VLLM_BASE_URLS=http://localhost:9001 \
  VLLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 \
  VLLM_TIMEOUT_S=2000 \
  VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
  PYTHONUNBUFFERED=1 \
  torchrun --nproc_per_node=5 --master-port 29501 train_grpo_optimized.py \
      --game tec \
      --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
      --compact-tools \
      --groups-per-batch 16 \
      --temperature 1.0 \
      --lr 1e-5 \
      --mini-batch-size 4 \
      --stats-chunk-size 4 \
      --save-every 5 \
      --sft-coef 0 \
      --no-step-rewards

################################################
  CUDA_VISIBLE_DEVICES=0,1,4,5,6 \
  NCCL_P2P_DISABLE=1 \
  WANDB_API_KEY=f4ef099e7073d103963e5c986e4f818f5a526ee8 \
  WANDB_PROJECT=games \
  VLLM_BASE_URLS=http://localhost:5051 \
  VLLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 \
  VLLM_TIMEOUT_S=2000 \
  VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
  PYTHONUNBUFFERED=1 \
  torchrun --nproc_per_node=5 --master-port 29501 train_grpo_optimized.py \
      --game tec_v2 \
      --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
      --compact-tools \
      --groups-per-batch 32 \
      --temperature 1.0 \
      --lr 5e-6 \
      --mini-batch-size 4 \
      --stats-chunk-size 8 \
      --save-every 5 \
      --sft-coef 0 \
      --no-step-rewards \
      --dynamic-sampling-max-batches 1




  CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
  NCCL_P2P_DISABLE=1 \
  WANDB_API_KEY=f4ef099e7073d103963e5c986e4f818f5a526ee8 \
  WANDB_PROJECT=games \
  VLLM_BASE_URLS=http://localhost:9001 \
  VLLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 \
  VLLM_TIMEOUT_S=2000 \
  VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
  PYTHONUNBUFFERED=1 \
  torchrun --nproc_per_node=6 --master-port 29501 train_grpo_optimized.py \
      --game tec \
      --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
      --compact-tools \
      --groups-per-batch 32 \
      --temperature 1.0 \
      --lr 5e-6 \
      --mini-batch-size 2 \
      --stats-chunk-size 2 \
      --save-every 5 \
      --no-filter-constant-groups \
      --sft-coef 0 \
      --no-step-rewards


  \begin{table}[t]
  \centering
  \caption{ToolSandbox results (129 base scenarios). We report perfect score (similarity $= 1.0$) and mean similarity across all scenarios.}
  \label{tab:toolsandbox}
  \resizebox{\columnwidth}{!}{
  \begin{tabular}{lcccccc}
  \toprule
  \textbf{Category} & \multicolumn{2}{c}{\textbf{Base}} & \multicolumn{2}{c}{\textbf{Precondition}} & \multicolumn{2}{c}{\textbf{Multi-step}} \\
   & Perfect & Mean Sim. & Perfect & Mean Sim. & Perfect & Mean Sim. \\
  \midrule
  Single Tool Call & 0/19 & 0.368 & 0/19 & 0.342 & 0/19 & 0.368 \\
  Multiple Tool Call & 4/82 & 0.408 & 4/82 & 0.403 & 4/82 & 0.404 \\
  Multiple User Turn & 0/28 & 0.211 & 0/28 & 0.211 & 0/28 & 0.226 \\
  Insufficient Info & 15/28 & 0.536 & 15/28 & 0.536 & 16/28 & 0.571 \\
  State Dependency & 0/24 & 0.381 & 0/24 & 0.356 & 0/24 & 0.381 \\
  Canonicalization & 4/59 & 0.456 & 4/59 & 0.448 & 4/59 & 0.450 \\
  \midrule
  \textbf{Overall} & 19/129 & 0.430 & 19/129 & 0.423 & 20/129 & \textbf{0.435} \\
  \bottomrule
  \end{tabular}
  }
  \end{table}

\begin{table}[t]
  \centering
  \caption{ToolSandbox results (129 base scenarios). We report perfect score (similarity $= 1.0$) and mean similarity.}
  \label{tab:toolsandbox}
  \begin{tabular}{lcc}
  \toprule
  \textbf{Model} & \textbf{Perfect} & \textbf{Mean Sim.} \\
  \midrule
  Base (Qwen3-30B-A3B) & 19/129 & 0.430 \\
  + Precondition & 19/129 & 0.423 \\
  + Multi-step & \textbf{20/129} & \textbf{0.435} \\
  \bottomrule
  \end{tabular}
  \end{table}