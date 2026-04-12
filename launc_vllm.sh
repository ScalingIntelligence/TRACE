conda activate trace
export VLLM_RPC_TIMEOUT=2000
export HF_HOME=/home/hangook/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
vllm serve Qwen/Qwen3.5-2B   --host 0.0.0.0   --port 8080   --dtype bfloat16   --max-model-len 32000   --enable-lora   --max-loras 2   --gpu-memory-utilization 0.85   --enable-auto-tool-choice   --tool-call-parser hermes 