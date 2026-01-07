# Qwen2.5-Math-Instruct Series
PROMPT_TYPE="qwen-self-play"

# Qwen2.5-Math-1.5B-Instruct
export CUDA_VISIBLE_DEVICES="4,5,6,7"
MODEL_NAME_OR_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
OUTPUT_DIR="data/eval/DeepSeek-R1-Distill-Qwen-1.5B"
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $OUTPUT_DIR
