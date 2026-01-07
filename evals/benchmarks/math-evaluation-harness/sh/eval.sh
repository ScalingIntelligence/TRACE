set -ex

# export CUDA_VISIBLE_DEVICES=7
PROMPT_TYPE=$1
MODEL_NAME_OR_PATH=$2
OUTPUT_DIR=$3
WANDB_ID=$4

SPLIT="test"
NUM_TEST_SAMPLE=-1

# English open datasets
# DATA_NAME="gsm8k,math500,minerva_math,gaokao2023en,olympiadbench,college_math,aime24,amc23"
DATA_NAME="math500,aime24,aime25,olympiadbench,amc23,minerva_math"
TOKENIZERS_PARALLELISM=false \
python3 -u math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --max_tokens_per_call 8192 \
    --seed 0 \
    --temperature 0.6 \
    --n_sampling 4 \
    --top_p 0.95 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite

# Extract step number from model name if it follows the pattern of having "step" followed by numbers
if [[ $MODEL_NAME_OR_PATH =~ step[_-]?([0-9]+) ]]; then
    STEP="${BASH_REMATCH[1]}"
    echo "Extracted step: $STEP"
else
    STEP="None"
    echo "No step number found in model name"
fi


RUN_NAME=$(basename "$MODEL_NAME_OR_PATH")
echo "Extracted run name: $RUN_NAME"

if [[ $MODEL_NAME_OR_PATH =~ ^(.+)[_-]?step[_-]?[0-9]+$ ]]; then
    WANDB_GROUP="${BASH_REMATCH[1]}"
    WANDB_GROUP="${WANDB_GROUP%-}"
    WANDB_GROUP="${WANDB_GROUP%_}"
else
    WANDB_GROUP="$MODEL_NAME_OR_PATH"
fi

python3 sh/collect_results.py \
    --base_dir ${OUTPUT_DIR} \
    --model_name ${MODEL_NAME_OR_PATH} \
    --output_path ${OUTPUT_DIR}/metrics.csv \
    --benchmarks ${DATA_NAME}