set -ex
proj_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROMPT_TYPE=$1
MODEL_NAME_OR_PATH=$2
OUTPUT_DIR=$3
SEED=$4
DATA_NAME=$5

# API configuration
API_BATCH_SIZE=5  # Number of requests to batch together
API_DELAY=0.5     # Delay in seconds between API calls (adjust for rate limits)

# Model parameters
TEMPERATURE=0.0   # Use 0 for deterministic outputs

# OpenAI Batch API configuration
# Set OPENAI_USE_BATCH=1 to submit requests through OpenAI Batch API
OPENAI_USE_BATCH=${OPENAI_USE_BATCH:-1}
OPENAI_BATCH_COMPLETION_WINDOW=${OPENAI_BATCH_COMPLETION_WINDOW:-24h}
OPENAI_BATCH_POLL_INTERVAL=${OPENAI_BATCH_POLL_INTERVAL:-15}

# Dataset configuration
SPLIT="test"
NUM_TEST_SAMPLE=-1  # -1 for full dataset

OPENAI_BATCH_ARGS=()
if [ "${OPENAI_USE_BATCH}" = "1" ]; then
    OPENAI_BATCH_ARGS+=(
        --openai_use_batch
        --openai_batch_completion_window "${OPENAI_BATCH_COMPLETION_WINDOW}"
        --openai_batch_poll_interval "${OPENAI_BATCH_POLL_INTERVAL}"
    )
fi


python math_eval_api.py \
    --data_names "${DATA_NAME}" \
    --data_dir "../../data/dataset" \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --prompt_type "${PROMPT_TYPE}" \
    --split "${SPLIT}" \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed ${SEED} \
    --temperature ${TEMPERATURE} \
    --max_tokens_per_call 2048 \
    --api_batch_size ${API_BATCH_SIZE} \
    --api_delay ${API_DELAY} \
    --save_outputs \
    "${OPENAI_BATCH_ARGS[@]}" \
    --openai_api_key "${OPENAI_API_KEY}"

echo "Completed evaluation for ${MODEL_NAME_OR_PATH}"
