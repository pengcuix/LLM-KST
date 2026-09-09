set -ex
proj_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROMPT_TYPE=$1
MODEL_NAME_OR_PATH=$2
OUTPUT_DIR=$3
SEED=$4
DATA_NAME=$5

# API configuration
API_BATCH_SIZE=5  # Number of requests to batch together (only used when batch API disabled)
API_DELAY=1.0     # Delay in seconds between API calls (only used when batch API disabled)

# Model parameters
TEMPERATURE=0.0   # Use 0 for deterministic outputs

# Dataset configuration
SPLIT="test"
NUM_TEST_SAMPLE=-1  # -1 for full dataset

# Batch API configuration (read from environment variables set in run_api_claude.sh)
ANTHROPIC_USE_BATCH_FLAG=""
if [ "${ANTHROPIC_USE_BATCH:-0}" = "1" ]; then
    ANTHROPIC_USE_BATCH_FLAG="--anthropic_use_batch"
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
    --anthropic_api_key "${ANTHROPIC_API_KEY}" \
    ${ANTHROPIC_USE_BATCH_FLAG} \
    --anthropic_batch_poll_interval ${ANTHROPIC_BATCH_POLL_INTERVAL:-15}

echo "Completed evaluation for ${MODEL_NAME_OR_PATH}"
