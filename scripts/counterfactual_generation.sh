#!/bin/bash
# Counterfactual Generation Script
# Generates counterfactual variations of Reddit comments along attribute dimensions
# using a strong teacher LM (e.g., Llama 3.1 405B).
#
# Prerequisites:
# - A vLLM endpoint serving the teacher model
# - Preprocessed Reddit data (posts + comments per subreddit)
#
# Usage: bash scripts/counterfactual_generation.sh

SUBREDDIT=${1:-"askdocs"}
MODEL_ENDPOINT=${2:-"http://localhost:8000"}
INPUT_DIR=${3:-"data/preprocessed"}
OUTPUT_DIR=${4:-"data/counterfactual"}

mkdir -p ${OUTPUT_DIR}
mkdir -p logs/counterfactual

python -m prefpalette.counterfactual_generation.generate \
    --subreddit ${SUBREDDIT} \
    --input_dir ${INPUT_DIR} \
    --output_filepath ${OUTPUT_DIR}/counterfactual_zeroshot_${SUBREDDIT}.jsonl \
    --log_filepath logs/counterfactual/${SUBREDDIT}.log \
    --model_endpoint ${MODEL_ENDPOINT} \
    --model_name meta-llama/Llama-3.1-405B-Instruct-FP8 \
    --max_samples 100 \
    --temperature 1.0 \
    --max_tokens 512
