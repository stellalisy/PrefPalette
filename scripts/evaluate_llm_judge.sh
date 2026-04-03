#!/bin/bash
# LLM Judge Evaluation Script
# Evaluates preference prediction accuracy using LLM judges (GPT-4o, etc.)
#
# Usage: bash scripts/evaluate_llm_judge.sh <subreddit>

SUBREDDIT=${1:-"askdocs"}
ANNOTATOR=${2:-"gpt4o_clf"}
DATA_DIR=${3:-"data/preference_pairs"}
OUTPUT_DIR=${4:-"outputs/llm_judge"}

mkdir -p ${OUTPUT_DIR}

python -m prefpalette.evaluation.llm_judge \
    --subreddit ${SUBREDDIT} \
    --input_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --annotator_dir configs/llm_judge \
    --annotator_name ${ANNOTATOR} \
    --test_split test_2022_comment \
    --include_time \
    --max_samples 1000
