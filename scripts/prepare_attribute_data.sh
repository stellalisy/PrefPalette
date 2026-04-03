#!/bin/bash
# Step 4: Prepare Attribute Training Data
# Converts counterfactual generations into pairwise training data for
# attribute predictors. Creates train/eval/test splits per attribute.
#
# Usage: bash scripts/prepare_attribute_data.sh

COUNTERFACTUAL_DIR=${1:-"data/counterfactual"}
OUTPUT_DIR=${2:-"data/attribute_training_data"}
SUBREDDITS_FILE=${3:-"data/subreddits.txt"}

mkdir -p ${OUTPUT_DIR}

python -m prefpalette.counterfactual_generation.prepare_attribute_data \
    --counterfactual_dir ${COUNTERFACTUAL_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --subreddits_file ${SUBREDDITS_FILE} \
    --max_per_subreddit 100 \
    --num_test_subreddits 10 \
    --seed 42
