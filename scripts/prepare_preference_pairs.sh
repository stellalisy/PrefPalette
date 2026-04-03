#!/bin/bash
# Step 2: Prepare Preference Pairs
# Creates preference pairs from preprocessed Reddit data by linking comments
# to their parent posts and pairing comments with different scores.
#
# Usage: bash scripts/prepare_preference_pairs.sh

INPUT_DIR=${1:-"data/preprocessed"}
OUTPUT_DIR=${2:-"data/preference_pairs"}
SUBREDDITS_FILE=${3:-"data/subreddits.txt"}

mkdir -p ${OUTPUT_DIR}

python -m prefpalette.preprocessing.prepare_preference_pairs \
    --input_dir ${INPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --subreddits_file ${SUBREDDITS_FILE} \
    --start_year 2022 \
    --end_year 2023 \
    --max_pair_per_post 50
