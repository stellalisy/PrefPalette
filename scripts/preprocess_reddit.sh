#!/bin/bash
# Step 1a: Extract Reddit Data
# Extracts raw Reddit .bz2 dumps into per-subreddit shard files.
#
# Prerequisites:
# - Raw Reddit data dumps (e.g., from pushshift.io) as .bz2 files
#
# Usage: bash scripts/preprocess_reddit.sh

INPUT_PATTERN=${1:-"data/raw/part-{idx}.bz2"}
OUTPUT_DIR=${2:-"data/preprocessed"}
START_IDX=${3:-0}
END_IDX=${4:-480}

mkdir -p ${OUTPUT_DIR}
mkdir -p logs/preprocessing

python -m prefpalette.preprocessing.preprocess_reddit extract \
    --input_pattern ${INPUT_PATTERN} \
    --output_dir ${OUTPUT_DIR} \
    --start_idx ${START_IDX} \
    --end_idx ${END_IDX} \
    --log_filepath logs/preprocessing/extract.log
