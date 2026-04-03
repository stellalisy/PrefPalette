#!/bin/bash
# Step 1b: Consolidate Shards
# Merges per-shard JSONL files into single {subreddit}_posts.jsonl and
# {subreddit}_comments.jsonl files needed by downstream steps.
#
# Usage: bash scripts/consolidate_shards.sh

OUTPUT_DIR=${1:-"data/preprocessed"}
SUBREDDITS_FILE=${2:-"data/subreddits.txt"}

python -m prefpalette.preprocessing.preprocess_reddit consolidate \
    --output_dir ${OUTPUT_DIR} \
    --subreddits_file ${SUBREDDITS_FILE}
