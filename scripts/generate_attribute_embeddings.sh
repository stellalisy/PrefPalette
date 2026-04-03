#!/bin/bash
# Attribute Embedding Generation Script
# Uses trained attribute predictors to generate embeddings for preference data.
#
# For each attribute, loads the trained attribute predictor from Step 5,
# runs it over the preference pair data from Step 2, and saves per-comment
# embeddings as .pkl files for use in the final preference model training.
#
# Usage: bash scripts/generate_attribute_embeddings.sh <attribute> <subreddit>

ATTRIBUTE=${1:-"verbosity"}
SUBREDDIT=${2:-"askdocs"}
MODEL_SIZE=${3:-"1B"}
LR=${4:-"1e-5"}
BS=${5:-"256"}
DATA_DIR=${6:-"data/preference_pairs"}
MODEL_DIR=${7:-"outputs/attribute_predictors"}
OUTPUT_DIR=${8:-"data/attribute_embeddings"}

PREDICTOR_DIR="${MODEL_DIR}/${ATTRIBUTE}/llama-3.2-${MODEL_SIZE}_lr${LR}_bs${BS}"

mkdir -p ${OUTPUT_DIR}/${SUBREDDIT}/${MODEL_SIZE}

python scripts/launch_training.py \
    --num_gpus 1 \
    --train_module openrlhf.cli.train_rm \
    --train_yaml_path configs/attribute_predictor/example.yaml \
    --train_overrides "dataset=${DATA_DIR}/${SUBREDDIT}|save_path=${PREDICTOR_DIR}|gen_norm=${ATTRIBUTE}|embedding_filepath=${OUTPUT_DIR}/${SUBREDDIT}/${MODEL_SIZE}/${ATTRIBUTE}.pkl|train_split=train_2022|eval_split=eval_2022_comment,eval_2022_post|test_split=test_2022_comment,test_2022_post"
