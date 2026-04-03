#!/bin/bash
# Attribute Predictor Training Script
# Trains per-attribute reward models using contrastive attribute distillation.
#
# This trains a small specialized model (e.g., Llama 3.2 1B) for each attribute
# dimension using counterfactual pairs from prepare_attribute_data.py.
#
# Usage: bash scripts/train_attribute_predictor.sh <attribute>

ATTRIBUTE=${1:-"verbosity"}
MODEL_SIZE=${2:-"1B"}
LR=${3:-"1e-5"}
BS=${4:-"256"}
DATA_DIR=${5:-"data/attribute_training_data"}

SAVE_DIR="outputs/attribute_predictors/${ATTRIBUTE}/llama-3.2-${MODEL_SIZE}_lr${LR}_bs${BS}"

python scripts/launch_training.py \
    --num_gpus 1 \
    --train_module openrlhf.cli.train_rm \
    --train_yaml_path configs/attribute_predictor/example.yaml \
    --train_overrides "dataset=${DATA_DIR}/${ATTRIBUTE}|save_path=${SAVE_DIR}|ckpt_path=${SAVE_DIR}|learning_rate=${LR}|train_batch_size=${BS}"
