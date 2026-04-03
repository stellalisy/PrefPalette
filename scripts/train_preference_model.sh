#!/bin/bash
# PrefPalette Preference Model Training Script
# Trains the attribute-mediated preference model with attention-based integration.
#
# Usage: bash scripts/train_preference_model.sh <subreddit>

SUBREDDIT=${1:-"askdocs"}
FEATURE_GROUP=${2:-"all"}
MODEL_SIZE=${3:-"1B"}
LR=${4:-"1e-5"}
BS=${5:-"256"}
DROPOUT=${6:-"0.1"}
DATA_DIR=${7:-"data/preference_pairs"}
EMBEDDING_DIR=${8:-"data/attribute_embeddings"}

if [ "$FEATURE_GROUP" = "norms" ]; then
    FEATURES="supportiveness,politeness,sarcasm,humor,formality,verbosity,directness,assertiveness,empathy"
elif [ "$FEATURE_GROUP" = "values" ]; then
    FEATURES="Self-Direction,Stimulation,Hedonism,Achievement,Power,Security,Conformity,Tradition,Benevolence,Universalism"
else
    FEATURES="supportiveness,politeness,sarcasm,humor,formality,verbosity,directness,assertiveness,empathy,Self-Direction,Stimulation,Hedonism,Achievement,Power,Security,Conformity,Tradition,Benevolence,Universalism"
fi

MAIN_KEY="${SUBREDDIT}_${MODEL_SIZE}_dr${DROPOUT}_lr${LR}_bs${BS}_valattn_${FEATURE_GROUP}_time"
SAVE_DIR="outputs/preference_models/${FEATURE_GROUP}_time/${MAIN_KEY}"

python scripts/launch_training.py \
    --num_gpus 1 \
    --train_module openrlhf.cli.train_rm \
    --train_yaml_path configs/preference_model/example.yaml \
    --train_overrides "dataset=${DATA_DIR}/${SUBREDDIT}|save_path=${SAVE_DIR}|ckpt_path=${SAVE_DIR}|subreddit=${SUBREDDIT}|feature_classifiers=${FEATURES}|feature_dataset=${EMBEDDING_DIR}/${SUBREDDIT}/${MODEL_SIZE}|feature_dropout=${DROPOUT}|learning_rate=${LR}|train_batch_size=${BS}|feature_group=${FEATURE_GROUP}|wandb_run_name=${MAIN_KEY}"
