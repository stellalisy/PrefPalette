#!/bin/bash
# PrefPalette Full Pipeline
# Runs all steps from raw Reddit data to trained preference model.
#
# This script is meant as a reference showing the full pipeline sequence.
# In practice, you'll likely run each step separately, especially the
# GPU-intensive steps (counterfactual generation, training).
#
# Usage: bash scripts/run_full_pipeline.sh

set -e

SUBREDDITS_FILE=${SUBREDDITS_FILE:-"data/subreddits.txt"}
RAW_DATA_PATTERN=${RAW_DATA_PATTERN:-"data/raw/part-{idx}.bz2"}
PREPROCESSED_DIR=${PREPROCESSED_DIR:-"data/preprocessed"}
PREF_PAIRS_DIR=${PREF_PAIRS_DIR:-"data/preference_pairs"}
COUNTERFACTUAL_DIR=${COUNTERFACTUAL_DIR:-"data/counterfactual"}
ATTR_DATA_DIR=${ATTR_DATA_DIR:-"data/attribute_training_data"}
EMBEDDING_DIR=${EMBEDDING_DIR:-"data/attribute_embeddings"}
MODEL_ENDPOINT=${MODEL_ENDPOINT:-"http://localhost:8000"}

ATTRIBUTES="supportiveness,politeness,sarcasm,humor,formality,verbosity,directness,assertiveness,empathy,Self-Direction,Stimulation,Hedonism,Achievement,Power,Security,Conformity,Tradition,Benevolence,Universalism"

echo "=================================================="
echo "Step 1a: Extract raw Reddit data into per-subreddit shards"
echo "=================================================="
bash scripts/preprocess_reddit.sh "${RAW_DATA_PATTERN}" "${PREPROCESSED_DIR}"

echo "=================================================="
echo "Step 1b: Consolidate shards into single files per subreddit"
echo "=================================================="
bash scripts/consolidate_shards.sh "${PREPROCESSED_DIR}" "${SUBREDDITS_FILE}"

echo "=================================================="
echo "Step 2: Create preference pairs"
echo "=================================================="
bash scripts/prepare_preference_pairs.sh "${PREPROCESSED_DIR}" "${PREF_PAIRS_DIR}" "${SUBREDDITS_FILE}"

echo "=================================================="
echo "Step 3: Generate counterfactual attribute variations"
echo "  (Requires a vLLM endpoint serving the teacher model)"
echo "=================================================="
while IFS= read -r subreddit || [ -n "$subreddit" ]; do
    [ -z "$subreddit" ] && continue
    echo "  Generating counterfactuals for ${subreddit}..."
    bash scripts/counterfactual_generation.sh "${subreddit}" "${MODEL_ENDPOINT}" "${PREPROCESSED_DIR}" "${COUNTERFACTUAL_DIR}"
done < "${SUBREDDITS_FILE}"

echo "=================================================="
echo "Step 4: Prepare attribute predictor training data"
echo "=================================================="
bash scripts/prepare_attribute_data.sh "${COUNTERFACTUAL_DIR}" "${ATTR_DATA_DIR}" "${SUBREDDITS_FILE}"

echo "=================================================="
echo "Step 5: Train attribute predictors (one per attribute)"
echo "=================================================="
IFS=',' read -ra ATTR_LIST <<< "${ATTRIBUTES}"
for attr in "${ATTR_LIST[@]}"; do
    echo "  Training attribute predictor for ${attr}..."
    bash scripts/train_attribute_predictor.sh "${attr}" 1B 1e-5 256 "${ATTR_DATA_DIR}"
done

echo "=================================================="
echo "Step 6: Generate attribute embeddings for preference data"
echo "=================================================="
while IFS= read -r subreddit || [ -n "$subreddit" ]; do
    [ -z "$subreddit" ] && continue
    for attr in "${ATTR_LIST[@]}"; do
        echo "  Generating ${attr} embeddings for ${subreddit}..."
        bash scripts/generate_attribute_embeddings.sh "${attr}" "${subreddit}" 1B 1e-5 256 "${PREF_PAIRS_DIR}" "outputs/attribute_predictors" "${EMBEDDING_DIR}"
    done
done < "${SUBREDDITS_FILE}"

echo "=================================================="
echo "Step 7: Train PrefPalette preference model"
echo "=================================================="
while IFS= read -r subreddit || [ -n "$subreddit" ]; do
    [ -z "$subreddit" ] && continue
    echo "  Training preference model for ${subreddit}..."
    bash scripts/train_preference_model.sh "${subreddit}" all 1B 1e-5 256 0.1 "${PREF_PAIRS_DIR}" "${EMBEDDING_DIR}"
done < "${SUBREDDITS_FILE}"

echo "=================================================="
echo "Step 8: Evaluate with LLM judge"
echo "=================================================="
while IFS= read -r subreddit || [ -n "$subreddit" ]; do
    [ -z "$subreddit" ] && continue
    echo "  Evaluating ${subreddit}..."
    bash scripts/evaluate_llm_judge.sh "${subreddit}" gpt4o_clf "${PREF_PAIRS_DIR}"
done < "${SUBREDDITS_FILE}"

echo "=================================================="
echo "Pipeline complete!"
echo "=================================================="
