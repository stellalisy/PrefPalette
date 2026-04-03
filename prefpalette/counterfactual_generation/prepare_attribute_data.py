"""Convert counterfactual generations into training data for attribute predictors.

For each attribute dimension, creates pairwise preference data from the 5-level
counterfactual rewrites. Level pairs (e.g., level 1 vs 5) form preference pairs
where the higher level is "chosen" and the lower level is "rejected".

Input: counterfactual JSONL files from `generate.py`, one per subreddit:
    counterfactual_zeroshot_{subreddit}.jsonl

Output: per-attribute train/eval/test splits:
    {output_dir}/{attribute}/train.jsonl
    {output_dir}/{attribute}/eval_comment.jsonl
    {output_dir}/{attribute}/eval_ood.jsonl
    {output_dir}/{attribute}/eval_level.jsonl
    {output_dir}/{attribute}/test_comment.jsonl
    {output_dir}/{attribute}/test_ood.jsonl
    {output_dir}/{attribute}/test_level.jsonl
"""

import os
import json
import random
import argparse


ATTRIBUTES = [
    "verbosity", "formality", "supportiveness", "politeness",
    "sarcasm", "humor", "directness", "assertiveness", "empathy",
    "Self-Direction", "Stimulation", "Hedonism", "Achievement",
    "Power", "Security", "Conformity", "Tradition", "Benevolence",
    "Universalism",
]

ORDERED_PAIRS = [
    ("1", "2"), ("1", "3"), ("1", "4"), ("1", "5"),
    ("2", "3"), ("2", "4"), ("2", "5"),
    ("3", "4"), ("3", "5"),
    ("4", "5"),
]


def build_pairs_for_comment(line, attribute, subreddit):
    """Create all 10 ordered pairs for one comment on one attribute dimension."""
    rewrite = line.get("rewrite", {}).get(attribute, {})
    pairs = []
    for lo, hi in ORDERED_PAIRS:
        if lo not in rewrite or hi not in rewrite:
            continue
        pairs.append({
            "id": f"{line['id']}_{attribute}_{lo}_{hi}",
            "subreddit": subreddit,
            "context": [{"role": "user", "content": f"{line['post_title']}\n{line['post_body']}"}],
            "rejected": [{"role": "assistant", "content": rewrite[lo]}],
            "chosen": [{"role": "assistant", "content": rewrite[hi]}],
        })
    return pairs


def gather_attribute_data(attribute, subreddits, counterfactual_dir, max_per_subreddit=100,
                          num_test_subreddits=10):
    """Gather train/eval/test splits for a single attribute."""
    shuffled = list(subreddits)
    random.shuffle(shuffled)
    test_subreddits = set(shuffled[:num_test_subreddits])

    train_data = []
    test_data_comment = []
    test_data_ood = []

    for subreddit in subreddits:
        filepath = os.path.join(counterfactual_dir, f"counterfactual_zeroshot_{subreddit}.jsonl")
        if not os.path.exists(filepath):
            continue
        with open(filepath) as f:
            lines = [json.loads(line) for line in f][:max_per_subreddit]
        if not lines:
            continue

        test_comment_idxs = set(random.choices(range(len(lines)), k=1))

        subreddit_data = []
        for i, line in enumerate(lines):
            pairs = build_pairs_for_comment(line, attribute, subreddit)
            if i in test_comment_idxs:
                if subreddit in test_subreddits:
                    test_data_ood.extend(pairs)
                else:
                    test_data_comment.extend(pairs)
            else:
                subreddit_data.extend(pairs)

        if subreddit in test_subreddits:
            test_data_ood.extend(subreddit_data)
        else:
            train_data.extend(subreddit_data)

    test_level_idxs = set(random.choices(range(len(train_data)), k=len(test_data_comment)))
    test_data_level = [train_data[i] for i in test_level_idxs]
    train_data = [train_data[i] for i in range(len(train_data)) if i not in test_level_idxs]

    for lst in [train_data, test_data_comment, test_data_ood, test_data_level]:
        random.shuffle(lst)

    def split_half(data):
        mid = len(data) // 2
        return data[:mid], data[mid:]

    eval_ood, test_data_ood = split_half(test_data_ood)
    eval_comment, test_data_comment = split_half(test_data_comment)
    eval_level, test_data_level = split_half(test_data_level)

    return {
        "train": train_data,
        "eval_comment": eval_comment,
        "eval_ood": eval_ood,
        "eval_level": eval_level,
        "test_comment": test_data_comment,
        "test_ood": test_data_ood,
        "test_level": test_data_level,
    }


def write_splits(splits, output_dir, attribute):
    attr_dir = os.path.join(output_dir, attribute)
    os.makedirs(attr_dir, exist_ok=True)
    for split_name, data in splits.items():
        filepath = os.path.join(attr_dir, f"{split_name}.jsonl")
        with open(filepath, "w") as f:
            for entry in data:
                json.dump(entry, f)
                f.write("\n")
        print(f"  {split_name}: {len(data)} pairs")


def main():
    parser = argparse.ArgumentParser(description="Prepare attribute predictor training data from counterfactual generations")
    parser.add_argument("--counterfactual_dir", type=str, required=True, help="Directory containing counterfactual_zeroshot_{subreddit}.jsonl files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for attribute training data")
    parser.add_argument("--subreddits_file", type=str, required=True, help="File listing subreddits (one per line)")
    parser.add_argument("--attributes", type=str, default=None, help="Comma-separated list of attributes (default: all 19)")
    parser.add_argument("--max_per_subreddit", type=int, default=100, help="Max comments to use per subreddit")
    parser.add_argument("--num_test_subreddits", type=int, default=10, help="Number of subreddits held out for OOD test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.subreddits_file) as f:
        subreddits = [line.strip() for line in f if line.strip()]

    attributes = ATTRIBUTES if args.attributes is None else [a.strip() for a in args.attributes.split(",")]

    for attribute in attributes:
        print(f"Gathering data for: {attribute}")
        splits = gather_attribute_data(
            attribute, subreddits, args.counterfactual_dir,
            max_per_subreddit=args.max_per_subreddit,
            num_test_subreddits=args.num_test_subreddits,
        )
        write_splits(splits, args.output_dir, attribute)
    print("Done.")


if __name__ == "__main__":
    main()
