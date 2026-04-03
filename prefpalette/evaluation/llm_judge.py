import os
import json
import logging
import time
from collections import defaultdict
from datetime import datetime
import argparse

from .annotator import Annotator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate preference prediction using LLM judges")
    parser.add_argument('--subreddit', type=str, required=True)
    parser.add_argument('--include_subreddit', action='store_true', default=False)
    parser.add_argument('--include_time', action='store_true', default=False)
    parser.add_argument('--input_dir', type=str, required=True, help="Directory containing preference pair files")
    parser.add_argument('--test_split', type=str, default='test_2022_comment')
    parser.add_argument('--output_dir', type=str, required=True, help="Directory for annotations and results")
    parser.add_argument('--annotator_dir', type=str, required=True, help="Path to annotator config directory")
    parser.add_argument('--annotator_name', type=str, default="gpt4o_clf")
    parser.add_argument('--model_endpoint', type=str, default=None, help="Override model endpoint")
    parser.add_argument('--api_account', type=str, default=None, help="API account to use (openai, azure)")
    parser.add_argument('--self_consistency', type=int, default=None)
    parser.add_argument('--max_samples', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log_filepath', type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    exp_key = f"{args.annotator_name}_judge-{args.subreddit}-{args.test_split}"
    if args.include_subreddit:
        exp_key += "-sid"
    if args.include_time:
        exp_key += "-time"
    exp_key += f"-s{args.seed}"

    annotation_filepath = os.path.join(args.output_dir, f"{exp_key}-annotations.jsonl")
    results_filepath = os.path.join(args.output_dir, f"{exp_key}-results.json")
    log_filepath = args.log_filepath or os.path.join(args.output_dir, f"{exp_key}.log")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_filepath), exist_ok=True)
    logging.basicConfig(filename=log_filepath, level=logging.INFO,
                        format='[%(asctime)s - %(levelname)s] %(message)s', datefmt='%Y-%m-%d,%H:%M:%S', filemode='a')

    override_config = {}
    if args.model_endpoint:
        override_config["model_endpoint"] = args.model_endpoint
    if args.self_consistency:
        override_config["self_consistency"] = args.self_consistency
    if args.api_account:
        override_config["api_account"] = args.api_account

    annotator = Annotator(os.path.join(args.annotator_dir, args.annotator_name), override_config=override_config)

    input_filepath = os.path.join(args.input_dir, args.subreddit, f"{args.test_split}.jsonl")
    with open(input_filepath, 'r') as f:
        comment_pairs = {d['id']: d for d in (json.loads(line) for line in f)}

    data_processed = {}
    if os.path.exists(annotation_filepath):
        with open(annotation_filepath, "r") as f:
            data_processed = {d['id']: d for d in (json.loads(line) for line in f)}

    logging.info(f"Loaded {len(comment_pairs)} pairs, {len(data_processed)} already processed")
    global_corrects = []
    usage = {"prompt_tokens": 0, "total_tokens": 0, "completion_tokens": 0}

    num_processed = 0
    for c_id, sample in comment_pairs.items():
        if num_processed >= args.max_samples:
            break

        if c_id in data_processed:
            global_corrects.append(data_processed[c_id]["score"])
            num_processed += 1
            continue

        context = sample["context"][0]["content"]
        chosen = sample["chosen"][0]["content"]
        rejected = sample["rejected"][0]["content"]

        if args.include_time:
            context = f"Post time: {datetime.fromtimestamp(sample['post_time']).strftime('%A %B %d %Y %H:%M:%S %p')} | " + context
            chosen = f"Response time: {datetime.fromtimestamp(sample['chosen_time']).strftime('%A %B %d %Y %H:%M:%S %p')} | " + chosen
            rejected = f"Response time: {datetime.fromtimestamp(sample['rejected_time']).strftime('%A %B %d %Y %H:%M:%S %p')} | " + rejected

        if args.include_subreddit:
            annotation = annotator.annotate_pair(subreddit=args.subreddit, context=context, comment_1=chosen, comment_2=rejected)
        else:
            annotation = annotator.annotate_pair(context=context, comment_1=chosen, comment_2=rejected)

        if annotation is None:
            continue

        out_dict = defaultdict(dict, sample)
        for k, v in annotation.items():
            out_dict[k] = v
        for k, v in annotation['total_usage'].items():
            usage[k] += v

        global_corrects.append(annotation["score"])

        with open(annotation_filepath, "a+") as f:
            json.dump(dict(out_dict), f)
            f.write("\n")

        num_processed += 1
        accuracy = sum(global_corrects) / len(global_corrects) if global_corrects else 0
        logging.info(f"Processed {c_id} ({num_processed}/{len(comment_pairs)}), accuracy={accuracy:.4f}")

        with open(results_filepath, "w") as f:
            json.dump({"subreddit": args.subreddit, "num_annotated": num_processed, "accuracy": accuracy}, f, indent=4)

    logging.info("Done")


if __name__ == "__main__":
    main()
