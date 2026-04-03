import os
import json
import logging
import random
import time
from collections import defaultdict
import argparse

from . import prompts
from .llm_client import VLLMClient


def parse_args():
    parser = argparse.ArgumentParser(description="Generate counterfactual attribute variations of Reddit comments")
    parser.add_argument('--dimensions', type=str, default=None, help="Comma-separated list of attribute dimensions")
    parser.add_argument('--levels', type=str, default=None, help="Comma-separated list of levels (1-5)")
    parser.add_argument('--subreddit', type=str, required=True)
    parser.add_argument('--input_dir', type=str, required=True, help="Directory with {subreddit}_posts.jsonl and {subreddit}_comments.jsonl")
    parser.add_argument('--output_filepath', type=str, required=True)
    parser.add_argument('--log_filepath', type=str, default=None)
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.1-405B-Instruct-FP8")
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--max_tokens', type=int, default=512)
    parser.add_argument('--model_endpoint', type=str, required=True, help="URL of the vLLM endpoint")
    parser.add_argument('--max_samples', type=int, default=100000)
    return parser.parse_args()


def process_comment(c_id, comment, title, post, dimensions, levels, usage, client, args, data_processed):
    log_comment = comment.replace('\n', '\\n')
    logging.info(f"Original comment: {log_comment}")

    if c_id in data_processed:
        out_dict = defaultdict(dict, {k: v for k, v in data_processed[c_id].items() if v})
        comment = comment if len(comment) > 0 else out_dict.get("synthetic_original", "")
    else:
        out_dict = defaultdict(dict, {
            'id': c_id, 'post_title': title, 'post_body': post,
            'original': comment,
            'usage': {"prompt_tokens": 0, "total_tokens": 0, "completion_tokens": 0}
        })

    comment_usage = {"prompt_tokens": 0, "total_tokens": 0, "completion_tokens": 0}

    for dim in dimensions:
        out_dict["rewrite"][dim] = out_dict["rewrite"].get(dim, {})
        for level in levels:
            if str(level) in out_dict["rewrite"][dim]:
                continue
            generate_counterfactual(dim, level, comment, title, post, out_dict, comment_usage, client, args)

    for k, v in comment_usage.items():
        out_dict["usage"][k] += v
        usage[k] += v

    logging.info(f"Processed comment {c_id}")
    return out_dict


def generate_counterfactual(dim, level, comment, title, post, out_dict, usage, client, args):
    if dim in prompts.schwartz_values:
        messages = prompts.create_prompt_schwartz(dim, level, title, post, comment)
    else:
        messages = prompts.get_zeroshot_prompts(dim, level, title, post, comment)

    response_obj = client.chat_completion(
        messages, model_name=args.model_name,
        max_tokens=args.max_tokens, temperature=args.temperature
    )
    assert response_obj["status_ok"], f"Failed to generate counterfactual for {dim} L-{level}"
    response_text = response_obj["content"].split("COMMENT:")[-1].strip()
    for k in usage.keys():
        usage[k] += response_obj["usage"][k]
    log_response = response_text.replace('\n', '\\n')
    logging.info(f"{dim} L-{level}: {log_response}")
    out_dict["rewrite"][dim][str(level)] = response_text


def main():
    args = parse_args()

    if args.log_filepath:
        os.makedirs(os.path.dirname(os.path.realpath(args.log_filepath)), exist_ok=True)
        logging.basicConfig(filename=args.log_filepath, level=logging.INFO, filemode='a',
                            format='[%(asctime)s - %(levelname)s] %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')

    if args.dimensions is None:
        dimensions = prompts.norms + prompts.schwartz_values
    else:
        dimensions = [s.strip() for s in args.dimensions.split(",")]

    levels = [1, 2, 3, 4, 5] if args.levels is None else [int(s.strip()) for s in args.levels.split(",")]
    logging.info(f"Dimensions: {dimensions}, Levels: {levels}")

    data_processed = {}
    if os.path.exists(args.output_filepath):
        with open(args.output_filepath, "r") as f:
            for line in f:
                d = json.loads(line)
                data_processed[d["id"]] = d
        logging.info(f"Loaded {len(data_processed)} previously processed comments")

    with open(os.path.join(args.input_dir, f"{args.subreddit}_posts.jsonl")) as f:
        posts = {p['name']: p for p in (json.loads(line.strip()) for line in f)}
    with open(os.path.join(args.input_dir, f"{args.subreddit}_comments.jsonl")) as f:
        comments = {
            c['name']: {**c, 'post_title': posts[c['link_id']]['title'], 'post_body': posts[c['link_id']]['selftext']}
            for c in (json.loads(line.strip()) for line in f) if c['link_id'] in posts
        }
    del posts

    client = VLLMClient(model_endpoint=args.model_endpoint, model_name=args.model_name)
    usage = {"prompt_tokens": 0, "total_tokens": 0, "completion_tokens": 0}

    new_ids = [c_id for c_id in comments.keys() if c_id not in data_processed]
    random.shuffle(new_ids)
    new_ids = new_ids[:max(0, args.max_samples - len(data_processed))]
    comments_items = [(c_id, comments[c_id]) for c_id in new_ids]
    del comments

    os.makedirs(os.path.dirname(os.path.realpath(args.output_filepath)), exist_ok=True)
    for i, (c_id, sample) in enumerate(comments_items):
        logging.info(f"Processing comment {c_id} [{i + 1}/{len(comments_items)}]")
        title = sample.get('post_title', 'None') or 'None'
        post_body = sample.get('post_body', 'None') or 'None'
        comment = sample['body']

        out_dict = process_comment(
            c_id, comment, title, post_body, dimensions, levels, usage, client, args, data_processed
        )
        with open(args.output_filepath, "a+") as f:
            json.dump(out_dict, f)
            f.write("\n")

    logging.info(f"Done. Processed {len(comments_items)} comments, usage: {usage}")


if __name__ == "__main__":
    main()
