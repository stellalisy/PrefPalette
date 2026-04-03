import os
import json
import logging
from collections import defaultdict
import argparse

from . import prompts
from .llm_client import VLLMClient


def parse_args():
    parser = argparse.ArgumentParser(description="Verify counterfactual generation quality")
    parser.add_argument('--dimensions', type=str, default=None, help="Comma-separated list of dimensions to verify")
    parser.add_argument('--subreddit', type=str, required=True)
    parser.add_argument('--input_filepath', type=str, required=True, help="Path to counterfactual generation output")
    parser.add_argument('--output_filepath', type=str, required=True, help="Path to write verification results")
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.1-405B-Instruct-FP8")
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--max_tokens', type=int, default=512)
    parser.add_argument('--model_endpoint', type=str, required=True)
    parser.add_argument('--self_consistency', type=int, default=1)
    parser.add_argument('--log_filepath', type=str, default=None)
    return parser.parse_args()


def verify_pair(dim, title, post, pair, client, args):
    all_messages = prompts.get_verifier_prompt(dim, title, post, pair[0], pair[1])
    correct_answer = ["B", "B", "A", "A", "A", "A", "B", "B"]
    corrects = []
    for i, messages in enumerate(all_messages):
        response_obj = client.chat_completion(
            messages, model_name=args.model_name,
            max_tokens=args.max_tokens, temperature=args.temperature
        )
        assert response_obj["status_ok"]
        response_text = response_obj["content"]
        if '[[A]]' in response_text:
            response = 'A'
        elif '[[B]]' in response_text:
            response = 'B'
        else:
            response = 'C'
        corrects.append(int(response == correct_answer[i]))
    return corrects


def verify_comment(c_id, sample, dimensions, client, args):
    comment = sample.get('synthetic_original', sample['original'])
    out_dict = defaultdict(dict, {'id': c_id, 'reference': comment})
    usage = {"prompt_tokens": 0, "total_tokens": 0, "completion_tokens": 0}

    for dim in dimensions:
        rewrites = sample['rewrite'][dim]
        rewrite_list = [rewrites[str(level)] for level in range(1, 6)]
        rewrite_pairs = [(rewrite_list[i], rewrite_list[j]) for i in range(len(rewrite_list)) for j in range(i + 1, len(rewrite_list))]

        all_corrects = []
        for pair in rewrite_pairs:
            pair_corrects = [verify_pair(dim, sample['post_title'], sample['post_body'], pair, client, args)
                             for _ in range(args.self_consistency)]
            all_corrects.append(pair_corrects)
        out_dict["results"][dim] = all_corrects
        logging.info(f"Verified {dim} for comment {c_id}")

    out_dict["usage"] = usage
    return out_dict


def main():
    args = parse_args()

    if args.log_filepath:
        os.makedirs(os.path.dirname(os.path.realpath(args.log_filepath)), exist_ok=True)
        logging.basicConfig(filename=args.log_filepath, level=logging.INFO,
                            format='[%(asctime)s - %(levelname)s] %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')

    dimensions = prompts.norms if args.dimensions is None else args.dimensions.split(',')
    client = VLLMClient(model_endpoint=args.model_endpoint, model_name=args.model_name)

    with open(args.input_filepath, 'r') as f:
        rewritten_comments = {d['id']: d for d in (json.loads(line) for line in f)}

    data_processed = {}
    if os.path.exists(args.output_filepath):
        with open(args.output_filepath, "r") as f:
            data_processed = {d['id']: d for d in (json.loads(line) for line in f)}

    logging.info(f"Loaded {len(rewritten_comments)} comments, {len(data_processed)} already verified")

    for c_id, sample in rewritten_comments.items():
        if c_id in data_processed:
            continue
        out_dict = verify_comment(c_id, sample, dimensions, client, args)
        os.makedirs(os.path.dirname(os.path.realpath(args.output_filepath)), exist_ok=True)
        with open(args.output_filepath, "a") as f:
            json.dump(out_dict, f)
            f.write("\n")
        logging.info(f"Processed comment {c_id}")


if __name__ == "__main__":
    main()
