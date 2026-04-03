import os
import json
import random
from collections import defaultdict
import logging
import datetime
import time
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Create preference pairs from Reddit data")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing {subreddit}_posts.jsonl and {subreddit}_comments.jsonl files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for preference pair files")
    parser.add_argument("--subreddits_file", type=str, required=True, help="File listing subreddits to process (one per line)")
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--shard_size", type=int, default=5)
    parser.add_argument("--start_year", type=int, default=2022)
    parser.add_argument("--end_year", type=int, default=2023)
    parser.add_argument("--max_pair_per_post", type=int, default=50)
    parser.add_argument("--max_pair_per_subreddit", type=int, default=1000000)
    parser.add_argument("--temporal_test_only", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_filepath", type=str, default=None)
    return parser.parse_args()


def load_subreddit_data(input_dir, subreddit, start_time=0, end_time=float('inf')):
    with open(os.path.join(input_dir, f"{subreddit}_posts.jsonl")) as f:
        posts = {
            p['name']: {
                "name": p['name'], "id": p['id'],
                "parent_id": p.get('parent_id'), "subreddit": p['subreddit'],
                "subreddit_id": p['subreddit_id'], "created_utc": float(p['created_utc']),
                "score": p['score'], "title": p['title'], "selftext": p['selftext'],
            }
            for p in (json.loads(line.strip()) for line in f)
            if start_time <= float(p['created_utc']) <= end_time
        }

    with open(os.path.join(input_dir, f"{subreddit}_comments.jsonl")) as f:
        comments = {
            c['name']: {
                "name": c['name'], "id": c['id'], "link_id": c['link_id'],
                "parent_id": c['parent_id'], "subreddit": c['subreddit'],
                "subreddit_id": c['subreddit_id'], "created_utc": float(c['created_utc']),
                "score": c['score'], "body": c['body'],
            }
            for c in (json.loads(line.strip()) for line in f)
            if c['link_id'] in posts and start_time <= float(c['created_utc']) <= end_time
        }

    return posts, comments


def link_posts_and_comments(posts, comments):
    comments_by_link_id = defaultdict(list)
    for comment in comments.values():
        comments_by_link_id[comment['link_id']].append(comment)

    processed_posts = {}
    for post_id, post in posts.items():
        if post_id not in comments_by_link_id:
            continue
        linked_comments = comments_by_link_id[post_id]
        last_comment_time = max(c["created_utc"] for c in linked_comments)

        out = {
            "post_id": post_id,
            "post_title": post['title'],
            "post_body": post['selftext'],
            "post_time": post['created_utc'],
            "post_score": post['score'],
            "post_last_comment_time": last_comment_time,
            "subreddit": post['subreddit'],
            "subreddit_id": post['subreddit_id'],
        }
        processed_comments = {
            cmt['id']: {
                "comment_id": cmt['id'],
                "comment_body": cmt['body'],
                "comment_time": cmt['created_utc'],
                "comment_score": cmt['score'],
                "parent_id": cmt['parent_id'],
                "link_id": cmt['link_id'],
            }
            for cmt in linked_comments
        }
        out['comments'] = processed_comments
        if len(processed_comments) > 1:
            processed_posts[post_id] = out

    return processed_posts


def create_pairs(post, max_pair_per_post, seed=42):
    random.seed(seed)
    samples = {}
    outer_keys = list(post['comments'].keys())
    random.shuffle(outer_keys)
    inner_keys = list(post['comments'].keys())
    random.shuffle(inner_keys)

    seen_pairs = defaultdict(list)
    for key_i in outer_keys:
        for key_j in inner_keys:
            if key_i == key_j or key_j in seen_pairs[key_i] or key_i in seen_pairs[key_j]:
                continue

            example_i = post['comments'][key_i]
            example_j = post['comments'][key_j]
            if not example_i['comment_score'] > example_j['comment_score']:
                continue

            sample_id = f"{post['subreddit']}-{post['post_id']}-{example_i['comment_id']}-{example_j['comment_id']}"
            samples[sample_id] = {
                "id": sample_id,
                "context": [{"role": "user", "content": f"{post['post_title']}\n{post['post_body']}"}],
                "chosen": [{"role": "assistant", "content": example_i['comment_body']}],
                "rejected": [{"role": "assistant", "content": example_j['comment_body']}],
                "subreddit": post['subreddit'],
                "margin": example_i['comment_score'] - example_j['comment_score'],
                "post_time": post['post_time'],
                "chosen_score": example_i['comment_score'],
                "rejected_score": example_j['comment_score'],
                "chosen_time": example_i['comment_time'],
                "rejected_time": example_j['comment_time'],
            }
            seen_pairs[key_i].append(key_j)
            seen_pairs[key_j].append(key_i)
            if len(samples) >= max_pair_per_post * 10:
                break

    sample_ids = list(samples.keys())
    random.seed(seed)
    random.shuffle(sample_ids)
    return [samples[sid] for sid in sample_ids[:max_pair_per_post]]


def split_and_write(processed_posts, subreddit, output_dir, args):
    max_pair_per_post = 20 if len(processed_posts) > 100000 else args.max_pair_per_post

    post_ids = list(processed_posts.keys())
    random.seed(args.seed)
    random.shuffle(post_ids)

    if args.temporal_test_only > 0:
        test_post_ids = []
    else:
        test_post_ids = post_ids[:max(min(300, int(len(post_ids) * 0.06)), 1)]

    test_post_samples, train_samples = [], []
    for post_id in test_post_ids:
        test_post_samples.extend(create_pairs(processed_posts[post_id], max_pair_per_post))
    for post_id in post_ids:
        if post_id in test_post_ids:
            continue
        train_samples.extend(create_pairs(processed_posts[post_id], max_pair_per_post))

    random.seed(args.seed)
    random.shuffle(train_samples)
    split_size = min(len(test_post_samples), len(train_samples))
    test_comment_samples = train_samples[:split_size]
    train_samples = train_samples[split_size:]

    eval_comment_samples = test_comment_samples[:len(test_comment_samples) // 2]
    test_comment_samples = test_comment_samples[len(test_comment_samples) // 2:]
    eval_post_samples = test_post_samples[:len(test_post_samples) // 2]
    test_post_samples = test_post_samples[len(test_post_samples) // 2:]

    logging.info(f"  Splits: train={len(train_samples)} eval_comment={len(eval_comment_samples)} eval_post={len(eval_post_samples)} test_comment={len(test_comment_samples)} test_post={len(test_post_samples)}")

    out_dir = os.path.join(output_dir, subreddit)
    os.makedirs(out_dir, exist_ok=True)

    def write_jsonl(filepath, data, max_samples):
        with open(filepath, "w") as f:
            for sample in data[:max_samples]:
                json.dump(sample, f)
                f.write("\n")

    if args.temporal_test_only > 0:
        write_jsonl(os.path.join(out_dir, f"test_{args.start_year}.jsonl"), train_samples, args.temporal_test_only)
    else:
        write_jsonl(os.path.join(out_dir, f"train_{args.start_year}.jsonl"), train_samples, args.max_pair_per_subreddit)
        write_jsonl(os.path.join(out_dir, f"eval_{args.start_year}_comment.jsonl"), eval_comment_samples, args.max_pair_per_subreddit)
        write_jsonl(os.path.join(out_dir, f"eval_{args.start_year}_post.jsonl"), eval_post_samples, args.max_pair_per_subreddit)
        write_jsonl(os.path.join(out_dir, f"test_{args.start_year}_comment.jsonl"), test_comment_samples, args.max_pair_per_subreddit)
        write_jsonl(os.path.join(out_dir, f"test_{args.start_year}_post.jsonl"), test_post_samples, args.max_pair_per_subreddit)


def main():
    args = parse_args()

    if args.log_filepath:
        os.makedirs(os.path.dirname(args.log_filepath), exist_ok=True)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                            filename=args.log_filepath, filemode='a')

    with open(args.subreddits_file, "r") as f:
        subreddits = [s.strip() for s in f.readlines()]
    subreddits = subreddits[args.shard_id * args.shard_size:min((args.shard_id + 1) * args.shard_size, len(subreddits))]

    start_time = time.mktime(datetime.datetime(args.start_year, 1, 1, 0, 0).timetuple())
    end_time = time.mktime(datetime.datetime(args.end_year, 1, 1, 0, 0).timetuple())

    for subreddit in subreddits:
        logging.info(f"Processing subreddit: {subreddit}")
        posts, comments = load_subreddit_data(args.input_dir, subreddit, start_time, end_time)
        logging.info(f"  Loaded {len(posts)} posts, {len(comments)} comments")
        processed_posts = link_posts_and_comments(posts, comments)
        logging.info(f"  After linking: {len(processed_posts)} posts with >1 comment")
        del posts, comments
        split_and_write(processed_posts, subreddit, args.output_dir, args)


if __name__ == "__main__":
    main()
