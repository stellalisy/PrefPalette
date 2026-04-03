import bz2
import json
import argparse
import os
import glob
from datetime import datetime
import logging


def extract(args):
    """Extract raw Reddit .bz2 dumps into per-subreddit shard files."""
    print(f"Processing indices {args.start_idx} to {args.end_idx} (exclusive)")

    for i in range(args.start_idx, args.end_idx):
        idx = str(i).zfill(5)
        input_filepath = args.input_pattern.format(idx=idx)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processing {input_filepath}")

        with bz2.open(input_filepath, "rt") as file:
            data = [json.loads(line.strip()) for line in file]
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loaded {len(data)} samples")

        categories = [
            "unknown" if 'name' not in sample
            else "comments" if "t1_" in sample['name']
            else "posts" if "t3_" in sample['name']
            else "unknown"
            for sample in data
        ]
        subreddits = [sample["subreddit"] for sample in data]
        file_ids = [f"{subreddits[i]}_{categories[i]}" for i in range(len(data))]
        file_ids_set = set(file_ids)
        subreddits_set = set(subreddits)
        categories_set = set(categories)

        num_comments, num_posts, num_unknown = 0, 0, 0
        for subreddit in subreddits_set:
            for category in categories_set:
                file_id = f"{subreddit}_{category}"
                if file_id not in file_ids_set:
                    continue

                out_dir = os.path.join(args.output_dir, subreddit)
                os.makedirs(out_dir, exist_ok=True)
                out_filepath = os.path.join(out_dir, f"{subreddit}_{category}_{idx}.jsonl")
                if os.path.exists(out_filepath + '.done'):
                    continue

                idxs = [i for i, x in enumerate(file_ids) if x == file_id]

                if category == "comments":
                    num_comments += len(idxs)
                elif category == "posts":
                    num_posts += len(idxs)
                else:
                    num_unknown += len(idxs)

                with open(out_filepath, "w") as f:
                    for index in idxs:
                        f.write(json.dumps(data[index]) + "\n")
                with open(out_filepath + '.done', "w") as f:
                    f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {file_id}: {len(idxs)}")

        del data
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Finished file {idx}: comments={num_comments}, posts={num_posts}, unknown={num_unknown}")


def consolidate(args):
    """Merge per-shard files into single {subreddit}_posts.jsonl and {subreddit}_comments.jsonl files.

    After extraction, each subreddit directory contains many shard files like:
        {subreddit}_comments_00001.jsonl, {subreddit}_comments_00002.jsonl, ...
    This step merges them into the consolidated format expected by downstream pipeline steps:
        {subreddit}_comments.jsonl, {subreddit}_posts.jsonl
    """
    if args.subreddits_file:
        with open(args.subreddits_file) as f:
            subreddits = [line.strip() for line in f if line.strip()]
    else:
        subreddits = [d for d in os.listdir(args.output_dir)
                      if os.path.isdir(os.path.join(args.output_dir, d))]

    for subreddit in sorted(subreddits):
        subreddit_dir = os.path.join(args.output_dir, subreddit)
        if not os.path.isdir(subreddit_dir):
            print(f"Skipping {subreddit}: directory not found")
            continue

        for category in ["comments", "posts"]:
            pattern = os.path.join(subreddit_dir, f"{subreddit}_{category}_*.jsonl")
            shard_files = sorted(glob.glob(pattern))
            if not shard_files:
                continue

            out_filepath = os.path.join(args.output_dir, f"{subreddit}_{category}.jsonl")
            seen_ids = set()
            count = 0
            with open(out_filepath, "w") as out_f:
                for shard_file in shard_files:
                    with open(shard_file) as in_f:
                        for line in in_f:
                            entry = json.loads(line)
                            entry_id = entry.get("name", entry.get("id"))
                            if entry_id not in seen_ids:
                                seen_ids.add(entry_id)
                                out_f.write(json.dumps(entry) + "\n")
                                count += 1

            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {subreddit} {category}: {count} entries from {len(shard_files)} shards")


def main():
    parser = argparse.ArgumentParser(description="Preprocess Reddit data dumps into per-subreddit files")
    subparsers = parser.add_subparsers(dest="command", help="Pipeline step to run")

    extract_parser = subparsers.add_parser("extract", help="Extract .bz2 dumps into per-subreddit shard files")
    extract_parser.add_argument("--input_pattern", type=str, required=True, help="Input file pattern with {idx} placeholder, e.g., /path/to/reddit/part-{idx}.bz2")
    extract_parser.add_argument("--output_dir", type=str, required=True, help="Output directory for preprocessed files")
    extract_parser.add_argument("--start_idx", type=int, default=0)
    extract_parser.add_argument("--end_idx", type=int, default=480)
    extract_parser.add_argument("--log_filepath", type=str, default=None)

    consolidate_parser = subparsers.add_parser("consolidate", help="Merge shard files into single per-subreddit files")
    consolidate_parser.add_argument("--output_dir", type=str, required=True, help="Directory containing per-subreddit shard subdirectories (same as extract --output_dir)")
    consolidate_parser.add_argument("--subreddits_file", type=str, default=None, help="Optional file listing subreddits to consolidate (one per line)")

    args = parser.parse_args()

    if args.command == "extract":
        if args.log_filepath:
            os.makedirs(os.path.dirname(args.log_filepath), exist_ok=True)
            logging.basicConfig(filename=args.log_filepath, level=logging.DEBUG)
        extract(args)
    elif args.command == "consolidate":
        consolidate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
