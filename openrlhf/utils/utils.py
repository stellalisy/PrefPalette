import os

from datasets import interleave_datasets, load_dataset, load_from_disk, Dataset
from transformers import AutoTokenizer

import datetime

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def get_tokenizer(pretrain, model, padding_side="left", strategy=None, use_fast=True):
    try:
        tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True, use_fast=use_fast)
    except:
        if "llama" in pretrain.lower():
            if "8b" in pretrain: tokenizer_path = "meta-llama/Llama-3.1-8B-Instruct"
            elif "3b" in pretrain: tokenizer_path = "meta-llama/Llama-3.2-3B-Instruct"
            else: tokenizer_path = "meta-llama/Llama-3.1-70B-Instruct"
        elif "mistral" in pretrain.lower():
            if "7b" in pretrain: tokenizer_path = "mistralai/Mistral-7B-Instruct-v0.1"
            elif "mixture" in pretrain: tokenizer_path = "mistralai/Mixtral-8x7B-Instruct-v0.1"
            else: tokenizer_path = "mistralai/Mistral-7B-Instruct-v0.1"
        elif "qwen" in pretrain.lower():
            if "7b" in pretrain: tokenizer_path = "Qwen/Qwen-7B-Chat"
            else: tokenizer_path = "Qwen/Qwen2-Model-Base"
        elif "gemma" in pretrain.lower():
            if "7b" in pretrain: tokenizer_path = "Gemma-7B"
            else: tokenizer_path = "Gemma-2-8B"
        elif "baichuan" in pretrain.lower():
            if "7b" in pretrain: tokenizer_path = "Baichuan-7B-Chat"
        else: tokenizer_path = pretrain
        print(f"Failed to load tokenizer from {pretrain}, try to load from inferred base model: {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=use_fast)


    tokenizer.padding_side = padding_side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer


def get_strategy(args):
    from openrlhf.utils.deepspeed import DeepspeedStrategy

    strategy = DeepspeedStrategy(
        seed=getattr(args, "seed", 42),
        max_norm=getattr(args, "max_norm", 1.0),
        micro_train_batch_size=getattr(args, "micro_train_batch_size", 1),
        train_batch_size=getattr(args, "train_batch_size", 128),
        zero_stage=args.zero_stage,
        bf16=getattr(args, "bf16", True),
        args=args,
    )
    return strategy


def blending_datasets(
    datasets,
    probabilities,
    strategy=None,
    seed=42,
    max_count=5000000,
    max_count_eval=None,
    return_eval=True,
    stopping_strategy="first_exhausted",
    train_split="train",
    eval_split="eval",
    test_split="test",
    data_filter=None,
    exclude_direction=None,
    norm_training=False,
    pref_training=False,
    verifier_filter=False,
    feature_classifiers=None,
    feature_dataset=None,
    generate_embeddings=False,
    subreddits=None,
    use_textual_features=False,
    test_only=False,
    use_attribute_scores=False,
):
    if max_count_eval is None:
        max_count_eval = max_count
    if generate_embeddings:
        return_eval = False

    datasets = datasets.split(",")
    if len(datasets) == 1 and subreddits is not None and len(subreddits.split(",")) > 1:
        datasets = [f"{datasets[0]}/{subreddit}" for subreddit in subreddits.split(",")]

    eval_split = eval_split.split(",") if eval_split else []
    test_split = test_split.split(",") if test_split else []
    probabilities = list(map(float, probabilities.split(",")))
    if len(probabilities) != len(datasets): probabilities = None
    feature_classifiers = feature_classifiers.split(",") if feature_classifiers and feature_classifiers != '' else []

    train_data_list = []
    eval_data_list = {e_split: [] for e_split in eval_split}
    test_data_list = {t_split: [] for t_split in test_split}
    for i, dataset in enumerate(datasets):
        dataset = dataset.strip()
        strategy.print(f"dataset: {dataset}")

        data_dir = dataset.split("@")[1].strip() if "@" in dataset else None
        dataset = dataset.split("@")[0].strip()
        dataset_basename = os.path.basename(dataset)

        ext = os.path.splitext(dataset)[-1]
        if norm_training or pref_training:
            
            dataset_dict = {} if test_only else {train_split: f"{dataset}/{train_split}.jsonl"}
            if return_eval:
                for e_split in eval_split:
                    if not test_only: dataset_dict[e_split] = f"{dataset}/{e_split}.jsonl"
                for t_split in test_split:
                    dataset_dict[t_split] = f"{dataset}/{t_split}.jsonl"
            
            if not (pref_training and len(feature_classifiers) > 0 and feature_dataset):
                data = load_dataset("json", data_files=dataset_dict)

            elif pref_training and len(feature_classifiers) > 0 and feature_dataset:
                import pickle
                import pandas as pd
                import json
                import numpy as np

                data_ids = {}
                all_needed_ids, all_needed_ids_temporal = set(), set()
                for data_split_name, data_filepath in dataset_dict.items():
                    strategy.print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z%z')}] Loading input data for {data_split_name}: {data_filepath}")
                    with open(data_filepath, 'r') as f:
                        lines = [json.loads(line) for line in f.readlines()][:int(max_count_eval if data_split_name in eval_split or data_split_name in test_split else max_count)]
                    data_split = pd.DataFrame(data=lines).set_index("id", drop=False)
                    if "chosen_score" in data_split.columns: data_split = data_split.rename(columns={"chosen_score": "chosen_upvote", "rejected_score": "rejected_upvote"})
                    
                    index_list = data_split.index.tolist()
                    id_pairs = [sample_id.split("-")[2:4] for sample_id in index_list]
                    chosen_ids = [id_pair[0] for id_pair in id_pairs]
                    rejected_ids = [id_pair[1] for id_pair in id_pairs]

                    if "_2023" in data_split_name:
                        all_needed_ids_temporal.update(chosen_ids)
                        all_needed_ids_temporal.update(rejected_ids)
                    else:
                        all_needed_ids.update(chosen_ids)
                        all_needed_ids.update(rejected_ids)

                    data_ids[data_split_name] = (data_split, chosen_ids, rejected_ids, index_list)
                
                all_needed_ids = list(all_needed_ids)
                all_needed_ids_temporal = list(all_needed_ids_temporal)
                print(f"Num needed ids: {len(all_needed_ids)}")

                feature_dataset_dir = feature_dataset.replace("/__need_to_replace__/", f"/{dataset_basename}/")

                feature_data, feature_data_temporal = {}, {}
                for feature_classifier in feature_classifiers:
                    strategy.print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z%z')}] Loading feature data for {feature_classifier}: {os.path.join(feature_dataset_dir, f'{feature_classifier}.pkl')}")
                    with open(os.path.join(feature_dataset_dir, f"{feature_classifier}.pkl"), "rb") as f:
                        feature_data[feature_classifier] = pickle.load(f)
                        # only keep the data where the id is in the all_needed_ids; note that feature_data[feature_classifier] is a pd dataframe, just use select
                        feature_data[feature_classifier] = feature_data[feature_classifier].loc[all_needed_ids]
                    if "_2023" in ''.join(list(dataset_dict.keys())):
                        with open(os.path.join(feature_dataset_dir.replace("_2022", "_2023"), f"{feature_classifier}.pkl"), "rb") as f:
                            feature_data_temporal[feature_classifier] = pickle.load(f)
                            feature_data_temporal[feature_classifier] = feature_data_temporal[feature_classifier].loc[all_needed_ids_temporal]

                feature_data = pd.concat(feature_data, axis=1)
                if feature_data_temporal:
                    feature_data_temporal = pd.concat(feature_data_temporal, axis=1)

                data = {}
                for data_split_name, ids_info in data_ids.items():
                    data_split, chosen_ids, rejected_ids, index_list = ids_info
                    
                    if "_2023" in data_split_name:
                        chosen_scores = feature_data_temporal.loc[chosen_ids, (feature_classifiers, "score")]
                        rejected_scores = feature_data_temporal.loc[rejected_ids, (feature_classifiers, "score")]
                    else:
                        chosen_scores = feature_data.loc[chosen_ids, (feature_classifiers, "score")]
                        rejected_scores = feature_data.loc[rejected_ids, (feature_classifiers, "score")]
                    
                    
                    chosen_scores = pd.DataFrame({"chosen_score": list(np.stack(chosen_scores.to_numpy().tolist()))}, index=index_list) #data_split.index.tolist())
                    # chosen_scores.index = index_list

                    rejected_scores = pd.DataFrame({"rejected_score": list(np.stack(rejected_scores.to_numpy().tolist()))}, index=index_list) #data_split.index.tolist())
                    # rejected_scores.index = index_list

                    print(f"check for potential duplicate index, before dropping duplicates: {len(data_split.index.tolist())} samples")
                    data_split = data_split[~data_split.index.duplicated(keep="first")]
                    chosen_scores = chosen_scores[~chosen_scores.index.duplicated(keep="first")]
                    rejected_scores = rejected_scores[~rejected_scores.index.duplicated(keep="first")]
                    print(f"check for potential duplicate index, after dropping duplicates: {len(data_split.index.tolist())} samples")

                    if use_textual_features or use_attribute_scores:
                        data_combined = pd.concat([data_split, chosen_scores, rejected_scores], axis=1)
                        now = datetime.datetime.now()
                        data[data_split_name] = Dataset.from_pandas(data_combined)
                        time_taken = datetime.datetime.now() - now
                        strategy.print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z%z')}] Loaded {data_split_name} data in {time_taken.total_seconds()} seconds")
                        continue
                    
                    if "_2023" in data_split_name:
                        chosen_embeddings = feature_data_temporal.loc[chosen_ids, (feature_classifiers, "embedding")]
                        rejected_embeddings = feature_data_temporal.loc[rejected_ids, (feature_classifiers, "embedding")]
                    else:
                        chosen_embeddings = feature_data.loc[chosen_ids, (feature_classifiers, "embedding")]
                        rejected_embeddings = feature_data.loc[rejected_ids, (feature_classifiers, "embedding")]
                    
                    chosen_embeddings = pd.DataFrame({"chosen_embedding": [[row[col] for col in chosen_embeddings.columns] for _, row in chosen_embeddings.iterrows()]})
                    chosen_embeddings.index = index_list
                    chosen_embeddings = chosen_embeddings[~chosen_embeddings.index.duplicated(keep="first")]
                    
                    rejected_embeddings = pd.DataFrame({"rejected_embedding": [[row[col] for col in rejected_embeddings.columns] for _, row in rejected_embeddings.iterrows()]})
                    rejected_embeddings.index = index_list
                    rejected_embeddings = rejected_embeddings[~rejected_embeddings.index.duplicated(keep="first")]

                    data_combined = pd.concat([data_split, chosen_embeddings, rejected_embeddings, chosen_scores, rejected_scores], axis=1) #.drop(columns=['__index_level_0__'])
                    
                    # start timer
                    now = datetime.datetime.now()
                    data[data_split_name] = Dataset.from_pandas(data_combined)
                    time_taken = datetime.datetime.now() - now
                    strategy.print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z%z')}] Loaded {data_split_name} data in {time_taken.total_seconds()} seconds")


            strategy.print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z%z')}] Loaded {dataset} with data_files={dataset_dict}")

        elif generate_embeddings:
            data = {}
            train_file = os.path.join(dataset, f"{train_split}.jsonl")

            import json
            import pandas as pd
            with open(train_file, 'r') as f:
                lines = [json.loads(line) for line in f.readlines()][:int(max_count)]
            for line in lines:
                data[line['id'].split('-')[2]] = line["context"] + line["chosen"]
                data[line['id'].split('-')[3]] = line["context"] + line["rejected"]
            
            for e_split in eval_split:
                eval_file = os.path.join(dataset, f"{e_split}.jsonl")
                with open(eval_file, 'r') as f:
                    lines = [json.loads(line) for line in f.readlines()][:int(max_count)]
                for line in lines:
                    data[line['id'].split('-')[2]] = line["context"] + line["chosen"]
                    data[line['id'].split('-')[3]] = line["context"] + line["rejected"]
                    
            for t_split in test_split:
                eval_file = os.path.join(dataset, f"{t_split}.jsonl")
                with open(eval_file, 'r') as f:
                    lines = [json.loads(line) for line in f.readlines()][:int(max_count)]
                for line in lines:
                    data[line['id'].split('-')[2]] = line["context"] + line["chosen"]
                    data[line['id'].split('-')[3]] = line["context"] + line["rejected"]
                    
            data = [{"id": k, "message": v} for k, v in data.items()]
            data = Dataset.from_pandas(pd.DataFrame(data=data))
                    
        elif ext == ".py" or (
            os.path.isdir(dataset) and os.path.exists(os.path.join(dataset, f"{dataset_basename}.py"))
        ):
            data = load_dataset(dataset, trust_remote_code=True)
            strategy.print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z%z')}] loaded {dataset} with python script")
        # local text file
        elif ext in [".json", ".jsonl", ".csv"]:
            ext = ext.lower().strip(".")
            if ext == "jsonl":
                ext = "json"
            data = load_dataset(ext, data_files=dataset)
            strategy.print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z%z')}] loaded {dataset} with data_files={dataset}")
        # local dataset saved with `datasets.Dataset.save_to_disk`
        elif os.path.isdir(dataset):
            data = load_from_disk(dataset)
            strategy.print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z%z')}] loaded {dataset} from disk")
        # remote/local folder or common file
        else:
            data = load_dataset(dataset, data_dir=data_dir)
            strategy.print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z%z')}] loaded {dataset} from files")
        
        if data_filter: 
            strategy.print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z%z')}] dataset before {data_filter} filtering: {data}")
            data = data.filter(lambda example: data_filter in example['id'])
            strategy.print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z%z')}] dataset after {data_filter} filtering: {data}")
        if exclude_direction:
            strategy.print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z%z')}] dataset before excluding {exclude_direction} direction: {data}")
            data = data.filter(lambda example: exclude_direction not in example['id'])
            strategy.print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z%z')}] dataset after {exclude_direction} filtering: {data}")

        if not test_only:
            if train_split and train_split in data:
                train_data = data[train_split].select(range(min(max_count, len(data[train_split]))))
            else:
                train_data = data.select(range(min(max_count, len(data))))
            train_data_list.append(train_data)

        if return_eval:
            for e_split in eval_split:
                if e_split in data:
                    eval_data_list[e_split].append(data[e_split].select(range(min(max_count_eval, len(data[e_split])))))
            for t_split in test_split:
                if t_split in data:
                    test_data_list[t_split].append(data[t_split].select(range(min(max_count_eval, len(data[t_split])))))
    train_dataset = interleave_datasets(
        train_data_list,
        probabilities=probabilities,
        seed=seed,
        stopping_strategy=stopping_strategy,
    ) if len(train_data_list) > 0 else None
    strategy.print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z%z')}] interleaved train_dataset: {train_dataset}")

    if return_eval:
        eval_dataset = {e_split: interleave_datasets
                            (
                                eval_data_list[e_split],
                                probabilities=probabilities,
                                seed=seed,
                                stopping_strategy=stopping_strategy,
                            ) if len(eval_data_list[e_split]) > 0 else None
                            for e_split in eval_split}
        strategy.print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z%z')}] interleaved eval_dataset: {eval_dataset}")
        test_dataset = {t_split: interleave_datasets
                            (
                                test_data_list[t_split],
                                probabilities=probabilities,
                                seed=seed,
                                stopping_strategy=stopping_strategy,
                            ) if len(test_data_list[t_split]) > 0 else None
                            for t_split in test_split}
        strategy.print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z%z')}] interleaved test_dataset: {test_dataset}")
        return train_dataset, eval_dataset, test_dataset
    else:
        return train_dataset


def convert_token_to_id(token, tokenizer):
    if isinstance(token, str):
        token = tokenizer.encode(token, add_special_tokens=False)
        assert len(token) == 1
        return token[0]
    else:
        raise ValueError("token should be int or str")
