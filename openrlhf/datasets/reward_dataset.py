from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import numpy as np
from datetime import datetime

from .utils import exist_and_not_none, zero_pad_sequences

dimension_pairs = {
    "verbosity": "concise-verbose",
    "formality": "casual-formal", 
    "supportiveness": "toxic-supportive", 
    "sarcasm": "genuine-sarcastic", 
    "humor": "serious-humorous", 
    "politeness": "rude-polite",
    "assertiveness": "passive-assertive",
    "empathy": "detached-empathetic",
    "directness": "indirect-direct"
}

def gen_text_features(scores, feature_classifiers):
    """
    given a list of scores for a list of features, convert the scores to likert scale description of the features
    """
    temp = "Below is a list of features and the degree to which the response exhibits each feature. Higher scores indicate a stronger presence of the feature:\n"
    feature_classifier_str = " | ".join([f"{feature}: {score}" for feature, score in zip(feature_classifiers, scores)])
    return temp + feature_classifier_str + " | "


def preprocess_data(
    data,
    input_template=None,
    prompt_key=None,
    chosen_key="chosen",
    rejected_key="rejected",
    apply_chat_template=None,
    is_dpo=False,
    feature_classifiers=None,
    include_time=False,
    include_subreddit=False,
    use_textual_features=False,
) -> str:
    if apply_chat_template:
        if prompt_key:
            if feature_classifiers is not None and len(feature_classifiers)>0 and use_textual_features:
                # chosen_score = [data[str(("chosen_score", feature))] for feature in feature_classifiers]
                # rejected_score = [data[str(("rejected_score", feature))] for feature in feature_classifiers]
                chosen_score = data["chosen_score"]
                rejected_score = data["rejected_score"]
                data[chosen_key][-1]["content"] = gen_text_features(chosen_score, feature_classifiers) + data[chosen_key][-1]["content"]
                data[rejected_key][-1]["content"] = gen_text_features(rejected_score, feature_classifiers) + data[rejected_key][-1]["content"]
            if include_time:
                data[prompt_key][0]["content"] = f"Post time: {datetime.fromtimestamp(data['post_time']).strftime('%A %B %d %Y %H:%M:%S %p')} | " + data[prompt_key][0]["content"]
                data[chosen_key][-1]["content"] = f"Response time: {datetime.fromtimestamp(data['chosen_time']).strftime('%A %B %d %Y %H:%M:%S %p')} | " + data[chosen_key][-1]["content"]
                data[rejected_key][-1]["content"] = f"Response time: {datetime.fromtimestamp(data['rejected_time']).strftime('%A %B %d %Y %H:%M:%S %p')} | " + data[rejected_key][-1]["content"]
            if include_subreddit:
                data[prompt_key][0]["content"] = f"Subreddit: r/{data['subreddit']} | " + data[prompt_key][0]["content"]
            prompt = apply_chat_template(data[prompt_key], tokenize=False, add_generation_prompt=True)
            chosen = apply_chat_template(data[prompt_key] + data[chosen_key], tokenize=False)[len(prompt) :]
            rejected = apply_chat_template(data[prompt_key] + data[rejected_key], tokenize=False)[len(prompt) :]
        else:
            prompt = ""
            chosen = apply_chat_template(data[chosen_key], tokenize=False)
            rejected = apply_chat_template(data[rejected_key], tokenize=False)

            if is_dpo:
                prompt = apply_chat_template(data[chosen_key][:-1], tokenize=False, add_generation_prompt=True)
                chosen = chosen[len(prompt) :]
                rejected = rejected[len(prompt) :]
    else:
        if prompt_key:
            prompt = data[prompt_key]
            if input_template:
                prompt = input_template.format(prompt)
        else:
            prompt = ""
        chosen = data[chosen_key]
        rejected = data[rejected_key]

    margin = data["margin"] if exist_and_not_none(data, "margin") else 0

    return prompt, chosen, rejected, {"margin": margin, "id": data["id"]}


class RewardDataset(Dataset):
    """
    Dataset for reward model

    Args:
        dataset: dataset for reward model
        self.tokenizer: self.tokenizer for reward model
        self.max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template=None,
        is_dpo=False,
        num_processors=8,
        multiple_of=1,
        feature_classifiers=None,
    ) -> None:
        super().__init__()
        self.is_dpo = is_dpo
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length
        self.multiple_of = multiple_of
        self.feature_classifiers = feature_classifiers.split(",") if feature_classifiers and feature_classifiers != '' else []

        # chat_template
        self.input_template = input_template
        self.prompt_key = getattr(self.strategy.args, "prompt_key", None)
        self.chosen_key = getattr(self.strategy.args, "chosen_key", None)
        self.rejected_key = getattr(self.strategy.args, "rejected_key", None)
        self.apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if self.apply_chat_template:
            self.apply_chat_template = self.tokenizer.apply_chat_template
            tokenizer_chat_template = getattr(self.strategy.args, "tokenizer_chat_template", None)
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template

        # Parallel loading datasets
        processed_dataset = dataset.map(
            self.process_data, remove_columns=dataset.column_names, num_proc=num_processors
        )

        # Filter out None values if necessary
        processed_dataset = processed_dataset.filter(lambda x: x["prompt"] is not None)

        # Store the processed data in class attributes
        self.prompts = processed_dataset["prompt"]
        self.chosens = processed_dataset["chosen"]
        self.rejects = processed_dataset["reject"]
        self.extras = processed_dataset["extra"]

        # add feature embeddings
        self.has_features = len(self.feature_classifiers) > 0 and not self.strategy.args.use_textual_features
        if self.has_features:
            if not self.strategy.args.use_attribute_scores:
                self.chosen_embeddings = dataset["chosen_embedding"]
                self.rejected_embeddings = dataset["rejected_embedding"]
            self.chosen_scores = dataset["chosen_score"]
            self.rejected_scores = dataset["rejected_score"]

    def process_data(self, data):
        prompt, chosen, reject, extra = preprocess_data(
            data,
            self.input_template,
            self.prompt_key,
            self.chosen_key,
            self.rejected_key,
            self.apply_chat_template,
            self.is_dpo,
            self.feature_classifiers,
            self.strategy.args.include_time,
            self.strategy.args.include_subreddit,
            self.strategy.args.use_textual_features,
        )

        if self.is_dpo:
            prompt_token = self.tokenizer(
                prompt,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            prompt_ids_len = prompt_token["attention_mask"].int().sum().item()

            # Filter the sample whose length is greater than max_length (2 for answer length)
            if prompt_ids_len >= self.max_length - 2:
                prompt = None
        else:
            prompt_ids_len = None

        return {
            "prompt": prompt,
            "chosen": chosen,
            "reject": reject,
            "extra": {
                "prompt_ids_len": prompt_ids_len,
                "margin": extra["margin"],
                # "feature_classifier_data": extra["feature_classifier_data"],
                "id": extra["id"],
            },
        }

    def __len__(self):
        length = len(self.chosens)
        return length

    def __getitem__(self, idx):
        prompt, chosen, reject, extra = self.prompts[idx], self.chosens[idx], self.rejects[idx], self.extras[idx]
        if self.has_features:
            extra["feature_classifier_data"] = {
                "chosen_score": self.chosen_scores[idx],
                "rejected_score": self.rejected_scores[idx],
            }
            if not self.strategy.args.use_attribute_scores:
                extra["feature_classifier_data"]["chosen_embedding"] = self.chosen_embeddings[idx]
                extra["feature_classifier_data"]["rejected_embedding"] = self.rejected_embeddings[idx]
        else:
            extra["feature_classifier_data"] = None

        chosen = (prompt + chosen).rstrip("\n")
        if not chosen.endswith(self.tokenizer.eos_token):
            chosen += " " + self.tokenizer.eos_token
        chosen_token = self.tokenizer(
            chosen,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )

        reject = (prompt + reject).rstrip("\n")
        if not reject.endswith(self.tokenizer.eos_token):
            reject += " " + self.tokenizer.eos_token
        reject_token = self.tokenizer(
            reject,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )

        # to avoid EOS_token truncation
        chosen_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        reject_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        chosen_token["attention_mask"][0][-1] = True
        reject_token["attention_mask"][0][-1] = True

        return (
            chosen_token["input_ids"],
            chosen_token["attention_mask"],
            reject_token["input_ids"],
            reject_token["attention_mask"],
            extra,
        )

    def collate_fn(self, item_list):
        chosen_ids = []
        chosen_masks = []
        reject_ids = []
        rejects_masks = []
        extras = []
        for chosen_id, chosen_mask, reject_id, rejects_mask, extra in item_list:
            chosen_ids.append(chosen_id)
            chosen_masks.append(chosen_mask)
            reject_ids.append(reject_id)
            rejects_masks.append(rejects_mask)
            extras.append(extra)

        if self.is_dpo:
            padding_side = "right"
        else:
            padding_side = "left"
        chosen_ids = zero_pad_sequences(chosen_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        chosen_masks = zero_pad_sequences(chosen_masks, side=padding_side)
        reject_ids = zero_pad_sequences(reject_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        rejects_masks = zero_pad_sequences(rejects_masks, side=padding_side)
        
        return chosen_ids, chosen_masks, reject_ids, rejects_masks, extras

    def packing_collate_fn(self, item_list):
        extras = {}

        chosen_ids = []
        chosen_att_masks = []
        chosen_seq_lens = []
        rejected_ids = []
        rejected_att_masks = []
        rejected_seq_lens = []
        index = 1
        
        for chosen_id, chosen_mask, reject_id, rejects_mask, extra in item_list:
            chosen_ids.append(chosen_id.flatten())
            chosen_att_masks.append(torch.full_like(chosen_id.flatten(), index))
            chosen_seq_lens.append(len(chosen_id.flatten()))
            for k, v in extra.items():
                if k == "feature_classifier_data":
                    if extra["feature_classifier_data"] == None:
                        extras["feature_classifier_data"] = None
                    else:
                        if k not in extras: extras[k] = {}
                        for kk, vv in v.items():
                            if kk not in extras[k]: extras[k][kk] = []
                            extras[k][kk].append(vv)
                else:
                    if k not in extras: extras[k] = []
                    extras[k].append(v)

            rejected_ids.append(reject_id.flatten())
            rejected_att_masks.append(torch.full_like(reject_id.flatten(), index + len(item_list)))
            rejected_seq_lens.append(len(reject_id.flatten()))
            index += 1

        packed_input_ids = torch.cat(chosen_ids + rejected_ids, dim=0).unsqueeze(0)
        packed_attention_masks = torch.cat(chosen_att_masks + rejected_att_masks, dim=0).unsqueeze(0)
        packed_seq_lens = chosen_seq_lens + rejected_seq_lens

        if extras["feature_classifier_data"] != None:
            for k, v in extras["feature_classifier_data"].items():
                extras["feature_classifier_data"][k] = np.array(v)

        if self.multiple_of > 1 and packed_input_ids.numel() % self.multiple_of != 0:
            padding_len = self.multiple_of - (packed_input_ids.numel() % self.multiple_of)
            packed_input_ids = F.pad(packed_input_ids, (0, padding_len), value=self.tokenizer.pad_token_id)
            packed_attention_masks = F.pad(packed_attention_masks, (0, padding_len), value=0)

        return packed_input_ids, packed_attention_masks, packed_seq_lens, extras
