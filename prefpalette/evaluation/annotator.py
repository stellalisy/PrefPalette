import os
import yaml
import logging
from . import completions


class Annotator:
    def __init__(self, annotator_config_filepath, override_config=None):
        if not annotator_config_filepath.endswith(".yaml"):
            annotator_config_filepath = os.path.join(annotator_config_filepath, "configs.yaml")
        self.annotator_config = yaml.safe_load(open(annotator_config_filepath))
        self.annotator_name = list(self.annotator_config.keys())[0]
        self.annotator_config = self.annotator_config[self.annotator_name]

        if override_config:
            for k, v in override_config.items():
                if k in self.annotator_config:
                    self.annotator_config[k] = v
                if k in self.annotator_config.get("completions_kwargs", {}):
                    self.annotator_config["completions_kwargs"][k] = v

        annotator_dir = os.path.dirname(annotator_config_filepath)
        with open(os.path.join(annotator_dir, self.annotator_config["system_prompt"]), "r") as f:
            self.system_prompt = f.read()
        with open(os.path.join(annotator_dir, self.annotator_config["prompt_template"]), "r") as f:
            self.prompt_template = f.read()

        subreddit_template_path = os.path.join(annotator_dir, self.annotator_config["prompt_template"].replace(".txt", "_subreddit.txt"))
        if os.path.exists(subreddit_template_path):
            with open(subreddit_template_path, "r") as f:
                self.prompt_template_subreddit = f.read()
        else:
            self.prompt_template_subreddit = None

        self.self_consistency = self.annotator_config["self_consistency"]
        self.completions_kwargs = self.annotator_config["completions_kwargs"]
        self.chat_completions_fn = getattr(completions, self.annotator_config["fn_completions"])

    def annotate_pair(self, subreddit=None, context=None, comment_1=None, comment_2=None):
        if subreddit and self.prompt_template_subreddit:
            prompt = [
                self.prompt_template_subreddit.format(subredditname=subreddit, question=context, output_1=comment_1, output_2=comment_2),
                self.prompt_template_subreddit.format(subredditname=subreddit, question=context, output_1=comment_2, output_2=comment_1),
            ]
        else:
            prompt = [
                self.prompt_template.format(question=context, output_1=comment_1, output_2=comment_2),
                self.prompt_template.format(question=context, output_1=comment_2, output_2=comment_1),
            ]
        messages_batch = [[{"role": "system", "content": self.system_prompt}, {"role": "user", "content": p}] for p in prompt]
        messages_batch = messages_batch * self.self_consistency

        try:
            output_dict = self.chat_completions_fn(messages_batch, **self.completions_kwargs)
            decisions = []
            for i, response in enumerate(output_dict["completions"]):
                if response is None:
                    continue
                decision = response[-1]
                if i % 2 == 0:
                    decision = decision == "m"
                else:
                    decision = decision == "M"
                decisions.append(decision)

            score = sum(decisions) / len(decisions) if decisions else 0.5
            return dict(score=score, decisions=decisions, total_usage=output_dict["usage_total"])
        except Exception as e:
            logging.error(f"Annotation failed: {e}")
            return None
