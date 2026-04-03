import os
from abc import ABC

import torch
from torch.optim import Optimizer
from tqdm import tqdm

from openrlhf.models import LogExpLoss, PairWiseLoss
from openrlhf.utils.distributed_sampler import DistributedSampler

import json
import random
import shutil
import numpy as np

class RewardModelTrainer(ABC):
    """
    Trainer for training a reward model.

    Args:
        model (torch.nn.Module): The model to be trained.
        strategy (Strategy): The training strategy to apply.
        optim (Optimizer): The optimizer to use during training.
        train_dataloader (DataLoader): The dataloader for the training dataset.
        eval_dataloader (DataLoader): The dataloader for the evaluation dataset.
        scheduler (Scheduler): The learning rate scheduler for dynamic adjustments during training.
        tokenizer (Tokenizer): The tokenizer for processing input text data.
        max_norm (float, defaults to 0.5): Maximum gradient norm for gradient clipping.
        max_epochs (int, defaults to 2): Maximum number of training epochs.
        loss (str, defaults to "sigmoid"): The loss function to use during training, e.g., "sigmoid".
    """

    def __init__(
        self,
        model,
        strategy,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
        scheduler,
        tokenizer,
        max_norm=0.5,
        max_epochs: int = 2,
        loss="sigmoid",
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.max_norm = max_norm
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.scheduler = scheduler
        self.optimizer = optim
        self.tokenizer = tokenizer
        self.args = strategy.args

        files = [f for f in os.listdir(self.args.ckpt_path) if "best_step" in f and f.endswith(".txt")] if os.path.exists(self.args.ckpt_path) else []
        if len(files) > 0:
            with open(os.path.join(self.args.ckpt_path, files[0]), "r") as f:
                self.best_model_eval_metric_val = float(f.read())
        else:
            self.best_model_eval_metric_val = -float("inf")

        if loss == "sigmoid":
            self.loss_fn = PairWiseLoss()
            self.strategy.print("LogSigmoid Loss")
        else:
            self.loss_fn = LogExpLoss()
            self.strategy.print("LogExp Loss")

        # Mixtral 8*7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        # packing samples
        self.packing_samples = strategy.args.packing_samples

        self.margin_loss = self.strategy.args.margin_loss
        self.compute_fp32_loss = self.strategy.args.compute_fp32_loss

        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if strategy.args.wandb_host: wandb.login(key=strategy.args.use_wandb, host=strategy.args.wandb_host)
            else: wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )
            
            if strategy.args.pref_training:
                wandb.config.update({"logging_config":
                            {
                                "subreddit": strategy.args.subreddit,
                                "base_model": strategy.args.pretrain,
                                "batch_size": strategy.args.train_batch_size,
                                "learning_rate": strategy.args.learning_rate,
                                "scheduler_name": strategy.args.scheduler_name,
                                "loss": loss,
                                "filter": strategy.args.verifier_filter,
                                "feature_group": strategy.args.feature_group,
                                "feature_classifiers": strategy.args.feature_classifiers,
                                "feature_dropout": strategy.args.feature_dropout,
                                "include_time": strategy.args.include_time,
                                "include_subreddit": strategy.args.include_subreddit,
                                "use_textual_features": strategy.args.use_textual_features,
                            }
                        })
            else:
                wandb.config.update({"logging_config":
                            {
                                "norm": strategy.args.data_dimension if strategy.args.data_dimension else strategy.args.gen_norm,
                                "subreddit": strategy.args.subreddit,
                                "base_model": strategy.args.pretrain,
                                "batch_size": strategy.args.train_batch_size,
                                "learning_rate": strategy.args.learning_rate,
                                "scheduler_name": strategy.args.scheduler_name,
                                "loss": loss,
                                "filter": strategy.args.verifier_filter,
                            }
                        })


            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

        # Initialize TensorBoard writer if wandb is not available
        if self.strategy.args.use_tensorboard and self._wandb is None and self.strategy.is_rank_0():
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.use_tensorboard, strategy.args.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)
    

    
    def generate_embedding(self, save_filepath, dataloader):
        os.makedirs(os.path.dirname(save_filepath), exist_ok=True)
        self.model.eval()
        
        progress_bar = tqdm(
            range(dataloader.__len__()),
            desc="Generating embeddings",
            disable=not self.strategy.is_rank_0(),
        )
        input_scores = []
        output_data = []
        with torch.no_grad():
            for data in dataloader:
                if not self.packing_samples:
                    raise NotImplementedError("generate_embedding is not implemented for non-packed samples.")

                _, inputs, attention_masks, infos = data
                packed_seq_lens = infos["input_length"]
                sample_ids = infos["id"]
                inputs = inputs.to(torch.cuda.current_device())
                attention_mask = attention_masks.to(torch.cuda.current_device())

                values, eos_hidden_states = self.model.generate_embedding(
                    inputs,
                    attention_mask=attention_mask,
                    ring_attn_group=self.strategy.ring_attn_group,
                    packed_seq_lens=packed_seq_lens,
                )

                input_embedding = eos_hidden_states.squeeze(0)
                input_score = values.squeeze(-1)

                input_scores.extend(input_score.tolist())

                out = [
                    {
                        "id": sample_id,
                        "score": input_score[i].item(),
                        "embedding": input_embedding[i].tolist(),
                    } for i, sample_id in enumerate(sample_ids)
                ]
                output_data.extend(out)

                if save_filepath.endswith(".jsonl"):
                    with open(save_filepath, "a+") as f:
                        for o in out:
                            json.dump(o, f)
                            f.write("\n")

                progress_bar.set_postfix({"score_avg": sum(input_scores) / len(input_scores)})
                progress_bar.update()

        if save_filepath.endswith(".pkl"):
            import pandas as pd
            df = pd.DataFrame(output_data).set_index("id")
            df.to_pickle(save_filepath)


    def fit(self, args, consumed_samples=0, num_update_steps_per_epoch=None):
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = num_update_steps_per_epoch  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt
        if args.test_steps == -1:
            args.test_steps = float("inf")  # do not run test during training

        # Restore step and start_epoch
        step = consumed_samples // args.train_batch_size * self.strategy.accumulated_gradient + 1
        start_epoch = consumed_samples // args.train_batch_size // num_update_steps_per_epoch
        consumed_samples = consumed_samples % (num_update_steps_per_epoch * args.train_batch_size)

        total_steps = self.epochs * num_update_steps_per_epoch

        epoch_bar = tqdm(range(start_epoch, self.epochs), desc="Train epoch", disable=not self.strategy.is_rank_0())
        for epoch in range(start_epoch, self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(
                    epoch, consumed_samples=0 if epoch > start_epoch else consumed_samples
                )

            #  train
            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            self.model.train()
            acc_mean = 0
            loss_mean = 0
            for data in self.train_dataloader:

                feature_dropout_prob = min(args.feature_dropout, step / (total_steps * 0.75))  # Reach 0.8 at 75% of training
                use_features = random.random() > feature_dropout_prob

                if not self.packing_samples:
                    chosen_ids, c_mask, reject_ids, r_mask, extra = data
                    margin = extra["margin"]
                    chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                    c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                    reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                    r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())

                    if extra["feature_classifier_data"] and use_features:
                        if args.use_attribute_scores:
                            chosen_feature_eos_hidden_states = torch.Tensor(extra["feature_classifier_data"]["chosen_score"]).to(torch.cuda.current_device())
                            reject_feature_eos_hidden_states = torch.Tensor(extra["feature_classifier_data"]["rejected_score"]).to(torch.cuda.current_device())
                        else:
                            chosen_feature_eos_hidden_states = torch.Tensor(extra["feature_classifier_data"]["chosen_embedding"]).to(torch.cuda.current_device())
                            reject_feature_eos_hidden_states = torch.Tensor(extra["feature_classifier_data"]["rejected_embedding"]).to(torch.cuda.current_device())
                    else:
                        chosen_feature_eos_hidden_states = None
                        reject_feature_eos_hidden_states = None

                    chosen_reward, reject_reward, aux_loss = self.concatenated_forward(
                        self.model, chosen_ids, c_mask, reject_ids, r_mask, chosen_feature_eos_hidden_states, reject_feature_eos_hidden_states
                    )
                else:
                    packed_input_ids, packed_attention_masks, packed_seq_lens, extra = data
                    margin = extra["margin"]
                    packed_input_ids, packed_attention_masks = packed_input_ids.to(
                        torch.cuda.current_device()
                    ), packed_attention_masks.to(torch.cuda.current_device())

                    if extra["feature_classifier_data"] and use_features:
                        if args.use_attribute_scores:
                            # chosen_feature_eos_hidden_states shape: num_features x per_device_batch_size x len(chosen_ids) x 1
                            chosen_feature_eos_hidden_states = torch.Tensor(extra["feature_classifier_data"]["chosen_score"]).to(torch.cuda.current_device())
                            reject_feature_eos_hidden_states = torch.Tensor(extra["feature_classifier_data"]["rejected_score"]).to(torch.cuda.current_device())
                        else:
                            # chosen_feature_eos_hidden_states shape: per_device_batch_size x num_features x hidden_size
                            chosen_feature_eos_hidden_states = torch.Tensor(extra["feature_classifier_data"]["chosen_embedding"]).to(torch.cuda.current_device())
                            reject_feature_eos_hidden_states = torch.Tensor(extra["feature_classifier_data"]["rejected_embedding"]).to(torch.cuda.current_device())
                    else:
                        chosen_feature_eos_hidden_states = None
                        reject_feature_eos_hidden_states = None

                    chosen_reward, reject_reward, aux_loss = self.packed_samples_forward(
                        self.model, packed_input_ids, packed_attention_masks, packed_seq_lens, chosen_feature_eos_hidden_states, reject_feature_eos_hidden_states
                    )

                    if step % 100 == 0:
                        for name, param in self.model.named_parameters():
                            if 'feature_attention' in name:
                                self.strategy.print(f"{name} requires_grad: {param.requires_grad}, has_grad: {param.grad is not None}")

                if self.margin_loss:
                    margin = torch.tensor(margin).to(torch.cuda.current_device())
                else:
                    margin = None

                # loss function
                if self.compute_fp32_loss:
                    chosen_reward = chosen_reward.float()
                    reject_reward = reject_reward.float()

                preference_loss = self.loss_fn(chosen_reward, reject_reward, margin)
                # mixtral
                if not self.aux_loss:
                    aux_loss = 0

                loss = preference_loss + aux_loss * self.args.aux_loss_coef

                self.strategy.backward(loss, self.model, self.optimizer)

                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                acc = (chosen_reward > reject_reward).float().mean().item()
                acc_mean = acc_mean * 0.9 + 0.1 * acc
                loss_mean = loss_mean * 0.9 + 0.1 * preference_loss.item()
                # optional rm info
                logs_dict = {
                    "loss": preference_loss.item(),
                    "acc": acc,
                    "chosen_reward": chosen_reward.mean().item(),
                    "reject_reward": reject_reward.mean().item(),
                    "loss_mean": loss_mean,
                    "acc_mean": acc_mean,
                    "lr": self.scheduler.get_last_lr()[0],
                }
                if self.aux_loss:
                    logs_dict["aux_loss"] = aux_loss.item()
                if extra["feature_classifier_data"] and use_features:
                    logs_dict["save_feature_ratio"] = self.model.save_feature_score
                    if self.strategy.is_rank_0():
                    
                        with open(os.path.join(args.ckpt_path, "feature_classifier.jsonl"), "a+") as f:
                            f.write(json.dumps({
                                "step": step,
                                "feature_attention_weights": np.array(self.model.save_attention_weights).reshape(np.array(self.model.save_attention_weights).shape[:2]).tolist(), # self.model.save_attention_weights,
                                "feature_ratio": self.model.save_feature_score,
                                }) + "\n")

                # step bar
                logs_dict = self.strategy.all_reduce(logs_dict)
                step_bar.set_postfix(logs_dict)
                
                # logs/checkpoints/evaluation
                if step % self.strategy.accumulated_gradient == 0:
                    global_step = step // self.strategy.accumulated_gradient
                    client_states = {"consumed_samples": global_step * args.train_batch_size}
                    self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict, client_states, has_feature=extra["feature_classifier_data"]!=None)
                
                step_bar.update()
                step += 1
            
            if isinstance(self.test_dataloader, dict):
                for key, test_dataloader in self.test_dataloader.items():
                    if len(test_dataloader) > 0:
                        self.evaluate(test_dataloader, global_step, log_key=key)
                        if extra["feature_classifier_data"]!=None:
                            self.evaluate(test_dataloader, global_step, log_key=key, use_features=True)
            else:
                if len(self.test_dataloader) > 0:
                    self.evaluate(self.test_dataloader, global_step)
                    if extra["feature_classifier_data"]!=None:
                        self.evaluate(self.test_dataloader, global_step, use_features=True)
            epoch_bar.update()

        if isinstance(self.test_dataloader, dict):
            for key, test_dataloader in self.test_dataloader.items():
                if len(test_dataloader) > 0:
                    self.evaluate(test_dataloader, "final", log_key=key)
                    if args.feature_group != "" and not args.use_textual_features:
                        self.evaluate(test_dataloader, "final", log_key=key, use_features=True)
        else:
            if len(self.test_dataloader) > 0:
                self.evaluate(self.test_dataloader, "final")
                if args.feature_group != "" and not args.use_textual_features:
                    self.evaluate(self.test_dataloader, "final", use_features=True)
        
        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

    # logs/checkpoints/evaluate
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}, has_feature=False):
        best_model_eval_metrics = {}

        if global_step % args.logging_steps == 0:
            # wandb
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)

        # eval
        if global_step % args.eval_steps == 0:
            best_model_eval_metrics = {}
            # do eval when len(dataloader) > 0, avoid zero division in eval.
            if isinstance(self.eval_dataloader, dict):
                for key, eval_dataloader in self.eval_dataloader.items():
                    if len(eval_dataloader) > 0:
                        best_model_eval_metrics[key] = self.evaluate(eval_dataloader, global_step, log_key=key)
                        if has_feature:
                            best_model_eval_metrics[f"{key}_feature"] = self.evaluate(eval_dataloader, global_step, log_key=key, use_features=True)
            else:
                if len(self.eval_dataloader) > 0:
                    best_model_eval_metrics["eval"] = self.evaluate(self.eval_dataloader, global_step)
                    if has_feature:
                        best_model_eval_metrics["eval_feature"] = self.evaluate(self.eval_dataloader, global_step, use_features=True)
        
        # test
        if global_step % args.test_steps == 0:
            if isinstance(self.test_dataloader, dict):
                for key, test_dataloader in self.test_dataloader.items():
                    if len(test_dataloader) > 0:
                        self.evaluate(test_dataloader, global_step, log_key=key)
                        if has_feature:
                            self.evaluate(test_dataloader, global_step, log_key=key, use_features=True)
            else:
                if len(self.test_dataloader) > 0:
                    self.evaluate(self.test_dataloader, global_step)
                    if has_feature:
                        self.evaluate(self.test_dataloader, global_step, use_features=True)

        # save ckpt
        # TODO: save best model on dev, use loss/perplexity on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self.strategy.save_ckpt(
                self.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem, client_states
            )
            # need a specified eval metric to decide best model -- currently should use eval acc_mean
            if self.strategy.is_rank_0() and self.best_model_eval_metric_val < best_model_eval_metrics.get(self.args.best_model_eval_split, -float("inf")):
                # find whether any folder within args.ckpt_path starts with "best"
                folders = [folder for folder in os.listdir(args.ckpt_path) if folder.startswith("best")]
                if len(folders) > 0:
                    # remove the folder
                    for folder in folders:
                        self.strategy.print(f"Removing old best model {folder}")
                        folder_path = os.path.join(args.ckpt_path, folder)
                        if os.path.isdir(folder_path):
                            shutil.rmtree(folder_path)
                        elif os.path.exists(folder_path):
                            os.remove(folder_path)
                # save the new best model
                self.strategy.print(f"Saving best model with {self.args.best_model_eval_metric}={best_model_eval_metrics.get(self.args.best_model_eval_split, -float('inf'))}")
                self.strategy.save_model(
                    self.model, self.tokenizer, os.path.join(args.ckpt_path, f"best_step{global_step}")
                )
                with open(os.path.join(args.ckpt_path, f"best_step{global_step}.txt"), "w") as f:
                    f.write(str(best_model_eval_metrics.get(self.args.best_model_eval_split, -float('inf'))))
                self.best_model_eval_metric_val = best_model_eval_metrics.get(self.args.best_model_eval_split, -float("inf"))
                


    def evaluate(self, eval_dataloader, steps=0, log_key="eval", use_features=False, write_output=None):
        step_bar_eval = tqdm(
            range(eval_dataloader.__len__()),
            desc=f"Eval stage of steps {steps}",
            disable=not self.strategy.is_rank_0(),
        )
        if use_features: log_key = f"{log_key}_usefeature"
        self.model.eval()
        with torch.no_grad():
            acc = 0
            rewards = []
            loss_sum = 0

            feature_ratios, feature_weights = [], []

            if write_output:
                from collections import defaultdict
                write_data = defaultdict(list)
            for data in eval_dataloader:

                if not self.packing_samples:
                    chosen_ids, c_mask, reject_ids, r_mask, extra = data
                    margin = extra["margin"] if isinstance(extra, dict) else extra
                    chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                    c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                    reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                    r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())

                    chosen_reward, reject_reward, _ = self.concatenated_forward(
                        self.model, chosen_ids, c_mask, reject_ids, r_mask
                    )
                else:
                    packed_input_ids, packed_attention_masks, packed_seq_lens, extra = data
                    margin = extra["margin"]
                    packed_input_ids, packed_attention_masks = packed_input_ids.to(
                        torch.cuda.current_device()
                    ), packed_attention_masks.to(torch.cuda.current_device())

                    if extra["feature_classifier_data"] and use_features:
                        # chosen_feature_eos_hidden_states shape: num_features x per_device_batch_size x hidden_size
                        if self.args.use_attribute_scores:
                            chosen_feature_eos_hidden_states = torch.Tensor(extra["feature_classifier_data"]["chosen_score"]).to(torch.cuda.current_device())
                            reject_feature_eos_hidden_states = torch.Tensor(extra["feature_classifier_data"]["rejected_score"]).to(torch.cuda.current_device())
                        else:
                            chosen_feature_eos_hidden_states = torch.Tensor(extra["feature_classifier_data"]["chosen_embedding"]).to(torch.cuda.current_device())
                            reject_feature_eos_hidden_states = torch.Tensor(extra["feature_classifier_data"]["rejected_embedding"]).to(torch.cuda.current_device())
                    else:
                        chosen_feature_eos_hidden_states = None
                        reject_feature_eos_hidden_states = None

                    chosen_reward, reject_reward, _ = self.packed_samples_forward(
                        self.model, packed_input_ids, packed_attention_masks, packed_seq_lens, chosen_feature_eos_hidden_states, reject_feature_eos_hidden_states
                    )

                if self.margin_loss:
                    margin = torch.tensor(margin).to(torch.cuda.current_device())
                else:
                    margin = None

                loss = self.loss_fn(chosen_reward, reject_reward, margin)

                rewards += [chosen_reward.flatten(), reject_reward.flatten()]
                acc += (chosen_reward > reject_reward).float().mean().item()
                loss_sum += loss.item()
                step_bar_eval.update()

                if write_output and self.packing_samples:
                    cr = chosen_reward.flatten().tolist()
                    rr = reject_reward.flatten().tolist()
                    write_data["log_key"].extend([log_key] * len(cr))
                    write_data["id"].extend(extra["id"])
                    write_data["accuracy"].extend([a > b for a,b in zip(cr, rr)])
                    write_data["chosen_reward"].extend(cr)
                    write_data["rejected_reward"].extend(rr)

                if self.packing_samples and extra.get("feature_classifier_data") and use_features:
                    feature_weights.append(np.array(self.model.save_attention_weights).reshape(np.array(self.model.save_attention_weights).shape[:2]))
                    feature_ratios.append(self.model.save_feature_score)

            acc_mean = acc / len(eval_dataloader)
            loss_mean = loss_sum / len(eval_dataloader)

            rewards = torch.cat(rewards).float()
            rewards = self.strategy.all_gather(rewards)
            reward_mean = torch.mean(rewards)
            reward_std = torch.std(rewards).clamp(min=1e-8)

            unwrap_model = self.strategy._unwrap_model(self.model)
            unwrap_model.config.mean = reward_mean.item()
            unwrap_model.config.std = reward_std.item()

            bar_dict = {
                "eval_loss": loss_mean,
                "acc_mean": acc_mean,
                "reward_mean": reward_mean.item(),
                "reward_std": reward_std.item(),
            }
            if len(feature_ratios) > 0:
                feature_ratios_ts = self.strategy.all_gather(feature_ratios)
                feature_ratios_mean = torch.mean(feature_ratios_ts)
                feature_ratios_std = torch.std(feature_ratios_ts).clamp(min=1e-8)
                bar_dict["feature_ratio_mean"] = feature_ratios_mean.item()
                bar_dict["feature_ratio_std"] = feature_ratios_std.item()

                feature_weights_ts = self.strategy.all_gather(np.vstack(feature_weights)).squeeze()
                feature_weights_mean = torch.mean(feature_weights_ts, dim=0)
                feature_weights_std = torch.std(feature_weights_ts, dim=0).clamp(min=1e-8)

            logs = self.strategy.all_reduce(bar_dict)
            step_bar_eval.set_postfix(logs)

            if self.strategy.is_rank_0():
                if self._wandb is not None:
                    logs = {"%s/%s" % (log_key, k): v for k, v in {**logs, "global_step": steps}.items()}
                    self._wandb.log(logs)
                elif self._tensorboard is not None:
                    for k, v in logs.items():
                        self._tensorboard.add_scalar(f"{log_key}/{k}", v, steps)
                if len(feature_ratios) > 0:
                    with open(os.path.join(self.strategy.args.ckpt_path, f"feature_classifier_{log_key}.jsonl"), "a+") as f:
                        f.write(json.dumps({
                            "step": steps,
                            "feature_attention_weights_mean": feature_weights_mean.tolist(),
                            "feature_attention_weights_std": feature_weights_std.tolist(),
                            "feature_ratio_mean": feature_ratios_mean.item(),
                            "feature_ratio": feature_ratios,
                            }) + "\n")
                if write_output:
                    os.makedirs(os.path.dirname(write_output), exist_ok=True)
                    with open(write_output, "a+") as f:
                        for i in range(len(write_data["id"])):
                            f.write(json.dumps({
                                k: v[i] for k,v in write_data.items()
                            }) + "\n")

        self.model.train()  # reset model state
        return bar_dict[self.args.best_model_eval_metric]

    def concatenated_forward(self, model, chosen_ids, c_mask, reject_ids, r_mask, chosen_feature_eos_hidden_states=None, reject_feature_eos_hidden_states=None):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.
        
        chosen_feature_eos_hidden_states shape: batch_size x num_features x len(chosen_ids) x hidden_size
        reject_feature_eos_hidden_states shape: batch_size x num_features x len(reject_ids) x hidden_size
        
        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        if chosen_feature_eos_hidden_states is not None and reject_feature_eos_hidden_states is not None:
            chosen_feature_eos_hidden_states = torch.sum(chosen_feature_eos_hidden_states, dim=1)/chosen_feature_eos_hidden_states.shape[1] # shape: batch_size x input_len x hidden_size
            reject_feature_eos_hidden_states = torch.sum(reject_feature_eos_hidden_states, dim=1)/reject_feature_eos_hidden_states.shape[1] # shape: batch_size x input_len x hidden_size
            # TODO: figure out dimensions for not packing samples and incorporate feature hidden states

        input_ids, att_masks = self.concatenated_inputs(chosen_ids, c_mask, reject_ids, r_mask)
        all_values, output = model(input_ids, attention_mask=att_masks, return_output=True)
        chosen_rewards = all_values[: chosen_ids.shape[0]]
        rejected_rewards = all_values[chosen_ids.shape[0] :]
        aux_loss = output.aux_loss if "aux_loss" in output else []
        return chosen_rewards, rejected_rewards, aux_loss

    def concatenated_inputs(self, chosen_ids, c_mask, reject_ids, r_mask):
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """

        def pad_to_length(tensor, length, pad_value, dim=-1):
            if tensor.size(dim) >= length:
                return tensor
            else:
                pad_size = list(tensor.shape)
                pad_size[dim] = length - tensor.size(dim)
                # left pad
                return torch.cat(
                    [pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device), tensor], dim=dim
                )

        max_length = max(chosen_ids.shape[1], reject_ids.shape[1])
        inputs_ids = torch.cat(
            (
                pad_to_length(chosen_ids, max_length, self.tokenizer.pad_token_id),
                pad_to_length(reject_ids, max_length, self.tokenizer.pad_token_id),
            ),
            dim=0,
        )
        max_length = max(c_mask.shape[1], r_mask.shape[1])
        att_masks = torch.cat((pad_to_length(c_mask, max_length, 0), pad_to_length(r_mask, max_length, 0)), dim=0)
        return inputs_ids, att_masks

    def packed_samples_forward(self, model, packed_input_ids, packed_attention_masks, packed_seq_lens, chosen_feature_eos_hidden_states=None, reject_feature_eos_hidden_states=None):
        """
        packed_input_ids shape: 1 x input_size x hidden_size
        per_device_batch_size = len(packed_seq_lens)//2
        feature_hidden_size should be the same with the hidden size of the model in most cases, but should also handle cases where they're different by learning a projection
        chosen_feature_eos_hidden_states shape: num_features x per_device_batch_size x feature_hidden_size
        reject_feature_eos_hidden_states shape: num_features x per_device_batch_size x feature_hidden_size
        feature_hidden_states shape: 1 x num_features x len(packed_seq_lens) x feature_hidden_size
        """
        if chosen_feature_eos_hidden_states is not None and reject_feature_eos_hidden_states is not None:
            if len(chosen_feature_eos_hidden_states.shape) == 2: chosen_feature_eos_hidden_states = chosen_feature_eos_hidden_states.unsqueeze(-1)
            if len(reject_feature_eos_hidden_states.shape) == 2: reject_feature_eos_hidden_states = reject_feature_eos_hidden_states.unsqueeze(-1)
            feature_hidden_states = torch.cat((chosen_feature_eos_hidden_states, reject_feature_eos_hidden_states), dim=0)
            if self.strategy.args.bf16:
                feature_hidden_states = feature_hidden_states.bfloat16()
        else:
            feature_hidden_states = None

        all_values, output = model(
            packed_input_ids,
            attention_mask=packed_attention_masks,
            return_output=True,
            ring_attn_group=self.strategy.ring_attn_group,
            packed_seq_lens=packed_seq_lens,
            feature_hidden_states=feature_hidden_states,
            context_attention=self.strategy.args.context_attention,
            features_only=self.strategy.args.features_only,
        )
        
        half_len = len(packed_seq_lens) // 2
        chosen_rewards = all_values[:half_len]
        rejected_rewards = all_values[half_len:]
        aux_loss = output.aux_loss if "aux_loss" in output else []

        return chosen_rewards, rejected_rewards, aux_loss
