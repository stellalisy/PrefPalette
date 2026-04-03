import os
from abc import ABC

import torch
import torch.distributed as dist
try:
    from flash_attn.utils.distributed import all_gather
except ModuleNotFoundError:
    all_gather = None
from torch.optim import Optimizer
from torch.nn import functional as F
from tqdm import tqdm

from openrlhf.models import GPTLMLoss
from openrlhf.utils.distributed_sampler import DistributedSampler

import pandas as pd
import random

class SFTTrainer(ABC):
    """
    Trainer for supervised fine-tuning (SFT).

    Args:
        model (torch.nn.Module): The model to be trained.
        strategy (Strategy): The training strategy to be applied.
        optim (Optimizer): The optimizer for model training.
        train_dataloader (DataLoader): The dataloader for the training dataset.
        eval_dataloader (DataLoader): The dataloader for the evaluation dataset.
        scheduler (Scheduler): The learning rate scheduler to adjust training rates.
        max_norm (float, defaults to 1): Maximum gradient norm for clipping to prevent exploding gradients.
        pretrain_mode (bool, defaults to False): Flag to indicate if the trainer is in pre-training mode.
        batch_size (int, defaults to 1): Batch size for training.
        max_epochs (int, defaults to 2): The maximum number of training epochs.
        tokenizer (Tokenizer, optional): The tokenizer for processing input data.
    """

    def __init__(
        self,
        model,
        strategy,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        max_norm: float = 1,
        pretrain_mode: bool = False,
        batch_size: int = 1,
        max_epochs: int = 2,
        tokenizer=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.batch_size = batch_size
        self.max_norm = max_norm
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.pretrain_mode = pretrain_mode
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optim
        self.args = strategy.args
        self.text_gen_df = None

        self.loss_fn = GPTLMLoss(ring_attn_group=self.strategy.ring_attn_group)

        # Mixtral 8*7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        # packing samples
        self.packing_samples = strategy.args.packing_samples

        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
        self.strategy.print(f"self.strategy.args.use_wandb: {self.strategy.args.use_wandb}")
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            self.strategy.print(f"initializing wandb: org: {self.strategy.args.wandb_org}, project: {self.strategy.args.wandb_project}, group: {self.strategy.args.wandb_group}, run_name: {self.strategy.args.wandb_run_name}")
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

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

    def fit(self, args, consumed_samples=0, num_update_steps_per_epoch=None):
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = num_update_steps_per_epoch  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        # Restore step and start_epoch
        step = consumed_samples // args.train_batch_size * self.strategy.accumulated_gradient + 1
        start_epoch = consumed_samples // args.train_batch_size // num_update_steps_per_epoch
        consumed_samples = consumed_samples % (num_update_steps_per_epoch * args.train_batch_size)

        epoch_bar = tqdm(
            range(start_epoch, self.epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )
        for epoch in range(start_epoch, self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(
                    epoch, consumed_samples=0 if epoch > start_epoch else consumed_samples
                )

            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            # train
            self.model.train()
            loss_mean = 0
            for prompt_id_lens, inputs, attention_masks, infos in self.train_dataloader:
                if self.packing_samples:
                    inputs = inputs.to(torch.cuda.current_device())
                    attention_mask = attention_masks.to(torch.cuda.current_device())
                else:
                    inputs = inputs.to(torch.cuda.current_device()).squeeze(1)
                    attention_mask = attention_masks.to(torch.cuda.current_device()).squeeze(1)
                
                if self.strategy.ring_attn_group is None:
                    output = self.model(inputs, attention_mask=attention_mask, return_output=True)
                else:
                    output = self.model(
                        inputs, 
                        attention_mask=attention_mask, 
                        return_output=True,
                        ring_attn_group=self.strategy.ring_attn_group,
                        packed_seq_lens=infos["input_length"],
                    )

                # loss function
                labels = torch.where(
                    attention_mask.bool(),
                    inputs,
                    self.loss_fn.IGNORE_INDEX,
                )
                # mixtral
                if self.aux_loss:
                    aux_loss = output.aux_loss
                else:
                    aux_loss = 0

                if not self.pretrain_mode:
                    if self.packing_samples:
                        index = 0
                        for input_length, source_len in zip(infos["input_length"], prompt_id_lens):
                            labels[0][index : index + source_len] = self.loss_fn.IGNORE_INDEX
                            index += input_length
                    else:
                        for label, source_len in zip(labels, prompt_id_lens):
                            label[:source_len] = self.loss_fn.IGNORE_INDEX

                gpt_loss = self.loss_fn(output.logits, labels)
                loss = gpt_loss + aux_loss * self.args.aux_loss_coef
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                loss_mean = loss_mean * 0.9 + 0.1 * gpt_loss.item()
                logs_dict = {
                    "gpt_loss": gpt_loss.item(),
                    "loss_mean": loss_mean,
                    "lr": self.scheduler.get_last_lr()[0],
                }
                if self.aux_loss:
                    logs_dict["aux_loss"] = aux_loss.item()
                # step bar
                logs_dict = self.strategy.all_reduce(logs_dict)
                step_bar.set_postfix(logs_dict)
                step_bar.update()

                # logs/checkpoints/evaluation
                if step % self.strategy.accumulated_gradient == 0:
                    global_step = step // self.strategy.accumulated_gradient
                    client_states = {"consumed_samples": global_step * args.train_batch_size}
                    self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict, client_states)

                step += 1

            epoch_bar.update()

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

    # logs/checkpoints/evaluation
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
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
            # do eval when len(dataloader) > 0, avoid zero division in eval.
            if len(self.eval_dataloader) > 0:
                self.evaluate(self.eval_dataloader, global_step)
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self.strategy.save_ckpt(
                self.model.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem, client_states
            )
            print(f"Saved checkpoint to {args.ckpt_path} with tag {tag}")

    def evaluate(self, eval_dataloader, steps=0):
        times = 0
        self.model.eval()
        with torch.no_grad():
            loss_sum = 0
            step_bar = tqdm(
                range(eval_dataloader.__len__()),
                desc="Eval stage of steps %d" % steps,
                disable=not self.strategy.is_rank_0(),
            )
            
            random_samples_generation = []
            for prompt_id_lens, inputs, attention_masks, infos in eval_dataloader:
                if self.packing_samples:
                    inputs = inputs.to(torch.cuda.current_device())
                    attention_mask = attention_masks.to(torch.cuda.current_device())
                    prompt_ids = [inputs[0].split(infos["input_length"])[i][:prompt_id_lens[i]] for i in range(len(prompt_id_lens))]
                else:
                    inputs = inputs.to(torch.cuda.current_device()).squeeze(1)
                    attention_mask = attention_masks.to(torch.cuda.current_device()).squeeze(1)
                    prompt_ids = [inputs[i, :prompt_id_lens[i]] for i in range(len(prompt_id_lens))]
                random_samples_generation.extend(prompt_ids)
                if len(random_samples_generation) > 10: random_samples_generation = random.choices(random_samples_generation, k=10)

                if self.strategy.ring_attn_group is None:
                    output = self.model(inputs, attention_mask=attention_mask, return_output=True)
                else:
                    output = self.model(
                        inputs, 
                        attention_mask=attention_mask, 
                        return_output=True,
                        ring_attn_group=self.strategy.ring_attn_group,
                        packed_seq_lens=infos["input_length"],
                    )
                
                # loss function
                labels = torch.where(
                    attention_mask.bool(),
                    inputs,
                    self.loss_fn.IGNORE_INDEX,
                )

                # for i, prompt_id_len in enumerate(prompt_id_lens):
                #     input_ids = inputs[i, :prompt_id_len].unsqueeze(0)
                #     completion_output = self.model.model.generate(input_ids, max_new_tokens=self.args.max_new_tokens, attention_mask=torch.Tensor(input_ids.shape).fill_(1).to(input_ids.device).long(), pad_token_id=self.tokenizer.pad_token_id)
                #     decoded_output = self.tokenizer.decode(completion_output[0], skip_special_tokens=True)
                #     decoded_input = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                #     generations["step"].append(steps)
                #     generations["input"].append(decoded_input)
                #     generations["output"].append(decoded_output)

                if not self.pretrain_mode:
                    if self.packing_samples:
                        index = 0
                        for input_length, source_len in zip(infos["input_length"], prompt_id_lens):
                            labels[0][index : index + source_len] = self.loss_fn.IGNORE_INDEX
                            index += input_length
                    else:
                        for label, source_len in zip(labels, prompt_id_lens):
                            label[:source_len] = self.loss_fn.IGNORE_INDEX

                loss = self.loss_fn(output.logits, labels)

                times += 1
                loss_sum += loss.item()
                bar_dict = {"eval gpt_loss": loss_sum / times}
                step_bar.update()
                logs = self.strategy.all_reduce(bar_dict)
                step_bar.set_postfix(logs)

            # input_texts_sample, output_texts_sample = [], []
            # for inputs, input_lengths, prompt_id_lens in random_samples_generation:
            #     sliced_inputs = self.prepare_inputs_for_generation(inputs, input_lengths, prompt_id_lens)
            #     sequences, decoded_texts = self.generate_completions(sliced_inputs, prompt_id_lens)
            #     input_texts = self.tokenizer.batch_decode([sequences[i][:prompt_id_lens[i]] for i in range(len(prompt_id_lens))], skip_special_tokens=False)
            #     input_texts_sample.extend(input_texts)
            #     output_texts_sample.extend(decoded_texts)
            # df = pd.DataFrame({"step": [steps]*len(input_texts_sample), "input_text": input_texts_sample, "generation": output_texts_sample})
            # df = pd.DataFrame(generations)
            # if self.text_gen_df is None: self.text_gen_df = df
            # else: self.text_gen_df = pd.concat([self.text_gen_df, df], ignore_index=True)
            if self.strategy.stage != 3 and not self.args.pretrain_mode:
                generations = {"step": [], "input": [], "output": []}
                for input_ids in random_samples_generation:
                    decoded_input = self.tokenizer.decode(input_ids, skip_special_tokens=False)
                    _, decoded_output = self.generate_completions([input_ids])
                    generations["step"].append(steps)
                    generations["input"].append(decoded_input)
                    generations["output"].append(decoded_output)
                df = pd.DataFrame(generations)
                if self.text_gen_df is None: self.text_gen_df = df
                else: self.text_gen_df = pd.concat([self.text_gen_df, df], ignore_index=True)

            if self.strategy.is_rank_0():
                if self._wandb is not None:
                    logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                    if self.strategy.stage != 3 and not self.args.pretrain_mode:logs["eval/completions"] = self._wandb.Table(dataframe=self.text_gen_df)
                    self._wandb.log(logs)
                elif self._tensorboard is not None:
                    for k, v in logs.items():
                        self._tensorboard.add_scalar(f"eval/{k}", v, steps)
        self.model.train()  # reset model state

    def generate_completions(
        self,
        inputs: torch.Tensor, # list of 1d vector
    ):
        """
        Generate token completions from self.model in a batched manner until the
        stop_token_id is reached or max_new_tokens is generated.

        Args:
            input_ids (torch.Tensor): The initial input token ids of shape [batch_size, seq_len].
            attention_mask (torch.Tensor, optional): The attention mask of shape [batch_size, seq_len].
                If None, an attention mask of all 1s will be created automatically.
            max_new_tokens (int, optional): The maximum number of tokens to generate.
            stop_token_id (int, optional): The token id that indicates end of generation.

        Returns:
            torch.Tensor: The generated sequences of shape [batch_size, seq_len + generated_tokens].
        """
        device = inputs[0].device
        batch_size = len(inputs)
        prompt_id_lens = [len(seq) for seq in inputs]
        seq_len = max(prompt_id_lens)

        total_seq_len = seq_len + self.args.max_new_tokens

        # Clone the input so we can append new tokens to 'sequences'
        # sequences = input_ids.clone()
        sequences = torch.full(
            (batch_size, total_seq_len),
            fill_value=self.tokenizer.pad_token_id,
            dtype=inputs[0].dtype,
            device=device,
        )
        # Build the attention mask: 1 for real tokens, 0 for pad
        attention_mask = torch.zeros(
            (batch_size, total_seq_len),
            dtype=torch.long,
            device=device,
        )
        # Copy in only the valid prompt tokens for each row
        for i in range(batch_size):
            valid_len_i = prompt_id_lens[i]
            sequences[i, :valid_len_i] = inputs[i][:valid_len_i]
            attention_mask[i, :valid_len_i] = 1
        
        # Track where each example will write its next token
        # (i.e., the current "effective" length, ignoring trailing pad)
        current_lens = torch.tensor(prompt_id_lens, device=device, dtype=torch.long)

        # Keep track of which sequences have finished generation (hit the stop token).
        finished = [False] * batch_size

        for step in range(self.args.max_new_tokens):
            # Model forward pass
            if self.strategy.ring_attn_group is None:
                output = self.model(
                    sequences,
                    attention_mask=attention_mask,
                    return_output=True
                )
            else:
                # Example for computing "packed_seq_lens" if needed:
                # This is typically the (non-padded) length of each sequence.
                packed_seq_lens = attention_mask.sum(dim=1).tolist()
                
                output = self.model(
                    sequences,
                    attention_mask=attention_mask,
                    return_output=True,
                    ring_attn_group=self.strategy.ring_attn_group,
                    packed_seq_lens=packed_seq_lens,
                )

            # `output.logits` should be of shape [batch_size, current_seq_len, vocab_size]
            # We want the distribution over the next token for each batch element.
            # output.logits: [batch_size, total_seq_len, vocab_size]
            logits = output.logits  # shape = [B, T, V]

            # ----------------------------------
            #  1) Gather the "next token" logits
            #     at the *real* last position
            #     for each example
            # ----------------------------------
            # For each i, the real "last token" is at index (current_lens[i] - 1)
            gather_indices = (current_lens - 1).view(-1, 1, 1)  # shape [B, 1, 1]
            gather_indices = gather_indices.expand(-1, 1, logits.size(-1))  # [B, 1, V]
            next_logits = logits.gather(1, gather_indices)  # [B, 1, V]
            next_logits = next_logits.squeeze(1)  # [B, V]
            
            # Greedy pick
            next_tokens = torch.argmax(next_logits, dim=-1)  # [B]
            
            # ----------------------------------
            #  2) Place the new token for each
            #     example at the correct position
            # ----------------------------------
            for i in range(batch_size):
                if not finished[i]:
                    # Write next token in position `current_lens[i]`
                    seq_pos = current_lens[i].item()
                    sequences[i, seq_pos] = next_tokens[i]
                    # Update the mask
                    attention_mask[i, seq_pos] = 1

                    # Check for stop token
                    if next_tokens[i].item() == self.tokenizer.eos_token_id:
                        finished[i] = True
                    else:
                        current_lens[i] += 1  # Increase length by 1

            # Break early if all sequences have encountered the stop token
            if all(finished):
                break
        
        decoded_texts = self.tokenizer.batch_decode([sequences[i][prompt_id_lens[i]:] for i in range(batch_size)], skip_special_tokens=False)
        return sequences, decoded_texts
