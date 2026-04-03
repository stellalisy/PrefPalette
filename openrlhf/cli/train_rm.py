import argparse
import math
import os
from datetime import datetime

from transformers.trainer import get_scheduler

from openrlhf.datasets import RewardDataset, SFTDataset
from openrlhf.models import get_llm_for_sequence_regression
from openrlhf.trainer import RewardModelTrainer
from openrlhf.utils import blending_datasets, get_strategy, get_tokenizer


def train(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    if args.gen_norm:
        # step 1: load and prepare the best model

        model_best = get_llm_for_sequence_regression(
            args.save_path,
            "reward",
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules,
            lora_dropout=args.lora_dropout,
            ds_config=strategy.get_ds_train_config(is_actor=False),
            init_value_head=True,
            value_head_prefix=args.value_head_prefix,
            packing_samples=args.packing_samples,
            pref_training=args.pref_training,
            context_attention=args.context_attention,
            feature_classifiers=args.feature_classifiers,
            use_textual_features=args.use_textual_features,
            use_attribute_scores=args.use_attribute_scores,
            hidden_size_ratio=args.hidden_size_ratio,
        )
        tokenizer_best = get_tokenizer(args.pretrain, model_best, "left", strategy, use_fast=not args.disable_fast_tokenizer)

        strategy.print(model_best)
        model_best = strategy.prepare(model_best)

        train_data = blending_datasets(
            args.dataset,
            args.dataset_probs,
            strategy,
            args.seed,
            max_count=args.max_samples_per_dataset,
            stopping_strategy="all_exhausted",
            train_split=args.train_split,
            eval_split=args.eval_split,
            test_split=args.test_split,
            generate_embeddings=True,
        )
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z%z')}] len(train_data): {len(train_data)}")

        train_data = train_data.select(range(min(args.max_samples, len(train_data))))
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z%z')}] len(train_data) after sampling with args.max_samples ({args.max_samples}): {len(train_data)}")

        train_dataset = SFTDataset(
            train_data,
            tokenizer_best,
            args.max_len,
            strategy,
            input_template=args.input_template,
        )
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z%z')}] type(train_dataset): SFTDataset, len(train_dataset): {len(train_dataset)}")

        train_dataloader = strategy.setup_dataloader(
            train_dataset,
            args.micro_train_batch_size,
            True,
            True,
            train_dataset.packing_collate_fn if args.packing_samples else train_dataset.collate_fn,
            drop_last=False,
        )
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z%z')}] len(train_dataloader): {len(train_dataloader)}")

        trainer_best = RewardModelTrainer(
            model=model_best,
            strategy=strategy,
            optim=None,
            tokenizer=tokenizer_best,
            train_dataloader=train_dataloader,
            eval_dataloader=None,
            test_dataloader=None,
            scheduler=None,
            max_norm=args.max_norm,
            max_epochs=args.max_epochs,
            loss=args.loss,
        )
        
        trainer_best.generate_embedding(
            args.embedding_filepath,
            train_dataloader,
        )

        return
    

    model = get_llm_for_sequence_regression(
        args.pretrain,
        "reward",
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        ds_config=strategy.get_ds_train_config(is_actor=False),
        init_value_head=True,
        value_head_prefix=args.value_head_prefix,
        packing_samples=args.packing_samples,
        pref_training=args.pref_training,
        context_attention=args.context_attention,
        feature_classifiers=args.feature_classifiers,
        use_textual_features=args.use_textual_features,
        use_attribute_scores=args.use_attribute_scores,
        hidden_size_ratio=args.hidden_size_ratio,
    )
    

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model, "left", strategy, use_fast=not args.disable_fast_tokenizer)
    

    strategy.print(model)
    

    # configure optimizer
    # prepare for data and dataset
    train_data, eval_data, test_data = blending_datasets(
        args.dataset,
        args.dataset_probs,
        strategy,
        args.seed,
        max_count=args.max_samples_per_dataset,
        max_count_eval=args.max_samples_per_dataset_eval,
        stopping_strategy="all_exhausted",
        train_split=args.train_split,
        eval_split=args.eval_split,
        test_split=args.test_split,
        data_filter=args.data_dimension,
        exclude_direction=args.exclude_direction,
        norm_training=args.norm_training,
        pref_training=args.pref_training,
        verifier_filter=args.verifier_filter,
        feature_classifiers=args.feature_classifiers,
        feature_dataset=args.feature_dataset,
        subreddits=args.subreddit,
        use_textual_features=args.use_textual_features,
        use_attribute_scores=args.use_attribute_scores,
    )

    train_data = train_data.select(range(min(args.max_samples, len(train_data))))
    eval_data = {k: e_data.select(range(min(args.max_samples, len(e_data)))) for k, e_data in eval_data.items()}
    test_data = {k: t_data.select(range(min(args.max_samples, len(t_data)))) for k, t_data in test_data.items()}

    train_dataset = RewardDataset(
            train_data,
            tokenizer,
            args.max_len,
            strategy,
            input_template=args.input_template,
            multiple_of=args.ring_attn_size,
            feature_classifiers=args.feature_classifiers,
        ) if train_data is not None else None
    eval_dataset = {
        k: RewardDataset(
            e_data,
            tokenizer,
            args.max_len,
            strategy,
            input_template=args.input_template,
            multiple_of=args.ring_attn_size,
            feature_classifiers=args.feature_classifiers,
        ) if e_data is not None else None
        for k, e_data in eval_data.items()
    }
    test_dataset = {
        k: RewardDataset(
            t_data,
            tokenizer,
            args.max_len,
            strategy,
            input_template=args.input_template,
            multiple_of=args.ring_attn_size,
            feature_classifiers=args.feature_classifiers,
        ) if t_data is not None else None
        for k, t_data in test_data.items()
    }

    train_dataloader = strategy.setup_dataloader(
        train_dataset,
        args.micro_train_batch_size,
        True,
        True,
        train_dataset.packing_collate_fn if args.packing_samples else train_dataset.collate_fn,
        drop_last=False,
    ) if train_dataset is not None else None
    eval_dataloader = {
        k: strategy.setup_dataloader(
            e_dataset,
            args.micro_train_batch_size,
            True,
            False,
            e_dataset.packing_collate_fn if args.packing_samples else e_dataset.collate_fn,
            drop_last=False,
        ) if e_dataset is not None else None
        for k, e_dataset in eval_dataset.items()
    }
    test_dataloader = {
        k: strategy.setup_dataloader(
            t_dataset,
            args.micro_train_batch_size,
            True,
            False,
            t_dataset.packing_collate_fn if args.packing_samples else t_dataset.collate_fn,
            drop_last=False,
        ) if t_dataset is not None else None
        for k, t_dataset in test_dataset.items()
    }

    optim = strategy.create_optimizer(model, lr=args.learning_rate, betas=args.adam_betas, weight_decay=args.l2)
    
    # scheduler
    num_update_steps_per_epoch = len(train_dataset) // args.train_batch_size
    max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)

    scheduler = get_scheduler(
        args.scheduler_name,
        optim,
        num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
        num_training_steps=max_steps,
        scheduler_specific_kwargs={"min_lr": args.learning_rate * 0.1},
    )
    

    # gradient_checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )
    
    # strategy prepare
    (model, optim, scheduler) = strategy.prepare((model, optim, scheduler))
    
    # load checkpoint
    consumed_samples = 0
    if not args.embedding_filepath and args.load_checkpoint and os.path.exists(args.ckpt_path):
        ckpt_dirs = [d for d in os.listdir(args.ckpt_path) if d.startswith("global_step")] # or d.startswith("best_step")]
        if len(ckpt_dirs) > 0:
            _, states = strategy.load_ckpt(model, args.ckpt_path)
            consumed_samples = states["consumed_samples"]
            strategy.print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z%z')}] Loaded the checkpoint: {args.ckpt_path}, consumed_samples: {consumed_samples}")

    if args.only_save_best_ckpt:
        strategy.save_model(model, tokenizer, os.path.join(args.save_path, "best_final"))
        return

    # batch_size here is micro_batch_size * 2
    # we use merged chosen + rejected response forward
    trainer = RewardModelTrainer(
        model=model,
        strategy=strategy,
        optim=optim,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        test_dataloader=test_dataloader,
        scheduler=scheduler,
        max_norm=args.max_norm,
        max_epochs=args.max_epochs,
        loss=args.loss,
    )

    if args.test_only:
        if isinstance(test_dataloader, dict):
            for key, dataloader in test_dataloader.items():
                if len(dataloader) > 0:
                    trainer.evaluate(dataloader, log_key=key, write_output=args.test_output_filepath)
        else:
            if len(test_dataloader) > 0:
                trainer.evaluate(test_dataloader, log_key="test", write_output=args.test_output_filepath)
        return

    os.makedirs(args.save_path, exist_ok=True)
    trainer.fit(args, consumed_samples, num_update_steps_per_epoch)

    # Save value_head_prefix
    strategy.print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z%z')}] Save value_head_prefix in config")
    unwrap_model = strategy._unwrap_model(model)
    unwrap_model.config.value_head_prefix = args.value_head_prefix
    
    best_dirs = [d for d in os.listdir(args.ckpt_path) if d.startswith("best_step") and not d.endswith(".txt")]
    if len(best_dirs) > 0:
        best_steps = [int(d.replace("best_step", "").replace(".txt", "")) for d in best_dirs]
        best_max_step_idx = best_steps.index(max(best_steps))
        best_dir = best_dirs[best_max_step_idx]
        
        model = get_llm_for_sequence_regression(
            os.path.join(args.ckpt_path, best_dir),
            "reward",
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules,
            lora_dropout=args.lora_dropout,
            ds_config=strategy.get_ds_train_config(is_actor=False),
            init_value_head=True,
            value_head_prefix=args.value_head_prefix,
            packing_samples=args.packing_samples,
            pref_training=args.pref_training,
            context_attention=args.context_attention,
            feature_classifiers=args.feature_classifiers,
            use_textual_features=args.use_textual_features,
            use_attribute_scores=args.use_attribute_scores,
            hidden_size_ratio=args.hidden_size_ratio,
        )
    
    # save model checkpoint after fitting on only rank0
    strategy.save_model(model, tokenizer, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Checkpoint
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--test_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_rm")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=int(1e8))
    parser.add_argument("--load_checkpoint", action="store_true", default=False)

    # DeepSpeed
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--overlap_comm", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)

    # Models
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--value_head_prefix", type=str, default="score")

    # Context Parallel
    parser.add_argument("--ring_attn_size", type=int, default=1, help="Ring attention group size")
    parser.add_argument(
        "--ring_head_stride",
        type=int,
        default=1,
        help="the number of heads to do ring attention each time. "
        "It should be a divisor of the number of heads. "
        "A larger value may results in faster training but will consume more memory.",
    )

    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")

    # RM training
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument("--compute_fp32_loss", action="store_true", default=False)
    parser.add_argument("--margin_loss", action="store_true", default=False)
    parser.add_argument("--scheduler_name", type=str, default="cosine_with_min_lr")
    parser.add_argument("--learning_rate", type=float, default=9e-6)
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--micro_train_batch_size", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--loss", type=str, default="sigmoid")
    parser.add_argument("--l2", type=float, default=0.0, help="weight decay loss")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")

    # packing samples using Flash Attention2
    parser.add_argument("--packing_samples", action="store_true", default=False)

    # Custom dataset
    parser.add_argument("--subreddit", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset_probs", type=str, default="1.0", help="sampling probs for datasets")
    parser.add_argument("--prompt_key", type=str, default=None)
    parser.add_argument("--chosen_key", type=str, default="chosen")
    parser.add_argument("--rejected_key", type=str, default="rejected")
    parser.add_argument("--input_key", type=str, default="message")
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )
    parser.add_argument("--tokenizer_chat_template", type=str, default=None)
    parser.add_argument("--train_split", type=str, default="train", help="train split of the HF dataset")
    parser.add_argument("--eval_split", type=str, default="test", help="test split of the dataset")
    parser.add_argument("--test_split", type=str, default="test", help="test split of the dataset")
    parser.add_argument("--best_model_eval_metric", type=str, default="acc_mean")
    parser.add_argument("--best_model_eval_split", type=str, default="eval")
    parser.add_argument("--max_samples", type=int, default=int(1e8), help="Max number of samples")
    parser.add_argument("--max_samples_per_dataset", type=int, default=int(1e8), help="Max number of samples")
    parser.add_argument("--max_samples_per_dataset_eval", type=int, default=int(1e8), help="Max number of samples for eval dataset")
    parser.add_argument("--max_len", type=int, default=512)

    parser.add_argument("--verifier_filter", action="store_true", default=False) # use relativity verifier to filter
    parser.add_argument("--data_dimension", type=str, default=None) # focus, clarity, relevance, accuracy, avoidbias, answerability 
    parser.add_argument("--gen_norm", type=str, default=None) # focus, clarity, relevance, accuracy, avoidbias, answerability 
    parser.add_argument("--exclude_direction", type=str, default=None) # exclude direction in the dataset
    parser.add_argument("--norm_training", action="store_true", default=False) 
    parser.add_argument("--pref_training", action="store_true", default=False)
    parser.add_argument("--only_save_best_ckpt", action="store_true", default=False)
    parser.add_argument("--embedding_filepath", type=str, default=None)
    parser.add_argument("--feature_group", type=str, default="") # for logging only
    parser.add_argument("--feature_classifiers", type=str, default="")
    parser.add_argument("--feature_dataset", type=str, default=None)
    parser.add_argument("--feature_dropout", type=float, default=0.5)
    parser.add_argument("--include_time", action="store_true", default=False)
    parser.add_argument("--include_subreddit", action="store_true", default=False)
    parser.add_argument("--use_textual_features", action="store_true", default=False)
    parser.add_argument("--context_attention", action="store_true", default=False)
    parser.add_argument("--test_only", action="store_true", default=False)
    parser.add_argument("--test_output_filepath", type=str, default=None)
    parser.add_argument("--use_attribute_scores", action="store_true", default=False)
    parser.add_argument("--features_only", action="store_true", default=False)
    parser.add_argument("--hidden_size_ratio", type=float, default=0.25)
    
    # wandb parameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_host", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_rm")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="rm_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    # TensorBoard parameters
    parser.add_argument("--use_tensorboard", type=str, default=None, help="TensorBoard logging path")

    args = parser.parse_args()

    if args.input_template and "{}" not in args.input_template:
        print("[Warning] {} not in args.input_template, set to None")
        args.input_template = None

    if args.input_template and "\\n" in args.input_template:
        print(
            "[Warning] input_template contains \\n chracters instead of newline. "
            "You likely want to pass $'\\n' in Bash or \"`n\" in PowerShell."
        )

    if args.packing_samples and not args.flash_attn:
        print("[Warning] Please --flash_attn to accelerate when --packing_samples is enabled.")
        args.flash_attn = True

    if args.ring_attn_size > 1:
        assert args.packing_samples, "packing_samples must be enabled when using ring attention"

    train(args)
