from typing import Optional, Union

import deepspeed
import torch
import torch.nn as nn
try:
    from flash_attn.utils.distributed import all_gather
except ModuleNotFoundError:
    all_gather = None
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoConfig, AutoModel, BitsAndBytesConfig
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from openrlhf.utils.logging_utils import init_logger

import copy

from .ring_attn_utils import convert_ring_attn_params
from .utils import reset_position_ids

logger = init_logger(__name__)


# Construct transformer with a value head for sequence classification.
# https://github.com/huggingface/transformers/blob/405b56269812056d9593869e22b7b264d806cb1e/src/transformers/models/llama/modeling_llama.py#L1254
def get_llm_for_sequence_regression(
    model_name_or_path: str,
    model_type: str,
    *,
    bf16=True,
    load_in_4bit=False,
    lora_rank=0,
    lora_alpha=16,
    target_modules=None,
    lora_dropout=0,
    normalize_reward=False,
    use_flash_attention_2=False,
    ds_config: dict = None,
    init_value_head: bool = False,
    value_head_prefix="score",
    device_map=None,
    packing_samples=False,
    pref_training=False,
    context_attention=False,
    feature_classifiers="",
    use_textual_features=False,
    use_attribute_scores=False,
    hidden_size_ratio=0.25,
    **kwargs,
) -> nn.Module:
    """Retrieve a transformer model with a sequence regression head on top.

    This function loads a pretrained transformer model and attaches a linear layer for sequence regression.

    Args:
        model_name_or_path (str): Path to the pretrained model.
        model_type (str): Type of the model, either "reward" or "critic".
        bf16 (bool, optional): Enable bfloat16 precision. Defaults to True.
        load_in_4bit (bool, optional): Load the model in 4-bit precision. Defaults to False.
        lora_rank (int, optional): Rank for LoRA adaptation. Defaults to 0.
        lora_alpha (int, optional): Alpha parameter for LoRA. Defaults to 16.
        target_modules (list, optional): List of target modules for LoRA. Defaults to None.
        lora_dropout (float, optional): Dropout rate for LoRA layers. Defaults to 0.
        normalize_reward (bool, optional): Normalize reward values. Defaults to False.
        use_flash_attention_2 (bool, optional): Use Flash Attention 2.0. Defaults to False.
        ds_config (dict, optional): Deepspeed configuration for model partitioning across multiple GPUs when ZeRO-3 is enabled. Defaults to None.
        init_value_head (bool, optional): Initialize the value head. Defaults to False.
        value_head_prefix (str, optional): Prefix for the value head. Defaults to "score".
        device_map (dict, optional): Map of devices for model loading. Defaults to None.
        packing_samples (bool, optional): Whether to pack samples during training. Defaults to False.

    Returns:
        nn.Module: A pretrained transformer model with a sequence regression head.
    """
    assert (
        model_type == "critic" or model_type == "reward"
    ), f"invalid model_type: {model_type}, should be critic or reward."

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    config.normalize_reward = normalize_reward
    config._attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

    # Prioritize using the value_head_prefix in the model configuration.
    value_head_prefix = getattr(config, "value_head_prefix", value_head_prefix)
    logger.info(f"set value_head_prefix to `{value_head_prefix}`")

    base_class = AutoModel._model_mapping[type(config)]
    base_pretrained_class = base_class.__base__
    if model_type == "reward":
        cls_class = _get_reward_model(base_pretrained_class, base_class, value_head_prefix, packing_samples, pref_training, context_attention, feature_classifiers, use_textual_features, use_attribute_scores, hidden_size_ratio)
    else:
        cls_class = _get_critic_model(base_pretrained_class, base_class, value_head_prefix, packing_samples)

    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None

    if load_in_4bit:
        assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        nf4_config = None

    model = cls_class.from_pretrained(
        model_name_or_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        quantization_config=nf4_config,
        device_map=device_map,
        **kwargs,
    )

    # LoRA
    if lora_rank > 0:
        model.enable_input_require_grads()
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, lora_config)

        if load_in_4bit:
            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    module = module.to(torch.bfloat16)
                if "norm" in name:
                    module = module.to(torch.float32)
                if value_head_prefix in name or "embed_tokens" in name:
                    if hasattr(module, "weight"):
                        module = module.to(torch.bfloat16)

    # MoE - balancing loss
    model_config = model.config.to_dict()
    if "output_router_logits" in model_config:
        print("[MoE] set output_router_logits as True")
        model.config.output_router_logits = True

    # https://github.com/huggingface/transformers/issues/26877
    model.config.use_cache = False

    # NOTE: For reward model training only, intialize value_head manually
    # because deepspeed.zero.Init() will not intialize them.
    # TODO: Find a better way to clarify reward model training.
    if init_value_head:
        value_head = getattr(model, value_head_prefix)
        if dschf is not None:
            logger.info("initialize value_head for ZeRO-3 reward model training.")
            with deepspeed.zero.GatheredParameters([value_head.weight], modifier_rank=0):
                if torch.distributed.get_rank() == 0:
                    value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
        else:
            value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))

    return model


def _get_reward_model(base_pretrained_model, base_llm_model, value_head_prefix="score", packing_samples=False, pref_training=False, context_attention=False, feature_classifiers="", use_textual_features=False, use_attribute_scores=False, hidden_size_ratio=0.25):
    class RewardModel(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))

            self.value_head_prefix = value_head_prefix
            setattr(self, value_head_prefix, nn.Linear(config.hidden_size, 1, bias=False))

            self.packing_samples = packing_samples

            # mean std
            self.normalize_reward = config.normalize_reward
            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)

            if pref_training and not use_textual_features and feature_classifiers != "":
                self.hidden_size = getattr(self, self.value_head_prefix).in_features
                if use_attribute_scores: attribute_dimension = 1
                else: attribute_dimension = self.hidden_size

                if not context_attention:
                    self.feature_attention = nn.Sequential(
                        nn.Linear(attribute_dimension, int(self.hidden_size * hidden_size_ratio)),
                        nn.Tanh(),
                        nn.Linear(int(self.hidden_size * hidden_size_ratio), 1)
                    )
                else:
                    self.query_projection = nn.Linear(self.hidden_size, int(self.hidden_size * hidden_size_ratio))
                    self.feature_projection = nn.Linear(attribute_dimension, int(self.hidden_size * hidden_size_ratio))
                    self.attention_score = nn.Linear(int(self.hidden_size * hidden_size_ratio), 1)

                self.feature_score = nn.Parameter(torch.tensor(0.5))
                self.save_attention_weights, self.save_feature_score = None, None

            # load mean/std from config.json
            if hasattr(config, "mean"):
                self.mean[0] = config.mean
                self.std[0] = config.std

        def generate_embedding(
                self,
                input_ids,
                attention_mask=None,
                ring_attn_group=None,
                packed_seq_lens=None,
                ) -> torch.Tensor:
            if not self.packing_samples:
                raise NotImplementedError("generate_embedding is not implemented for non-packed samples.")
            if ring_attn_group is not None:
                input_ids, attention_mask, position_ids = convert_ring_attn_params(
                    input_ids, attention_mask, packed_seq_lens, ring_attn_group
                )
            else:
                position_ids = reset_position_ids(attention_mask)
            # explicitly ignore attention_mask for packing_samples
            attention_mask = None
            outputs = getattr(self, self.base_model_prefix)(
                input_ids, attention_mask=attention_mask, position_ids=position_ids
            )
            last_hidden_states = outputs["last_hidden_state"]
            packed_seq_lens = torch.tensor(packed_seq_lens, device=last_hidden_states.device)
            eos_indices = packed_seq_lens.cumsum(dim=0) - 1
            eos_hidden_states = last_hidden_states.gather(dim=1, index=eos_indices.unsqueeze(-1).unsqueeze(0).expand(last_hidden_states.shape[0], len(eos_indices), last_hidden_states.shape[-1]))
            values = getattr(self, self.value_head_prefix)(eos_hidden_states).squeeze(0)
            return values, eos_hidden_states


        def aggregate_feature_hiddens(self, feature_hidden_states):
            """
            Aggregates feature hidden states using attention mechanism
            
            Args:
                feature_hidden_states: Shape [1, batch_size*2, num_features, hidden_size] (for packing_samples)
                
            Returns:
                aggregated_feature: Shape [1, batch_size*2, hidden_size] (for packing_samples)
            """
            if not self.packing_samples:
                raise NotImplementedError("aggregate_feature_hiddens is not implemented for non-packed samples.")
            
            attention_scores = self.feature_attention(feature_hidden_states).squeeze(-1)  # [batch_size*2, num_features]
            
            attention_weights = torch.softmax(attention_scores, dim=1).unsqueeze(-1)  # [batch_size, num_features, 1]
            self.save_attention_weights = copy.deepcopy(attention_weights.tolist())

            # Apply attention weights to feature hidden states
            weighted_features = feature_hidden_states * attention_weights  # [batch_size, num_features, hidden_size]
            aggregated_feature = weighted_features.sum(dim=-2)  # [1, batch_size, hidden_size]
            
            return aggregated_feature

        def aggregate_feature_hiddens_context_attn(self, feature_hidden_states, eos_hidden_states):
            """
            Aggregates feature hidden states using attention mechanism
            
            Args:
                feature_hidden_states: Shape [1, batch_size*2, num_features, hidden_size] (for packing_samples)
                
            Returns:
                aggregated_feature: Shape [1, batch_size*2, hidden_size] (for packing_samples)
            """
            if not self.packing_samples:
                raise NotImplementedError("aggregate_feature_hiddens is not implemented for non-packed samples.")
            
            query = self.query_projection(eos_hidden_states) # [batch_size*2, hidden_size]
            keys = self.feature_projection(feature_hidden_states) # [batch_size*2, num_features, hidden_size]
            interaction = torch.tanh(query.unsqueeze(1) + keys) # [batch_size*2, num_features, hidden_size]
            attention_scores = self.attention_score(interaction).squeeze(-1)  # [batch_size, num_features]

            attention_weights = torch.softmax(attention_scores, dim=1).unsqueeze(-1)
            self.save_attention_weights = copy.deepcopy(attention_weights.tolist())

            weighted_features = feature_hidden_states * attention_weights
            aggregated_feature = weighted_features.sum(dim=1)
            
            return aggregated_feature


        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
            ring_attn_group=None,
            packed_seq_lens=None,
            feature_hidden_states=None, # [batch_size*2, num_features, hidden_size] (for packing_samples)
            context_attention=False,
            features_only=False,
        ) -> torch.Tensor:
            if not self.packing_samples:
                # https://github.com/OpenRLHF/OpenRLHF/issues/217
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
            else:
                # convert attention_mask to position_ids
                if ring_attn_group is not None:
                    input_ids, attention_mask, position_ids = convert_ring_attn_params(
                        input_ids, attention_mask, packed_seq_lens, ring_attn_group
                    )
                else:
                    position_ids = reset_position_ids(attention_mask)
                # explicitly ignore attention_mask for packing_samples
                attention_mask = None

            outputs = getattr(self, self.base_model_prefix)(
                input_ids, attention_mask=attention_mask, position_ids=position_ids
            )

            last_hidden_states = outputs["last_hidden_state"]
            
            if self.packing_samples:
                packed_seq_lens = torch.tensor(packed_seq_lens, device=last_hidden_states.device)
                eos_indices = packed_seq_lens.cumsum(dim=0) - 1
                # eos_hidden_states: torch.Size([1, 4, 2048]) == 1 x local_batch_size*2 x hidden_size of model
                eos_hidden_states = last_hidden_states.gather(dim=1, index=eos_indices.unsqueeze(-1).unsqueeze(0).expand(last_hidden_states.shape[0], len(eos_indices), last_hidden_states.shape[-1]))
                eos_hidden_states = eos_hidden_states.squeeze(0) # [batch_size*2, hidden_size]

                # feature_hidden_states shape: 1 x len(packed_seq_lens) x num_features x hidden_size
                if feature_hidden_states is not None:
                    if not context_attention:
                        feature_hidden_states = self.aggregate_feature_hiddens(feature_hidden_states) # len(packed_seq_lens) x num_features x hidden_size (for packing_samples)
                    else:
                        feature_hidden_states = self.aggregate_feature_hiddens_context_attn(feature_hidden_states, eos_hidden_states)
                    
                    if features_only:
                        eos_hidden_states = feature_hidden_states
                    else:
                        eos_hidden_states = eos_hidden_states + self.feature_score * feature_hidden_states
                    self.save_feature_score = copy.deepcopy(self.feature_score.tolist())

                values = getattr(self, self.value_head_prefix)(eos_hidden_states) # [batch_size*2, 1].squeeze(0) == [batch_size*2, 1]

                if ring_attn_group is not None:
                    reward = all_gather(values, ring_attn_group).reshape(1, -1)
                else:
                    reward = values
                
            else:
                values = getattr(self, self.value_head_prefix)(last_hidden_states).squeeze(-1)
                eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
                reward = values.gather(dim=1, index=eos_indices).squeeze(1)

            if not self.training and self.normalize_reward:
                reward = (reward - self.mean) / self.std

            return (reward, outputs) if return_output else reward

    return RewardModel


def _get_critic_model(base_pretrained_model, base_llm_model, value_head_prefix="score", packing_samples=False):
    class CriticModel(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))

            self.value_head_prefix = value_head_prefix
            setattr(self, value_head_prefix, nn.Linear(config.hidden_size, 1, bias=False))

            self.packing_samples = packing_samples

            # mean std
            self.normalize_reward = config.normalize_reward
            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)

            # load mean/std from config.json
            if hasattr(config, "mean"):
                self.mean[0] = config.mean
                self.std[0] = config.std

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            num_actions: Optional[Union[int, list[int]]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
            packed_seq_lens=None,
        ) -> torch.Tensor:
            if not self.packing_samples:
                # https://github.com/OpenRLHF/OpenRLHF/issues/217
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
            else:
                # convert attention_mask to position_ids
                position_ids = reset_position_ids(attention_mask)
                # explicitly ignore attention_mask for packing_samples
                attention_mask = None

            outputs = getattr(self, self.base_model_prefix)(
                input_ids, attention_mask=attention_mask, position_ids=position_ids
            )
            last_hidden_states = outputs["last_hidden_state"]
            values = getattr(self, self.value_head_prefix)(last_hidden_states).squeeze(-1)[:, :-1]

            # normalize reward
            if self.normalize_reward:
                values = (values - self.mean) / self.std

            if num_actions is None:
                assert return_output
                return outputs

            if not self.packing_samples:
                action_values = values[:, -num_actions:]
            else:
                assert isinstance(num_actions, list) and len(num_actions) == len(packed_seq_lens)
                action_values = []
                offset = 0
                for num_action, seq_len in zip(num_actions, packed_seq_lens):
                    start, end = max(0, offset + seq_len - num_action - 1), offset + seq_len - 1
                    action_values.append(values[:, start:end])
                    offset += seq_len
                action_values = torch.cat(action_values, dim=1)

            if return_output:
                return (action_values, outputs)
            else:
                return action_values

    return CriticModel
