# -*- coding: utf-8 -*-
# Stripped-down version: conversion + inference only (no training / deepspeed / fused_kl).

from __future__ import annotations
import math
import warnings
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.utils.deprecation import deprecate_kwarg

from distill_model.config_distilled_student import StudentConfig
import importlib

if torch.cuda.is_available():
    from fla.layers.attn import Attention as _FLAAttention
    from fla.models.utils import Cache
    from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss
    from fla.modules import RMSNorm
    from fla.modules.mlp import SwiGLULinear, swiglu
    from fla.modules.fused_kl_div import fused_kl_div_loss

    class Attention(_FLAAttention):
        """Extended FLA Attention that accepts explicit head_dim for models
        where hidden_size != num_heads * head_dim."""

        def __init__(self, hidden_size=2048, num_heads=32, num_kv_heads=None,
                     head_dim=None, **kwargs):
            super().__init__(hidden_size=hidden_size, num_heads=num_heads,
                             num_kv_heads=num_kv_heads, **kwargs)
            if head_dim is not None and head_dim != self.head_dim:
                self.head_dim = head_dim
                self.kv_dim = self.num_kv_heads * head_dim
                self.q_proj = nn.Linear(hidden_size, num_heads * head_dim,
                                        bias=kwargs.get('qkv_bias', False))
                self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * head_dim,
                                        bias=kwargs.get('qkv_bias', False))
                self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * head_dim,
                                        bias=kwargs.get('qkv_bias', False))
                self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
                if kwargs.get('qk_norm', False):
                    self.q_norm = nn.RMSNorm(head_dim)
                    self.k_norm = nn.RMSNorm(head_dim)

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

logger = logging.get_logger(__name__)


def get_student_attention_class(model_name: str):
    STUDENT_ATTENTION_MAP = {
        'path_v1': 'distill_model.student_layers.PaTHAttentionStudentV1',
        'path_fox_v1': 'distill_model.student_layers.PaTHFoXAttentionStudentV1',
        'fox_v1': 'distill_model.student_layers.PaTHFoXAttentionStudentV1',
        'gdn_v1': 'distill_model.student_layers.GatedDeltaNetStudentV1',
        'gdn_v2': 'distill_model.student_layers.GatedDeltaNetStudentV2',
        'gdn_v3': 'distill_model.student_layers.GatedDeltaNetStudentV3',
        'gsa_v1': 'distill_model.student_layers.GatedSlotAttentionStudentV1',
        'gla_v1': 'distill_model.student_layers.GatedLinearAttentionStudentV1',
        'gdn_v4': 'distill_model.student_layers.GatedDeltaNetStudentV4',
        'gdn_v4_no_silu': 'distill_model.student_layers.GatedDeltaNetStudentV4NoSilu',
        'gdn_v5': 'distill_model.student_layers.GatedDeltaNetStudentV5',
        'gdn_v6': 'distill_model.student_layers.GatedDeltaNetStudentV6',
        'swa_v1': 'distill_model.student_layers.SlidingWindowAttentionStudentV1',
    }
    if model_name not in STUDENT_ATTENTION_MAP:
        raise ValueError(f"Unknown student attention: {model_name}")
    module_path, class_name = STUDENT_ATTENTION_MAP[model_name].rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class StudentMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        hidden_ratio: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        hidden_act: str = 'swish',
        fuse_swiglu: bool = True
    ) -> StudentMLP:
        super().__init__()
        self.hidden_size = hidden_size
        if hidden_ratio is None:
            hidden_ratio = 4
        if intermediate_size is None:
            intermediate_size = int(hidden_size * hidden_ratio * 2 / 3)
            intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.fuse_swiglu = fuse_swiglu

        if hidden_act != 'swish':
            raise ValueError(f'Unsupported hidden_act: {hidden_act}')

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        if self.fuse_swiglu:
            self.swiglu_linear = SwiGLULinear()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        gate, y = self.gate_proj(x), self.up_proj(x)
        if self.fuse_swiglu:
            return self.swiglu_linear(
                gate, y,
                self.down_proj.weight,
                self.down_proj.bias if self.down_proj.bias is not None else None
            )
        else:
            return self.down_proj(swiglu(gate, y))


class StudentMoEMLP(nn.Module):
    """MoE MLP layer compatible with Qwen3-style sparse MoE."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([
            StudentMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                hidden_act=config.hidden_act,
                fuse_swiglu=False,
            )
            for _ in range(config.num_experts)
        ])

    def forward(self, hidden_states, **kwargs):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate(hidden_states_flat)
        routing_weights = F.softmax(router_logits, dim=1)
        topk_weights, topk_ids = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        final_hidden = torch.zeros_like(hidden_states_flat)
        expert_mask = F.one_hot(topk_ids, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.numel() == 0:
                continue
            current_hidden = hidden_states_flat[top_x]
            current_output = self.experts[expert_idx](current_hidden)
            final_hidden.index_add_(0, top_x, current_output * topk_weights[top_x, idx, None])

        return final_hidden.view(batch_size, seq_len, hidden_dim)


class StudentBlock(nn.Module):

    def __init__(self, config: StudentConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        head_dim = getattr(config, 'head_dim', None)
        attn_kwargs = dict(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=head_dim,
            qkv_bias=config.qkv_bias,
            qk_norm=config.qk_norm,
            window_size=config.window_size,
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
            layer_idx=layer_idx,
        )

        self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        if getattr(config, "force_window_on_all_layers", False):
            self.attn = Attention(**attn_kwargs)
        elif layer_idx in config.keep_full_attention_layers:
            self.attn = Attention(**attn_kwargs)
        else:
            student_attn_class = get_student_attention_class(config.student_name)
            self.attn = student_attn_class(config, layer_idx)

        self.mlp_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        if getattr(config, 'num_experts', None) is not None:
            self.mlp = StudentMoEMLP(config)
        else:
            self.mlp = StudentMLP(
                hidden_size=config.hidden_size,
                hidden_ratio=config.hidden_ratio,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                fuse_swiglu=config.fuse_swiglu
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs
        )
        if self.config.fuse_norm:
            hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states, **kwargs)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attentions,)
        if use_cache:
            outputs += (past_key_values,)
        return outputs


class StudentPreTrainedModel(PreTrainedModel):

    config_class = StudentConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['StudentBlock']
    _supports_cache_class = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(
        self,
        module: nn.Module,
        rescale_prenorm_residual: bool = False,
        num_residuals_per_layer: int = 2,
    ):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif hasattr(module, 'reset_parameters'):
            module.reset_parameters()

        if rescale_prenorm_residual:
            p = None
            if hasattr(module, 'o_proj'):
                p = module.o_proj.weight
            elif hasattr(module, 'down_proj'):
                p = module.down_proj.weight
            if p is not None:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)


class StudentModel(StudentPreTrainedModel):

    def __init__(self, config: StudentConfig) -> StudentModel:
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([StudentBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if output_attentions:
            warnings.warn("`StudentModel` does not support output attention weights now, setting to False.")
            output_attentions = False
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if use_cache and not isinstance(past_key_values, Cache):
            past_key_values = Cache.from_legacy_cache(past_key_values)

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing.")
                use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        next_cache = None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    **kwargs
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    **kwargs
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_attns
        )


class StudentForCausalLM(StudentPreTrainedModel, GenerationMixin):

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.model = StudentModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None
        self.use_zero3 = getattr(config, "use_zero3", False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        logits_to_keep: Optional[int] = None,
        **kwargs
    ):
        if past_key_values is not None and len(past_key_values) > 0:
            input_ids = input_ids[:, -1:]
        if inputs_embeds is not None and (past_key_values is None or len(past_key_values) == 0):
            model_inputs = {'inputs_embeds': inputs_embeds}
        else:
            model_inputs = {'input_ids': input_ids.contiguous()}

        if logits_to_keep is not None:
            model_inputs['logits_to_keep'] = logits_to_keep

        model_inputs.update({
            'past_key_values': past_key_values,
            'use_cache': use_cache,
            'attention_mask': attention_mask,
        })
        return model_inputs

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Optional[int] = 0,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states[:, -logits_to_keep:])

        loss = None
        if labels is not None:
            criterion = nn.CrossEntropyLoss()
            labels = labels.to(hidden_states.device)
            labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], criterion.ignore_index)), 1)
            loss = criterion(logits.view(labels.numel(), -1), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def forward_kl(
        self,
        teacher,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> torch.Tensor:
        output_student = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )[0].view(-1, self.config.hidden_size)

        with torch.no_grad():
            output_teacher = teacher.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                **kwargs
            )[0].view(-1, self.config.hidden_size)

        teacher_lm_head = teacher.lm_head.weight.data.clone()
        loss = fused_kl_div_loss(
            output_student, output_teacher,
            self.lm_head.weight, teacher_lm_head
        )
        return loss
