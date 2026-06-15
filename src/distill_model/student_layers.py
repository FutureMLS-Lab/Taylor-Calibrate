# student_only_attention.py
# -*- coding: utf-8 -*-
# Copyright (c) 2025, Songlin Yang, Yanhong Li

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import logging
import sys
import os


def get_logger(name: str = None) -> logging.Logger:
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S"
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    if 'RANK' in os.environ and int(os.environ['RANK']) == 0:
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

    return logger
logger = get_logger(__name__)


from fla.layers.path_attn import PaTHAttention
class PaTHAttentionStudentV1(PaTHAttention):
    def __init__(self, config, layer_idx: int):
        super().__init__(hidden_size=config.hidden_size,
                         num_heads=config.num_heads,
                         num_kv_heads=config.num_kv_heads,
                         layer_idx=layer_idx,
                         use_w_shortconv=False,
                         use_low_rank_w=False,
                         )
    def init_from_teacher(self, teacher_attn):
        self.q_proj.weight.data.copy_(teacher_attn.q_proj.weight.data)
        self.k_proj.weight.data.copy_(teacher_attn.k_proj.weight.data)
        self.w_proj.weight.data.copy_(teacher_attn.w_proj.weight.data)
        self.v_proj.weight.data.copy_(teacher_attn.v_proj.weight.data)
        self.o_proj.weight.data.copy_(teacher_attn.o_proj.weight.data)

class PaTHFoXAttentionStudentV1(PaTHAttention):
    def __init__(self, config, layer_idx: int):
        super().__init__(hidden_size=config.hidden_size,
                         num_heads=config.num_heads,
                         num_kv_heads=config.num_kv_heads,
                         layer_idx=layer_idx,
                         use_w_shortconv=False,
                         use_low_rank_w=False,
                         use_forget_gate=True,
                         )
    def init_from_teacher(self, teacher_attn):
        self.q_proj.weight.data.copy_(teacher_attn.q_proj.weight.data)
        self.k_proj.weight.data.copy_(teacher_attn.k_proj.weight.data)
        self.w_proj.weight.data.copy_(teacher_attn.w_proj.weight.data)
        self.v_proj.weight.data.copy_(teacher_attn.v_proj.weight.data)
        self.o_proj.weight.data.copy_(teacher_attn.o_proj.weight.data)


from fla.layers.gated_deltanet import GatedDeltaNet
class GatedDeltaNetStudentV1(GatedDeltaNet):
    def __init__(self, config, layer_idx: int):
        super().__init__(hidden_size=config.hidden_size,
                         expand_v=1,
                         head_dim=getattr(config, 'head_dim', config.hidden_size // config.num_heads),
                         use_gate=False,
                         use_short_conv=True,
                         num_heads=config.num_heads,
                         layer_idx=layer_idx,
                         )

    def init_from_teacher(self, teacher_attn):
        k_weight = teacher_attn.k_proj.weight.data
        v_weight = teacher_attn.v_proj.weight.data

        k_weight_repeat = repeat(k_weight, 'h d -> (h g) d', g=teacher_attn.num_heads // teacher_attn.num_kv_heads)
        v_weight_repeat = repeat(v_weight, 'h d -> (h g) d', g=teacher_attn.num_heads // teacher_attn.num_kv_heads)

        self.q_proj.weight.data.copy_(teacher_attn.q_proj.weight.data)
        self.o_proj.weight.data.copy_(teacher_attn.o_proj.weight.data)
        self.k_proj.weight.data.copy_(k_weight_repeat)
        self.v_proj.weight.data.copy_(v_weight_repeat)
        logger.info(f"✅ GDN layer {self.layer_idx} init from teacher done.")


from fla.layers.gated_deltanet import GatedDeltaNet
class GatedDeltaNetStudentV2(GatedDeltaNet):
    def __init__(self, config, layer_idx: int):
        super().__init__(hidden_size=config.hidden_size,
                         expand_v=1,
                         head_dim=getattr(config, 'head_dim', config.hidden_size // config.num_heads),
                         use_gate=False,
                         use_short_conv=False,
                         num_heads=config.num_heads,
                         layer_idx=layer_idx,
                         )
        self.description = "Compared with GDN V1, this version does not use short conv"


    def init_from_teacher(self, teacher_attn):
        k_weight = teacher_attn.k_proj.weight.data
        v_weight = teacher_attn.v_proj.weight.data

        k_weight_repeat = repeat(k_weight, 'h d -> (h g) d', g=teacher_attn.num_heads // teacher_attn.num_kv_heads)
        v_weight_repeat = repeat(v_weight, 'h d -> (h g) d', g=teacher_attn.num_heads // teacher_attn.num_kv_heads)

        self.q_proj.weight.data.copy_(teacher_attn.q_proj.weight.data)
        self.o_proj.weight.data.copy_(teacher_attn.o_proj.weight.data)
        self.k_proj.weight.data.copy_(k_weight_repeat)
        self.v_proj.weight.data.copy_(v_weight_repeat)
        logger.info(f"✅ GDN V2 layer {self.layer_idx} init from teacher done.")


class GatedDeltaNetStudentV3(GatedDeltaNet):
    def __init__(self, config, layer_idx: int):
        super().__init__(hidden_size=config.hidden_size,
                         expand_v=1,
                         head_dim=getattr(config, 'head_dim', config.hidden_size // config.num_heads),
                         use_short_conv=True,
                         num_heads=config.num_heads,
                         layer_idx=layer_idx,
                         use_gate=True,
                         )
        self.description = "Compared with GDN V1, this version uses output gate and short conv"


    def init_from_teacher(self, teacher_attn):
        k_weight = teacher_attn.k_proj.weight.data
        v_weight = teacher_attn.v_proj.weight.data

        k_weight_repeat = repeat(k_weight, 'h d -> (h g) d', g=teacher_attn.num_heads // teacher_attn.num_kv_heads)
        v_weight_repeat = repeat(v_weight, 'h d -> (h g) d', g=teacher_attn.num_heads // teacher_attn.num_kv_heads)

        self.q_proj.weight.data.copy_(teacher_attn.q_proj.weight.data)
        self.o_proj.weight.data.copy_(teacher_attn.o_proj.weight.data)
        self.k_proj.weight.data.copy_(k_weight_repeat)
        self.v_proj.weight.data.copy_(v_weight_repeat)
        logger.info(f"✅ GDN V3 layer {self.layer_idx} init from teacher done.")



class GatedDeltaNetStudentV4(GatedDeltaNet):
    def __init__(self, config, layer_idx: int):
        super().__init__(hidden_size=config.hidden_size,
                         expand_v=1,
                         head_dim=getattr(config, 'head_dim', config.hidden_size // config.num_heads),
                         use_short_conv=False,
                         num_heads=config.num_heads,
                         layer_idx=layer_idx,
                         use_gate=True,
                         )
        self.description = "Compared with GDN V1, this version uses output gate and no short conv"

    def init_from_teacher(self, teacher_attn):
        k_weight = teacher_attn.k_proj.weight.data
        v_weight = teacher_attn.v_proj.weight.data
        k_weight_repeat = repeat(k_weight, 'h d -> (h g) d', g=teacher_attn.num_heads // teacher_attn.num_kv_heads)
        v_weight_repeat = repeat(v_weight, 'h d -> (h g) d', g=teacher_attn.num_heads // teacher_attn.num_kv_heads)
        self.q_proj.weight.data.copy_(teacher_attn.q_proj.weight.data)
        self.o_proj.weight.data.copy_(teacher_attn.o_proj.weight.data)
        self.k_proj.weight.data.copy_(k_weight_repeat)
        self.v_proj.weight.data.copy_(v_weight_repeat)
        logger.info(f"GDN V4 layer {self.layer_idx} init from teacher done.")


from fla.layers.gated_deltanet import (
    chunk_gated_delta_rule, fused_recurrent_gated_delta_rule,
)
from fla.layers.utils import get_layer_cache, get_unpad_data, index_first_axis, pad_input, update_layer_cache


class GatedDeltaNetStudentV4NoSilu(GatedDeltaNet):
    """V4 variant that removes SiLU from Q/K/V projections."""
    def __init__(self, config, layer_idx: int):
        super().__init__(hidden_size=config.hidden_size,
                         expand_v=1,
                         head_dim=getattr(config, 'head_dim', config.hidden_size // config.num_heads),
                         use_short_conv=False,
                         num_heads=config.num_heads,
                         layer_idx=layer_idx,
                         use_gate=True,
                         )

    def forward(self, hidden_states, attention_mask=None, past_key_values=None,
                use_cache=False, output_attentions=False, **kwargs):
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2

        batch_size, q_len, _ = hidden_states.shape
        mode = 'fused_recurrent' if (q_len <= 64 and not self.training) else self.mode
        if self.training:
            assert mode == 'chunk'

        last_state = get_layer_cache(self, past_key_values)
        cu_seqlens = kwargs.get('cu_seqlens')
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(
                rearrange(hidden_states, "b s ... -> (b s) ..."), indices
            ).unsqueeze(0)

        # NO SiLU — direct linear projection
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q, k = map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_k_dim), (q, k))
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)

        if self.num_v_heads > self.num_heads:
            q, k = map(lambda x: repeat(x, '... h d -> ... (h g) d', g=self.num_v_heads // self.num_heads), (q, k))

        beta = self.b_proj(hidden_states).sigmoid()
        if self.allow_neg_eigval:
            beta = beta * 2.
        g = -self.A_log.float().exp() * F.softplus(self.a_proj(hidden_states).float() + self.dt_bias)

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        if mode == 'chunk':
            o, recurrent_state = chunk_gated_delta_rule(
                q=q, k=k, v=v, g=g, beta=beta,
                initial_state=recurrent_state, output_final_state=use_cache,
                cu_seqlens=cu_seqlens, use_qk_l2norm_in_kernel=True,
            )
        elif mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_gated_delta_rule(
                q=q, k=k, v=v, g=g, beta=beta,
                initial_state=recurrent_state, output_final_state=use_cache,
                cu_seqlens=cu_seqlens, use_qk_l2norm_in_kernel=True,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        update_layer_cache(
            self, past_key_values, recurrent_state=recurrent_state,
            conv_state=None, offset=q_len,
        )

        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values

    def init_from_teacher(self, teacher_attn):
        k_weight = teacher_attn.k_proj.weight.data
        v_weight = teacher_attn.v_proj.weight.data
        k_weight_repeat = repeat(k_weight, 'h d -> (h g) d', g=teacher_attn.num_heads // teacher_attn.num_kv_heads)
        v_weight_repeat = repeat(v_weight, 'h d -> (h g) d', g=teacher_attn.num_heads // teacher_attn.num_kv_heads)
        self.q_proj.weight.data.copy_(teacher_attn.q_proj.weight.data)
        self.o_proj.weight.data.copy_(teacher_attn.o_proj.weight.data)
        self.k_proj.weight.data.copy_(k_weight_repeat)
        self.v_proj.weight.data.copy_(v_weight_repeat)
        logger.info(f"✅ GDN V4-NoSiLU layer {self.layer_idx} init from teacher done.")


from distill_model.custom_gdn import GatedDeltaNet_custom
class GatedDeltaNetStudentV6(GatedDeltaNet_custom):
    def __init__(self, config, layer_idx: int):
        super().__init__(hidden_size=config.hidden_size,
                         expand_v=1,
                         head_dim=getattr(config, 'head_dim', config.hidden_size // config.num_heads),
                         num_heads=config.num_heads,
                         layer_idx=layer_idx,
                         )
        self.description = "Compared with GDN V1, this version uses output gate and no short conv"



    def init_from_teacher(self, teacher_attn):
        k_weight = teacher_attn.k_proj.weight.data
        v_weight = teacher_attn.v_proj.weight.data

        k_weight_repeat = repeat(k_weight, 'h d -> (h g) d', g=teacher_attn.num_heads // teacher_attn.num_kv_heads)
        v_weight_repeat = repeat(v_weight, 'h d -> (h g) d', g=teacher_attn.num_heads // teacher_attn.num_kv_heads)

        self.q_proj.weight.data.copy_(teacher_attn.q_proj.weight.data)
        self.o_proj.weight.data.copy_(teacher_attn.o_proj.weight.data)
        self.k_proj.weight.data.copy_(k_weight_repeat)
        self.v_proj.weight.data.copy_(v_weight_repeat)
        logger.info(f"✅ GDN V6 layer {self.layer_idx} init from teacher done.")



class GatedDeltaNetStudentV5(GatedDeltaNet):
    def __init__(self, config, layer_idx: int):
        super().__init__(hidden_size=config.hidden_size,
                         expand_v=1,
                         head_dim=getattr(config, 'head_dim', config.hidden_size // config.num_heads),
                         use_short_conv=False,
                         num_heads=config.num_heads,
                         layer_idx=layer_idx,
                         use_gate=True,
                         )


        self.k_proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim),
            nn.Linear(config.num_kv_heads * self.head_dim, config.num_heads * self.head_dim),
        )
        self.v_proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim),
            nn.Linear(config.num_kv_heads * self.head_dim, config.num_heads * self.head_dim),
        )

    def init_from_teacher(self, teacher_attn):
        # Initialize the first layer of nn.Sequential for k_proj and v_proj
        # with the teacher's k_proj and v_proj weights, respectively.
        # The second layer can be left as default or initialized as needed.
        # Here, we copy the teacher's weights to the first layer of each Sequential.
        k_weight = teacher_attn.k_proj.weight.data
        v_weight = teacher_attn.v_proj.weight.data

        # Copy weights and biases to the first Linear layer in the Sequential
        self.k_proj[0].weight.data.copy_(k_weight)
        self.v_proj[0].weight.data.copy_(v_weight)

        self.q_proj.weight.data.copy_(teacher_attn.q_proj.weight.data)
        self.o_proj.weight.data.copy_(teacher_attn.o_proj.weight.data)
        logger.info(f"✅ GDN V5 layer {self.layer_idx} init from teacher done.")



from fla.layers.gsa import GatedSlotAttention
class GatedSlotAttentionStudentV1(GatedSlotAttention):
    def __init__(self, config, layer_idx: int):
        super().__init__(hidden_size=config.hidden_size,
                         num_heads=config.num_heads,
                         num_kv_heads=config.num_heads,
                         num_slots=64,
                         layer_idx=layer_idx,
                         )

    def init_from_teacher(self, teacher_attn):
        k_weight = teacher_attn.k_proj.weight.data
        v_weight = teacher_attn.v_proj.weight.data

        k_weight_repeat = repeat(k_weight, 'h d -> (h g) d', g=teacher_attn.num_heads // teacher_attn.num_kv_heads)
        v_weight_repeat = repeat(v_weight, 'h d -> (h g) d', g=teacher_attn.num_heads // teacher_attn.num_kv_heads)

        self.q_proj.weight.data.copy_(teacher_attn.q_proj.weight.data)
        self.o_proj.weight.data.copy_(teacher_attn.o_proj.weight.data)
        self.k_proj.weight.data.copy_(k_weight_repeat)
        self.v_proj.weight.data.copy_(v_weight_repeat)
        logger.info(f"✅ GSA V1 layer {self.layer_idx} init from teacher done.")



from fla.layers.gla import GatedLinearAttention
class GatedLinearAttentionStudentV1(GatedLinearAttention):
    def __init__(self, config, layer_idx: int):
        super().__init__(hidden_size=config.hidden_size,
                         expand_k=1,
                         expand_v=1,
                         num_heads=config.num_heads,
                         layer_idx=layer_idx,
                         )

    def init_from_teacher(self, teacher_attn):
        k_weight = teacher_attn.k_proj.weight.data
        v_weight = teacher_attn.v_proj.weight.data

        k_weight_repeat = repeat(k_weight, 'h d -> (h g) d', g=teacher_attn.num_heads // teacher_attn.num_kv_heads)
        v_weight_repeat = repeat(v_weight, 'h d -> (h g) d', g=teacher_attn.num_heads // teacher_attn.num_kv_heads)

        self.q_proj.weight.data.copy_(teacher_attn.q_proj.weight.data)
        self.o_proj.weight.data.copy_(teacher_attn.o_proj.weight.data)
        self.k_proj.weight.data.copy_(k_weight_repeat)
        self.v_proj.weight.data.copy_(v_weight_repeat)
        logger.info(f"✅ GLA V1 layer {self.layer_idx} init from teacher done.")

from fla.layers.attn import Attention
class SlidingWindowAttentionStudentV1(nn.Module):
    """
    Student attention that *is* FLA Attention with a fixed window size.
    """
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            qkv_bias=config.qkv_bias,
            qk_norm=config.qk_norm,
            window_size=config.window_size,  # <- sliding window
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
            layer_idx=layer_idx
        )

    def forward(self, hidden_states, attention_mask=None, past_key_values=None,
                use_cache=False, output_attentions=False, **kwargs):
        return self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs
        )

    @torch.no_grad()
    def init_from_teacher(self, teacher_attn):
        # copy Q,K,V,O projections if names match the teacher's Attention module
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            getattr(self.attn, name).weight.copy_(getattr(teacher_attn, name).weight)
            if getattr(self.attn, name).bias is not None and getattr(teacher_attn, name).bias is not None:
                getattr(self.attn, name).bias.copy_(getattr(teacher_attn, name).bias)

