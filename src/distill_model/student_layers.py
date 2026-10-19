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


def build_swa_sink_mask(
    seq_len: int,
    window_size: int,
    num_sink_tokens: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the causal SWA mask with a fixed prefix sink.

    ``window_size`` is the total number of visible positions: the first
    ``num_sink_tokens`` keys plus the latest
    ``window_size - num_sink_tokens`` local keys.
    """
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if num_sink_tokens < 0 or num_sink_tokens >= window_size:
        raise ValueError(
            "num_sink_tokens must satisfy 0 <= num_sink_tokens < window_size, "
            f"got {num_sink_tokens} and {window_size}"
        )

    query = torch.arange(seq_len, device=device)[:, None]
    key = torch.arange(seq_len, device=device)[None, :]
    causal = key <= query
    local_width = window_size - num_sink_tokens
    local = key >= (query - local_width + 1)
    sink = key < num_sink_tokens
    return causal & (local | sink)


class SlidingWindowSinkAttentionStudentV1(nn.Module):
    """SWA with four persistent prefix sinks and an exact shared softmax.

    For a total window ``w`` and ``s`` sinks, positions ``i >= w`` attend to
    keys ``[0, s)`` and the latest ``w-s`` causal keys.  The two disjoint
    groups are evaluated with FlashAttention and merged using their returned
    log-sum-exp normalizers, avoiding a dense ``T x T`` attention matrix.

    Training uses ``use_cache=False``.  Generation retains the full KV history
    for correctness and selects the sink/local union at each decode step.
    Sink-aware cache eviction is intentionally deferred because the app-to-app
    comparison makes no serving-speed or memory claim.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.window_size = int(config.window_size)
        self.num_sink_tokens = int(getattr(config, "num_sink_tokens", 0))
        if self.num_sink_tokens <= 0:
            raise ValueError("swa_sink_v1 requires num_sink_tokens > 0")
        build_swa_sink_mask(
            seq_len=1,
            window_size=self.window_size,
            num_sink_tokens=self.num_sink_tokens,
        )

        self.attn = Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            qkv_bias=config.qkv_bias,
            qk_norm=config.qk_norm,
            window_size=self.window_size,
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
            layer_idx=layer_idx,
        )

    @staticmethod
    def _normalize_lse(lse: torch.Tensor, q_len: int) -> torch.Tensor:
        """Convert FlashAttention LSE to ``[batch, query, heads, 1]``."""
        if lse.ndim != 3:
            raise RuntimeError(f"Unexpected FlashAttention LSE shape: {tuple(lse.shape)}")
        if lse.shape[-1] == q_len:
            # flash-attn returns [batch, heads, query]
            lse = lse.transpose(1, 2)
        elif lse.shape[1] != q_len:
            raise RuntimeError(f"Unexpected FlashAttention LSE shape: {tuple(lse.shape)}")
        return lse.unsqueeze(-1)

    @staticmethod
    def _flash_with_lse(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool,
        window_size: tuple[int, int] = (-1, -1),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from flash_attn import flash_attn_func

        result = flash_attn_func(
            q,
            k,
            v,
            dropout_p=0.0,
            causal=causal,
            window_size=window_size,
            return_attn_probs=True,
        )
        if not isinstance(result, tuple) or len(result) < 2:
            raise RuntimeError(
                "Installed flash-attn must support return_attn_probs=True "
                "and return (output, softmax_lse, ...)"
            )
        return result[0], result[1]

    def _sink_window_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        from flash_attn import flash_attn_func

        q_len = q.shape[1]
        if q_len <= self.window_size:
            return flash_attn_func(q, k, v, dropout_p=0.0, causal=True)

        prefix_len = self.window_size
        prefix = flash_attn_func(
            q[:, :prefix_len],
            k[:, :prefix_len],
            v[:, :prefix_len],
            dropout_p=0.0,
            causal=True,
        )

        q_suffix = q[:, prefix_len:]
        local_width = self.window_size - self.num_sink_tokens
        local_out, local_lse = self._flash_with_lse(
            q_suffix,
            k,
            v,
            causal=True,
            window_size=(local_width - 1, 0),
        )
        sink_out, sink_lse = self._flash_with_lse(
            q_suffix,
            k[:, :self.num_sink_tokens],
            v[:, :self.num_sink_tokens],
            causal=False,
        )

        suffix_len = q_suffix.shape[1]
        local_lse = self._normalize_lse(local_lse, suffix_len)
        sink_lse = self._normalize_lse(sink_lse, suffix_len)
        total_lse = torch.logaddexp(local_lse, sink_lse)
        local_weight = torch.exp(local_lse - total_lse).to(local_out.dtype)
        sink_weight = torch.exp(sink_lse - total_lse).to(sink_out.dtype)
        suffix = local_weight * local_out + sink_weight * sink_out
        return torch.cat((prefix, suffix), dim=1)

    def _project_qkv(
        self,
        hidden_states: torch.Tensor,
        seqlen_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_len = hidden_states.shape[1]
        q = rearrange(
            self.attn.q_proj(hidden_states),
            "... (h d) -> ... h d",
            d=self.attn.head_dim,
        )
        k = rearrange(
            self.attn.k_proj(hidden_states),
            "... (h d) -> ... h d",
            d=self.attn.head_dim,
        )
        v = rearrange(
            self.attn.v_proj(hidden_states),
            "... (h d) -> ... h d",
            d=self.attn.head_dim,
        )
        if self.attn.qk_norm:
            q, k = self.attn.q_norm(q), self.attn.k_norm(k)

        max_seqlen = max(
            q_len + seqlen_offset,
            self.attn.max_position_embeddings or q_len,
        )
        q, k = self.attn.rotary(
            q,
            k,
            seqlen_offset=seqlen_offset,
            max_seqlen=max_seqlen,
            cu_seqlens=None,
        )
        return q, k, v

    def _forward_unpadded(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, q_len, _ = hidden_states.shape
        q, k, v = self._project_qkv(hidden_states)
        output = self._sink_window_attention(q, k, v)
        output = output.reshape(batch_size, q_len, -1)
        return self.attn.o_proj(output)

    def _forward_with_cache(self, hidden_states, past_key_values):
        from flash_attn import flash_attn_func

        batch_size, q_len, _ = hidden_states.shape
        seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
        if seqlen_offset and q_len != 1:
            raise NotImplementedError(
                "swa_sink_v1 cache supports a full prefill followed by single-token decode"
            )

        q, k, v = self._project_qkv(hidden_states, seqlen_offset=seqlen_offset)
        state = past_key_values.update(
            attn_state=(k.flatten(-2, -1), v.flatten(-2, -1)),
            layer_idx=self.layer_idx,
            offset=q_len,
            # Preserve the full history for quality-only evaluation. A serving
            # implementation should evict to sinks + local tokens.
            cache_kwargs={"window_size": None},
        )["attn_state"]
        cached_k, cached_v = state
        cached_k = rearrange(cached_k, "... (h d) -> ... h d", d=self.attn.head_dim)
        cached_v = rearrange(cached_v, "... (h d) -> ... h d", d=self.attn.head_dim)

        if seqlen_offset == 0:
            output = self._sink_window_attention(q, cached_k, cached_v)
        else:
            if cached_k.shape[1] > self.window_size:
                local_width = self.window_size - self.num_sink_tokens
                cached_k = torch.cat(
                    (cached_k[:, :self.num_sink_tokens], cached_k[:, -local_width:]),
                    dim=1,
                )
                cached_v = torch.cat(
                    (cached_v[:, :self.num_sink_tokens], cached_v[:, -local_width:]),
                    dim=1,
                )
            output = flash_attn_func(
                q,
                cached_k,
                cached_v,
                dropout_p=0.0,
                causal=False,
            )

        output = output.reshape(batch_size, q_len, -1)
        return self.attn.o_proj(output)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        output_attentions=False,
        **kwargs,
    ):
        if kwargs.get("cu_seqlens") is not None:
            raise NotImplementedError("Packed sequences are not supported by swa_sink_v1")
        if attention_mask is not None and attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape [batch, sequence]")

        if use_cache:
            if past_key_values is None:
                raise ValueError("use_cache=True requires an initialized FLA cache")
            if attention_mask is not None and not bool(torch.all(attention_mask)):
                raise NotImplementedError(
                    "swa_sink_v1 cached generation does not support padded batches"
                )
            output = self._forward_with_cache(hidden_states, past_key_values)
        elif attention_mask is None or bool(torch.all(attention_mask)):
            output = self._forward_unpadded(hidden_states)
        else:
            # Evaluation may left-pad variable-length examples.  Process each
            # contiguous valid span independently; Stage-1/2 fixed blocks use
            # the fast all-valid path above.
            output = torch.zeros_like(hidden_states)
            for batch_idx in range(hidden_states.shape[0]):
                valid = torch.nonzero(attention_mask[batch_idx], as_tuple=False).flatten()
                if valid.numel() == 0:
                    continue
                start, stop = int(valid[0]), int(valid[-1]) + 1
                expected = torch.arange(start, stop, device=valid.device)
                if not torch.equal(valid, expected):
                    raise ValueError("swa_sink_v1 requires contiguous padding masks")
                sample = self._forward_unpadded(hidden_states[batch_idx:batch_idx + 1, start:stop])
                output[batch_idx:batch_idx + 1, start:stop] = sample

        return output, None if not output_attentions else None, past_key_values

    @torch.no_grad()
    def init_from_teacher(self, teacher_attn):
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            student_proj = getattr(self.attn, name)
            teacher_proj = getattr(teacher_attn, name)
            student_proj.weight.copy_(teacher_proj.weight)
            if student_proj.bias is not None and teacher_proj.bias is not None:
                student_proj.bias.copy_(teacher_proj.bias)
        for name in ["q_norm", "k_norm"]:
            if hasattr(self.attn, name) and hasattr(teacher_attn, name):
                student_norm = getattr(self.attn, name)
                teacher_norm = getattr(teacher_attn, name)
                student_norm.weight.copy_(teacher_norm.weight)
                if (
                    getattr(student_norm, "bias", None) is not None
                    and getattr(teacher_norm, "bias", None) is not None
                ):
                    student_norm.bias.copy_(teacher_norm.bias)

