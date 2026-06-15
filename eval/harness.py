# -*- coding: utf-8 -*-

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import fla  # noqa: F401
import math
import torch
import yaml
import torch.nn.functional as F
from torch import nn
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

from distill_model.config_distilled_student import StudentConfig
from distill_model.modeling_distilled_student import StudentModel, StudentForCausalLM
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

AutoConfig.register("student", StudentConfig, exist_ok=True)
AutoModelForCausalLM.register(StudentConfig, StudentForCausalLM, exist_ok=True)


def _parse_keep_layers(keep_layers: str | None, cfg: str | None) -> set[int]:
    if cfg:
        with open(cfg) as f:
            cfg_dict = yaml.safe_load(f)
        return set(cfg_dict["student_model"].get("keep_full_attention_layers", []))
    if not keep_layers:
        return set()
    return {int(part.strip()) for part in keep_layers.split(",") if part.strip()}


class SecondOrderQwen2Attention(nn.Module):
    def __init__(self, base_attn: nn.Module, order: int = 2, decay: float = 0.9, mode: str = "taylor"):
        super().__init__()
        self.config = base_attn.config
        self.layer_idx = base_attn.layer_idx
        self.head_dim = base_attn.head_dim
        self.hidden_size = base_attn.config.hidden_size
        self.num_heads = base_attn.config.num_attention_heads
        self.num_key_value_heads = base_attn.config.num_key_value_heads
        self.num_key_value_groups = base_attn.num_key_value_groups
        self.scaling = base_attn.scaling
        self.attention_dropout = base_attn.attention_dropout
        self.is_causal = base_attn.is_causal
        self.q_proj = base_attn.q_proj
        self.k_proj = base_attn.k_proj
        self.v_proj = base_attn.v_proj
        self.o_proj = base_attn.o_proj
        self.sliding_window = getattr(base_attn, "sliding_window", None)
        self.order = order
        self.decay = decay
        self.mode = "taylor" if mode == "gdn_taylor" else mode
        self.state_dtype = torch.float32
        self._cached_state = None

    def _reset_cached_state(self):
        self._cached_state = None

    def _is_query_valid(self, attention_mask: torch.Tensor | None, t: int) -> torch.Tensor | None:
        if attention_mask is None:
            return None
        if attention_mask.dim() == 4:
            return (attention_mask[:, 0, t, t] == 0)
        if attention_mask.dim() == 2:
            return attention_mask[:, t].bool()
        return None

    def _step_update(
        self,
        q_t: torch.Tensor,
        k_t: torch.Tensor,
        v_t: torch.Tensor,
        valid_t: torch.Tensor | None,
        states: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, heads, d_k = q_t.shape
        device = q_t.device
        dtype = self.state_dtype

        q_t = q_t.to(dtype)
        k_t = k_t.to(dtype)
        v_t = v_t.to(dtype)

        if valid_t is None:
            valid = torch.ones(batch, 1, 1, device=device, dtype=dtype)
        else:
            valid = valid_t.to(dtype).view(batch, 1, 1)

        kk_t = torch.einsum("bhd,bhe->bhde", k_t, k_t)
        vk_t = torch.einsum("bhv,bhk->bhvk", v_t, k_t)

        states["s0"] = states["s0"] + valid * v_t
        states["z0"] = states["z0"] + valid.squeeze(-1)
        states["z1"] = states["z1"] + valid * k_t
        states["s1"] = states["s1"] + valid * vk_t

        if self.order >= 2:
            vkk_t = torch.einsum("bhv,bhk,bhl->bhvkl", v_t, k_t, k_t)
            states["s2"] = states["s2"] + valid.unsqueeze(-1) * vkk_t
            states["z2"] = states["z2"] + valid.unsqueeze(-1) * kk_t

        q_scaled = q_t / math.sqrt(self.head_dim)
        r1 = torch.einsum("bhvk,bhk->bhv", states["s1"], q_scaled)
        numerator = states["s0"] + r1
        denom = states["z0"] + torch.einsum("bhk,bhk->bh", states["z1"], q_scaled)

        if self.order >= 2:
            r2 = torch.einsum("bhvkl,bhk,bhl->bhv", states["s2"], q_t, q_t)
            numerator = numerator + r2 / (2.0 * self.head_dim)
            quad = torch.einsum("bhkl,bhk,bhl->bh", states["z2"], q_t, q_t)
            denom = denom + quad / (2.0 * self.head_dim)

        denom = denom.clamp_min(1e-6).unsqueeze(-1)
        out_t = numerator / denom
        if valid_t is not None:
            out_t = out_t * valid_t.to(dtype).view(batch, 1, 1)
        return out_t, states

    def _allocate_states(self, batch: int, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
        value_dim = self.head_dim
        states = {
            "s0": torch.zeros(batch, self.num_heads, value_dim, device=device, dtype=self.state_dtype),
            "s1": torch.zeros(batch, self.num_heads, value_dim, self.head_dim, device=device, dtype=self.state_dtype),
            "z0": torch.zeros(batch, self.num_heads, device=device, dtype=self.state_dtype),
            "z1": torch.zeros(batch, self.num_heads, self.head_dim, device=device, dtype=self.state_dtype),
        }
        if self.order >= 2:
            states["s2"] = torch.zeros(
                batch, self.num_heads, value_dim, self.head_dim, self.head_dim, device=device, dtype=self.state_dtype
            )
            states["z2"] = torch.zeros(
                batch, self.num_heads, self.head_dim, self.head_dim, device=device, dtype=self.state_dtype
            )
        return states

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_value=None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        query_states = query_states.float()
        key_states = key_states.float()
        value_states = value_states.float()

        batch, _, seq_len, _ = query_states.shape
        use_recurrent_cache = past_key_value is not None and seq_len == 1
        if use_recurrent_cache and cache_position is not None and int(cache_position.min().item()) == 0:
            self._reset_cached_state()

        if use_recurrent_cache:
            if self._cached_state is None or self._cached_state["s0"].shape[0] != batch:
                self._cached_state = self._allocate_states(batch, query_states.device, query_states.dtype)
            valid_t = self._is_query_valid(attention_mask, 0)
            context, self._cached_state = self._step_update(
                query_states[:, :, 0, :],
                key_states[:, :, 0, :],
                value_states[:, :, 0, :],
                valid_t,
                self._cached_state,
            )
            attn_output = context.reshape(batch, 1, -1)
        else:
            states = self._allocate_states(batch, query_states.device, query_states.dtype)
            outputs = []
            for t in range(seq_len):
                valid_t = self._is_query_valid(attention_mask, t)
                context_t, states = self._step_update(
                    query_states[:, :, t, :],
                    key_states[:, :, t, :],
                    value_states[:, :, t, :],
                    valid_t,
                    states,
                )
                outputs.append(context_t)
            attn_output = torch.stack(outputs, dim=2).transpose(1, 2).contiguous().reshape(*input_shape, -1)

        attn_output = self.o_proj(attn_output.to(self.o_proj.weight.dtype))
        attn_weights = None

        if not kwargs.get("output_attentions", False):
            attn_weights = None
        return attn_output, attn_weights


def _patch_qwen2_with_second_order(
    model,
    order: int = 2,
    keep_layers: set[int] | None = None,
    decay: float = 0.9,
    mode: str = "taylor",
):
    keep_layers = keep_layers or set()
    replaced = 0
    for idx, layer in enumerate(model.model.layers):
        if idx in keep_layers:
            continue
        attn = getattr(layer, "self_attn", None)
        if attn is None or attn.__class__.__name__ != "Qwen2Attention":
            continue
        patched_attn = SecondOrderQwen2Attention(attn, order=order, decay=decay, mode=mode)
        patched_attn = patched_attn.to(device=attn.q_proj.weight.device, dtype=attn.q_proj.weight.dtype)
        layer.self_attn = patched_attn
        replaced += 1
    if replaced == 0:
        raise ValueError("No Qwen2Attention layers were patched. Check the model family and keep-layer selection.")


@register_model("fla")
class FlashLinearAttentionLMWrapper(HFLM):
    def __init__(self, **kwargs) -> FlashLinearAttentionLMWrapper:
        super().__init__(**kwargs)


@register_model("qwen2_second_order")
class Qwen2SecondOrderLMWrapper(HFLM):
    def __init__(
        self,
        pretrained,
        order=2,
        decay=0.9,
        mode="taylor",
        keep_layers=None,
        cfg=None,
        **kwargs,
    ):
        keep_layer_set = _parse_keep_layers(keep_layers, cfg)
        kwargs.setdefault("trust_remote_code", True)
        kwargs.setdefault("dtype", torch.bfloat16)
        kwargs.setdefault("tokenizer", pretrained)
        kwargs.setdefault("use_fast_tokenizer", True)
        kwargs.setdefault("truncation", False)
        kwargs["attn_implementation"] = "eager"
        super().__init__(pretrained=pretrained, **kwargs)
        _patch_qwen2_with_second_order(
            self.model,
            order=int(order),
            keep_layers=keep_layer_set,
            decay=float(decay),
            mode=mode,
        )


if __name__ == "__main__":
    cli_evaluate()
