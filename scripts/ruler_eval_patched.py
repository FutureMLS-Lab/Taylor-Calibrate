"""RULER eval with two defensive patches (a.k.a. "v6 patch").

Bug 1 — fla.layers.attn.Attention.forward unpad_input crash with KV cache.
    Fix: bypass attention_mask path entirely by passing None; FLA still uses
    the dense flash-attn path under causal=True, just without unpadding logic
    that mismatched the cache shape.

Bug 2 — StudentForCausalLM.prepare_inputs_for_generation slices to last token
    even on first prefill, because newer transformers pre-create a non-empty
    DynamicCache. Fix: replace it with FLA's GenerationMixin version which
    keys on cache_position rather than len(past_key_values).

Usage matches eval/harness.py — same lm_eval CLI args. Run via:
    python scripts/ruler_eval_patched.py --model hf \\
        --model_args "pretrained=<ckpt>,dtype=bfloat16,trust_remote_code=True,max_length=4096" \\
        --tasks ruler --batch_size 1 --device cuda --output_path <out>
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Make HF checkpoints load with the legacy default
import torch
_orig_load = torch.load
def _load_relaxed(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _load_relaxed

import fla  # noqa: F401
from fla.layers.attn import Attention as FLAAttention
from fla.models.utils import FLAGenerationMixin

from distill_model.config_distilled_student import StudentConfig
from distill_model.modeling_distilled_student import StudentForCausalLM
from transformers import AutoConfig, AutoModelForCausalLM

AutoConfig.register("student", StudentConfig, exist_ok=True)
AutoModelForCausalLM.register(StudentConfig, StudentForCausalLM, exist_ok=True)

# --- Patch 1: strip attention_mask in FLA Attention.forward ----------------
_orig_attn_forward = FLAAttention.forward
def _patched_attn_forward(self, hidden_states, attention_mask=None, *args, **kwargs):
    # Force-bypass the unpad path; flash-attn handles causal mask internally.
    return _orig_attn_forward(self, hidden_states, attention_mask=None, *args, **kwargs)
FLAAttention.forward = _patched_attn_forward
print("[PATCH v6] FLA Attention patched: stripped attention_mask")

# --- Patch 2: use FLA's prepare_inputs_for_generation ----------------------
StudentForCausalLM.prepare_inputs_for_generation = FLAGenerationMixin.prepare_inputs_for_generation
print("[PATCH v6] StudentForCausalLM.prepare_inputs_for_generation replaced with FLAGenerationMixin version")

# --- Hand off to lm_eval CLI -----------------------------------------------
from lm_eval.__main__ import cli_evaluate

if __name__ == "__main__":
    cli_evaluate()
