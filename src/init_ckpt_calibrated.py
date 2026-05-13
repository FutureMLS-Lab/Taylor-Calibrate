#!/usr/bin/env python3
"""
Better initialization for GDN hybrid student models.

Key insight: the fla GatedDeltaNet output path is:
  o = rms_norm(gdn_recurrence_output) * silu(g_proj(x))
  output = o_proj(o)

With random g_proj (2.4M params), silu(g_proj(x)) injects noise into
every GDN layer, giving PPL ~1M. By zeroing g_proj, silu(0)=0,
the GDN layers contribute nothing, and the model runs coherently through
residual connections + MLPs + 7 kept attention layers.

This gives a MUCH better starting point for distillation than random noise.

Variants:
  --strategy zero_gate       : zero g_proj only
  --strategy zero_all        : zero g_proj + a_proj + b_proj (simplest dynamics)
  --strategy zero_calibrate  : zero g_proj + calibrated A_log/dt_bias/b_proj
  --strategy small_gate      : g_proj *= 0.02 (tiny non-zero for immediate gradients)
  --strategy calibrate_gproj : per-layer gradient descent on g_proj to match teacher attn output
"""

import argparse
import math
import os
import yaml

import fla  # noqa: F401
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from distill_model.config_distilled_student import StudentConfig
from distill_model.modeling_distilled_student import StudentForCausalLM

AutoConfig.register("student", StudentConfig, exist_ok=True)
AutoModelForCausalLM.register(StudentConfig, StudentForCausalLM, exist_ok=True)

from init_ckpt_from_teacher import build_student_from_teacher, parse_config

STRATEGIES = [
    "zero_gate", "zero_all", "zero_calibrate", "small_gate",
    "calibrate_gproj", "calibrate_gproj_reg", "calibrate_gates_gproj",
    "gate_data_calib",
    "gproj_from_v", "gproj_from_o",
    "calib_gates_gproj_v",
    "ols_calibrate",
    "taylor_calibrate_init",
    "small_gate_calibrate",
    "layer_distill",
    "taylor_calibrate",
    "alignment_only",
]


def apply_zero_gate(student, keep_layers):
    """Zero out g_proj so GDN layers contribute nothing initially."""
    count = 0
    for idx, layer in enumerate(student.model.layers):
        if idx in keep_layers:
            continue
        attn = layer.attn
        if hasattr(attn, 'g_proj'):
            attn.g_proj.weight.data.zero_()
            count += 1
    print(f"  Zeroed g_proj in {count} GDN layers")


def apply_zero_all(student, keep_layers):
    """Zero g_proj + a_proj + b_proj for simplest initial dynamics.
    beta=sigmoid(0)=0.5, decay determined only by A_log/dt_bias."""
    count = 0
    for idx, layer in enumerate(student.model.layers):
        if idx in keep_layers:
            continue
        attn = layer.attn
        if hasattr(attn, 'g_proj'):
            attn.g_proj.weight.data.zero_()
            attn.a_proj.weight.data.zero_()
            attn.b_proj.weight.data.zero_()
            count += 1
    print(f"  Zeroed g_proj + a_proj + b_proj in {count} GDN layers")


def apply_small_gate(student, keep_layers, scale=0.02):
    """Scale g_proj by a small factor so attention contribution is tiny but nonzero.
    This gives immediate gradients for all parameters (no two-phase learning)."""
    count = 0
    for idx, layer in enumerate(student.model.layers):
        if idx in keep_layers:
            continue
        attn = layer.attn
        if hasattr(attn, 'g_proj'):
            attn.g_proj.weight.data.mul_(scale)
            count += 1
    print(f"  Scaled g_proj by {scale} in {count} GDN layers")


@torch.no_grad()
def apply_zero_calibrate(student, keep_layers, hf_model, tokenizer, device):
    """Zero g_proj + calibrate A_log/dt_bias/b_proj from teacher attention stats.
    Sets decay half-life from attention distance, beta from entropy."""
    from datasets import load_dataset
    import torch.nn.functional as F

    for idx, layer in enumerate(student.model.layers):
        if idx in keep_layers:
            continue
        attn = layer.attn
        if hasattr(attn, 'g_proj'):
            attn.g_proj.weight.data.zero_()

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    full_text = "\n\n".join(t for t in ds["text"] if t.strip())
    enc = tokenizer(full_text, return_tensors="pt", truncation=False)
    all_ids = enc["input_ids"][0]
    seqlen, n_samples = 512, 4
    samples = [all_ids[i * seqlen:(i + 1) * seqlen] for i in range(n_samples)]
    input_ids = torch.stack(samples).to(device)
    attention_mask = torch.ones_like(input_ids)

    outputs = hf_model(
        input_ids=input_ids, attention_mask=attention_mask,
        output_attentions=True, use_cache=False,
    )

    B = input_ids.size(0)
    count = 0
    for layer_idx, attn_w in enumerate(outputs.attentions):
        if layer_idx in keep_layers:
            continue
        attn = student.model.layers[layer_idx].attn
        if not hasattr(attn, 'A_log'):
            continue

        attn_w = attn_w.float()
        _, H, T, _ = attn_w.shape

        token_mask = attention_mask[:, :T].bool()
        m = token_mask.float().unsqueeze(1).expand(B, H, T)
        cnt = m.sum(dim=(0, 2)).clamp_min(1)

        pos = torch.arange(T, device=device).float()
        dist = (pos.unsqueeze(1) - pos.unsqueeze(0)).clamp_min(0)
        avg_d = (attn_w * dist[None, None]).sum(-1)
        avg_distance = ((avg_d * m).sum(dim=(0, 2)) / cnt).clamp_min(0.5)

        causal = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
        valid = causal[None, None] & token_mask[:, None, None, :]
        lp = torch.log(attn_w.clamp_min(1e-10))
        ent = -(attn_w * lp * valid.float()).sum(-1)
        avg_entropy = (ent * m).sum(dim=(0, 2)) / cnt

        attn.A_log.data.zero_()
        target_g = (math.log(2.0) / avg_distance).clamp(0.001, 5.0).clamp_min(1e-4)
        inv_dt = target_g + torch.log(-torch.expm1(-target_g))
        attn.dt_bias.data.copy_(inv_dt)

        max_e, min_e = avg_entropy.max(), avg_entropy.min()
        ent_range = (max_e - min_e).clamp_min(1e-3)
        concentration = 1.0 - (avg_entropy - min_e) / ent_range
        target_beta = 0.3 + 0.4 * concentration
        target_logit = torch.log(target_beta / (1.0 - target_beta))
        fan_in = attn.b_proj.weight.shape[1]
        w_abs = attn.b_proj.weight.data.abs().mean(dim=1, keepdim=True).clamp_min(1e-8)
        scale = target_logit / math.sqrt(fan_in)
        attn.b_proj.weight.data.copy_(
            scale.unsqueeze(1) * attn.b_proj.weight.data / w_abs
        )

        count += 1
        dist_str = ", ".join(f"{d:.0f}" for d in avg_distance.cpu().tolist())
        print(f"    Layer {layer_idx:2d}: dist=[{dist_str}]")

    print(f"  Calibrated {count} GDN layers (g_proj zeroed + A_log/dt_bias/b_proj calibrated)")


def _get_calibration_data(tokenizer, device, seqlen=256, num_samples=4):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    full_text = "\n\n".join(t for t in ds["text"] if t.strip())
    enc = tokenizer(full_text, return_tensors="pt", truncation=False)
    all_ids = enc["input_ids"][0]
    samples = [all_ids[i * seqlen:(i + 1) * seqlen] for i in range(num_samples)]
    input_ids = torch.stack(samples).to(device)
    return input_ids, torch.ones_like(input_ids)


def _gdn_recurrence_core(student_attn, x, attention_mask, v_scale=None, use_silu=True):
    """Prepare GDN projections and run recurrence, yielding per-timestep outputs.

    Returns (S_state, per_step_generator) or use via _gdn_recurrence / _gdn_recurrence_ols.
    """
    import torch.nn.functional as F

    B, T, D = x.shape
    H = student_attn.num_heads
    hd = student_attn.head_k_dim
    device = x.device

    q = student_attn.q_proj(x.to(student_attn.q_proj.weight.dtype)).float()
    k = student_attn.k_proj(x.to(student_attn.k_proj.weight.dtype)).float()
    v = student_attn.v_proj(x.to(student_attn.v_proj.weight.dtype)).float()

    if use_silu:
        q = F.silu(q)
        k = F.silu(k)
        v = F.silu(v)

    q = F.normalize(q.view(B, T, H, hd), p=2, dim=-1)
    k = F.normalize(k.view(B, T, H, hd), p=2, dim=-1)
    v = v.view(B, T, H, hd)

    if v_scale is not None:
        v = v * v_scale.view(1, 1, H, 1)

    beta = student_attn.b_proj(x.to(student_attn.b_proj.weight.dtype)).float().sigmoid()
    g = -student_attn.A_log.float().exp() * F.softplus(
        student_attn.a_proj(x.to(student_attn.a_proj.weight.dtype)).float()
        + student_attn.dt_bias
    )

    if attention_mask is not None:
        mask = attention_mask[:, :T].bool()
    else:
        mask = torch.ones(B, T, device=device, dtype=torch.bool)

    return B, T, H, hd, q, k, v, beta, g, mask


def _gdn_recurrence(student_attn, x, attention_mask, v_scale=None, use_silu=True):
    """Run GDN recurrence manually and return pre-o_norm output [B, T, H, hd]."""
    B, T, H, hd, q, k, v, beta, g, mask = _gdn_recurrence_core(
        student_attn, x, attention_mask, v_scale=v_scale, use_silu=use_silu
    )
    device = x.device
    S = torch.zeros(B, H, hd, hd, device=device, dtype=torch.float32)
    outs = []
    for t in range(T):
        kt, vt, qt = k[:, t], v[:, t], q[:, t]
        bt = beta[:, t].unsqueeze(-1).unsqueeze(-1)
        gt = g[:, t].unsqueeze(-1).unsqueeze(-1)
        kk = kt.unsqueeze(-1) @ kt.unsqueeze(-2)
        vk = vt.unsqueeze(-1) @ kt.unsqueeze(-2)
        S = torch.exp(gt) * (S - bt * (S @ kk)) + bt * vk
        ot = (S @ qt.unsqueeze(-1)).squeeze(-1)
        if not mask[:, t].all():
            ot = ot * mask[:, t].float().view(B, 1, 1)
        outs.append(ot)
    return torch.stack(outs, dim=1)  # [B, T, H, hd]


def _gdn_recurrence_ols(student_attn, x, attention_mask, teacher_ctx,
                         v_scale=None, use_silu=True):
    """Run GDN recurrence and compute OLS statistics in a streaming fashion.

    Instead of storing all T timestep outputs (which OOMs on large models),
    accumulates sigma_num and sigma_den incrementally.

    Returns (sigma_num, sigma_den) both of shape [H].
    """
    B, T, H, hd, q, k, v, beta, g, mask = _gdn_recurrence_core(
        student_attn, x, attention_mask, v_scale=v_scale, use_silu=use_silu
    )
    device = x.device
    S = torch.zeros(B, H, hd, hd, device=device, dtype=torch.float32)
    sigma_num = torch.zeros(H, device=device, dtype=torch.float32)
    sigma_den = torch.zeros(H, device=device, dtype=torch.float32)
    for t in range(T):
        kt, vt, qt = k[:, t], v[:, t], q[:, t]
        bt = beta[:, t].unsqueeze(-1).unsqueeze(-1)
        gt = g[:, t].unsqueeze(-1).unsqueeze(-1)
        kk = kt.unsqueeze(-1) @ kt.unsqueeze(-2)
        vk = vt.unsqueeze(-1) @ kt.unsqueeze(-2)
        S = torch.exp(gt) * (S - bt * (S @ kk)) + bt * vk
        ot = (S @ qt.unsqueeze(-1)).squeeze(-1)  # [B, H, hd]
        if not mask[:, t].all():
            ot = ot * mask[:, t].float().view(B, 1, 1)
        tc_t = teacher_ctx[:, t]  # [B, H, hd]
        sigma_num += (tc_t * ot).sum(dim=(0, 2))
        sigma_den += (ot ** 2).sum(dim=(0, 2))
    return sigma_num, sigma_den


def _calibrate_gproj_core(student, hf_model, tokenizer, cfg, device,
                          n_steps=300, lr=0.01, weight_decay=0.0,
                          pre_calibrate_gates=False):
    """Core g_proj calibration via per-layer gradient descent.

    For each GDN layer:
      target:  rms_norm(gdn_out) * silu(g_proj(x)) ≈ teacher_attn_out
      solve:   min_{g_proj.weight} MSE + weight_decay * ||W||^2

    gdn_out depends on (optionally pre-calibrated) a/b/A_log/dt_bias.
    teacher_attn_out = softmax_attn_weights @ V_teacher (pre-o_proj).
    x = teacher's hidden states at this layer (layer-wise teacher-forcing).
    """
    import torch.nn.functional as F

    keep_layers = set(cfg["student_model"].get("keep_full_attention_layers", []))
    input_ids, attention_mask = _get_calibration_data(tokenizer, device, seqlen=256, num_samples=4)
    B, T = input_ids.shape

    print("  Running teacher forward...")
    with torch.no_grad():
        t_out = hf_model(
            input_ids=input_ids, attention_mask=attention_mask,
            output_attentions=True, output_hidden_states=True, use_cache=False,
        )

    if pre_calibrate_gates:
        print("  Pre-calibrating recurrence gates (A_log/dt_bias/b_proj)...")
        _calibrate_gates_from_attn(student, keep_layers, t_out, attention_mask, device)

    num_kv = hf_model.config.num_key_value_heads
    num_q = hf_model.config.num_attention_heads
    gqa_g = num_q // num_kv

    count = 0
    for layer_idx in range(len(student.model.layers)):
        if layer_idx in keep_layers:
            continue
        sa = student.model.layers[layer_idx].attn
        if not hasattr(sa, 'g_proj'):
            continue

        H, hd = sa.num_heads, sa.head_k_dim

        with torch.no_grad():
            hi = t_out.hidden_states[layer_idx].to(device).float()
            tln = hf_model.model.layers[layer_idx].input_layernorm
            x = tln(hi.to(tln.weight.dtype)).float()

            v_t = hf_model.model.layers[layer_idx].self_attn.v_proj(
                x.to(hf_model.model.layers[layer_idx].self_attn.v_proj.weight.dtype)
            ).float().view(B, T, num_kv, hd).permute(0, 2, 1, 3)
            v_t = v_t.repeat_interleave(gqa_g, dim=1)
            aw = t_out.attentions[layer_idx].to(device).float()
            teacher_target = (aw @ v_t).permute(0, 2, 1, 3)

            gdn_out = _gdn_recurrence(sa, x, attention_mask)
            eps = getattr(sa.o_norm, 'eps', 1e-5)
            rms = torch.sqrt(gdn_out.pow(2).mean(dim=-1, keepdim=True) + eps)
            gdn_normed = (gdn_out / rms) * sa.o_norm.weight.float()

        sa.g_proj.weight.data.zero_()
        sa.g_proj.weight.requires_grad_(True)
        optimizer = torch.optim.AdamW([sa.g_proj.weight], lr=lr, weight_decay=weight_decay)
        mask4 = attention_mask[:, :T].bool().float().unsqueeze(-1).unsqueeze(-1)

        x_det, gn_det, tgt_det = x.detach(), gdn_normed.detach(), teacher_target.detach()

        for step in range(n_steps):
            optimizer.zero_grad()
            g = sa.g_proj(x_det.to(sa.g_proj.weight.dtype)).float().view(B, T, H, hd)
            pred = gn_det * F.silu(g)
            loss = ((pred - tgt_det).pow(2) * mask4).sum() / mask4.sum() / (H * hd)
            loss.backward()
            optimizer.step()

        sa.g_proj.weight.requires_grad_(False)
        count += 1

        with torch.no_grad():
            cos = F.cosine_similarity(
                (gn_det * F.silu(sa.g_proj(x_det.to(sa.g_proj.weight.dtype)).float().view(B, T, H, hd))).reshape(-1, hd),
                tgt_det.reshape(-1, hd), dim=-1,
            ).mean()
            wnorm = sa.g_proj.weight.data.norm().item()
        print(f"    Layer {layer_idx:2d}: loss={loss.item():.6f}  cos={cos.item():.4f}  |W|={wnorm:.1f}")

    print(f"\n  Calibrated g_proj in {count} GDN layers.")


def _calibrate_gates_from_attn(student, keep_layers, t_out, attention_mask, device):
    """Calibrate A_log/dt_bias/b_proj from teacher attention stats (reused from zero_calibrate)."""
    B = attention_mask.size(0)
    for layer_idx, attn_w in enumerate(t_out.attentions):
        if layer_idx in keep_layers:
            continue
        attn = student.model.layers[layer_idx].attn
        if not hasattr(attn, 'A_log'):
            continue

        attn_w = attn_w.float()
        _, H, T, _ = attn_w.shape
        token_mask = attention_mask[:, :T].bool()
        m = token_mask.float().unsqueeze(1).expand(B, H, T)
        cnt = m.sum(dim=(0, 2)).clamp_min(1)

        pos = torch.arange(T, device=device).float()
        dist = (pos.unsqueeze(1) - pos.unsqueeze(0)).clamp_min(0)
        avg_d = (attn_w * dist[None, None]).sum(-1)
        avg_distance = ((avg_d * m).sum(dim=(0, 2)) / cnt).clamp_min(0.5)

        causal = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
        valid = causal[None, None] & token_mask[:, None, None, :]
        lp = torch.log(attn_w.clamp_min(1e-10))
        ent = -(attn_w * lp * valid.float()).sum(-1)
        avg_entropy = (ent * m).sum(dim=(0, 2)) / cnt

        attn.A_log.data.zero_()
        target_g = (math.log(2.0) / avg_distance).clamp(0.001, 5.0).clamp_min(1e-4)
        inv_dt = target_g + torch.log(-torch.expm1(-target_g))
        attn.dt_bias.data.copy_(inv_dt)

        max_e, min_e = avg_entropy.max(), avg_entropy.min()
        ent_range = (max_e - min_e).clamp_min(1e-3)
        concentration = 1.0 - (avg_entropy - min_e) / ent_range
        target_beta = 0.3 + 0.4 * concentration
        target_logit = torch.log(target_beta / (1.0 - target_beta))
        fan_in = attn.b_proj.weight.shape[1]
        w_abs = attn.b_proj.weight.data.abs().mean(dim=1, keepdim=True).clamp_min(1e-8)
        scale = target_logit / math.sqrt(fan_in)
        attn.b_proj.weight.data.copy_(
            scale.unsqueeze(1) * attn.b_proj.weight.data / w_abs
        )
        print(f"    Layer {layer_idx:2d}: gates calibrated")


def apply_calibrate_gproj(student, hf_model, tokenizer, cfg, device):
    """g_proj calibration only (random gates, no regularization)."""
    _calibrate_gproj_core(student, hf_model, tokenizer, cfg, device,
                          n_steps=300, lr=0.01, weight_decay=0.0)


def apply_calibrate_gproj_reg(student, hf_model, tokenizer, cfg, device):
    """g_proj calibration with strong weight decay to prevent overfitting."""
    _calibrate_gproj_core(student, hf_model, tokenizer, cfg, device,
                          n_steps=300, lr=0.01, weight_decay=1.0)


def apply_calibrate_gates_gproj(student, hf_model, tokenizer, cfg, device):
    """Calibrate recurrence gates first, then g_proj with moderate regularization."""
    _calibrate_gproj_core(student, hf_model, tokenizer, cfg, device,
                          n_steps=300, lr=0.01, weight_decay=0.1,
                          pre_calibrate_gates=True)


@torch.no_grad()
def _set_gproj_from_v(student, hf_model, t_out, cfg, device, damping=0.1):
    """Set g_proj = v_proj * alpha, where alpha is calibrated to produce
    output magnitude = damping * teacher_attn_rms."""
    import torch.nn.functional as F

    keep_layers = set(cfg["student_model"].get("keep_full_attention_layers", []))
    num_kv = hf_model.config.num_key_value_heads
    num_q = hf_model.config.num_attention_heads
    gqa_g = num_q // num_kv
    B, T = t_out.hidden_states[0].shape[:2]

    count = 0
    for layer_idx in range(len(student.model.layers)):
        if layer_idx in keep_layers:
            continue
        sa = student.model.layers[layer_idx].attn
        if not hasattr(sa, 'g_proj'):
            continue

        H, hd = sa.num_heads, sa.head_k_dim

        t_dev = next(hf_model.parameters()).device
        hi = t_out.hidden_states[layer_idx].to(t_dev).float()
        tln = hf_model.model.layers[layer_idx].input_layernorm
        x_t = tln(hi.to(tln.weight.dtype)).float()

        v_t = hf_model.model.layers[layer_idx].self_attn.v_proj(
            x_t.to(hf_model.dtype)
        ).float().view(B, T, num_kv, hd).permute(0, 2, 1, 3)
        v_t = v_t.repeat_interleave(gqa_g, dim=1)
        aw = t_out.attentions[layer_idx].to(t_dev).float()
        teacher_attn = (aw @ v_t).permute(0, 2, 1, 3)
        teacher_rms = teacher_attn.pow(2).mean(dim=-1).sqrt().mean().item()

        x = x_t.to(device)
        v_student = sa.v_proj(x.to(sa.v_proj.weight.dtype)).float()
        from distill_model.student_layers import GatedDeltaNetStudentV4NoSilu
        if isinstance(sa, GatedDeltaNetStudentV4NoSilu):
            gate_v_rms = v_student.pow(2).mean().sqrt().item()
        else:
            gate_v_rms = F.silu(v_student).pow(2).mean().sqrt().item()

        alpha = damping * teacher_rms / max(gate_v_rms, 1e-8)
        sa.g_proj.weight.data.copy_(sa.v_proj.weight.data * alpha)

        count += 1
        print(f"    Layer {layer_idx:2d}: alpha={alpha:.6f}  teacher_rms={teacher_rms:.4f}  v_rms={gate_v_rms:.4f}")

    print(f"\n  Set g_proj = v_proj * alpha in {count} GDN layers (damping={damping}).")


@torch.no_grad()
def apply_gproj_from_v(student, hf_model, tokenizer, cfg, device):
    """g_proj = v_proj * alpha (damping=0.1)."""
    input_ids, attention_mask = _get_calibration_data(tokenizer, device, seqlen=256, num_samples=4)
    with torch.no_grad():
        t_out = hf_model(
            input_ids=input_ids, attention_mask=attention_mask,
            output_attentions=True, output_hidden_states=True, use_cache=False,
        )
    _set_gproj_from_v(student, hf_model, t_out, cfg, device, damping=0.1)


@torch.no_grad()
def apply_gproj_from_o(student, hf_model, tokenizer, cfg, device):
    """Initialize g_proj from teacher's o_proj transposed, scaled by calibrated alpha.

    Rationale: o_proj maps attention-space → hidden-space.
    o_proj^T maps hidden-space → attention-space (inverse direction).
    g_proj(x) = alpha * o_proj^T @ x projects hidden_states into the
    attention output space, telling the gate "what the attention output
    should look like" based on the current context.
    """
    import torch.nn.functional as F

    keep_layers = set(cfg["student_model"].get("keep_full_attention_layers", []))
    input_ids, attention_mask = _get_calibration_data(tokenizer, device, seqlen=256, num_samples=4)
    B, T = input_ids.shape

    with torch.no_grad():
        t_out = hf_model(
            input_ids=input_ids, attention_mask=attention_mask,
            output_attentions=True, output_hidden_states=True, use_cache=False,
        )

    num_kv = hf_model.config.num_key_value_heads
    num_q = hf_model.config.num_attention_heads
    gqa_g = num_q // num_kv

    count = 0
    for layer_idx in range(len(student.model.layers)):
        if layer_idx in keep_layers:
            continue
        sa = student.model.layers[layer_idx].attn
        if not hasattr(sa, 'g_proj'):
            continue

        H, hd = sa.num_heads, sa.head_k_dim

        hi = t_out.hidden_states[layer_idx].to(device).float()
        tln = hf_model.model.layers[layer_idx].input_layernorm
        x = tln(hi.to(tln.weight.dtype)).float()

        # Teacher pre-o_proj attention output RMS
        v_t = hf_model.model.layers[layer_idx].self_attn.v_proj(
            x.to(hf_model.dtype)
        ).float().view(B, T, num_kv, hd).permute(0, 2, 1, 3)
        v_t = v_t.repeat_interleave(gqa_g, dim=1)
        aw = t_out.attentions[layer_idx].to(device).float()
        teacher_attn = (aw @ v_t).permute(0, 2, 1, 3)
        teacher_rms = teacher_attn.pow(2).mean(dim=-1).sqrt().mean().item()

        # o_proj^T applied to hidden_states, then silu magnitude
        o_weight_t = sa.o_proj.weight.data.T  # [value_dim, hidden_size]
        proj = F.linear(x.to(o_weight_t.dtype), o_weight_t).float()
        silu_proj_rms = F.silu(proj).pow(2).mean().sqrt().item()

        damping = 0.1
        alpha = damping * teacher_rms / max(silu_proj_rms, 1e-8)
        sa.g_proj.weight.data.copy_(o_weight_t * alpha)

        count += 1
        print(f"    Layer {layer_idx:2d}: alpha={alpha:.6f}  teacher_rms={teacher_rms:.4f}  silu_oT_rms={silu_proj_rms:.4f}")

    print(f"\n  Set g_proj = o_proj^T * alpha in {count} GDN layers (damping={damping}).")


@torch.no_grad()
def apply_calib_gates_gproj_v(student, hf_model, tokenizer, cfg, device):
    """Combined: calibrate recurrence gates from attention stats, then g_proj = v_proj * alpha.

    1. A_log/dt_bias from avg attention distance (decay rate)
    2. b_proj from attention entropy (write gate concentration)
    3. g_proj = v_proj * alpha with small damping

    This gives the recurrence a structured starting point AND g_proj
    a semantically meaningful initialization in value-space.
    """
    keep_layers = set(cfg["student_model"].get("keep_full_attention_layers", []))
    input_ids, attention_mask = _get_calibration_data(tokenizer, device, seqlen=256, num_samples=4)
    B, T = input_ids.shape

    with torch.no_grad():
        t_out = hf_model(
            input_ids=input_ids, attention_mask=attention_mask,
            output_attentions=True, output_hidden_states=True, use_cache=False,
        )

    token_mask = attention_mask[:, :T].float()

    print("  [Step 1/2] Calibrating recurrence gates (A_log, dt_bias, b_proj)...")
    for layer_idx in range(len(student.model.layers)):
        if layer_idx in keep_layers:
            continue
        sa = student.model.layers[layer_idx].attn
        if not hasattr(sa, 'A_log'):
            continue

        attn_w = t_out.attentions[layer_idx][:, :, :T, :T].to(device).float()
        m = token_mask[:, None, :T]
        cnt = m.sum(dim=(0, 2)).clamp_min(1)

        dist = torch.arange(T, device=device).float().unsqueeze(0) - \
               torch.arange(T, device=device).float().unsqueeze(1)
        dist = dist.abs().clamp_min(0.5)
        dist = torch.tril(dist)
        avg_d = (attn_w * dist[None, None]).sum(-1)
        avg_distance = ((avg_d * m).sum(dim=(0, 2)) / cnt).clamp_min(0.5)

        causal = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
        valid = causal[None, None] & token_mask[:, None, None, :].bool()
        lp = torch.log(attn_w.clamp_min(1e-10))
        ent = -(attn_w * lp * valid.float()).sum(-1)
        avg_entropy = (ent * m).sum(dim=(0, 2)) / cnt

        sa.A_log.data.zero_()
        target_g = (math.log(2.0) / avg_distance).clamp(0.001, 5.0).clamp_min(1e-4)
        inv_dt = target_g + torch.log(-torch.expm1(-target_g))
        sa.dt_bias.data.copy_(inv_dt)

        max_e, min_e = avg_entropy.max(), avg_entropy.min()
        ent_range = (max_e - min_e).clamp_min(1e-3)
        concentration = 1.0 - (avg_entropy - min_e) / ent_range
        target_beta = 0.3 + 0.4 * concentration
        target_logit = torch.log(target_beta / (1.0 - target_beta))
        fan_in = sa.b_proj.weight.shape[1]
        w_abs = sa.b_proj.weight.data.abs().mean(dim=1, keepdim=True).clamp_min(1e-8)
        scale = target_logit / math.sqrt(fan_in)
        sa.b_proj.weight.data.copy_(
            scale.unsqueeze(1) * sa.b_proj.weight.data / w_abs
        )

        dist_str = ", ".join(f"{d:.0f}" for d in avg_distance.cpu().tolist())
        print(f"    Layer {layer_idx:2d}: dist=[{dist_str}]")

    print("\n  [Step 2/2] Setting g_proj = v_proj * alpha (damping=0.01)...")
    _set_gproj_from_v(student, hf_model, t_out, cfg, device, damping=0.01)


def _calibrate_gates_inplace(student, hf_model, t_out, cfg, device):
    """Calibrate A_log/dt_bias/b_proj from teacher attention stats (shared helper)."""
    keep_layers = set(cfg["student_model"].get("keep_full_attention_layers", []))
    B, T = t_out.hidden_states[0].shape[:2]
    token_mask = torch.ones(B, T, device=device).float()

    for layer_idx in range(len(student.model.layers)):
        if layer_idx in keep_layers:
            continue
        sa = student.model.layers[layer_idx].attn
        if not hasattr(sa, 'A_log'):
            continue

        attn_w = t_out.attentions[layer_idx][:, :, :T, :T].to(device).float()
        m = token_mask[:, None, :T]
        cnt = m.sum(dim=(0, 2)).clamp_min(1)

        dist = torch.arange(T, device=device).float().unsqueeze(0) - \
               torch.arange(T, device=device).float().unsqueeze(1)
        dist = dist.abs().clamp_min(0.5)
        dist = torch.tril(dist)
        avg_d = (attn_w * dist[None, None]).sum(-1)
        avg_distance = ((avg_d * m).sum(dim=(0, 2)) / cnt).clamp_min(0.5)

        causal = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
        valid = causal[None, None] & token_mask[:, None, None, :].bool()
        lp = torch.log(attn_w.clamp_min(1e-10))
        ent = -(attn_w * lp * valid.float()).sum(-1)
        avg_entropy = (ent * m).sum(dim=(0, 2)) / cnt

        sa.A_log.data.zero_()
        target_g = (math.log(2.0) / avg_distance).clamp(0.001, 5.0).clamp_min(1e-4)
        inv_dt = target_g + torch.log(-torch.expm1(-target_g))
        sa.dt_bias.data.copy_(inv_dt)

        max_e, min_e = avg_entropy.max(), avg_entropy.min()
        ent_range = (max_e - min_e).clamp_min(1e-3)
        concentration = 1.0 - (avg_entropy - min_e) / ent_range
        target_beta = 0.3 + 0.4 * concentration
        target_logit = torch.log(target_beta / (1.0 - target_beta))
        fan_in = sa.b_proj.weight.shape[1]
        w_abs = sa.b_proj.weight.data.abs().mean(dim=1, keepdim=True).clamp_min(1e-8)
        scale = target_logit / math.sqrt(fan_in)
        sa.b_proj.weight.data.copy_(
            scale.unsqueeze(1) * sa.b_proj.weight.data / w_abs
        )

        dist_str = ", ".join(f"{d:.0f}" for d in avg_distance.cpu().tolist())
        print(f"    Layer {layer_idx:2d}: dist=[{dist_str}]")


@torch.no_grad()
def apply_ols_calibrate(student, hf_model, tokenizer, cfg, device):
    """Full OLS calibration ported from GPT-2 attention_approx_compare.py:

    1. Calibrate gates (A_log, dt_bias, b_proj) from teacher attention stats
    2. Run GDN recurrence with calibrated gates
    3. OLS per-head sigma: min_sigma ||teacher_ctx - sigma_h * gdn_ctx_h||^2
    4. Scale v_proj weights by sigma (adjusts V per head)
    5. Set g_proj = scaled_v_proj * damping
    """
    import torch.nn.functional as F

    keep_layers = set(cfg["student_model"].get("keep_full_attention_layers", []))
    input_ids, attention_mask = _get_calibration_data(tokenizer, device, seqlen=256, num_samples=4)
    B, T = input_ids.shape

    with torch.no_grad():
        t_out = hf_model(
            input_ids=input_ids, attention_mask=attention_mask,
            output_attentions=True, output_hidden_states=True, use_cache=False,
        )
    _offload_t_out(t_out)

    num_kv = hf_model.config.num_key_value_heads
    num_q = hf_model.config.num_attention_heads
    gqa_g = num_q // num_kv

    from distill_model.student_layers import GatedDeltaNetStudentV4NoSilu
    first_gdn = next(
        (l.attn for l in student.model.layers
         if hasattr(l, 'attn') and hasattr(l.attn, 'A_log')), None
    )
    use_silu = not isinstance(first_gdn, GatedDeltaNetStudentV4NoSilu)
    print(f"  Detected use_silu={use_silu}")

    print("  [Step 1/3] Calibrating recurrence gates...")
    _calibrate_gates_inplace(student, hf_model, t_out, cfg, device)

    print("\n  [Step 2/3] OLS sigma per head (scaling V)...")
    for layer_idx in range(len(student.model.layers)):
        if layer_idx in keep_layers:
            continue
        sa = student.model.layers[layer_idx].attn
        if not hasattr(sa, 'g_proj'):
            continue

        H, hd = sa.num_heads, sa.head_k_dim

        hi = t_out.hidden_states[layer_idx].to(device).float()
        tln = hf_model.model.layers[layer_idx].input_layernorm
        x = tln(hi.to(tln.weight.dtype)).float()

        # Teacher pre-o_proj attention output: softmax(QK^T) @ V
        v_t = hf_model.model.layers[layer_idx].self_attn.v_proj(
            x.to(hf_model.dtype)
        ).float().view(B, T, num_kv, hd).permute(0, 2, 1, 3)
        v_t = v_t.repeat_interleave(gqa_g, dim=1)
        aw = t_out.attentions[layer_idx].to(device).float()
        teacher_ctx = (aw @ v_t).permute(0, 2, 1, 3)  # [B, T, H, hd]

        # OLS via streaming recurrence (avoids storing full [B,T,H,hd] cal_ctx)
        sigma_num, sigma_den = _gdn_recurrence_ols(
            sa, x, attention_mask, teacher_ctx, use_silu=use_silu
        )
        sigmas = (sigma_num / sigma_den.clamp_min(1e-8)).clamp(0.1, 10.0)
        del sigma_num, sigma_den

        # Scale v_proj weights per head
        for h in range(H):
            sa.v_proj.weight.data[h * hd:(h + 1) * hd, :] *= sigmas[h].item()

        # Cosine similarity between teacher and sigma-scaled GDN output
        cal_ctx_scaled = _gdn_recurrence(sa, x, attention_mask, use_silu=use_silu)
        cos = F.cosine_similarity(
            teacher_ctx.reshape(-1, hd), cal_ctx_scaled.reshape(-1, hd), dim=-1
        ).mean().item()

        sigma_str = ", ".join(f"{s:.3f}" for s in sigmas.cpu().tolist())
        print(f"    Layer {layer_idx:2d}: sigma=[{sigma_str}]  cos_after={cos:.4f}")

    print("\n  [Step 3/3] Setting g_proj = v_proj * alpha (damping=0.01)...")
    _set_gproj_from_v(student, hf_model, t_out, cfg, device, damping=0.01)


# ---------------------------------------------------------------------------
# Per-layer mini-distillation: adapt Q/K/V + gates + g_proj for GDN dynamics
# ---------------------------------------------------------------------------

def apply_layer_distill(student, hf_model, tokenizer, cfg, device,
                        n_steps=300, lr_new=1e-3, lr_qkv=1e-4):
    """Per-layer mini-distillation: optimize GDN params to match teacher attention output.

    For each GDN layer independently:
    - Target: teacher's attention output (pre-residual)
    - Optimized params: q/k/v_proj (small LR), gates + g_proj (larger LR)
    - Frozen: o_proj, layernorm, FFN, embeddings
    """
    import torch.nn.functional as F

    keep_layers = set(cfg["student_model"].get("keep_full_attention_layers", []))
    input_ids, attention_mask = _get_calibration_data(tokenizer, device, seqlen=256, num_samples=8)

    with torch.no_grad():
        t_out = hf_model(
            input_ids=input_ids, attention_mask=attention_mask,
            output_attentions=True, output_hidden_states=True, use_cache=False,
        )

    num_kv = hf_model.config.num_key_value_heads
    num_q = hf_model.config.num_attention_heads
    gqa_g = num_q // num_kv

    print("  [Step 1/2] Gate pre-calibration from attention stats...")
    _calibrate_gates_inplace(student, hf_model, t_out, cfg, device)

    print("\n  [Step 2/2] Per-layer mini-distillation...")
    for layer_idx in range(len(student.model.layers)):
        if layer_idx in keep_layers:
            continue
        sa = student.model.layers[layer_idx].attn
        if not hasattr(sa, 'g_proj'):
            continue

        H, hd = sa.num_heads, sa.head_k_dim

        hi = t_out.hidden_states[layer_idx].to(device).float()
        tln = hf_model.model.layers[layer_idx].input_layernorm
        x = tln(hi.to(tln.weight.dtype)).to(device)

        # Teacher attention output (post-o_proj, pre-residual)
        v_t = hf_model.model.layers[layer_idx].self_attn.v_proj(
            x.to(hf_model.dtype)
        ).float().view(x.shape[0], x.shape[1], num_kv, hd).permute(0, 2, 1, 3)
        v_t = v_t.repeat_interleave(gqa_g, dim=1)
        aw = t_out.attentions[layer_idx].to(device).float()
        teacher_pre_o = (aw @ v_t).permute(0, 2, 1, 3)  # [B, T, H, hd]
        teacher_pre_o = teacher_pre_o.reshape(x.shape[0], x.shape[1], -1)
        teacher_out = hf_model.model.layers[layer_idx].self_attn.o_proj(
            teacher_pre_o.to(hf_model.dtype)
        ).float().detach()

        # Param groups: QKV with small LR, new params with large LR
        qkv_params = [sa.q_proj.weight, sa.k_proj.weight, sa.v_proj.weight]
        new_params = [p for n, p in sa.named_parameters()
                      if p.requires_grad and not any(p is q for q in qkv_params)
                      and 'o_proj' not in n and 'o_norm' not in n]

        for p in sa.parameters():
            p.requires_grad_(False)
        for p in qkv_params:
            p.requires_grad_(True)
        for p in new_params:
            p.requires_grad_(True)

        optimizer = torch.optim.AdamW([
            {"params": qkv_params, "lr": lr_qkv},
            {"params": new_params, "lr": lr_new},
        ], weight_decay=0.01)

        x_input = x.detach()
        best_loss = float('inf')
        best_state = None

        for step in range(n_steps):
            optimizer.zero_grad()
            student_out, _, _ = sa(x_input, attention_mask=None)
            student_out = student_out.float()

            loss = F.mse_loss(student_out, teacher_out)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(qkv_params + new_params, 1.0)
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {n: p.data.clone() for n, p in sa.named_parameters()}

            if step % 100 == 0 or step == n_steps - 1:
                with torch.no_grad():
                    cos = F.cosine_similarity(
                        student_out.reshape(-1, student_out.shape[-1]),
                        teacher_out.reshape(-1, teacher_out.shape[-1]),
                        dim=-1,
                    ).mean().item()
                print(f"    Layer {layer_idx:2d} step {step:3d}: loss={loss.item():.6f}  cos={cos:.4f}")

        # Restore best checkpoint
        if best_state is not None:
            for n, p in sa.named_parameters():
                if n in best_state:
                    p.data.copy_(best_state[n])

        for p in sa.parameters():
            p.requires_grad_(False)

        # Final eval
        with torch.no_grad():
            final_out, _, _ = sa(x_input, attention_mask=None)
            final_cos = F.cosine_similarity(
                final_out.float().reshape(-1, final_out.shape[-1]),
                teacher_out.reshape(-1, teacher_out.shape[-1]),
                dim=-1,
            ).mean().item()
            final_loss = F.mse_loss(final_out.float(), teacher_out).item()
        print(f"    Layer {layer_idx:2d} FINAL: loss={final_loss:.6f}  cos={final_cos:.4f}")

    print("\n  Per-layer mini-distillation done.")


def _taylor_analytical_init(student, hf_model, t_out, cfg, device):
    """Taylor-based analytical initialization (from attention_approx_compare.py):

    1. Gate calibration: decay half-life from avg attention distance,
       beta from attention entropy (already in _calibrate_gates_inplace)
    2. OLS sigma: per-head scaling of V to min ||teacher_ctx - sigma * gdn_ctx||²
    3. g_proj = scaled_v_proj * damping
    """
    import torch.nn.functional as F

    keep_layers = set(cfg["student_model"].get("keep_full_attention_layers", []))
    num_kv = hf_model.config.num_key_value_heads
    num_q = hf_model.config.num_attention_heads
    gqa_g = num_q // num_kv
    B, T = t_out.hidden_states[0].shape[:2]

    from distill_model.student_layers import GatedDeltaNetStudentV4NoSilu
    first_gdn = next(
        (l.attn for l in student.model.layers
         if hasattr(l, 'attn') and hasattr(l.attn, 'A_log')), None
    )
    use_silu = not isinstance(first_gdn, GatedDeltaNetStudentV4NoSilu)

    print(f"  [Taylor Step 1/3] Gate calibration from attention stats...")
    _calibrate_gates_inplace(student, hf_model, t_out, cfg, device)

    print(f"\n  [Taylor Step 2/3] OLS sigma per head (scaling V, streaming)...")
    for layer_idx in range(len(student.model.layers)):
        if layer_idx in keep_layers:
            continue
        sa = student.model.layers[layer_idx].attn
        if not hasattr(sa, 'g_proj'):
            continue

        H, hd = sa.num_heads, sa.head_k_dim
        t_dev = next(hf_model.parameters()).device
        hi = t_out.hidden_states[layer_idx].to(t_dev).float()
        tln = hf_model.model.layers[layer_idx].input_layernorm
        x_t = tln(hi.to(tln.weight.dtype)).float()

        v_t = hf_model.model.layers[layer_idx].self_attn.v_proj(
            x_t.to(hf_model.dtype)
        ).float().view(B, T, num_kv, hd).permute(0, 2, 1, 3)
        v_t = v_t.repeat_interleave(gqa_g, dim=1)
        aw = t_out.attentions[layer_idx].to(t_dev).float()
        teacher_ctx = (aw @ v_t).permute(0, 2, 1, 3)
        del aw, v_t

        x = x_t.to(device)
        sigma_num, sigma_den = _gdn_recurrence_ols(
            sa, x, None, teacher_ctx.to(device), use_silu=use_silu
        )
        del teacher_ctx
        sigmas = (sigma_num / sigma_den.clamp_min(1e-8)).clamp(0.1, 10.0)
        del sigma_num, sigma_den

        for h in range(H):
            sa.v_proj.weight.data[h * hd:(h + 1) * hd, :] *= sigmas[h].item()

        sigma_str = ", ".join(f"{s:.3f}" for s in sigmas.cpu().tolist())
        print(f"    Layer {layer_idx:2d}: sigma=[{sigma_str}]")
        del hi, x, x_t, sigmas
        torch.cuda.empty_cache()

    print(f"\n  [Taylor Step 3/3] g_proj = v_proj * alpha (damping=0.01)...")
    _set_gproj_from_v(student, hf_model, t_out, cfg, device, damping=0.01)


def _offload_t_out(t_out):
    """Move t_out hidden_states and attentions to CPU to free GPU memory."""
    t_out.hidden_states = tuple(h.cpu() for h in t_out.hidden_states)
    t_out.attentions = tuple(a.cpu() for a in t_out.attentions)
    torch.cuda.empty_cache()
    return t_out


def apply_taylor_calibrate_init(student, hf_model, tokenizer, cfg, device):
    """Taylor analytical init only (Phase 1, no gradient fine-tuning).

    Gate calibration + OLS sigma + g_proj from v_proj.
    """
    input_ids, attention_mask = _get_calibration_data(tokenizer, device, seqlen=256, num_samples=4)
    with torch.no_grad():
        t_out = hf_model(
            input_ids=input_ids, attention_mask=attention_mask,
            output_attentions=True, output_hidden_states=True, use_cache=False,
        )
    _offload_t_out(t_out)
    _taylor_analytical_init(student, hf_model, t_out, cfg, device)
    print("\n  Taylor analytical init done (no gradient).")


def apply_small_gate_calibrate(student, hf_model, tokenizer, cfg, device, scale=0.02):
    """Small gate + Taylor analytical init.

    1. Taylor gate calibration (A_log/dt_bias/b_proj from attention stats)
    2. OLS sigma scaling on V
    3. g_proj scaled by small factor (instead of v_proj * alpha)
    """
    import torch.nn.functional as F

    keep_layers = set(cfg["student_model"].get("keep_full_attention_layers", []))
    input_ids, attention_mask = _get_calibration_data(tokenizer, device, seqlen=256, num_samples=4)

    with torch.no_grad():
        t_out = hf_model(
            input_ids=input_ids, attention_mask=attention_mask,
            output_attentions=True, output_hidden_states=True, use_cache=False,
        )
    _offload_t_out(t_out)

    num_kv = hf_model.config.num_key_value_heads
    num_q = hf_model.config.num_attention_heads
    gqa_g = num_q // num_kv
    B, T = t_out.hidden_states[0].shape[:2]

    from distill_model.student_layers import GatedDeltaNetStudentV4NoSilu
    first_gdn = next(
        (l.attn for l in student.model.layers
         if hasattr(l, 'attn') and hasattr(l.attn, 'A_log')), None
    )
    use_silu = not isinstance(first_gdn, GatedDeltaNetStudentV4NoSilu)

    print(f"  [Step 1/3] Gate calibration from attention stats...")
    _calibrate_gates_inplace(student, hf_model, t_out, cfg, device)

    print(f"\n  [Step 2/3] OLS sigma per head (scaling V)...")
    for layer_idx in range(len(student.model.layers)):
        if layer_idx in keep_layers:
            continue
        sa = student.model.layers[layer_idx].attn
        if not hasattr(sa, 'g_proj'):
            continue

        H, hd = sa.num_heads, sa.head_k_dim
        hi = t_out.hidden_states[layer_idx].to(device).float()
        tln = hf_model.model.layers[layer_idx].input_layernorm
        x = tln(hi.to(tln.weight.dtype)).float()

        v_t = hf_model.model.layers[layer_idx].self_attn.v_proj(
            x.to(hf_model.dtype)
        ).float().view(B, T, num_kv, hd).permute(0, 2, 1, 3)
        v_t = v_t.repeat_interleave(gqa_g, dim=1)
        aw = t_out.attentions[layer_idx].to(device).float()
        teacher_ctx = (aw @ v_t).permute(0, 2, 1, 3)

        cal_ctx = _gdn_recurrence(sa, x, None, use_silu=use_silu)

        sigma_num = (teacher_ctx.float() * cal_ctx.float()).sum(dim=(0, 1, 3))
        sigma_den = (cal_ctx.float() ** 2).sum(dim=(0, 1, 3))
        sigmas = (sigma_num / sigma_den.clamp_min(1e-8)).clamp(0.1, 10.0)

        for h in range(H):
            sa.v_proj.weight.data[h * hd:(h + 1) * hd, :] *= sigmas[h].item()

        sigma_str = ", ".join(f"{s:.3f}" for s in sigmas.cpu().tolist())
        print(f"    Layer {layer_idx:2d}: sigma=[{sigma_str}]")

    print(f"\n  [Step 3/3] Small gate scaling: g_proj *= {scale}...")
    count = 0
    for idx, layer in enumerate(student.model.layers):
        if idx in keep_layers:
            continue
        attn = layer.attn
        if hasattr(attn, 'g_proj'):
            attn.g_proj.weight.data.mul_(scale)
            count += 1
    print(f"  Scaled g_proj by {scale} in {count} GDN layers")
    print("\n  Small gate + Taylor calibrate done.")


def apply_taylor_calibrate(student, hf_model, tokenizer, cfg, device,
                         n_steps=300, lr_new=1e-3, lr_qkv=1e-4):
    """Taylor analytical init + per-layer gradient fine-tuning.

    Phase 1: Analytical init from attention_approx_compare.py
      - Gate calibration from softmax attention stats (decay, beta)
      - OLS sigma scaling on V
      - g_proj from scaled v_proj
    Phase 2: Per-layer gradient fine-tuning
      - Optimize Q/K/V (small LR) + gates + g_proj (large LR)
      - Target: match teacher attention output
    """
    import torch.nn.functional as F

    keep_layers = set(cfg["student_model"].get("keep_full_attention_layers", []))
    input_ids, attention_mask = _get_calibration_data(tokenizer, device, seqlen=256, num_samples=4)

    with torch.no_grad():
        t_out = hf_model(
            input_ids=input_ids, attention_mask=attention_mask,
            output_attentions=True, output_hidden_states=True, use_cache=False,
        )
    _offload_t_out(t_out)

    num_kv = hf_model.config.num_key_value_heads
    num_q = hf_model.config.num_attention_heads
    gqa_g = num_q // num_kv

    hf_model.cpu()
    torch.cuda.empty_cache()
    student.to(device)

    # Phase 1: Taylor analytical init
    print("=" * 60)
    print("  Phase 1: Taylor analytical initialization")
    print("=" * 60)
    _taylor_analytical_init(student, hf_model, t_out, cfg, device)

    # Phase 2: Per-layer gradient fine-tuning
    print("\n" + "=" * 60)
    print("  Phase 2: Per-layer gradient fine-tuning")
    print("=" * 60)
    B, T = input_ids.shape

    for layer_idx in range(len(student.model.layers)):
        if layer_idx in keep_layers:
            continue
        sa = student.model.layers[layer_idx].attn
        if not hasattr(sa, 'g_proj'):
            continue

        H, hd = sa.num_heads, sa.head_k_dim

        t_dev = next(hf_model.parameters()).device
        hi = t_out.hidden_states[layer_idx].to(t_dev).float()
        tln = hf_model.model.layers[layer_idx].input_layernorm
        x_t = tln(hi.to(tln.weight.dtype)).float()

        v_t = hf_model.model.layers[layer_idx].self_attn.v_proj(
            x_t.to(hf_model.dtype)
        ).float().view(B, T, num_kv, hd).permute(0, 2, 1, 3)
        v_t = v_t.repeat_interleave(gqa_g, dim=1)
        aw = t_out.attentions[layer_idx].to(t_dev).float()
        teacher_pre_o = (aw @ v_t).permute(0, 2, 1, 3).reshape(B, T, -1)
        teacher_out = hf_model.model.layers[layer_idx].self_attn.o_proj(
            teacher_pre_o.to(hf_model.dtype)
        ).float().detach().to(device)

        qkv_params = [sa.q_proj.weight, sa.k_proj.weight, sa.v_proj.weight]
        new_params = [p for n, p in sa.named_parameters()
                      if p.requires_grad and not any(p is q for q in qkv_params)
                      and 'o_proj' not in n and 'o_norm' not in n]

        for p in sa.parameters():
            p.requires_grad_(False)
        for p in qkv_params:
            p.requires_grad_(True)
        for p in new_params:
            p.requires_grad_(True)

        optimizer = torch.optim.AdamW([
            {"params": qkv_params, "lr": lr_qkv},
            {"params": new_params, "lr": lr_new},
        ], weight_decay=0.01)

        x_input = x_t.to(device).to(sa.q_proj.weight.dtype).detach()
        best_loss = float('inf')
        best_state = None

        for step in range(n_steps):
            optimizer.zero_grad()
            student_out, _, _ = sa(x_input, attention_mask=None)
            student_out = student_out.float()
            loss = F.mse_loss(student_out, teacher_out)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(qkv_params + new_params, 1.0)
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {n: p.data.clone() for n, p in sa.named_parameters()}

            if step % 100 == 0 or step == n_steps - 1:
                with torch.no_grad():
                    cos = F.cosine_similarity(
                        student_out.reshape(-1, student_out.shape[-1]),
                        teacher_out.reshape(-1, teacher_out.shape[-1]),
                        dim=-1,
                    ).mean().item()
                print(f"    Layer {layer_idx:2d} step {step:3d}: loss={loss.item():.6f}  cos={cos:.4f}")

        if best_state is not None:
            for n, p in sa.named_parameters():
                if n in best_state:
                    p.data.copy_(best_state[n])

        for p in sa.parameters():
            p.requires_grad_(False)

        with torch.no_grad():
            final_out, _, _ = sa(x_input, attention_mask=None)
            final_cos = F.cosine_similarity(
                final_out.float().reshape(-1, final_out.shape[-1]),
                teacher_out.reshape(-1, teacher_out.shape[-1]),
                dim=-1,
            ).mean().item()
            final_loss = F.mse_loss(final_out.float(), teacher_out).item()
        print(f"    Layer {layer_idx:2d} FINAL: loss={final_loss:.6f}  cos={final_cos:.4f}")

    print("\n  Taylor + gradient fine-tuning done.")


def apply_alignment_only(student, hf_model, tokenizer, cfg, device,
                         n_steps=300, lr_new=1e-3, lr_qkv=1e-4):
    """Phase 2 gradient alignment only (no Phase 1 analytical calibration).

    Skips gate calibration, OLS sigma, and g_proj-from-v_proj.
    Directly runs per-layer gradient fine-tuning from baseline-initialized weights.
    """
    import torch.nn.functional as F

    keep_layers = set(cfg["student_model"].get("keep_full_attention_layers", []))
    input_ids, attention_mask = _get_calibration_data(tokenizer, device, seqlen=256, num_samples=4)

    with torch.no_grad():
        t_out = hf_model(
            input_ids=input_ids, attention_mask=attention_mask,
            output_attentions=True, output_hidden_states=True, use_cache=False,
        )
    _offload_t_out(t_out)

    num_kv = hf_model.config.num_key_value_heads
    num_q = hf_model.config.num_attention_heads
    gqa_g = num_q // num_kv

    hf_model.cpu()
    torch.cuda.empty_cache()
    student.to(device)

    # No Phase 1 — skip analytical calibration entirely
    print("=" * 60)
    print("  Phase 1: SKIPPED (alignment-only mode)")
    print("=" * 60)

    # Phase 2: Per-layer gradient fine-tuning (same as taylor_calibrate)
    print("\n" + "=" * 60)
    print("  Phase 2: Per-layer gradient fine-tuning")
    print("=" * 60)
    B, T = input_ids.shape

    for layer_idx in range(len(student.model.layers)):
        if layer_idx in keep_layers:
            continue
        sa = student.model.layers[layer_idx].attn
        if not hasattr(sa, 'g_proj'):
            continue

        H, hd = sa.num_heads, sa.head_k_dim

        t_dev = next(hf_model.parameters()).device
        hi = t_out.hidden_states[layer_idx].to(t_dev).float()
        tln = hf_model.model.layers[layer_idx].input_layernorm
        x_t = tln(hi.to(tln.weight.dtype)).float()

        v_t = hf_model.model.layers[layer_idx].self_attn.v_proj(
            x_t.to(hf_model.dtype)
        ).float().view(B, T, num_kv, hd).permute(0, 2, 1, 3)
        v_t = v_t.repeat_interleave(gqa_g, dim=1)
        aw = t_out.attentions[layer_idx].to(t_dev).float()
        teacher_pre_o = (aw @ v_t).permute(0, 2, 1, 3).reshape(B, T, -1)
        teacher_out = hf_model.model.layers[layer_idx].self_attn.o_proj(
            teacher_pre_o.to(hf_model.dtype)
        ).float().detach().to(device)

        qkv_params = [sa.q_proj.weight, sa.k_proj.weight, sa.v_proj.weight]
        new_params = [p for n, p in sa.named_parameters()
                      if p.requires_grad and not any(p is q for q in qkv_params)
                      and 'o_proj' not in n and 'o_norm' not in n]

        for p in sa.parameters():
            p.requires_grad_(False)
        for p in qkv_params:
            p.requires_grad_(True)
        for p in new_params:
            p.requires_grad_(True)

        optimizer = torch.optim.AdamW([
            {"params": qkv_params, "lr": lr_qkv},
            {"params": new_params, "lr": lr_new},
        ], weight_decay=0.01)

        x_input = x_t.to(device).to(sa.q_proj.weight.dtype).detach()
        best_loss = float('inf')
        best_state = None

        for step in range(n_steps):
            optimizer.zero_grad()
            student_out, _, _ = sa(x_input, attention_mask=None)
            student_out = student_out.float()
            loss = F.mse_loss(student_out, teacher_out)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(qkv_params + new_params, 1.0)
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {n: p.data.clone() for n, p in sa.named_parameters()}

            if step % 100 == 0 or step == n_steps - 1:
                with torch.no_grad():
                    cos = F.cosine_similarity(
                        student_out.reshape(-1, student_out.shape[-1]),
                        teacher_out.reshape(-1, teacher_out.shape[-1]),
                        dim=-1,
                    ).mean().item()
                print(f"    Layer {layer_idx:2d} step {step:3d}: loss={loss.item():.6f}  cos={cos:.4f}")

        if best_state is not None:
            for n, p in sa.named_parameters():
                if n in best_state:
                    p.data.copy_(best_state[n])

        for p in sa.parameters():
            p.requires_grad_(False)

        with torch.no_grad():
            final_out, _, _ = sa(x_input, attention_mask=None)
            final_cos = F.cosine_similarity(
                final_out.float().reshape(-1, final_out.shape[-1]),
                teacher_out.reshape(-1, teacher_out.shape[-1]),
                dim=-1,
            ).mean().item()
            final_loss = F.mse_loss(final_out.float(), teacher_out).item()
        print(f"    Layer {layer_idx:2d} FINAL: loss={final_loss:.6f}  cos={final_cos:.4f}")

    print("\n  Alignment-only gradient fine-tuning done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--strategy", choices=STRATEGIES, default="zero_gate")
    parser.add_argument("--output", default=None)
    parser.add_argument("--hf-teacher", default="Qwen/Qwen2.5-1.5B-Instruct")
    args = parser.parse_args()

    cfg = parse_config(args.cfg)
    suffix = f"init-{args.strategy.replace('_', '-')}"
    output_dir = args.output or os.path.join(cfg["train"]["output_dir"], suffix)
    os.makedirs(output_dir, exist_ok=True)

    keep_layers = set(cfg["student_model"].get("keep_full_attention_layers", []))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Strategy: {args.strategy}")
    print("Building student from FLA teacher...")
    student = build_student_from_teacher(cfg)
    student = student.eval()

    if args.strategy == "zero_gate":
        apply_zero_gate(student, keep_layers)

    elif args.strategy == "zero_all":
        apply_zero_all(student, keep_layers)

    elif args.strategy == "small_gate":
        apply_small_gate(student, keep_layers)

    elif args.strategy == "zero_calibrate":
        print(f"Loading HF teacher: {args.hf_teacher}")
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.hf_teacher, torch_dtype=torch.bfloat16, attn_implementation="eager",
        ).to(device).eval()
        tokenizer_hf = AutoTokenizer.from_pretrained(args.hf_teacher)
        if tokenizer_hf.pad_token is None:
            tokenizer_hf.pad_token = tokenizer_hf.eos_token
        apply_zero_calibrate(student, keep_layers, hf_model, tokenizer_hf, device)
        del hf_model
        torch.cuda.empty_cache()

    elif args.strategy in ("calibrate_gproj", "calibrate_gproj_reg", "calibrate_gates_gproj",
                           "gproj_from_v", "gproj_from_o", "calib_gates_gproj_v",
                           "ols_calibrate", "taylor_calibrate_init", "small_gate_calibrate",
                           "layer_distill", "taylor_calibrate", "alignment_only"):
        student.cpu()
        torch.cuda.empty_cache()
        print(f"Loading HF teacher: {args.hf_teacher}")
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.hf_teacher, torch_dtype=torch.bfloat16, attn_implementation="eager",
        ).to(device).eval()
        tokenizer_hf = AutoTokenizer.from_pretrained(args.hf_teacher)
        if tokenizer_hf.pad_token is None:
            tokenizer_hf.pad_token = tokenizer_hf.eos_token
        dispatch = {
            "calibrate_gproj": apply_calibrate_gproj,
            "calibrate_gproj_reg": apply_calibrate_gproj_reg,
            "calibrate_gates_gproj": apply_calibrate_gates_gproj,
            "gproj_from_v": apply_gproj_from_v,
            "gproj_from_o": apply_gproj_from_o,
            "calib_gates_gproj_v": apply_calib_gates_gproj_v,
            "ols_calibrate": apply_ols_calibrate,
            "taylor_calibrate_init": apply_taylor_calibrate_init,
            "small_gate_calibrate": apply_small_gate_calibrate,
            "layer_distill": apply_layer_distill,
            "taylor_calibrate": apply_taylor_calibrate,
            "alignment_only": apply_alignment_only,
        }
        if args.strategy not in ("taylor_calibrate", "alignment_only"):
            student.to(device)
        dispatch[args.strategy](student, hf_model, tokenizer_hf, cfg, device)
        del hf_model
        torch.cuda.empty_cache()

    print(f"\nSaving to {output_dir}")
    student = student.cpu()
    student.save_pretrained(output_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg["teacher_model"]["name"])
    tokenizer.save_pretrained(output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
