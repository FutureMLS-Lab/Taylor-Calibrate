#!/usr/bin/env python3
"""GA-S2 layer selection via PPL probing with configurable init strategy.

For each layer i, builds a student with only layer i as full attention,
applies the chosen initialization strategy, and measures zero-shot PPL.
Ranks layers by PPL (lower = better = more important to keep).

Usage:
  # Probe with taylor_calibrate (original behavior)
  CUDA_VISIBLE_DEVICES=0 python greedy_layer_selection.py \
    --cfg configs/qwen2_1.5b/qwen2_1_5b_gdn_v4.yaml \
    --hf-teacher Qwen/Qwen2.5-1.5B-Instruct \
    --strategy taylor_calibrate --layers 0,1,2,3 \
    --output results/ga_s2_calibrate_distill_gpu0.json

  # Probe with zero_gate (fast, no teacher needed for init)
  CUDA_VISIBLE_DEVICES=0 python greedy_layer_selection.py \
    --cfg configs/qwen2_1.5b/qwen2_1_5b_gdn_v4.yaml \
    --hf-teacher Qwen/Qwen2.5-1.5B-Instruct \
    --strategy zero_gate --layers 0,1,2,3 \
    --output results/ga_s2_zero_gate_gpu0.json
"""

import argparse
import gc
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
import fla  # noqa: F401
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from distill_model.config_distilled_student import StudentConfig
from distill_model.modeling_distilled_student import StudentForCausalLM
from init_ckpt_from_teacher import build_student_from_teacher, parse_config
from init_ckpt_calibrated import (
    apply_zero_gate, apply_small_gate, _taylor_analytical_init,
)

AutoConfig.register("student", StudentConfig, exist_ok=True)
AutoModelForCausalLM.register(StudentConfig, StudentForCausalLM, exist_ok=True)

STRATEGIES = ["baseline", "zero_gate", "small_gate", "taylor_calibrate_init", "taylor_calibrate"]
NEEDS_HF_TEACHER = {"taylor_calibrate_init", "taylor_calibrate"}


def get_calibration_data(tokenizer, device, seqlen=256, num_samples=8):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join([t for t in ds["text"] if len(t) > 100])
    enc = tokenizer(text, return_tensors="pt")
    ids = enc.input_ids[:, : seqlen * num_samples].view(num_samples, seqlen).to(device)
    mask = torch.ones_like(ids)
    return ids, mask


def apply_taylor_calibrate_fast(student, hf_model, tokenizer, cfg, device,
                               n_steps=300, lr_qkv=1e-4, lr_new=3e-3,
                               calib_seqlen=256, calib_samples=8):
    """Streamlined taylor_calibrate: analytical init + per-layer gradient fine-tuning."""
    keep_layers = set(cfg["student_model"].get("keep_full_attention_layers", []))
    input_ids, attention_mask = get_calibration_data(tokenizer, device,
                                                     seqlen=calib_seqlen,
                                                     num_samples=calib_samples)

    with torch.no_grad():
        t_out = hf_model(
            input_ids=input_ids, attention_mask=attention_mask,
            output_attentions=True, output_hidden_states=True, use_cache=False,
        )

    from init_ckpt_calibrated import _taylor_analytical_init
    _taylor_analytical_init(student, hf_model, t_out, cfg, device)

    num_kv = hf_model.config.num_key_value_heads
    num_q = hf_model.config.num_attention_heads
    gqa_g = num_q // num_kv
    B, T = input_ids.shape

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

        v_t = hf_model.model.layers[layer_idx].self_attn.v_proj(
            x.to(hf_model.dtype)
        ).float().view(B, T, num_kv, hd).permute(0, 2, 1, 3)
        v_t = v_t.repeat_interleave(gqa_g, dim=1)
        aw = t_out.attentions[layer_idx].float()
        teacher_pre_o = (aw @ v_t).permute(0, 2, 1, 3).reshape(B, T, -1)
        teacher_out = hf_model.model.layers[layer_idx].self_attn.o_proj(
            teacher_pre_o.to(hf_model.dtype)
        ).float().detach()

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

        if best_state is not None:
            for n, p in sa.named_parameters():
                if n in best_state:
                    p.data.copy_(best_state[n])

        for p in sa.parameters():
            p.requires_grad_(False)

        del optimizer, x_input, teacher_out, best_state
        torch.cuda.empty_cache()

    del t_out
    torch.cuda.empty_cache()


@torch.no_grad()
def evaluate_ppl_fast(model, tokenizer, seqlen=2048, device='cuda'):
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    text = "\n\n".join(ds['text'])
    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings.input_ids.to(device)

    total_nll = 0.0
    total_tokens = 0
    n_chunks = input_ids.size(1) // seqlen

    for i in range(n_chunks):
        chunk = input_ids[:, i * seqlen:(i + 1) * seqlen]
        outputs = model(chunk)
        logits = outputs.logits[:, :-1, :].contiguous()
        targets = chunk[:, 1:].contiguous()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                               targets.view(-1), reduction='sum')
        total_nll += loss.item()
        total_tokens += targets.numel()

    return torch.exp(torch.tensor(total_nll / total_tokens)).item()


def probe_layer(layer_idx, cfg, hf_teacher_name, device, strategy="taylor_calibrate",
                low_mem=False):
    """Build student keeping only layer_idx as full attention, apply strategy init, eval PPL."""
    probe_cfg = {
        "teacher_model": cfg["teacher_model"],
        "student_model": {
            "name": cfg["student_model"]["name"],
            "keep_full_attention_layers": [layer_idx],
        },
        "train": cfg.get("train", {}),
    }

    print(f"\n{'='*60}")
    print(f"  Probing layer {layer_idx} | strategy={strategy}")
    print(f"{'='*60}")

    t0 = time.time()
    student = build_student_from_teacher(probe_cfg)
    student = student.to(device)
    keep_layers = {layer_idx}

    tokenizer = AutoTokenizer.from_pretrained(hf_teacher_name)

    if strategy in NEEDS_HF_TEACHER:
        hf_model = AutoModelForCausalLM.from_pretrained(
            hf_teacher_name, torch_dtype=torch.bfloat16, attn_implementation="eager",
        ).to(device).eval()

    if strategy == "baseline":
        pass
    elif strategy == "zero_gate":
        apply_zero_gate(student, keep_layers)
    elif strategy == "small_gate":
        apply_small_gate(student, keep_layers)
    elif strategy == "taylor_calibrate_init":
        input_ids, attention_mask = get_calibration_data(tokenizer, device)
        with torch.no_grad():
            t_out = hf_model(
                input_ids=input_ids, attention_mask=attention_mask,
                output_attentions=True, output_hidden_states=True, use_cache=False,
            )
        _taylor_analytical_init(student, hf_model, t_out, probe_cfg, device)
        del t_out
    elif strategy == "taylor_calibrate":
        kwargs = {}
        if low_mem:
            kwargs = dict(n_steps=100, calib_seqlen=128, calib_samples=4)
        apply_taylor_calibrate_fast(student, hf_model, tokenizer, probe_cfg, device, **kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    init_time = time.time() - t0
    print(f"  Init ({strategy}) done in {init_time:.1f}s")

    if strategy in NEEDS_HF_TEACHER:
        del hf_model
    gc.collect()
    torch.cuda.empty_cache()

    student.eval()
    t1 = time.time()
    ppl = evaluate_ppl_fast(student, tokenizer, device=device)
    eval_time = time.time() - t1
    print(f"  Layer {layer_idx}: PPL = {ppl:.2f}  (eval {eval_time:.1f}s)")

    del student, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return {"layer": layer_idx, "ppl": ppl, "init_time": init_time, "eval_time": eval_time}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="YAML config for the model")
    parser.add_argument("--hf-teacher", required=True, help="HF teacher model name/path")
    parser.add_argument("--strategy", default="taylor_calibrate", choices=STRATEGIES,
                        help="Init strategy for probing (default: taylor_calibrate)")
    parser.add_argument("--layers", default=None,
                        help="Comma-separated layer indices (e.g. '0,1,2,3'). Default: all")
    parser.add_argument("--num-layers", type=int, default=28, help="Total layers in model")
    parser.add_argument("--top-k", type=int, default=7, help="Number of layers to select")
    parser.add_argument("--output", default="greedy_layer_results.json")
    parser.add_argument("--low-mem", action="store_true",
                        help="Reduce calibration data for taylor_calibrate to avoid OOM on large models")
    args = parser.parse_args()

    cfg = parse_config(args.cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Strategy: {args.strategy}")
    print(f"Teacher (FLA): {cfg['teacher_model']['name']}")
    print(f"Teacher (HF):  {args.hf_teacher}")

    if args.layers:
        layer_indices = [int(x) for x in args.layers.split(",")]
    else:
        layer_indices = list(range(args.num_layers))

    results = []
    existing = {}
    if os.path.exists(args.output):
        with open(args.output) as f:
            existing_data = json.load(f)
        existing = {r["layer"]: r for r in existing_data.get("results", [])}
        print(f"Loaded {len(existing)} existing results from {args.output}")

    for idx in layer_indices:
        if idx in existing:
            print(f"  Layer {idx} already done: PPL={existing[idx]['ppl']:.2f}, skipping")
            results.append(existing[idx])
            continue
        result = probe_layer(idx, cfg, args.hf_teacher, device, strategy=args.strategy,
                             low_mem=args.low_mem)
        results.append(result)

        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        out = {"results": sorted(results, key=lambda x: x["layer"])}
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)

    results_sorted = sorted(results, key=lambda x: x["ppl"])
    print(f"\n{'='*60}")
    print(f"  RANKING [{args.strategy}] (lower PPL = more important to keep)")
    print(f"{'='*60}")
    for rank, r in enumerate(results_sorted, 1):
        print(f"  #{rank:2d}  Layer {r['layer']:2d}  PPL = {r['ppl']:.2f}")

    top_k = [r["layer"] for r in results_sorted[:args.top_k]]
    top_k_sorted = sorted(top_k)
    print(f"\n  Top-{args.top_k} layers (sorted): {top_k_sorted}")
    print(f"  keep_full_attention_layers: {top_k_sorted}")

    out = {
        "results": sorted(results, key=lambda x: x["layer"]),
        "ranking": [{"rank": i+1, "layer": r["layer"], "ppl": r["ppl"]}
                     for i, r in enumerate(results_sorted)],
        "top_k": top_k_sorted,
        "config": {
            "teacher": cfg["teacher_model"]["name"],
            "hf_teacher": args.hf_teacher,
            "student": cfg["student_model"]["name"],
            "strategy": args.strategy,
            "top_k": args.top_k,
        }
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved to {args.output}")


if __name__ == "__main__":
    main()
