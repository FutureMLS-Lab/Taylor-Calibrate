#!/usr/bin/env python3
"""Held-out teacher-fidelity controls for Phase-1 calibration choices.

Converts a fixed set of teacher layers to the student operator, applies one
initialization variant per run, and reports on held-out WikiText-2 validation
text: converted-layer attention-output MSE and cosine under teacher-forced
inputs, token-level KL(p_T || p_S), and perplexity on a separate held-out
stream. Calibration uses the WikiText-2 test prefix (init_ckpt_calibrated).

Results are appended to a JSONL file; finished (variant, seed) pairs are
skipped on restart.

Example:
  PYTHONPATH=src:. python scripts/fidelity_control.py \
    --teacher-fla ./teachers/Qwen2.5-1.5B-Instruct \
    --hf-teacher Qwen/Qwen2.5-1.5B-Instruct \
    --seeds 0,1,2 --output results/fidelity_control/qwen25_1p5b.jsonl
"""

import argparse
import json
import math
import os
import statistics
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

import init_ckpt_calibrated as icc
from init_ckpt_from_teacher import build_student_from_teacher

VARIANTS = {
    "projection_copy": (None, {}),
    "taylor_phase1": ("taylor_calibrate_init", {}),
    "alignment_only": ("alignment_only", {}),
    "output_only": ("output_only", {}),
    "rms_ols": ("rms_ols", {}),
    "taylor_calibrate": ("taylor_calibrate", {}),
    "shuffled_stats": ("taylor_calibrate", {"stat_perturb": "shuffle"}),
    "constant_half_life": ("taylor_calibrate", {"stat_perturb": "const_hl"}),
    "constant_beta": ("taylor_calibrate", {"stat_perturb": "const_beta"}),
}

DISPATCH = {
    "taylor_calibrate_init": icc.apply_taylor_calibrate_init,
    "alignment_only": icc.apply_alignment_only,
    "output_only": icc.apply_output_only,
    "rms_ols": icc.apply_rms_ols,
    "taylor_calibrate": icc.apply_taylor_calibrate,
}

METRICS = ("mse", "cosine", "kl", "ppl")


def validation_ids(tokenizer):
    from datasets import load_dataset
    ds = load_dataset(
        "wikitext", "wikitext-2-raw-v1", split="validation",
        revision=icc.WIKITEXT_REVISION,
    )
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    return tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]


@torch.no_grad()
def teacher_reference(hf_model, fid_ids, layers):
    out = hf_model(
        input_ids=fid_ids, output_attentions=True, output_hidden_states=True,
        use_cache=False,
    )
    cfg = hf_model.config
    num_kv = cfg.num_key_value_heads
    gqa = cfg.num_attention_heads // num_kv
    refs = {}
    for li in layers:
        tl = hf_model.model.layers[li]
        x = tl.input_layernorm(out.hidden_states[li])
        B, T, _ = x.shape
        v = tl.self_attn.v_proj(x).float()
        v = v.view(B, T, num_kv, -1).transpose(1, 2).repeat_interleave(gqa, dim=1)
        ctx = (out.attentions[li].float() @ v).transpose(1, 2).reshape(B, T, -1)
        attn_out = tl.self_attn.o_proj(ctx.to(x.dtype)).float()
        refs[li] = (x.float().cpu(), attn_out.cpu())
    return refs, out.logits.to(torch.bfloat16).cpu()


@torch.no_grad()
def layer_metrics(student, refs, device):
    mses, coss = [], []
    for li, (x, t_attn) in refs.items():
        sa = student.model.layers[li].attn
        s_attn = sa(x.to(device).to(sa.q_proj.weight.dtype), attention_mask=None)[0].float()
        t_attn = t_attn.to(device)
        mses.append(F.mse_loss(s_attn, t_attn).item())
        coss.append(F.cosine_similarity(
            s_attn.flatten(0, 1), t_attn.flatten(0, 1), dim=-1).mean().item())
    return statistics.mean(mses), statistics.mean(coss)


@torch.no_grad()
def kl_to_teacher(student, fid_ids, t_logits, device):
    total, count = 0.0, 0
    for i in range(fid_ids.shape[0]):
        s = student(input_ids=fid_ids[i:i + 1]).logits.float()
        t = t_logits[i:i + 1].to(device).float()
        kl = F.kl_div(F.log_softmax(s, -1), F.log_softmax(t, -1),
                      log_target=True, reduction="none").sum(-1)
        total += kl.sum().item()
        count += kl.numel()
    return total / count


@torch.no_grad()
def perplexity(model, stream, window, device):
    nll, count = 0.0, 0
    for start in range(0, stream.numel(), window):
        ids = stream[start:start + window].unsqueeze(0).to(device)
        if ids.shape[1] < 2:
            break
        logits = model(input_ids=ids).logits.float()
        nll += F.cross_entropy(logits[0, :-1], ids[0, 1:], reduction="sum").item()
        count += ids.shape[1] - 1
    return math.exp(nll / count)


def summarize(path):
    rows = [json.loads(line) for line in open(path)]
    print(f"\n{'variant':<20}{'n':>3}" + "".join(f"{m:>22}" for m in METRICS))
    for name in VARIANTS:
        group = [r for r in rows if r["variant"] == name]
        if not group:
            continue
        cells = []
        for m in METRICS:
            vals = [r[m] for r in group]
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            cells.append(f"{statistics.mean(vals):>13.4f} ± {sd:<6.4f}")
        print(f"{name:<20}{len(group):>3}" + "".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-fla", required=True)
    ap.add_argument("--hf-teacher", required=True)
    ap.add_argument("--student", default="gdn_v4")
    ap.add_argument("--converted", default="3,7,11,15,19,23,27")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--const-half-life", type=float, default=71.0)
    ap.add_argument("--fid-seqs", type=int, default=8)
    ap.add_argument("--fid-len", type=int, default=512)
    ap.add_argument("--ppl-tokens", type=int, default=32768)
    ap.add_argument("--ppl-window", type=int, default=2048)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    device = "cuda"
    converted = [int(x) for x in args.converted.split(",")]
    variants = args.variants.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}")

    tokenizer = AutoTokenizer.from_pretrained(args.hf_teacher)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.hf_teacher, torch_dtype=torch.bfloat16, attn_implementation="eager",
    ).to(device).eval()
    keep = [i for i in range(hf_model.config.num_hidden_layers) if i not in converted]
    cfg = {
        "teacher_model": {"name": args.teacher_fla},
        "student_model": {"name": args.student, "keep_full_attention_layers": keep},
        "train": {},
    }

    ids = validation_ids(tokenizer)
    n_fid = args.fid_seqs * args.fid_len
    if ids.numel() < n_fid + args.ppl_tokens:
        raise SystemExit(f"validation split has only {ids.numel()} tokens")
    fid_ids = ids[:n_fid].view(args.fid_seqs, args.fid_len).to(device)
    stream = ids[n_fid:n_fid + args.ppl_tokens]

    refs, t_logits = teacher_reference(hf_model, fid_ids, converted)
    teacher_ppl = perplexity(hf_model, stream, args.ppl_window, device)
    print(f"teacher PPL on {stream.numel()} held-out tokens: {teacher_ppl:.3f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    done = set()
    if os.path.exists(args.output):
        for line in open(args.output):
            r = json.loads(line)
            done.add((r["variant"], r["seed"]))

    for seed in seeds:
        for name in variants:
            if (name, seed) in done:
                continue
            strategy, switches = VARIANTS[name]
            icc.CALIB_OPTIONS.clear()
            icc.CALIB_OPTIONS.update(icc.DEFAULT_CALIB_OPTIONS)
            icc.CALIB_OPTIONS.update(
                switches, perturb_seed=seed, const_half_life=args.const_half_life)
            print(f"\n######## variant={name} seed={seed} ########")
            set_seed(seed)
            student = build_student_from_teacher(cfg).eval()
            start = time.perf_counter()
            if strategy is not None:
                if strategy not in icc.LOCAL_ALIGNMENT_STRATEGIES:
                    student.to(device)
                DISPATCH[strategy](student, hf_model, tokenizer, cfg, device)
            init_seconds = time.perf_counter() - start
            hf_model.to(device)
            student.to(device).eval()
            for p in student.parameters():
                p.requires_grad_(False)

            mse, cosine = layer_metrics(student, refs, device)
            record = {
                "variant": name,
                "seed": seed,
                "strategy": strategy,
                "switches": switches,
                "converted_layers": converted,
                "mse": mse,
                "cosine": cosine,
                "kl": kl_to_teacher(student, fid_ids, t_logits, device),
                "ppl": perplexity(student, stream, args.ppl_window, device),
                "teacher_ppl": teacher_ppl,
                "init_seconds": init_seconds,
                "eval": {
                    "fidelity_tokens": n_fid,
                    "fidelity_seq_len": args.fid_len,
                    "ppl_tokens": stream.numel(),
                    "ppl_window": args.ppl_window,
                    "split": "wikitext-2-raw-v1/validation",
                },
            }
            with open(args.output, "a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record))
            del student
            torch.cuda.empty_cache()

    summarize(args.output)


if __name__ == "__main__":
    main()
