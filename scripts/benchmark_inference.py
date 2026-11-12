#!/usr/bin/env python3
"""Prefill latency, decode throughput, and memory: softmax teacher vs. hybrid student.

Token ids are random (content does not change the cost). Both models run their
native caches: KV for softmax layers, recurrent state for GDN layers. Logits
are computed for the last position only, so vocabulary projection over the
prompt does not dominate prefill memory. Peak memory is
torch.cuda.max_memory_allocated over prefill plus decode.

The student can be loaded from a checkpoint (--student) or built from the
converted FLA teacher with a retained-layer list (--student-fla-teacher,
--keep); weights do not affect speed.

Example:
  PYTHONPATH=src:. python scripts/benchmark_inference.py \
    --teacher Qwen/Qwen2.5-1.5B-Instruct \
    --student-fla-teacher ./teachers/Qwen2.5-1.5B-Instruct --keep 0,4,8,12,16,20,24 \
    --contexts 2048,8192,32768 --batch-sizes 1,16 --output results/benchmark.jsonl
"""

import argparse
import json
import os
import statistics
import time

import fla  # noqa: F401
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from distill_model.config_distilled_student import StudentConfig
from distill_model.modeling_distilled_student import StudentForCausalLM
from init_ckpt_from_teacher import build_student_from_teacher

AutoConfig.register("student", StudentConfig, exist_ok=True)
AutoModelForCausalLM.register(StudentConfig, StudentForCausalLM, exist_ok=True)


def tensor_bytes(obj, seen=None):
    seen = set() if seen is None else seen
    if torch.is_tensor(obj):
        if obj.data_ptr() in seen:
            return 0
        seen.add(obj.data_ptr())
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(tensor_bytes(v, seen) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(tensor_bytes(v, seen) for v in obj)
    if hasattr(obj, "__dict__"):
        return tensor_bytes(vars(obj), seen)
    return 0


@torch.no_grad()
def prefill(model, ids):
    out = model.model(input_ids=ids, use_cache=True)
    next_tok = model.lm_head(out.last_hidden_state[:, -1:]).argmax(-1)
    return out.past_key_values, next_tok


@torch.no_grad()
def decode(model, cache, next_tok, steps):
    for _ in range(steps):
        out = model.model(input_ids=next_tok, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        next_tok = model.lm_head(out.last_hidden_state[:, -1:]).argmax(-1)
    return cache


def timed(fn, *args):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn(*args)
    torch.cuda.synchronize()
    return result, time.perf_counter() - start


def measure(model, vocab, context, batch, decode_steps, repeats):
    ids = torch.randint(0, vocab, (batch, context), device="cuda")
    for _ in range(2):
        cache, tok = prefill(model, ids)
        decode(model, cache, tok, 8)
        del cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()

    prefill_s, decode_s = [], []
    for _ in range(repeats):
        (cache, tok), t_pre = timed(prefill, model, ids)
        cache_bytes = tensor_bytes(cache)
        cache, t_dec = timed(decode, model, cache, tok, decode_steps)
        prefill_s.append(t_pre)
        decode_s.append(t_dec)
        del cache
    peak = torch.cuda.max_memory_allocated()
    return {
        "context": context,
        "batch": batch,
        "prefill_ms": 1e3 * statistics.median(prefill_s),
        "prefill_tok_per_s": batch * context / statistics.median(prefill_s),
        "decode_steps": decode_steps,
        "decode_ms_per_step": 1e3 * statistics.median(decode_s) / decode_steps,
        "decode_tok_per_s": batch * decode_steps / statistics.median(decode_s),
        "cache_after_prefill_gb": cache_bytes / 2**30,
        "peak_gb": peak / 2**30,
        "peak_over_weights_gb": (peak - base) / 2**30,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, help="HF softmax teacher")
    ap.add_argument("--student", default=None, help="student checkpoint directory")
    ap.add_argument("--student-fla-teacher", default=None)
    ap.add_argument("--student-name", default="gdn_v4")
    ap.add_argument("--keep", default=None, help="retained softmax layers, e.g. 0,4,8")
    ap.add_argument("--contexts", default="2048,8192,32768")
    ap.add_argument("--batch-sizes", default="1,16")
    ap.add_argument("--decode-steps", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--attn-implementation", default="flash_attention_2")
    ap.add_argument("--max-batch-tokens", type=int, default=262144,
                    help="skip context x batch above this (FLA kernels fault on larger prefills)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if args.student is None and not (args.student_fla_teacher and args.keep):
        raise SystemExit("pass --student or both --student-fla-teacher and --keep")

    contexts = [int(x) for x in args.contexts.split(",")]
    batches = [int(x) for x in args.batch_sizes.split(",")]
    gpu = torch.cuda.get_device_name()

    def load(kind):
        if kind == "teacher":
            return AutoModelForCausalLM.from_pretrained(
                args.teacher, torch_dtype=torch.bfloat16,
                attn_implementation=args.attn_implementation,
            )
        if args.student:
            return AutoModelForCausalLM.from_pretrained(args.student, torch_dtype=torch.bfloat16)
        keep = [int(x) for x in args.keep.split(",")]
        return build_student_from_teacher({
            "teacher_model": {"name": args.student_fla_teacher},
            "student_model": {"name": args.student_name, "keep_full_attention_layers": keep},
        })

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    for kind in ("teacher", "student"):
        model = load(kind).cuda().eval()
        vocab = model.config.vocab_size
        n_layers = model.config.num_hidden_layers
        keep = getattr(model.config, "keep_full_attention_layers", list(range(n_layers)))
        for context in contexts:
            for batch in batches:
                if context * batch > args.max_batch_tokens:
                    continue
                try:
                    row = measure(model, vocab, context, batch, args.decode_steps, args.repeats)
                except torch.cuda.OutOfMemoryError:
                    row = {"context": context, "batch": batch, "oom": True}
                    torch.cuda.empty_cache()
                row.update(model=kind, softmax_layers=len(keep), layers=n_layers, gpu=gpu,
                           teacher=args.teacher)
                print(json.dumps(row))
                with open(args.output, "a") as handle:
                    handle.write(json.dumps(row) + "\n")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
