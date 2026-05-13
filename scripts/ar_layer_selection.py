#!/usr/bin/env python3
"""LM-PPL based layer selection (used as AR proxy).

For each layer in the teacher model, bypasses that layer's attention
(zeros the mixing output, keeps only residual) and measures the
resulting PPL increase on Wikitext-2. Layers causing the largest PPL
increase when bypassed are the most important to keep as softmax.

This is init-independent (operates on the teacher, no student involved).

Usage:
  # Run all 28 layers across 8 GPUs
  for GPU in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$GPU python scripts/ar_layer_selection.py \
      --model Qwen/Qwen2.5-1.5B-Instruct \
      --layers $((GPU*4)),$((GPU*4+1)),$((GPU*4+2)),$((GPU*4+3)) \
      --output results/ar_lmppl_1.5b_gpu${GPU}.json &
  done
  wait

  # Merge and rank
  python scripts/ar_layer_selection.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --merge-only --output results/ar_lmppl_1.5b.json \
    results/ar_lmppl_1.5b_gpu*.json
"""

import argparse
import gc
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def evaluate_ppl(model, tokenizer, seqlen=2048, device="cuda"):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test",
                      cache_dir=os.environ.get("HF_DATASETS_CACHE", None))
    text = "\n\n".join(ds["text"])
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(device)

    total_nll = 0.0
    total_tokens = 0
    n_chunks = input_ids.size(1) // seqlen

    for i in range(n_chunks):
        chunk = input_ids[:, i * seqlen : (i + 1) * seqlen]
        outputs = model(chunk)
        logits = outputs.logits[:, :-1, :].contiguous()
        targets = chunk[:, 1:].contiguous()
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)).float(),
            targets.view(-1),
            reduction="sum",
        )
        total_nll += loss.item()
        total_tokens += targets.numel()

    return torch.exp(torch.tensor(total_nll / total_tokens)).item()


def bypass_layer_attention(model, layer_idx):
    """Register a hook that zeros the attention output for a given layer.
    The residual connection in the transformer block preserves the input,
    effectively bypassing that layer's attention mixing."""
    layer = model.model.layers[layer_idx]
    handle = layer.self_attn.register_forward_hook(
        lambda module, inp, out: (torch.zeros_like(out[0]),) + out[1:]
    )
    return handle


def probe_layer(model, tokenizer, layer_idx, baseline_ppl, device):
    """Bypass one layer's attention, measure PPL increase."""
    print(f"  Probing layer {layer_idx}...", end=" ", flush=True)
    t0 = time.time()

    handle = bypass_layer_attention(model, layer_idx)
    ppl = evaluate_ppl(model, tokenizer, device=device)
    handle.remove()

    delta = ppl - baseline_ppl
    elapsed = time.time() - t0
    print(f"PPL={ppl:.2f}  delta=+{delta:.2f}  ({elapsed:.1f}s)")

    return {
        "layer": layer_idx,
        "ppl_bypassed": ppl,
        "ppl_delta": delta,
        "time": elapsed,
    }


def merge_results(input_files, top_k=7):
    """Merge per-GPU JSON results into a single ranked file."""
    all_results = []
    for f in input_files:
        with open(f) as fh:
            data = json.load(fh)
        all_results.extend(data.get("results", []))

    seen = {}
    for r in all_results:
        seen[r["layer"]] = r
    merged = sorted(seen.values(), key=lambda x: x["layer"])

    ranked = sorted(merged, key=lambda x: -x["ppl_delta"])
    top_k_layers = sorted([r["layer"] for r in ranked[:top_k]])

    return {
        "results": merged,
        "ranking": [
            {"rank": i + 1, "layer": r["layer"],
             "ppl_bypassed": r["ppl_bypassed"], "ppl_delta": r["ppl_delta"]}
            for i, r in enumerate(ranked)
        ],
        "top_k": top_k_layers,
    }


def main():
    parser = argparse.ArgumentParser(description="LM-PPL layer importance scoring")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="HF teacher model name/path")
    parser.add_argument("--layers", default=None,
                        help="Comma-separated layer indices. Default: all")
    parser.add_argument("--num-layers", type=int, default=28)
    parser.add_argument("--top-k", type=int, default=7)
    parser.add_argument("--output", default="results/ar_lmppl_layers.json")
    parser.add_argument("--merge-only", action="store_true",
                        help="Merge existing per-GPU JSONs (pass them as positional args)")
    parser.add_argument("input_files", nargs="*", help="JSON files to merge (with --merge-only)")
    args = parser.parse_args()

    if args.merge_only:
        if not args.input_files:
            parser.error("--merge-only requires input JSON files as positional args")
        print(f"Merging {len(args.input_files)} files...")
        out = merge_results(args.input_files, top_k=args.top_k)
        out["config"] = {"model": args.model, "method": "lm_ppl", "top_k": args.top_k}
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)

        print(f"\n{'='*60}")
        print(f"  RANKING (higher delta = more important to keep as softmax)")
        print(f"{'='*60}")
        for r in out["ranking"]:
            print(f"  #{r['rank']:2d}  Layer {r['layer']:2d}  "
                  f"PPL_bypass={r['ppl_bypassed']:.2f}  delta=+{r['ppl_delta']:.2f}")
        print(f"\n  Top-{args.top_k} layers: {out['top_k']}")
        print(f"  Saved to {args.output}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading teacher: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print("Computing baseline PPL (no bypass)...")
    baseline_ppl = evaluate_ppl(model, tokenizer, device=device)
    print(f"  Baseline PPL = {baseline_ppl:.2f}")

    if args.layers:
        layer_indices = [int(x) for x in args.layers.split(",")]
    else:
        layer_indices = list(range(args.num_layers))

    results = []
    existing = {}
    if os.path.exists(args.output):
        with open(args.output) as f:
            data = json.load(f)
        existing = {r["layer"]: r for r in data.get("results", [])}
        print(f"Loaded {len(existing)} existing results")

    for idx in layer_indices:
        if idx >= args.num_layers:
            continue
        if idx in existing:
            print(f"  Layer {idx} already done, skipping")
            results.append(existing[idx])
            continue
        result = probe_layer(model, tokenizer, idx, baseline_ppl, device)
        results.append(result)

        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        out_data = {
            "baseline_ppl": baseline_ppl,
            "results": sorted(results, key=lambda x: x["layer"]),
        }
        with open(args.output, "w") as f:
            json.dump(out_data, f, indent=2)

    ranked = sorted(results, key=lambda x: -x["ppl_delta"])
    top_k = sorted([r["layer"] for r in ranked[: args.top_k]])

    print(f"\n{'='*60}")
    print(f"  RANKING (higher delta = more important to keep as softmax)")
    print(f"{'='*60}")
    for i, r in enumerate(ranked, 1):
        print(f"  #{i:2d}  Layer {r['layer']:2d}  "
              f"PPL_bypass={r['ppl_bypassed']:.2f}  delta=+{r['ppl_delta']:.2f}")

    print(f"\n  Top-{args.top_k} layers: {top_k}")

    out_data = {
        "baseline_ppl": baseline_ppl,
        "results": sorted(results, key=lambda x: x["layer"]),
        "ranking": [
            {"rank": i + 1, "layer": r["layer"],
             "ppl_bypassed": r["ppl_bypassed"], "ppl_delta": r["ppl_delta"]}
            for i, r in enumerate(ranked)
        ],
        "top_k": top_k,
        "config": {"model": args.model, "method": "lm_ppl", "top_k": args.top_k},
    }
    with open(args.output, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"  Saved to {args.output}")


if __name__ == "__main__":
    main()
