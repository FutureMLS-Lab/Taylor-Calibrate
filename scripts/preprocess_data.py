"""
Download, tokenize, and chunk a HuggingFace dataset for distillation training.

Usage:
    python preprocess_data.py \
        --dataset_name HuggingFaceFW/fineweb-edu \
        --dataset_config sample-10BT \
        --tokenizer Qwen/Qwen2.5-1.5B-Instruct \
        --context_length 4096 \
        --output_dir ./data \
        --max_tokens 2000000000
"""
import argparse
import os
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
import pyarrow as pa
from datasets import Dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset_config", type=str, default="sample-10BT")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--text_field", type=str, default="text")
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--context_length", type=int, default=4096)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_tokens", type=int, default=2_000_000_000,
                        help="Stop after collecting this many tokens (saves time/disk)")
    parser.add_argument("--num_proc", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--streaming", action="store_true",
                        help="Use streaming to avoid downloading entire dataset")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    npy_path = os.path.join(args.output_dir, "all_tokens.npy")

    if os.path.exists(npy_path):
        print(f"Loading cached tokens from {npy_path}")
        all_tokens = np.load(npy_path, mmap_mode='r')
    elif args.streaming:
        print(f"Streaming {args.dataset_name} ({args.dataset_config})...")
        ds = load_dataset(args.dataset_name, args.dataset_config,
                          split=args.split, streaming=True)

        token_chunks = []
        total = 0
        for i, example in enumerate(ds):
            ids = tok.encode(example[args.text_field], add_special_tokens=False)
            token_chunks.append(ids)
            total += len(ids)
            if i % 10000 == 0:
                print(f"  {i:,} docs, {total:,} tokens")
            if total >= args.max_tokens:
                print(f"Reached {args.max_tokens:,} token limit")
                break

        print(f"Concatenating {total:,} tokens...")
        from itertools import chain
        all_tokens = np.array(list(chain.from_iterable(token_chunks)), dtype=np.uint32)
        print(f"Saving token cache to {npy_path}")
        np.save(npy_path, all_tokens)
    else:
        print(f"Loading {args.dataset_name} ({args.dataset_config})...")
        ds_kwargs = {}
        if args.dataset_config:
            ds_kwargs["name"] = args.dataset_config
        ds = load_dataset(args.dataset_name, **ds_kwargs, split=args.split)

        print(f"Tokenizing {len(ds):,} examples...")
        def tokenize_batch(batch):
            return {"input_ids": tok(batch[args.text_field], add_special_tokens=False)["input_ids"]}

        tokenized = ds.map(
            tokenize_batch, batched=True, batch_size=args.batch_size,
            num_proc=args.num_proc, remove_columns=ds.column_names,
            desc="Tokenizing",
        )

        print("Concatenating all tokens...")
        chunked_arr = tokenized.data.column("input_ids")
        flat_chunks = [chunk.flatten() for chunk in chunked_arr.chunks]
        flat_array = pa.concat_arrays(flat_chunks)
        all_tokens = flat_array.to_numpy(zero_copy_only=False).astype(np.uint32, copy=False)

        print(f"Saving token cache ({len(all_tokens):,} tokens) to {npy_path}")
        np.save(npy_path, all_tokens)

    ctx = args.context_length
    total_len = (len(all_tokens) // ctx) * ctx
    all_tokens = all_tokens[:total_len]
    chunks = all_tokens.reshape(-1, ctx)

    print(f"Total tokens: {total_len:,}, Chunks: {len(chunks):,} x {ctx}")

    arrow_type = pa.uint32()
    flat_arr = pa.array(chunks.flatten(), type=arrow_type)
    arrow_arr = pa.FixedSizeListArray.from_arrays(flat_arr, ctx)
    table = pa.Table.from_arrays([arrow_arr], names=["input_ids"])
    chunked_ds = Dataset(table)

    out_path = os.path.join(args.output_dir, f"chunked_context{ctx}")
    print(f"Saving chunked dataset to {out_path}...")
    chunked_ds.save_to_disk(out_path)

    print(f"Done! {len(chunks):,} chunks of length {ctx}")
    print(f"Total tokens: {total_len:,} ({total_len/1e9:.2f}B)")


if __name__ == "__main__":
    main()
