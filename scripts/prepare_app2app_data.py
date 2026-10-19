"""Build the immutable 100M-token cleaned-Alpaca app-to-app stream.

The source corpus is formatted with the canonical Stanford Alpaca templates,
tokenized once in dataset order, and repeated cyclically to exactly 100M
tokens.  Stage 1 and Stage 2 receive disjoint 50M-token slices represented as
50,000 fixed 1,000-token examples each.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer


DATASET_NAME = "yahma/alpaca-cleaned"
DATASET_REVISION = "12567cabf869d7c92e573c7c783905fc160e9639"
TOKENIZER_NAME = "Qwen/Qwen3-8B"
TOKENIZER_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
FORMAT_VERSION = "stanford-alpaca-v1+eos"

PROMPT_WITH_INPUT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

PROMPT_NO_INPUT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""


def _sha256_array(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    view = memoryview(np.ascontiguousarray(array)).cast("B")
    for start in range(0, len(view), 16 * 1024 * 1024):
        digest.update(view[start:start + 16 * 1024 * 1024])
    return digest.hexdigest()


def _save_fixed_dataset(tokens: np.ndarray, seq_len: int, output_dir: Path) -> None:
    if len(tokens) % seq_len:
        raise ValueError(f"{len(tokens):,} tokens is not divisible by seq_len={seq_len}")
    chunks = tokens.reshape(-1, seq_len)
    flat = pa.array(chunks.reshape(-1), type=pa.uint32())
    lists = pa.FixedSizeListArray.from_arrays(flat, seq_len)
    Dataset(pa.Table.from_arrays([lists], names=["input_ids"])).save_to_disk(str(output_dir))


def _format_example(example: dict[str, str]) -> str:
    values = {
        "instruction": (example.get("instruction") or "").strip(),
        "input": (example.get("input") or "").strip(),
        "output": (example.get("output") or "").strip(),
    }
    template = PROMPT_WITH_INPUT if values["input"] else PROMPT_NO_INPUT
    return template.format(**values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--total-tokens", type=int, default=100_000_000)
    parser.add_argument("--stage-tokens", type=int, default=50_000_000)
    parser.add_argument("--seq-len", type=int, default=1_000)
    args = parser.parse_args()

    if args.total_tokens != 2 * args.stage_tokens:
        raise ValueError("total_tokens must equal exactly two stage_tokens slices")
    if args.stage_tokens % args.seq_len:
        raise ValueError("stage_tokens must be divisible by seq_len")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        revision=DATASET_REVISION,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME,
        revision=TOKENIZER_REVISION,
        use_fast=True,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id")

    documents: list[np.ndarray] = []
    for example in dataset:
        ids = tokenizer.encode(_format_example(example), add_special_tokens=False)
        ids.append(tokenizer.eos_token_id)
        documents.append(np.asarray(ids, dtype=np.uint32))
    base_tokens = np.concatenate(documents)

    repeats, remainder = divmod(args.total_tokens, len(base_tokens))
    stream = np.empty(args.total_tokens, dtype=np.uint32)
    cursor = 0
    for _ in range(repeats):
        stream[cursor:cursor + len(base_tokens)] = base_tokens
        cursor += len(base_tokens)
    stream[cursor:] = base_tokens[:remainder]

    np.save(output_dir / "all_tokens.npy", stream)
    stage1 = stream[:args.stage_tokens]
    stage2 = stream[args.stage_tokens:]
    _save_fixed_dataset(stage1, args.seq_len, output_dir / f"stage1_context{args.seq_len}")
    _save_fixed_dataset(stage2, args.seq_len, output_dir / f"stage2_context{args.seq_len}")

    manifest = {
        "dataset": {
            "name": DATASET_NAME,
            "revision": DATASET_REVISION,
            "split": "train",
            "fingerprint": dataset._fingerprint,
            "examples": len(dataset),
        },
        "format_version": FORMAT_VERSION,
        "tokenizer": {
            "name": TOKENIZER_NAME,
            "revision": TOKENIZER_REVISION,
            "eos_token_id": tokenizer.eos_token_id,
        },
        "base_corpus": {
            "tokens": int(len(base_tokens)),
            "sha256_uint32_le": _sha256_array(base_tokens),
        },
        "stream": {
            "tokens": int(len(stream)),
            "repetitions": repeats,
            "remainder_tokens": remainder,
            "sha256_uint32_le": _sha256_array(stream),
        },
        "stages": {
            "stage1": {
                "start": 0,
                "stop": args.stage_tokens,
                "tokens": args.stage_tokens,
                "sequence_length": args.seq_len,
                "examples": args.stage_tokens // args.seq_len,
                "sha256_uint32_le": _sha256_array(stage1),
            },
            "stage2": {
                "start": args.stage_tokens,
                "stop": args.total_tokens,
                "tokens": args.stage_tokens,
                "sequence_length": args.seq_len,
                "examples": args.stage_tokens // args.seq_len,
                "sha256_uint32_le": _sha256_array(stage2),
            },
        },
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
