"""Create a parameter-count manifest without materializing checkpoint tensors."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from safetensors import safe_open


def _group(name: str) -> str:
    if ".attn." in name:
        return "attention"
    if ".mlp." in name:
        return "mlp"
    if "embedding" in name or "lm_head" in name:
        return "embedding_and_lm_head"
    return "norm_and_other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    with open(checkpoint / "config.json", encoding="utf-8") as handle:
        config = json.load(handle)
    with open(checkpoint / "model.safetensors.index.json", encoding="utf-8") as handle:
        index = json.load(handle)

    shard_to_keys: dict[str, list[str]] = defaultdict(list)
    for key, shard in index["weight_map"].items():
        shard_to_keys[shard].append(key)

    groups: dict[str, int] = defaultdict(int)
    tensor_count = 0
    total = 0
    for shard, keys in sorted(shard_to_keys.items()):
        path = checkpoint / shard
        if not path.is_file():
            raise FileNotFoundError(path)
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in keys:
                shape = handle.get_slice(key).get_shape()
                numel = math.prod(shape)
                total += numel
                groups[_group(key)] += numel
                tensor_count += 1

    report = {
        "checkpoint": str(checkpoint),
        "method": config.get("student_name"),
        "student_config": {
            "hidden_size": config.get("hidden_size"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "num_heads": config.get("num_heads"),
            "num_kv_heads": config.get("num_kv_heads"),
            "keep_full_attention_layers": config.get("keep_full_attention_layers"),
            "window_size": config.get("window_size"),
            "num_sink_tokens": config.get("num_sink_tokens"),
        },
        "parameter_count": {
            "total": total,
            "by_group": dict(sorted(groups.items())),
            "checkpoint_tensors": tensor_count,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
