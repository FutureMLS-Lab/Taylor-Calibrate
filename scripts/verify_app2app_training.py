"""Verify exact token accounting and matched app-to-app training settings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MATCHED_KEYS = (
    "target_tokens",
    "batch_size",
    "micro_batch_size",
    "train_seq_len",
    "seed",
    "lr",
    "lr_attn",
    "lr_scheduler_type",
)


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-final", required=True)
    parser.add_argument("--stage2-final", required=True)
    parser.add_argument("--expected-stage-tokens", type=int, default=50_000_000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    stage1 = _load(Path(args.stage1_final) / "training_manifest.json")
    stage2 = _load(Path(args.stage2_final) / "training_manifest.json")
    expected = args.expected_stage_tokens
    for number, manifest in ((1, stage1), (2, stage2)):
        if manifest["stage"] != number:
            raise ValueError(f"Expected Stage {number} manifest")
        if manifest["completed_tokens"] != expected:
            raise ValueError(
                f"Stage {number} completed {manifest['completed_tokens']:,}, "
                f"expected {expected:,}"
            )
        if manifest["target_tokens"] != expected:
            raise ValueError(f"Stage {number} target token mismatch")
        if manifest["global_steps"] != 1250:
            raise ValueError(f"Stage {number} expected 1,250 optimizer steps")

    report = {
        "stage1_tokens": stage1["completed_tokens"],
        "stage2_tokens": stage2["completed_tokens"],
        "total_recovery_training_tokens": (
            stage1["completed_tokens"] + stage2["completed_tokens"]
        ),
        "stage1_steps": stage1["global_steps"],
        "stage2_steps": stage2["global_steps"],
        "world_size": {
            "stage1": stage1["world_size"],
            "stage2": stage2["world_size"],
        },
        "effective_batch_size": {
            "stage1": stage1["effective_batch_size"],
            "stage2": stage2["effective_batch_size"],
        },
        "sequence_length": {
            "stage1": stage1["sequence_length"],
            "stage2": stage2["sequence_length"],
        },
        "train_settings": {
            "stage1": {
                key: stage1["config"]["train"].get(key)
                for key in MATCHED_KEYS
            },
            "stage2": {
                key: stage2["config"]["train"].get(key)
                for key in MATCHED_KEYS
            },
        },
    }
    if report["world_size"] != {"stage1": 8, "stage2": 8}:
        raise ValueError(f"Both stages must run on 8 GPUs: {report['world_size']}")
    if report["total_recovery_training_tokens"] != 100_000_000:
        raise ValueError("Total recovery token count is not exactly 100M")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
