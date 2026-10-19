"""Validate and hash every file required by a sharded model checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise FileNotFoundError("Expected config.json and model.safetensors.index.json")

    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    with open(index_path, encoding="utf-8") as handle:
        index = json.load(handle)

    shard_names = sorted(set(index["weight_map"].values()))
    required = [config_path, index_path, *(model_dir / name for name in shard_names)]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing model files: {missing}")
    if config.get("num_hidden_layers") != 36 or config.get("hidden_size") != 4096:
        raise ValueError("Converted artifact is not the expected Qwen3-8B architecture")

    manifest = {
        "source": {
            "name": args.source_name,
            "revision": args.source_revision,
        },
        "model_dir": str(model_dir),
        "architecture": {
            "model_type": config.get("model_type"),
            "hidden_size": config.get("hidden_size"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "num_heads": config.get("num_heads"),
            "num_kv_heads": config.get("num_kv_heads"),
            "vocab_size": config.get("vocab_size"),
        },
        "weight_tensors": len(index["weight_map"]),
        "files": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in required
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
