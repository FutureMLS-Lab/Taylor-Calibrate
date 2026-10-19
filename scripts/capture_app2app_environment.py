"""Record software and hardware versions used for an app-to-app run."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import accelerate
import deepspeed
import flash_attn
import fla
import torch
import transformers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = {
        "method": args.method,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
        "deepspeed": deepspeed.__version__,
        "flash_attn": flash_attn.__version__,
        "fla": getattr(fla, "__version__", "unknown"),
        "gpu_count": torch.cuda.device_count(),
        "gpus": [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
