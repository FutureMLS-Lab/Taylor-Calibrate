"""Hash the source/config files that define an app-to-app run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


INCLUDE_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".sh"}
INCLUDE_ROOTS = ("src", "scripts", "convert", "eval", "configs", "tests")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    files = []
    for root_name in INCLUDE_ROOTS:
        root = repo / root_name
        if not root.exists():
            continue
        files.extend(
            path for path in root.rglob("*")
            if path.is_file() and path.suffix in INCLUDE_SUFFIXES
        )
    manifest = {
        "repo": str(repo),
        "files": {
            str(path.relative_to(repo)): {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(files)
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Hashed {len(files)} source/config files")


if __name__ == "__main__":
    main()
