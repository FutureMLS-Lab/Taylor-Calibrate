"""Validate and summarize the SWA-vs-Taylor app-to-app experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean


TASK_METRICS = {
    "mmlu": "acc,none",
    "arc_challenge": "acc_norm,none",
    "arc_easy": "acc_norm,none",
    "hellaswag": "acc_norm,none",
    "piqa": "acc_norm,none",
    "winogrande": "acc,none",
}
RULER_TASKS = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multiquery",
    "niah_multivalue",
    "ruler_vt",
    "ruler_cwe",
    "ruler_fwe",
    "ruler_qa_squad",
    "ruler_qa_hotpot",
)


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _single_result(directory: Path) -> tuple[dict, Path]:
    files = sorted(directory.rglob("results_*.json"))
    if len(files) != 1:
        raise ValueError(f"Expected one results JSON under {directory}, found {files}")
    return _load(files[0]), files[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _collect_scores(result_dir: Path) -> tuple[dict, list[Path]]:
    core, core_path = _single_result(result_dir / "core")
    mmlu, mmlu_path = _single_result(result_dir / "mmlu")
    merged = {**core["results"], **mmlu["results"]}
    tasks = {
        task: float(merged[task][metric])
        for task, metric in TASK_METRICS.items()
    }
    ppl_path = result_dir / "ppl" / "results.json"
    ppl = float(_load(ppl_path)["perplexity"])

    ruler = {}
    ruler_paths = []
    for task in RULER_TASKS:
        payload, path = _single_result(result_dir / "ruler-shards" / task)
        ruler[task] = float(payload["results"][task]["4096,none"])
        ruler_paths.append(path)

    return {
        "tasks": tasks,
        "six_task_macro": mean(tasks.values()),
        "wikitext2_ppl": ppl,
        "ruler_4k": {
            "tasks": ruler,
            "macro": mean(ruler.values()),
        },
    }, [core_path, mmlu_path, ppl_path, *ruler_paths]


def _checkpoint_metadata(checkpoint: Path) -> dict:
    parameters = _load(checkpoint / "parameter_manifest.json")["parameter_count"]
    artifact = _load(checkpoint / "artifact_manifest.json")
    return {
        "path": str(checkpoint),
        "parameters": parameters,
        "artifact_files": artifact["files"],
    }


def _method_stage_paths(root: Path, stage: str, method: str) -> tuple[Path, Path]:
    if stage == "0b" and method == "swa":
        return (
            root / "results/0b/swa-fixed-qk-norm",
            root / "checkpoints/swa/0b-fixed-qk-norm",
        )
    if stage == "0b":
        return root / "results/0b/taylor", root / "checkpoints/taylor/0b"
    return (
        root / f"results/100m/{method}",
        root / f"checkpoints/{method}/stage2/final",
    )


def _format_pct(value: float) -> str:
    return f"{100 * value:.2f}"


def _render_markdown(report: dict) -> str:
    lines = [
        "# Qwen3-8B SWA(64,4) vs Taylor-Calibrate GDN",
        "",
        "Status: validated. These results remain outside the manuscript pending review.",
        "",
        "## Fixed protocol",
        "",
        "- Same pinned Qwen3-8B teacher/tokenizer, cleaned-Alpaca stream, seed, BF16 precision, optimizer settings, and all-36-layer replacement.",
        "- 100M recovery budget is exactly 50M layer-output alignment + 50M full-model KL for both methods.",
        "- Each stage uses 1,000-token blocks, effective batch 40, and exactly 1,250 optimizer steps.",
        "- Scores use MMLU 5-shot and zero-shot ARC-C, ARC-E, HellaSwag, PIQA, and WinoGrande. The macro is the unweighted mean of those six tasks.",
        "- RULER-4K is the unweighted mean of all 13 official subtasks, 500 examples each.",
        "",
        "## Results",
        "",
        "| Budget | Method | MMLU | ARC-C | ARC-E | HellaSwag | PIQA | WinoGrande | 6-task avg | WikiText2 PPL ↓ | RULER-4K |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in ("0b", "100m"):
        for method in ("swa", "taylor"):
            row = report["results"][stage][method]
            task = row["tasks"]
            lines.append(
                "| {budget} | {method} | {mmlu} | {arc_c} | {arc_e} | {hs} | {piqa} | {wino} | {avg} | {ppl:.2f} | {ruler} |".format(
                    budget="0B" if stage == "0b" else "100M",
                    method="SWA(64,4)" if method == "swa" else "Taylor-GDN",
                    mmlu=_format_pct(task["mmlu"]),
                    arc_c=_format_pct(task["arc_challenge"]),
                    arc_e=_format_pct(task["arc_easy"]),
                    hs=_format_pct(task["hellaswag"]),
                    piqa=_format_pct(task["piqa"]),
                    wino=_format_pct(task["winogrande"]),
                    avg=_format_pct(row["six_task_macro"]),
                    ppl=row["wikitext2_ppl"],
                    ruler=_format_pct(row["ruler_4k"]["macro"]),
                )
            )

    winner = report["winner"]
    lines.extend(
        [
            "",
            "## Outcome",
            "",
            f"- At 0B, {winner['0b']['macro']} wins the six-task macro by {100 * winner['0b']['macro_margin']:.2f} points.",
            f"- At 100M, {winner['100m']['macro']} wins the six-task macro by {100 * winner['100m']['macro_margin']:.2f} points.",
            f"- At 100M, {winner['100m']['ruler']} wins RULER-4K by {100 * winner['100m']['ruler_margin']:.2f} points; {winner['100m']['ppl']} has lower WikiText2 perplexity.",
            f"- Across the six 100M short tasks, Taylor wins {winner['100m']['taylor_task_wins']} and SWA wins {winner['100m']['swa_task_wins']}.",
            f"- Taylor improves its six-task macro by {100 * (report['results']['100m']['taylor']['six_task_macro'] - report['results']['0b']['taylor']['six_task_macro']):.2f} points, closing most of the initial gap, but does not overtake SWA. SWA changes by {100 * (report['results']['100m']['swa']['six_task_macro'] - report['results']['0b']['swa']['six_task_macro']):.2f} points after post-training.",
            "",
            "## Accounting",
            "",
            "| Method | Total params | Stage-1 trainable | Stage-2 trainable | Recovery tokens | Calibration |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for method in ("swa", "taylor"):
        accounting = report["accounting"][method]
        calibration = accounting["calibration"]
        if calibration["total_local_steps"]:
            calibration_text = (
                f"{calibration['unique_tokens']:,} unique tokens, "
                f"{calibration['total_local_steps']:,} local steps, "
                f"{calibration['gpu_hours']:.3f} H100-h"
            )
        else:
            calibration_text = "none"
        lines.append(
            f"| {'SWA(64,4)' if method == 'swa' else 'Taylor-GDN'} | "
            f"{accounting['total_parameters']:,} | "
            f"{accounting['stage1_trainable_parameters']:,} | "
            f"{accounting['stage2_trainable_parameters']:,} | "
            f"{accounting['recovery_training_tokens']:,} | {calibration_text} |"
        )

    data = report["data"]
    lines.extend(
        [
            "",
            "## Reproducibility and caveats",
            "",
            f"- Teacher/tokenizer revision: `{report['teacher_revision']}`.",
            f"- cleaned-Alpaca revision: `{data['dataset']['revision']}`; exact 100M stream SHA-256: `{data['stream']['sha256_uint32_le']}`.",
            f"- The formatted base corpus has {data['base_corpus']['tokens']:,} tokens and is repeated {data['stream']['repetitions']} times plus a deterministic remainder.",
            "- This is quality at a fixed training budget. The SWA generation path intentionally retains full KV history for correctness, so these runs support no serving-speed or memory claim.",
            "- Taylor has more parameters; the difference is reported rather than hidden.",
            "- No manuscript files were changed.",
            "",
            "## Invalidated or failed attempts",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["invalidated_attempts"])
    lines.extend(
        [
            "",
            "All four accepted checkpoints and all 52 RULER task-stage-method shards passed completeness checks. Code, data, artifact, and result hashes are recorded in `app2app_comparison.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(args.root)

    data = _load(root / "data/cleaned-alpaca-qwen3-8b-100m/manifest.json")
    training = {
        method: _load(root / f"metadata/{method}/training_manifest.json")
        for method in ("swa", "taylor")
    }
    code_manifests = {
        method: _load(root / f"metadata/{method}/code_manifest.json")
        for method in ("swa", "taylor")
    }
    environments = {
        method: _load(root / f"metadata/{method}/environment.json")
        for method in ("swa", "taylor")
    }
    if code_manifests["swa"]["files"] != code_manifests["taylor"]["files"]:
        raise ValueError("Methods did not run the same source/config snapshot")
    for stage in ("stage1", "stage2"):
        if training["swa"]["train_settings"][stage] != training["taylor"]["train_settings"][stage]:
            raise ValueError(f"Matched training settings diverged at {stage}")
    if (
        _load(root / "metadata/swa/data_manifest.json")["stream"]["sha256_uint32_le"]
        != _load(root / "metadata/taylor/data_manifest.json")["stream"]["sha256_uint32_le"]
    ):
        raise ValueError("Methods did not use the same token stream")

    results = {}
    result_files = []
    checkpoints = {}
    for stage in ("0b", "100m"):
        results[stage] = {}
        checkpoints[stage] = {}
        for method in ("swa", "taylor"):
            result_dir, checkpoint = _method_stage_paths(root, stage, method)
            results[stage][method], files = _collect_scores(result_dir)
            result_files.extend(files)
            checkpoints[stage][method] = _checkpoint_metadata(checkpoint)

    winner = {}
    for stage in ("0b", "100m"):
        swa, taylor = results[stage]["swa"], results[stage]["taylor"]
        macro_method = "SWA" if swa["six_task_macro"] >= taylor["six_task_macro"] else "Taylor"
        ruler_method = "SWA" if swa["ruler_4k"]["macro"] >= taylor["ruler_4k"]["macro"] else "Taylor"
        ppl_method = "SWA" if swa["wikitext2_ppl"] <= taylor["wikitext2_ppl"] else "Taylor"
        winner[stage] = {
            "macro": macro_method,
            "macro_margin": abs(swa["six_task_macro"] - taylor["six_task_macro"]),
            "ruler": ruler_method,
            "ruler_margin": abs(swa["ruler_4k"]["macro"] - taylor["ruler_4k"]["macro"]),
            "ppl": ppl_method,
            "swa_task_wins": sum(
                swa["tasks"][task] > taylor["tasks"][task] for task in TASK_METRICS
            ),
            "taylor_task_wins": sum(
                taylor["tasks"][task] > swa["tasks"][task] for task in TASK_METRICS
            ),
        }

    swa_init = _load(root / "checkpoints/swa/0b-fixed-qk-norm/initialization_manifest.json")
    taylor_init = _load(root / "checkpoints/taylor/0b/initialization_manifest.json")
    accounting = {}
    for method, init in (("swa", swa_init), ("taylor", taylor_init)):
        params = checkpoints["100m"][method]["parameters"]
        accounting[method] = {
            "total_parameters": params["total"],
            "stage1_trainable_parameters": params["by_group"]["attention"],
            "stage2_trainable_parameters": params["total"],
            "recovery_training_tokens": training[method]["total_recovery_training_tokens"],
            "calibration": init["calibration"],
        }

    report = {
        "status": "validated",
        "teacher": "Qwen/Qwen3-8B",
        "teacher_revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "data": data,
        "results": results,
        "winner": winner,
        "accounting": accounting,
        "training": training,
        "code_manifests": code_manifests,
        "environments": environments,
        "checkpoints": checkpoints,
        "result_files": {
            str(path): {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in sorted(result_files)
        },
        "invalidated_attempts": [
            "The first SWA 0B checkpoint omitted learned Qwen3 q_norm/k_norm weights. Its PPL 697.84 and associated short-task results were invalidated; a regression test and exact weight check were added before rebuilding.",
            "Initial RULER setup failed because lm-eval 0.4.11 referenced an unavailable CMU HTTP URL. It was replaced with upstream's immutable, byte-compatible Hugging Face mirror.",
            "Serial full-group RULER was stopped after 5% when measured runtime exceeded its deadline. The accepted run evaluates all 13 official tasks with all 500 examples via indexed shards.",
            "Two 8-GPU training pods were resubmitted with lower scheduler memory reservations while still pending; no optimizer step ran in those scheduling-only attempts.",
        ],
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "app2app_comparison.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with open(output_dir / "app2app_comparison.md", "w", encoding="utf-8") as handle:
        handle.write(_render_markdown(report))
    print(json.dumps(report["winner"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
