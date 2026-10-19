"""Audit the structurally matched 3:1 SWA-vs-Taylor comparison."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from collect_app2app_results import TASK_METRICS, _checkpoint_metadata, _collect_scores, _load, _sha256
from collect_hybrid_app2app_results import CORE_RUNTIME_FILES, _macro_independent_se


def _pct(value: float) -> str:
    return f"{100 * value:.2f}"


def _render(report: dict) -> str:
    swa = report["results"]["swa_hybrid_ar_100m"]
    taylor = report["results"]["taylor_hybrid_ar_100m"]
    lines = [
        "# Structurally Matched Qwen3-8B 3:1 Comparison",
        "",
        "Status: validated.",
        "",
        "Both students retain the same nine AR-selected full-attention layers. The remaining 27 layers use SWA(64,4) or Taylor-Calibrate GDN. Teacher, data, seed, optimizer, effective batch, and the exact 50M Stage-1 + 50M Stage-2 budget are shared.",
        "",
        "| Model | MMLU | ARC-C | ARC-E | HellaSwag | PIQA | WinoGrande | 6-task avg | WikiText2 PPL ↓ | RULER-4K |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in (
        ("3:1 SWA(64,4)", swa),
        ("3:1 Taylor-GDN", taylor),
    ):
        task = row["tasks"]
        lines.append(
            f"| {label} | {_pct(task['mmlu'])} | {_pct(task['arc_challenge'])} | "
            f"{_pct(task['arc_easy'])} | {_pct(task['hellaswag'])} | "
            f"{_pct(task['piqa'])} | {_pct(task['winogrande'])} | "
            f"{_pct(row['six_task_macro'])} | {row['wikitext2_ppl']:.2f} | "
            f"{_pct(row['ruler_4k']['macro'])} |"
        )
    outcome = report["outcome"]
    lines.extend(
        [
            "",
            "## Outcome",
            "",
            f"- SWA numerically leads the six-task average by {outcome['swa_short_lead_points']:.2f} points and wins four of six tasks.",
            f"- Taylor wins MMLU and HellaSwag, lowers PPL by {outcome['taylor_ppl_reduction_percent']:.1f}%, and improves RULER-4K by {outcome['taylor_ruler_lead_points']:.2f} points ({outcome['taylor_ruler_ratio']:.2f}×).",
            f"- The {outcome['swa_short_lead_points']:.2f}-point short-task gap is smaller than the independent difference SE estimate ({100 * report['uncertainty']['difference_independent_se']:.2f} points), so short-context accuracy should be treated as a tie without paired bootstrap or more seeds.",
            "",
            "## Accounting",
            "",
            "| Model | Total parameters | Stage-1 trainable | Stage-2 trainable | Recovery tokens |",
            "|---|---:|---:|---:|---:|",
            f"| 3:1 SWA(64,4) | {report['accounting']['swa']['total_parameters']:,} | "
            f"{report['accounting']['swa']['stage1_trainable_parameters']:,} | "
            f"{report['accounting']['swa']['stage2_trainable_parameters']:,} | 100,000,000 |",
            f"| 3:1 Taylor-GDN | {report['accounting']['taylor']['total_parameters']:,} | "
            f"{report['accounting']['taylor']['stage1_trainable_parameters']:,} | "
            f"{report['accounting']['taylor']['stage2_trainable_parameters']:,} | 100,000,000 |",
            "",
            "Both models retain the same growing KV cache in nine full-attention layers. The converted layers differ: SWA stores a 64-token cache, whereas GDN maintains recurrent state. End-to-end memory and throughput were not measured here.",
            "",
            "The accepted SWA evaluation was requeued before execution with a longer deadline after six hours of GPU scarcity; no incomplete evaluation result was reused.",
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

    prior = _load(root / "report/app2app_hybrid_comparison.json")
    swa_dir = root / "results/100m/swa-hybrid-ar"
    swa, result_files = _collect_scores(swa_dir)
    taylor = prior["results"]["taylor_hybrid_ar_100m"]

    swa_checkpoint = root / "checkpoints/swa-hybrid-ar/stage2/final"
    swa_checkpoint_meta = _checkpoint_metadata(swa_checkpoint)
    taylor_checkpoint_meta = prior["checkpoint"]
    swa_training = _load(root / "metadata/swa-hybrid-ar/training_manifest.json")
    taylor_training = prior["training"]
    swa_selection = _load(root / "metadata/swa-hybrid-ar/swa_hybrid_ar_selection.json")
    taylor_selection = prior["selection"]

    if swa_selection["retained_full_attention_layers"] != taylor_selection["retained_full_attention_layers"]:
        raise ValueError("SWA and Taylor retained different full-attention layers")
    if swa_selection["converted_layers"] != taylor_selection["converted_layers"]:
        raise ValueError("SWA and Taylor converted different layer sets")
    if swa_training["total_recovery_training_tokens"] != 100_000_000:
        raise ValueError("SWA hybrid did not complete exactly 100M tokens")
    if swa_training["world_size"] != {"stage1": 8, "stage2": 8}:
        raise ValueError("SWA hybrid did not train at world size 8")
    if swa_training["train_settings"] != taylor_training["train_settings"]:
        raise ValueError("SWA and Taylor hybrid training settings differ")
    if (
        _load(root / "metadata/swa-hybrid-ar/data_manifest.json")["stream"]["sha256_uint32_le"]
        != _load(root / "metadata/taylor-hybrid-ar/data_manifest.json")["stream"]["sha256_uint32_le"]
    ):
        raise ValueError("SWA and Taylor hybrid token streams differ")

    swa_code = _load(root / "metadata/swa-hybrid-ar/code_manifest.json")
    taylor_code = prior["code_manifest"]
    for path in CORE_RUNTIME_FILES:
        if swa_code["files"][path]["sha256"] != taylor_code["files"][path]["sha256"]:
            raise ValueError(f"Runtime source differs between hybrid runs: {path}")

    swa_macro_se = _macro_independent_se(swa_dir)
    taylor_macro_se = prior["uncertainty"]["hybrid_macro_independent_se"]
    task_wins = {
        "swa": [task for task in TASK_METRICS if swa["tasks"][task] > taylor["tasks"][task]],
        "taylor": [task for task in TASK_METRICS if taylor["tasks"][task] > swa["tasks"][task]],
    }
    outcome = {
        "swa_short_lead_points": 100 * (swa["six_task_macro"] - taylor["six_task_macro"]),
        "taylor_ppl_reduction_percent": 100 * (
            1 - taylor["wikitext2_ppl"] / swa["wikitext2_ppl"]
        ),
        "taylor_ruler_lead_points": 100 * (
            taylor["ruler_4k"]["macro"] - swa["ruler_4k"]["macro"]
        ),
        "taylor_ruler_ratio": taylor["ruler_4k"]["macro"] / swa["ruler_4k"]["macro"],
        "task_wins": task_wins,
    }

    report = {
        "status": "validated",
        "teacher": prior["teacher"],
        "teacher_revision": prior["teacher_revision"],
        "selection": {
            "retained_full_attention_layers": swa_selection["retained_full_attention_layers"],
            "converted_layers": swa_selection["converted_layers"],
        },
        "results": {
            "swa_hybrid_ar_100m": swa,
            "taylor_hybrid_ar_100m": taylor,
        },
        "outcome": outcome,
        "uncertainty": {
            "swa_macro_independent_se": swa_macro_se,
            "taylor_macro_independent_se": taylor_macro_se,
            "difference_independent_se": math.sqrt(
                swa_macro_se * swa_macro_se + taylor_macro_se * taylor_macro_se
            ),
            "note": "Independence approximation from lm-eval per-task standard errors.",
        },
        "accounting": {
            "swa": {
                "total_parameters": swa_checkpoint_meta["parameters"]["total"],
                "stage1_trainable_parameters": 1_132_468_992,
                "stage2_trainable_parameters": swa_checkpoint_meta["parameters"]["total"],
            },
            "taylor": {
                "total_parameters": taylor_checkpoint_meta["parameters"]["total"],
                "stage1_trainable_parameters": prior["accounting"]["hybrid"]["stage1_trainable_parameters"],
                "stage2_trainable_parameters": taylor_checkpoint_meta["parameters"]["total"],
            },
        },
        "training": {
            "swa": swa_training,
            "taylor": taylor_training,
        },
        "checkpoints": {
            "swa": swa_checkpoint_meta,
            "taylor": taylor_checkpoint_meta,
        },
        "result_files": {
            str(path): {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in sorted(result_files)
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "app2app_matched_hybrid_comparison.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with open(output_dir / "app2app_matched_hybrid_comparison.md", "w", encoding="utf-8") as handle:
        handle.write(_render(report))
    print(json.dumps(outcome, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
