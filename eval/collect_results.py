"""Collect lm-eval results and format into comparison tables."""
import argparse
import json
import os
import glob
from pathlib import Path


def find_result_files(result_dir):
    """Find all lm-eval result JSON files in a directory tree."""
    patterns = [
        os.path.join(result_dir, "**", "results_*.json"),
        os.path.join(result_dir, "**", "results.json"),
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))
    return sorted(set(files))


def load_results(result_file):
    """Load results from a single lm-eval JSON output."""
    with open(result_file) as f:
        data = json.load(f)
    results = data.get("results", {})
    out = {}
    for task_name, metrics in results.items():
        for metric_key, value in metrics.items():
            if metric_key.startswith("alias"):
                continue
            if isinstance(value, (int, float)):
                clean_key = metric_key.replace(",none", "")
                out[f"{task_name}/{clean_key}"] = value
    return out


def collect_model_results(model_dir):
    """Collect all results for a single model."""
    files = find_result_files(model_dir)
    combined = {}
    for f in files:
        combined.update(load_results(f))
    return combined


def format_table(all_results, metric_filter=None):
    """Format results as a markdown comparison table."""
    all_metrics = set()
    for results in all_results.values():
        all_metrics.update(results.keys())

    if metric_filter:
        all_metrics = {m for m in all_metrics if any(f in m for f in metric_filter)}

    key_metrics = [
        "arc_challenge/acc_norm",
        "arc_easy/acc_norm",
        "hellaswag/acc_norm",
        "piqa/acc_norm",
        "winogrande/acc",
        "boolq/acc",
        "lambada_openai/acc",
        "lambada_openai/perplexity",
        "openbookqa/acc_norm",
        "sciq/acc_norm",
        "copa/acc",
        "mmlu/acc",
    ]

    ruler_metrics = sorted(m for m in all_metrics if "ruler" in m.lower())

    display_metrics = []
    for m in key_metrics:
        if m in all_metrics:
            display_metrics.append(m)
    for m in ruler_metrics:
        display_metrics.append(m)
    for m in sorted(all_metrics - set(display_metrics)):
        display_metrics.append(m)

    if not display_metrics:
        return "No results found."

    model_names = list(all_results.keys())
    header = "| Metric | " + " | ".join(model_names) + " |"
    sep = "|" + "---|" * (len(model_names) + 1)

    rows = [header, sep]
    for metric in display_metrics:
        row_vals = []
        for model in model_names:
            val = all_results[model].get(metric, None)
            if val is None:
                row_vals.append("-")
            elif "perplexity" in metric:
                row_vals.append(f"{val:.2f}")
            elif isinstance(val, float):
                row_vals.append(f"{val*100:.1f}" if val <= 1.0 else f"{val:.2f}")
            else:
                row_vals.append(str(val))

        best_val = None
        best_idx = -1
        for i, model in enumerate(model_names):
            v = all_results[model].get(metric, None)
            if v is None:
                continue
            is_lower_better = "perplexity" in metric
            if best_val is None:
                best_val = v
                best_idx = i
            elif is_lower_better and v < best_val:
                best_val = v
                best_idx = i
            elif not is_lower_better and v > best_val:
                best_val = v
                best_idx = i

        if best_idx >= 0 and len(model_names) > 1:
            row_vals[best_idx] = f"**{row_vals[best_idx]}**"

        row = f"| {metric} | " + " | ".join(row_vals) + " |"
        rows.append(row)

    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description="Collect and compare lm-eval results.")
    parser.add_argument("result_dirs", nargs="+",
                        help="Directories containing lm-eval results. Format: 'name:path' or just 'path'.")
    parser.add_argument("--output", "-o", default=None,
                        help="Save comparison table to file (markdown).")
    parser.add_argument("--filter", nargs="*", default=None,
                        help="Filter metrics containing any of these strings.")
    args = parser.parse_args()

    all_results = {}
    for entry in args.result_dirs:
        if ":" in entry and not entry.startswith("/"):
            name, path = entry.split(":", 1)
        else:
            path = entry
            name = os.path.basename(path.rstrip("/"))
        results = collect_model_results(path)
        if results:
            all_results[name] = results
            print(f"Loaded {len(results)} metrics for '{name}' from {path}")
        else:
            print(f"WARNING: No results found in {path}")

    if not all_results:
        print("No results to compare.")
        return

    table = format_table(all_results, metric_filter=args.filter)
    print("\n" + table)

    if args.output:
        with open(args.output, "w") as f:
            f.write(table + "\n")
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
