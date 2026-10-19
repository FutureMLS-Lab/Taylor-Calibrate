# Qwen3-8B SWA(64,4) vs Taylor-Calibrate GDN

Status: validated. These results remain outside the manuscript pending review.

## Fixed protocol

- Same pinned Qwen3-8B teacher/tokenizer, cleaned-Alpaca stream, seed, BF16 precision, optimizer settings, and all-36-layer replacement.
- 100M recovery budget is exactly 50M layer-output alignment + 50M full-model KL for both methods.
- Each stage uses 1,000-token blocks, effective batch 40, and exactly 1,250 optimizer steps.
- Scores use MMLU 5-shot and zero-shot ARC-C, ARC-E, HellaSwag, PIQA, and WinoGrande. The macro is the unweighted mean of those six tasks.
- RULER-4K is the unweighted mean of all 13 official subtasks, 500 examples each.

## Results

| Budget | Method | MMLU | ARC-C | ARC-E | HellaSwag | PIQA | WinoGrande | 6-task avg | WikiText2 PPL ↓ | RULER-4K |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0B | SWA(64,4) | 70.29 | 56.40 | 80.77 | 74.11 | 77.80 | 67.72 | 71.18 | 16.41 | 7.22 |
| 0B | Taylor-GDN | 23.57 | 25.34 | 27.95 | 26.12 | 52.50 | 50.67 | 34.36 | 5571.40 | 0.08 |
| 100M | SWA(64,4) | 60.56 | 55.38 | 80.47 | 71.67 | 77.58 | 68.51 | 69.03 | 16.53 | 4.52 |
| 100M | Taylor-GDN | 47.88 | 52.82 | 77.74 | 70.19 | 77.42 | 61.96 | 64.67 | 27.21 | 2.12 |

## Outcome

- At 0B, SWA wins the six-task macro by 36.82 points.
- At 100M, SWA wins the six-task macro by 4.36 points.
- At 100M, SWA wins RULER-4K by 2.40 points; SWA has lower WikiText2 perplexity.
- Across the six 100M short tasks, Taylor wins 0 and SWA wins 6.
- Taylor improves its six-task macro by 30.31 points, closing most of the initial gap, but does not overtake SWA. SWA changes by -2.15 points after post-training.

## Accounting

| Method | Total params | Stage-1 trainable | Stage-2 trainable | Recovery tokens | Calibration |
|---|---:|---:|---:|---:|---|
| SWA(64,4) | 8,190,735,360 | 1,509,958,656 | 8,190,735,360 | 100,000,000 | none |
| Taylor-GDN | 9,710,119,680 | 3,029,342,976 | 9,710,119,680 | 100,000,000 | 1,024 unique tokens, 10,800 local steps, 0.049 H100-h |

## Reproducibility and caveats

- Teacher/tokenizer revision: `b968826d9c46dd6066d109eabc6255188de91218`.
- cleaned-Alpaca revision: `12567cabf869d7c92e573c7c783905fc160e9639`; exact 100M stream SHA-256: `6bec134dc8392fb40b0966f7b457e89026241669ee6a91ea25a4a05f6342c3af`.
- The formatted base corpus has 9,710,407 tokens and is repeated 10 times plus a deterministic remainder.
- This is quality at a fixed training budget. The SWA generation path intentionally retains full KV history for correctness, so these runs support no serving-speed or memory claim.
- Taylor has more parameters; the difference is reported rather than hidden.
- No manuscript files were changed.

## Invalidated or failed attempts

- The first SWA 0B checkpoint omitted learned Qwen3 q_norm/k_norm weights. Its PPL 697.84 and associated short-task results were invalidated; a regression test and exact weight check were added before rebuilding.
- Initial RULER setup failed because lm-eval 0.4.11 referenced an unavailable CMU HTTP URL. It was replaced with upstream's immutable, byte-compatible Hugging Face mirror.
- Serial full-group RULER was stopped after 5% when measured runtime exceeded its deadline. The accepted run evaluates all 13 official tasks with all 500 examples via indexed shards.
- Two 8-GPU training pods were resubmitted with lower scheduler memory reservations while still pending; no optimizer step ran in those scheduling-only attempts.

All four accepted checkpoints and all 52 RULER task-stage-method shards passed completeness checks. Code, data, artifact, and result hashes are recorded in `app2app_comparison.json`.
