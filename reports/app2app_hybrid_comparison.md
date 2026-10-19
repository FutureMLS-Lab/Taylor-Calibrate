# Qwen3-8B 3:1 Taylor Hybrid vs SWA

Status: validated. This report remains outside the manuscript pending review.

## Protocol

- Taylor hybrid retains the nine highest LM-PPL-importance softmax layers and converts the other 27 layers to GDN.
- The trained comparison uses the exact same pinned teacher, 100M-token cleaned-Alpaca stream, seed, BF16 precision, optimizer settings, and 50M Stage-1 + 50M Stage-2 split as the prior SWA run.
- MMLU is 5-shot; ARC-C, ARC-E, HellaSwag, PIQA, and WinoGrande are zero-shot. The six-task average is unweighted.
- RULER-4K is the unweighted average of all 13 official subtasks, 500 examples each.

## Results

| Model | MMLU | ARC-C | ARC-E | HellaSwag | PIQA | WinoGrande | 6-task avg | WikiText2 PPL ↓ | RULER-4K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SWA(64,4), 0M | 70.29 | 56.40 | 80.77 | 74.11 | 77.80 | 67.72 | 71.18 | 16.41 | 7.22 |
| SWA(64,4), 100M | 60.56 | 55.38 | 80.47 | 71.67 | 77.58 | 68.51 | 69.03 | 16.53 | 4.52 |
| Taylor all-GDN, 100M | 47.88 | 52.82 | 77.74 | 70.19 | 77.42 | 61.96 | 64.67 | 27.21 | 2.12 |
| Taylor 3:1 AR hybrid, 100M | 64.67 | 55.46 | 79.25 | 71.77 | 76.93 | 67.40 | 69.25 | 11.02 | 32.80 |

## Outcome

- Against trained SWA at the same 100M-token budget, the 3:1 Taylor hybrid numerically leads the six-task average by 0.22 points.
- The trained six tasks split 3–3: Taylor wins MMLU, ARC-C, and HellaSwag; SWA wins ARC-E, PIQA, and WinoGrande.
- Taylor has lower WikiText2 PPL (11.02 vs 16.53) and higher RULER-4K (32.80 vs 4.52).
- Against SWA's stronger training-free checkpoint, SWA still leads the short-task average by 1.93 points, while Taylor retains the PPL and RULER advantages.
- Relative to all-36-layer GDN at 100M, the 3:1 hybrid improves the six-task average by 4.58 points and RULER by 30.68 points.
- The 0.22-point trained short-task margin is below the simple independent macro standard errors (Taylor 0.40, SWA 0.40 points). Without paired bootstrap or additional seeds, the short-task result should be treated as a tie rather than a statistically established win.

## Accounting

| Model | Total params | Stage-1 trainable | Stage-2 trainable | Recovery tokens | Calibration |
|---|---:|---:|---:|---:|---|
| SWA(64,4) | 8,190,735,360 | 1,509,958,656 | 8,190,735,360 | 100,000,000 | none |
| Taylor 3:1 AR hybrid | 9,330,273,600 | 2,272,007,232 | 9,330,273,600 | 100,000,000 | 1,024 unique tokens, 8,100 local steps, 0.047 H100-h |

## Interpretation

- The 3:1 architecture is the paper's actual hybrid regime; the previous all-GDN experiment was a stricter operator-matched ablation.
- This establishes short-task parity with trained SWA and a large long-context/PPL numerical advantage.
- It is not a fixed-memory win: nine retained full-attention layers keep a growing KV cache. Any efficiency comparison must report that separately.

## Invalidated or failed attempts

- Single-node 8-GPU submissions remained pending due cluster fragmentation; no optimizer step ran.
- The first 4x2 launch hit a busy CUDA device on h100-087 before distributed initialization; no optimizer step ran.
- The second 4x2 launch reached the first forward but shared partial-node GPU visibility caused OOM; no optimizer step completed.
- The accepted training waited for and used one exclusive 8xH100 node.
- The first short-task evaluation completed the five-task core set, then MMLU batch 8 OOMed on a shared GPU. The accepted MMLU rerun used batch 1 with the same task examples and scoring protocol.
- The first RULER Job landed one shard on h100-087 and produced no accepted results. The accepted Job excluded that node and reran all 13 tasks.

The accepted training completed exactly 50M + 50M tokens at world size 8. All accepted short-task files and all 13 RULER shards passed completeness checks; hashes are in `app2app_hybrid_comparison.json`.
