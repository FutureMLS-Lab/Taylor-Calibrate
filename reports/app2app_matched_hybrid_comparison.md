# Structurally Matched Qwen3-8B 3:1 Comparison

Status: validated.

Both students retain the same nine AR-selected full-attention layers. The remaining 27 layers use SWA(64,4) or Taylor-Calibrate GDN. Teacher, data, seed, optimizer, effective batch, and the exact 50M Stage-1 + 50M Stage-2 budget are shared.

| Model | MMLU | ARC-C | ARC-E | HellaSwag | PIQA | WinoGrande | 6-task avg | WikiText2 PPL ↓ | RULER-4K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3:1 SWA(64,4) | 61.79 | 56.57 | 81.44 | 71.69 | 77.75 | 68.82 | 69.68 | 15.88 | 4.71 |
| 3:1 Taylor-GDN | 64.67 | 55.46 | 79.25 | 71.77 | 76.93 | 67.40 | 69.25 | 11.02 | 32.80 |

## Outcome

- SWA numerically leads the six-task average by 0.43 points and wins four of six tasks.
- Taylor wins MMLU and HellaSwag, lowers PPL by 30.6%, and improves RULER-4K by 28.09 points (6.96×).
- The 0.43-point short-task gap is smaller than the independent difference SE estimate (0.57 points), so short-context accuracy should be treated as a tie without paired bootstrap or more seeds.

## Accounting

| Model | Total parameters | Stage-1 trainable | Stage-2 trainable | Recovery tokens |
|---|---:|---:|---:|---:|
| 3:1 SWA(64,4) | 8,190,735,360 | 1,132,468,992 | 8,190,735,360 | 100,000,000 |
| 3:1 Taylor-GDN | 9,330,273,600 | 2,272,007,232 | 9,330,273,600 | 100,000,000 |

Both models retain the same growing KV cache in nine full-attention layers. The converted layers differ: SWA stores a 64-token cache, whereas GDN maintains recurrent state. End-to-end memory and throughput were not measured here.

The accepted SWA evaluation was requeued before execution with a longer deadline after six hours of GPU scarcity; no incomplete evaluation result was reused.
