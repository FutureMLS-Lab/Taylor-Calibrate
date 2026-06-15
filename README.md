# Taylor-Calibrate

Distilling softmax-attention Transformers (Qwen, Llama) into hybrid linear-attention students built on **GatedDeltaNet (GDN)**, with a Taylor-series-informed initialization.

The student keeps a small set of **full-attention layers** and replaces the rest with a GDN variant. Training is two stages preceded by an analytical/Taylor initialization.

```
Teacher (HF) ──► [convert] ──► FLA-format teacher
                                   │
                                   │  (optional)
                              [layer selection] ──► keep_full_attention_layers
                                   │
                                   ▼
                              [init student]
                              · zero_gate  (baseline)
                              · taylor_calibrate_init / taylor_calibrate
                                   │
                                   ▼
                          [Stage 1: per-layer MSE alignment]
                                   │
                              [convert stage 1]   ◄── extract clean student
                                   │
                                   ▼
                       [Stage 2: end-to-end KL distillation]
                                   │
                                   ▼
                       [Eval: PPL · lm-eval · RULER]
```

---

## 0 · Environment

```bash
# Python 3.11+ recommended
conda create -n taylor python=3.11 -y && conda activate taylor
pip install -r requirements.txt

# Make src/ importable from anywhere
export PYTHONPATH="$PWD/src:$PWD:$PYTHONPATH"
```

`requirements.txt` includes `fla` (Flash Linear Attention), `lm-eval==0.4.11`, `transformers`, `deepspeed`. CUDA + Triton are required for `fla` kernels.

`student_model='gdn_v4'` is the default attention class. To use a different student attention (`gdn_v1..v6`, `gsa`, `gla`, etc.), change the `name` field in the YAML — see `src/distill_model/student_layers.py` for the full list.

---

## 1 · Convert teacher

Convert a HuggingFace teacher into the FLA-compatible format expected by `train_stage1.py`.

```bash
# pick the matching converter
python convert/convert_from_qwen2.5.py    --hf Qwen/Qwen2.5-1.5B-Instruct   --out ./teachers/Qwen2.5-1.5B-Instruct
python convert/convert_from_qwen3.py      --hf Qwen/Qwen3-8B                --out ./teachers/Qwen3-8B
python convert/convert_from_llama3.2.py   --hf meta-llama/Llama-3.2-3B-Instruct --out ./teachers/Llama-3.2-3B-Instruct
```

The output dir is then referenced as `teacher_model.name` in the YAML configs.

---

## 2 · Layer selection (optional, run once per teacher)

The student keeps `keep_full_attention_layers` as full-attention; the rest become GDN. Two methods to pick which layers to keep:

```bash
# AR: bypass each layer, measure Wikitext-2 PPL, rank.
python scripts/ar_layer_selection.py \
    --teacher Qwen/Qwen2.5-1.5B-Instruct \
    --num-keep 7 \
    --output ./results/ar_qwen2_1.5b.json

# GA-S2: greedy selection where each candidate is a full init+zero-shot eval.
python scripts/greedy_layer_selection.py \
    --teacher Qwen/Qwen2.5-1.5B-Instruct \
    --strategy taylor_calibrate \
    --num-keep 7 \
    --output ./results/ga_s2_qwen2_1.5b.json
```

Copy the chosen list into the `student_model.keep_full_attention_layers` field of your YAML. The `configs/qwen2_1.5b/*.yaml` already encode the picks used in the paper for that model.

---

## 3 · Initialize student

Two strategies — pick **one**.

```bash
# (a) Baseline: copy teacher weights, zero the GDN gate.
python src/init_ckpt_from_teacher.py \
    --cfg configs/qwen2_1.5b/uniform_baseline_stage1.yaml \
    --strategy zero_gate

# (b) Taylor-calibrated: closed-form analytical init + ~300 per-layer SGD steps
#     against the teacher attention output on a tiny calibration set.
python src/init_ckpt_calibrated.py \
    --cfg configs/qwen2_1.5b/uniform_calibrate_stage1.yaml \
    --strategy taylor_calibrate \
    --hf-teacher Qwen/Qwen2.5-1.5B-Instruct
```

Other strategies are listed in `src/init_ckpt_calibrated.py::STRATEGIES`. The output checkpoint path is read from `train.student_init_ckpt` in the YAML and is the input for Stage 1.

This step alone gives a usable **zero-shot** student — eval before Stage 1 if you want a clean Stage-2-baseline RULER number.

---

## 4 · Preprocess training data (once)

Stage 1 and Stage 2 both load chunked, tokenized text from a local cache.

```bash
# Stage 1 uses short-context (e.g. 512) chunks
python scripts/prep_stage1_data.py \
    --tokenizer Qwen/Qwen2.5-1.5B-Instruct \
    --seq-len 512 \
    --out ./data/chunked_context512

# Stage 2 uses the long-context (e.g. 4096) version
python scripts/preprocess_data.py \
    --tokenizer Qwen/Qwen2.5-1.5B-Instruct \
    --seq-len 4096 \
    --out ./data/chunked_context4096
```

Edit the YAML's `data.cache_dir` to match.

---

## 5 · Stage 1 — per-layer MSE alignment

Wraps each non-kept teacher attention in an `AttentionDistillationWrapper` that runs both teacher (frozen, no_grad) and student attention on the same input; per-layer L2 loss is averaged into the training objective. Teacher-forced — layers train independently.

```bash
torchrun --nproc_per_node=8 scripts/train_stage1.py \
    --cfg configs/qwen2_1.5b/uniform_calibrate_stage1.yaml
```

Do not enable gradient checkpointing here — see [Future work](#future-work).

---

## 6 · Convert Stage 1 → clean student

Stage 1 saves the wrapped (teacher+student) graph. Strip it down to a vanilla `StudentForCausalLM` checkpoint that Stage 2 can load.

```bash
python scripts/convert_stage1.py \
    --stage1_dir ./checkpoints/calibrate_stage1 \
    --student_name gdn_v4 \
    --keep_layers 0,4,8,12,16,20,24 \
    --teacher ./teachers/Qwen2.5-1.5B-Instruct
# writes ./checkpoints/calibrate_stage1/converted-hf/
```

The Stage-2 YAML's `train.student_init_ckpt` should point at the `converted-hf/` dir.

---

## 7 · Stage 2 — end-to-end KL distillation

KL divergence between student and teacher logits over the full sequence. Teacher is loaded once (CPU → GPU on first batch).

```bash
torchrun --nproc_per_node=8 scripts/train.py \
    --cfg configs/qwen2_1.5b/uniform_calibrate_stage2.yaml
```

DeepSpeed configs in `configs/deepspeed/`:
- `stage_1.json` — ZeRO-1 (Stage 1 default)
- `stage_2.json` — ZeRO-2 (Stage 2 default)
- `stage_2_offload.json` — OOM escape hatches.

---

## 8 · (Optional) SFT for instruction tuning

After Stage 2, an instruction SFT pass on a chat-format dataset.

```bash
torchrun --nproc_per_node=8 scripts/sft.py \
    --cfg configs/<model>/uniform_calibrate_sft.yaml
```

> **Note:** No SFT config is shipped in this repo. Create one following the Stage 2 YAML layout, replacing `student_init_ckpt` with the Stage 2 final dir and adding a `sft_dataset` field.

The default config streams a chat-format dataset, applies the tokenizer's chat template, and masks all non-assistant tokens (`-100`) so cross-entropy is computed only on assistant responses.

---

## 9 · Evaluation

All eval scripts default their output dir to `results/eval/<basename(ckpt)>/<task_group>/`.

```bash
# Wikitext-2 perplexity (single GPU, fast)
python scripts/quick_ppl_eval.py --ckpt ./checkpoints/calibrate_stage2/final

# lm-eval harness — core (ARC, HellaSwag, etc.) / MMLU 5-shot
bash scripts/eval_benchmarks.sh ./checkpoints/calibrate_stage2/final --tasks core
bash scripts/eval_benchmarks.sh ./checkpoints/calibrate_stage2/final --tasks mmlu

# RULER long-context — defaults to seq 4096
bash scripts/eval_ruler.sh ./checkpoints/calibrate_stage2/final --seq_lengths '4096,8192'
```

### RULER eval — patched variant

Some `transformers` × `fla` versions silently collapse RULER multi-needle scores to near-zero. `scripts/ruler_eval_patched.py` mirrors `eval/harness.py` with defensive monkey-patches — use it in place of the `eval_ruler.sh` wrapper if you see scores collapse. See [Future work](#future-work) for the underlying interactions.

```bash
python scripts/ruler_eval_patched.py \
    --model hf \
    --model_args "pretrained=./checkpoints/calibrate_stage2/final,dtype=bfloat16,trust_remote_code=True,max_length=4096" \
    --tasks ruler --batch_size 1 --device cuda \
    --output_path ./results/eval/calibrate_stage2/ruler/seq4096
```

---

## End-to-end one-shot recipe (Qwen2.5-1.5B, uniform-calibrate variant)

```bash
# 0 · env
export PYTHONPATH="$PWD/src:$PWD:$PYTHONPATH"

# 1 · convert teacher
python convert/convert_from_qwen2.5.py \
    --hf Qwen/Qwen2.5-1.5B-Instruct \
    --out ./teachers/Qwen2.5-1.5B-Instruct

# 2 · prep data
python scripts/prep_stage1_data.py --tokenizer Qwen/Qwen2.5-1.5B-Instruct --seq-len 512  --out ./data/chunked_context512
python scripts/preprocess_data.py  --tokenizer Qwen/Qwen2.5-1.5B-Instruct --seq-len 4096 --out ./data/chunked_context4096

# 3 · taylor calibrate
python src/init_ckpt_calibrated.py \
    --cfg configs/qwen2_1.5b/uniform_calibrate_stage1.yaml \
    --strategy taylor_calibrate \
    --hf-teacher Qwen/Qwen2.5-1.5B-Instruct

# 4 · stage 1
torchrun --nproc_per_node=8 scripts/train_stage1.py \
    --cfg configs/qwen2_1.5b/uniform_calibrate_stage1.yaml

# 5 · convert stage 1
python scripts/convert_stage1.py \
    --stage1_dir ./checkpoints/calibrate_stage1 \
    --student_name gdn_v4 \
    --keep_layers 0,4,8,12,16,20,24 \
    --teacher ./teachers/Qwen2.5-1.5B-Instruct

# 6 · stage 2
torchrun --nproc_per_node=8 scripts/train.py \
    --cfg configs/qwen2_1.5b/uniform_calibrate_stage2.yaml

# 7 · eval
python scripts/quick_ppl_eval.py    --ckpt ./checkpoints/calibrate_stage2/final
bash   scripts/eval_benchmarks.sh   ./checkpoints/calibrate_stage2/final --tasks core
bash   scripts/eval_benchmarks.sh   ./checkpoints/calibrate_stage2/final --tasks mmlu
python scripts/ruler_eval_patched.py \
    --model hf \
    --model_args "pretrained=./checkpoints/calibrate_stage2/final,dtype=bfloat16,trust_remote_code=True,max_length=4096" \
    --tasks ruler --batch_size 1 --device cuda \
    --output_path ./results/eval/calibrate_stage2/ruler/seq4096
```

To reproduce the **baseline-init** ablation (no Taylor), substitute step 3 with
`python src/init_ckpt_from_teacher.py --cfg configs/qwen2_1.5b/uniform_baseline_stage1.yaml --strategy zero_gate` and use the matching `uniform_baseline_*` configs in steps 4-6.

---

## Variants

The `configs/qwen2_1.5b/` directory ships 6 (variant × init-strategy) configurations:

| layer-selection | init      | stage 1 yaml                          | stage 2 yaml                          |
|-----------------|-----------|---------------------------------------|---------------------------------------|
| Uniform (every-Nth) | zero-gate baseline | `uniform_baseline_stage1.yaml` | `uniform_baseline_stage2.yaml` |
| Uniform | Taylor calibrate | `uniform_calibrate_stage1.yaml`   | `uniform_calibrate_stage2.yaml`   |
| AR (PPL ranking) | zero-gate baseline | `ar_baseline_stage1.yaml`      | `ar_baseline_stage2.yaml`      |
| AR | Taylor calibrate | `ar_calibrate_stage1.yaml`        | `ar_calibrate_stage2.yaml`        |
| GA-S2 (greedy approx) | zero-gate baseline | `ga_s2_baseline_stage1.yaml`   | `ga_s2_baseline_stage2.yaml`   |
| GA-S2 | Taylor calibrate | `ga_s2_calibrate_stage1.yaml`     | `ga_s2_calibrate_stage2.yaml`     |

`configs/qwen2_1.5b/qwen2_1_5b_gdn_v4.yaml` is a standalone single-file config (no separate stages) used by ad-hoc scripts.

Add new (model, variant) combinations under `configs/<model>/<variant>_stage{1,2}.yaml` following the same field layout.

---

## Repository layout

```
.
├── README.md                              # this file
├── requirements.txt
├── src/
│   ├── distill_model/
│   │   ├── config_distilled_student.py    # StudentConfig (Auto-registered)
│   │   ├── modeling_distilled_student.py  # StudentForCausalLM, layer dispatch
│   │   ├── student_layers.py              # GDN-v1..v6, GSA, GLA, PaTH, SWA students
│   │   └── custom_gdn.py                  # local fork of fla GDN internals
│   ├── data.py                            # streaming/chunk dataloaders
│   ├── hf_trainer.py                      # KDTrainer (Stage 2 KL-distillation trainer)
│   ├── init_ckpt_from_teacher.py          # zero-gate / vanilla student init
│   └── init_ckpt_calibrated.py            # Taylor / taylor-calibrate (~15 strategies)
├── scripts/
│   ├── train_stage1.py                    # Stage 1 (per-layer MSE)
│   ├── convert_stage1.py                  # extract clean student from Stage 1 wrapper
│   ├── train.py                           # Stage 2 (full KL distillation)
│   ├── sft.py                             # optional instruction-tuning SFT
│   ├── prep_stage1_data.py                # short-ctx tokenize+chunk for Stage 1
│   ├── preprocess_data.py                 # long-ctx tokenize+chunk for Stage 2
│   ├── ar_layer_selection.py              # PPL-based keep-layer selection
│   ├── greedy_layer_selection.py          # GA-S2 greedy selection
│   ├── quick_ppl_eval.py                  # Wikitext-2 perplexity
│   ├── eval_benchmarks.sh                 # lm-eval harness wrapper
│   ├── eval_ruler.sh                      # RULER wrapper
│   └── ruler_eval_patched.py              # RULER eval w/ defensive FLA patches
├── convert/
│   ├── convert_from_qwen2.5.py
│   ├── convert_from_qwen3.py
│   └── convert_from_llama3.2.py
├── eval/
│   ├── harness.py                         # lm-eval entry point + StudentConfig registration
│   ├── ppl_eval.py
│   ├── collect_results.py
│   └── __init__.py
└── configs/
    ├── deepspeed/
    │   ├── stage_1.json                   # ZeRO-1
    │   ├── stage_2.json                   # ZeRO-2
    │   ├── stage_2_offload.json           # ZeRO-2 + CPU offload
    └── qwen2_1.5b/                        # 6 variants × stage{1,2} = 12 yamls
```

---

## Conventions

- **Output directories** in YAMLs are relative `./checkpoints/...` paths. Override per-machine via the YAML or by symlinking `./checkpoints` to a fast disk.
- **Teacher names** in `teacher_model.name`: Stage 1 wants the **converted** FLA path (e.g. `./teachers/Qwen2.5-1.5B-Instruct`); Stage 2 wants the original HF hub ID. The converter scripts produce the FLA path.
- **`student_name` is consumed only when *building* the model.** Once a checkpoint is saved, switching the field in `config.json` and reloading silently loads weights into the wrong attention class. Always check `config.json` before reloading.
- **Triton cache**: each rank should have its own dir (`$TRITON_CACHE_DIR/<local_rank>`); `train.py` and `train_stage1.py` already set this.
- **`weights_only` torch.load**: HF checkpoints predate the safe-load default; both training scripts monkey-patch `torch.load` to set `weights_only=False`.

---

## Future work

- **Gradient checkpointing for Stage 1.** Not yet supported — per-layer losses are collected through a module-global list, which collides with reentrant recomputation (`use_reentrant=True` zeros gradients silently; `use_reentrant=False` double-appends and shape-mismatches). For very large teachers, int-quantize frozen layers as a stand-in.
- **Drop the RULER eval patches.** `scripts/ruler_eval_patched.py` defensively monkey-patches around two upstream interactions on long-context eval: a KV-cache + `unpad_input` crash in `fla.layers.attn.Attention.forward`, and first-prefill truncation in `StudentForCausalLM.prepare_inputs_for_generation` with transformers ≥4.56. Plan: retire the patched runner once the upstream fixes land.
