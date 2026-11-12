<h1 align="center">Taylor-Calibrate: Principled Initialization for Hybrid Linear Attention Distillation</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2606.16429"><img alt="Paper Link" src="https://img.shields.io/badge/Paper%20Link-arXiv%3A2606.16429-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white"></a>
</p>

This repo converts a softmax-attention Transformer (Qwen, Llama) into a hybrid model. The hybrid keeps a few softmax layers and replaces the rest with Gated DeltaNet (GDN) layers.

Taylor-Calibrate initializes each new GDN layer in two phases:

- **Phase 1** sets the layer's decay, write strength, and output scale from teacher-attention statistics.
- **Phase 2** aligns the layer's output with the teacher's for 300 steps.

The student then goes through two recovery stages:

- **Stage 1** aligns hidden states layer by layer.
- **Stage 2** runs end-to-end KL distillation.

## Install

```bash
conda create -n taylor python=3.11 -y && conda activate taylor
pip install -r requirements.txt
export PYTHONPATH="$PWD/src:$PWD:$PYTHONPATH"
```

Inside an NGC PyTorch container, keep the container's PyTorch:

```bash
pip list --format=freeze | grep -E "^(torch|triton|numpy)==" > constraints.txt
pip install -c constraints.txt -r requirements.txt
```

## Run

Example: Qwen2.5-1.5B with 7 of its 28 softmax layers kept.

```bash
# 1. Convert the teacher (--config must match the model size)
python convert/convert_from_qwen2.5.py --model Qwen/Qwen2.5-1.5B-Instruct --config 1.5b \
    --output ./teachers/Qwen2.5-1.5B-Instruct

# 2. Tokenize FineWeb-Edu into ./data (default cap 2B tokens; set --max_tokens to change it)
python scripts/preprocess_data.py --tokenizer Qwen/Qwen2.5-1.5B-Instruct --context_length 4096 \
    --output_dir ./data --streaming
SRC_NPY=./data/all_tokens.npy DST_DIR=./data python scripts/prep_stage1_data.py

# 3. Initialize with Taylor-Calibrate (writes ./checkpoints/init-calibrate)
python src/init_ckpt_calibrated.py --cfg configs/qwen2_1.5b/uniform_calibrate_stage1.yaml \
    --strategy taylor_calibrate --hf-teacher Qwen/Qwen2.5-1.5B-Instruct --seed 0

# 4. Stage 1, then extract the student
torchrun --nproc_per_node=8 scripts/train_stage1.py --cfg configs/qwen2_1.5b/uniform_calibrate_stage1.yaml
python scripts/convert_stage1.py --stage1_dir ./checkpoints/calibrate_stage1 --student_name gdn_v4 \
    --keep_layers 0,4,8,12,16,20,24 --teacher ./teachers/Qwen2.5-1.5B-Instruct

# 5. Stage 2
torchrun --nproc_per_node=8 scripts/train.py --cfg configs/qwen2_1.5b/uniform_calibrate_stage2.yaml
```

Variations:

- **Baseline (projection copy):** skip step 3 and use the `uniform_baseline_*` configs.
- **Qwen3-8B:** use `convert_from_qwen3.py --config 8b`, the `configs/qwen3_8b/uniform_*` configs, and `--keep_layers 0,4,8,12,16,20,24,28,32`.
- **Llama-3.2-3B:** use `convert_from_llama3.2.py --config 3b`.

## Initialization strategies

Pass one of these to `--strategy`:

| strategy | what it does |
|---|---|
| `taylor_calibrate` | Phase 1 + Phase 2 (the method) |
| `taylor_calibrate_init` | Phase 1 only |
| `alignment_only` | Phase 2 only |
| `zero_gate` / `small_gate` | zero the output gate, or scale it by 0.02 |
| `zero_gate_align` | zero the output gate, then Phase 2 |
| `rms_ols` / `output_only` | Phase-1 ablations |

## Evaluate

```bash
CKPT=./checkpoints/calibrate_stage2/final
python scripts/quick_ppl_eval.py --ckpt $CKPT                      # WikiText-2 test split
python scripts/quick_ppl_eval.py --ckpt $CKPT --split validation   # disjoint from the calibration tokens
bash scripts/eval_benchmarks.sh $CKPT --tasks core                 # ARC, HellaSwag, PIQA, ...
bash scripts/eval_benchmarks.sh $CKPT --tasks mmlu
bash scripts/eval_ruler.sh $CKPT --seq_lengths '4096,8192'         # add --limit N for a quick check
```

## Notes

- **Init checkpoint:** Stage 1 loads `train.student_init_ckpt` from its YAML and stops with an error if that directory is missing.
- **Teacher path:** Stage 1 uses the converted teacher (`./teachers/...`); Stage 2 uses the Hugging Face model ID.
- **Gradient checkpointing:** not supported in Stage 1.
- **RULER:** `eval_ruler.sh` runs `scripts/ruler_eval_patched.py`, which avoids RULER score collapse with some transformers × fla versions.
- **Layer selection:** the shipped configs already contain the paper's choices. `scripts/ar_layer_selection.py` and `scripts/greedy_layer_selection.py` produce new ones.

## Citation

```bibtex
@article{zhou2026taylorcalibrate,
  title   = {Taylor-Calibrate: Principled Initialization for Hybrid Linear Attention Distillation},
  author  = {Zhou, Zhongzhu and Wu, Qingyang and Wang, Junxiong and Mishra, Mayank and Song, Shuaiwen Leon and Athiwaratkun, Ben and Xu, Chenfeng},
  journal = {arXiv preprint arXiv:2606.16429},
  year    = {2026},
  url     = {https://arxiv.org/abs/2606.16429}
}
```
