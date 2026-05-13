#!/bin/bash
set -euo pipefail

usage() {
    echo "Usage: $0 <model_path> [--gpus GPU_IDS] [--seq_lengths '4096,8192,16384'] [--batch_size BS] [--output_dir DIR]"
    echo ""
    echo "RULER subtasks: ruler_cwe, ruler_fwe, ruler_qa_hotpot, ruler_qa_squad, ruler_vt"
    echo "  cwe       : Common Words Extraction"
    echo "  fwe       : Frequent Words Extraction"
    echo "  qa_hotpot : QA (HotpotQA)"
    echo "  qa_squad  : QA (SQuAD)"
    echo "  vt        : Variable Tracking"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/checkpoint"
    echo "  $0 /path/to/checkpoint --seq_lengths '4096,8192'"
    echo "  $0 /path/to/checkpoint --gpus 0,1,2,3 --batch_size 4"
    exit 1
}

[[ $# -lt 1 ]] && usage

MODEL_PATH="$1"
shift

GPU_IDS="0,1,2,3,4,5,6,7"
SEQ_LENGTHS="4096"
BATCH_SIZE="1"
OUTPUT_DIR=""
RULER_TASKS="ruler"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)        GPU_IDS="$2"; shift 2 ;;
        --seq_lengths) SEQ_LENGTHS="$2"; shift 2 ;;
        --batch_size)  BATCH_SIZE="$2"; shift 2 ;;
        --output_dir)  OUTPUT_DIR="$2"; shift 2 ;;
        --tasks)       RULER_TASKS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; usage ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Activate conda environment (disable nounset temporarily for conda compatibility)
set +u
for CONDA_BASE in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
  if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    break
  fi
done
conda activate taylor
set -u

MODEL_NAME=$(basename "$MODEL_PATH")
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="results/eval/${MODEL_NAME}/ruler"
fi
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo " RULER Long-Context Evaluation"
echo "=============================================="
echo "  Model:       $MODEL_PATH"
echo "  Tasks:       $RULER_TASKS"
echo "  Seq lengths: $SEQ_LENGTHS"
echo "  GPUs:        $GPU_IDS"
echo "  Batch size:  $BATCH_SIZE"
echo "  Output:      $OUTPUT_DIR"
echo "=============================================="

EFFECTIVE_GPUS="${CUDA_VISIBLE_DEVICES:-$GPU_IDS}"
FIRST_GPU=$(echo "$EFFECTIVE_GPUS" | cut -d',' -f1)
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TRITON_CACHE_DIR:-/tmp/triton_cache}}/gpu${FIRST_GPU}"
mkdir -p "$TRITON_CACHE_DIR"

IFS=',' read -ra LENGTHS <<< "$SEQ_LENGTHS"
for LEN in "${LENGTHS[@]}"; do
    LEN=$(echo "$LEN" | tr -d ' ')
    echo ""
    echo ">>> Evaluating at max_seq_length = $LEN"
    echo "-------------------------------------------"

    LEN_DIR="${OUTPUT_DIR}/seq${LEN}"
    mkdir -p "$LEN_DIR"

    HF_ALLOW_CODE_EVAL=1 PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_IDS}" \
    python eval/harness.py \
        --model hf \
        --model_args "pretrained=${MODEL_PATH},dtype=bfloat16,trust_remote_code=True,max_length=${LEN}" \
        --tasks "$RULER_TASKS" \
        --batch_size "$BATCH_SIZE" \
        --device cuda \
        --output_path "$LEN_DIR" \
        --metadata="{\"max_seq_lengths\":[${LEN}]}" \
        --seed 0 \
        2>&1 | tee "${LEN_DIR}/eval.log"

    echo "  Results for seq_len=$LEN saved to: $LEN_DIR"
done

echo ""
echo "All RULER results saved to: $OUTPUT_DIR"
