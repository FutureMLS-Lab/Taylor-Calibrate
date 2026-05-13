#!/bin/bash
set -euo pipefail

usage() {
    echo "Usage: $0 <model_path> [--tasks TASK_GROUP] [--gpus GPU_IDS] [--batch_size BS] [--output_dir DIR]"
    echo ""
    echo "Task groups:"
    echo "  core      : arc_challenge,arc_easy,hellaswag,piqa,winogrande,boolq,lambada_openai,openbookqa,race,sciq,copa"
    echo "  mmlu      : mmlu (57 subtasks, 5-shot)"
    echo "  all       : core + mmlu"
    echo "  custom    : pass --tasks_str directly, e.g. --tasks_str 'arc_challenge,hellaswag'"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/student_checkpoint"
    echo "  $0 /path/to/student_checkpoint --tasks core --gpus 0,1,2,3"
    echo "  $0 /path/to/student_checkpoint --tasks mmlu --batch_size 8"
    echo "  $0 Qwen/Qwen2.5-7B-Instruct --tasks core  # HF teacher model"
    exit 1
}

[[ $# -lt 1 ]] && usage

MODEL_PATH="$1"
shift

TASK_GROUP="core"
GPU_IDS="0,1,2,3,4,5,6,7"
BATCH_SIZE="auto:4"
OUTPUT_DIR=""
TASKS_STR=""
NUM_FEWSHOT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tasks)      TASK_GROUP="$2"; shift 2 ;;
        --gpus)       GPU_IDS="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --tasks_str)  TASKS_STR="$2"; shift 2 ;;
        --num_fewshot) NUM_FEWSHOT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; usage ;;
    esac
done

CORE_TASKS="arc_challenge,arc_easy,hellaswag,piqa,winogrande,boolq,lambada_openai,openbookqa,race,sciq,copa"

if [[ -n "$TASKS_STR" ]]; then
    TASKS="$TASKS_STR"
elif [[ "$TASK_GROUP" == "core" ]]; then
    TASKS="$CORE_TASKS"
elif [[ "$TASK_GROUP" == "mmlu" ]]; then
    TASKS="mmlu"
    NUM_FEWSHOT="${NUM_FEWSHOT:-5}"
elif [[ "$TASK_GROUP" == "all" ]]; then
    TASKS="${CORE_TASKS},mmlu"
else
    echo "Unknown task group: $TASK_GROUP"
    usage
fi

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
    OUTPUT_DIR="results/eval/${MODEL_NAME}/${TASK_GROUP}"
fi
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo " LM-Eval Benchmark Evaluation"
echo "=============================================="
echo "  Model:       $MODEL_PATH"
echo "  Tasks:       $TASKS"
echo "  GPUs:        $GPU_IDS"
echo "  Batch size:  $BATCH_SIZE"
echo "  Output:      $OUTPUT_DIR"
echo "=============================================="

EFFECTIVE_GPUS="${CUDA_VISIBLE_DEVICES:-$GPU_IDS}"
FIRST_GPU=$(echo "$EFFECTIVE_GPUS" | cut -d',' -f1)
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TRITON_CACHE_DIR:-/tmp/triton_cache}}/gpu${FIRST_GPU}"
mkdir -p "$TRITON_CACHE_DIR"

FEWSHOT_ARG=""
if [[ -n "$NUM_FEWSHOT" ]]; then
    FEWSHOT_ARG="--num_fewshot $NUM_FEWSHOT"
fi

HF_ALLOW_CODE_EVAL=1 PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR" CUDA_VISIBLE_DEVICES="$EFFECTIVE_GPUS" \
python eval/harness.py \
    --model hf \
    --model_args "pretrained=${MODEL_PATH},dtype=bfloat16,trust_remote_code=True" \
    --tasks "$TASKS" \
    --batch_size "$BATCH_SIZE" \
    --device cuda \
    --output_path "$OUTPUT_DIR" \
    --seed 0 \
    $FEWSHOT_ARG \
    2>&1 | tee "${OUTPUT_DIR}/eval.log"

echo ""
echo "Results saved to: $OUTPUT_DIR"
