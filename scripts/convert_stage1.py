"""
Convert a Stage 1 (wrapper-based) checkpoint into a clean StudentForCausalLM checkpoint.

After Stage 1 training, the model contains AttentionDistillationWrappers with
.teacher_attn and .student_attn sub-modules. This script extracts the
.student_attn weights, builds a clean StudentForCausalLM, and saves it.

Usage:
    python convert_stage1.py \
        --stage1_dir /path/to/checkpoints/baseline_stage1 \
        --student_name gdn_v4 \
        --keep_layers 0,4,8,12,16,20,24 \
        --teacher /path/to/converted/teacher

    # All-linear conversion (keep no full-attention layers)
    python convert_stage1.py \
        --stage1_dir /path/to/checkpoints/baseline_stage1 \
        --student_name gdn_v4 \
        --keep_layers "" \
        --teacher /path/to/converted/teacher
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import argparse
import json

import torch
from accelerate import init_empty_weights
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from distill_model.config_distilled_student import StudentConfig
from distill_model.modeling_distilled_student import StudentForCausalLM

AutoConfig.register('student', StudentConfig, exist_ok=True)
AutoModelForCausalLM.register(StudentConfig, StudentForCausalLM, exist_ok=True)


def find_latest_checkpoint(base_dir):
    candidates = [d for d in os.listdir(base_dir) if d.startswith("checkpoint-")]
    if not candidates:
        final = os.path.join(base_dir, "final")
        if os.path.exists(final):
            return final
        raise FileNotFoundError(f"No checkpoint found under {base_dir}")
    candidates.sort(key=lambda x: int(x.split("-")[1]))
    return os.path.join(base_dir, candidates[-1])


def load_state_dict_from_dir(ckpt_dir):
    index_path = os.path.join(ckpt_dir, 'model.safetensors.index.json')
    sf_path = os.path.join(ckpt_dir, 'model.safetensors')
    bin_path = os.path.join(ckpt_dir, 'pytorch_model.bin')

    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        sd = {}
        for shard in set(index['weight_map'].values()):
            sd.update(load_file(os.path.join(ckpt_dir, shard), device="cpu"))
        return sd
    elif os.path.exists(sf_path):
        return load_file(sf_path, device="cpu")
    elif os.path.exists(bin_path):
        return torch.load(bin_path, map_location="cpu", weights_only=False)
    raise FileNotFoundError(f"No weights found in {ckpt_dir}")


def parse_keep_layers(raw: str) -> list[int]:
    raw = (raw or "").strip()
    if raw.lower() in {"", "none", "null", "[]"}:
        return []
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_dir", required=True, help="Stage 1 output dir")
    parser.add_argument("--student_name", default="gdn_v4")
    parser.add_argument("--keep_layers", default="0,4,8,12,16,20,24")
    parser.add_argument("--teacher", required=True, help="Teacher model path")
    parser.add_argument("--output", default=None, help="Output dir (default: stage1_dir/converted-hf)")
    args = parser.parse_args()

    keep_layers = parse_keep_layers(args.keep_layers)
    output_dir = args.output or os.path.join(args.stage1_dir, "converted-hf")
    os.makedirs(output_dir, exist_ok=True)

    ckpt_dir = find_latest_checkpoint(args.stage1_dir)
    print(f"Using checkpoint: {ckpt_dir}")

    base_config = AutoConfig.from_pretrained(ckpt_dir)
    config_dict = base_config.to_dict()
    config_dict['student_name'] = args.student_name
    config_dict['name'] = 'student'
    config_dict['keep_full_attention_layers'] = keep_layers
    student_config = StudentConfig(**config_dict)

    with init_empty_weights():
        student_model = AutoModelForCausalLM.from_config(student_config)
    student_model.to_empty(device='cpu')
    student_model = student_model.to(torch.bfloat16)

    raw_sd = load_state_dict_from_dir(ckpt_dir)

    clean_keys = [k for k in raw_sd if k.startswith("module.") or k.startswith("_forward_module.")]
    for k in clean_keys:
        raw_sd[k.replace("module.", "").replace("_forward_module.", "")] = raw_sd.pop(k)

    purified = {}
    for k, v in raw_sd.items():
        if ".student_attn." in k:
            purified[k.replace(".student_attn", "")] = v
        elif ".teacher_attn" not in k:
            purified[k] = v

    student_sd = student_model.state_dict()
    filtered = {}
    skipped_size_mismatch = []
    for k, v in purified.items():
        if k in student_sd and student_sd[k].shape != v.shape:
            skipped_size_mismatch.append(k)
        else:
            filtered[k] = v

    if skipped_size_mismatch:
        print(f"Skipped {len(skipped_size_mismatch)} keys due to size mismatch (will load from teacher):")
        for k in skipped_size_mismatch:
            print(f"  {k}: ckpt={purified[k].shape} vs model={student_sd[k].shape}")

    result = student_model.load_state_dict(filtered, strict=False)
    print(f"Loaded student weights. Missing: {len(result.missing_keys)}, Unexpected: {len(result.unexpected_keys)}")

    all_missing = list(set(result.missing_keys + skipped_size_mismatch))
    if all_missing:
        print(f"\n{len(all_missing)} keys to fill from teacher:")
        for k in sorted(all_missing):
            print(f"  {k}")

        print(f"\nLoading teacher to fill missing/mismatched keys...")
        teacher = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=torch.bfloat16)
        teacher_sd = teacher.state_dict()

        fill = {}
        for k in all_missing:
            if k in teacher_sd:
                fill[k] = teacher_sd[k]
            elif k.replace("self_attn", "attn") in teacher_sd:
                fill[k] = teacher_sd[k.replace("self_attn", "attn")]

        if fill:
            student_model.load_state_dict({**filtered, **fill}, strict=False)
            print(f"Filled {len(fill)} keys from teacher")
        del teacher

    student_model.tie_weights()
    student_model.save_pretrained(output_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Saved clean student model to: {output_dir}")


if __name__ == "__main__":
    main()
