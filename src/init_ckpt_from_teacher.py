import argparse
import gc
import os
import yaml

import fla  # noqa: F401
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from distill_model.config_distilled_student import StudentConfig
from distill_model.modeling_distilled_student import StudentForCausalLM


AutoConfig.register("student", StudentConfig, exist_ok=True)
AutoModelForCausalLM.register(StudentConfig, StudentForCausalLM, exist_ok=True)


def parse_config(path: str):
    with open(path) as f:
        return yaml.safe_load(f)


def _copy_if_present(dst_module, src_module, name: str):
    if hasattr(dst_module, name) and hasattr(src_module, name):
        dst = getattr(dst_module, name)
        src = getattr(src_module, name)
        if hasattr(dst, "weight") and hasattr(src, "weight") and dst.weight is not None and src.weight is not None:
            dst.weight.data.copy_(src.weight.data)
        if hasattr(dst, "bias") and hasattr(src, "bias") and dst.bias is not None and src.bias is not None:
            dst.bias.data.copy_(src.bias.data)
        if hasattr(dst, "eps") and hasattr(src, "eps"):
            dst.eps = src.eps
        if hasattr(dst, "variance_epsilon") and hasattr(src, "variance_epsilon") and hasattr(dst, "eps"):
            dst.eps = src.variance_epsilon


def build_student_from_teacher(cfg):
    teacher_name = cfg["teacher_model"]["name"]
    student_name = cfg["student_model"]["name"]
    keep_layers = cfg["student_model"].get("keep_full_attention_layers", [])

    teacher_config = AutoConfig.from_pretrained(teacher_name)
    config_dict = teacher_config.to_dict()
    config_dict["name"] = "student"
    config_dict["student_name"] = student_name
    config_dict["keep_full_attention_layers"] = list(keep_layers)
    student_config = StudentConfig(**config_dict)

    dtype = torch.bfloat16
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_name, torch_dtype=dtype, low_cpu_mem_usage=True,
    )
    student = AutoModelForCausalLM.from_config(student_config, torch_dtype=dtype)

    student.model.embeddings.weight.data.copy_(teacher.model.embeddings.weight.data)
    if hasattr(student.model, "norm") and hasattr(teacher.model, "norm"):
        _copy_if_present(student.model, teacher.model, "norm")

    if hasattr(student, "lm_head") and hasattr(teacher, "lm_head"):
        student.lm_head.weight.data.copy_(teacher.lm_head.weight.data)
        if getattr(student.lm_head, "bias", None) is not None and getattr(teacher.lm_head, "bias", None) is not None:
            student.lm_head.bias.data.copy_(teacher.lm_head.bias.data)

    for idx, (student_layer, teacher_layer) in enumerate(zip(student.model.layers, teacher.model.layers)):
        _copy_if_present(student_layer, teacher_layer, "attn_norm")
        _copy_if_present(student_layer, teacher_layer, "mlp_norm")

        if idx in keep_layers:
            student_layer.attn.load_state_dict(teacher_layer.attn.state_dict(), strict=False)
        else:
            student_layer.attn.init_from_teacher(teacher_layer.attn)

        student_layer.mlp.load_state_dict(teacher_layer.mlp.state_dict(), strict=False)
        teacher.model.layers[idx] = None

    del teacher
    gc.collect()
    return student


def main():
    parser = argparse.ArgumentParser(description="Initialize a student/hybrid checkpoint from a converted teacher without training.")
    parser.add_argument("--cfg", required=True, help="Path to stage1/stage2 yaml config.")
    parser.add_argument("--output", default=None, help="Where to save the initialized HF checkpoint.")
    args = parser.parse_args()

    cfg = parse_config(args.cfg)
    output_dir = args.output or os.path.join(cfg["train"]["output_dir"], "init-from-teacher")
    os.makedirs(output_dir, exist_ok=True)

    model = build_student_from_teacher(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg["teacher_model"]["name"])

    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved initialized checkpoint to: {output_dir}")


if __name__ == "__main__":
    main()
