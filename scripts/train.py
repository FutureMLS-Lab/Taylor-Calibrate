"""
Stage 2 KL distillation training for GDN student model.

Usage (single node, 8 GPUs):
    torchrun --nproc_per_node=8 train.py --cfg configs/qwen2_1.5b/qwen2_1_5b_gdn_v4_stage2.yaml
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import argparse
import math
import logging
import json
import gc

import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
from omegaconf import OmegaConf
from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForCausalLM, TrainingArguments
)
from safetensors.torch import load_file

from distill_model.config_distilled_student import StudentConfig
from distill_model.modeling_distilled_student import StudentForCausalLM
from hf_trainer import KDTrainer

AutoConfig.register('student', StudentConfig, exist_ok=True)
AutoModelForCausalLM.register(StudentConfig, StudentForCausalLM, exist_ok=True)

local_rank = int(os.environ.get("LOCAL_RANK", 0))
os.environ["TRITON_CACHE_DIR"] = os.environ.get("TRITON_CACHE_DIR", f"/tmp/triton_cache/{local_rank}")


def get_logger(name=None):
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    if int(os.environ.get("RANK", "0")) == 0:
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
    return logger


logger = get_logger(__name__)

_original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = patched_torch_load


def count_params(model, trainable_only=True):
    params = filter(lambda p: p.requires_grad, model.parameters()) if trainable_only else model.parameters()
    return sum(p.numel() for p in params)


def build_student(cfg):
    """Load student model from our taylor_calibrate checkpoint."""
    ckpt_path = cfg.train.student_init_ckpt
    logger.info(f"Loading student from: {ckpt_path}")

    config_path = os.path.join(ckpt_path, "config.json")
    with open(config_path) as f:
        config_dict = json.load(f)
    config_dict["fuse_swiglu"] = False
    config_dict["use_cache"] = False
    student_config = StudentConfig(**config_dict)

    from accelerate import init_empty_weights
    with init_empty_weights():
        student = StudentForCausalLM(student_config)
    student.to_empty(device='cpu')
    student = student.to(torch.bfloat16)

    sf_single = os.path.join(ckpt_path, "model.safetensors")
    sf_index = os.path.join(ckpt_path, "model.safetensors.index.json")
    if os.path.exists(sf_single):
        weights = load_file(sf_single, device="cpu")
    elif os.path.exists(sf_index):
        with open(sf_index) as f:
            index = json.load(f)
        weights = {}
        for shard in set(index["weight_map"].values()):
            weights.update(load_file(os.path.join(ckpt_path, shard), device="cpu"))
    else:
        raise FileNotFoundError(f"No safetensors found in {ckpt_path}")
    result = student.load_state_dict(weights, strict=False)
    student.tie_weights()
    logger.info(f"Tied weights; lm_head == embeddings: "
                f"{student.lm_head.weight.data_ptr() == student.model.embeddings.weight.data_ptr()}")
    if result.unexpected_keys:
        logger.warning(f"Unexpected keys: {result.unexpected_keys}")
    logger.info(f"Loaded student weights from {ckpt_path}")

    for p in student.parameters():
        p.requires_grad_(True)

    if cfg.train.get('gradient_checkpointing', False):
        logger.info("Enabling gradient checkpointing for student")
        student.gradient_checkpointing_enable()

    tr = count_params(student, True)
    tot = count_params(student, False)
    logger.info(f"Student: Trainable={tr/1e6:.1f}M | Total={tot/1e6:.1f}M ({tr/tot:.2%})")
    return student


def _quantize_model(model, bits=4):
    """Replace all Linear layers with quantized versions to reduce memory.
    bits=8: per-row int8 (model_size / 2)
    bits=4: per-row int4 packed in int8 (model_size / 4)
    """
    import torch.nn.functional as F

    class QuantizedLinear(nn.Module):
        def __init__(self, linear, bits):
            super().__init__()
            w = linear.weight.data.float()
            if bits == 8:
                scale = (w.abs().amax(dim=1, keepdim=True) / 127.0).to(torch.bfloat16)
                w_q = (w / scale.float()).round().clamp(-127, 127).to(torch.int8)
            else:
                scale = (w.abs().amax(dim=1, keepdim=True) / 7.0).to(torch.bfloat16)
                w_q = (w / scale.float()).round().clamp(-7, 7).to(torch.int8)
            self.register_buffer('weight_q', w_q)
            self.register_buffer('_scale', scale)
            if linear.bias is not None:
                self.register_buffer('_bias', linear.bias.data.clone())
            else:
                self._bias = None
            self.in_features = linear.in_features
            self.out_features = linear.out_features

        def forward(self, x):
            w = self.weight_q.to(x.dtype) * self._scale.to(x.dtype)
            return F.linear(x, w, self._bias)

    count = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        parts = name.rsplit('.', 1)
        if len(parts) == 2:
            parent = dict(model.named_modules())[parts[0]]
            setattr(parent, parts[1], QuantizedLinear(module, bits))
            count += 1
    logger.info(f"  Quantized {count} Linear layers to int{bits}")


def build_teacher(cfg):
    """Load teacher model (frozen), quantize to reduce memory."""
    logger.info(f"Loading teacher from: {cfg.teacher_model.name}")

    teacher = AutoModelForCausalLM.from_pretrained(
        cfg.teacher_model.name, torch_dtype=torch.bfloat16
    )
    teacher.config.use_cache = False
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    param_gb = sum(p.numel() * p.element_size() for p in teacher.parameters()) / 1e9
    if param_gb > 20:
        logger.info(f"  Teacher is {param_gb:.1f}GB, quantizing to int4...")
        _quantize_model(teacher, bits=4)

    logger.info(f"Teacher ready on CPU ({count_params(teacher, False)/1e6:.1f}M params), will move to GPU after student init")
    return teacher


def get_optimizer(model, cfg):
    attn_params, other_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "attn" in name:
            attn_params.append(param)
        else:
            other_params.append(param)

    logger.info(f"Optimizer groups: attn={len(attn_params)} params (lr={cfg.train.lr_attn}), "
                f"other={len(other_params)} params (lr={cfg.train.lr})")

    return torch.optim.AdamW(
        [
            {"params": attn_params, "lr": cfg.train.lr_attn},
            {"params": other_params, "lr": cfg.train.lr},
        ],
        betas=(0.9, 0.95),
        fused=True,
    )


def main(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.teacher_model.name, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    teacher = build_teacher(cfg)
    student = build_student(cfg)

    from data import get_dataloader
    train_loader = get_dataloader(
        cfg.data.cache_dir, batch_size=cfg.train.batch_size, shuffle=True, num_workers=4
    )
    logger.info(f"Dataset: {len(train_loader.dataset)} samples from {cfg.data.cache_dir}")

    world_size = int(os.environ.get("WORLD_SIZE", max(1, torch.cuda.device_count())))
    seq_len = cfg.train.train_seq_len
    micro_total = cfg.train.micro_batch_size * world_size
    g_accum = max(1, math.ceil(cfg.train.batch_size / max(1, micro_total)))
    effective_batch = micro_total * g_accum
    target_tokens = cfg.train.get('target_tokens', None)

    if target_tokens:
        max_steps = max(1, int(target_tokens // (effective_batch * seq_len)))
    else:
        max_steps = cfg.train.get('max_steps', 10000)

    logger.info(f"world_size={world_size}, micro_batch={cfg.train.micro_batch_size}, "
                f"grad_accum={g_accum}, effective_batch={effective_batch}")
    logger.info(f"seq_len={seq_len}, target_tokens={target_tokens}, max_steps={max_steps}")
    logger.info(f"Tokens per step: {effective_batch * seq_len:,}")

    ds_config = cfg.train.get('deepspeed_config', None)
    if ds_config and str(ds_config) != 'None':
        ds_config = str(ds_config)
    else:
        ds_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "deepspeed", "stage_2.json")

    save_steps = cfg.train.get('save_steps', 500)
    training_args = TrainingArguments(
        per_device_train_batch_size=cfg.train.micro_batch_size,
        gradient_accumulation_steps=g_accum,
        max_steps=max_steps,
        bf16=True,
        logging_steps=10,
        eval_strategy="no",
        save_steps=save_steps,
        save_total_limit=cfg.train.get('save_total_limit', 20),
        save_only_model=True,
        output_dir=cfg.train.output_dir,
        deepspeed=ds_config,
        report_to='none',
        gradient_checkpointing=cfg.train.get('gradient_checkpointing', False),
        learning_rate=cfg.train.lr,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        warmup_steps=cfg.train.get('warmup_steps', 0),
    )

    optimizer = get_optimizer(student, cfg)

    trainer = KDTrainer(
        teacher_model=teacher,
        model=student,
        args=training_args,
        train_dataset=train_loader.dataset,
        eval_dataset=None,
        optimizers=(optimizer, None),
        tokenizer=tokenizer,
    )

    resume_ckpt = cfg.train.get('resume_from_checkpoint', None)
    if resume_ckpt == "None" or resume_ckpt is None:
        trainer.train(resume_from_checkpoint=None)
    else:
        trainer.train(resume_from_checkpoint=resume_ckpt)

    if dist.is_initialized() and dist.get_rank() == 0 or not dist.is_initialized():
        final_dir = os.path.join(cfg.train.output_dir, "final")
        logger.info(f"Saving final model to {final_dir}")
        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="Path to YAML config")
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = OmegaConf.create(yaml.safe_load(f))
    main(cfg)
