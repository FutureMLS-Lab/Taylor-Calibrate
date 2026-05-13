"""
Stage 1: Per-layer MSE hidden-state alignment.

Follows the RADLADS pipeline (Goldstein et al., 2025):
- Load full teacher model
- Wrap each converted layer with AttentionDistillationWrapper
  (teacher attention frozen, student GDN attention trainable)
- Teacher hidden states flow through the network
- Student attention trained to match teacher attention output per-layer
- Only .student_attn. parameters are trainable

Usage:
    torchrun --nproc_per_node=8 train_stage1.py --cfg configs/qwen2_1.5b/uniform_baseline_stage1.yaml
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import argparse
import math
import logging
import json

import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
from omegaconf import OmegaConf
from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
)

from distill_model.config_distilled_student import StudentConfig
from distill_model.modeling_distilled_student import (
    StudentForCausalLM, get_student_attention_class
)

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


_DISTILL_LOSSES = []
_COLLECTING_LOSSES = False


class AttentionDistillationWrapper(nn.Module):
    def __init__(self, teacher_attn, student_cls, config, layer_idx):
        super().__init__()
        self.teacher_attn = teacher_attn.eval()
        for p in self.teacher_attn.parameters():
            p.requires_grad_(False)
        self.student_attn = student_cls(config, layer_idx)
        self.student_attn.init_from_teacher(self.teacher_attn)

    def forward(self, *args, **kwargs):
        kwargs["output_attentions"] = False
        kwargs["use_cache"] = False
        with torch.no_grad():
            t_hidden, _, _ = self.teacher_attn(*args, **kwargs)
        s_hidden, _, _ = self.student_attn(*args, **kwargs)
        distill_loss = torch.linalg.vector_norm(
            t_hidden - s_hidden, dim=-1
        ).mean() * (t_hidden.size(-1) ** -0.5)
        if _COLLECTING_LOSSES:
            _DISTILL_LOSSES.append(distill_loss)
        return t_hidden, None, None


class DistillTrainer(Trainer):
    def __init__(self, *args, mse_factor=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.mse_factor = mse_factor

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        global _COLLECTING_LOSSES
        inputs = {k: v.to(model.device) for k, v in inputs.items() if k != "labels"}
        _DISTILL_LOSSES.clear()
        _COLLECTING_LOSSES = True
        model(**inputs)
        _COLLECTING_LOSSES = False

        if _DISTILL_LOSSES:
            loss = torch.stack(_DISTILL_LOSSES).mean() * self.mse_factor
        else:
            loss = torch.tensor(0.0, device=model.device, requires_grad=True)

        _DISTILL_LOSSES.clear()
        return (loss, None) if return_outputs else loss


def count_params(model, trainable_only=True):
    params = filter(lambda p: p.requires_grad, model.parameters()) if trainable_only else model.parameters()
    return sum(p.numel() for p in params)


def _quantize_frozen_linears(model):
    """Replace frozen Linear layers with int8 to halve memory (60GB -> 30GB)."""
    import torch.nn.functional as F

    class Int8Linear(nn.Module):
        def __init__(self, linear):
            super().__init__()
            w = linear.weight.data.float()
            scale = (w.abs().amax(dim=1, keepdim=True) / 127.0).to(torch.bfloat16)
            w_int8 = (w / scale.float()).round().clamp(-127, 127).to(torch.int8)
            self.register_buffer('weight_int8', w_int8)
            self.register_buffer('_scale', scale)
            if linear.bias is not None:
                self.register_buffer('_bias', linear.bias.data.clone())
            else:
                self._bias = None
            self.in_features = linear.in_features
            self.out_features = linear.out_features

        @property
        def weight(self):
            """Dequantize on-the-fly for fla swiglu_linear compatibility."""
            return self.weight_int8.to(self._scale.dtype) * self._scale

        @property
        def bias(self):
            return self._bias

        def forward(self, x):
            w = self.weight_int8.to(x.dtype) * self._scale.to(x.dtype)
            return F.linear(x, w, self._bias)

    count = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if any(p.requires_grad for p in module.parameters()):
            continue
        parts = name.rsplit('.', 1)
        if len(parts) == 2:
            parent = dict(model.named_modules())[parts[0]]
            setattr(parent, parts[1], Int8Linear(module))
            count += 1
    logger.info(f"  Quantized {count} frozen Linear layers to int8")


def build_stage1_model(cfg):
    teacher_name = cfg.teacher_model.name
    logger.info(f"Loading teacher model: {teacher_name}")

    base_config = AutoConfig.from_pretrained(teacher_name)
    base_config.use_cache = False

    model = AutoModelForCausalLM.from_pretrained(
        teacher_name, config=base_config, torch_dtype=torch.bfloat16
    )

    student_attn_class = get_student_attention_class(cfg.student_model.name)
    keep_layers = list(cfg.student_model.get('keep_full_attention_layers', []))
    logger.info(f"Student attention: {student_attn_class.__name__}")
    logger.info(f"Keep full attention layers: {keep_layers}")

    for idx, layer in enumerate(model.model.layers):
        if idx in keep_layers:
            logger.info(f"  Layer {idx}: keep as full attention (frozen)")
            for param in layer.attn.parameters():
                param.requires_grad_(False)
        else:
            logger.info(f"  Layer {idx}: wrap with student GDN attention")
            wrapper = AttentionDistillationWrapper(
                layer.attn, student_attn_class, base_config, idx
            )
            layer.attn = wrapper

    student_init_ckpt = cfg.train.get('student_init_ckpt', None)
    if student_init_ckpt and str(student_init_ckpt) != 'None':
        logger.info(f"Loading student attention weights from: {student_init_ckpt}")
        ckpt_path = os.path.join(str(student_init_ckpt), "model.safetensors")
        index_path = os.path.join(str(student_init_ckpt), "model.safetensors.index.json")
        ckpt_sd = None
        if os.path.exists(ckpt_path):
            from safetensors.torch import load_file
            ckpt_sd = load_file(ckpt_path)
        elif os.path.exists(index_path):
            import glob as glob_mod
            from safetensors.torch import load_file
            shard_files = sorted(glob_mod.glob(os.path.join(str(student_init_ckpt), "model-*.safetensors")))
            logger.info(f"  Loading {len(shard_files)} sharded safetensors files")
            ckpt_sd = {}
            for sf in shard_files:
                ckpt_sd.update(load_file(sf))
        else:
            logger.warning(f"  Checkpoint not found: {ckpt_path}, using init_from_teacher")
        if ckpt_sd is not None:
            loaded = 0
            for idx, layer_mod in enumerate(model.model.layers):
                if idx in keep_layers:
                    continue
                wrapper = layer_mod.attn
                if not hasattr(wrapper, 'student_attn'):
                    continue
                prefix = f"model.layers.{idx}.attn."
                student_sd = {}
                for k, v in ckpt_sd.items():
                    if k.startswith(prefix):
                        student_sd[k[len(prefix):]] = v
                if student_sd:
                    result = wrapper.student_attn.load_state_dict(student_sd, strict=False)
                    loaded += 1
                    if result.missing_keys:
                        logger.info(f"  Layer {idx}: missing keys: {result.missing_keys}")
            logger.info(f"  Loaded student weights for {loaded} GDN layers from checkpoint")

    for name, p in model.named_parameters():
        p.requires_grad_(".student_attn." in name)

    if cfg.train.get('quantize_frozen', True):
        logger.info("Quantizing frozen layers to int8 to reduce memory")
        _quantize_frozen_linears(model)

    if cfg.train.get('gradient_checkpointing', False):
        logger.info("Enabling gradient checkpointing (use_reentrant=False)")
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    tr = count_params(model, True)
    tot = count_params(model, False)
    logger.info(f"Trainable = {tr/1e6:.1f}M | Total = {tot/1e6:.1f}M ({tr/tot:.2%})")
    return model


def main(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.teacher_model.name, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info("==== Stage 1 (MSE Hidden-State Alignment) ====")
    model = build_stage1_model(cfg)

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
        ds_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "deepspeed", "stage_1.json")
    if not os.path.exists(ds_config):
        ds_config = None

    save_steps = cfg.train.get('save_steps', 500)
    training_args = TrainingArguments(
        per_device_train_batch_size=cfg.train.micro_batch_size,
        gradient_accumulation_steps=g_accum,
        max_steps=max_steps,
        bf16=True,
        logging_steps=10,
        eval_strategy="no",
        save_steps=save_steps,
        save_total_limit=cfg.train.get('save_total_limit', 9999),
        save_only_model=True,
        output_dir=cfg.train.output_dir,
        deepspeed=ds_config,
        report_to='none',
        gradient_checkpointing=cfg.train.get('gradient_checkpointing', False),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=cfg.train.lr,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
    )

    attn_params, other_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "attn" in name:
            attn_params.append(param)
        else:
            other_params.append(param)
    logger.info(f"Optimizer: attn={len(attn_params)} params (lr={cfg.train.lr_attn}), "
                f"other={len(other_params)} params (lr={cfg.train.lr})")
    optimizer = torch.optim.AdamW(
        [
            {"params": attn_params, "lr": cfg.train.lr_attn},
            {"params": other_params, "lr": cfg.train.lr},
        ],
        betas=(0.9, 0.95), fused=True,
    )

    trainer = DistillTrainer(
        mse_factor=1.0,
        model=model,
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

    if (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized():
        final_dir = os.path.join(cfg.train.output_dir, "final")
        logger.info(f"Saving final model to {final_dir}")
        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = OmegaConf.create(yaml.safe_load(f))
    main(cfg)
