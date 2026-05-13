"""SFT training on chat-format instruction data with assistant-only loss.

Resumes from a Stage 2 final checkpoint, streams togethercomputer/sft-mix-v0.2
(default-with-system config), applies the tokenizer's chat template, and
masks all non-assistant tokens (-100) so cross-entropy is computed only on
assistant responses.

Usage:
    torchrun --nproc_per_node=8 scripts/sft.py --cfg configs/qwen2_1.5b_run/uniform_calibrate_sft.yaml
"""
import sys, os, math, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import torch
torch.serialization.add_safe_globals([])
_orig_load = torch.load
def _load_relaxed(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _load_relaxed

import fla  # noqa
from datasets import load_dataset
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          TrainingArguments, Trainer, DataCollatorForSeq2Seq)


class SFTTrainer(Trainer):
    """Drop num_items_in_batch before calling model — student layers don't accept it."""
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        return (loss, outputs) if return_outputs else loss
from omegaconf import OmegaConf

from distill_model.config_distilled_student import StudentConfig
from distill_model.modeling_distilled_student import StudentForCausalLM

AutoConfig.register("student", StudentConfig, exist_ok=True)
AutoModelForCausalLM.register(StudentConfig, StudentForCausalLM, exist_ok=True)


def build_encode_fn(tokenizer, max_seqlen):
    """Return a row->{input_ids, attention_mask, labels} encoder."""
    def encode(ex):
        conversations = ex["conversations"]
        input_ids, labels = [], []
        prev_text = ""
        for i, turn in enumerate(conversations):
            full_msgs = conversations[: i + 1]
            full_text = tokenizer.apply_chat_template(
                full_msgs, tokenize=False, add_generation_prompt=False
            )
            new_text = full_text[len(prev_text):]
            new_ids = tokenizer(new_text, add_special_tokens=False)["input_ids"]
            if turn["role"] == "assistant":
                labels.extend(new_ids)
            else:
                labels.extend([-100] * len(new_ids))
            input_ids.extend(new_ids)
            prev_text = full_text

        if len(input_ids) > max_seqlen:
            input_ids = input_ids[:max_seqlen]
            labels = labels[:max_seqlen]

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }
    return encode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)
    rank = int(os.environ.get("RANK", 0))

    if rank == 0:
        print(f"[SFT] resume_ckpt: {cfg.train.resume_ckpt}")
        print(f"[SFT] dataset: {cfg.data.name} ({cfg.data.config})")
        print(f"[SFT] target_tokens: {cfg.train.target_tokens:,}")

    # Tokenizer (use teacher's, since student keeps the same vocab)
    tokenizer = AutoTokenizer.from_pretrained(cfg.train.resume_ckpt)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        cfg.train.resume_ckpt, torch_dtype=torch.bfloat16
    )
    model.gradient_checkpointing_enable()

    # Streaming dataset
    raw = load_dataset(cfg.data.name, name=cfg.data.config, split="train", streaming=True)
    encode = build_encode_fn(tokenizer, cfg.train.train_seq_len)
    ds = raw.map(encode, remove_columns=raw.column_names)
    # Filter out examples with no assistant tokens (all -100)
    def has_loss(ex):
        return any(x != -100 for x in ex["labels"])
    ds = ds.filter(has_loss)

    # Training args
    g_accum = cfg.train.batch_size // (cfg.train.micro_batch_size * int(os.environ.get("WORLD_SIZE", 1)))
    tokens_per_step = cfg.train.batch_size * cfg.train.train_seq_len
    max_steps = cfg.train.target_tokens // tokens_per_step
    save_steps = cfg.train.save_steps_tokens // tokens_per_step

    if rank == 0:
        print(f"[SFT] world={os.environ.get('WORLD_SIZE',1)}, micro={cfg.train.micro_batch_size}, "
              f"grad_accum={g_accum}, effective_batch={cfg.train.batch_size}")
        print(f"[SFT] tokens/step={tokens_per_step:,}, max_steps={max_steps}, save_every={save_steps}")

    ds_config = "configs/deepspeed/stage_2.json"

    training_args = TrainingArguments(
        per_device_train_batch_size=cfg.train.micro_batch_size,
        gradient_accumulation_steps=g_accum,
        max_steps=max_steps,
        bf16=True,
        logging_steps=10,
        eval_strategy="no",
        save_steps=save_steps,
        save_total_limit=99,
        save_only_model=True,
        output_dir=cfg.train.output_dir,
        deepspeed=ds_config,
        report_to="none",
        gradient_checkpointing=True,
        learning_rate=cfg.train.lr,
        lr_scheduler_type=cfg.train.get("lr_scheduler_type", "constant"),
        warmup_steps=cfg.train.get("warmup_steps", 100),
        max_grad_norm=cfg.train.get("max_grad_norm", 1.0),
        dataloader_num_workers=cfg.train.get("dataloader_num_workers", 2),
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding="max_length",
                                             max_length=cfg.train.train_seq_len,
                                             label_pad_token_id=-100),
    )

    trainer.train()
    trainer.save_model(os.path.join(cfg.train.output_dir, "final"))
    if rank == 0:
        print("[SFT] Done.")


if __name__ == "__main__":
    main()
