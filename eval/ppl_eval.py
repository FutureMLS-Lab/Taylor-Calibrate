from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


CARE_SRC = Path(__file__).resolve().parents[2] / "CARE" / "src"
if str(CARE_SRC) not in sys.path:
    sys.path.insert(0, str(CARE_SRC))

from utils import get_dataset, prepare_test_dataloader  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate perplexity with CARE's Wikitext loader on hybrid-distillation models.")
    parser.add_argument("--model", choices=["hf", "qwen2_second_order"], required=True)
    parser.add_argument("--pretrained", required=True, help="HF repo id or local checkpoint path.")
    parser.add_argument("--cfg", default=None, help="Stage config path used to read keep_full_attention_layers for qwen2_second_order.")
    parser.add_argument("--keep_layers", default=None, help="Comma-separated kept full-attention layers, overrides cfg.")
    parser.add_argument("--order", type=int, default=2, help="Approximation order for qwen2_second_order.")
    parser.add_argument("--decay", type=float, default=0.9, help="Unused for the current Taylor-state implementation; kept for CLI compatibility.")
    parser.add_argument("--mode", default="taylor", choices=["taylor", "naive_linear", "gdn_taylor"])
    parser.add_argument("--dataset", default="wikitext2", help="CARE dataset name. Recommended: wikitext2.")
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def _dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[name]


def _parse_keep_layers(keep_layers: str | None, cfg: str | None) -> set[int]:
    if keep_layers:
        return {int(part.strip()) for part in keep_layers.split(",") if part.strip()}
    if cfg:
        with open(cfg) as f:
            cfg_dict = yaml.safe_load(f)
        return set(cfg_dict["student_model"].get("keep_full_attention_layers", []))
    return set()


def _model_device(model: torch.nn.Module) -> torch.device:
    input_embeddings = model.get_input_embeddings()
    if input_embeddings is not None and hasattr(input_embeddings, "weight"):
        return input_embeddings.weight.device
    return next(model.parameters()).device


def _register_student_model_if_needed(pretrained: str):
    config_path = Path(pretrained) / "config.json"
    if not config_path.exists():
        return
    with open(config_path) as f:
        config_dict = json.load(f)
    if config_dict.get("model_type") != "student":
        return
    import fla  # noqa: F401
    from distill_model.config_distilled_student import StudentConfig
    from distill_model.modeling_distilled_student import StudentForCausalLM

    AutoConfig.register("student", StudentConfig, exist_ok=True)
    AutoModelForCausalLM.register(StudentConfig, StudentForCausalLM, exist_ok=True)


def _load_model_and_tokenizer(args):
    dtype = _dtype_from_name(args.dtype)
    common_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": args.trust_remote_code,
    }

    if args.model == "hf":
        _register_student_model_if_needed(args.pretrained)
        model = AutoModelForCausalLM.from_pretrained(args.pretrained, **common_kwargs)
    else:
        from eval.harness import _patch_qwen2_with_second_order

        keep_layers = _parse_keep_layers(args.keep_layers, args.cfg)
        model = AutoModelForCausalLM.from_pretrained(
            args.pretrained,
            attn_implementation="eager",
            **common_kwargs,
        )
        _patch_qwen2_with_second_order(
            model,
            order=int(args.order),
            keep_layers=keep_layers,
            decay=float(args.decay),
            mode=args.mode,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained, trust_remote_code=args.trust_remote_code, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device(args.device)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def evaluate_ppl(model: torch.nn.Module, testloader, pad_token_id: int | None) -> float:
    model.eval()
    device = _model_device(model)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_id)

    start = time.time()
    nlls = []
    for batch_idx, batch in enumerate(tqdm(testloader, desc="Evaluating perplexity")):
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch, use_cache=False).logits
        if not torch.isfinite(logits).all():
            raise RuntimeError(f"Non-finite logits detected at batch {batch_idx}.")
        logits = logits[:, :-1, :]
        shift_labels = batch["input_ids"][:, 1:]
        nll = loss_fn(logits.permute(0, 2, 1), shift_labels).float()
        if not torch.isfinite(nll).all():
            raise RuntimeError(f"Non-finite token losses detected at batch {batch_idx}.")
        valid = shift_labels != loss_fn.ignore_index
        nll_mean = (nll * valid).sum(dim=1) / valid.sum(dim=1)
        if not torch.isfinite(nll_mean).all():
            raise RuntimeError(f"Non-finite mean losses detected at batch {batch_idx}.")
        nlls.append(nll_mean)

    ppl = torch.exp(torch.cat(nlls).mean()).item()
    logging.info("PPL evaluation finished in %.2fs", time.time() - start)
    return ppl


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    logging.info("Loading model: %s", args.pretrained)
    model, tokenizer = _load_model_and_tokenizer(args)

    logging.info("Loading dataset via CARE: %s", args.dataset)
    dataset = get_dataset(args.dataset)
    test_loader = prepare_test_dataloader(
        dataset=dataset["test"],
        tokenizer=tokenizer,
        seqlen=args.seqlen,
        batch_size=args.batch_size,
    )

    ppl = evaluate_ppl(model, test_loader, tokenizer.pad_token_id)
    print(f"PPL ({args.dataset}, seqlen={args.seqlen}): {ppl:.4f}")


if __name__ == "__main__":
    main()
