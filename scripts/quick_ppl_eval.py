"""Quick PPL evaluation on Wikitext2 for student checkpoints."""
import argparse
import sys
import json
import os
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from safetensors.torch import load_file

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from distill_model.config_distilled_student import StudentConfig
from distill_model.modeling_distilled_student import StudentForCausalLM

AutoConfig.register('student', StudentConfig, exist_ok=True)
AutoModelForCausalLM.register(StudentConfig, StudentForCausalLM, exist_ok=True)


def load_student(ckpt_path, device='cuda'):
    config_path = os.path.join(ckpt_path, "config.json")
    with open(config_path) as f:
        config_dict = json.load(f)
    config_dict.setdefault("fuse_swiglu", False)
    config_dict["use_cache"] = False
    config = StudentConfig(**config_dict)

    from accelerate import init_empty_weights
    with init_empty_weights():
        model = StudentForCausalLM(config)
    model.to_empty(device='cpu')
    model = model.to(torch.bfloat16)

    single = os.path.join(ckpt_path, "model.safetensors")
    if os.path.exists(single):
        weights = load_file(single, device="cpu")
    else:
        import glob
        weights = {}
        for shard in sorted(glob.glob(os.path.join(ckpt_path, "model-*.safetensors"))):
            weights.update(load_file(shard, device="cpu"))
    model.load_state_dict(weights, strict=False)
    if config.tie_word_embeddings:
        model.lm_head.weight = model.model.embeddings.weight
    return model.to(device).eval()


@torch.no_grad()
def evaluate_ppl(model, tokenizer, seqlen=2048, device='cuda'):
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    text = "\n\n".join(ds['text'])
    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings.input_ids.to(device)

    total_nll = 0.0
    total_tokens = 0
    n_chunks = input_ids.size(1) // seqlen

    for i in range(n_chunks):
        chunk = input_ids[:, i * seqlen:(i + 1) * seqlen]
        outputs = model(chunk)
        logits = outputs.logits[:, :-1, :].contiguous()
        targets = chunk[:, 1:].contiguous()

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                               targets.view(-1), reduction='sum')
        total_nll += loss.item()
        total_tokens += targets.numel()

        if (i + 1) % 20 == 0:
            ppl_so_far = torch.exp(torch.tensor(total_nll / total_tokens)).item()
            print(f"  chunk {i+1}/{n_chunks}: PPL={ppl_so_far:.2f}")

    avg_nll = total_nll / total_tokens
    ppl = torch.exp(torch.tensor(avg_nll)).item()
    return ppl


def is_student_checkpoint(path):
    """Check if a path is a local student checkpoint vs an HF model name."""
    config_path = os.path.join(path, "config.json")
    if not os.path.exists(config_path):
        return False
    with open(config_path) as f:
        cfg = json.load(f)
    return cfg.get("model_type") == "student"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    tok_path = args.tokenizer or args.ckpt
    tokenizer = AutoTokenizer.from_pretrained(tok_path)

    print(f"Loading model from {args.ckpt}")
    if is_student_checkpoint(args.ckpt):
        model = load_student(args.ckpt, args.device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.ckpt, torch_dtype=torch.bfloat16, trust_remote_code=True,
        ).to(args.device).eval()
    print(f"Model loaded. Evaluating PPL on wikitext2 (seqlen={args.seqlen})...")

    ppl = evaluate_ppl(model, tokenizer, args.seqlen, args.device)
    print(f"\n{'='*50}")
    print(f"Wikitext2 PPL = {ppl:.2f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
