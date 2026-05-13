
# scripts for converting pretrained hf model weights to fla style

import argparse
import warnings
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import fla  # noqa
from fla.models.transformer.configuration_transformer import TransformerConfig

config_dict = {
    "3b": {
        "attention_bias": False,
        "bos_token_id": 128000,
        "eos_token_id": [
            128001,
            128008,
            128009
        ],
        "fuse_cross_entropy": False,
        "fuse_norm": True,
        "hidden_act": "swish",
        "hidden_ratio": 4,
        "hidden_size": 3072,
        "initializer_range": 0.02,
        "intermediate_size": 8192,
        "model_type": "transformer",
        "norm_eps": 1e-05,
        "num_heads": 24,
        "num_hidden_layers": 28,
        "num_kv_heads": 8,
        "rope_theta": 500000.0,
        "tie_word_embeddings": True,
        "use_cache": True,
        "vocab_size": 128256,
        "max_position_embeddings": 131072,
        "qkv_bias": False,
    }
}

def sizeof_fmt(num, suffix='B'):
    for unit in ('', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi'):
        if abs(num) < 1024.0:
            return f'{num:.2f}{unit}{suffix}'
        num /= 1024.0
    return f'{num:.2f}Yi{suffix}'



def convert(
    llama: str,
    config: str,
    output: str,
    precision: str = 'float32'
):
    AutoTokenizer.from_pretrained(llama).save_pretrained(output)
    llama = AutoModelForCausalLM.from_pretrained(llama, torch_dtype=precision)
    print(f"Loading Llama ...\n{llama}")

    # Check if config is a directory with config.json, otherwise use config_dict
    if os.path.isdir(config) and os.path.isfile(os.path.join(config, "config.json")):
        config_obj = AutoConfig.from_pretrained(config)
    elif config in config_dict:
        config_obj = TransformerConfig(**config_dict[config])
    else:
        raise ValueError(f"Config '{config}' is neither a known key nor a directory with config.json")

    config_obj.torch_dtype = precision
    model = AutoModelForCausalLM.from_config(config_obj)
    if precision in ['float16', 'fp16']:
        model = model.to(torch.float16)
    elif precision in ['bfloat16', 'bf16']:
        model = model.to(torch.bfloat16)
    num_parameters = model.num_parameters()
    print(f"Initializing the model from the config:\n{config_obj}\n{model}")
    print(f"Number of parameters in total: {num_parameters} ({sizeof_fmt(num_parameters)})")

    print("Copying the weights from Llama to the model ...")
    vocab_size = llama.model.embed_tokens.weight.shape[0]
    if model.model.embeddings.weight.shape[0] != vocab_size:
        warnings.warn(f"Llama and the model have different embedding sizes "
                      f"({vocab_size} vs {model.model.embeddings.weight.shape[0]}), "
                      f"the model embeddings will be extended with randomly initialized values or truncated")
        vocab_size = min(model.model.embeddings.weight.shape[0], vocab_size)
    print("llama.model.embed_tokens                        -> model.model.embeddings")
    model.model.embeddings.weight.data[:vocab_size].copy_(llama.model.embed_tokens.weight[:vocab_size])
    torch.testing.assert_close(model.model.embeddings.weight[:vocab_size], llama.model.embed_tokens.weight[:vocab_size])
    for i in range(config_obj.num_hidden_layers):
        if hasattr(model.model.layers[i], 'attn_norm'):
            if model.model.layers[i].attn_norm.weight is not None:
                print(f"llama.model.layers{i}.input_layernorm.weight -> model.model.layers{i}.attn_norm.weight")
                model.model.layers[i].attn_norm.weight.data.copy_(llama.model.layers[i].input_layernorm.weight)
                torch.testing.assert_close(model.model.layers[i].attn_norm.weight,
                                           llama.model.layers[i].input_layernorm.weight)
            if model.model.layers[i].attn_norm.bias is not None:
                print(f"llama.model.layers{i}.input_layernorm.bias -> model.model.layers{i}.attn_norm.bias")
                model.model.layers[i].attn_norm.bias.data.copy_(llama.model.layers[i].input_layernorm.bias)
                torch.testing.assert_close(model.model.layers[i].attn_norm.bias,
                                           llama.model.layers[i].input_layernorm.bias)
            model.model.layers[i].attn_norm.eps = llama.model.layers[i].input_layernorm.variance_epsilon
        if hasattr(model.model.layers[i].attn, 'norm'):
            if model.model.layers[i].attn.norm.weight is not None:
                print(f"llama.model.layers{i}.input_layernorm.weight -> model.model.layers{i}.attn.norm.weight")
                model.model.layers[i].attn.norm.weight.data.copy_(llama.model.layers[i].input_layernorm.weight)
                torch.testing.assert_close(model.model.layers[i].attn.norm.weight,
                                           llama.model.layers[i].input_layernorm.weight)
            if model.model.layers[i].attn.norm.bias is not None:
                print(f"llama.model.layers{i}.input_layernorm.bias -> model.model.layers{i}.attn.norm.bias")
                model.model.layers[i].attn.norm.bias.data.copy_(llama.model.layers[i].input_layernorm.bias)
                torch.testing.assert_close(model.model.layers[i].attn.norm.bias,
                                           llama.model.layers[i].input_layernorm.bias)
            model.model.layers[i].attn.norm.eps = llama.model.layers[i].input_layernorm.variance_epsilon

        print(f"llama.model.layers{i}.attn.q_proj.weight  -> model.model.layers{i}.attn.q_proj.weight")
        model.model.layers[i].attn.q_proj.weight.data.copy_(llama.model.layers[i].self_attn.q_proj.weight)
        torch.testing.assert_close(model.model.layers[i].attn.q_proj.weight, llama.model.layers[i].self_attn.q_proj.weight)
        if llama.model.layers[i].self_attn.q_proj.bias is not None and model.model.layers[i].attn.q_proj.bias is not None:
            print(f"llama.model.layers{i}.attn.q_proj.bias  -> model.model.layers{i}.attn.q_proj.bias")
            model.model.layers[i].attn.q_proj.bias.data.copy_(llama.model.layers[i].self_attn.q_proj.bias)
            torch.testing.assert_close(model.model.layers[i].attn.q_proj.bias, llama.model.layers[i].self_attn.q_proj.bias)
        print(f"llama.model.layers.{i}.attn.k_proj.weight -> model.model.layers.{i}.attn.k_proj.weight")
        model.model.layers[i].attn.k_proj.weight.data.copy_(llama.model.layers[i].self_attn.k_proj.weight)
        torch.testing.assert_close(model.model.layers[i].attn.k_proj.weight, llama.model.layers[i].self_attn.k_proj.weight)
        if llama.model.layers[i].self_attn.k_proj.bias is not None and model.model.layers[i].attn.k_proj.bias is not None:
            print(f"llama.model.layers{i}.attn.k_proj.bias  -> model.model.layers{i}.attn.k_proj.bias")
            model.model.layers[i].attn.k_proj.bias.data.copy_(llama.model.layers[i].self_attn.k_proj.bias)
            torch.testing.assert_close(model.model.layers[i].attn.k_proj.bias, llama.model.layers[i].self_attn.k_proj.bias)
        print(f"llama.model.layers.{i}.attn.v_proj.weight -> model.model.layers.{i}.attn.v_proj.weight")
        model.model.layers[i].attn.v_proj.weight.data.copy_(llama.model.layers[i].self_attn.v_proj.weight)
        torch.testing.assert_close(model.model.layers[i].attn.v_proj.weight, llama.model.layers[i].self_attn.v_proj.weight)
        if llama.model.layers[i].self_attn.v_proj.bias is not None and model.model.layers[i].attn.v_proj.bias is not None:
            print(f"llama.model.layers{i}.attn.v_proj.bias  -> model.model.layers{i}.attn.v_proj.bias")
            model.model.layers[i].attn.v_proj.bias.data.copy_(llama.model.layers[i].self_attn.v_proj.bias)
            torch.testing.assert_close(model.model.layers[i].attn.v_proj.bias, llama.model.layers[i].self_attn.v_proj.bias)

        print(f"llama.model.layers.{i}.attn.o_proj.weight -> model.model.layers.{i}.attn.o_proj.weight")
        model.model.layers[i].attn.o_proj.weight.data.copy_(llama.model.layers[i].self_attn.o_proj.weight)
        torch.testing.assert_close(model.model.layers[i].attn.o_proj.weight, llama.model.layers[i].self_attn.o_proj.weight)

        if hasattr(model.model.layers[i], 'mlp_norm'):
            if model.model.layers[i].mlp_norm.weight is not None:
                print(f"llama.model.layers{i}.post_attention_layernorm.weight -> model.model.layers{i}.mlp_norm.weight")
                model.model.layers[i].mlp_norm.weight.data.copy_(llama.model.layers[i].post_attention_layernorm.weight)
                torch.testing.assert_close(model.model.layers[i].mlp_norm.weight,
                                           llama.model.layers[i].post_attention_layernorm.weight)
            if model.model.layers[i].mlp_norm.bias is not None:
                print(f"llama.model.layers{i}.post_attention_layernorm.bias -> model.model.layers{i}.mlp_norm.bias")
                model.model.layers[i].mlp_norm.bias.data.copy_(llama.model.layers[i].post_attention_layernorm.bias)
                torch.testing.assert_close(model.model.layers[i].mlp_norm.bias,
                                           llama.model.layers[i].post_attention_layernorm.bias)
            model.model.layers[i].mlp_norm.eps = llama.model.layers[i].post_attention_layernorm.variance_epsilon
        if hasattr(model.model.layers[i].mlp, 'norm'):
            if model.model.layers[i].mlp.norm.weight is not None:
                print(f"llama.model.layers{i}.post_attention_layernorm.weight -> model.model.layers{i}.mlp.norm.weight")
                model.model.layers[i].mlp.norm.weight.data.copy_(llama.model.layers[i].post_attention_layernorm.weight)
                torch.testing.assert_close(model.model.layers[i].mlp.norm.weight,
                                           llama.model.layers[i].post_attention_layernorm.weight)
            if model.model.layers[i].mlp.norm.bias is not None:
                print(f"llama.model.layers{i}.post_attention_layernorm.bias -> model.model.layers{i}.mlp.norm.bias")
                model.model.layers[i].mlp.norm.bias.data.copy_(llama.model.layers[i].post_attention_layernorm.bias)
                torch.testing.assert_close(model.model.layers[i].mlp.norm.bias,
                                           llama.model.layers[i].post_attention_layernorm.bias)
            model.model.layers[i].mlp.norm.eps = llama.model.layers[i].post_attention_layernorm.variance_epsilon

        print(f"llama.model.layers.{i}.mlp.gate_proj.weight -> model.model.layers.{i}.mlp.gate_proj.weight")
        model.model.layers[i].mlp.gate_proj.weight.data.copy_(llama.model.layers[i].mlp.gate_proj.weight)
        torch.testing.assert_close(model.model.layers[i].mlp.gate_proj.weight, llama.model.layers[i].mlp.gate_proj.weight)
        print(f"llama.model.layers.{i}.mlp.up_proj.weight -> model.model.layers.{i}.mlp.up_proj.weight")
        model.model.layers[i].mlp.up_proj.weight.data.copy_(llama.model.layers[i].mlp.up_proj.weight)
        torch.testing.assert_close(model.model.layers[i].mlp.up_proj.weight, llama.model.layers[i].mlp.up_proj.weight)

        print(f"llama.model.layers.{i}.mlp.down_proj.weight -> model.model.layers.{i}.mlp.down_proj.weight")
        model.model.layers[i].mlp.down_proj.weight.data.copy_(llama.model.layers[i].mlp.down_proj.weight)
        torch.testing.assert_close(model.model.layers[i].mlp.down_proj.weight,
                                   llama.model.layers[i].mlp.down_proj.weight)

    if model.model.norm.weight is not None:
        print("llama.model.norm.weight -> model.model.norm.weight")
        model.model.norm.weight.data.copy_(llama.model.norm.weight)
        torch.testing.assert_close(model.model.norm.weight, llama.model.norm.weight)
    if model.model.norm.bias is not None:
        print("llama.model.norm.bias -> model.model.norm.bias")
        model.model.norm.bias.data.copy_(llama.model.norm.bias)
        torch.testing.assert_close(model.model.norm.bias, llama.model.norm.bias)
    model.model.norm.eps = llama.model.norm.variance_epsilon

    if not model.config.tie_word_embeddings:
        print("llama.model.lm_head.weight -> model.lm_head.weight")
        model.lm_head.weight.data[:vocab_size].copy_(llama.lm_head.weight[:vocab_size])
        torch.testing.assert_close(model.lm_head.weight[:vocab_size], llama.lm_head.weight[:vocab_size])
    model.config.rope_theta = llama.config.rope_theta

    print(f"Saving converted model to {output} ...\n{model}")
    model.save_pretrained(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='meta-llama/Llama-3.2-3B-Instruct')
    parser.add_argument("--config", default='3b')
    parser.add_argument("--output", default='converted/Llama-3.2-3B-Instruct')
    parser.add_argument('--precision', type=str, default='bfloat16')
    args = parser.parse_args()
    convert(args.model, args.config, args.output, precision=args.precision)
