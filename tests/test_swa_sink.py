import copy
from types import SimpleNamespace

import pytest
import torch
from einops import rearrange, repeat

from distill_model.student_layers import (
    SlidingWindowSinkAttentionStudentV1,
    build_swa_sink_mask,
)


def _config(window_size=64, num_sink_tokens=4):
    return SimpleNamespace(
        hidden_size=128,
        num_heads=4,
        num_kv_heads=2,
        qkv_bias=False,
        qk_norm=False,
        window_size=window_size,
        num_sink_tokens=num_sink_tokens,
        rope_theta=1_000_000.0,
        max_position_embeddings=256,
    )


def _dense_reference(module, hidden_states):
    attn = module.attn
    q = rearrange(attn.q_proj(hidden_states), "... (h d) -> ... h d", d=attn.head_dim)
    k = rearrange(attn.k_proj(hidden_states), "... (h d) -> ... h d", d=attn.head_dim)
    v = rearrange(attn.v_proj(hidden_states), "... (h d) -> ... h d", d=attn.head_dim)
    if attn.qk_norm:
        q, k = attn.q_norm(q), attn.k_norm(k)
    q, k = attn.rotary(
        q,
        k,
        seqlen_offset=0,
        max_seqlen=max(hidden_states.shape[1], attn.max_position_embeddings),
        cu_seqlens=None,
    )
    k = repeat(k, "b t h d -> b t (h g) d", g=attn.num_kv_groups)
    v = repeat(v, "b t h d -> b t (h g) d", g=attn.num_kv_groups)
    scores = torch.einsum("bthd,bshd->bhts", q.float(), k.float()) * (attn.head_dim ** -0.5)
    mask = build_swa_sink_mask(
        hidden_states.shape[1],
        module.window_size,
        module.num_sink_tokens,
        hidden_states.device,
    )
    scores = scores.masked_fill(~mask[None, None], torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores, dim=-1).to(v.dtype)
    output = torch.einsum("bhts,bshd->bthd", probs, v)
    return attn.o_proj(output.reshape(hidden_states.shape[0], hidden_states.shape[1], -1))


def test_mask_keeps_four_sinks_and_sixty_recent_tokens():
    mask = build_swa_sink_mask(seq_len=96, window_size=64, num_sink_tokens=4)
    final = mask[-1]
    assert final[:4].all()
    assert not final[4:36].any()
    assert final[36:].all()
    assert int(final.sum()) == 64


def test_mask_is_full_causal_before_window_fills():
    mask = build_swa_sink_mask(seq_len=32, window_size=64, num_sink_tokens=4)
    expected = torch.ones(32, 32, dtype=torch.bool).tril()
    assert torch.equal(mask, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlashAttention requires CUDA")
@pytest.mark.parametrize("seq_len", [32, 96])
def test_flash_output_matches_dense_reference(seq_len):
    pytest.importorskip("flash_attn")
    torch.manual_seed(7)
    module = SlidingWindowSinkAttentionStudentV1(_config(), layer_idx=0).cuda().bfloat16()
    hidden = torch.randn(2, seq_len, 128, device="cuda", dtype=torch.bfloat16)
    actual, _, _ = module(hidden, use_cache=False)
    expected = _dense_reference(module, hidden)
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlashAttention requires CUDA")
def test_short_sequence_matches_full_attention():
    pytest.importorskip("flash_attn")
    torch.manual_seed(11)
    module = SlidingWindowSinkAttentionStudentV1(_config(), layer_idx=0).cuda().bfloat16()
    hidden = torch.randn(2, 32, 128, device="cuda", dtype=torch.bfloat16)
    actual, _, _ = module(hidden, use_cache=False)
    expected, _, _ = module.attn(hidden, use_cache=False)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlashAttention requires CUDA")
def test_old_nonsink_is_invisible_but_prefix_sink_is_visible():
    pytest.importorskip("flash_attn")
    torch.manual_seed(17)
    module = SlidingWindowSinkAttentionStudentV1(_config(), layer_idx=0).cuda().bfloat16()
    hidden = torch.randn(1, 96, 128, device="cuda", dtype=torch.bfloat16)
    reference, _, _ = module(hidden, use_cache=False)

    changed_old = hidden.clone()
    changed_old[:, 10] += 10
    old_output, _, _ = module(changed_old, use_cache=False)
    torch.testing.assert_close(reference[:, -1], old_output[:, -1], atol=0, rtol=0)

    changed_sink = hidden.clone()
    changed_sink[:, 0] += 10
    sink_output, _, _ = module(changed_sink, use_cache=False)
    assert not torch.allclose(reference[:, -1], sink_output[:, -1], atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlashAttention requires CUDA")
def test_cached_decode_matches_full_forward():
    pytest.importorskip("flash_attn")
    from fla.models.utils import Cache

    torch.manual_seed(23)
    module = SlidingWindowSinkAttentionStudentV1(_config(), layer_idx=0).cuda().bfloat16()
    hidden = torch.randn(1, 96, 128, device="cuda", dtype=torch.bfloat16)
    expected, _, _ = module(hidden, use_cache=False)

    cache = Cache()
    outputs = []
    prefill, _, cache = module(hidden[:, :80], past_key_values=cache, use_cache=True)
    outputs.append(prefill)
    for position in range(80, hidden.shape[1]):
        decoded, _, cache = module(
            hidden[:, position:position + 1],
            past_key_values=cache,
            use_cache=True,
        )
        outputs.append(decoded)
    actual = torch.cat(outputs, dim=1)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)


def test_rejects_invalid_sink_count():
    with pytest.raises(ValueError):
        build_swa_sink_mask(seq_len=8, window_size=4, num_sink_tokens=4)


def test_init_copies_qwen_qk_norm_weights():
    config = _config()
    config.qk_norm = True
    module = SlidingWindowSinkAttentionStudentV1(config, layer_idx=0)
    teacher = copy.deepcopy(module.attn)
    teacher.q_norm.weight.data.copy_(torch.linspace(0.5, 2.0, teacher.q_norm.weight.numel()))
    teacher.k_norm.weight.data.copy_(torch.linspace(1.0, 34.0, teacher.k_norm.weight.numel()))
    module.init_from_teacher(teacher)
    torch.testing.assert_close(module.attn.q_norm.weight, teacher.q_norm.weight)
    torch.testing.assert_close(module.attn.k_norm.weight, teacher.k_norm.weight)
