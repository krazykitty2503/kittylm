"""Milestone B model tests (CPU; GPU-only checks live in test_model_gpu.py)."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch
from torch import nn

from kittylm.config import ConfigError, from_dict
from kittylm.model.accounting import (
    AccountingError,
    count_parameters,
    render_parameter_tree,
)
from kittylm.model.attention import (
    AttentionBackendUnavailable,
    attend,
    causal_mask,
    reference_attention,
)
from kittylm.model.block import SEQ_MIXERS, _lookup
from kittylm.model.config import ATTENTION_BACKENDS, ModelConfig
from kittylm.model.kv_cache import KVCache
from kittylm.model.normalization import RMSNorm
from kittylm.model.positional import RotaryEmbedding, apply_rotary, rotate_half
from kittylm.model.transformer import KittyLM

SMALL = ModelConfig(
    name="test-small",
    vocab_size=97,
    d_model=32,
    n_layers=2,
    n_heads=4,
    ffn_dim=64,
    context_length=48,
    attention_backend="reference",
)
TINY = ModelConfig(
    name="tiny",
    vocab_size=16384,
    d_model=384,
    n_layers=6,
    n_heads=6,
    ffn_dim=1024,
    context_length=1024,
    attention_backend="reference",
)
CPU_REQUIRED_BACKENDS = ("reference", "sdpa_math")


def build(config: ModelConfig = SMALL, seed: int = 0) -> KittyLM:
    torch.manual_seed(seed)
    return KittyLM(config).eval()


def ids(batch: int, length: int, vocab: int = SMALL.vocab_size, seed: int = 1) -> torch.Tensor:
    return torch.randint(0, vocab, (batch, length), generator=torch.Generator().manual_seed(seed))


# --- configuration --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"d_model": 30}, "divisible by n_heads"),
        ({"d_model": 12, "n_heads": 4}, "must be even"),
        ({"n_layers": 0}, "n_layers must be >= 1"),
        ({"context_length": 0}, "context_length"),
        ({"init_std": 0.0}, "must be > 0"),
    ],
)
def test_config_validation(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        replace(SMALL, **changes)  # type: ignore[arg-type]


def test_config_rejects_unknown_backend_and_mixers() -> None:
    base = {
        "name": "x",
        "vocab_size": 8,
        "d_model": 8,
        "n_layers": 1,
        "n_heads": 2,
        "ffn_dim": 8,
        "context_length": 4,
        "attention_backend": "reference",
    }
    assert from_dict(ModelConfig, base).head_dim == 4
    for key, value in (
        ("attention_backend", "sdpa_magic"),
        ("seq_mixer", "gdn"),
        ("channel_mixer", "moe"),
    ):
        with pytest.raises(ConfigError, match="expected one of"):
            from_dict(ModelConfig, {**base, key: value})
    with pytest.raises(KeyError, match="registered"):
        _lookup(SEQ_MIXERS, "gdn", "sequence")


# --- building blocks ------------------------------------------------------------------------------


def test_rmsnorm_scale_invariance_and_shape() -> None:
    norm = RMSNorm(16)
    x = torch.randn(2, 3, 16, generator=torch.Generator().manual_seed(0))
    y = norm(x)
    assert y.shape == x.shape
    assert torch.allclose(norm(7.5 * x), y, atol=1e-5)
    assert torch.allclose(y.pow(2).mean(-1), torch.ones(2, 3), atol=1e-4)


def test_rope_properties() -> None:
    rope = RotaryEmbedding(head_dim=8, max_positions=16)
    x = torch.randn(1, 2, 16, 8, generator=torch.Generator().manual_seed(0))
    cos, sin = rope(0, 16)
    y = apply_rotary(x, cos, sin)
    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-5)  # rotation preserves norm
    assert torch.allclose(y[:, :, 0], x[:, :, 0])  # position 0 is the identity
    cos5, sin5 = rope(5, 3)
    assert torch.equal(cos5, cos[5:8]) and torch.equal(sin5, sin[5:8])
    assert torch.equal(
        rotate_half(torch.tensor([1.0, 2.0, 3.0, 4.0])), torch.tensor([-3.0, -4.0, 1.0, 2.0])
    )
    with pytest.raises(ValueError, match="exceed"):
        rope(10, 7)
    with pytest.raises(ValueError, match="even head_dim"):
        RotaryEmbedding(head_dim=7, max_positions=4)


def test_rope_relative_position_property() -> None:
    rope = RotaryEmbedding(head_dim=16, max_positions=64)
    gen = torch.Generator().manual_seed(3)
    q = torch.randn(16, generator=gen)
    k = torch.randn(16, generator=gen)
    cos, sin = rope(0, 64)

    def dot(i: int, j: int) -> float:
        qi = apply_rotary(q.view(1, 1, 1, 16), cos[i : i + 1], sin[i : i + 1])
        kj = apply_rotary(k.view(1, 1, 1, 16), cos[j : j + 1], sin[j : j + 1])
        return float((qi * kj).sum())

    assert math.isclose(dot(10, 7), dot(40, 37), rel_tol=1e-4, abs_tol=1e-5)


def test_causal_mask_is_bottom_right_aligned() -> None:
    assert torch.equal(
        causal_mask(2, 4, torch.device("cpu")), torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]]).bool()
    )
    assert torch.equal(causal_mask(3, 3, torch.device("cpu")), torch.ones(3, 3).tril().bool())


# --- attention paths ------------------------------------------------------------------------------


def qkv(q_len: int, k_len: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return (
        torch.randn(2, 4, q_len, 8, generator=gen),
        torch.randn(2, 4, k_len, 8, generator=gen),
        torch.randn(2, 4, k_len, 8, generator=gen),
    )


@pytest.mark.parametrize("backend", ATTENTION_BACKENDS)
@pytest.mark.parametrize(("q_len", "k_len"), [(12, 12), (1, 12), (5, 12)])
def test_each_path_matches_reference_or_is_explicitly_unavailable(
    backend: str, q_len: int, k_len: int
) -> None:
    q, k, v = qkv(q_len, k_len)
    expected = reference_attention(q, k, v)
    try:
        got = attend(q, k, v, backend)
    except AttentionBackendUnavailable as exc:
        assert backend not in CPU_REQUIRED_BACKENDS, f"{backend} must be available on CPU"
        assert exc.backend == backend
        return
    assert got.shape == expected.shape
    assert torch.allclose(got, expected, atol=1e-5)


def test_attend_rejects_bad_inputs() -> None:
    q, k, v = qkv(6, 4)
    with pytest.raises(ValueError, match="exceeds key length"):
        attend(q, k, v, "reference")
    q, k, v = qkv(4, 4)
    with pytest.raises(ValueError, match="unknown attention backend"):
        attend(q, k, v, "sdpa_magic")


def test_reference_rows_are_causal_probability_distributions() -> None:
    q, k, _ = qkv(6, 6)
    eye = torch.eye(6).expand(2, 4, 6, 6)
    weights = reference_attention(q, k, eye)  # V = identity exposes the attention weights
    assert torch.allclose(weights.sum(-1), torch.ones(2, 4, 6), atol=1e-6)
    assert torch.all(weights.triu(diagonal=1) == 0)


# --- full model -----------------------------------------------------------------------------------


def test_forward_shapes_and_errors() -> None:
    model = build()
    logits = model(ids(3, 10))
    assert logits.shape == (3, 10, SMALL.vocab_size)
    with pytest.raises(ValueError, match="exceeds context_length"):
        model(ids(1, SMALL.context_length + 1))
    with pytest.raises(ValueError, match="at least one token"):
        model(torch.zeros(1, 0, dtype=torch.long))
    with pytest.raises(ValueError, match=r"\[batch, length\]"):
        model(torch.zeros(4, dtype=torch.long))


@pytest.mark.parametrize("backend", CPU_REQUIRED_BACKENDS)
def test_causality(backend: str) -> None:
    model = build(replace(SMALL, attention_backend=backend))
    base = ids(2, 20)
    changed = base.clone()
    changed[:, 12:] = (changed[:, 12:] + 7) % SMALL.vocab_size
    with torch.no_grad():
        a = model(base)
        b = model(changed)
    assert torch.allclose(a[:, :12], b[:, :12], atol=1e-6)
    assert not torch.allclose(a[:, 12:], b[:, 12:])


def test_backends_produce_identical_models() -> None:
    reference = build(SMALL)
    math_model = build(replace(SMALL, attention_backend="sdpa_math"))
    x = ids(2, 16)
    with torch.no_grad():
        assert torch.allclose(reference(x), math_model(x), atol=1e-5)


def test_tied_and_untied_embeddings() -> None:
    tied = build()
    assert tied.lm_head.weight is tied.tok_embeddings.weight
    untied = build(replace(SMALL, tie_embeddings=False))
    assert untied.lm_head.weight is not untied.tok_embeddings.weight


def test_initialization_scales_residual_projections() -> None:
    config = replace(SMALL, d_model=256, n_heads=4, ffn_dim=512, n_layers=8, vocab_size=512)
    model = build(config)
    residual_std = config.init_std / math.sqrt(2 * config.n_layers)
    o_proj = model.layers[0].seq_mixer.o_proj.weight
    q_proj = model.layers[0].seq_mixer.q_proj.weight
    down = model.layers[0].channel_mixer.down_proj.weight
    assert math.isclose(float(o_proj.detach().std()), residual_std, rel_tol=0.05)
    assert math.isclose(float(down.detach().std()), residual_std, rel_tol=0.05)
    assert math.isclose(float(q_proj.detach().std()), config.init_std, rel_tol=0.05)
    assert torch.equal(model.norm.weight, torch.ones(config.d_model))


def test_bf16_autocast_forward_on_cpu() -> None:
    model = build()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        logits = model(ids(1, 8))
    assert logits.dtype == torch.bfloat16
    assert torch.isfinite(logits).all()


# --- KV cache -------------------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ("reference", "sdpa_math", "sdpa_flash"))
def test_kv_cache_matches_full_forward(backend: str) -> None:
    model = build(replace(SMALL, attention_backend=backend))
    x = ids(2, 24)
    try:
        with torch.no_grad():
            full = model(x)
            cache = KVCache.for_config(
                model.config, batch_size=2, dtype=torch.float32, device="cpu"
            )
            steps = [model(x[:, :10], cache=cache)]
            for t in range(10, 24):
                steps.append(model(x[:, t : t + 1], cache=cache))
    except AttentionBackendUnavailable:
        assert backend not in CPU_REQUIRED_BACKENDS
        return
    incremental = torch.cat(steps, dim=1)
    assert cache.length == 24
    assert torch.allclose(incremental, full, atol=1e-5)


@pytest.mark.parametrize("backend", CPU_REQUIRED_BACKENDS)
def test_kv_cache_chunked_prefill(backend: str) -> None:
    model = build(replace(SMALL, attention_backend=backend))
    x = ids(1, 30)
    with torch.no_grad():
        full = model(x)
        cache = KVCache.for_config(model.config, 1, torch.float32, "cpu")
        chunks = [model(x[:, s : s + 7], cache=cache) for s in range(0, 30, 7)]
    assert torch.allclose(torch.cat(chunks, dim=1), full, atol=1e-5)


def test_kv_cache_guards() -> None:
    model = build()
    cache = KVCache.for_config(model.config, 1, torch.float32, "cpu", max_length=8)
    with torch.no_grad():
        model(ids(1, 6), cache=cache)
        with pytest.raises(ValueError, match="exceeds context_length|overflow"):
            model(ids(1, 3), cache=cache)
    bf16_cache = KVCache.for_config(model.config, 1, torch.bfloat16, "cpu")
    with torch.no_grad(), pytest.raises(TypeError, match="dtype"):
        model(ids(1, 4), cache=bf16_cache)
    grad_cache = KVCache.for_config(model.config, 1, torch.float32, "cpu")
    with pytest.raises(RuntimeError, match="inference-only"):
        model(ids(1, 4), cache=grad_cache)
    cache.reset()
    assert cache.length == 0


# --- parameter accounting -------------------------------------------------------------------------


def test_tiny_parameter_budget_is_exact() -> None:
    with torch.device("meta"):
        model = KittyLM(TINY)
    counts = count_parameters(model)
    assert counts.total == 16_913_280
    assert counts.embedding == 16_384 * 384 == 6_291_456
    assert counts.attention == 6 * 4 * 384 * 384 == 3_538_944
    assert counts.mlp == 6 * 3 * 384 * 1024 == 7_077_888
    assert counts.normalization == (2 * 6 + 1) * 384 == 4_992
    assert counts.positional == 0
    assert counts.output == 0
    assert counts.tied_parameters == 6_291_456
    assert counts.non_embedding == 10_621_824
    assert counts.trainable == counts.total
    parts = (
        counts.embedding
        + counts.attention
        + counts.mlp
        + counts.normalization
        + counts.positional
        + counts.output
    )
    assert parts == counts.total
    tree = render_parameter_tree(model)
    assert "16,913,280" in tree and "10,621,824" in tree


def test_untied_budget_counts_output_separately() -> None:
    config = replace(TINY, tie_embeddings=False)
    with torch.device("meta"):
        counts = count_parameters(KittyLM(config))
    assert counts.output == 16_384 * 384
    assert counts.tied_parameters == 0
    assert counts.total == 16_913_280 + 6_291_456


def test_accounting_matches_torch_numel_and_frozen_params() -> None:
    model = build()
    counts = count_parameters(model)
    assert counts.total == sum(p.numel() for p in model.parameters())  # parameters() dedups ties
    model.norm.weight.requires_grad_(False)
    assert count_parameters(model).trainable == counts.total - SMALL.d_model


def test_unclassified_parameters_are_rejected() -> None:
    model = build()
    model.extra = nn.Linear(2, 2)  # type: ignore[assignment]
    with pytest.raises(AccountingError, match="extra.weight"):
        count_parameters(model)
