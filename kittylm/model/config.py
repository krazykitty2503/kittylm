"""Model configuration (``kind: model``).

Purpose:
    Describe a decoder-only Transformer completely and strictly: every shape, the attention
    backend and the mixer types. The configuration is validated when it is built, so an
    impossible model (for example ``d_model`` not divisible by ``n_heads``) fails before any
    tensor is allocated.

Public API:
    ATTENTION_BACKENDS
        Every attention path the model can use: ``reference`` (explicit PyTorch operations,
        the correctness oracle) and the four PyTorch SDPA kernels restricted one at a time.
    ModelConfig
        Frozen dataclass registered as ``kind: model``; ``head_dim`` is derived.

Invariants:
    - ``d_model == n_heads * head_dim`` and ``head_dim`` is even (RoPE rotates pairs).
    - Only mixers registered in 0.1 are accepted: ``attention`` and ``swiglu`` (D-004).
    - The attention backend is an explicit configuration value; the default for real runs is
      decided by BENCH-ATTN-001 (D-014), never assumed.

Failure modes:
    - Invalid shapes or values raise ConfigError when the dataclass is constructed.

See:
    D-004 (LLaMA-style baseline), D-008 (RoPE layout), D-014 (attention kernel), plan rev 3.3
    section 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from kittylm.config import ConfigError, register_config_kind

__all__ = ["ATTENTION_BACKENDS", "AttentionBackend", "ModelConfig"]

AttentionBackend = Literal["reference", "sdpa_math", "sdpa_efficient", "sdpa_flash", "sdpa_cudnn"]
ATTENTION_BACKENDS: tuple[str, ...] = (
    "reference",
    "sdpa_math",
    "sdpa_efficient",
    "sdpa_flash",
    "sdpa_cudnn",
)


@dataclass(frozen=True)
class ModelConfig:
    """``kind: model`` configuration of the decoder-only Transformer."""

    name: str
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    ffn_dim: int
    context_length: int
    attention_backend: AttentionBackend
    seq_mixer: Literal["attention"] = "attention"
    channel_mixer: Literal["swiglu"] = "swiglu"
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    init_std: float = 0.02
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        """Validate shapes and numeric ranges."""
        for field_name in ("vocab_size", "d_model", "n_layers", "n_heads", "ffn_dim"):
            if getattr(self, field_name) < 1:
                raise ConfigError(f"{field_name} must be >= 1")
        if self.context_length < 1:
            raise ConfigError("context_length must be >= 1")
        if self.d_model % self.n_heads:
            raise ConfigError("d_model must be divisible by n_heads")
        if (self.d_model // self.n_heads) % 2:
            raise ConfigError("head_dim (d_model / n_heads) must be even for RoPE")
        if self.rope_theta <= 0 or self.norm_eps <= 0 or self.init_std <= 0:
            raise ConfigError("rope_theta, norm_eps and init_std must be > 0")

    @property
    def head_dim(self) -> int:
        """Per-head dimension ``d_model // n_heads``."""
        return self.d_model // self.n_heads


register_config_kind("model", ModelConfig)
