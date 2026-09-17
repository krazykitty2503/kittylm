"""Pre-norm Transformer block with registered sequence and channel mixers.

Purpose:
    A block alternates two kinds of mixing, each wrapped in a residual connection with
    normalization first (pre-norm): a *sequence mixer* moves information between positions
    (attention in 0.1) and a *channel mixer* transforms each position (SwiGLU in 0.1). Mixers
    are looked up by name from registries. This is the seam where future experiments plug in a
    Gated DeltaNet sequence mixer or a mixture-of-experts channel mixer without rewriting the
    model (deferred until EXP-001 exists).

Public API:
    SEQ_MIXERS, CHANNEL_MIXERS
        Registries ``name -> factory(config, layer_index) -> nn.Module``; 0.1 registers only
        ``attention`` and ``swiglu``.
    TransformerBlock(config, layer_index)
        ``norm1``, ``seq_mixer``, ``norm2``, ``channel_mixer``.

Shapes:
    ``[B, T, d_model]`` -> ``[B, T, d_model]``; ``cos``/``sin`` ``[T, head_dim]``.

Dtype:
    Parameters float32; activations follow autocast.

Device:
    Any device.

Math:
    ``h = x + seq_mixer(norm1(x))``; ``y = h + channel_mixer(norm2(h))``.

Invariants:
    - Residual stream shape is preserved.
    - Pre-norm: each mixer sees normalized input; the residual path itself is never normalized
      inside the block (the final norm lives in the model).

Failure modes:
    - An unregistered mixer name raises KeyError with the known names.

See:
    D-004, plan rev 3.3 section 3 (mixer seam).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from kittylm.model.attention import CausalSelfAttention
from kittylm.model.config import ModelConfig
from kittylm.model.kv_cache import KVCache
from kittylm.model.mlp import SwiGLU
from kittylm.model.normalization import RMSNorm

__all__ = ["CHANNEL_MIXERS", "SEQ_MIXERS", "TransformerBlock"]

MixerFactory = Callable[[ModelConfig, int], nn.Module]

SEQ_MIXERS: dict[str, MixerFactory] = {
    "attention": lambda config, index: CausalSelfAttention(
        config.d_model, config.n_heads, config.attention_backend, index
    ),
}
CHANNEL_MIXERS: dict[str, MixerFactory] = {
    "swiglu": lambda config, index: SwiGLU(config.d_model, config.ffn_dim),
}


def _lookup(registry: dict[str, MixerFactory], name: str, kind: str) -> MixerFactory:
    if name not in registry:
        raise KeyError(f"unknown {kind} mixer {name!r}; registered: {sorted(registry)}")
    return registry[name]


class TransformerBlock(nn.Module):
    """One pre-norm residual block: sequence mixer then channel mixer."""

    def __init__(self, config: ModelConfig, layer_index: int) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.d_model, config.norm_eps)
        self.seq_mixer = _lookup(SEQ_MIXERS, config.seq_mixer, "sequence")(config, layer_index)
        self.norm2 = RMSNorm(config.d_model, config.norm_eps)
        self.channel_mixer = _lookup(CHANNEL_MIXERS, config.channel_mixer, "channel")(
            config, layer_index
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        """Apply both residual sublayers."""
        x = x + self.seq_mixer(self.norm1(x), cos=cos, sin=sin, cache=cache)
        return x + self.channel_mixer(self.norm2(x))
