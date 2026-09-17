"""KittyLM: the decoder-only Transformer baseline.

Purpose:
    Map token ids to next-token logits. Tokens are embedded, passed through ``n_layers``
    pre-norm blocks (attention + SwiGLU), normalized once more, and projected back to the
    vocabulary with the *same* matrix used for the embedding (tied weights, D-004/D-008).
    This dense, LLaMA-style model is the baseline every later architecture is compared against.

Public API:
    KittyLM(config)
        ``forward(input_ids, cache=None) -> logits``; ``reset_parameters()``; ``config``;
        ``output_module_names`` (used by parameter accounting).

Shapes:
    ``input_ids`` ``[B, T]`` (int64) -> logits ``[B, T, vocab_size]``. With a cache holding
    ``L`` tokens, positions are ``L .. L + T - 1`` and ``L + T <= context_length``.

Dtype:
    Parameters float32 by default; logits follow the dtype of the final projection (bf16 under
    autocast). Loss computation (Milestone C) is responsible for upcasting.

Device:
    Parameters, RoPE buffers, inputs and cache must share one device. Construction under
    ``torch.device("meta")`` allocates nothing (used for parameter accounting of large configs).

Math:
    ``logits = W_E^T · RMSNorm(block_N(... block_1(W_E[input_ids])))``.

Invariants:
    - Causal: logits at position t depend only on ids at positions <= t.
    - With ``tie_embeddings``, ``lm_head.weight is tok_embeddings.weight`` (one tensor).
    - Initialization: every linear/embedding weight ``~ N(0, init_std)``; residual output
      projections (attention ``o_proj``, SwiGLU ``down_proj``) use
      ``init_std / sqrt(2 * n_layers)`` so the residual stream does not grow with depth;
      norm gains start at 1.

Failure modes:
    - Empty input, a non-2-D ``input_ids``, or ``cache.length + T > context_length`` raises
      ValueError.
    - Attention kernel problems surface as AttentionBackendUnavailable or a device error.

See:
    D-004, D-008, D-014, plan rev 3.3 section 3.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from kittylm.model.block import TransformerBlock
from kittylm.model.config import ModelConfig
from kittylm.model.kv_cache import KVCache
from kittylm.model.normalization import RMSNorm
from kittylm.model.positional import RotaryEmbedding

__all__ = ["KittyLM"]


class KittyLM(nn.Module):
    """Decoder-only Transformer language model."""

    output_module_names: tuple[str, ...] = ("lm_head",)

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.rope = RotaryEmbedding(config.head_dim, config.context_length, config.rope_theta)
        self.layers = nn.ModuleList(TransformerBlock(config, i) for i in range(config.n_layers))
        self.norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Initialize weights (see Invariants in the module docstring)."""
        std = self.config.init_std
        residual_std = std / math.sqrt(2 * self.config.n_layers)
        nn.init.normal_(self.tok_embeddings.weight, mean=0.0, std=std)
        for module in self.modules():
            if isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
            elif isinstance(module, nn.Linear):
                if module is self.lm_head and self.config.tie_embeddings:
                    continue  # shares the embedding tensor initialized above
                scale = residual_std if getattr(module, "is_residual_projection", False) else std
                nn.init.normal_(module.weight, mean=0.0, std=scale)

    def forward(self, input_ids: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        """Return next-token logits ``[B, T, vocab_size]`` for ``input_ids`` ``[B, T]``."""
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be [batch, length], got {tuple(input_ids.shape)}")
        length = input_ids.shape[1]
        if length == 0:
            raise ValueError("input_ids must contain at least one token")
        offset = cache.length if cache is not None else 0
        if offset + length > self.config.context_length:
            raise ValueError(
                f"sequence of {offset + length} tokens exceeds context_length "
                f"{self.config.context_length}"
            )
        cos, sin = self.rope(offset, length)
        h = self.tok_embeddings(input_ids)
        for layer in self.layers:
            h = layer(h, cos=cos, sin=sin, cache=cache)
        logits: torch.Tensor = self.lm_head(self.norm(h))
        if cache is not None:
            cache.advance(length)
        return logits
