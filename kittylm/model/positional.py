"""Rotary position embeddings (RoPE) in the HF-LLaMA ``rotate_half`` layout.

Purpose:
    Give attention a sense of order without learned position parameters. Each query and key
    vector is rotated by an angle proportional to its position, with a different frequency per
    dimension pair. The dot product of a rotated query and key then depends on their *relative*
    distance. The ``rotate_half`` layout (first half / second half of the head dimension)
    matches HF-LLaMA, so weights map onto that format without re-deriving (D-008).

Public API:
    RotaryEmbedding(head_dim, max_positions, theta=10000.0)
        ``forward(offset, length)`` returns ``(cos, sin)`` for positions
        ``offset .. offset + length - 1``.
    rotate_half(x), apply_rotary(x, cos, sin)

Shapes:
    ``cos``/``sin``: ``[length, head_dim]``. ``apply_rotary``: ``x`` ``[B, H, T, head_dim]`` with
    ``cos``/``sin`` ``[T, head_dim]`` -> ``[B, H, T, head_dim]``.

Dtype:
    Tables are computed in float64 and stored as float32; ``apply_rotary`` casts them to the
    dtype of ``x``.

Device:
    The tables are non-persistent buffers: they move with the module (``.to(device)``) and are
    not stored in checkpoints, because they are fully determined by the configuration.

Math:
    ``inv_freq_i = theta^(-2i / head_dim)``, angle ``= position * inv_freq_i``,
    ``rope(x) = x * cos(angle) + rotate_half(x) * sin(angle)`` with
    ``rotate_half([x1, x2]) = [-x2, x1]``.

Invariants:
    - Rotation preserves vector norms.
    - Position 0 is the identity rotation.
    - No parameters: RoPE contributes 0 to the ``positional`` parameter count.

Failure modes:
    - ``offset + length > max_positions`` raises ValueError.
    - An odd ``head_dim`` raises ValueError.

See:
    D-004, D-008.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["RotaryEmbedding", "apply_rotary", "rotate_half"]


class RotaryEmbedding(nn.Module):
    """Precomputed RoPE cos/sin tables for ``max_positions`` positions."""

    cos_table: torch.Tensor
    sin_table: torch.Tensor

    def __init__(self, head_dim: int, max_positions: int, theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE requires an even head_dim")
        self.head_dim = head_dim
        self.max_positions = max_positions
        inv_freq = 1.0 / theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim)
        positions = torch.arange(max_positions, dtype=torch.float64)
        angles = torch.outer(positions, inv_freq)
        emb = torch.cat((angles, angles), dim=-1)
        self.register_buffer("cos_table", emb.cos().float(), persistent=False)
        self.register_buffer("sin_table", emb.sin().float(), persistent=False)

    def forward(self, offset: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(cos, sin)`` for positions ``offset .. offset + length - 1``."""
        end = offset + length
        if offset < 0 or end > self.max_positions:
            raise ValueError(
                f"positions {offset}..{end - 1} exceed RoPE table of {self.max_positions}"
            )
        return self.cos_table[offset:end], self.sin_table[offset:end]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Map ``[x1, x2]`` (halves of the last dimension) to ``[-x2, x1]``."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` ``[B, H, T, head_dim]`` by ``cos``/``sin`` ``[T, head_dim]``."""
    cos = cos.to(dtype=x.dtype)
    sin = sin.to(dtype=x.dtype)
    return x * cos + rotate_half(x) * sin
