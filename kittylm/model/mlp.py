"""SwiGLU feed-forward block (the channel mixer).

Purpose:
    After attention mixes information *across positions*, the feed-forward block transforms each
    position independently. SwiGLU uses a gated unit: one projection is passed through SiLU and
    multiplies another projection element-wise, then a third projection maps back to
    ``d_model``. The gate lets the network choose which features to pass, which trains better
    than a plain ReLU/GELU MLP at the same parameter count (D-004).

Public API:
    SwiGLU(d_model, ffn_dim)
        Parameters ``gate_proj`` and ``up_proj`` (``d_model -> ffn_dim``) and ``down_proj``
        (``ffn_dim -> d_model``), all without bias; ``down_proj`` is a residual output
        projection (scaled at initialization).

Shapes:
    input ``[B, T, d_model]`` -> hidden ``[B, T, ffn_dim]`` -> output ``[B, T, d_model]``.

Dtype:
    Parameters float32; activations follow autocast (bf16 on GPU during training).

Device:
    Any device.

Math:
    ``SwiGLU(x) = down(SiLU(gate(x)) * up(x))``; parameter count ``3 * d_model * ffn_dim``.

Invariants:
    - Position-wise: output at position t depends only on input at position t.
    - No biases (LLaMA layout, D-008); classified as ``mlp`` in accounting.

Failure modes:
    - A last dimension different from ``d_model`` raises a shape error from PyTorch.

See:
    D-004, D-008.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["SwiGLU"]


class SwiGLU(nn.Module):
    """Gated SiLU feed-forward block without biases."""

    def __init__(self, d_model: int, ffn_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, ffn_dim, bias=False)
        self.up_proj = nn.Linear(d_model, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, d_model, bias=False)
        self.down_proj.is_residual_projection = True  # type: ignore[assignment]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gated feed-forward transform position-wise."""
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
