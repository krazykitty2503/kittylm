"""RMSNorm: root-mean-square layer normalization.

Purpose:
    Keep activations at a stable scale before each sublayer (pre-norm, D-004). RMSNorm divides
    by the root mean square of the features and applies a learned per-feature gain. Unlike
    LayerNorm it neither subtracts the mean nor has a bias, which is cheaper and works as well
    in modern decoder-only models.

Public API:
    RMSNorm(dim, eps=1e-5)
        ``weight`` has shape ``[dim]`` and is initialized to ones.

Shapes:
    input ``[..., dim]`` -> output ``[..., dim]``.

Dtype:
    The normalization is computed in float32 for numerical stability and cast back to the
    input dtype before the gain is applied (under bf16 autocast the result follows PyTorch's
    promotion rules with the float32 gain).

Device:
    Any device; ``weight`` lives with the module.

Math:
    ``y = weight * x / sqrt(mean(x^2, last dim) + eps)``

Invariants:
    - Scale invariance: ``RMSNorm(c * x) == RMSNorm(x)`` for ``c > 0`` (up to eps).
    - One parameter vector of size ``dim``; classified as ``normalization`` in accounting.

Failure modes:
    - A last dimension different from ``dim`` raises a shape error from PyTorch.

See:
    D-004 (LLaMA-style baseline).
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["RMSNorm"]


class RMSNorm(nn.Module):
    """Root-mean-square normalization with a learned gain."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize ``x`` over its last dimension."""
        input_dtype = x.dtype
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * h.to(input_dtype)
