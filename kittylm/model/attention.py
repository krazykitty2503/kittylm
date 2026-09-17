"""Causal multi-head self-attention with explicitly selectable kernels.

Purpose:
    Attention lets every position read information from earlier positions. Each position
    produces a query, key and value; the query is compared with every earlier key, the scores
    become weights through a softmax, and the output is the weighted sum of values. Several
    kernels compute the same mathematics with different speed and memory. KittyLM selects the
    kernel explicitly (D-014) and keeps a plain PyTorch ``reference`` implementation as the
    correctness oracle, because a fast kernel is only useful if it is provably equivalent.

Public API:
    attend(q, k, v, backend) -> Tensor
        Causal attention through one explicitly chosen path. The model and BENCH-ATTN-001 both
        call this function, so the benchmark measures exactly what the model runs.
    reference_attention(q, k, v), causal_mask(q_len, k_len, device)
    AttentionBackendUnavailable(backend, reason)
        Raised when the chosen SDPA kernel does not exist for these inputs/hardware.
    CausalSelfAttention(d_model, n_heads, backend, layer_index)
        ``q_proj``/``k_proj``/``v_proj``/``o_proj`` (no bias) + RoPE + optional KV cache.

Shapes:
    ``attend``: ``q`` ``[B, H, Tq, Dh]``, ``k``/``v`` ``[B, H, Tk, Dh]`` with ``Tq <= Tk`` ->
    ``[B, H, Tq, Dh]``. ``CausalSelfAttention``: ``[B, T, D]`` -> ``[B, T, D]``, ``D = H * Dh``.

Dtype:
    Parameters float32; ``q``/``k``/``v`` share one dtype (bf16 under GPU autocast). The
    reference path computes scores and softmax in the input dtype, like SDPA's math kernel.

Device:
    Any device. SDPA kernel availability depends on device, dtype and build (see
    BENCH-ATTN-001); CPU supports ``sdpa_math`` and ``sdpa_flash``.

Math:
    ``softmax(Q K^T / sqrt(Dh) + M) V`` where ``M[i, j] = -inf`` for ``j > i + (Tk - Tq)``
    (bottom-right aligned, so cached decoding sees every earlier token).

Invariants:
    - Causality: output at position t never depends on positions after t.
    - Every path computes the same function; differences are floating-point only.
    - Mask handling: ``Tq == Tk`` uses ``is_causal=True``; ``Tq == 1`` needs no mask; otherwise
      an explicit bottom-right causal mask is passed.

Failure modes:
    - An SDPA kernel that is unavailable for the inputs raises AttentionBackendUnavailable
      (never a silent fallback to another kernel). ``sdpa_flash`` cannot take the explicit mask
      needed for multi-token prefill on top of a non-empty cache.
    - Device-level kernel errors (for example ``CUDA error: invalid argument`` from AOTriton on
      unsupported GPUs) propagate unchanged.
    - ``Tq > Tk`` or an unknown backend raises ValueError.

See:
    D-004, D-008, D-014, experiments/BENCH-ATTN-001.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from kittylm.model.config import ATTENTION_BACKENDS
from kittylm.model.kv_cache import KVCache
from kittylm.model.positional import apply_rotary

__all__ = [
    "AttentionBackendUnavailable",
    "CausalSelfAttention",
    "attend",
    "causal_mask",
    "reference_attention",
]

_SDPA_BACKENDS: dict[str, SDPBackend] = {
    "sdpa_math": SDPBackend.MATH,
    "sdpa_efficient": SDPBackend.EFFICIENT_ATTENTION,
    "sdpa_flash": SDPBackend.FLASH_ATTENTION,
    "sdpa_cudnn": SDPBackend.CUDNN_ATTENTION,
}
_UNAVAILABLE_MARKERS = ("No available kernel", "No viable backend")


class AttentionBackendUnavailable(RuntimeError):
    """The requested attention kernel cannot run for these inputs on this build/hardware."""

    def __init__(self, backend: str, reason: str) -> None:
        self.backend = backend
        self.reason = reason
        super().__init__(f"attention backend {backend!r} unavailable: {reason}")


def causal_mask(q_len: int, k_len: int, device: torch.device) -> torch.Tensor:
    """Boolean mask ``[q_len, k_len]``; True where query i may attend key j (bottom-right)."""
    offset = k_len - q_len
    rows = torch.arange(q_len, device=device).unsqueeze(1)
    cols = torch.arange(k_len, device=device).unsqueeze(0)
    return cols <= rows + offset


def reference_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Explicit causal attention: the correctness oracle for every kernel."""
    q_len, k_len = q.shape[-2], k.shape[-2]
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    mask = causal_mask(q_len, k_len, q.device)
    scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, backend: str) -> torch.Tensor:
    """Causal attention through exactly the requested path (no silent fallback)."""
    q_len, k_len = q.shape[-2], k.shape[-2]
    if q_len > k_len:
        raise ValueError(f"query length {q_len} exceeds key length {k_len}")
    if backend == "reference":
        return reference_attention(q, k, v)
    if backend not in _SDPA_BACKENDS:
        raise ValueError(f"unknown attention backend {backend!r}; known: {ATTENTION_BACKENDS}")

    attn_mask: torch.Tensor | None = None
    is_causal = False
    if q_len == k_len:
        is_causal = True
    elif q_len > 1:
        attn_mask = causal_mask(q_len, k_len, q.device)
    try:
        with sdpa_kernel([_SDPA_BACKENDS[backend]]):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
    except RuntimeError as exc:
        message = str(exc)
        if any(marker in message for marker in _UNAVAILABLE_MARKERS):
            raise AttentionBackendUnavailable(backend, message.splitlines()[0]) from exc
        raise


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE, the ``attention`` sequence mixer."""

    def __init__(self, d_model: int, n_heads: int, backend: str, layer_index: int) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if backend not in ATTENTION_BACKENDS:
            raise ValueError(f"unknown attention backend {backend!r}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.backend = backend
        self.layer_index = layer_index
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj.is_residual_projection = True  # type: ignore[assignment]

    def forward(
        self,
        x: torch.Tensor,
        *,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        """Attend over ``x`` ``[B, T, D]`` (and the cache, if given)."""
        batch, length, d_model = x.shape
        shape = (batch, length, self.n_heads, self.head_dim)
        q = self.q_proj(x).view(shape).transpose(1, 2)
        k = self.k_proj(x).view(shape).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        if cache is not None:
            k, v = cache.update(self.layer_index, k, v)
        out = attend(q, k, v, self.backend)
        return self.o_proj(out.transpose(1, 2).reshape(batch, length, d_model))
