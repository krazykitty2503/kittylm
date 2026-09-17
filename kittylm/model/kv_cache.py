"""Preallocated key/value cache for fast autoregressive decoding.

Purpose:
    Generating token t with a plain forward pass recomputes keys and values for all previous
    tokens, making generation quadratic. The KV cache stores each layer's keys and values once
    they are computed, so each new token only projects itself and attends over the stored
    history. Buffers are preallocated to ``max_length`` so decoding never reallocates memory.

Public API:
    KVCache(n_layers=, batch_size=, n_heads=, head_dim=, max_length=, dtype=, device=)
        ``update(layer, k, v)`` writes at the current length and returns the key/value views up
        to the end of the new tokens; ``advance(count)`` moves the length after all layers have
        been updated; ``reset()``; ``length``; ``max_length``.
    KVCache.for_config(config, batch_size, dtype, device, max_length=None)

Shapes:
    Per layer: keys/values ``[B, H, max_length, head_dim]``. ``update`` takes ``k``/``v``
    ``[B, H, T, head_dim]`` and returns views ``[B, H, length + T, head_dim]``.

Dtype:
    Chosen at construction and must equal the dtype of the keys written (for example bf16 under
    autocast). A mismatch raises TypeError instead of silently casting.

Device:
    Chosen at construction; writes must come from the same device.

Invariants:
    - All layers write at the same offset (``length``); ``advance`` is called once per forward.
    - ``length <= max_length`` always.
    - Inference only: the cache holds no autograd history.

Failure modes:
    - Writing past ``max_length``, a shape/dtype/device mismatch, or writing tensors that
      require grad while grad mode is enabled raises an error.

See:
    Plan rev 3.3 section 3; used by generation in Milestone D.
"""

from __future__ import annotations

import torch

from kittylm.model.config import ModelConfig

__all__ = ["KVCache"]


class KVCache:
    """Per-layer preallocated key/value buffers."""

    def __init__(
        self,
        *,
        n_layers: int,
        batch_size: int,
        n_heads: int,
        head_dim: int,
        max_length: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        if min(n_layers, batch_size, n_heads, head_dim, max_length) < 1:
            raise ValueError("KVCache dimensions must all be >= 1")
        shape = (batch_size, n_heads, max_length, head_dim)
        self.keys = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(n_layers)]
        self.values = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(n_layers)]
        self.max_length = max_length
        self.dtype = dtype
        self.length = 0

    @classmethod
    def for_config(
        cls,
        config: ModelConfig,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device | str,
        max_length: int | None = None,
    ) -> KVCache:
        """Build a cache sized for ``config`` (``max_length`` defaults to its context length)."""
        return cls(
            n_layers=config.n_layers,
            batch_size=batch_size,
            n_heads=config.n_heads,
            head_dim=config.head_dim,
            max_length=max_length or config.context_length,
            dtype=dtype,
            device=device,
        )

    def update(
        self, layer: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Store ``k``/``v`` for ``layer`` at the current length and return the full views."""
        if torch.is_grad_enabled() and (k.requires_grad or v.requires_grad):
            raise RuntimeError("KVCache is inference-only; run decoding under torch.no_grad()")
        if k.dtype != self.dtype or v.dtype != self.dtype:
            raise TypeError(f"KVCache dtype is {self.dtype}, got {k.dtype}/{v.dtype}")
        buffer = self.keys[layer]
        if k.shape != v.shape or k.shape[:2] != buffer.shape[:2] or k.shape[3] != buffer.shape[3]:
            raise ValueError(f"key/value shape {tuple(k.shape)} does not fit cache {buffer.shape}")
        end = self.length + k.shape[2]
        if end > self.max_length:
            raise ValueError(f"KVCache overflow: {end} > max_length {self.max_length}")
        self.keys[layer][:, :, self.length : end] = k
        self.values[layer][:, :, self.length : end] = v
        return self.keys[layer][:, :, :end], self.values[layer][:, :, :end]

    def advance(self, count: int) -> None:
        """Advance the length by ``count`` tokens after every layer has been updated."""
        if count < 0 or self.length + count > self.max_length:
            raise ValueError("invalid KVCache advance")
        self.length += count

    def reset(self) -> None:
        """Forget all cached tokens (buffers are reused, not reallocated)."""
        self.length = 0
