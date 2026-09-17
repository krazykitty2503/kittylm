"""Causal language-modeling loss.

Purpose:
    A language model is trained to assign high probability to the actual next token. The loss is
    the average negative natural-log probability of each target token (cross-entropy). Its
    exponential is perplexity; divided by ``ln 2`` and normalized by bytes it becomes
    bits-per-byte (Milestone D).

Public API:
    causal_lm_loss(logits, targets) -> scalar tensor

Shapes:
    ``logits`` ``[B, T, V]``, ``targets`` ``[B, T]`` -> scalar.

Dtype:
    Logits are upcast to float32 before the softmax, so bf16 autocast cannot underflow the
    log-probabilities.

Device:
    Same device as the inputs.

Math:
    ``loss = -(1 / (B*T)) * sum log softmax(logits)[target]`` (natural log).

Invariants:
    - Uniform logits over V classes give ``ln(V)``.

Failure modes:
    - Shape mismatches raise ValueError.

See:
    D-003 (bits-per-byte), plan rev 3.3 section 4.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["causal_lm_loss"]


def causal_lm_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean natural-log cross-entropy of ``targets`` under ``logits``."""
    if logits.dim() != 3 or targets.shape != logits.shape[:2]:
        raise ValueError(
            f"expected logits [B, T, V] and targets [B, T], got {tuple(logits.shape)} and "
            f"{tuple(targets.shape)}"
        )
    vocab = logits.shape[-1]
    return F.cross_entropy(logits.float().reshape(-1, vocab), targets.reshape(-1))
