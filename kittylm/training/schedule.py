"""Learning-rate schedule: linear warmup, then cosine decay to a minimum.

Purpose:
    Large updates at the start of training, when weights are random, can destabilize a
    Transformer, so the learning rate ramps up linearly over ``warmup_steps``. Afterwards it
    follows half a cosine from the peak down to ``min_learning_rate`` at ``max_steps``, which
    lowers update noise as the model converges. The schedule is a pure function of the step
    number, so its state is a single integer and resumes exactly.

Public API:
    learning_rate_at(step, *, max_lr, min_lr, warmup_steps, max_steps) -> float
    WarmupCosineSchedule(optimizer, ...)
        ``apply()`` sets the learning rate for the upcoming step, ``advance()``,
        ``state_dict()`` / ``load_state_dict()``, ``step``, ``last_lr``.

Shapes:
    Not applicable.

Dtype:
    Python floats.

Device:
    Not applicable.

Math:
    For 0-based step ``s``: ``lr = max_lr * (s + 1) / warmup`` while ``s < warmup``; then
    ``lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * p))`` with
    ``p = (s - warmup) / (max_steps - warmup)``; ``lr = min_lr`` for ``s >= max_steps``.

Invariants:
    - ``min_lr <= lr <= max_lr`` for every step; the peak is reached at the last warmup step.
    - Identical step -> identical learning rate (no hidden state).

Failure modes:
    - Loading a state dict without ``step`` raises KeyError.

See:
    Plan rev 3.3 section 4.
"""

from __future__ import annotations

import math
from typing import Any

import torch

__all__ = ["WarmupCosineSchedule", "learning_rate_at"]


def learning_rate_at(
    step: int, *, max_lr: float, min_lr: float, warmup_steps: int, max_steps: int
) -> float:
    """Learning rate for 0-based optimizer step ``step``."""
    if warmup_steps > 0 and step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


class WarmupCosineSchedule:
    """Sets optimizer learning rates from ``learning_rate_at`` and tracks the step."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        max_lr: float,
        min_lr: float,
        warmup_steps: int,
        max_steps: int,
    ) -> None:
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.step = 0
        self.last_lr = 0.0

    def apply(self) -> float:
        """Set the learning rate for the upcoming optimizer step and return it."""
        lr = learning_rate_at(
            self.step,
            max_lr=self.max_lr,
            min_lr=self.min_lr,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.last_lr = lr
        return lr

    def advance(self) -> None:
        """Move to the next step after the optimizer step completed."""
        self.step += 1

    def state_dict(self) -> dict[str, Any]:
        """Checkpointable schedule state."""
        return {"step": self.step, "last_lr": self.last_lr}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore schedule state."""
        self.step = int(state["step"])
        self.last_lr = float(state["last_lr"])
