"""Training configuration (``kind: training``).

Purpose:
    Every optimization knob of a run in one strict, typed place, so a run's resolved
    configuration can be hashed into its checkpoints and records. A resumed run with a different
    configuration hash is refused.

Public API:
    TrainingConfig
        Frozen dataclass registered as ``kind: training``.

Invariants:
    - ``min_learning_rate <= learning_rate``; ``warmup_steps <= max_steps``.
    - Every floating-point hyperparameter (FLOAT_FIELDS) is finite: NaN and +/-Inf are rejected
      before any range check, because a range comparison with NaN is always False.
    - ``gradient_accumulation`` micro-batches of ``batch_size`` form one optimizer step, so each
      step consumes ``batch_size * gradient_accumulation * context_length`` tokens.

Failure modes:
    - Invalid values raise ConfigError at construction.

See:
    Plan rev 3.3 section 4, D-011 (strict configuration), D-018.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Literal

from kittylm.config import ConfigError, register_config_kind

__all__ = ["FLOAT_FIELDS", "TrainingConfig"]

FLOAT_FIELDS: tuple[str, ...] = (
    "learning_rate",
    "min_learning_rate",
    "weight_decay",
    "beta1",
    "beta2",
    "adam_eps",
    "grad_clip",
)


@dataclass(frozen=True)
class TrainingConfig:
    """``kind: training`` configuration."""

    seed: int
    batch_size: int
    gradient_accumulation: int
    max_steps: int
    learning_rate: float
    min_learning_rate: float
    warmup_steps: int
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    adam_eps: float = 1e-8
    grad_clip: float = 1.0
    precision: Literal["bf16", "fp16", "fp32"] = "bf16"
    determinism: Literal["deterministic", "semi_deterministic", "nondeterministic"] = (
        "semi_deterministic"
    )
    log_every: int = 10
    eval_every: int = 0  # validate every N steps; > 0 requires validation batches (engine)
    checkpoint_every: int = 0
    keep_last: int = 3
    timing_sync_every: int = 10
    cpu_threads: int = 0

    def __post_init__(self) -> None:
        """Validate ranges (every floating-point field must be finite first)."""
        for name in FLOAT_FIELDS:
            if not math.isfinite(getattr(self, name)):
                raise ConfigError(f"{name} must be finite, got {getattr(self, name)!r}")
        float_fields = {f.name for f in fields(self) if f.type in ("float", float)}
        if float_fields != set(FLOAT_FIELDS):  # a new float field must be added to FLOAT_FIELDS
            raise ConfigError(f"FLOAT_FIELDS is out of date: {sorted(float_fields)}")
        for name in ("batch_size", "gradient_accumulation", "max_steps", "keep_last"):
            if getattr(self, name) < 1:
                raise ConfigError(f"{name} must be >= 1")
        for name in ("warmup_steps", "log_every", "eval_every", "checkpoint_every"):
            if getattr(self, name) < 0:
                raise ConfigError(f"{name} must be >= 0")
        if self.timing_sync_every < 0 or self.cpu_threads < 0:
            raise ConfigError("timing_sync_every and cpu_threads must be >= 0")
        if self.warmup_steps > self.max_steps:
            raise ConfigError("warmup_steps must be <= max_steps")
        if not 0 < self.learning_rate or not 0 <= self.min_learning_rate <= self.learning_rate:
            raise ConfigError(
                "require 0 <= min_learning_rate <= learning_rate and learning_rate > 0"
            )
        if self.weight_decay < 0 or self.grad_clip <= 0 or self.adam_eps <= 0:
            raise ConfigError("weight_decay must be >= 0; grad_clip and adam_eps must be > 0")
        if not (0 <= self.beta1 < 1 and 0 <= self.beta2 < 1):
            raise ConfigError("beta1 and beta2 must be in [0, 1)")


register_config_kind("training", TrainingConfig)
