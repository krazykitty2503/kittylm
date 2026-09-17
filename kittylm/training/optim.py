"""AdamW optimizer with weight decay on 2-D weights only.

Purpose:
    AdamW adapts each parameter's step size from running estimates of its gradient mean and
    variance, and applies weight decay directly to the weights (decoupled from the gradient).
    Decay shrinks weight matrices toward zero as regularization; applying it to vectors (norm
    gains) would push those toward zero and hurt training, so only parameters with two or more
    dimensions (linear layers and the tied embedding) are decayed.

Public API:
    parameter_groups(model, weight_decay) -> list[dict]
    build_optimizer(model, config) -> torch.optim.AdamW

Shapes:
    Optimizer state tensors (``exp_avg``, ``exp_avg_sq``) have the shape of their parameter.

Dtype:
    Parameters and optimizer state are float32 regardless of autocast.

Device:
    State lives on the parameters' device.

Math:
    ``m = b1 m + (1-b1) g``; ``v = b2 v + (1-b2) g^2``;
    ``w <- w - lr * (m_hat / (sqrt(v_hat) + eps) + wd * w)``.

Invariants:
    - Each unique parameter appears in exactly one group (tied weights once).
    - Group order and parameter order follow ``model.named_parameters()``, so optimizer state
      maps back to parameters identically after a checkpoint reload.

Failure modes:
    - A model with no parameters raises ValueError.

See:
    Plan rev 3.3 section 4.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from kittylm.training.config import TrainingConfig

__all__ = ["build_optimizer", "parameter_groups"]


def parameter_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """Split unique trainable parameters into decayed (ndim >= 2) and non-decayed groups."""
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for _, param in model.named_parameters():  # named_parameters() de-duplicates tied tensors
        if not param.requires_grad:
            continue
        (decay if param.ndim >= 2 else no_decay).append(param)
    if not decay and not no_decay:
        raise ValueError("model has no trainable parameters")
    return [
        {"params": decay, "weight_decay": weight_decay, "name": "decay"},
        {"params": no_decay, "weight_decay": 0.0, "name": "no_decay"},
    ]


def build_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.AdamW:
    """AdamW over ``parameter_groups`` with the configured hyperparameters."""
    return torch.optim.AdamW(
        parameter_groups(model, config.weight_decay),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.adam_eps,
    )
