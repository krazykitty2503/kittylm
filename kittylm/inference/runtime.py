"""Small runtime helpers shared by generation and evaluation.

Purpose:
    Run a model in inference mode consistently: optional autocast, the dtype a KV cache must
    use under that autocast, and device synchronization for honest wall-clock timing (GPU
    kernels are asynchronous, so a timer must wait for queued work to finish).

Public API:
    autocast_context(device, dtype)
    cache_dtype(model, dtype) -> torch.dtype
    model_config(model) -> ModelConfig
    resolve_context_length(model, context_length) -> int
    synchronize(device)
    eval_mode(model)

Shapes:
    Not applicable (no tensors are created).

Dtype:
    ``dtype=None`` means no autocast: the model runs in its parameter dtype.

Device:
    ``synchronize`` is a no-op on CPU and waits for all queued kernels on CUDA/ROCm.

Invariants:
    - ``eval_mode`` restores the model's previous train/eval mode on exit.
    - ``cache_dtype`` equals the dtype attention keys actually have under the context.
    - ``model_config`` only returns a real ModelConfig (models and test doubles alike).

Failure modes:
    - Autocast to a dtype the device does not support raises inside PyTorch.
    - A model without a ``ModelConfig`` ``config`` attribute raises TypeError.
    - A context override outside ``[1, model context_length]`` raises ValueError before any
      tensor or cache is allocated.

See:
    kittylm/inference/generate.py, kittylm/evaluation/inference_speed.py.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import torch
from torch import nn

from kittylm.model.config import ModelConfig

__all__ = [
    "autocast_context",
    "cache_dtype",
    "eval_mode",
    "model_config",
    "resolve_context_length",
    "synchronize",
]


@contextlib.contextmanager
def autocast_context(device: torch.device, dtype: torch.dtype | None) -> Iterator[None]:
    """Autocast to ``dtype`` on ``device`` (no-op when ``dtype`` is None)."""
    if dtype is None:
        yield
    else:
        with torch.autocast(device_type=device.type, dtype=dtype):
            yield


def cache_dtype(model: nn.Module, dtype: torch.dtype | None) -> torch.dtype:
    """Dtype of attention keys/values: the autocast dtype, else the parameter dtype."""
    if dtype is not None:
        return dtype
    return next(model.parameters()).dtype


def synchronize(device: torch.device) -> None:
    """Wait for queued GPU work (no-op on CPU)."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextlib.contextmanager
def eval_mode(model: nn.Module) -> Iterator[nn.Module]:
    """Put ``model`` in eval mode for the block, then restore its previous mode."""
    was_training = model.training
    model.eval()
    try:
        yield model
    finally:
        model.train(was_training)


def model_config(model: nn.Module) -> ModelConfig:
    """The model's ``ModelConfig`` (context length, vocabulary and cache shape)."""
    config = getattr(model, "config", None)
    if not isinstance(config, ModelConfig):
        raise TypeError("model must expose its ModelConfig as `.config`")
    return config


def resolve_context_length(model: nn.Module, context_length: int | None) -> int:
    """The context to use: the model's own, or a validated smaller override."""
    limit = model_config(model).context_length
    if context_length is None:
        return limit
    if not 1 <= context_length <= limit:
        raise ValueError(
            f"context_length must be in [1, {limit}] for this model, got {context_length}"
        )
    return context_length
