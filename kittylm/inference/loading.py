"""Load a trained model for evaluation or generation, refusing mismatched artifacts.

Purpose:
    Evaluation and generation need three artifacts that must belong together: the model
    configuration, a checkpoint, and the tokenizer the checkpoint was trained with. A model
    decoding with the wrong tokenizer produces fluent-looking nonsense and wrong bits-per-byte,
    so the loader checks that they match before returning anything.

Public API:
    LoadedModel(model, config, tokenizer, metadata, global_step)
    load_for_inference(model_config_path, checkpoint_path, tokenizer_path, device,
                       overrides=()) -> LoadedModel
        ``overrides`` are ``key=value`` strings, e.g. ``vocab_size=512`` for the smoke vocabulary.
    precision_dtype(name) -> torch.dtype | None

Shapes:
    Not applicable (loads parameters of the configured shapes).

Dtype:
    Parameters load as saved (float32 master weights); inference precision is applied later
    through autocast (``precision_dtype``).

Device:
    The checkpoint is verified and read on CPU, then the model is moved to ``device``.

Invariants:
    - The checkpoint passes the checksum and ``weights_only`` loading of D-021.
    - ``tokenizer.sha256 == metadata["tokenizer_sha256"]`` and
      ``tokenizer.vocab_size == config.vocab_size``.
    - The returned model is in eval mode.

Failure modes:
    - A config file without ``kind: model``, a tokenizer/vocabulary mismatch, or a checkpoint
      whose weights do not fit the configuration raises LoadError.
    - Damaged checkpoints raise CheckpointError (from kittylm/training/checkpoint.py).

See:
    D-021, kittylm/training/checkpoint.py, scripts/evaluate.py, scripts/generate.py.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from kittylm.config import apply_overrides, from_dict, load_yaml
from kittylm.model.config import ModelConfig
from kittylm.model.transformer import KittyLM
from kittylm.tokenizer.bpe import BPETokenizer
from kittylm.training.checkpoint import read_checkpoint

__all__ = ["LoadError", "LoadedModel", "load_for_inference", "precision_dtype"]

_PRECISIONS: dict[str, torch.dtype | None] = {"fp32": None, "bf16": torch.bfloat16}


class LoadError(ValueError):
    """The model configuration, checkpoint and tokenizer do not belong together."""


@dataclass(frozen=True)
class LoadedModel:
    """A model ready for inference, with the artifacts it was verified against."""

    model: KittyLM
    config: ModelConfig
    tokenizer: BPETokenizer
    metadata: dict[str, Any]
    global_step: int


def precision_dtype(name: str) -> torch.dtype | None:
    """Autocast dtype for an inference precision name (``fp32`` means no autocast)."""
    if name not in _PRECISIONS:
        raise LoadError(f"inference precision must be one of {sorted(_PRECISIONS)}")
    return _PRECISIONS[name]


def load_for_inference(
    model_config_path: Path,
    checkpoint_path: Path,
    tokenizer_path: Path,
    device: torch.device,
    overrides: Sequence[str] = (),
) -> LoadedModel:
    """Load and cross-check config, checkpoint and tokenizer; return an eval-mode model."""
    data = load_yaml(model_config_path)
    if data.get("kind") != "model":
        raise LoadError(f"{model_config_path.name} must declare kind: model")
    fields = apply_overrides({k: v for k, v in data.items() if k != "kind"}, list(overrides))
    config = from_dict(ModelConfig, fields)
    tokenizer = BPETokenizer.load(tokenizer_path)
    if tokenizer.vocab_size != config.vocab_size:
        raise LoadError(
            f"tokenizer vocab_size {tokenizer.vocab_size} != model vocab_size {config.vocab_size}"
        )
    state = read_checkpoint(checkpoint_path)
    metadata = dict(state["metadata"])
    if metadata.get("tokenizer_sha256") != tokenizer.sha256:
        raise LoadError("checkpoint was trained with a different tokenizer (sha256 mismatch)")
    model = KittyLM(config)
    try:
        model.load_state_dict(state["model"])
    except RuntimeError as exc:
        raise LoadError(f"checkpoint weights do not fit {config.name}: {exc}") from exc
    model.to(device).eval()
    return LoadedModel(model, config, tokenizer, metadata, int(state["global_step"]))
