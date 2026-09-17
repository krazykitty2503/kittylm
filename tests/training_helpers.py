"""Shared builders for training tests (tiny, CPU-fast, deterministic)."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kittylm.config import config_hash, to_dict
from kittylm.data.loader import TOKEN_DTYPE, TrainWindowSampler, open_token_file
from kittylm.model.config import ModelConfig
from kittylm.model.transformer import KittyLM
from kittylm.training.checkpoint import RunIdentity
from kittylm.training.config import TrainingConfig
from kittylm.training.determinism import seed_everything
from kittylm.training.engine import RunInfo, TrainingEngine

COMMIT = hashlib.sha1(b"training-tests").hexdigest()  # fixture commit id, not a security use

TINY_MODEL = ModelConfig(
    name="train-test",
    vocab_size=64,
    d_model=32,
    n_layers=2,
    n_heads=4,
    ffn_dim=64,
    context_length=16,
    attention_backend="reference",
)
TINY_TRAINING = TrainingConfig(
    seed=1234,
    batch_size=4,
    gradient_accumulation=2,
    max_steps=12,
    learning_rate=3e-3,
    min_learning_rate=3e-4,
    warmup_steps=2,
    precision="fp32",
    determinism="deterministic",
    log_every=1,
    timing_sync_every=1,
    cpu_threads=1,
)


def write_tokens(path: Path, count: int, vocab: int, seed: int = 7) -> Path:
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng.integers(0, vocab, size=count, dtype=np.int64).astype(TOKEN_DTYPE).tofile(path)
    return path


def identity_for(model: ModelConfig, training: TrainingConfig, token_file: Path) -> RunIdentity:
    return RunIdentity(
        config_hash=config_hash({"model": to_dict(model), "training": to_dict(training)}),
        tokenizer_sha256=hashlib.sha256(b"test-tokenizer").hexdigest(),
        dataset_version=hashlib.sha256(token_file.read_bytes()).hexdigest(),
    )


def make_engine(
    tmp_path: Path,
    *,
    model_config: ModelConfig = TINY_MODEL,
    training: TrainingConfig = TINY_TRAINING,
    token_file: Path | None = None,
    run_name: str = "run",
    run: RunInfo | None = None,
    **training_changes: Any,
) -> TrainingEngine:
    if training_changes:
        training = replace(training, **training_changes)
    if token_file is None:
        token_file = tmp_path / "tokens.bin"
        # Deterministic contents; never rewritten while another engine may still memory-map it.
        if not token_file.exists():
            write_tokens(token_file, 4000, model_config.vocab_size)
    seed_everything(training.seed)
    model = KittyLM(model_config)
    sampler = TrainWindowSampler(
        open_token_file(token_file),
        context_length=model_config.context_length,
        batch_size=training.batch_size,
        seed=training.seed,
    )
    return TrainingEngine(
        model=model,
        model_config=model_config,
        training_config=training,
        sampler=sampler,
        device=torch.device("cpu"),
        run_dir=tmp_path / run_name,
        identity=identity_for(model_config, training, token_file),
        run=run
        or RunInfo(kind="engineering", experiment_id=run_name, git_commit=COMMIT, git_dirty=False),
    )
