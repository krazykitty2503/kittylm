"""Unit tests for training building blocks: loader, schedule, optimizer, loss, precision,
timing, logger, determinism and configuration."""

from __future__ import annotations

import json
import math
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from kittylm.config import CONFIG_KINDS, ConfigError, load_config_kinds
from kittylm.data.loader import TrainWindowSampler, open_token_file, validation_batches
from kittylm.model.transformer import KittyLM
from kittylm.training.config import TrainingConfig
from kittylm.training.determinism import capture_rng_state, restore_rng_state
from kittylm.training.logger import ExperimentLogger
from kittylm.training.loss import causal_lm_loss
from kittylm.training.optim import build_optimizer, parameter_groups
from kittylm.training.precision import make_precision
from kittylm.training.schedule import WarmupCosineSchedule, learning_rate_at
from kittylm.training.timing import StepTimer, ThroughputMeter
from tests.training_helpers import TINY_MODEL, TINY_TRAINING, write_tokens

# --- loader ---------------------------------------------------------------------------------------


def test_sampler_is_deterministic_and_targets_are_shifted(tmp_path: Path) -> None:
    tokens = open_token_file(write_tokens(tmp_path / "t.bin", 500, 64))
    a = TrainWindowSampler(tokens, context_length=8, batch_size=3, seed=5)
    b = TrainWindowSampler(tokens, context_length=8, batch_size=3, seed=5)
    for _ in range(4):
        x, y = a.next_batch(), b.next_batch()
        assert x.starts == y.starts
        assert torch.equal(x.inputs, y.inputs)
        assert x.inputs.shape == (3, 8) and x.inputs.dtype == torch.int64
        assert torch.equal(x.inputs[:, 1:], x.targets[:, :-1])
        for row, start in enumerate(x.starts):
            assert x.inputs[row, 0] == int(tokens[start])


def test_preview_does_not_advance_and_state_round_trips(tmp_path: Path) -> None:
    tokens = open_token_file(write_tokens(tmp_path / "t.bin", 500, 64))
    sampler = TrainWindowSampler(tokens, context_length=8, batch_size=3, seed=9)
    sampler.next_batch()
    preview = sampler.preview_starts()
    assert sampler.preview_starts() == preview
    state = sampler.state_dict()
    assert sampler.next_batch().starts == preview
    restored = TrainWindowSampler(tokens, context_length=8, batch_size=3, seed=0)
    restored.load_state_dict(state)
    assert restored.next_batch().starts == preview
    assert restored.batches_drawn == 2


def test_sampler_state_mismatch_and_short_streams(tmp_path: Path) -> None:
    tokens = open_token_file(write_tokens(tmp_path / "t.bin", 500, 64))
    state = TrainWindowSampler(tokens, context_length=8, batch_size=3, seed=1).state_dict()
    with pytest.raises(ValueError, match="batch_size"):
        TrainWindowSampler(tokens, context_length=8, batch_size=4, seed=1).load_state_dict(state)
    with pytest.raises(ValueError, match="shorter"):
        TrainWindowSampler(np.zeros(8, dtype=np.uint16), context_length=8, batch_size=1, seed=1)


def test_validation_batches_are_fixed_and_non_overlapping() -> None:
    tokens = np.arange(50, dtype=np.uint16)
    batches = validation_batches(tokens, context_length=10, batch_size=2)
    starts = [s for b in batches for s in b.starts]
    assert starts == [0, 10, 20, 30]
    again = validation_batches(tokens, context_length=10, batch_size=2)
    assert all(
        torch.equal(a.inputs, b.inputs) and torch.equal(a.targets, b.targets)
        for a, b in zip(batches, again, strict=True)
    )
    assert len(validation_batches(tokens, context_length=10, batch_size=2, max_batches=1)) == 1


# --- schedule -------------------------------------------------------------------------------------


def test_learning_rate_schedule_shape() -> None:
    kw = {"max_lr": 1.0, "min_lr": 0.1, "warmup_steps": 4, "max_steps": 20}
    assert learning_rate_at(0, **kw) == pytest.approx(0.25)
    assert learning_rate_at(3, **kw) == pytest.approx(1.0)
    assert learning_rate_at(4, **kw) == pytest.approx(1.0)
    assert learning_rate_at(12, **kw) == pytest.approx(0.55)
    assert learning_rate_at(20, **kw) == pytest.approx(0.1)
    assert learning_rate_at(99, **kw) == pytest.approx(0.1)
    values = [learning_rate_at(s, **kw) for s in range(25)]
    assert all(0.1 - 1e-12 <= v <= 1.0 + 1e-12 for v in values)
    assert learning_rate_at(0, max_lr=1.0, min_lr=0.0, warmup_steps=0, max_steps=10) == 1.0


def test_schedule_sets_optimizer_and_round_trips() -> None:
    param = torch.nn.Parameter(torch.zeros(2))
    opt = torch.optim.AdamW([param], lr=0.0)
    schedule = WarmupCosineSchedule(opt, max_lr=1.0, min_lr=0.0, warmup_steps=2, max_steps=10)
    assert schedule.apply() == 0.5 and opt.param_groups[0]["lr"] == 0.5
    schedule.advance()
    other = WarmupCosineSchedule(opt, max_lr=1.0, min_lr=0.0, warmup_steps=2, max_steps=10)
    other.load_state_dict(schedule.state_dict())
    assert other.step == 1 and other.apply() == schedule.apply()


# --- optimizer, loss, precision -------------------------------------------------------------------


def test_parameter_groups_decay_only_matrices_and_count_ties_once() -> None:
    model = KittyLM(TINY_MODEL)
    decay, no_decay = parameter_groups(model, 0.1)
    unique = {id(p) for p in model.parameters()}
    assert len(decay["params"]) + len(no_decay["params"]) == len(unique)
    assert all(p.ndim >= 2 for p in decay["params"]) and all(
        p.ndim == 1 for p in no_decay["params"]
    )
    assert any(p is model.tok_embeddings.weight for p in decay["params"])
    assert no_decay["weight_decay"] == 0.0
    optimizer = build_optimizer(model, TINY_TRAINING)
    assert optimizer.param_groups[0]["betas"] == (0.9, 0.95)


def test_loss_matches_uniform_distribution() -> None:
    logits = torch.zeros(2, 5, 17, dtype=torch.bfloat16)
    targets = torch.randint(0, 17, (2, 5))
    assert float(causal_lm_loss(logits, targets)) == pytest.approx(math.log(17), rel=1e-6)
    with pytest.raises(ValueError, match="expected logits"):
        causal_lm_loss(torch.zeros(2, 17), targets)


def test_precision_policies() -> None:
    cpu = torch.device("cpu")
    assert make_precision("fp32", cpu).autocast_dtype is None
    bf16 = make_precision("bf16", cpu)
    with bf16.autocast():
        assert torch.get_autocast_dtype("cpu") == torch.bfloat16
    assert bf16.scaler is None
    with pytest.raises(ValueError, match="requires a GPU"):
        make_precision("fp16", cpu)
    with pytest.raises(ValueError, match="unknown precision"):
        make_precision("int8", cpu)


# --- timing and logging ---------------------------------------------------------------------------


def test_step_timer_records_only_synced_steps() -> None:
    timer = StepTimer(torch.device("cpu"), sync_every=2)
    for step in range(4):
        timer.begin_step(step)
        with timer.phase("forward"):
            pass
    assert len(timer.samples["forward"]) == 2
    summary = timer.summary()["forward"]
    assert set(summary) == {"mean_ms", "p50_ms", "p95_ms"}
    disabled = StepTimer(torch.device("cpu"), sync_every=0)
    disabled.begin_step(0)
    with disabled.phase("forward"):
        pass
    assert disabled.summary() == {}
    meter = ThroughputMeter(torch.device("cpu"))
    with pytest.raises(RuntimeError, match="start"):
        meter.lap(1, 1)
    meter.start()
    tokens_per_s, samples_per_s = meter.lap(1000, 10)
    assert tokens_per_s > 0 and samples_per_s > 0


def test_logger_writes_json_lines_with_non_finite_strings(tmp_path: Path) -> None:
    logger = ExperimentLogger(tmp_path / "run", "unit")
    logger.log_metrics(3, {"loss": float("nan"), "lr": 0.1, "nested": {"x": float("inf")}})
    logger.close()
    record = json.loads((tmp_path / "run" / "metrics.jsonl").read_text(encoding="utf-8"))
    assert record == {"step": 3, "loss": "nan", "lr": 0.1, "nested": {"x": "inf"}}
    assert (tmp_path / "run" / "train.log").exists()


# --- determinism and configuration ----------------------------------------------------------------


def test_rng_capture_and_restore_is_weights_only_safe(tmp_path: Path) -> None:
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    state = capture_rng_state(include_cuda=False)
    expected = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    path = tmp_path / "rng.pt"
    torch.save(state, path)
    loaded = torch.load(path, weights_only=True)
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    restore_rng_state(loaded)
    assert (random.random(), float(np.random.rand()), float(torch.rand(1))) == expected


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"batch_size": 0}, "batch_size"),
        ({"warmup_steps": 50}, "warmup_steps must be <= max_steps"),
        ({"min_learning_rate": 1.0}, "min_learning_rate"),
        ({"grad_clip": 0.0}, "grad_clip"),
        ({"beta2": 1.0}, "beta1 and beta2"),
        ({"checkpoint_every": -1}, "checkpoint_every"),
    ],
)
def test_training_config_validation(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        replace(TINY_TRAINING, **changes)  # type: ignore[arg-type]


def test_training_kind_is_registered() -> None:
    load_config_kinds()
    assert CONFIG_KINDS["training"] is TrainingConfig
