"""TrainingEngine behaviour on CPU: overfitting, gates, guards, metrics, in-process resume."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn
from torch.amp import GradScaler

from kittylm.data.loader import Batch, validation_batches
from kittylm.ledger import REQUIRED_CI_JOBS, CiEvidence, CiJob
from kittylm.model.accounting import AccountingError
from kittylm.training.checkpoint import read_checkpoint
from kittylm.training.engine import (
    MAX_CONSECUTIVE_SKIPS,
    CiGateError,
    NonFiniteError,
    RunInfo,
    TrainingEngine,
)
from kittylm.training.precision import Precision
from tests.training_helpers import COMMIT, TINY_MODEL, make_engine, write_tokens


def test_overfits_a_single_fixed_batch(tmp_path: Path) -> None:
    # A token stream of exactly context_length + 1 tokens has one possible window.
    token_file = write_tokens(tmp_path / "one.bin", TINY_MODEL.context_length + 1, 64, seed=3)
    engine = make_engine(
        tmp_path,
        token_file=token_file,
        batch_size=1,
        gradient_accumulation=1,
        max_steps=300,
        warmup_steps=10,
        learning_rate=1e-2,
        min_learning_rate=1e-3,
        weight_decay=0.0,
        log_every=0,
        timing_sync_every=0,
    )
    results = engine.train(300)
    engine.close()
    assert all(r.batch_starts == [[0]] for r in results)
    assert results[-1].loss < 0.05, f"final loss {results[-1].loss}"
    assert results[-1].loss < results[0].loss / 10


def test_counters_schedule_and_accumulation(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    results = engine.train(5)
    engine.close()
    cfg = engine.config
    assert engine.global_step == 5 == engine.schedule.step
    assert engine.tokens_seen == 5 * cfg.batch_size * cfg.gradient_accumulation * 16
    assert [len(r.batch_starts) for r in results] == [cfg.gradient_accumulation] * 5
    assert results[0].learning_rate == pytest.approx(cfg.learning_rate / cfg.warmup_steps)
    assert results[1].learning_rate == pytest.approx(cfg.learning_rate)


def test_metrics_timing_and_summary(tmp_path: Path) -> None:
    engine = make_engine(tmp_path, run_name="metrics")
    engine.train(4)
    summary = engine.summary()
    engine.close()
    lines = (tmp_path / "metrics" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    assert [r["step"] for r in records] == [1, 2, 3, 4]
    expected = {
        "step",
        "loss",
        "ppl",
        "lr",
        "grad_norm",
        "tokens_seen",
        "tokens_per_second",
        "samples_per_second",
    }
    assert set(records[0]) == expected
    assert records[0]["ppl"] == pytest.approx(math.exp(records[0]["loss"]))
    assert set(summary["timing_breakdown"]) == {"data", "forward", "backward", "optimizer"}
    assert summary["parameters"]["total"] > 0 and summary["tokens_per_second"] > 0


def test_numerical_guard_writes_crash_checkpoint(tmp_path: Path) -> None:
    engine = make_engine(tmp_path, run_name="nan")
    engine.train(1)
    with torch.no_grad():
        engine.model.tok_embeddings.weight[:, 0] = float("nan")
    with pytest.raises(NonFiniteError, match="non-finite"):
        engine.train_step()
    engine.close()
    assert engine.global_step == 1  # the bad step was not applied
    crash = read_checkpoint(tmp_path / "nan" / "checkpoints" / "crash.pt")
    assert crash["global_step"] == 1


def test_validation_and_best_checkpoint(tmp_path: Path) -> None:
    engine = make_engine(tmp_path, run_name="val")
    tokens = np.arange(200, dtype=np.uint16) % 64
    batches = validation_batches(tokens, context_length=16, batch_size=4)
    first = engine.validate(batches)
    engine.train(10)
    second = engine.validate(batches)
    engine.close()
    assert math.isfinite(first) and math.isfinite(second)
    assert (tmp_path / "val" / "checkpoints" / "best_val.pt").exists()


def test_bf16_autocast_training_on_cpu(tmp_path: Path) -> None:
    engine = make_engine(
        tmp_path, run_name="bf16", precision="bf16", determinism="semi_deterministic"
    )
    results = engine.train(3)
    engine.close()
    assert all(math.isfinite(r.loss) for r in results)
    assert engine.model.tok_embeddings.weight.dtype == torch.float32  # master weights stay fp32


def test_in_process_resume_matches_uninterrupted(tmp_path: Path) -> None:
    full = make_engine(tmp_path, run_name="full")
    full_results = full.train(6)
    full.close()

    first = make_engine(tmp_path, run_name="split")
    first.train(3)
    first.save_checkpoint()
    first.close()
    resumed = make_engine(tmp_path, run_name="split")
    assert resumed.resume_latest() and resumed.global_step == 3
    tail = resumed.train(6)
    resumed.close()
    assert [r.loss.hex() for r in full_results[3:]] == [r.loss.hex() for r in tail]
    for (name, a), b in zip(
        full.model.state_dict().items(), resumed.model.state_dict().values(), strict=True
    ):
        assert torch.equal(a, b), name


def evidence(**changes: object) -> CiEvidence:
    jobs = {
        name: CiJob(job_id=i + 1, conclusion="success") for i, name in enumerate(REQUIRED_CI_JOBS)
    }
    base = CiEvidence(commit=COMMIT, workflow="Test", run_id=42, event="push", jobs=jobs)
    return replace(base, **changes)  # type: ignore[arg-type]


def formal(**changes: object) -> RunInfo:
    base = RunInfo(
        kind="formal",
        experiment_id="EXP-900",
        git_commit=COMMIT,
        git_dirty=False,
        ci_evidence=evidence(),
    )
    return replace(base, **changes)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("run", "message"),
    [
        (formal(ci_evidence=None), "ci_evidence is required"),
        (formal(ci_evidence=evidence(commit="f" * 40)), "is for commit"),
        (formal(git_dirty=True), "uncommitted changes"),
        (formal(ci_evidence=evidence(event="pull_request")), "event must be 'push'"),
        (formal(ci_evidence=evidence(event="workflow_dispatch")), "event must be 'push'"),
        (
            formal(ci_evidence=evidence(jobs={"Quality": CiJob(1, "success")})),
            "missing required job",
        ),
    ],
)
def test_formal_runs_require_green_ci_on_the_exact_commit(
    tmp_path: Path, run: RunInfo, message: str
) -> None:
    with pytest.raises(CiGateError, match=message):
        make_engine(tmp_path, run=run)


def test_formal_run_with_valid_evidence_starts(tmp_path: Path) -> None:
    engine = make_engine(tmp_path, run=formal())
    engine.close()


def test_engine_refuses_unaccountable_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests import training_helpers

    real = training_helpers.KittyLM

    def with_extra(config):  # type: ignore[no-untyped-def]
        model = real(config)
        model.extra = nn.Linear(2, 2)
        return model

    monkeypatch.setattr(training_helpers, "KittyLM", with_extra)
    with pytest.raises(AccountingError, match="extra.weight"):
        make_engine(tmp_path)


# --- review regressions (PR #4) -------------------------------------------------------------------


def test_train_refuses_steps_beyond_max_steps(tmp_path: Path) -> None:
    # Regression: train() silently clamped to max_steps, so callers (the resume harness) could
    # believe a larger step count had run.
    engine = make_engine(tmp_path, max_steps=4, warmup_steps=1)
    with pytest.raises(ValueError, match="exceeds max_steps 4"):
        engine.train(5)
    assert engine.global_step == 0
    engine.train(4)
    engine.close()
    assert engine.global_step == 4


def fixed_validation(windows: int, batch_size: int) -> list[Batch]:
    rng = np.random.default_rng(21)
    tokens = rng.integers(0, 64, size=windows * 16 + 1).astype(np.uint16)
    return validation_batches(tokens, context_length=16, batch_size=batch_size)


def test_eval_every_runs_validation_on_schedule(tmp_path: Path) -> None:
    batches = fixed_validation(4, 2)
    engine = make_engine(tmp_path, run_name="eval", eval_every=2, validation_batches=batches)
    engine.train(5)
    engine.close()
    metrics = (tmp_path / "eval" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in metrics]
    assert [r["step"] for r in records if "val_loss" in r] == [2, 4]
    assert (tmp_path / "eval" / "checkpoints" / "best_val.pt").exists()
    assert math.isfinite(engine.best_val_loss)


def test_eval_every_without_validation_batches_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="eval_every > 0 requires validation_batches"):
        make_engine(tmp_path, eval_every=2)


def test_validation_weights_a_partial_final_batch_by_tokens(tmp_path: Path) -> None:
    # Regression: batch means were averaged equally, over-weighting a smaller final batch.
    engine = make_engine(tmp_path, run_name="partial")
    engine.train(3)
    per_window = fixed_validation(3, 1)
    grouped = fixed_validation(3, 2)
    assert [len(b.starts) for b in grouped] == [2, 1]
    window_losses = [engine.validate([b]) for b in per_window]
    reference = sum(window_losses) / 3
    unweighted = (sum(window_losses[:2]) / 2 + window_losses[2]) / 2
    assert abs(unweighted - reference) > 1e-4  # the fixture actually exposes the bias
    assert engine.validate(grouped) == pytest.approx(reference, rel=1e-6)
    assert engine.validate(per_window) == pytest.approx(reference, rel=1e-6)
    engine.close()


def test_best_val_loss_survives_checkpoint_resume(tmp_path: Path) -> None:
    # Regression: best_val_loss was not checkpointed, so the first validation after a resume
    # always replaced best_val.pt, even with a loss that was not better (here: equal).
    batches = fixed_validation(2, 2)
    first = make_engine(tmp_path, run_name="best")
    first.train(3)
    best = first.validate(batches)
    first.save_checkpoint()
    first.close()

    resumed = make_engine(tmp_path, run_name="best")
    assert resumed.resume_latest()
    assert resumed.best_val_loss == best
    best_file = tmp_path / "best" / "checkpoints" / "best_val.pt"
    best_file.unlink()
    assert resumed.validate(batches) == best  # identical weights: not an improvement
    assert not best_file.exists()
    resumed.close()


def test_crash_checkpoint_replays_the_failed_step(tmp_path: Path) -> None:
    # Regression: the crash checkpoint held the loader state *after* the failing micro-batches
    # while global_step was unchanged, so retrying from it skipped the failed batch.
    engine = make_engine(tmp_path, run_name="crash")
    engine.train(1)
    failing_starts = engine.sampler.preview_starts()
    torch_rng_before = torch.get_rng_state()
    loader_before = engine.sampler.state_dict()
    with torch.no_grad():
        engine.model.tok_embeddings.weight[:, 0] = float("nan")
    with pytest.raises(NonFiniteError):
        engine.train_step()
    engine.close()

    crash = read_checkpoint(tmp_path / "crash" / "checkpoints" / "crash.pt")
    assert crash["global_step"] == 1
    assert crash["loader"]["batches_drawn"] == loader_before["batches_drawn"]
    assert torch.equal(crash["loader"]["generator"], loader_before["generator"])
    assert torch.equal(crash["rng"]["torch"], torch_rng_before)

    retry = make_engine(tmp_path, run_name="retry")
    retry.load_state_dict(crash)
    assert retry.sampler.next_batch().starts == failing_starts
    retry.close()


def cpu_fp16_engine(tmp_path: Path, run_name: str) -> TrainingEngine:
    """An fp32 engine given a CPU GradScaler, to exercise the loss-scaler path without a GPU."""
    engine = make_engine(tmp_path, run_name=run_name)
    engine.precision = Precision("fp16", "cpu", None, GradScaler("cpu", init_scale=2.0**16))
    return engine


def poison_gradients(engine: TrainingEngine, calls: int | None) -> Any:
    """Make the next ``calls`` backward passes (every one if None) produce inf gradients."""
    remaining = [calls]

    def hook(grad: torch.Tensor) -> torch.Tensor:
        left = remaining[0]
        if left is None or left > 0:
            if left is not None:
                remaining[0] = left - 1
            return torch.full_like(grad, float("inf"))
        return grad

    return engine.model.tok_embeddings.weight.register_hook(hook)


def test_fp16_skipped_update_does_not_advance_counters(tmp_path: Path) -> None:
    # Regression: an overflow-skipped GradScaler step still advanced the schedule, global_step,
    # tokens_seen and history although no update was applied.
    engine = cpu_fp16_engine(tmp_path, "skip")
    before = {k: v.clone() for k, v in engine.model.state_dict().items()}
    handle = poison_gradients(engine, calls=None)
    result = engine.train_step()
    handle.remove()
    assert result.skipped
    assert (engine.global_step, engine.schedule.step, engine.tokens_seen) == (0, 0, 0)
    assert engine.history == [] and engine.skipped_steps == 1
    scaler = engine.precision.scaler
    assert scaler is not None and scaler.get_scale() == 2.0**15
    for name, value in engine.model.state_dict().items():
        assert torch.equal(value, before[name]), name
    # The skipped attempt consumed its micro-batches (data is not replayed).
    assert engine.sampler.batches_drawn == engine.config.gradient_accumulation

    applied = engine.train_step()
    assert not applied.skipped and engine.global_step == 1 == len(engine.history)
    engine.close()
    log = (tmp_path / "skip" / "train.log").read_text(encoding="utf-8")
    assert "fp16 overflow: skipped update at step 0" in log


def test_train_retries_after_skips_and_returns_applied_steps(tmp_path: Path) -> None:
    engine = cpu_fp16_engine(tmp_path, "retry-skip")
    accumulation = engine.config.gradient_accumulation
    poison_gradients(engine, calls=accumulation)  # exactly the first attempt overflows
    results = engine.train(2)
    engine.close()
    assert [r.step for r in results] == [1, 2] and not any(r.skipped for r in results)
    assert engine.skipped_steps == 1
    assert engine.sampler.batches_drawn == 3 * accumulation
    assert engine.tokens_seen == 2 * engine.config.batch_size * accumulation * 16


def test_persistent_overflow_stops_with_a_crash_checkpoint(tmp_path: Path) -> None:
    engine = cpu_fp16_engine(tmp_path, "overflow")
    poison_gradients(engine, calls=None)
    with pytest.raises(NonFiniteError, match=f"{MAX_CONSECUTIVE_SKIPS} consecutive"):
        engine.train(1)
    engine.close()
    assert engine.skipped_steps == MAX_CONSECUTIVE_SKIPS and engine.global_step == 0
    crash = read_checkpoint(tmp_path / "overflow" / "checkpoints" / "crash.pt")
    assert crash["global_step"] == 0 and crash["skipped_steps"] == MAX_CONSECUTIVE_SKIPS
    assert crash["scaler"] is not None


def test_loss_scaler_state_mismatch_is_refused(tmp_path: Path) -> None:
    scaled = cpu_fp16_engine(tmp_path, "scaled")
    state = scaled.state_dict()
    scaled.close()
    plain = make_engine(tmp_path, run_name="plain")
    with pytest.raises(ValueError, match="loss-scaler state"):
        plain.load_state_dict(state)
    plain.close()
