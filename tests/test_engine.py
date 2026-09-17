"""TrainingEngine behaviour on CPU: overfitting, gates, guards, metrics, in-process resume."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from kittylm.data.loader import validation_batches
from kittylm.ledger import REQUIRED_CI_JOBS, CiEvidence, CiJob
from kittylm.model.accounting import AccountingError
from kittylm.training.checkpoint import read_checkpoint
from kittylm.training.engine import CiGateError, NonFiniteError, RunInfo
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
