"""Ledger tests. All numbers here are synthetic test fixtures, not experiment results."""

from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
from typing import Any

import pytest

from kittylm.ledger import (
    REQUIRED_CI_JOBS,
    RESUME_CATEGORIES,
    RESUME_EXACT_REQUIRED,
    RESUME_HARNESS,
    LedgerError,
    load_all_records,
    load_record,
    parse_record,
    render_ablation_table,
    render_record,
    write_record,
)
from tests.conftest import ROOT
from tests.fakes import fake_github_token

SHA = hashlib.sha256(b"fixture").hexdigest()
COMMIT = hashlib.sha1(b"fixture").hexdigest()  # fixture commit id, not a security use


def valid_record() -> dict[str, Any]:
    loss = 2.0
    return {
        "schema_version": 2,
        "experiment": {
            "id": "EXP-900",
            "kind": "formal",
            "name": "fixture",
            "description": "test fixture",
        },
        "repository": {"git_commit": COMMIT, "git_dirty": False},
        "environment": {
            "python": "3.12.10",
            "pytorch": "2.13.0",
            "backend": "cpu",
            "backend_version": None,
            "device": "cpu",
            "dtype": "fp32",
            "os": "test-os",
        },
        "dataset": {
            "name": "fixture-v1",
            "version": SHA,
            "manifest_sha256": SHA,
            "recipe_version": 1,
            "sources": 1,
            "documents": 3,
            "training_bytes": 4000,
            "validation_bytes": 400,
            "training_tokens": 1000,
            "validation_tokens": 100,
        },
        "tokenizer": {
            "version": "bpe-512@fixture",
            "vocab_size": 512,
            "tokenizer_sha256": SHA,
            "bytes_per_token": 4.0,
            "tokens_per_byte": 0.25,
        },
        "model": {
            "architecture": "transformer",
            "d_model": 16,
            "layers": 1,
            "heads": 2,
            "ffn_dim": 32,
            "context_length": 8,
            "parameters": {
                "total": 10000,
                "trainable": 10000,
                "embedding": 8192,
                "attention": 1024,
                "mlp": 752,
                "normalization": 32,
                "positional": 0,
                "output": 0,
                "tied_parameters": 8192,
                "non_embedding": 1808,
            },
        },
        "training": {
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "schedule": "cosine",
            "batch_size": 4,
            "gradient_accumulation": 1,
            "max_steps": 10,
            "training_steps": 10,
            "tokens_seen": 320,
            "bytes_seen": 1280,
            "precision": "fp32",
            "attention_kernel": "math",
        },
        "results": {
            "final_loss": loss,
            "final_ppl": math.exp(loss),
            "final_bpb": 0.72,
            "val_bpb_by_category": {"code": 0.7},
            "test_ppl": None,
            "tokens_per_second": 100.0,
            "samples_per_second": 1.0,
            "timing_breakdown": {"forward": {"mean_ms": 1.0, "p50_ms": 1.0, "p95_ms": 2.0}},
            "peak_vram_gb": None,
            "allocated_vram_gb": None,
            "duration_seconds": 3.5,
            "inference": None,
        },
        "reproducibility": {
            "determinism_mode": "deterministic",
            "checkpoint_resume_test": "not_run",
            "resume_evidence": None,
        },
        "ci_evidence": {
            "commit": COMMIT,
            "workflow": "Test",
            "run_id": 1,
            "event": "push",
            "jobs": {
                name: {"job_id": i + 1, "conclusion": "success"}
                for i, name in enumerate(REQUIRED_CI_JOBS)
            },
        },
        "limitations": ["synthetic fixture"],
        "notes": "",
    }


def test_valid_record_parses_and_round_trips() -> None:
    record = parse_record(valid_record())
    assert record.model.parameters.total == 10000
    assert render_record(record) == render_record(record)


def mutate(path: str, value: Any) -> dict[str, Any]:
    data = copy.deepcopy(valid_record())
    node = data
    keys = path.split(".")
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = value
    return data


@pytest.mark.parametrize(
    ("path", "value", "problem"),
    [
        ("model.parameters.attention", 1000, "!= total"),
        ("model.parameters.non_embedding", 1, "non_embedding"),
        ("results.final_ppl", 9.0, "exp(final_loss)"),
        ("tokenizer.bytes_per_token", 3.0, "bytes_per_token"),
        ("repository.git_commit", "abc", "40-character"),
        ("dataset.version", "v1", "sha256"),
        ("experiment.id", "exp-1", "EXP-###"),
        ("reproducibility.checkpoint_resume_test", "passed", "requires resume_evidence"),
        ("limitations", [], "at least one"),
        ("notes", "ran from C:\\Users\\someone\\kittylm", "absolute path"),
        ("notes", "see /home/someone/runs", "absolute path"),
        ("results.tokens_per_second", -1.0, ">= 0"),
        ("training.precision", "int8", "expected one of"),
        ("unexpected", 1, "unknown key"),
    ],
)
def test_invalid_records_are_rejected(path: str, value: Any, problem: str) -> None:
    with pytest.raises(LedgerError) as exc:
        parse_record(mutate(path, value))
    assert problem in str(exc.value)


def full_comparison(device_type: str = "cpu", **verdicts: str) -> dict[str, str]:
    comparison = {category: "exact" for category in RESUME_CATEGORIES}
    comparison.update(verdicts)
    comparison["device_type"] = device_type
    return comparison


def resume_record(status: str, comparison: dict[str, str]) -> dict[str, Any]:
    data = mutate("reproducibility.checkpoint_resume_test", status)
    data["reproducibility"]["resume_evidence"] = {
        "harness": RESUME_HARNESS,
        "run_id": "run-1",
        "kill_step": 5,
        "resumed_to_step": 10,
        "metrics_sha256": SHA,
        "comparison": comparison,
    }
    return data


def test_resume_evidence_rules() -> None:
    data = resume_record("passed", full_comparison())
    evidence = data["reproducibility"]["resume_evidence"]
    assert parse_record(data).reproducibility.checkpoint_resume_test == "passed"

    forged = copy.deepcopy(data)
    forged["reproducibility"]["resume_evidence"]["harness"] = "typed-by-hand"
    with pytest.raises(LedgerError, match="harness"):
        parse_record(forged)

    stale = mutate("reproducibility.resume_evidence", evidence)
    with pytest.raises(LedgerError, match="must be null"):
        parse_record(stale)


@pytest.mark.parametrize("missing", RESUME_CATEGORIES)
def test_resume_comparison_must_be_complete(missing: str) -> None:
    comparison = full_comparison()
    del comparison[missing]
    for status in ("passed", "failed"):
        with pytest.raises(LedgerError, match=f"missing categories \\['{missing}'\\]"):
            parse_record(resume_record(status, comparison))


@pytest.mark.parametrize("category", RESUME_EXACT_REQUIRED["cpu"])
def test_passed_requires_every_exact_category_on_cpu(category: str) -> None:
    comparison = full_comparison(**{category: "max_abs_dev=1.000e-07"})
    with pytest.raises(LedgerError, match="not exact"):
        parse_record(resume_record("passed", comparison))
    # The same divergence is a consistent record when the status says failed.
    assert parse_record(resume_record("failed", comparison))


def test_gpu_passed_allows_measured_tensor_deviations_only() -> None:
    deviations = {
        c: "max_abs_dev=2.000e-06"
        for c in RESUME_CATEGORIES
        if c not in RESUME_EXACT_REQUIRED["cuda"]
    }
    assert set(deviations) == {"optimizer_state", "model_parameters", "loss_series"}
    assert parse_record(resume_record("passed", full_comparison("cuda", **deviations)))
    for category in RESUME_EXACT_REQUIRED["cuda"]:
        broken = full_comparison("cuda", **{category: "values differ"})
        with pytest.raises(LedgerError, match="not exact"):
            parse_record(resume_record("passed", broken))


@pytest.mark.parametrize(
    ("comparison", "status", "problem"),
    [
        (full_comparison(), "failed", "every required category is exact"),
        (full_comparison("rocm"), "passed", "device_type must be one of"),
        ({**full_comparison(), "vibes": "exact"}, "passed", "unknown categories"),
        ({"global_step": "exact"}, "passed", "device_type must be one of"),
        ({}, "passed", "device_type must be one of"),
    ],
)
def test_inconsistent_resume_comparisons_are_rejected(
    comparison: dict[str, str], status: str, problem: str
) -> None:
    with pytest.raises(LedgerError, match=problem):
        parse_record(resume_record(status, comparison))


def test_forbidden_identifiers_and_secrets() -> None:
    with pytest.raises(LedgerError, match="username/hostname"):
        parse_record(mutate("notes", "trained on kittybox"), forbidden_identifiers=["kittybox"])
    with pytest.raises(LedgerError, match="secret scanner"):
        parse_record(mutate("notes", f"oops {fake_github_token()}"))


def test_write_and_load_record(tmp_path: Path) -> None:
    record = parse_record(valid_record())
    path = write_record(record, tmp_path)
    assert path == tmp_path / "experiments" / "EXP-900" / "record.yaml"
    assert load_record(path) == record
    assert load_all_records(tmp_path / "experiments") == [record]


def test_record_directory_must_match_id(tmp_path: Path) -> None:
    record = parse_record(valid_record())
    wrong = tmp_path / "EXP-901" / "record.yaml"
    wrong.parent.mkdir()
    wrong.write_text(render_record(record), encoding="utf-8")
    with pytest.raises(LedgerError, match="!= experiment.id"):
        load_record(wrong)


def test_ablation_table_is_deterministic() -> None:
    a = parse_record(valid_record())
    b = parse_record(mutate("experiment.id", "EXP-100"))
    table = render_ablation_table([a, b])
    assert table == render_ablation_table([b, a])
    assert table.index("EXP-100") < table.index("EXP-900")
    assert "81.9" in table  # embedding share 8192 / 10000
    assert "No formal experiment records yet" in render_ablation_table([])


def test_committed_ablation_table_is_in_sync() -> None:
    records = load_all_records(ROOT / "experiments")
    committed = (ROOT / "experiments" / "ablations.md").read_text(encoding="utf-8")
    assert committed == render_ablation_table(records)
