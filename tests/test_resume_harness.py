"""Deterministic fresh-process resume validation (plan rev 3.3 section 4, D-012)."""

from __future__ import annotations

import ast
import copy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest
import torch

from kittylm.config import from_dict, load_yaml, to_dict
from kittylm.ledger import RESUME_HARNESS, parse_record
from kittylm.model.config import ModelConfig
from kittylm.training.resume_harness import (
    CPU_EXACT_CATEGORIES,
    HarnessError,
    HarnessSpec,
    compare_runs,
    validate_resume,
)
from tests.conftest import ROOT
from tests.test_ledger import mutate
from tests.training_helpers import COMMIT, TINY_TRAINING, write_tokens


def nano_spec(tmp_path: Path) -> HarnessSpec:
    data = load_yaml(ROOT / "configs" / "model" / "nano.yaml")
    nano = from_dict(ModelConfig, {k: v for k, v in data.items() if k != "kind"})
    token_file = write_tokens(tmp_path / "tokens.bin", 20_000, nano.vocab_size, seed=11)
    training = replace(
        TINY_TRAINING,
        batch_size=2,
        gradient_accumulation=2,
        max_steps=8,
        warmup_steps=2,
        precision="fp32",
        determinism="deterministic",
        log_every=0,
        timing_sync_every=0,
        cpu_threads=1,
    )
    return HarnessSpec(
        run_id="test-cpu-nano",
        model=to_dict(nano),
        training=to_dict(training),
        token_file=str(token_file),
        device="cpu",
        kill_step=4,
        total_steps=8,
        git_commit=COMMIT,
    )


@pytest.fixture(scope="module")
def cpu_report(tmp_path_factory: pytest.TempPathFactory) -> Any:
    tmp = tmp_path_factory.mktemp("resume")
    spec = nano_spec(tmp)
    return spec, validate_resume(spec, tmp / "work")


def test_cpu_nano_fp32_fresh_process_resume_is_bit_exact(cpu_report: Any) -> None:
    spec, report = cpu_report
    assert report.passed, report.categories
    assert report.exact_required == CPU_EXACT_CATEGORIES
    for category in CPU_EXACT_CATEGORIES:
        assert report.categories[category] == "exact", (category, report.categories[category])
    assert report.categories["device_type"] == "cpu"
    assert report.checkpoint_resume_test == "passed"


def test_evidence_is_complete_and_accepted_by_the_ledger(cpu_report: Any) -> None:
    spec, report = cpu_report
    ev = report.evidence
    assert ev.harness == RESUME_HARNESS == "kittylm.training.resume_harness"
    assert (ev.kill_step, ev.resumed_to_step) == (spec.kill_step, spec.total_steps)
    assert len(ev.metrics_sha256) == 64
    assert set(CPU_EXACT_CATEGORIES) <= set(ev.comparison)
    data = mutate("reproducibility.checkpoint_resume_test", "passed")
    data["reproducibility"]["resume_evidence"] = to_dict(ev)
    record = parse_record(data)
    assert record.reproducibility.resume_evidence is not None


def test_harness_rejects_invalid_split(tmp_path: Path) -> None:
    spec = replace(nano_spec(tmp_path), kill_step=8)
    with pytest.raises(HarnessError, match="kill_step"):
        validate_resume(spec, tmp_path / "work")


def test_harness_reports_worker_failures(tmp_path: Path) -> None:
    spec = replace(nano_spec(tmp_path), token_file=str(tmp_path / "missing.bin"))
    with pytest.raises(HarnessError, match="full worker failed"):
        validate_resume(spec, tmp_path / "work")


# --- the comparison detects every kind of divergence ----------------------------------------------


def fake_runs() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]
]:
    loss = [float(i).hex() for i in (1, 2, 3, 4)]
    lrs = [float(i / 10).hex() for i in (1, 2, 3, 4)]
    starts = [[[1, 2]], [[3, 4]], [[5, 6]], [[7, 8]]]
    full = {
        "global_step": 4,
        "losses": loss,
        "learning_rates": lrs,
        "batch_starts": starts,
        "next_batch_starts": [9, 10],
    }
    first = {
        "global_step": 2,
        "losses": loss[:2],
        "learning_rates": lrs[:2],
        "batch_starts": starts[:2],
        "next_batch_starts": [5, 6],
    }
    resume = {
        "global_step": 4,
        "losses": loss[2:],
        "learning_rates": lrs[2:],
        "batch_starts": starts[2:],
        "next_batch_starts": [9, 10],
    }
    state = {
        "global_step": 4,
        "scheduler": {"step": 4, "last_lr": 0.4},
        "optimizer": {
            "state": {0: {"exp_avg": torch.ones(3), "step": torch.tensor(4.0)}},
            "param_groups": [{"lr": 0.4, "params": [0]}],
        },
        "model": {"w": torch.arange(6, dtype=torch.float32)},
        "rng": {
            "python": {"version": 3, "state": [1, 2], "gauss": None},
            "numpy": {"keys": torch.arange(4)},
            "torch": torch.arange(8, dtype=torch.uint8),
            "cuda": None,
        },
        "loader": {"generator": torch.arange(8, dtype=torch.uint8), "batches_drawn": 8},
    }
    return full, first, resume, state, copy.deepcopy(state)


def test_identical_fake_runs_compare_exact() -> None:
    categories = compare_runs(*fake_runs(), device_type="cpu")
    assert all(categories[name] == "exact" for name in CPU_EXACT_CATEGORIES)


@pytest.mark.parametrize(
    ("category", "tamper"),
    [
        ("global_step", lambda f, a, r, s, t: t.update(global_step=5)),
        ("scheduler", lambda f, a, r, s, t: t["scheduler"].update(last_lr=0.5)),
        ("learning_rates", lambda f, a, r, s, t: r["learning_rates"].__setitem__(0, (9.0).hex())),
        ("optimizer_state", lambda f, a, r, s, t: t["optimizer"]["state"][0]["exp_avg"].add_(1e-7)),
        ("model_parameters", lambda f, a, r, s, t: t["model"]["w"].add_(1e-7)),
        ("rng_python", lambda f, a, r, s, t: t["rng"]["python"]["state"].append(3)),
        ("rng_numpy", lambda f, a, r, s, t: t["rng"]["numpy"]["keys"].add_(1)),
        (
            "rng_torch",
            lambda f, a, r, s, t: t["rng"].update(torch=torch.zeros(8, dtype=torch.uint8)),
        ),
        (
            "rng_cuda",
            lambda f, a, r, s, t: t["rng"].update(cuda=[torch.zeros(2, dtype=torch.uint8)]),
        ),
        ("loader_state", lambda f, a, r, s, t: t["loader"].update(batches_drawn=7)),
        ("loss_series", lambda f, a, r, s, t: r["losses"].__setitem__(1, (4.5).hex())),
        ("batch_indices", lambda f, a, r, s, t: r["batch_starts"].__setitem__(0, [[0, 0]])),
        ("next_batch_indices", lambda f, a, r, s, t: r.update(next_batch_starts=[1, 1])),
    ],
)
def test_each_divergence_is_detected(category: str, tamper: Any) -> None:
    full, first, resume, state, resumed_state = fake_runs()
    tamper(full, first, resume, state, resumed_state)
    categories = compare_runs(full, first, resume, state, resumed_state, device_type="cpu")
    assert categories[category] != "exact", categories


# --- only the harness may produce resume evidence -------------------------------------------------


def test_only_the_harness_constructs_resume_evidence() -> None:
    offenders = []
    for path in sorted([*ROOT.glob("kittylm/**/*.py"), *ROOT.glob("scripts/*.py")]):
        rel = path.relative_to(ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
                if name == "ResumeEvidence" and rel != "kittylm/training/resume_harness.py":
                    offenders.append(rel)
            if isinstance(node, ast.keyword) and node.arg == "resume_evidence":
                if rel != "kittylm/training/resume_harness.py":
                    offenders.append(rel)
    assert offenders == []


def test_spec_round_trips_through_json(tmp_path: Path) -> None:
    spec = nano_spec(tmp_path)
    assert HarnessSpec(**asdict(spec)) == spec
