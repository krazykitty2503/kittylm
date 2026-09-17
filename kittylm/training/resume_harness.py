"""Deterministic checkpoint/resume validation harness (the only producer of resume evidence).

Purpose:
    Prove, rather than assume, that a checkpoint captures the complete training state. Three
    fresh Python processes run the same configuration:

    - run A trains uninterrupted to ``total_steps`` and saves its final state;
    - run B1 trains to ``kill_step``, writes a step checkpoint and exits (the "kill");
    - run B2 is a new process that resumes from that checkpoint and trains to ``total_steps``.

    The harness then compares A with B1+B2 category by category: global step, scheduler state and
    learning-rate series, every optimizer state tensor, every model parameter, every RNG state
    (Python, NumPy, PyTorch CPU, CUDA), the loader generator and draw count, the per-step loss
    series, the per-step batch window starts, and the window starts of the *next* batch. On CPU
    every category must be bit-identical. On GPUs the counters, schedule, RNG states and data order
    must be identical, while tensor and loss deviations are measured and reported (GPU kernels may
    be nondeterministic). Only this module constructs ``ResumeEvidence`` (D-012).

Public API:
    HarnessSpec(run_id, model, training, token_file, device, kill_step, total_steps, git_commit)
    validate_resume(spec, work_dir, python=sys.executable) -> ResumeReport
    compare_runs(full, first, resume, full_state, resumed_state, device_type) -> dict[str, str]
    ResumeReport(passed, checkpoint_resume_test, categories, exact_required, evidence)
    HarnessError
    ``python -m kittylm.training.resume_harness --worker ...`` (internal worker entry point)

Shapes:
    Compared tensors keep their saved shapes; a shape or structure difference is a mismatch.

Dtype:
    Losses and learning rates are compared as exact float hex strings; tensors with
    ``torch.equal``.

Device:
    Workers run on ``spec.device``; checkpoints are compared on CPU.

Invariants:
    - Every run is a fresh process, so no in-memory state can leak across the "kill".
    - ``passed`` is True only if every category in ``exact_required`` is ``exact``.
    - The evidence's ``metrics_sha256`` hashes all three runs' series and the comparison.

Failure modes:
    - A worker that exits non-zero raises HarnessError (the harness never reports a pass it did
      not observe).
    - A resume that does not land exactly on ``kill_step`` raises HarnessError.

See:
    D-012, plan rev 3.3 section 4, kittylm/ledger.py (ResumeEvidence validation).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from kittylm.config import canonical_json, config_hash, from_dict
from kittylm.data.loader import TrainWindowSampler, open_token_file
from kittylm.ledger import RESUME_HARNESS, ResumeEvidence
from kittylm.model.config import ModelConfig
from kittylm.model.transformer import KittyLM
from kittylm.training.checkpoint import RunIdentity, read_checkpoint
from kittylm.training.config import TrainingConfig
from kittylm.training.determinism import seed_everything
from kittylm.training.engine import RunInfo, TrainingEngine

__all__ = [
    "CPU_EXACT_CATEGORIES",
    "GPU_EXACT_CATEGORIES",
    "HarnessError",
    "HarnessSpec",
    "ResumeReport",
    "compare_runs",
    "validate_resume",
]

HARNESS_TOKENIZER_SHA256 = hashlib.sha256(b"kittylm resume harness: pre-tokenized data").hexdigest()
CPU_EXACT_CATEGORIES: tuple[str, ...] = (
    "global_step",
    "scheduler",
    "learning_rates",
    "optimizer_state",
    "model_parameters",
    "rng_python",
    "rng_numpy",
    "rng_torch",
    "rng_cuda",
    "loader_state",
    "loss_series",
    "batch_indices",
    "next_batch_indices",
)
GPU_EXACT_CATEGORIES: tuple[str, ...] = (
    "global_step",
    "scheduler",
    "learning_rates",
    "rng_python",
    "rng_numpy",
    "rng_torch",
    "rng_cuda",
    "loader_state",
    "batch_indices",
    "next_batch_indices",
)


class HarnessError(RuntimeError):
    """The harness could not run a comparison it can vouch for."""


@dataclass(frozen=True)
class HarnessSpec:
    """Everything a worker needs to rebuild the same run."""

    run_id: str
    model: dict[str, Any]
    training: dict[str, Any]
    token_file: str
    device: str
    kill_step: int
    total_steps: int
    git_commit: str


@dataclass(frozen=True)
class ResumeReport:
    """Outcome of a resume validation."""

    passed: bool
    checkpoint_resume_test: Literal["passed", "failed"]
    categories: dict[str, str]
    exact_required: tuple[str, ...]
    evidence: ResumeEvidence


# ------------------------------------------------------------------------------------ comparison


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key in sorted(value, key=str):
            out.update(_flatten(value[key], f"{prefix}/{key}"))
        return out
    if isinstance(value, list | tuple):
        out = {f"{prefix}#len": len(value)}
        for index, item in enumerate(value):
            out.update(_flatten(item, f"{prefix}[{index}]"))
        return out
    return {prefix: value}


def _compare_values(a: Any, b: Any) -> str:
    """``exact``, ``max_abs_dev=<x>`` (same structure, other numbers) or ``structure differs``."""
    flat_a, flat_b = _flatten(a), _flatten(b)
    if flat_a.keys() != flat_b.keys():
        return "structure differs"
    max_dev = 0.0
    exact = True
    for key, left in flat_a.items():
        right = flat_b[key]
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            if left.shape != right.shape or left.dtype != right.dtype:
                return "structure differs"
            if not torch.equal(left, right):
                exact = False
                if left.is_floating_point():
                    diff = (left.double() - right.double()).abs().max()
                    max_dev = max(max_dev, float(diff))
                else:
                    max_dev = max(max_dev, float("inf"))
        elif isinstance(left, float) and isinstance(right, float):
            if left.hex() != right.hex():
                exact = False
                max_dev = max(max_dev, abs(left - right))
        elif left != right:
            return (
                "structure differs" if type(left) is not type(right) else f"values differ at {key}"
            )
    return "exact" if exact else f"max_abs_dev={max_dev:.3e}"


def _compare_series(full: list[Any], split: list[Any]) -> str:
    if len(full) != len(split):
        return f"length differs ({len(full)} vs {len(split)})"
    if full == split:
        return "exact"
    try:
        dev = max(
            abs(float.fromhex(a) - float.fromhex(b)) for a, b in zip(full, split, strict=True)
        )
        return f"max_abs_dev={dev:.3e}"
    except (TypeError, ValueError):
        first = next(i for i, (a, b) in enumerate(zip(full, split, strict=True)) if a != b)
        return f"differs from step {first + 1}"


def compare_runs(
    full: dict[str, Any],
    first: dict[str, Any],
    resume: dict[str, Any],
    full_state: dict[str, Any],
    resumed_state: dict[str, Any],
    device_type: str,
) -> dict[str, str]:
    """Compare run A with runs B1+B2 in every category (see module docstring)."""
    categories: dict[str, str] = {}
    steps_ok = (
        full["global_step"]
        == resume["global_step"]
        == full_state["global_step"]
        == (resumed_state["global_step"])
    )
    categories["global_step"] = "exact" if steps_ok else "values differ"
    categories["scheduler"] = _compare_values(full_state["scheduler"], resumed_state["scheduler"])
    categories["learning_rates"] = _compare_series(
        full["learning_rates"], first["learning_rates"] + resume["learning_rates"]
    )
    categories["optimizer_state"] = _compare_values(
        full_state["optimizer"], resumed_state["optimizer"]
    )
    categories["model_parameters"] = _compare_values(full_state["model"], resumed_state["model"])
    for source in ("python", "numpy", "torch", "cuda"):
        categories[f"rng_{source}"] = _compare_values(
            full_state["rng"][source], resumed_state["rng"][source]
        )
    categories["loader_state"] = _compare_values(full_state["loader"], resumed_state["loader"])
    categories["loss_series"] = _compare_series(full["losses"], first["losses"] + resume["losses"])
    categories["batch_indices"] = (
        "exact"
        if full["batch_starts"] == first["batch_starts"] + resume["batch_starts"]
        else "values differ"
    )
    categories["next_batch_indices"] = (
        "exact" if full["next_batch_starts"] == resume["next_batch_starts"] else "values differ"
    )
    categories["device_type"] = device_type
    return categories


# ------------------------------------------------------------------------------------ workers


def _dataset_version(token_file: Path) -> str:
    return hashlib.sha256(token_file.read_bytes()).hexdigest()


def run_worker(spec: HarnessSpec, mode: str, run_dir: Path, out_path: Path) -> None:
    """Worker body: ``full`` (A), ``first`` (B1) or ``resume`` (B2)."""
    model_config = from_dict(ModelConfig, spec.model)
    training_config = from_dict(TrainingConfig, spec.training)
    if training_config.cpu_threads:
        torch.set_num_threads(training_config.cpu_threads)
    seed_everything(training_config.seed)
    device = torch.device(spec.device)
    model = KittyLM(model_config)
    token_file = Path(spec.token_file)
    sampler = TrainWindowSampler(
        open_token_file(token_file),
        context_length=model_config.context_length,
        batch_size=training_config.batch_size,
        seed=training_config.seed,
    )
    identity = RunIdentity(
        config_hash=config_hash({"model": spec.model, "training": spec.training}),
        tokenizer_sha256=HARNESS_TOKENIZER_SHA256,
        dataset_version=_dataset_version(token_file),
    )
    run = RunInfo(
        kind="engineering",
        experiment_id=f"resume-{spec.run_id}-{mode}",
        git_commit=spec.git_commit,
        git_dirty=False,
    )
    engine = TrainingEngine(
        model=model,
        model_config=model_config,
        training_config=training_config,
        sampler=sampler,
        device=device,
        run_dir=run_dir,
        identity=identity,
        run=run,
    )
    try:
        if mode == "resume":
            if not engine.resume_latest() or engine.global_step != spec.kill_step:
                raise HarnessError(f"resume did not land on kill_step {spec.kill_step}")
        target = spec.kill_step if mode == "first" else spec.total_steps
        results = engine.train(target)
        if mode == "first":
            engine.save_checkpoint()
        else:
            engine.checkpoints.save_named(engine.state_dict(), "final")
        payload = {
            "mode": mode,
            "global_step": engine.global_step,
            "losses": [r.loss.hex() for r in results],
            "learning_rates": [r.learning_rate.hex() for r in results],
            "batch_starts": [r.batch_starts for r in results],
            "next_batch_starts": engine.sampler.preview_starts(),
        }
        out_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    finally:
        engine.close()


# ------------------------------------------------------------------------------------ harness


def _run_mode(
    python: str, spec_path: Path, mode: str, run_dir: Path, work_dir: Path
) -> dict[str, Any]:
    out_path = work_dir / f"{mode}.json"
    cmd = [
        python,
        "-m",
        "kittylm.training.resume_harness",
        "--worker",
        "--spec",
        str(spec_path),
        "--mode",
        mode,
        "--run-dir",
        str(run_dir),
        "--out",
        str(out_path),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False
    )
    if result.returncode != 0 or not out_path.exists():
        tail = "\n".join(result.stderr.strip().splitlines()[-8:])
        raise HarnessError(f"{mode} worker failed (exit {result.returncode}):\n{tail}")
    data: dict[str, Any] = json.loads(out_path.read_text(encoding="utf-8"))
    return data


def validate_resume(
    spec: HarnessSpec, work_dir: Path, python: str = sys.executable
) -> ResumeReport:
    """Run A, B1 and B2 in fresh processes and compare them (see module docstring)."""
    if not 0 < spec.kill_step < spec.total_steps:
        raise HarnessError("require 0 < kill_step < total_steps")
    work_dir.mkdir(parents=True, exist_ok=True)
    spec_path = work_dir / "spec.json"
    spec_path.write_text(json.dumps(asdict(spec), sort_keys=True), encoding="utf-8")
    full_dir, split_dir = work_dir / "full", work_dir / "split"

    full = _run_mode(python, spec_path, "full", full_dir, work_dir)
    first = _run_mode(python, spec_path, "first", split_dir, work_dir)
    resume = _run_mode(python, spec_path, "resume", split_dir, work_dir)

    full_state = read_checkpoint(full_dir / "checkpoints" / "final.pt")
    resumed_state = read_checkpoint(split_dir / "checkpoints" / "final.pt")
    device_type = torch.device(spec.device).type
    categories = compare_runs(full, first, resume, full_state, resumed_state, device_type)
    exact_required = GPU_EXACT_CATEGORIES if device_type == "cuda" else CPU_EXACT_CATEGORIES
    passed = all(categories[name] == "exact" for name in exact_required)

    digest = hashlib.sha256(
        canonical_json(
            {"full": full, "first": first, "resume": resume, "comparison": categories}
        ).encode("utf-8")
    ).hexdigest()
    evidence = ResumeEvidence(
        harness=RESUME_HARNESS,
        run_id=spec.run_id,
        kill_step=spec.kill_step,
        resumed_to_step=spec.total_steps,
        metrics_sha256=digest,
        comparison=dict(categories),
    )
    return ResumeReport(
        passed=passed,
        checkpoint_resume_test="passed" if passed else "failed",
        categories=categories,
        exact_required=exact_required,
        evidence=evidence,
    )


def main(argv: list[str] | None = None) -> int:
    """Worker entry point (``--worker``); the harness itself is ``validate_resume``."""
    parser = argparse.ArgumentParser(description="KittyLM resume harness worker")
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--mode", choices=["full", "first", "resume"], required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    spec = HarnessSpec(**json.loads(args.spec.read_text(encoding="utf-8")))
    run_worker(spec, args.mode, args.run_dir, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
