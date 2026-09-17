"""GPU checks for Milestone C (run locally with `pytest -m gpu`; never in CI).

bf16 training on the ROCm device, and the fresh-process resume harness on the GPU, where the
plan requires exact counters, scheduler, RNG and loader state and *measured* (not assumed)
tensor and loss deviations.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest

from kittylm.config import to_dict
from kittylm.data.loader import TrainWindowSampler, open_token_file
from kittylm.model.transformer import KittyLM
from kittylm.training.determinism import seed_everything
from kittylm.training.engine import RunInfo, TrainingEngine
from kittylm.training.resume_harness import GPU_EXACT_CATEGORIES, HarnessSpec, validate_resume
from tests.test_model_gpu import model_config, require_gpu, selected
from tests.training_helpers import COMMIT, TINY_TRAINING, identity_for, write_tokens

pytestmark = pytest.mark.gpu

GPU_TRAINING = replace(
    TINY_TRAINING,
    batch_size=2,
    gradient_accumulation=1,
    max_steps=8,
    warmup_steps=2,
    precision="bf16",
    determinism="deterministic",
    log_every=0,
    timing_sync_every=1,
    cpu_threads=0,
)


def test_bf16_overfit_on_gpu(tmp_path: Path) -> None:
    device = require_gpu()
    selected()  # the attention path and flag must match D-014
    nano = model_config("nano")
    token_file = write_tokens(
        tmp_path / "one.bin", nano.context_length + 1, nano.vocab_size, seed=5
    )
    training = replace(
        GPU_TRAINING,
        batch_size=1,
        max_steps=150,
        warmup_steps=10,
        learning_rate=3e-3,
        weight_decay=0.0,
    )
    seed_everything(training.seed)
    engine = TrainingEngine(
        model=KittyLM(nano).to(device),
        model_config=nano,
        training_config=training,
        sampler=TrainWindowSampler(
            open_token_file(token_file),
            context_length=nano.context_length,
            batch_size=1,
            seed=training.seed,
        ),
        device=device,
        run_dir=tmp_path / "gpu-overfit",
        identity=identity_for(nano, training, token_file),
        run=RunInfo(
            kind="engineering", experiment_id="gpu-overfit", git_commit=COMMIT, git_dirty=False
        ),
    )
    results = engine.train(150)
    summary = engine.summary()
    engine.close()
    assert all(math.isfinite(r.loss) and math.isfinite(r.grad_norm) for r in results)
    assert results[-1].loss < results[0].loss / 5, (results[0].loss, results[-1].loss)
    assert set(summary["timing_breakdown"]) == {"data", "forward", "backward", "optimizer"}
    assert summary["peak_vram_gb"] is not None


def test_gpu_fresh_process_resume(tmp_path: Path) -> None:
    require_gpu()
    selected()
    nano = model_config("nano")
    token_file = write_tokens(tmp_path / "tokens.bin", 20_000, nano.vocab_size, seed=11)
    spec = HarnessSpec(
        run_id="test-gpu-nano",
        model=to_dict(nano),
        training=to_dict(GPU_TRAINING),
        token_file=str(token_file),
        device="cuda",
        kill_step=4,
        total_steps=8,
        git_commit=COMMIT,
    )
    report = validate_resume(spec, tmp_path / "work")
    for category in GPU_EXACT_CATEGORIES:
        assert report.categories[category] == "exact", (category, report.categories)
    # Tensors and losses: exact, or a measured deviation — never missing.
    for category in ("optimizer_state", "model_parameters", "loss_series"):
        verdict = report.categories[category]
        assert verdict == "exact" or verdict.startswith("max_abs_dev="), (category, verdict)
    assert report.passed and report.categories["device_type"] == "cuda"
