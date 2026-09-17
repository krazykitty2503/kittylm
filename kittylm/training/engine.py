"""TrainingEngine: the optimization loop, its state, and its safety gates.

Purpose:
    Run optimizer steps on a model and keep *everything* needed to continue the run exactly:
    model weights, optimizer moments, schedule step, loss scaler, counters, every RNG state and
    the data sampler's generator. One optimizer step draws ``gradient_accumulation``
    micro-batches, averages their losses, backpropagates, clips the global gradient norm, and
    updates the weights at the scheduled learning rate. The engine also refuses to start a
    formal experiment without green required CI on the exact commit (D-018), refuses a model
    whose parameter accounting does not add up, and stops with a ``crash`` checkpoint on
    non-finite loss or gradients instead of silently continuing.

Public API:
    RunInfo(kind, experiment_id, git_commit, git_dirty, ci_evidence)
    TrainingEngine(model, model_config, training_config, sampler, device, run_dir, identity, run)
        ``train_step()``, ``train(until_step)``, ``validate(batches)``, ``state_dict()``,
        ``load_state_dict(state)``, ``save_checkpoint()``, ``resume_latest()``, ``summary()``;
        ``global_step``, ``tokens_seen``, ``history``.
    StepResult, NonFiniteError, CiGateError

Shapes:
    Micro-batches ``[batch_size, context_length]``; logits ``[B, T, vocab]``.

Dtype:
    Master weights and optimizer state float32; forward under the configured autocast
    (bf16 by default, D-014); loss computed in float32.

Device:
    Model, batches and loss on ``device``; checkpoints are saved and loaded on CPU and moved.

Math:
    ``loss_step = (1/A) * sum_a CE(micro_a)``; ``g <- clip(g, grad_clip)`` by global L2 norm;
    AdamW update with ``lr = schedule(step)``.

Invariants:
    - ``global_step`` equals the schedule step and the number of completed optimizer steps.
    - ``tokens_seen = global_step * batch_size * gradient_accumulation * context_length``.
    - Resuming from ``state_dict()`` in a fresh process continues bit-exactly on CPU in
      deterministic mode (validated by ``resume_harness``).
    - Checkpoint metadata contains only whitelisted identity fields.

Failure modes:
    - ``CiGateError``: formal run without valid ``ci_evidence`` for ``git_commit``.
    - ``AccountingError``: parameter categories do not sum to the total.
    - ``NonFiniteError``: non-finite loss or gradient norm (fp32/bf16; with fp16 the loss
      scaler skips overflowing steps instead). A ``crash`` checkpoint holding the pre-update state
      is written first.
    - ``CheckpointError``: a resume checkpoint is damaged or belongs to another run.

See:
    D-012, D-014, D-018, plan rev 3.3 section 4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from kittylm.config import to_dict
from kittylm.data.loader import Batch, TrainWindowSampler
from kittylm.ledger import CiEvidence, ci_evidence_problems
from kittylm.model.accounting import count_parameters
from kittylm.model.config import ModelConfig
from kittylm.training.checkpoint import CheckpointManager, RunIdentity
from kittylm.training.config import TrainingConfig
from kittylm.training.determinism import (
    capture_rng_state,
    configure_determinism,
    restore_rng_state,
)
from kittylm.training.logger import ExperimentLogger
from kittylm.training.loss import causal_lm_loss
from kittylm.training.optim import build_optimizer
from kittylm.training.precision import make_precision
from kittylm.training.schedule import WarmupCosineSchedule
from kittylm.training.timing import StepTimer, ThroughputMeter

__all__ = ["CiGateError", "NonFiniteError", "RunInfo", "StepResult", "TrainingEngine"]


class NonFiniteError(RuntimeError):
    """Loss or gradient norm became NaN/Inf."""


class CiGateError(RuntimeError):
    """A formal experiment was started without green required CI on its exact commit."""


@dataclass(frozen=True)
class RunInfo:
    """What kind of run this is and which code it runs."""

    kind: Literal["formal", "smoke", "engineering"]
    experiment_id: str
    git_commit: str
    git_dirty: bool
    ci_evidence: CiEvidence | None = None


@dataclass(frozen=True)
class StepResult:
    """Outcome of one optimizer step."""

    step: int
    loss: float
    learning_rate: float
    grad_norm: float
    tokens_seen: int
    batch_starts: list[list[int]] = field(default_factory=list)


class TrainingEngine:
    """Optimizer-step loop with exact, checkpointable state."""

    def __init__(
        self,
        *,
        model: nn.Module,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        sampler: TrainWindowSampler,
        device: torch.device,
        run_dir: Path,
        identity: RunIdentity,
        run: RunInfo,
    ) -> None:
        if run.kind == "formal":
            problems = ci_evidence_problems(run.ci_evidence, run.git_commit)
            if problems:
                raise CiGateError("formal run refused: " + "; ".join(problems))
            if run.git_dirty:
                raise CiGateError("formal run refused: working tree has uncommitted changes")
        if sampler.context_length > model_config.context_length:
            raise ValueError("sampler context_length exceeds the model context_length")
        self.parameter_counts = count_parameters(model)  # raises AccountingError if inconsistent

        self.config = training_config
        self.model_config = model_config
        self.model = model.to(device)
        self.sampler = sampler
        self.device = device
        self.identity = identity
        self.run = run
        if training_config.cpu_threads:
            torch.set_num_threads(training_config.cpu_threads)
        configure_determinism(training_config.determinism)
        self.precision = make_precision(training_config.precision, device)
        self.optimizer = build_optimizer(self.model, training_config)
        self.schedule = WarmupCosineSchedule(
            self.optimizer,
            max_lr=training_config.learning_rate,
            min_lr=training_config.min_learning_rate,
            warmup_steps=training_config.warmup_steps,
            max_steps=training_config.max_steps,
        )
        self.checkpoints = CheckpointManager(run_dir / "checkpoints", training_config.keep_last)
        self.logger = ExperimentLogger(run_dir, run.experiment_id)
        self.timer = StepTimer(device, training_config.timing_sync_every)
        self.throughput = ThroughputMeter(device)
        self.global_step = 0
        self.tokens_seen = 0
        self.history: list[StepResult] = []
        self.best_val_loss = math.inf
        self._tokens_since_lap = 0
        self._samples_since_lap = 0
        self._throughput_samples: list[tuple[float, float]] = []
        self._throughput_started = False

    # ------------------------------------------------------------------ state

    def _metadata(self) -> dict[str, Any]:
        return {
            "config_hash": self.identity.config_hash,
            "tokenizer_sha256": self.identity.tokenizer_sha256,
            "dataset_version": self.identity.dataset_version,
            "git_commit": self.run.git_commit,
            "git_dirty": self.run.git_dirty,
            "precision": self.config.precision,
            "device_type": self.device.type,
            "determinism": self.config.determinism,
        }

    def state_dict(self) -> dict[str, Any]:
        """Complete resumable state (CPU tensors)."""
        scaler = self.precision.scaler
        return {
            "metadata": self._metadata(),
            "model": {k: v.detach().to("cpu") for k, v in self.model.state_dict().items()},
            "optimizer": _to_cpu(self.optimizer.state_dict()),
            "scheduler": self.schedule.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "global_step": self.global_step,
            "tokens_seen": self.tokens_seen,
            "rng": capture_rng_state(include_cuda=self.device.type == "cuda"),
            "loader": self.sampler.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore state produced by ``state_dict`` (identity must already be verified)."""
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.schedule.load_state_dict(state["scheduler"])
        if self.precision.scaler is not None and state["scaler"] is not None:
            self.precision.scaler.load_state_dict(state["scaler"])
        self.global_step = int(state["global_step"])
        self.tokens_seen = int(state["tokens_seen"])
        if self.schedule.step != self.global_step:
            raise ValueError("checkpoint scheduler step does not match global_step")
        self.sampler.load_state_dict(state["loader"])
        restore_rng_state(state["rng"])

    def save_checkpoint(self) -> Path:
        """Write a step checkpoint and update ``latest.json``."""
        return self.checkpoints.save_step(self.state_dict(), self.global_step)

    def resume_latest(self) -> bool:
        """Resume from ``latest.json`` if present; refuses checkpoints of another run."""
        state = self.checkpoints.load_latest(self.identity)
        if state is None:
            return False
        self.load_state_dict(state)
        self.logger.info(f"resumed at step {self.global_step}")
        return True

    # ------------------------------------------------------------------ training

    def _to_device(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        return batch.inputs.to(self.device), batch.targets.to(self.device)

    def train_step(self) -> StepResult:
        """Run one optimizer step (``gradient_accumulation`` micro-batches)."""
        cfg = self.config
        if not self._throughput_started:
            self.throughput.start()
            self._throughput_started = True
        self.model.train()
        self.timer.begin_step(self.global_step)
        lr = self.schedule.apply()
        self.optimizer.zero_grad(set_to_none=True)
        scaler = self.precision.scaler
        loss_total = 0.0
        starts: list[list[int]] = []
        for _ in range(cfg.gradient_accumulation):
            with self.timer.phase("data"):
                batch = self.sampler.next_batch()
                inputs, targets = self._to_device(batch)
                starts.append(batch.starts)
            with self.timer.phase("forward"):
                with self.precision.autocast():
                    logits = self.model(inputs)
                loss = causal_lm_loss(logits, targets) / cfg.gradient_accumulation
            with self.timer.phase("backward"):
                (scaler.scale(loss) if scaler is not None else loss).backward()
            loss_total += float(loss.detach())

        with self.timer.phase("optimizer"):
            if scaler is not None:
                scaler.unscale_(self.optimizer)
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            )
            if not math.isfinite(loss_total) or (scaler is None and not math.isfinite(grad_norm)):
                self.checkpoints.save_named(self.state_dict(), "crash")
                self.logger.info(
                    f"non-finite values at step {self.global_step}: loss={loss_total} "
                    f"grad_norm={grad_norm}; wrote crash checkpoint"
                )
                raise NonFiniteError(
                    f"non-finite loss or gradient norm at step {self.global_step} "
                    f"(loss={loss_total}, grad_norm={grad_norm})"
                )
            if scaler is not None:
                scaler.step(self.optimizer)
                scaler.update()
            else:
                self.optimizer.step()

        self.schedule.advance()
        self.global_step += 1
        tokens = cfg.batch_size * cfg.gradient_accumulation * self.sampler.context_length
        self.tokens_seen += tokens
        self._tokens_since_lap += tokens
        self._samples_since_lap += cfg.batch_size * cfg.gradient_accumulation
        result = StepResult(self.global_step, loss_total, lr, grad_norm, self.tokens_seen, starts)
        self.history.append(result)

        if cfg.log_every and self.global_step % cfg.log_every == 0:
            self._log(result)
        if cfg.checkpoint_every and self.global_step % cfg.checkpoint_every == 0:
            self.save_checkpoint()
        return result

    def _log(self, result: StepResult) -> None:
        tokens_per_s, samples_per_s = self.throughput.lap(
            self._tokens_since_lap, self._samples_since_lap
        )
        self._tokens_since_lap = self._samples_since_lap = 0
        self._throughput_samples.append((tokens_per_s, samples_per_s))
        metrics: dict[str, Any] = {
            "loss": result.loss,
            "ppl": math.exp(result.loss) if result.loss < 700 else math.inf,
            "lr": result.learning_rate,
            "grad_norm": result.grad_norm,
            "tokens_seen": result.tokens_seen,
            "tokens_per_second": tokens_per_s,
            "samples_per_second": samples_per_s,
        }
        if self.device.type == "cuda":
            metrics["allocated_vram_mib"] = torch.cuda.memory_allocated(self.device) / 2**20
            metrics["peak_vram_mib"] = torch.cuda.max_memory_allocated(self.device) / 2**20
        self.logger.log_metrics(result.step, metrics)

    def train(self, until_step: int) -> list[StepResult]:
        """Train until ``global_step == until_step`` (at most ``max_steps``)."""
        target = min(until_step, self.config.max_steps)
        results = []
        while self.global_step < target:
            results.append(self.train_step())
        return results

    @torch.no_grad()
    def validate(self, batches: list[Batch]) -> float:
        """Mean validation loss over fixed batches; saves ``best_val`` on improvement."""
        if not batches:
            raise ValueError("validation needs at least one batch")
        self.model.eval()
        total = 0.0
        for batch in batches:
            inputs, targets = self._to_device(batch)
            with self.precision.autocast():
                logits = self.model(inputs)
            total += float(causal_lm_loss(logits, targets))
        loss = total / len(batches)
        self.logger.log_metrics(self.global_step, {"val_loss": loss})
        if loss < self.best_val_loss:
            self.best_val_loss = loss
            self.checkpoints.save_named(self.state_dict(), "best_val")
        return loss

    def summary(self) -> dict[str, Any]:
        """Run summary: parameter budget, compute consumed, timing and throughput."""
        measured = self._throughput_samples[1:]  # the first window includes warmup
        out: dict[str, Any] = {
            "parameters": self.parameter_counts.as_dict(),
            "global_step": self.global_step,
            "tokens_seen": self.tokens_seen,
            "timing_breakdown": self.timer.summary(),
            "tokens_per_second": (
                sum(t for t, _ in measured) / len(measured) if measured else None
            ),
            "samples_per_second": (
                sum(s for _, s in measured) / len(measured) if measured else None
            ),
            "config": {"model": to_dict(self.model_config), "training": to_dict(self.config)},
        }
        if self.device.type == "cuda":
            out["peak_vram_gb"] = torch.cuda.max_memory_allocated(self.device) / 2**30
        return out

    def close(self) -> None:
        """Release the logger's file handles."""
        self.logger.close()


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu")
    if isinstance(value, dict):
        return {k: _to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(v) for v in value)
    return value
