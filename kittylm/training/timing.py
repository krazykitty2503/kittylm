"""Phase timing and throughput that account for asynchronous GPU execution.

Purpose:
    GPU operations are queued and return immediately, so a Python timer around ``loss.backward()``
    measures how long it took to *enqueue* work, not to do it. Honest timings require
    synchronizing the device at phase boundaries, but synchronizing every step slows training.
    ``StepTimer`` therefore synchronizes and records phase durations only on every
    ``sync_every``-th step. ``ThroughputMeter`` synchronizes at window boundaries to measure
    tokens/s and samples/s over many steps.

Public API:
    StepTimer(device, sync_every)
        ``begin_step(step)``, ``phase(name)`` context manager, ``summary()``.
    ThroughputMeter(device)
        ``start()``, ``lap(tokens, samples) -> (tokens_per_s, samples_per_s)``.

Shapes:
    Not applicable.

Dtype:
    Durations in milliseconds (float).

Device:
    ``torch.cuda.synchronize`` on CUDA/ROCm devices; no-op on CPU.

Invariants:
    - Phase samples are only recorded on synchronized steps, so no recorded duration measures
      queueing instead of computation.
    - ``summary()`` reports mean, p50 and p95 per phase.

Failure modes:
    - ``sync_every = 0`` disables phase timing (summary is empty).

See:
    Plan rev 3.3 section 5 (performance monitoring).
"""

from __future__ import annotations

import contextlib
import statistics
import time
from collections.abc import Iterator

import torch

__all__ = ["StepTimer", "ThroughputMeter"]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class StepTimer:
    """Records per-phase durations on synchronized steps."""

    def __init__(self, device: torch.device, sync_every: int) -> None:
        self.device = device
        self.sync_every = sync_every
        self.samples: dict[str, list[float]] = {}
        self._active = False

    def begin_step(self, step: int) -> None:
        """Decide whether this step's phases are synchronized and recorded."""
        self._active = self.sync_every > 0 and step % self.sync_every == 0

    @contextlib.contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Time a phase (recorded only on synchronized steps)."""
        if not self._active:
            yield
            return
        _sync(self.device)
        start = time.perf_counter()
        try:
            yield
        finally:
            _sync(self.device)
            self.samples.setdefault(name, []).append((time.perf_counter() - start) * 1000.0)

    def summary(self) -> dict[str, dict[str, float]]:
        """Mean / p50 / p95 milliseconds per phase."""
        out: dict[str, dict[str, float]] = {}
        for name, values in self.samples.items():
            ordered = sorted(values)
            p95_index = min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))
            out[name] = {
                "mean_ms": statistics.fmean(ordered),
                "p50_ms": statistics.median(ordered),
                "p95_ms": ordered[p95_index],
            }
        return out


class ThroughputMeter:
    """Synchronized wall-clock throughput over windows of steps."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._start: float | None = None

    def start(self) -> None:
        """Begin a measurement window."""
        _sync(self.device)
        self._start = time.perf_counter()

    def lap(self, tokens: int, samples: int) -> tuple[float, float]:
        """Close the window, return (tokens/s, samples/s), and start the next window."""
        if self._start is None:
            raise RuntimeError("ThroughputMeter.start() was not called")
        _sync(self.device)
        now = time.perf_counter()
        elapsed = max(now - self._start, 1e-9)
        self._start = now
        return tokens / elapsed, samples / elapsed
