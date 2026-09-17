"""Experiment logger: standard logging plus a machine-readable metrics stream.

Purpose:
    Every training run writes human-readable progress to ``train.log`` and one JSON object per
    logged step to ``metrics.jsonl``. The metrics stream is what summaries and later analysis are
    computed from; it is not a separate monitoring system (plan rev 3.3 section 5).

Public API:
    ExperimentLogger(run_dir, run_name)
        ``log_metrics(step, metrics)``, ``info(message)``, ``close()``; ``metrics_path``.

Shapes:
    Not applicable.

Dtype:
    Metric values are Python numbers; tensors are converted by the caller.

Device:
    Not applicable (host-side files).

Invariants:
    - Each line of ``metrics.jsonl`` is valid JSON with sorted keys; non-finite floats are written
      as the strings ``"nan"``, ``"inf"`` or ``"-inf"`` (JSON has no NaN).
    - Log files live in the run directory under ``runs/``, which is never committed.
    - The logger never writes environment variables, hostnames or credentials.

Failure modes:
    - An unwritable run directory raises OSError at construction.

See:
    Plan rev 3.3 section 5.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

__all__ = ["ExperimentLogger"]


def _jsonable(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    return value


class ExperimentLogger:
    """Writes ``train.log`` and ``metrics.jsonl`` inside ``run_dir``."""

    def __init__(self, run_dir: Path, run_name: str) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = run_dir / "metrics.jsonl"
        self._logger = logging.getLogger(f"kittylm.training.{run_name}.{id(self)}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        self._handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
        self._handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        self._logger.addHandler(self._handler)

    def info(self, message: str) -> None:
        """Log a human-readable message."""
        self._logger.info(message)

    def log_metrics(self, step: int, metrics: dict[str, Any]) -> None:
        """Append one metrics record for ``step``."""
        record = _jsonable({"step": step, **metrics})
        with self.metrics_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        summary = " ".join(
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in record.items()
            if not isinstance(v, dict)
        )
        self._logger.info(summary)

    def close(self) -> None:
        """Detach and close the file handler."""
        self._logger.removeHandler(self._handler)
        self._handler.close()
