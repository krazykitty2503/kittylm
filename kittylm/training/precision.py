"""Mixed-precision handling: bf16 autocast, fp16 with loss scaling, or plain fp32.

Purpose:
    Training in 16-bit halves activation memory and speeds up matrix multiplications, but
    numerical range differs. bf16 has float32's range, so it needs no loss scaling. fp16 has a
    small range, so gradients are multiplied by a dynamic scale before backward and divided
    afterwards (``GradScaler``); steps whose gradients overflow are skipped. Master weights and
    optimizer state always stay in float32.

Public API:
    Precision(name, device_type, autocast_dtype, scaler)
        ``autocast()`` context manager.
    make_precision(name, device) -> Precision

Shapes:
    Not applicable.

Dtype:
    ``bf16`` -> autocast ``torch.bfloat16``; ``fp16`` -> autocast ``torch.float16`` + GradScaler;
    ``fp32`` -> no autocast.

Device:
    bf16 works on CPU and supported GPUs; fp16 is GPU-only.

Invariants:
    - Parameters and optimizer state are never cast; only autocast-eligible operations run in
      16-bit.
    - A scaler exists if and only if precision is fp16.

Failure modes:
    - fp16 on CPU, or bf16 on a GPU without bf16 support, raises ValueError.

See:
    Plan rev 3.3 section 4, D-014 (attention runs in bf16).
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch.amp import GradScaler

__all__ = ["Precision", "make_precision"]


@dataclass
class Precision:
    """Resolved precision policy for one device."""

    name: str
    device_type: str
    autocast_dtype: torch.dtype | None
    scaler: GradScaler | None

    @contextlib.contextmanager
    def autocast(self) -> Iterator[None]:
        """Autocast context for forward passes (no-op for fp32)."""
        if self.autocast_dtype is None:
            yield
        else:
            with torch.autocast(device_type=self.device_type, dtype=self.autocast_dtype):
                yield


def make_precision(name: str, device: torch.device) -> Precision:
    """Build the precision policy for ``name`` on ``device``."""
    device_type = device.type
    if name == "fp32":
        return Precision(name, device_type, None, None)
    if name == "bf16":
        if device_type == "cuda" and not torch.cuda.is_bf16_supported():
            raise ValueError("bf16 requested but this GPU does not support bfloat16")
        return Precision(name, device_type, torch.bfloat16, None)
    if name == "fp16":
        if device_type != "cuda":
            raise ValueError("fp16 training requires a GPU (use bf16 or fp32 on CPU)")
        return Precision(name, device_type, torch.float16, GradScaler("cuda"))
    raise ValueError(f"unknown precision {name!r}")
