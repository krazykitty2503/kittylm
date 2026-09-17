"""Seeds, determinism modes and checkpointable random-number-generator state.

Purpose:
    A resumed run can only reproduce an uninterrupted run if every random source continues from
    exactly where it stopped. This module seeds all generators, switches PyTorch between
    deterministic and faster modes, and captures/restores the state of every generator in a form
    that ``torch.load(weights_only=True)`` accepts (tensors, ints, floats, lists, None).

Public API:
    seed_everything(seed)
    configure_determinism(mode)
    capture_rng_state(include_cuda) -> dict
    restore_rng_state(state)

Shapes:
    RNG states are 1-D ``uint8``/``int64`` tensors (PyTorch, NumPy) or lists (Python).

Dtype:
    Not applicable to model tensors.

Device:
    CPU generator state always; CUDA/ROCm generator state per device when ``include_cuda``.

Invariants:
    - ``restore_rng_state(capture_rng_state(...))`` makes every generator produce the same
      subsequent values.
    - ``deterministic`` mode enables ``torch.use_deterministic_algorithms`` (errors instead of
      silently nondeterministic kernels) and disables cuDNN benchmarking.

Failure modes:
    - In ``deterministic`` mode an operation without a deterministic implementation raises at
      runtime.
    - Restoring CUDA state on a machine with a different device count raises ValueError.
    - On ROCm, deterministic mode fixes seeds and data order but GPU kernels may still differ
      between runs ("semi-deterministic"); only CPU runs are guaranteed bit-exact.

See:
    D-012, plan rev 3.3 section 4.
"""

from __future__ import annotations

import os
import random
from typing import Any, Literal, cast

import numpy as np
import torch

__all__ = ["capture_rng_state", "configure_determinism", "restore_rng_state", "seed_everything"]

DeterminismMode = Literal["deterministic", "semi_deterministic", "nondeterministic"]


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and all GPUs)."""
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_determinism(mode: DeterminismMode) -> None:
    """Configure PyTorch determinism for ``mode``."""
    if mode == "deterministic":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif mode == "semi_deterministic":
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
    elif mode == "nondeterministic":
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = True
    else:
        raise ValueError(f"unknown determinism mode {mode!r}")


def capture_rng_state(include_cuda: bool) -> dict[str, Any]:
    """Capture every random generator's state in a ``weights_only``-loadable form."""
    version, internal, gauss = random.getstate()
    legacy_state = cast(tuple[Any, ...], np.random.get_state(legacy=True))
    kind, keys, pos, has_gauss, cached = legacy_state
    state: dict[str, Any] = {
        "python": {"version": version, "state": list(internal), "gauss": gauss},
        "numpy": {
            "kind": kind,
            "keys": torch.from_numpy(np.asarray(keys, dtype=np.int64)),
            "pos": int(pos),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached),
        },
        "torch": torch.get_rng_state(),
        "cuda": [s for s in torch.cuda.get_rng_state_all()] if include_cuda else None,
    }
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore generator states captured by ``capture_rng_state``."""
    py = state["python"]
    random.setstate((py["version"], tuple(py["state"]), py["gauss"]))
    npy = state["numpy"]
    np.random.set_state(
        (
            npy["kind"],
            npy["keys"].numpy().astype(np.uint32),
            npy["pos"],
            npy["has_gauss"],
            npy["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch"])
    cuda_states = state.get("cuda")
    if cuda_states is not None:
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError("checkpoint CUDA RNG state does not match the number of devices")
        torch.cuda.set_rng_state_all(cuda_states)
