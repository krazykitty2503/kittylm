"""Token-window loading: random training windows and fixed validation windows.

Purpose:
    Turn a packed token stream (one flat array of token ids, documents separated by
    ``<|endoftext|>``) into training batches. Training draws random windows so every step sees a
    different slice of the corpus; the random generator is explicit and its state is saved in
    checkpoints, so a resumed run draws exactly the batches an uninterrupted run would have
    drawn. Validation uses fixed, non-overlapping windows so every evaluation measures the same
    tokens.

Public API:
    TOKEN_DTYPE, open_token_file(path) -> np.memmap
    Batch(inputs, targets, starts)
    TrainWindowSampler(tokens, context_length, batch_size, seed)
        ``next_batch()``, ``preview_starts()`` (next batch's window starts without advancing),
        ``state_dict()`` / ``load_state_dict()``, ``batches_drawn``.
    validation_batches(tokens, context_length, batch_size, max_batches=None) -> list[Batch]

Shapes:
    ``inputs`` and ``targets`` are ``[batch_size, context_length]``; ``targets`` is ``inputs``
    shifted left by one token (each window reads ``context_length + 1`` tokens).

Dtype:
    Token files are ``uint16`` (vocabularies up to 65,536); batches are ``int64`` for embedding
    lookup and cross-entropy.

Device:
    Batches are built on CPU; the caller moves them to the training device. The generator is a
    CPU ``torch.Generator``, so window order is identical on CPU and GPU runs.

Math:
    Window starts are drawn uniformly from ``[0, N - context_length - 1]`` with replacement.

Invariants:
    - Same tokens, context length, batch size and seed -> same sequence of batches.
    - ``preview_starts()`` never changes the sampler state.
    - ``load_state_dict`` refuses state from a sampler with a different context length, batch
      size or token count, so a checkpoint cannot silently resume on different data.

Failure modes:
    - A token stream shorter than ``context_length + 1`` raises ValueError.
    - A mismatched sampler state raises ValueError.

See:
    Plan rev 3.3 section 4, D-005 (split by document), D-012 (resume evidence).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

__all__ = [
    "TOKEN_DTYPE",
    "Batch",
    "TrainWindowSampler",
    "open_token_file",
    "validation_batches",
]

TOKEN_DTYPE = np.uint16


def open_token_file(path: Path) -> np.ndarray:
    """Memory-map a packed ``uint16`` token file read-only."""
    return np.memmap(path, dtype=TOKEN_DTYPE, mode="r")


@dataclass(frozen=True)
class Batch:
    """One batch of input/target windows and the window start offsets that produced it."""

    inputs: torch.Tensor
    targets: torch.Tensor
    starts: list[int]


def _windows(tokens: np.ndarray, starts: list[int], context_length: int) -> Batch:
    rows = np.stack([np.asarray(tokens[s : s + context_length + 1]) for s in starts])
    data = torch.from_numpy(rows.astype(np.int64))
    return Batch(inputs=data[:, :-1].contiguous(), targets=data[:, 1:].contiguous(), starts=starts)


class TrainWindowSampler:
    """Random training windows with a checkpointable generator."""

    def __init__(
        self, tokens: np.ndarray, *, context_length: int, batch_size: int, seed: int
    ) -> None:
        if context_length < 1 or batch_size < 1:
            raise ValueError("context_length and batch_size must be >= 1")
        if len(tokens) < context_length + 1:
            raise ValueError(
                f"token stream of {len(tokens)} tokens is shorter than context_length + 1"
            )
        self.tokens = tokens
        self.context_length = context_length
        self.batch_size = batch_size
        self.seed = seed
        self.max_start = len(tokens) - context_length - 1
        self.generator = torch.Generator().manual_seed(seed)
        self.batches_drawn = 0

    def _draw(self, generator: torch.Generator) -> list[int]:
        starts = torch.randint(0, self.max_start + 1, (self.batch_size,), generator=generator)
        return [int(s) for s in starts]

    def next_batch(self) -> Batch:
        """Draw the next random batch and advance the generator."""
        starts = self._draw(self.generator)
        self.batches_drawn += 1
        return _windows(self.tokens, starts, self.context_length)

    def preview_starts(self) -> list[int]:
        """Window starts of the next batch, without advancing the sampler."""
        clone = torch.Generator()
        clone.set_state(self.generator.get_state())
        return self._draw(clone)

    def state_dict(self) -> dict[str, Any]:
        """Checkpointable sampler state."""
        return {
            "generator": self.generator.get_state(),
            "batches_drawn": self.batches_drawn,
            "seed": self.seed,
            "context_length": self.context_length,
            "batch_size": self.batch_size,
            "num_tokens": len(self.tokens),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore sampler state saved by ``state_dict`` (structure must match)."""
        for key, expected in (
            ("context_length", self.context_length),
            ("batch_size", self.batch_size),
            ("num_tokens", len(self.tokens)),
        ):
            if state.get(key) != expected:
                raise ValueError(
                    f"sampler state {key}={state.get(key)!r} does not match this sampler "
                    f"({expected!r})"
                )
        self.generator.set_state(state["generator"])
        self.batches_drawn = int(state["batches_drawn"])
        self.seed = int(state["seed"])


def validation_batches(
    tokens: np.ndarray, *, context_length: int, batch_size: int, max_batches: int | None = None
) -> list[Batch]:
    """Fixed, non-overlapping validation windows (the last partial window is dropped)."""
    if len(tokens) < context_length + 1:
        raise ValueError("validation stream is shorter than context_length + 1")
    starts = list(range(0, len(tokens) - context_length, context_length))
    batches = [
        _windows(tokens, starts[i : i + batch_size], context_length)
        for i in range(0, len(starts), batch_size)
    ]
    return batches[:max_batches] if max_batches is not None else batches
