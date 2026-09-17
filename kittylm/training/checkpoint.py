"""Checkpoints: atomic, checksummed, safely loaded, and bound to their run identity.

Purpose:
    A checkpoint must let a fresh process continue a run *exactly*, and a damaged or mismatched
    checkpoint must never be loaded silently. Each file is written to a temporary name, flushed
    and fsynced, then atomically renamed; it starts with a JSON header holding the payload's
    sha256 and size, which are verified before unpickling. Loading uses
    ``torch.load(weights_only=True)``, which refuses arbitrary Python objects. A ``latest.json``
    pointer is only updated after the checkpoint it names is complete. Metadata binds the
    checkpoint to the configuration, tokenizer and dataset it was trained with.

Public API:
    CHECKPOINT_KEYS, METADATA_KEYS
    RunIdentity(config_hash, tokenizer_sha256, dataset_version)
    CheckpointError
    write_checkpoint(path, state) -> payload sha256
    read_checkpoint(path, map_location="cpu") -> state
    identity_mismatches(metadata, identity) -> list[str]
    CheckpointManager(directory, keep_last)
        ``save_step(state, step)``, ``save_named(state, name)``, ``latest_path()``,
        ``load_latest(identity)``, ``step_paths()``.

Shapes:
    Tensor shapes are those of the saved model/optimizer/RNG state.

Dtype:
    Stored as saved (float32 master weights and optimizer state; uint8/int64 RNG state).

Device:
    Loaded onto ``map_location`` (CPU by default); the engine moves state to its device.

Invariants:
    - A file either does not exist or is complete: interrupted writes leave only a temporary
      file, which is ignored and cleaned up, and never replace an existing checkpoint.
    - ``latest.json`` only names a completed checkpoint and records its payload sha256.
    - Step files are content-addressed (``step-XXXXXXXX-<first 16 hex of sha256>.pt``), so a
      repeated save of the same step can never modify the file ``latest.json`` names.
    - Top-level keys are exactly CHECKPOINT_KEYS; metadata keys are exactly METADATA_KEYS
      (a whitelist: no environment variables, paths, hostnames or credentials).
    - Step checkpoints beyond ``keep_last`` are pruned; the checkpoint named by ``latest.json``
      is never pruned.

Failure modes:
    - Wrong magic/version, truncated file, checksum mismatch, unexpected keys, or a pointer whose
      checksum disagrees with the file raise CheckpointError.
    - ``load_latest`` raises CheckpointError when the configuration hash, tokenizer sha256 or
      dataset version differ from the expected run identity.

See:
    D-012, plan rev 3.3 section 4.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

__all__ = [
    "CHECKPOINT_KEYS",
    "METADATA_KEYS",
    "CheckpointError",
    "CheckpointManager",
    "RunIdentity",
    "identity_mismatches",
    "read_checkpoint",
    "write_checkpoint",
]

MAGIC = "kittylm-checkpoint"
FORMAT_VERSION = 2  # 2: best_val_loss, skipped_steps; content-addressed step files
CHECKPOINT_KEYS = frozenset(
    {
        "metadata",
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "global_step",
        "tokens_seen",
        "skipped_steps",
        "best_val_loss",
        "rng",
        "loader",
    }
)
METADATA_KEYS = frozenset(
    {
        "config_hash",
        "tokenizer_sha256",
        "dataset_version",
        "git_commit",
        "git_dirty",
        "precision",
        "device_type",
        "determinism",
    }
)
_IDENTITY_FIELDS = ("config_hash", "tokenizer_sha256", "dataset_version")
_STEP_FILE = re.compile(r"^step-(\d{8})-([0-9a-f]{16})\.pt$")
_NAMED = frozenset({"best_val", "crash", "final"})


class CheckpointError(RuntimeError):
    """A checkpoint is damaged, malformed, or belongs to a different run."""


@dataclass(frozen=True)
class RunIdentity:
    """What a checkpoint must match to be resumed."""

    config_hash: str
    tokenizer_sha256: str
    dataset_version: str


def _validate_structure(state: dict[str, Any]) -> None:
    keys = set(state)
    if keys != CHECKPOINT_KEYS:
        raise CheckpointError(
            f"checkpoint keys {sorted(keys)} differ from required {sorted(CHECKPOINT_KEYS)}"
        )
    metadata = state["metadata"]
    if not isinstance(metadata, dict) or set(metadata) != METADATA_KEYS:
        raise CheckpointError(f"checkpoint metadata keys must be exactly {sorted(METADATA_KEYS)}")


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":  # directory handles cannot be fsynced on Windows
        return
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        _fsync_directory(path.parent)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _encode(state: dict[str, Any]) -> tuple[str, bytes]:
    """Validate and serialize ``state``; return ``(payload sha256, file bytes)``."""
    _validate_structure(state)
    buffer = io.BytesIO()
    torch.save(state, buffer)
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    header = {
        "magic": MAGIC,
        "format_version": FORMAT_VERSION,
        "payload_sha256": digest,
        "payload_bytes": len(payload),
    }
    return digest, json.dumps(header, sort_keys=True).encode("ascii") + b"\n" + payload


def write_checkpoint(path: Path, state: dict[str, Any]) -> str:
    """Atomically write ``state`` with a checksummed header; return the payload sha256."""
    digest, data = _encode(state)
    _atomic_write(path, data)
    return digest


def _read_verified(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise CheckpointError(f"cannot read checkpoint {path.name}: {exc}") from exc
    newline = data.find(b"\n")
    if newline < 0:
        raise CheckpointError(f"{path.name}: missing checkpoint header")
    try:
        header = json.loads(data[:newline].decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"{path.name}: unreadable checkpoint header") from exc
    if not isinstance(header, dict) or header.get("magic") != MAGIC:
        raise CheckpointError(f"{path.name}: not a KittyLM checkpoint")
    if header.get("format_version") != FORMAT_VERSION:
        raise CheckpointError(f"{path.name}: unsupported checkpoint format version")
    payload = data[newline + 1 :]
    if len(payload) != header.get("payload_bytes"):
        raise CheckpointError(f"{path.name}: truncated checkpoint payload")
    if hashlib.sha256(payload).hexdigest() != header.get("payload_sha256"):
        raise CheckpointError(f"{path.name}: checkpoint checksum mismatch")
    return header, payload


def read_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Verify and load a checkpoint written by ``write_checkpoint``."""
    _, payload = _read_verified(path)
    try:
        state = torch.load(io.BytesIO(payload), map_location=map_location, weights_only=True)
    except Exception as exc:  # unpickling failures are reported as checkpoint errors
        raise CheckpointError(f"{path.name}: payload could not be loaded safely: {exc}") from exc
    if not isinstance(state, dict):
        raise CheckpointError(f"{path.name}: payload is not a checkpoint dict")
    _validate_structure(state)
    return state


def identity_mismatches(metadata: dict[str, Any], identity: RunIdentity) -> list[str]:
    """Names of identity fields whose values differ from ``identity``."""
    return [name for name in _IDENTITY_FIELDS if metadata.get(name) != getattr(identity, name)]


class CheckpointManager:
    """Step checkpoints with a ``latest.json`` pointer, pruning, and named checkpoints."""

    def __init__(self, directory: Path, keep_last: int) -> None:
        if keep_last < 1:
            raise ValueError("keep_last must be >= 1")
        self.directory = directory
        self.keep_last = keep_last
        self.pointer = directory / "latest.json"

    def step_paths(self) -> list[Path]:
        """Existing step checkpoints in ascending step order (temporary files excluded)."""
        if not self.directory.is_dir():
            return []
        paths = [p for p in self.directory.iterdir() if _STEP_FILE.match(p.name)]
        return sorted(paths, key=lambda p: (int(p.name[5:13]), p.stat().st_mtime_ns, p.name))

    def save_step(self, state: dict[str, Any], step: int) -> Path:
        """Write ``step-XXXXXXXX-<digest>.pt``, then update ``latest.json``, then prune.

        The file name contains the payload digest, so saving a step again never modifies the
        file the current pointer names: identical bytes get the same name, a different state
        gets a new file. The pointer moves only after that file is durable, and pruning never
        removes the file the pointer names.
        """
        digest, data = _encode(state)
        path = self.directory / f"step-{step:08d}-{digest[:16]}.pt"
        _atomic_write(path, data)
        pointer = {"file": path.name, "global_step": step, "payload_sha256": digest}
        _atomic_write(self.pointer, json.dumps(pointer, sort_keys=True).encode("ascii") + b"\n")
        for old in self.step_paths()[: -self.keep_last]:
            if old.name != path.name:
                old.unlink(missing_ok=True)
        return path

    def save_named(self, state: dict[str, Any], name: str) -> Path:
        """Write a named checkpoint: ``best_val``, ``crash`` or ``final``."""
        if name not in _NAMED:
            raise ValueError(f"named checkpoint must be one of {sorted(_NAMED)}")
        path = self.directory / f"{name}.pt"
        write_checkpoint(path, state)
        return path

    def latest_path(self) -> Path | None:
        """The checkpoint named by ``latest.json``, or None if there is none."""
        if not self.pointer.exists():
            return None
        try:
            pointer = json.loads(self.pointer.read_text(encoding="ascii"))
            name = pointer["file"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise CheckpointError("latest.json is unreadable") from exc
        if not isinstance(name, str) or not _STEP_FILE.match(name):
            raise CheckpointError("latest.json does not name a step checkpoint")
        path = self.directory / name
        if not path.exists():
            raise CheckpointError(f"latest.json names missing checkpoint {name}")
        header, _ = _read_verified(path)
        if header["payload_sha256"] != pointer.get("payload_sha256"):
            raise CheckpointError("latest.json checksum does not match the checkpoint it names")
        return path

    def load_latest(
        self, identity: RunIdentity, map_location: str | torch.device = "cpu"
    ) -> dict[str, Any] | None:
        """Load the latest checkpoint after verifying it belongs to ``identity``."""
        path = self.latest_path()
        if path is None:
            return None
        state = read_checkpoint(path, map_location=map_location)
        mismatches = identity_mismatches(state["metadata"], identity)
        if mismatches:
            raise CheckpointError(
                f"refusing to resume {path.name}: {', '.join(mismatches)} differ from this run"
            )
        return state
