"""Checkpoint safety: atomic writes, checksums, safe loading, identity checks, pruning."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from kittylm.training.checkpoint import (
    CHECKPOINT_KEYS,
    FORMAT_VERSION,
    METADATA_KEYS,
    CheckpointError,
    CheckpointManager,
    RunIdentity,
    read_checkpoint,
    write_checkpoint,
)
from tests.conftest import ROOT
from tests.training_helpers import make_engine

IDENTITY = RunIdentity("a" * 64, "b" * 64, "c" * 64)


def state(step: int = 1, **metadata_changes: Any) -> dict[str, Any]:
    metadata = {
        "config_hash": IDENTITY.config_hash,
        "model_config_sha256": "f" * 64,
        "tokenizer_sha256": IDENTITY.tokenizer_sha256,
        "dataset_version": IDENTITY.dataset_version,
        "git_commit": "d" * 40,
        "git_dirty": False,
        "precision": "fp32",
        "device_type": "cpu",
        "determinism": "deterministic",
        **metadata_changes,
    }
    return {
        "metadata": metadata,
        "model": {"w": torch.full((3,), float(step))},
        "optimizer": {"state": {}, "param_groups": []},
        "scheduler": {"step": step, "last_lr": 0.1},
        "scaler": None,
        "global_step": step,
        "tokens_seen": step * 10,
        "skipped_steps": 0,
        "best_val_loss": float("inf"),
        "rng": {"torch": torch.get_rng_state()},
        "loader": {"generator": torch.Generator().get_state(), "batches_drawn": step},
    }


def test_round_trip_and_header(tmp_path: Path) -> None:
    path = tmp_path / "ck.pt"
    digest = write_checkpoint(path, state(3))
    loaded = read_checkpoint(path)
    assert loaded["global_step"] == 3 and torch.equal(loaded["model"]["w"], torch.full((3,), 3.0))
    header = json.loads(path.read_bytes().split(b"\n", 1)[0])
    assert header["payload_sha256"] == digest and header["magic"] == "kittylm-checkpoint"


def test_whitelists_are_enforced_on_write(tmp_path: Path) -> None:
    extra_key = state()
    extra_key["environment"] = {"HOME": "x"}
    with pytest.raises(CheckpointError, match="keys"):
        write_checkpoint(tmp_path / "a.pt", extra_key)
    with pytest.raises(CheckpointError, match="metadata keys"):
        write_checkpoint(tmp_path / "b.pt", state(hostname="box"))
    assert set(state()) == CHECKPOINT_KEYS and set(state()["metadata"]) == METADATA_KEYS
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        (lambda b: b[:-10], "truncated"),
        (lambda b: b[:-1] + bytes([b[-1] ^ 0xFF]), "checksum mismatch"),
        (lambda b: b.replace(b"kittylm-checkpoint", b"other-checkpoint!!"), "not a KittyLM"),
        (
            lambda b: b.replace(
                f'"format_version": {FORMAT_VERSION}'.encode(), b'"format_version": 9'
            ),
            "format version",
        ),
        (lambda b: b"garbage without header", "missing checkpoint header"),
        (lambda b: b"{not json\n" + b, "unreadable checkpoint header"),
    ],
)
def test_damaged_checkpoints_are_rejected(tmp_path: Path, damage: Any, message: str) -> None:
    path = tmp_path / "ck.pt"
    write_checkpoint(path, state())
    path.write_bytes(damage(path.read_bytes()))
    with pytest.raises(CheckpointError, match=message):
        read_checkpoint(path)


class NotAllowed:
    """An arbitrary class: weights_only loading must refuse to unpickle it."""


def test_arbitrary_objects_are_refused_even_with_a_valid_checksum(tmp_path: Path) -> None:
    evil = state()
    evil["model"] = {"w": NotAllowed()}
    buffer = io.BytesIO()
    torch.save(evil, buffer)
    payload = buffer.getvalue()
    header = {
        "magic": "kittylm-checkpoint",
        "format_version": FORMAT_VERSION,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "payload_bytes": len(payload),
    }
    path = tmp_path / "evil.pt"
    path.write_bytes(json.dumps(header, sort_keys=True).encode() + b"\n" + payload)
    with pytest.raises(CheckpointError, match="could not be loaded safely"):
        read_checkpoint(path)


def test_manager_pointer_pruning_and_named(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path / "ck", keep_last=2)
    assert manager.latest_path() is None and manager.load_latest(IDENTITY) is None
    for step in (1, 2, 3, 4):
        manager.save_step(state(step), step)
    assert [p.name[:13] for p in manager.step_paths()] == ["step-00000003", "step-00000004"]
    latest = manager.latest_path()
    assert latest is not None and latest.name.startswith("step-00000004-")
    loaded = manager.load_latest(IDENTITY)
    assert loaded is not None and loaded["global_step"] == 4
    assert manager.save_named(state(9), "crash").name == "crash.pt"
    with pytest.raises(ValueError, match="named checkpoint"):
        manager.save_named(state(), "whatever")


@pytest.mark.parametrize("field", ["config_hash", "tokenizer_sha256", "dataset_version"])
def test_identity_mismatch_refuses_resume(tmp_path: Path, field: str) -> None:
    manager = CheckpointManager(tmp_path / "ck", keep_last=1)
    manager.save_step(state(1, **{field: "e" * 64}), 1)
    with pytest.raises(CheckpointError, match=f"refusing to resume.*{field}"):
        manager.load_latest(IDENTITY)


def test_pointer_problems_are_detected(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path / "ck", keep_last=3)
    manager.save_step(state(1), 1)
    manager.save_step(state(2), 2)
    pointer = json.loads(manager.pointer.read_text(encoding="ascii"))
    manager.pointer.write_text(
        json.dumps({**pointer, "payload_sha256": "0" * 64}), encoding="ascii"
    )
    with pytest.raises(CheckpointError, match="checksum does not match"):
        manager.latest_path()
    manager.pointer.write_text(
        json.dumps({**pointer, "file": "step-00000099-0123456789abcdef.pt"}), encoding="ascii"
    )
    with pytest.raises(CheckpointError, match="missing checkpoint"):
        manager.latest_path()
    manager.pointer.write_text("{broken", encoding="ascii")
    with pytest.raises(CheckpointError, match="unreadable"):
        manager.latest_path()


def test_interrupted_save_keeps_previous_latest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = CheckpointManager(tmp_path / "ck", keep_last=5)
    manager.save_step(state(1), 1)

    def failing_replace(src: str, dst: str) -> None:
        raise OSError("simulated crash during rename")

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(OSError, match="simulated crash"):
        manager.save_step(state(2), 2)
    monkeypatch.undo()
    names = sorted(p.name for p in (tmp_path / "ck").iterdir())
    assert len(names) == 2 and names[0] == "latest.json" and names[1].startswith("step-00000001-")
    loaded = manager.load_latest(IDENTITY)
    assert loaded is not None and loaded["global_step"] == 1


def test_resaving_a_step_cannot_invalidate_the_current_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression: step files used to be named only by step, so a second save of the same step
    # overwrote the file latest.json named before the pointer moved; a crash in between left a
    # pointer whose checksum no longer matched, and the only latest checkpoint was refused.
    manager = CheckpointManager(tmp_path / "ck", keep_last=3)
    first = state(5)
    manager.save_step(first, 5)
    pointed = manager.latest_path()
    assert pointed is not None
    original_bytes = pointed.read_bytes()

    changed = state(5)
    changed["model"]["w"] = torch.full((3,), -1.0)  # same step, different content
    real_replace = os.replace

    def pointer_write_fails(src: str, dst: str) -> None:
        if Path(dst).name == "latest.json":
            raise OSError("simulated crash while replacing latest.json")
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", pointer_write_fails)
    with pytest.raises(OSError, match="latest.json"):
        manager.save_step(changed, 5)
    monkeypatch.undo()

    assert pointed.read_bytes() == original_bytes  # the pointed-at file was never touched
    loaded = manager.load_latest(IDENTITY)
    assert loaded is not None and torch.equal(loaded["model"]["w"], first["model"]["w"])

    # A completed re-save points at the new content; identical content maps to the same file.
    new_path = manager.save_step(changed, 5)
    assert new_path != pointed and manager.latest_path() == new_path
    assert manager.save_step(changed, 5) == new_path


HARD_KILL = """
import os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
from tests.test_checkpoint import state
from kittylm.training.checkpoint import CheckpointManager
manager = CheckpointManager(Path(sys.argv[1]), keep_last=5)
manager.save_step(state(1), 1)
real_fsync = os.fsync
def die(fd):
    os._exit(9)  # the process dies mid-write: no cleanup, no exception handling
os.fsync = die
manager.save_step(state(2), 2)
"""


def test_hard_kill_during_save_leaves_a_loadable_previous_checkpoint(tmp_path: Path) -> None:
    directory = tmp_path / "ck"
    result = subprocess.run(
        [sys.executable, "-c", HARD_KILL, str(directory), str(ROOT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 9, result.stderr[-1000:]
    names = sorted(p.name for p in directory.iterdir())
    # the partial file was never renamed into place
    assert not any(n.startswith("step-00000002-") for n in names)
    assert any(n.startswith(".step-00000002-") and n.endswith(".tmp") for n in names)
    manager = CheckpointManager(directory, keep_last=5)
    assert [p.name[:13] for p in manager.step_paths()] == ["step-00000001"]
    loaded = manager.load_latest(IDENTITY)
    assert loaded is not None and loaded["global_step"] == 1


def test_engine_refuses_checkpoint_from_different_config(tmp_path: Path) -> None:
    engine = make_engine(tmp_path, run_name="a", max_steps=12)
    engine.train(2)
    engine.save_checkpoint()
    engine.close()
    different = make_engine(tmp_path, run_name="a", max_steps=13)  # same run dir, different config
    with pytest.raises(CheckpointError, match="config_hash"):
        different.resume_latest()
    different.close()
