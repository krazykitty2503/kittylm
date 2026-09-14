"""Artifact guard: keeps data, checkpoints, weights and credentials out of git.

Rules (plan rev 3.1, section 7):
    - Forbidden suffixes: model/tensor/pickle/database artifacts.
    - Forbidden directories: ``data/raw/``, ``data/cleaned/``, ``data/tokenized/``, ``runs/``.
    - ``.env*`` files, except ``*.example`` templates.
    - Any file larger than 1 MB, unless it lives under an explicitly allowlisted prefix.
    - Allowlisted manifests (``data/manifests/``) must be UTF-8 text without NUL bytes.
      Their schema, relative-path and no-secret requirements are enforced by the manifest
      tooling (step 3) and the repository secret scan.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import PurePosixPath

from kittylm.utils.git import IndexEntry

__all__ = [
    "FORBIDDEN_DIRS",
    "FORBIDDEN_SUFFIXES",
    "MAX_FILE_BYTES",
    "SIZE_ALLOWLIST_PREFIXES",
    "Violation",
    "check_entries",
]

FORBIDDEN_SUFFIXES: tuple[str, ...] = (
    ".pt",
    ".pth",
    ".ckpt",
    ".bin",
    ".safetensors",
    ".gguf",
    ".npy",
    ".npz",
    ".pkl",
    ".pickle",
    ".sqlite",
    ".sqlite3",
    ".db",
)
FORBIDDEN_DIRS: tuple[str, ...] = ("data/raw/", "data/cleaned/", "data/tokenized/", "runs/")
MAX_FILE_BYTES = 1_000_000
SIZE_ALLOWLIST_PREFIXES: tuple[str, ...] = ("data/manifests/",)
_TEXT_ONLY_PREFIXES: tuple[str, ...] = ("data/manifests/",)


@dataclass(frozen=True, slots=True, order=True)
class Violation:
    """A tracked or staged file that must not be committed."""

    path: str
    reason: str


def _is_env_file(name: str) -> bool:
    return name.startswith(".env") and not name.endswith(".example")


def check_entries(
    entries: Iterable[IndexEntry], read: Callable[[list[str]], dict[str, bytes]]
) -> list[Violation]:
    """Return artifact-guard violations for index entries.

    Args:
        entries: Files to check.
        read: Callable mapping blob ids to contents; only used for text-only prefixes.
    """
    violations: list[Violation] = []
    text_only: list[IndexEntry] = []
    for entry in entries:
        path = PurePosixPath(entry.path)
        lowered = entry.path.lower()
        if any(lowered.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES):
            violations.append(Violation(entry.path, f"forbidden artifact type '{path.suffix}'"))
        if any(entry.path.startswith(prefix) for prefix in FORBIDDEN_DIRS):
            violations.append(Violation(entry.path, "forbidden generated-data/run directory"))
        if _is_env_file(path.name):
            violations.append(Violation(entry.path, "forbidden environment/credential file"))
        allowlisted = any(entry.path.startswith(p) for p in SIZE_ALLOWLIST_PREFIXES)
        if entry.size > MAX_FILE_BYTES and not allowlisted:
            violations.append(
                Violation(entry.path, f"file is {entry.size} bytes (> {MAX_FILE_BYTES} limit)")
            )
        if any(entry.path.startswith(p) for p in _TEXT_ONLY_PREFIXES):
            text_only.append(entry)

    if text_only:
        contents = read([e.blob for e in text_only])
        for entry in text_only:
            data = contents[entry.blob]
            try:
                data.decode("utf-8")
                is_text = b"\0" not in data
            except UnicodeDecodeError:
                is_text = False
            if not is_text:
                violations.append(Violation(entry.path, "manifest must be UTF-8 text"))
    return sorted(violations)
