"""Read-only git helpers used by the ledger, secret scan and artifact guard.

Scanning reads blobs from the git *index* rather than the working tree. In CI the index
equals the checked-out commit; in the pre-commit hook it is exactly what is about to be
committed. One reader therefore serves both "tracked files" and "staged files".
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "GitError",
    "IndexEntry",
    "head_commit",
    "index_entries",
    "is_dirty",
    "read_blobs",
    "repo_root",
]

_REGULAR_FILE_MODES = {"100644", "100755"}


class GitError(RuntimeError):
    """A git command failed."""


@dataclass(frozen=True, slots=True)
class IndexEntry:
    """One regular file in the git index."""

    path: str  # repository-relative, forward slashes
    blob: str
    size: int


def _git(args: list[str], cwd: Path, stdin: bytes | None = None) -> bytes:
    result = subprocess.run(["git", *args], cwd=cwd, input=stdin, capture_output=True, check=False)
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise GitError(f"git {' '.join(args)} failed: {message}")
    return result.stdout


def repo_root(cwd: Path | None = None) -> Path:
    """Return the top-level directory of the repository containing ``cwd``."""
    out = _git(["rev-parse", "--show-toplevel"], cwd or Path.cwd())
    return Path(out.decode("utf-8").strip())


def head_commit(cwd: Path) -> str:
    """Return the full hash of HEAD."""
    return _git(["rev-parse", "HEAD"], cwd).decode("ascii").strip()


def is_dirty(cwd: Path) -> bool:
    """Return True if tracked files differ from HEAD (untracked files are ignored)."""
    out = _git(["status", "--porcelain", "--untracked-files=no"], cwd)
    return bool(out.strip())


def index_entries(cwd: Path, staged_only: bool = False) -> list[IndexEntry]:
    """List regular files in the index with their blob ids and sizes.

    Args:
        cwd: Any directory inside the repository.
        staged_only: Restrict to files added, copied, modified or renamed in the index
            relative to HEAD (what the next commit would change).
    """
    raw = _git(["ls-files", "-s", "-z"], cwd)
    entries: list[tuple[str, str]] = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        meta, _, path = record.partition(b"\t")
        mode, blob, _stage = meta.decode("ascii").split(" ")
        if mode in _REGULAR_FILE_MODES:
            entries.append((path.decode("utf-8"), blob))

    if staged_only:
        changed = _git(["diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR"], cwd)
        wanted = {p.decode("utf-8") for p in changed.split(b"\0") if p}
        entries = [(p, b) for p, b in entries if p in wanted]

    sizes = _blob_sizes(cwd, (blob for _, blob in entries))
    return [IndexEntry(path=p, blob=b, size=sizes[b]) for p, b in sorted(entries)]


def _blob_sizes(cwd: Path, blobs: Iterable[str]) -> dict[str, int]:
    unique = sorted(set(blobs))
    if not unique:
        return {}
    out = _git(["cat-file", "--batch-check"], cwd, stdin="\n".join(unique).encode() + b"\n")
    sizes: dict[str, int] = {}
    for line in out.decode("ascii").splitlines():
        blob, _kind, size = line.split(" ")
        sizes[blob] = int(size)
    return sizes


def read_blobs(cwd: Path, blobs: Iterable[str]) -> dict[str, bytes]:
    """Read blob contents by id using a single ``git cat-file --batch`` call."""
    unique = sorted(set(blobs))
    if not unique:
        return {}
    out = _git(["cat-file", "--batch"], cwd, stdin="\n".join(unique).encode() + b"\n")
    contents: dict[str, bytes] = {}
    pos = 0
    for blob in unique:
        header_end = out.index(b"\n", pos)
        header_blob, _kind, size = out[pos:header_end].decode("ascii").split(" ")
        start = header_end + 1
        end = start + int(size)
        contents[header_blob] = out[start:end]
        pos = end + 1  # skip the trailing newline after each object
        if header_blob != blob:
            raise GitError(f"unexpected blob order from cat-file: {header_blob} != {blob}")
    return contents
