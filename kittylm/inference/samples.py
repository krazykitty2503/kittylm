"""Persist generated samples only after secret scanning and redaction.

Purpose:
    A language model can reproduce credential-like strings it saw in training data, or invent
    ones that look real. Generated text is therefore scanned with the repository's secret
    scanner *before* anything touches disk: every line with a finding is replaced by a
    redaction marker naming the rules (never the value), the result is re-scanned, and only
    clean text is written. The raw text is never written, not even to a temporary file.

Public API:
    SavedSample(path, redacted_lines, rules)
    SampleSecretError
    redact_secrets(text) -> (clean_text, redacted_lines, rules)
    save_sample(directory, name, text) -> SavedSample

Invariants:
    - The written file and its ``.json`` sidecar contain no text the scanner flags.
    - Redaction markers carry rule names and line numbers only, never matched values.
    - Files are written atomically (temporary file in the same directory, then rename).
    - ``name`` is a plain file stem: no path separators, no ``..``; the file lands directly in
      ``directory``.

Failure modes:
    - An invalid ``name`` raises ValueError.
    - If text is still flagged after redaction (defensive: a rule spanning redaction markers),
      SampleSecretError is raised and nothing is written.
    - The scanner has false negatives (see kittylm/data/secrets.py); this is defense-in-depth.

See:
    D-006, docs/safety.md, kittylm/data/secrets.py.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from kittylm.data.secrets import scan_text

__all__ = ["SampleSecretError", "SavedSample", "redact_secrets", "save_sample"]

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
MAX_REDACTION_PASSES = 3


class SampleSecretError(RuntimeError):
    """Generated text could not be made clean by redaction; nothing was written."""


@dataclass(frozen=True)
class SavedSample:
    """Where a sample was written and what was redacted (line numbers and rule names only)."""

    path: Path
    redacted_lines: tuple[int, ...]
    rules: tuple[str, ...]


def redact_secrets(text: str) -> tuple[str, tuple[int, ...], tuple[str, ...]]:
    """Replace every flagged line with a marker; return clean text, lines and rule names."""
    redacted: set[int] = set()
    rules: set[str] = set()
    for _ in range(MAX_REDACTION_PASSES):
        findings = scan_text(text)
        if not findings:
            return text, tuple(sorted(redacted)), tuple(sorted(rules))
        lines = text.split("\n")
        by_line: dict[int, set[str]] = {}
        for finding in findings:
            by_line.setdefault(finding.line, set()).add(finding.rule)
        for line, line_rules in by_line.items():
            lines[line - 1] = f"[REDACTED by secret scan: {', '.join(sorted(line_rules))}]"
            redacted.add(line)
            rules.update(line_rules)
        text = "\n".join(lines)
    if scan_text(text):
        raise SampleSecretError("generated text is still flagged after redaction; not saved")
    return text, tuple(sorted(redacted)), tuple(sorted(rules))


def _atomic_write_text(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def save_sample(directory: Path, name: str, text: str) -> SavedSample:
    """Scan and redact ``text``, then write ``<name>.txt`` and ``<name>.json`` atomically."""
    if not _NAME.match(name) or ".." in name:
        raise ValueError(f"invalid sample name {name!r}: use letters, digits, '_', '-', '.'")
    clean, lines, rules = redact_secrets(text)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.txt"
    sidecar = {"redacted_lines": list(lines), "rules": list(rules), "secret_scan": "clean"}
    _atomic_write_text(path, clean)
    _atomic_write_text(directory / f"{name}.json", json.dumps(sidecar, sort_keys=True) + "\n")
    return SavedSample(path=path, redacted_lines=lines, rules=rules)
