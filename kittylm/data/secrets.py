"""Secret scanning for data ingestion and repository hygiene.

Purpose:
    Detect credentials (service tokens, private keys, password-style assignments and
    high-entropy literals) in text *before* that text is written anywhere: the dataset,
    git, manifests, logs, experiment records or generated samples. One scanner guards
    both the data pipeline and the repository, so the rules are defined exactly once.
    It is deliberately small and offline so every rule can be read and audited.

Public API:
    scan_text(text) -> list[Finding]
        All findings in ``text``, ordered by line then rule name.
    Finding(line, rule)
        A detection. It carries the 1-based line number and rule name only.
    RULE_NAMES
        Tuple of every rule name the scanner can emit.

Invariants:
    - A Finding never contains the matched value, and its repr cannot reveal it.
    - scan_text is pure: no I/O, no global state, deterministic for a given input.
    - Rules err toward false positives. The data pipeline drops a whole document on any
      finding instead of trying to redact it (D-006).

Failure modes:
    - False negatives: credential formats without a rule are not detected. The scanner is
      one layer of defense-in-depth, not a guarantee.
    - False positives: long random-looking literals (for example base64 test fixtures)
      are flagged; in the data pipeline those documents are dropped.
    - Input is ``str``; callers decode bytes first (the repo scanner uses UTF-8 with
      replacement, so undecodable bytes cannot hide a match inside valid text).

See:
    D-006 (drop the whole file on a secret hit), SECURITY.md (secret-inclusion runbook).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

__all__ = ["RULE_NAMES", "Finding", "scan_text"]


@dataclass(frozen=True, slots=True, order=True)
class Finding:
    """A secret-scanner detection that deliberately omits the matched value."""

    line: int
    rule: str


# Characters that may form a token-like credential value.
_TOKEN_CHARS = re.compile(r"^[A-Za-z0-9+/=_.~-]+$")
# Substrings that mark a value as an obvious placeholder rather than a real credential.
_PLACEHOLDER_HINTS = (
    "example",
    "placeholder",
    "changeme",
    "change_me",
    "your",
    "xxxx",
    "dummy",
    "redacted",
    "sample",
    "insert",
)

_NOT_TOKEN_CHAR = r"(?![A-Za-z0-9_-])"

# Service-specific formats. Each pattern matches the credential shape itself.
_SERVICE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private_key_block", re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----")),
    ("aws_access_key_id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    (
        "github_token",
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,255}|github_pat_[A-Za-z0-9_]{22,255})\b"),
    ),
    ("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}" + _NOT_TOKEN_CHAR)),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}" + _NOT_TOKEN_CHAR)),
    (
        "discord_bot_token",
        re.compile(
            r"\b[MNO][A-Za-z0-9_-]{23,27}\.[A-Za-z0-9_-]{6}\.[A-Za-z0-9_-]{27,40}" + _NOT_TOKEN_CHAR
        ),
    ),
    ("telegram_bot_token", re.compile(r"\b\d{8,10}:AA[A-Za-z0-9_-]{33}" + _NOT_TOKEN_CHAR)),
)

# OpenAI/Anthropic-style keys share the ``sk-`` prefix. Validated by _looks_like_credential
# so that prose such as "sk-learn-contrib-..." is not flagged.
_SK_KEY = re.compile(r"\bsk-[A-Za-z0-9_-]{20,}" + _NOT_TOKEN_CHAR)

_KEY_NAMES = (
    r"api[_-]?key|secret(?:[_-]?key)?|client[_-]?secret|access[_-]?key|auth[_-]?token"
    r"|bot[_-]?token|token|password|passwd|pwd"
)
# key = "value" / key: 'value' / "key": "value"
_QUOTED_ASSIGNMENT = re.compile(
    rf"""(?i)(?<![A-Za-z0-9])(?:{_KEY_NAMES})["']?\s*[:=]\s*["']([^"'\s]{{12,}})["']"""
)
# KEY=value / key: value on its own line (.env and YAML style, unquoted)
_BARE_ASSIGNMENT = re.compile(
    r"(?im)^[ \t]*(?:export[ \t]+)?[A-Za-z0-9_]*(?<![A-Za-z0-9])"
    rf"(?:{_KEY_NAMES})[ \t]*[:=][ \t]*"
    r"""([^\s"'#]{12,})[ \t]*$"""
)
# Quoted literals that may be random credentials.
_QUOTED_LITERAL = re.compile(r"""["']([A-Za-z0-9+/=_-]{32,})["']""")
_HEX_ONLY = re.compile(r"^[0-9a-fA-F]+$")
_ENTROPY_THRESHOLD_BITS = 4.5

RULE_NAMES: tuple[str, ...] = (
    *(name for name, _ in _SERVICE_PATTERNS),
    "sk_api_key",
    "credential_assignment",
    "high_entropy_string",
)


def _has_mixed_classes(value: str) -> bool:
    return (
        any(c.isdigit() for c in value)
        and any(c.islower() for c in value)
        and any(c.isupper() for c in value)
    )


def _is_placeholder(value: str) -> bool:
    lowered = value.lower()
    return any(hint in lowered for hint in _PLACEHOLDER_HINTS)


def _looks_like_credential(value: str) -> bool:
    """Heuristic for assignment values: token charset, letters and digits, not a placeholder."""
    if not _TOKEN_CHARS.match(value) or _is_placeholder(value):
        return False
    has_letter = any(c.isalpha() for c in value)
    has_digit = any(c.isdigit() for c in value)
    return has_letter and (has_digit or len(value) >= 20)


def _shannon_entropy_bits(value: str) -> float:
    counts = Counter(value)
    total = len(value)
    return -sum((n / total) * math.log2(n / total) for n in counts.values())


def _is_high_entropy_literal(value: str) -> bool:
    # Hex strings are overwhelmingly hashes (sha256, git commits) in this project; they are
    # not treated as secrets. Base64-like values must mix character classes.
    if _HEX_ONLY.match(value) or not _has_mixed_classes(value) or _is_placeholder(value):
        return False
    return _shannon_entropy_bits(value) >= _ENTROPY_THRESHOLD_BITS


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def scan_text(text: str) -> list[Finding]:
    """Return all secret findings in ``text`` without exposing matched values.

    Args:
        text: Decoded text to scan.

    Returns:
        Findings sorted by line and rule. At most one finding per (line, rule).
    """
    found: set[Finding] = set()

    def add_matches(
        rule: str,
        pattern: re.Pattern[str],
        accept: Callable[[str], bool] | None = None,
        group: int = 0,
    ) -> None:
        for match in pattern.finditer(text):
            if accept is None or accept(match.group(group)):
                found.add(Finding(line=_line_of(text, match.start()), rule=rule))

    for rule, pattern in _SERVICE_PATTERNS:
        add_matches(rule, pattern)
    add_matches("sk_api_key", _SK_KEY, lambda v: _has_mixed_classes(v[3:]))
    add_matches("credential_assignment", _QUOTED_ASSIGNMENT, _looks_like_credential, group=1)
    add_matches("credential_assignment", _BARE_ASSIGNMENT, _looks_like_credential, group=1)
    add_matches("high_entropy_string", _QUOTED_LITERAL, _is_high_entropy_literal, group=1)
    return sorted(found)
