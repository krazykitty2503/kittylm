r"""Code-aware pre-tokenization: split text into chunks that BPE merges never cross.

Purpose:
    Byte-pair encoding learns merges between adjacent tokens. Without pre-tokenization a
    frequent cross-word pair (for example the end of one identifier and the start of the
    next) could become a token, and training would have to consider every position of the
    corpus. Splitting text into chunks first (words with an optional leading space, digit
    groups, punctuation runs, newline+indentation runs, spaces) keeps tokens meaningful and
    lets the trainer count *unique* chunks instead of raw positions.

    The pattern follows the GPT-4 (cl100k) style and is adapted for source code:
    - newline runs absorb the following indentation (``"\n    "``) so indentation levels
      become learnable units;
    - a word may carry one leading non-letter (``" name"``, ``"_private"``, ``".attr"``);
    - digits split into groups of at most three.

Public API:
    PRETOKENIZER_ID
        Stable identifier of this pattern. Any change to PATTERN requires a new id, because
        tokenizer artifacts record the id and the exact pattern and refuse a mismatch.
    PATTERN
        The regular expression (``regex`` module syntax: ``\p{L}``, possessive quantifiers).
    pretokenize(text) -> list[str]
        Chunks in order.

Invariants:
    - Lossless: ``"".join(pretokenize(text)) == text`` for every ``str``, including strings
      with lone surrogates (used to carry undecodable bytes). A final catch-all alternative
      guarantees every character is covered.
    - No chunk is empty; the function is pure and deterministic.

Failure modes:
    - None for valid ``str`` input. Combining marks form their own chunks (they are not
      ``\p{L}``); this affects compression only, never round-trip correctness.

See:
    D-001 (byte-level BPE from scratch), plan rev 3.3 section 2.
"""

from __future__ import annotations

import regex

__all__ = ["PATTERN", "PRETOKENIZER_ID", "pretokenize"]

PRETOKENIZER_ID = "kittylm-code-v1"

PATTERN = (
    r"'(?i:[sdmt]|ll|ve|re)"  # English contractions
    r"|[^\r\n\p{L}\p{N}]?+\p{L}++"  # word with at most one leading non-letter
    r"|\p{N}{1,3}"  # digit groups of 1-3
    r"| ?[^\s\p{L}\p{N}]++"  # punctuation/symbol run with optional leading space
    r"|(?:\r\n|\r|\n)++[ \t]*+"  # newline run plus the following indentation
    r"|[ \t]+(?!\S)"  # spaces/tabs, leaving one space to attach to the next word
    r"|\s+"  # any other whitespace
    r"|(?s:.)"  # catch-all: guarantees lossless coverage
)

_COMPILED = regex.compile(PATTERN)


def pretokenize(text: str) -> list[str]:
    """Split ``text`` into BPE chunks; concatenating the result reproduces ``text``."""
    chunks: list[str] = _COMPILED.findall(text)
    return chunks
