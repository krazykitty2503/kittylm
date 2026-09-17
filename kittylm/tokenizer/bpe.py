"""Byte-level BPE tokenizer: encoding, decoding and the deterministic artifact format.

Purpose:
    Map any text or byte string to integer token ids and back, losslessly. Starting from the
    256 possible byte values means there is no unknown token: code, shell, JSON, emoji or
    invalid UTF-8 all encode. Learned merges (see ``trainer.py``) combine frequent adjacent
    tokens into longer ones, so common text needs fewer ids.

    Vocabulary layout for a tokenizer of size V with M merges and S special tokens:
        ids 0..255              raw bytes
        ids 256..256+M-1        merges, in the order they were learned (rank = id - 256)
        ids V-S..V-1            special tokens (S = 32), in SPECIAL_TOKENS order
    Special-token ids are fixed and deterministic within an artifact/configuration, but they
    differ across vocabulary sizes because they occupy the top of each vocabulary. Every
    artifact records its exact special-token mapping; loading rejects a mapping that does not
    match this layout, so ids are never silently renumbered.

Public API:
    SPECIAL_TOKENS, NAMED_SPECIAL_TOKENS
        Reserved special-token strings (named protocol tokens + reserved slots).
    BPETokenizer(merges, *, corpus_sha256, min_pair_count, cache_size=65536)
        encode(text, allowed_special=frozenset()) -> list[int]
        encode_bytes(data) -> list[int]
        decode(ids) -> str                 (UTF-8, undecodable bytes replaced)
        decode_bytes(ids) -> bytes          (exact)
        special_token_ids -> dict[str, int]; eot_id; pad_id; vocab_size; n_merges
        truncated(vocab_size) -> BPETokenizer
        to_json_bytes() / sha256 / save(path) / load(path) / from_json_bytes(data)
    TokenizerError

Invariants:
    - ``decode_bytes(encode_bytes(b)) == b`` for all bytes; ``decode(encode(s)) == s`` for all
      strings without lone surrogates.
    - ``encode``/``encode_bytes`` never emit a special-token id unless that token's name is in
      ``allowed_special``: literal special-token text in data or tool output is ordinary bytes.
    - Artifact bytes are canonical (sorted keys, compact separators, UTF-8, trailing newline,
      no timestamps, paths, hostnames or usernames); ``sha256`` identifies the tokenizer.
    - ``truncated(n).to_json_bytes()`` equals the artifact of training directly to ``n``.
    - The encoding cache only memoizes; results are identical with ``cache_size=0``.

Failure modes:
    - ``encode`` raises TokenizerError for strings containing lone surrogates (not valid
      Unicode scalar text); use ``encode_bytes`` for arbitrary bytes.
    - ``decode``/``decode_bytes`` raise TokenizerError for ids outside ``[0, vocab_size)``.
    - ``load``/``from_json_bytes`` raise TokenizerError on unknown format, changed
      pre-tokenizer, malformed merges, or a special-token mapping that does not match.
    - ``encode`` raises TokenizerError if ``allowed_special`` names an unknown token.

See:
    D-001 (byte-level BPE), D-002 (vocabulary sizes), D-019 (special-token ids),
    plan rev 3.3 section 2.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from collections.abc import Set as AbstractSet
from pathlib import Path
from typing import Any

import regex

from kittylm.tokenizer.pretokenize import PATTERN, PRETOKENIZER_ID, pretokenize

__all__ = [
    "ARTIFACT_FORMAT",
    "ARTIFACT_FORMAT_VERSION",
    "NAMED_SPECIAL_TOKENS",
    "SPECIAL_TOKENS",
    "BPETokenizer",
    "TokenizerError",
]

ARTIFACT_FORMAT = "kittylm-bpe"
ARTIFACT_FORMAT_VERSION = 1

NAMED_SPECIAL_TOKENS: tuple[str, ...] = (
    "<|endoftext|>",
    "<|pad|>",
    "<|system|>",
    "<|user|>",
    "<|context|>",
    "<|plan|>",
    "<|tool_request|>",
    "<|tool_result|>",
    "<|validation|>",
    "<|final|>",
)
RESERVED_SPECIAL_COUNT = 22
SPECIAL_TOKENS: tuple[str, ...] = NAMED_SPECIAL_TOKENS + tuple(
    f"<|reserved_{i}|>" for i in range(RESERVED_SPECIAL_COUNT)
)
BYTE_VOCAB = 256
MIN_VOCAB_SIZE = BYTE_VOCAB + len(SPECIAL_TOKENS)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_KEYS = {
    "corpus_sha256",
    "format",
    "format_version",
    "merges",
    "min_pair_count",
    "pretokenizer",
    "pretokenizer_pattern",
    "special_tokens",
    "vocab_size",
}


class TokenizerError(ValueError):
    """Invalid tokenizer input, artifact or configuration."""


def expected_special_token_ids(vocab_size: int) -> dict[str, int]:
    """Return the special-token mapping for a vocabulary of ``vocab_size`` ids."""
    first = vocab_size - len(SPECIAL_TOKENS)
    return {name: first + i for i, name in enumerate(SPECIAL_TOKENS)}


class BPETokenizer:
    """A trained byte-level BPE tokenizer (see module docstring for the id layout)."""

    def __init__(
        self,
        merges: Sequence[tuple[int, int]],
        *,
        corpus_sha256: str,
        min_pair_count: int,
        cache_size: int = 65536,
    ) -> None:
        if not _HEX64.match(corpus_sha256):
            raise TokenizerError("corpus_sha256 must be a 64-character lowercase hex digest")
        if min_pair_count < 1:
            raise TokenizerError("min_pair_count must be >= 1")
        if cache_size < 0:
            raise TokenizerError("cache_size must be >= 0")

        token_bytes: list[bytes] = [bytes([i]) for i in range(BYTE_VOCAB)]
        ranks: dict[tuple[int, int], int] = {}
        validated: list[tuple[int, int]] = []
        for rank, pair in enumerate(merges):
            if len(pair) != 2 or not all(type(x) is int for x in pair):
                raise TokenizerError(f"merge {rank} must be a pair of int ids")
            left, right = pair[0], pair[1]
            limit = BYTE_VOCAB + rank  # a merge may only reference earlier tokens
            if not (0 <= left < limit and 0 <= right < limit):
                raise TokenizerError(f"merge {rank} references an id that does not exist yet")
            if (left, right) in ranks:
                raise TokenizerError(f"merge {rank} duplicates an earlier merge")
            ranks[(left, right)] = rank
            validated.append((left, right))
            token_bytes.append(token_bytes[left] + token_bytes[right])

        # Rank order is the explicit validated sequence, not dict iteration order.
        self._merges: tuple[tuple[int, int], ...] = tuple(validated)
        self._ranks = ranks
        self._corpus_sha256 = corpus_sha256
        self._min_pair_count = min_pair_count
        self._vocab_size = BYTE_VOCAB + len(self._merges) + len(SPECIAL_TOKENS)
        self._special_ids = expected_special_token_ids(self._vocab_size)
        self._id_to_special = {i: name for name, i in self._special_ids.items()}
        token_bytes.extend(name.encode("utf-8") for name in SPECIAL_TOKENS)
        self._token_bytes = token_bytes
        self._cache: dict[bytes, tuple[int, ...]] = {}
        self._cache_size = cache_size

    # ------------------------------------------------------------------ properties

    @property
    def vocab_size(self) -> int:
        """Total number of ids: bytes + merges + special tokens."""
        return self._vocab_size

    @property
    def n_merges(self) -> int:
        """Number of learned merges."""
        return len(self._merges)

    @property
    def merges(self) -> tuple[tuple[int, int], ...]:
        """Merges in rank order."""
        return self._merges

    @property
    def corpus_sha256(self) -> str:
        """Order-independent hash of the training corpus documents."""
        return self._corpus_sha256

    @property
    def min_pair_count(self) -> int:
        """Minimum pair frequency required for a merge during training."""
        return self._min_pair_count

    @property
    def special_token_ids(self) -> dict[str, int]:
        """Exact special-token mapping (name -> id) of this tokenizer."""
        return dict(self._special_ids)

    @property
    def first_special_id(self) -> int:
        """Lowest special-token id; every ordinary token id is below it."""
        return self._vocab_size - len(SPECIAL_TOKENS)

    @property
    def eot_id(self) -> int:
        """Id of ``<|endoftext|>``."""
        return self._special_ids["<|endoftext|>"]

    @property
    def pad_id(self) -> int:
        """Id of ``<|pad|>``."""
        return self._special_ids["<|pad|>"]

    def token_bytes(self, token_id: int) -> bytes:
        """Return the bytes a token id decodes to."""
        self._check_id(token_id)
        return self._token_bytes[token_id]

    # ------------------------------------------------------------------ encoding

    def _bpe(self, chunk: bytes) -> tuple[int, ...]:
        """Apply merges to one chunk, always merging the lowest-rank adjacent pair first."""
        if self._cache_size:
            cached = self._cache.get(chunk)
            if cached is not None:
                return cached
        ids = list(chunk)
        ranks = self._ranks
        while len(ids) >= 2:
            best_rank = -1
            for pair in zip(ids, ids[1:], strict=False):
                rank = ranks.get(pair)
                if rank is not None and (best_rank < 0 or rank < best_rank):
                    best_rank = rank
            if best_rank < 0:
                break
            left, right = self._merges[best_rank]
            new_id = BYTE_VOCAB + best_rank
            merged: list[int] = []
            i = 0
            while i < len(ids):
                if i + 1 < len(ids) and ids[i] == left and ids[i + 1] == right:
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(ids[i])
                    i += 1
            ids = merged
        result = tuple(ids)
        if self._cache_size:
            if len(self._cache) >= self._cache_size:
                self._cache.clear()
            self._cache[chunk] = result
        return result

    def _encode_chunks(self, chunks: Iterable[bytes]) -> list[int]:
        out: list[int] = []
        for chunk in chunks:
            out.extend(self._bpe(chunk))
        return out

    def encode_bytes(self, data: bytes) -> list[int]:
        """Encode arbitrary bytes (including invalid UTF-8). Never emits special ids."""
        # surrogateescape maps undecodable bytes to lone surrogates and back, exactly.
        text = bytes(data).decode("utf-8", errors="surrogateescape")
        return self._encode_chunks(
            chunk.encode("utf-8", errors="surrogateescape") for chunk in pretokenize(text)
        )

    def _encode_ordinary(self, text: str) -> list[int]:
        return self._encode_chunks(chunk.encode("utf-8") for chunk in pretokenize(text))

    def encode(self, text: str, allowed_special: AbstractSet[str] = frozenset()) -> list[int]:
        """Encode text to ids.

        Args:
            text: Unicode text without lone surrogates.
            allowed_special: Special-token names that may be emitted as special ids when they
                appear literally in ``text``. Every other occurrence is encoded as ordinary
                bytes.

        Raises:
            TokenizerError: For lone surrogates or unknown names in ``allowed_special``.
        """
        try:
            text.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise TokenizerError(
                "text contains lone surrogates; use encode_bytes for arbitrary bytes"
            ) from exc
        unknown = set(allowed_special) - set(SPECIAL_TOKENS)
        if unknown:
            raise TokenizerError(f"unknown special tokens in allowed_special: {sorted(unknown)}")
        if not allowed_special:
            return self._encode_ordinary(text)

        names = sorted(allowed_special, key=lambda n: (-len(n), n))
        splitter = regex.compile("(" + "|".join(regex.escape(n) for n in names) + ")")
        out: list[int] = []
        for piece in splitter.split(text):
            if not piece:
                continue
            if piece in allowed_special:
                out.append(self._special_ids[piece])
            else:
                out.extend(self._encode_ordinary(piece))
        return out

    # ------------------------------------------------------------------ decoding

    def _check_id(self, token_id: int) -> None:
        if not 0 <= token_id < self._vocab_size:
            raise TokenizerError(f"token id {token_id} outside [0, {self._vocab_size})")

    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        """Decode ids to the exact bytes they represent."""
        parts: list[bytes] = []
        for token_id in ids:
            self._check_id(token_id)
            parts.append(self._token_bytes[token_id])
        return b"".join(parts)

    def decode(self, ids: Iterable[int]) -> str:
        """Decode ids to text; invalid UTF-8 (e.g. a cut-off sequence) becomes U+FFFD."""
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ artifacts

    def truncated(self, vocab_size: int) -> BPETokenizer:
        """Return the tokenizer with the first ``vocab_size - 288`` merges (BPE is greedy)."""
        if not MIN_VOCAB_SIZE <= vocab_size <= self._vocab_size:
            raise TokenizerError(
                f"cannot truncate a {self._vocab_size}-token tokenizer to {vocab_size}; "
                f"valid range is [{MIN_VOCAB_SIZE}, {self._vocab_size}]"
            )
        n_merges = vocab_size - MIN_VOCAB_SIZE
        return BPETokenizer(
            self._merges[:n_merges],
            corpus_sha256=self._corpus_sha256,
            min_pair_count=self._min_pair_count,
            cache_size=self._cache_size,
        )

    def to_artifact(self) -> dict[str, Any]:
        """Return the artifact as plain data (see ``to_json_bytes`` for canonical bytes)."""
        return {
            "corpus_sha256": self._corpus_sha256,
            "format": ARTIFACT_FORMAT,
            "format_version": ARTIFACT_FORMAT_VERSION,
            "merges": [list(pair) for pair in self._merges],
            "min_pair_count": self._min_pair_count,
            "pretokenizer": PRETOKENIZER_ID,
            "pretokenizer_pattern": PATTERN,
            "special_tokens": dict(self._special_ids),
            "vocab_size": self._vocab_size,
        }

    def to_json_bytes(self) -> bytes:
        """Canonical artifact bytes: sorted keys, compact, UTF-8, trailing newline."""
        text = json.dumps(
            self.to_artifact(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return (text + "\n").encode("utf-8")

    @property
    def sha256(self) -> str:
        """sha256 of the canonical artifact bytes (the ``tokenizer_sha256``)."""
        return hashlib.sha256(self.to_json_bytes()).hexdigest()

    def save(self, path: Path) -> None:
        """Write the canonical artifact bytes to ``path``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.to_json_bytes())

    @classmethod
    def from_json_bytes(cls, data: bytes, *, cache_size: int = 65536) -> BPETokenizer:
        """Load and validate an artifact.

        Raises:
            TokenizerError: If the artifact is malformed, from another format or
                pre-tokenizer, or records a special-token mapping that does not match.
        """
        try:
            raw = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TokenizerError(f"artifact is not valid UTF-8 JSON: {exc}") from exc
        return cls.from_artifact(raw, cache_size=cache_size)

    @classmethod
    def from_artifact(cls, raw: Any, *, cache_size: int = 65536) -> BPETokenizer:
        """Validate plain artifact data (as produced by ``to_artifact``) and build a tokenizer.

        Raises:
            TokenizerError: On missing/extra keys, wrong types, unsupported format or
                pre-tokenizer, invalid values, or a special-token mapping that does not match.
        """
        if type(raw) is not dict or set(raw) != _ARTIFACT_KEYS:
            raise TokenizerError(f"artifact keys must be exactly {sorted(_ARTIFACT_KEYS)}")
        # Types first (bool is an int subclass in Python, so compare exact types).
        expected_types: dict[str, type] = {
            "corpus_sha256": str,
            "format": str,
            "format_version": int,
            "merges": list,
            "min_pair_count": int,
            "pretokenizer": str,
            "pretokenizer_pattern": str,
            "special_tokens": dict,
            "vocab_size": int,
        }
        for key, expected in expected_types.items():
            if type(raw[key]) is not expected:
                raise TokenizerError(
                    f"artifact field {key!r} must be {expected.__name__}, "
                    f"got {type(raw[key]).__name__}"
                )
        merges = raw["merges"]
        if not all(
            type(m) is list and len(m) == 2 and all(type(x) is int for x in m) for m in merges
        ):
            raise TokenizerError("merges must be a list of [int, int] pairs")
        if not all(type(k) is str and type(v) is int for k, v in raw["special_tokens"].items()):
            raise TokenizerError("special_tokens must map token names (str) to ids (int)")
        if raw["format"] != ARTIFACT_FORMAT or raw["format_version"] != ARTIFACT_FORMAT_VERSION:
            raise TokenizerError("unsupported tokenizer artifact format/version")
        if raw["pretokenizer"] != PRETOKENIZER_ID or raw["pretokenizer_pattern"] != PATTERN:
            raise TokenizerError(
                "artifact was built with a different pre-tokenizer; retrain the tokenizer"
            )
        if raw["vocab_size"] < MIN_VOCAB_SIZE:
            raise TokenizerError(f"artifact vocab_size must be >= {MIN_VOCAB_SIZE}")
        if raw["min_pair_count"] < 1:
            raise TokenizerError("artifact min_pair_count must be >= 1")

        tokenizer = cls(
            [(m[0], m[1]) for m in merges],
            corpus_sha256=raw["corpus_sha256"],
            min_pair_count=raw["min_pair_count"],
            cache_size=cache_size,
        )
        if raw["vocab_size"] != tokenizer.vocab_size:
            raise TokenizerError(
                f"artifact vocab_size {raw['vocab_size']} does not match its merges "
                f"({tokenizer.vocab_size})"
            )
        if raw["special_tokens"] != tokenizer.special_token_ids:
            raise TokenizerError(
                "artifact special-token mapping does not match the layout for its vocabulary "
                "size; special tokens are never renumbered"
            )
        return tokenizer

    @classmethod
    def load(cls, path: Path, *, cache_size: int = 65536) -> BPETokenizer:
        """Load and validate an artifact file."""
        return cls.from_json_bytes(path.read_bytes(), cache_size=cache_size)
