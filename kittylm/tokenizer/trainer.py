"""BPE training: learn merges from a corpus, deterministically.

Purpose:
    Learn the merge list of a byte-level BPE tokenizer. Starting from bytes, repeatedly find
    the most frequent adjacent token pair across the corpus and merge it into a new token,
    until the requested vocabulary size is reached. The trainer works on *unique chunks*
    (from ``pretokenize``) weighted by frequency, and keeps incremental pair counts, an index
    from pair to the chunks containing it, and a max-heap with lazy invalidation, so each
    merge only touches the chunks it affects.

    Also defines the ``kind: tokenizer`` configuration (``TokenizerConfig``).

Public API:
    train_bpe(documents, vocab_size, *, min_pair_count=2) -> BPETokenizer
    corpus_sha256(documents) -> str
    read_corpus_files(paths) -> list[str]
    TokenizerConfig, train_from_config(config, repo_root) -> BPETokenizer
    TokenizerTrainingError

Invariants:
    - Deterministic: identical documents (in any order, from any directory) give identical
      merges and artifact bytes. Ties are broken by highest count, then smallest left id, then
      smallest right id.
    - Greedy: the first N merges of a larger run equal a run trained directly to the smaller
      size, so smaller vocabularies are exact truncations.
    - Training treats text literally: special-token strings in the corpus are ordinary bytes
      and never produce or renumber special ids.

Failure modes:
    - If the corpus cannot provide enough merges whose pair count is at least
      ``min_pair_count``, training raises TokenizerTrainingError stating how many merges were
      possible and how many were required. It never returns a smaller vocabulary.
    - ``vocab_size`` below 288 (256 bytes + 32 special tokens) raises TokenizerTrainingError.
    - Corpus files must be strict UTF-8; invalid files raise TokenizerTrainingError.

See:
    D-001 (byte-level BPE), D-002 (train once to 32,768 and truncate), D-020 (fail on an
    insufficient corpus), plan rev 3.3 section 2.
"""

from __future__ import annotations

import hashlib
import heapq
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from kittylm.config import ConfigError, register_config_kind
from kittylm.tokenizer.bpe import (
    BYTE_VOCAB,
    MIN_VOCAB_SIZE,
    BPETokenizer,
    TokenizerError,
)
from kittylm.tokenizer.pretokenize import pretokenize

__all__ = [
    "TokenizerConfig",
    "TokenizerTrainingError",
    "corpus_sha256",
    "read_corpus_files",
    "train_bpe",
    "train_from_config",
]


class TokenizerTrainingError(TokenizerError):
    """Tokenizer training could not produce the requested tokenizer."""


def corpus_sha256(documents: Iterable[str]) -> str:
    """Order-independent corpus hash: sha256 over the sorted per-document sha256 digests."""
    digests = sorted(hashlib.sha256(doc.encode("utf-8")).hexdigest() for doc in documents)
    return hashlib.sha256("\n".join(digests).encode("ascii")).hexdigest()


def _pairs(seq: Sequence[int]) -> Counter[tuple[int, int]]:
    return Counter(zip(seq, seq[1:], strict=False))


def _merge(seq: list[int], left: int, right: int, new_id: int) -> list[int]:
    out: list[int] = []
    i = 0
    while i < len(seq):
        if i + 1 < len(seq) and seq[i] == left and seq[i + 1] == right:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def train_bpe(
    documents: Iterable[str], vocab_size: int, *, min_pair_count: int = 2
) -> BPETokenizer:
    """Train a byte-level BPE tokenizer of exactly ``vocab_size`` ids.

    Args:
        documents: Training documents (valid Unicode text).
        vocab_size: Target size including 256 byte ids and 32 special tokens.
        min_pair_count: A merge requires its pair to occur at least this many times.

    Raises:
        TokenizerTrainingError: If ``vocab_size`` is too small, a document is not valid
            Unicode, or the corpus cannot supply enough merges.
    """
    if vocab_size < MIN_VOCAB_SIZE:
        raise TokenizerTrainingError(
            f"vocab_size must be >= {MIN_VOCAB_SIZE} (256 bytes + special tokens)"
        )
    if min_pair_count < 1:
        raise TokenizerTrainingError("min_pair_count must be >= 1")
    target_merges = vocab_size - MIN_VOCAB_SIZE

    docs = list(documents)
    chunk_counts: Counter[bytes] = Counter()
    for index, doc in enumerate(docs):
        try:
            doc.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise TokenizerTrainingError(f"document {index} is not valid Unicode text") from exc
        for chunk in pretokenize(doc):
            chunk_counts[chunk.encode("utf-8")] += 1

    words = sorted(chunk_counts)  # deterministic order, independent of document order
    seqs: list[list[int]] = [list(word) for word in words]
    freqs: list[int] = [chunk_counts[word] for word in words]

    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    pair_words: dict[tuple[int, int], set[int]] = defaultdict(set)
    for wi, seq in enumerate(seqs):
        for pair, n in _pairs(seq).items():
            pair_counts[pair] += n * freqs[wi]
            pair_words[pair].add(wi)

    heap: list[tuple[int, int, int]] = [(-c, a, b) for (a, b), c in pair_counts.items()]
    heapq.heapify(heap)

    merges: list[tuple[int, int]] = []
    while len(merges) < target_merges:
        best: tuple[int, int] | None = None
        while heap:
            neg_count, a, b = heapq.heappop(heap)
            if pair_counts.get((a, b), 0) == -neg_count:  # skip stale heap entries
                best = (a, b)
                break
        if best is None or pair_counts[best] < min_pair_count:
            raise TokenizerTrainingError(
                f"corpus supports only {len(merges)} merges with pair count >= "
                f"{min_pair_count}, but vocab_size {vocab_size} requires {target_merges}; "
                "use a larger corpus, a smaller vocab_size or a lower min_pair_count"
            )

        new_id = BYTE_VOCAB + len(merges)
        merges.append(best)
        left, right = best
        changed: set[tuple[int, int]] = set()
        for wi in sorted(pair_words.pop(best, set())):
            old = seqs[wi]
            new = _merge(old, left, right, new_id)
            if new == old:
                continue  # stale index entry
            delta = _pairs(new)
            delta.subtract(_pairs(old))
            for pair, n in delta.items():
                if n:
                    pair_counts[pair] += n * freqs[wi]
                    changed.add(pair)
                    if n > 0:
                        pair_words[pair].add(wi)
            seqs[wi] = new
        pair_counts.pop(best, None)
        for pair in changed:
            count = pair_counts.get(pair, 0)
            if count > 0:
                heapq.heappush(heap, (-count, pair[0], pair[1]))
            else:
                pair_counts.pop(pair, None)

    return BPETokenizer(merges, corpus_sha256=corpus_sha256(docs), min_pair_count=min_pair_count)


def read_corpus_files(paths: Sequence[Path]) -> list[str]:
    """Read corpus files as strict UTF-8 bytes (no newline translation)."""
    documents: list[str] = []
    for path in paths:
        try:
            documents.append(path.read_bytes().decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise TokenizerTrainingError(f"{path.name} is not valid UTF-8") from exc
    return documents


@dataclass(frozen=True)
class TokenizerConfig:
    """``kind: tokenizer`` configuration (see ``configs/tokenizer/``).

    Two vocabulary floors exist on purpose. ``train_bpe`` accepts the theoretical minimum of
    288 ids (256 bytes + 32 special tokens, zero merges), which tests use. Config-driven
    tokenizers must additionally be a multiple of 64 (D-002), so the smallest valid
    ``vocab_size`` here is 320.
    """

    name: str
    purpose: Literal["smoke", "research"]
    vocab_size: int
    corpus_files: list[str]
    min_pair_count: int = 2
    notes: str = ""

    def __post_init__(self) -> None:
        """Validate sizes and require repository-relative POSIX corpus paths."""
        if self.vocab_size < MIN_VOCAB_SIZE:
            raise ConfigError(f"vocab_size must be >= {MIN_VOCAB_SIZE}")
        if self.vocab_size % 64:
            raise ConfigError("vocab_size must be a multiple of 64 (D-002)")
        if self.min_pair_count < 1:
            raise ConfigError("min_pair_count must be >= 1")
        if not self.corpus_files:
            raise ConfigError("corpus_files must list at least one file")
        for entry in self.corpus_files:
            path = PurePosixPath(entry)
            if path.is_absolute() or ":" in entry or "\\" in entry or ".." in path.parts:
                raise ConfigError(
                    f"corpus_files entry {entry!r} must be a repository-relative POSIX path"
                )


register_config_kind("tokenizer", TokenizerConfig)


def train_from_config(config: TokenizerConfig, repo_root: Path) -> BPETokenizer:
    """Train the tokenizer described by ``config`` from files under ``repo_root``.

    Corpus paths must resolve inside ``repo_root`` (symlinks escaping it are rejected).
    """
    root = repo_root.resolve()
    paths: list[Path] = []
    for entry in config.corpus_files:
        path = (root / PurePosixPath(entry)).resolve()
        if not path.is_relative_to(root):
            raise TokenizerTrainingError(f"corpus file {entry!r} resolves outside the repository")
        if not path.is_file():
            raise TokenizerTrainingError(f"corpus file not found: {entry!r}")
        paths.append(path)
    return train_bpe(
        read_corpus_files(paths),
        config.vocab_size,
        min_pair_count=config.min_pair_count,
    )
