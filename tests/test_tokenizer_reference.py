"""Differential tests: the optimized BPE trainer against a slow, obviously-correct reference.

The optimized trainer keeps incremental pair counts, a pair -> chunk index with stale entries,
a lazy heap, and ``Counter.subtract`` deltas with zero/negative entries. That bookkeeping is the
highest-risk correctness area, so these tests (not manual reasoning) are its oracle: the
reference recomputes every pair count from scratch after every merge.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence

import pytest

from kittylm.tokenizer.bpe import MIN_VOCAB_SIZE, BPETokenizer
from kittylm.tokenizer.pretokenize import pretokenize
from kittylm.tokenizer.trainer import TokenizerTrainingError, corpus_sha256, train_bpe


def naive_merge(seq: Sequence[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out: list[int] = []
    i = 0
    while i < len(seq):
        if i < len(seq) - 1 and (seq[i], seq[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def reference_merges(
    documents: Sequence[str], max_merges: int, min_pair_count: int
) -> list[tuple[int, int]]:
    """Slow BPE: recount every adjacent pair over every chunk before each merge."""
    counts = Counter(chunk.encode("utf-8") for doc in documents for chunk in pretokenize(doc))
    words = {word: list(word) for word in counts}
    merges: list[tuple[int, int]] = []
    while len(merges) < max_merges:
        pair_counts: Counter[tuple[int, int]] = Counter()
        for word, seq in words.items():
            for i in range(len(seq) - 1):
                pair_counts[(seq[i], seq[i + 1])] += counts[word]
        if not pair_counts:
            break
        best = min(pair_counts, key=lambda p: (-pair_counts[p], p[0], p[1]))
        if pair_counts[best] < min_pair_count:
            break
        new_id = 256 + len(merges)
        merges.append(best)
        words = {word: naive_merge(seq, best, new_id) for word, seq in words.items()}
    return merges


def assert_matches_reference(documents: Sequence[str], min_pair_count: int) -> None:
    """Compare merge sequences, artifact bytes and failure behaviour at several sizes."""
    reference = reference_merges(documents, 400, min_pair_count)
    possible = len(reference)
    sizes = sorted({MIN_VOCAB_SIZE, MIN_VOCAB_SIZE + possible // 2, MIN_VOCAB_SIZE + possible})
    for vocab_size in sizes:
        n = vocab_size - MIN_VOCAB_SIZE
        trained = train_bpe(documents, vocab_size, min_pair_count=min_pair_count)
        assert list(trained.merges) == reference[:n], f"merge mismatch at vocab {vocab_size}"
        expected = BPETokenizer(
            reference[:n],
            corpus_sha256=corpus_sha256(documents),
            min_pair_count=min_pair_count,
        )
        assert trained.to_json_bytes() == expected.to_json_bytes()
        assert trained.sha256 == expected.sha256
    if possible < 400:
        with pytest.raises(TokenizerTrainingError, match=f"supports only {possible} merges"):
            train_bpe(documents, MIN_VOCAB_SIZE + possible + 1, min_pair_count=min_pair_count)


# --- hand-checked overlapping-pair cases ---------------------------------------------------------


def test_overlapping_run_aaaa() -> None:
    # (a,a) occurs 3 times (overlapping) -> "aaaa" becomes [X, X] -> (X,X) -> [Y].
    tok = train_bpe(["aaaa"], MIN_VOCAB_SIZE + 2, min_pair_count=1)
    assert list(tok.merges) == [(97, 97), (256, 256)]
    assert tok.encode("aaaa") == [257]


def test_alternating_abababa() -> None:
    # ab=3 and ba=3 tie -> smallest pair (a,b). "abababa" -> [X,X,X,a]; (X,X)=2 wins;
    # left-to-right merge gives [Y,X,a]; (Y,X)=1 and (X,a)=1 tie -> (X,a)=(256,97) is smaller.
    tok = train_bpe(["abababa"], MIN_VOCAB_SIZE + 4, min_pair_count=1)
    assert list(tok.merges) == [(97, 98), (256, 256), (256, 97), (257, 258)]
    assert tok.encode("abababa") == [259]


@pytest.mark.parametrize(
    "documents",
    [
        ["aaaa"],
        ["aaaaaaa"],
        ["abababa"],
        ["aabaab aabaab aab"],
        ["abcabcabc abcabc abc"],
        ["aaaa aaa aa a aaaa aaa"],
        ["xyxyxyxy", "yxyxyx", "xxyyxxyy"],
        ["aaaa"] * 5 + ["aaa"] * 3,
        ["zzzzzzzzzzzzzzzz"],
    ],
)
@pytest.mark.parametrize("min_pair_count", [1, 2])
def test_repeated_and_overlapping_runs_match_reference(
    documents: list[str], min_pair_count: int
) -> None:
    assert_matches_reference(documents, min_pair_count)


# --- seeded random corpora ------------------------------------------------------------------------


def random_corpus(seed: int) -> list[str]:
    rng = random.Random(seed)
    alphabet = rng.choice(["ab", "abc", "abcd", "ab \n", "ab_(). ", "aé1 \t"])
    return [
        "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 60)))
        for _ in range(rng.randint(1, 8))
    ]


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("min_pair_count", [1, 2, 3])
def test_random_corpora_match_reference(seed: int, min_pair_count: int) -> None:
    assert_matches_reference(random_corpus(seed), min_pair_count)


def test_code_like_corpus_matches_reference() -> None:
    corpus = [
        "def f(x):\n    return x + 1\n\n\ndef g(y):\n    return y * 2\n",
        "for i in range(10):\n    total += i\n",
        '{"a": 1, "b": [1, 2, 3], "a": 1}\n',
    ] * 2
    assert_matches_reference(corpus, 2)
