"""Milestone A tokenizer acceptance tests (plan rev 3.3, section 2).

The module-scoped tokenizer is trained on a small fixed corpus defined here, so edits to the
smoke fixture corpus cannot change these expectations.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import pytest

from kittylm.tokenizer.bpe import (
    MIN_VOCAB_SIZE,
    SPECIAL_TOKENS,
    BPETokenizer,
    TokenizerError,
)
from kittylm.tokenizer.pretokenize import pretokenize
from kittylm.tokenizer.trainer import (
    TokenizerTrainingError,
    corpus_sha256,
    read_corpus_files,
    train_bpe,
)

FIXED_CORPUS: tuple[str, ...] = (
    "KittyLM tokenizer acceptance corpus. The quick brown fox jumps over the lazy dog.\n",
    "def add(a: int, b: int) -> int:\n    return a + b\n\n\nclass Box:\n    pass\n",
    'config = {"name": "kitty", "layers": 4, "dropout": 0.0, "tags": ["a", "b"]}\n',
    "model:\n  name: nano\n  heads: 4\nsteps:\n  - train\n  - eval\n",
    "# Heading\n\nSome *markdown* with `inline code` and a [link](docs/x.md).\n",
    "$ git status --short && python -m pytest -q | tail -n 5\n",
    "const f = (x) => { return x * 2; };\nconsole.log(f(21));\n",
    "Ünïcödé text, emoji 😺 and 日本語 appear too.\n",
) * 3
VOCAB = 512  # the fixed corpus supports at most 244 merges (measured); 512 needs 224

BOUNDARY_CODE_POINTS = (0x0, 0x7F, 0x80, 0x7FF, 0x800, 0xD7FF, 0xE000, 0xFFFD, 0xFFFF, 0x10000)
UNICODE_SAMPLES: tuple[str, ...] = (
    "",
    "hello world",
    "".join(chr(cp) for cp in (*BOUNDARY_CODE_POINTS, 0x10FFFF)),
    "e\u0301 a\u0308\u0323 n\u0303",  # combining marks
    # ZWJ sequences, a flag, a skin-tone modifier
    "\U0001f469\u200d\U0001f4bb \U0001f468\u200d\U0001f469\u200d\U0001f467 "
    "\U0001f1ec\U0001f1e7 \U0001f44d\U0001f3fd",
    "مرحبا بالعالم",  # Arabic (RTL)
    "שלום עולם",  # Hebrew (RTL)
    "日本語のテキスト 中文 한국어",  # CJK
    "\tdef f():\r\n\t\treturn 1\r\n",  # tabs, CRLF, indentation
    "    four spaces\n        eight spaces\n\n\n",
    "  leading and trailing  ",
    "\x00\x01\x7f\u2028\u2029\x0b\x0c",
    'def f(x):\n    """Doc."""\n    return {"k": [1, 2.5, None]}\n',
    '{"json": true, "n": -1.5e10, "s": "a\\"b"}',
    "key: value\nlist:\n  - one\n  - two\n",
    "## Markdown\n\n- item `code` **bold** _em_\n",
    "$ ls -la | grep '*.py' && echo \"$HOME\" > /dev/null 2>&1\n",
    "a" * 1000,
    "0123456789 1e-3 0xFF 3.14159",
)


@pytest.fixture(scope="module")
def tok() -> BPETokenizer:
    return train_bpe(FIXED_CORPUS, VOCAB, min_pair_count=1)


def assert_ordinary(tokenizer: BPETokenizer, ids: list[int]) -> None:
    """Ordinary encoding never produces a special-token id."""
    assert all(0 <= i < tokenizer.first_special_id for i in ids)


def random_scalar_string(rng: random.Random, length: int) -> str:
    chars = []
    while len(chars) < length:
        cp = rng.randrange(0, 0x110000)
        if 0xD800 <= cp <= 0xDFFF:  # surrogates are not Unicode scalar values
            continue
        chars.append(chr(cp))
    return "".join(chars)


# --- byte round-trip ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"\x00" * 8,
        b"\xff",
        b"\xc0\xaf",  # overlong encoding
        b"\xe2\x82",  # truncated sequence
        b"\x80\x80\x80",  # lone continuation bytes
        b"\xed\xa0\x80",  # encoded surrogate (invalid UTF-8)
        b"abc\xffdef\n\xfe",
        bytes(range(256)),
    ],
)
def test_byte_round_trip_edge_cases(tok: BPETokenizer, data: bytes) -> None:
    ids = tok.encode_bytes(data)
    assert tok.decode_bytes(ids) == data
    assert_ordinary(tok, ids)


def test_byte_round_trip_random(tok: BPETokenizer) -> None:
    rng = random.Random(20260917)
    for _ in range(200):
        data = bytes(rng.randrange(256) for _ in range(rng.randrange(0, 300)))
        ids = tok.encode_bytes(data)
        assert tok.decode_bytes(ids) == data
        assert_ordinary(tok, ids)


# --- Unicode round-trip ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", UNICODE_SAMPLES)
def test_unicode_round_trip_samples(tok: BPETokenizer, text: str) -> None:
    ids = tok.encode(text)
    assert tok.decode(ids) == text
    assert_ordinary(tok, ids)
    assert ids == tok.encode_bytes(text.encode("utf-8"))  # str and bytes paths agree


def test_unicode_round_trip_random_scalars(tok: BPETokenizer) -> None:
    rng = random.Random(7)
    for _ in range(150):
        text = random_scalar_string(rng, rng.randrange(0, 80))
        ids = tok.encode(text)
        assert tok.decode(ids) == text
        assert_ordinary(tok, ids)


@pytest.mark.parametrize("text", ["\ud800", "abc\udfffdef", "\udc80", "x\ud83d"])
def test_lone_surrogates_are_rejected_by_encode(tok: BPETokenizer, text: str) -> None:
    with pytest.raises(TokenizerError, match="lone surrogates"):
        tok.encode(text)


def test_pretokenize_is_lossless() -> None:
    rng = random.Random(3)
    samples = [*UNICODE_SAMPLES, *FIXED_CORPUS, "\udc80abc\udcff"]
    samples += [random_scalar_string(rng, rng.randrange(0, 60)) for _ in range(100)]
    for text in samples:
        chunks = pretokenize(text)
        assert "".join(chunks) == text
        assert all(chunks)


# --- special-token collision safety ---------------------------------------------------------------


def test_special_token_layout(tok: BPETokenizer) -> None:
    assert len(SPECIAL_TOKENS) == 32 and len(set(SPECIAL_TOKENS)) == 32
    first = tok.vocab_size - 32
    assert tok.first_special_id == first == 256 + tok.n_merges
    assert tok.special_token_ids == {name: first + i for i, name in enumerate(SPECIAL_TOKENS)}
    assert tok.eot_id == first and tok.pad_id == first + 1
    byte_ids = set(range(256))
    merge_ids = set(range(256, 256 + tok.n_merges))
    special_ids = set(tok.special_token_ids.values())
    assert not (byte_ids & merge_ids) and not (byte_ids & special_ids)
    assert not (merge_ids & special_ids)
    assert byte_ids | merge_ids | special_ids == set(range(tok.vocab_size))


@pytest.mark.parametrize("name", SPECIAL_TOKENS)
def test_every_special_literal_is_ordinary_unless_allowed(tok: BPETokenizer, name: str) -> None:
    sid = tok.special_token_ids[name]
    for text in (name, f"before {name} after", f"{name}{name}", f"x{name}y\n{name}"):
        ids = tok.encode(text)
        assert sid not in ids
        assert_ordinary(tok, ids)
        assert tok.decode(ids) == text

    assert tok.encode(name, allowed_special={name}) == [sid]
    embedded = tok.encode(f"before {name} after", allowed_special={name})
    assert embedded == tok.encode("before ") + [sid] + tok.encode(" after")
    adjacent = tok.encode(f"{name}{name}", allowed_special={name})
    assert adjacent == [sid, sid]
    assert tok.decode(embedded) == f"before {name} after"


@pytest.mark.parametrize("allowed", SPECIAL_TOKENS)
def test_allowing_one_token_never_emits_others(tok: BPETokenizer, allowed: str) -> None:
    text = "".join(f"a{name}" for name in SPECIAL_TOKENS)
    ids = tok.encode(text, allowed_special={allowed})
    special = set(tok.special_token_ids.values())
    assert [i for i in ids if i in special] == [tok.special_token_ids[allowed]]
    assert tok.decode(ids) == text


def test_system_token_exactly(tok: BPETokenizer) -> None:
    sid = tok.special_token_ids["<|system|>"]
    assert sid not in tok.encode("<|system|>")
    assert tok.encode("<|system|>", allowed_special={"<|system|>"}) == [sid]
    mixed = tok.encode("say <|system|> then <|user|>!", allowed_special={"<|system|>"})
    assert mixed == tok.encode("say ") + [sid] + tok.encode(" then <|user|>!")
    assert tok.special_token_ids["<|user|>"] not in mixed


@pytest.mark.parametrize(
    "text",
    [
        "<|endo<|endoftext|>ftext|>",
        "<|endoftext|",
        "|endoftext|>",
        "<|ENDOFTEXT|>",
        "< |endoftext|>",
        "<||>",
        "<|endoftext|><|endoftext|>",
        "<|reserved_1",
        "reserved_10|>",
        "<|reserved_100|>",
        "<|reserved_22|>",
        "<<|pad|>>",
    ],
)
def test_partial_and_nested_forms_are_ordinary(tok: BPETokenizer, text: str) -> None:
    ids = tok.encode(text)
    assert_ordinary(tok, ids)
    assert tok.decode(ids) == text


def test_nested_form_with_allowed_inner_token(tok: BPETokenizer) -> None:
    eot = tok.eot_id
    ids = tok.encode("<|endo<|endoftext|>ftext|>", allowed_special={"<|endoftext|>"})
    assert ids == tok.encode("<|endo") + [eot] + tok.encode("ftext|>")


def test_prefix_like_reserved_names(tok: BPETokenizer) -> None:
    r1 = tok.special_token_ids["<|reserved_1|>"]
    r10 = tok.special_token_ids["<|reserved_10|>"]
    text = "<|reserved_1|><|reserved_10|><|reserved_100|>"
    only_1 = tok.encode(text, allowed_special={"<|reserved_1|>"})
    assert only_1 == [r1] + tok.encode("<|reserved_10|><|reserved_100|>")
    only_10 = tok.encode(text, allowed_special={"<|reserved_10|>"})
    assert only_10 == tok.encode("<|reserved_1|>") + [r10] + tok.encode("<|reserved_100|>")
    both = tok.encode(text, allowed_special={"<|reserved_1|>", "<|reserved_10|>"})
    assert both == [r1, r10] + tok.encode("<|reserved_100|>")
    assert tok.decode(both) == text


def test_mixed_allowed_and_disallowed(tok: BPETokenizer) -> None:
    allowed = {"<|user|>", "<|final|>"}
    text = "<|system|>s<|user|>u<|plan|>p<|final|>f<|endoftext|>"
    ids = tok.encode(text, allowed_special=allowed)
    special = set(tok.special_token_ids.values())
    emitted = [i for i in ids if i in special]
    assert emitted == [tok.special_token_ids["<|user|>"], tok.special_token_ids["<|final|>"]]
    assert tok.decode(ids) == text


def test_unknown_allowed_special_is_rejected(tok: BPETokenizer) -> None:
    with pytest.raises(TokenizerError, match="unknown special tokens"):
        tok.encode("x", allowed_special={"<|made_up|>"})
    assert tok.encode("<|pad|>", allowed_special=frozenset()) == tok.encode("<|pad|>")


@pytest.mark.parametrize("name", SPECIAL_TOKENS)
def test_decoding_special_ids_yields_literal_markers(tok: BPETokenizer, name: str) -> None:
    sid = tok.special_token_ids[name]
    assert tok.decode_bytes([sid]) == name.encode("utf-8")
    assert tok.decode([sid]) == name


def test_corpus_with_special_literals_never_renumbers(tok: BPETokenizer) -> None:
    noisy = [doc + "<|endoftext|><|system|>" * 3 for doc in FIXED_CORPUS]
    noisy_tok = train_bpe(noisy, VOCAB, min_pair_count=1)
    assert noisy_tok.special_token_ids == tok.special_token_ids
    for name in SPECIAL_TOKENS:
        assert_ordinary(noisy_tok, noisy_tok.encode(name))


def test_special_ids_are_fixed_per_size_but_differ_across_sizes(tok: BPETokenizer) -> None:
    smaller = tok.truncated(VOCAB - 64)
    assert smaller.eot_id == tok.eot_id - 64
    assert json.loads(smaller.to_json_bytes())["special_tokens"] == smaller.special_token_ids
    assert json.loads(tok.to_json_bytes())["special_tokens"] == tok.special_token_ids


# --- deterministic artifact hashes ----------------------------------------------------------------


def test_training_twice_is_byte_identical(tok: BPETokenizer) -> None:
    again = train_bpe(FIXED_CORPUS, VOCAB, min_pair_count=1)
    assert again.to_json_bytes() == tok.to_json_bytes()
    assert again.sha256 == tok.sha256


def test_shuffled_document_order_gives_same_hash(tok: BPETokenizer) -> None:
    docs = list(FIXED_CORPUS)
    random.Random(11).shuffle(docs)
    assert train_bpe(docs, VOCAB, min_pair_count=1).sha256 == tok.sha256


def test_input_directory_does_not_change_hash(tmp_path: Path) -> None:
    hashes = []
    for folder in ("first_location", "a/much/deeper/other_location"):
        root = tmp_path / folder
        root.mkdir(parents=True)
        paths = []
        for i, doc in enumerate(FIXED_CORPUS):
            path = root / f"doc{i}.txt"
            path.write_bytes(doc.encode("utf-8"))
            paths.append(path)
        hashes.append(train_bpe(read_corpus_files(paths), 352, min_pair_count=1).sha256)
    assert hashes[0] == hashes[1]


@pytest.mark.parametrize("size", [MIN_VOCAB_SIZE, 352, 416, VOCAB])
def test_truncation_equals_direct_training(tok: BPETokenizer, size: int) -> None:
    direct = train_bpe(FIXED_CORPUS, size, min_pair_count=1)
    assert tok.truncated(size).to_json_bytes() == direct.to_json_bytes()


@pytest.mark.parametrize("size", [MIN_VOCAB_SIZE - 1, VOCAB + 64])
def test_invalid_truncation_sizes(tok: BPETokenizer, size: int) -> None:
    with pytest.raises(TokenizerError, match="cannot truncate"):
        tok.truncated(size)


def test_save_load_save_is_byte_identical(tok: BPETokenizer, tmp_path: Path) -> None:
    first = tmp_path / "a" / "tokenizer.json"
    tok.save(first)
    loaded = BPETokenizer.load(first)
    second = tmp_path / "b" / "tokenizer.json"
    loaded.save(second)
    assert first.read_bytes() == second.read_bytes() == tok.to_json_bytes()
    assert loaded.encode(UNICODE_SAMPLES[12]) == tok.encode(UNICODE_SAMPLES[12])


def test_artifact_is_canonical_and_path_free(tok: BPETokenizer) -> None:
    data = tok.to_json_bytes()
    assert data.endswith(b"\n") and b"\r" not in data
    text = data.decode("utf-8")
    canonical = json.dumps(
        json.loads(text), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    assert text == canonical + "\n"
    parsed = json.loads(text)
    assert set(parsed) == set(tok.to_artifact())
    parsed.pop("pretokenizer_pattern")  # the regex legitimately contains backslashes
    scanned = json.dumps(parsed)
    for forbidden in (":\\\\", "/Users/", "/home/", "C:/", "timestamp", "created"):
        assert forbidden not in scanned


def test_corpus_hash_order_and_multiplicity() -> None:
    assert corpus_sha256(["a", "b"]) == corpus_sha256(["b", "a"])
    assert corpus_sha256(["a"]) != corpus_sha256(["a", "a"])
    assert corpus_sha256(["ab"]) != corpus_sha256(["a", "b"])


# --- core behaviour -------------------------------------------------------------------------------

ZERO_HASH = "0" * 64


def test_merges_apply_in_rank_order() -> None:
    first = BPETokenizer([(97, 98), (98, 99)], corpus_sha256=ZERO_HASH, min_pair_count=1)
    assert first.encode("abc") == [256, 99]
    second = BPETokenizer([(98, 99), (97, 98)], corpus_sha256=ZERO_HASH, min_pair_count=1)
    assert second.encode("abc") == [97, 256]


def test_cache_does_not_change_results(tok: BPETokenizer) -> None:
    uncached = BPETokenizer.from_json_bytes(tok.to_json_bytes(), cache_size=0)
    tiny_cache = BPETokenizer.from_json_bytes(tok.to_json_bytes(), cache_size=2)
    for text in (*UNICODE_SAMPLES, *FIXED_CORPUS, *UNICODE_SAMPLES):
        expected = uncached.encode(text)
        assert tok.encode(text) == expected
        assert tiny_cache.encode(text) == expected


def test_vocab_size_is_exact(tok: BPETokenizer) -> None:
    assert tok.vocab_size == VOCAB
    assert tok.n_merges == VOCAB - MIN_VOCAB_SIZE
    for token_id in range(tok.vocab_size):
        assert tok.token_bytes(token_id)
    with pytest.raises(TokenizerError, match="outside"):
        tok.decode([VOCAB])
    with pytest.raises(TokenizerError, match="outside"):
        tok.decode_bytes([-1])


def test_insufficient_corpus_fails_clearly() -> None:
    with pytest.raises(TokenizerTrainingError) as exc:
        train_bpe(["abab"], MIN_VOCAB_SIZE + 64, min_pair_count=1)
    message = str(exc.value)
    assert "supports only" in message and "requires 64" in message


def test_vocab_size_below_minimum_fails() -> None:
    with pytest.raises(TokenizerTrainingError, match=f">= {MIN_VOCAB_SIZE}"):
        train_bpe(FIXED_CORPUS, MIN_VOCAB_SIZE - 1)


def test_invalid_corpus_file_encoding(tmp_path: Path) -> None:
    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"ok\xff")
    with pytest.raises(TokenizerTrainingError, match="not valid UTF-8"):
        read_corpus_files([bad])


# --- artifact malformation ------------------------------------------------------------------------


def artifact(tok: BPETokenizer) -> dict[str, Any]:
    return copy.deepcopy(tok.truncated(MIN_VOCAB_SIZE + 4).to_artifact())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("format_version", True, "'format_version' must be int"),
        ("format_version", "1", "'format_version' must be int"),
        ("vocab_size", True, "'vocab_size' must be int"),
        ("vocab_size", "292", "'vocab_size' must be int"),
        ("vocab_size", 292.0, "'vocab_size' must be int"),
        ("min_pair_count", True, "'min_pair_count' must be int"),
        ("min_pair_count", 1.0, "'min_pair_count' must be int"),
        ("merges", {"0": [97, 98]}, "'merges' must be list"),
        ("special_tokens", list(SPECIAL_TOKENS), "'special_tokens' must be dict"),
        ("corpus_sha256", 0, "'corpus_sha256' must be str"),
        ("pretokenizer_pattern", None, "'pretokenizer_pattern' must be str"),
        ("format", 1, "'format' must be str"),
    ],
)
def test_artifact_wrong_types(tok: BPETokenizer, field: str, value: Any, message: str) -> None:
    raw = artifact(tok)
    raw[field] = value
    with pytest.raises(TokenizerError, match=message):
        BPETokenizer.from_artifact(raw)


@pytest.mark.parametrize(
    "merges",
    [[[97.0, 98]], [[True, 98]], [[97, False]], [[97]], [[97, 98, 99]], [(97, 98)], ["ab"]],
)
def test_artifact_bad_merge_entries(tok: BPETokenizer, merges: list[Any]) -> None:
    raw = artifact(tok)
    raw["merges"] = merges
    with pytest.raises(TokenizerError, match="merges must be a list of"):
        BPETokenizer.from_artifact(raw)


@pytest.mark.parametrize(
    "mapping_change",
    ["non_string_name", "string_id", "bool_id"],
)
def test_artifact_bad_special_token_types(tok: BPETokenizer, mapping_change: str) -> None:
    raw = artifact(tok)
    mapping = raw["special_tokens"]
    if mapping_change == "non_string_name":
        mapping[7] = mapping.pop("<|pad|>")
    elif mapping_change == "string_id":
        mapping["<|pad|>"] = str(mapping["<|pad|>"])
    else:
        mapping["<|pad|>"] = True
    with pytest.raises(TokenizerError, match="special_tokens must map"):
        BPETokenizer.from_artifact(raw)


def mutate_values(raw: dict[str, Any], case: str) -> None:
    mapping = raw["special_tokens"]
    if case == "format":
        raw["format"] = "other-bpe"
    elif case == "format_version":
        raw["format_version"] = 2
    elif case == "pretokenizer":
        raw["pretokenizer"] = "kittylm-code-v0"
    elif case == "pattern":
        raw["pretokenizer_pattern"] = raw["pretokenizer_pattern"] + "|x"
    elif case == "vocab_mismatch":
        raw["vocab_size"] += 64
    elif case == "vocab_too_small":
        raw["vocab_size"] = 100
    elif case == "min_pair_count":
        raw["min_pair_count"] = 0
    elif case == "future_id":
        raw["merges"][0] = [300, 97]
    elif case == "duplicate_merge":
        raw["merges"][1] = list(raw["merges"][0])
    elif case == "corpus_hash":
        raw["corpus_sha256"] = "abc"
    elif case == "renumbered":
        eot = mapping["<|endoftext|>"]
        mapping["<|endoftext|>"] = mapping["<|pad|>"]
        mapping["<|pad|>"] = eot
    elif case == "missing_special":
        mapping.pop("<|final|>")
    elif case == "extra_special":
        mapping["<|extra|>"] = 999
    elif case == "missing_key":
        raw.pop("min_pair_count")
    elif case == "extra_key":
        raw["created_at"] = "2026-09-17"


VALUE_CASES = {
    "format": "unsupported tokenizer artifact format",
    "format_version": "unsupported tokenizer artifact format",
    "pretokenizer": "different pre-tokenizer",
    "pattern": "different pre-tokenizer",
    "vocab_mismatch": "does not match its merges",
    "vocab_too_small": f">= {MIN_VOCAB_SIZE}",
    "min_pair_count": "min_pair_count must be >= 1",
    "future_id": "does not exist yet",
    "duplicate_merge": "duplicates an earlier merge",
    "corpus_hash": "corpus_sha256 must be",
    "renumbered": "never renumbered",
    "missing_special": "never renumbered",
    "extra_special": "never renumbered",
    "missing_key": "keys must be exactly",
    "extra_key": "keys must be exactly",
}


@pytest.mark.parametrize(("case", "message"), sorted(VALUE_CASES.items()))
def test_artifact_wrong_values(tok: BPETokenizer, case: str, message: str) -> None:
    raw = artifact(tok)
    mutate_values(raw, case)
    with pytest.raises(TokenizerError, match=message):
        BPETokenizer.from_artifact(raw)
    data = (json.dumps(raw, sort_keys=True) + "\n").encode("utf-8")
    with pytest.raises(TokenizerError, match=message):
        BPETokenizer.from_json_bytes(data)


@pytest.mark.parametrize("data", [b"not json", b"\xff\xfe", b"[1, 2]", b"null", b"{}"])
def test_artifact_not_an_object(data: bytes) -> None:
    with pytest.raises(TokenizerError):
        BPETokenizer.from_json_bytes(data)


def test_valid_artifact_round_trips(tok: BPETokenizer) -> None:
    raw = artifact(tok)
    assert BPETokenizer.from_artifact(raw).to_artifact() == raw
