"""Smoke tokenizer configuration and CLI, through the real config system."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from kittylm.config import CONFIG_KINDS, ConfigError, from_dict, load_yaml, validate_config_tree
from kittylm.tokenizer.bpe import BPETokenizer
from kittylm.tokenizer.trainer import (
    TokenizerConfig,
    TokenizerTrainingError,
    read_corpus_files,
    train_bpe,
    train_from_config,
)
from tests.conftest import ROOT

SMOKE = ROOT / "configs" / "tokenizer" / "smoke.yaml"


def load_smoke() -> TokenizerConfig:
    data = load_yaml(SMOKE)
    assert data["kind"] == "tokenizer"
    return from_dict(TokenizerConfig, {k: v for k, v in data.items() if k != "kind"})


def test_smoke_config_through_config_system() -> None:
    config = load_smoke()
    assert config.purpose == "smoke"
    assert config.vocab_size == 512
    assert config.min_pair_count == 1
    assert config.corpus_files == ["tests/fixtures/smoke_corpus.txt"]
    assert config.notes.startswith("Temporary SMOKE-GPU-001 vocabulary")  # top-level field
    assert (ROOT / config.corpus_files[0]).is_file()
    assert CONFIG_KINDS["tokenizer"] is TokenizerConfig


def test_all_configs_validate() -> None:
    count, errors = validate_config_tree(ROOT / "configs")
    assert errors == []
    assert count >= 1


def test_smoke_tokenizer_trains_exactly() -> None:
    config = load_smoke()
    tokenizer = train_from_config(config, ROOT)
    assert tokenizer.vocab_size == 512
    assert tokenizer.min_pair_count == 1
    (corpus,) = read_corpus_files([ROOT / config.corpus_files[0]])
    ids = tokenizer.encode(corpus)
    assert tokenizer.decode(ids) == corpus
    assert all(i < tokenizer.first_special_id for i in ids)
    assert train_from_config(config, ROOT).sha256 == tokenizer.sha256


def test_research_default_min_pair_count_is_insufficient_for_smoke_fixture() -> None:
    # Documents why smoke.yaml uses min_pair_count: 1 (research tokenizers keep 2).
    config = load_smoke()
    strict = TokenizerConfig(
        name=config.name,
        purpose="research",
        vocab_size=config.vocab_size,
        corpus_files=config.corpus_files,
    )
    assert strict.min_pair_count == 2
    with pytest.raises(TokenizerTrainingError, match="supports only 158 merges"):
        train_from_config(strict, ROOT)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"vocab_size": 500}, "multiple of 64"),
        ({"vocab_size": 256}, ">= 288"),
        ({"min_pair_count": 0}, "min_pair_count"),
        ({"corpus_files": []}, "at least one file"),
        ({"corpus_files": ["/etc/passwd"]}, "repository-relative"),
        ({"corpus_files": ["C:/data/x.txt"]}, "repository-relative"),
        ({"corpus_files": ["..\\x.txt"]}, "repository-relative"),
        ({"corpus_files": ["tests/../../x.txt"]}, "repository-relative"),
        ({"purpose": "production"}, "expected one of"),
    ],
)
def test_config_validation(changes: dict[str, object], message: str) -> None:
    data = {
        "name": "x",
        "purpose": "smoke",
        "vocab_size": 512,
        "corpus_files": ["tests/fixtures/smoke_corpus.txt"],
        **changes,
    }
    with pytest.raises(ConfigError, match=message):
        from_dict(TokenizerConfig, data)


def test_vocab_floors_are_intentionally_different() -> None:
    # train_bpe accepts the theoretical minimum (288 = 256 bytes + 32 specials, 0 merges) ...
    assert train_bpe(["abc"], 288, min_pair_count=1).vocab_size == 288
    # ... but config-driven tokenizers must be a multiple of 64 (D-002): 288 is rejected,
    # and 320 is the smallest valid configured vocabulary.
    base = {"name": "x", "purpose": "smoke", "corpus_files": ["tests/fixtures/smoke_corpus.txt"]}
    with pytest.raises(ConfigError, match="multiple of 64"):
        from_dict(TokenizerConfig, {**base, "vocab_size": 288})
    assert from_dict(TokenizerConfig, {**base, "vocab_size": 320}).vocab_size == 320


def test_symlink_escaping_repo_root_is_rejected(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("secret-ish text outside the repository\n", encoding="utf-8")
    repo = tmp_path / "repo"
    repo.mkdir()
    link = repo / "linked.txt"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError):
        pytest.skip("creating symlinks is not permitted on this platform/account")
    config = TokenizerConfig(name="x", purpose="smoke", vocab_size=320, corpus_files=["linked.txt"])
    with pytest.raises(TokenizerTrainingError, match="outside the repository"):
        train_from_config(config, repo)


def test_missing_corpus_file(tmp_path: Path) -> None:
    config = TokenizerConfig(name="x", purpose="smoke", vocab_size=320, corpus_files=["nope.txt"])
    with pytest.raises(TokenizerTrainingError, match="not found"):
        train_from_config(config, tmp_path)


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/train_tokenizer.py", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def test_cli_trains_smoke_tokenizer(tmp_path: Path) -> None:
    out = tmp_path / "smoke" / "tokenizer.json"
    result = run_cli("--config", "configs/tokenizer/smoke.yaml", "--out", str(out))
    assert result.returncode == 0, result.stdout + result.stderr
    tokenizer = BPETokenizer.load(out)
    assert tokenizer.vocab_size == 512
    assert f"tokenizer_sha256: {tokenizer.sha256}" in result.stdout
    assert str(tmp_path) not in result.stdout  # no absolute paths printed


def test_cli_refuses_to_write_inside_repo_outside_runs() -> None:
    target = ROOT / "docs" / "tokenizer.json"
    result = run_cli("--config", "configs/tokenizer/smoke.yaml", "--out", str(target))
    assert result.returncode == 2
    assert not target.exists()


def test_cli_reports_insufficient_corpus(tmp_path: Path) -> None:
    result = run_cli(
        "--config",
        "configs/tokenizer/smoke.yaml",
        "--set",
        "min_pair_count=2",
        "--out",
        str(tmp_path / "t.json"),
    )
    assert result.returncode == 1
    assert "supports only 158 merges" in result.stdout
    assert not (tmp_path / "t.json").exists()
