from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from kittylm.utils.artifacts import MAX_FILE_BYTES, check_entries
from kittylm.utils.git import IndexEntry, index_entries, read_blobs
from tests.conftest import ROOT
from tests.fakes import fake_github_token


def _entry(path: str, size: int = 10) -> IndexEntry:
    return IndexEntry(path=path, blob="0" * 40, size=size)


def _no_read(_: list[str]) -> dict[str, bytes]:
    raise AssertionError("content should not be read")


@pytest.mark.parametrize(
    "path",
    [
        "runs/EXP-001/checkpoint.pt",
        "model.safetensors",
        "weights/kittylm.gguf",
        "data/tokenized/local-v1/train.bin",
        "data/raw/stdlib/os.py",
        "data/cleaned/doc.md",
        "runs/EXP-001/metrics.jsonl",
        ".env",
        "service/.env.local",
        "cache.pkl",
        "state.sqlite",
    ],
)
def test_forbidden_paths_are_violations(path: str) -> None:
    assert check_entries([_entry(path)], _no_read), path


@pytest.mark.parametrize(
    "path", ["kittylm/ledger.py", ".env.example", "docs/safety.md", "configs/README.md"]
)
def test_allowed_paths_pass(path: str) -> None:
    assert check_entries([_entry(path)], _no_read) == []


def test_size_limit_and_manifest_allowlist() -> None:
    big = MAX_FILE_BYTES + 1
    assert check_entries([_entry("docs/huge.md", big)], _no_read)

    manifest = IndexEntry("data/manifests/local-v1/manifest.json", "a" * 40, big)
    assert check_entries([manifest], lambda _: {"a" * 40: b'{"ok": true}'}) == []
    binary = check_entries([manifest], lambda _: {"a" * 40: b"\x00\x01binary"})
    assert [v.reason for v in binary] == ["manifest must be UTF-8 text"]


# --- real git repository tests -------------------------------------------------------------


# Environment variable names that may hold credentials. They are removed from the environment
# handed to child git/hook processes, and the sandbox never renders its environment, so a test
# failure cannot print a developer's or CI runner's secrets into logs.
_SENSITIVE_ENV = re.compile(r"TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH|API_?KEY|PRIVATE", re.I)


def sanitized_environment(base: Mapping[str, str]) -> dict[str, str]:
    return {k: v for k, v in base.items() if not _SENSITIVE_ENV.search(k)}


@dataclass(frozen=True)
class GitSandbox:
    """A throwaway repository with the KittyLM hook installed. Its repr omits the env."""

    repo: Path
    env: dict[str, str] = field(repr=False)

    def git(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=self.repo,
            env=self.env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

    def commit(self, name: str, content: bytes) -> tuple[int, str]:
        (self.repo / name).parent.mkdir(parents=True, exist_ok=True)
        (self.repo / name).write_bytes(content)
        assert self.git("add", "-f", name).returncode == 0
        result = self.git("commit", "-q", "-m", f"add {name}")
        return result.returncode, result.stdout + result.stderr


def test_sanitized_environment_drops_credential_names() -> None:
    env = sanitized_environment(
        {"PATH": "p", "SOME_AUTH_TOKEN": "x", "OPENAI_API_KEY": "y", "DB_PASSWORD": "z"}
    )
    assert env == {"PATH": "p"}
    assert "env" not in repr(GitSandbox(Path("r"), {"A_SECRET": "hidden"}))


@pytest.fixture
def hooked_repo(tmp_path: Path) -> GitSandbox:
    if shutil.which("git") is None:
        pytest.skip("git not available")
    repo = tmp_path / "repo"
    repo.mkdir()
    empty = tmp_path / "empty-gitconfig"
    empty.write_bytes(b"")
    env = sanitized_environment(os.environ)
    env.update(
        GIT_CONFIG_GLOBAL=str(empty),  # isolate from the developer's git configuration
        GIT_CONFIG_NOSYSTEM="1",
        KITTYLM_PYTHON=sys.executable.replace("\\", "/"),
    )
    sandbox = GitSandbox(repo=repo, env=env)
    for cmd in (
        ["init", "-q", "-b", "main"],
        ["config", "user.name", "KittyLM Test"],
        ["config", "user.email", "test@example.invalid"],
        ["config", "commit.gpgsign", "false"],
        ["config", "core.hooksPath", ".githooks"],
    ):
        assert sandbox.git(*cmd).returncode == 0
    shutil.copytree(ROOT / ".githooks", repo / ".githooks")
    (repo / "scripts").mkdir()
    for script in ("check.py", "scan_secrets.py"):
        shutil.copy2(ROOT / "scripts" / script, repo / "scripts" / script)
    return sandbox


def test_index_entries_and_blob_reading(hooked_repo: GitSandbox) -> None:
    repo = hooked_repo.repo
    (repo / "a.txt").write_bytes(b"alpha\n")  # bytes: no newline translation on Windows
    assert hooked_repo.git("add", "a.txt").returncode == 0
    entries = [e for e in index_entries(repo) if e.path == "a.txt"]
    assert len(entries) == 1 and entries[0].size == 6
    assert read_blobs(repo, [entries[0].blob])[entries[0].blob] == b"alpha\n"
    staged = {e.path for e in index_entries(repo, staged_only=True)}
    assert "a.txt" in staged


def test_pre_commit_hook_allows_clean_commit(hooked_repo: GitSandbox) -> None:
    code, output = hooked_repo.commit("notes.md", b"# clean\n")
    assert code == 0, output


@pytest.mark.parametrize(
    ("name", "content", "expected"),
    [
        ("checkpoint.pt", b"\x00weights", "forbidden artifact type"),
        (".env", b"SETTING=1\n", "forbidden environment/credential file"),
        ("runs/EXP-001/notes.txt", b"run output\n", "forbidden generated-data/run directory"),
    ],
)
def test_pre_commit_hook_blocks_artifacts(
    hooked_repo: GitSandbox, name: str, content: bytes, expected: str
) -> None:
    code, output = hooked_repo.commit(name, content)
    assert code != 0
    assert expected in output


def test_pre_commit_hook_blocks_secrets_without_printing_them(hooked_repo: GitSandbox) -> None:
    token = fake_github_token()
    code, output = hooked_repo.commit("bot.py", f"TOKEN_HINT = 1\nvalue = {token}\n".encode())
    assert code != 0
    assert "bot.py:2: github_token" in output
    assert token not in output
