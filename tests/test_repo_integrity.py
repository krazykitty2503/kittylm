"""Guards against the web-editor damage recorded as X-003 (Markdown-mangled source files)."""

from __future__ import annotations

import ast
import re
import subprocess

import pytest

from tests.conftest import ROOT

MANGLED_DUNDER = re.compile(r"(?<![\w*])\*\*(future|name|main|init|file|doc|all|path|pycache)\*\*")


def tracked(*patterns: str) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--", *patterns],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return sorted(line for line in result.stdout.splitlines() if line)


@pytest.mark.parametrize("path", tracked("*.py"))
def test_python_files_parse(path: str) -> None:
    source = (ROOT / path).read_text(encoding="utf-8")
    ast.parse(source, filename=path)


def test_no_markdown_mangled_dunders() -> None:
    hits = []
    for path in tracked("*.py", "*.toml", "*.yml", "*.yaml", ".gitignore", ".gitattributes"):
        text = (ROOT / path).read_text(encoding="utf-8")
        hits += [f"{path}: {m.group(0)}" for m in MANGLED_DUNDER.finditer(text)]
    assert hits == []


def test_detector_catches_known_damage() -> None:
    bold = "**"  # assembled at runtime so this file does not trip its own scan
    assert MANGLED_DUNDER.search(f"from {bold}future{bold} import annotations")
    assert MANGLED_DUNDER.search(f'if {bold}name{bold} == "{bold}main{bold}":')
    assert not MANGLED_DUNDER.search("def f(**kwargs): return 2**power")
