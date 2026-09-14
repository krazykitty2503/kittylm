"""Package boundaries (plan rev 3.1 section 7).

- kittylm.model must never import kittylm.training or kittylm.data: the architecture is
  independent of how it is trained or what data it sees.
- kittylm.tokenizer must never import torch: tokenization is plain Python and testable
  without a deep-learning stack.

There is deliberately no ban on network-module imports (see docs/safety.md).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.conftest import ROOT

RULES: dict[str, tuple[str, ...]] = {
    "kittylm/model": ("kittylm.training", "kittylm.data"),
    "kittylm/tokenizer": ("torch",),
}


def module_name(path: Path, root: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def imported_modules(path: Path, root: Path) -> set[str]:
    """Absolute names of every module imported by ``path`` (relative imports resolved)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    package = module_name(path, root)
    if path.name != "__init__.py":
        package = package.rpartition(".")[0]
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")
                base = base[: len(base) - (node.level - 1)] if node.level > 1 else base
                prefix = ".".join(base)
                target = f"{prefix}.{node.module}" if node.module else prefix
            else:
                target = node.module or ""
            names.add(target)
            names.update(f"{target}.{alias.name}" for alias in node.names)
    return names


def violations(root: Path, rules: dict[str, tuple[str, ...]]) -> list[str]:
    found: list[str] = []
    for directory, forbidden in rules.items():
        for path in sorted((root / directory).rglob("*.py")):
            for name in sorted(imported_modules(path, root)):
                if any(name == f or name.startswith(f + ".") for f in forbidden):
                    found.append(f"{path.relative_to(root).as_posix()} imports {name}")
    return found


def test_repository_respects_package_boundaries() -> None:
    assert violations(ROOT, RULES) == []


@pytest.mark.parametrize(
    ("relpath", "source", "expected"),
    [
        ("kittylm/model/block.py", "from kittylm.training import engine\n", True),
        ("kittylm/model/block.py", "from ..data import loader\n", True),
        ("kittylm/model/sub/deep.py", "from ...training.engine import X\n", True),
        ("kittylm/model/block.py", "from . import attention\n", False),
        ("kittylm/model/block.py", "import kittylm.config\n", False),
        ("kittylm/tokenizer/bpe.py", "import torch.nn as nn\n", True),
        ("kittylm/tokenizer/bpe.py", "import regex\n", False),
    ],
)
def test_checker_detects_violations(
    tmp_path: Path, relpath: str, source: str, expected: bool
) -> None:
    target = tmp_path / relpath
    target.parent.mkdir(parents=True)
    target.write_text(source, encoding="utf-8")
    assert bool(violations(tmp_path, RULES)) is expected
