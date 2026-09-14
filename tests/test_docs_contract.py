"""Documentation rule (plan rev 3.1 section 8): teachable code is enforced, not hoped for.

Every module in the major subpackages must carry a module docstring with the required
sections. Modules that import torch are "tensor modules" and must also document shapes,
dtypes and devices.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from tests.conftest import ROOT

MAJOR_SUBPACKAGES = ("tokenizer", "data", "model", "training", "evaluation")
BASE_SECTIONS = ("Purpose:", "Public API:", "Invariants:", "Failure modes:", "See:")
TENSOR_SECTIONS = ("Shapes:", "Dtype:", "Device:")


def missing_sections(source: str) -> list[str]:
    """Return required section headings missing from a module's docstring."""
    tree = ast.parse(source)
    docstring = ast.get_docstring(tree) or ""
    headings = {m.group(1) for m in re.finditer(r"^\s*([A-Z][A-Za-z ]*:)", docstring, re.M)}
    required = list(BASE_SECTIONS)
    if _imports_torch(tree):
        required += TENSOR_SECTIONS
    return [section for section in required if section not in headings]


def _imports_torch(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(
            a.name.split(".")[0] == "torch" for a in node.names
        ):
            return True
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "torch":
            return True
    return False


def major_modules() -> list[Path]:
    files: list[Path] = []
    for sub in MAJOR_SUBPACKAGES:
        pkg = ROOT / "kittylm" / sub
        if pkg.is_dir():
            files += sorted(p for p in pkg.rglob("*.py") if p.name != "__init__.py")
    return files


@pytest.mark.parametrize("path", major_modules(), ids=lambda p: p.relative_to(ROOT).as_posix())
def test_major_module_docstring_sections(path: Path) -> None:
    missing = missing_sections(path.read_text(encoding="utf-8"))
    assert not missing, f"{path.relative_to(ROOT)} docstring is missing sections: {missing}"


def test_every_package_module_has_a_docstring() -> None:
    undocumented = [
        p.relative_to(ROOT).as_posix()
        for p in sorted((ROOT / "kittylm").rglob("*.py"))
        if not ast.get_docstring(ast.parse(p.read_text(encoding="utf-8")))
    ]
    assert not undocumented


def test_checker_requires_tensor_sections_only_for_torch_modules() -> None:
    base = (
        '"""Doc.\n\nPurpose:\n x\nPublic API:\n x\nInvariants:\n x\n'
        'Failure modes:\n x\nSee:\n D-001\n"""\n'
    )
    assert missing_sections(base) == []
    assert missing_sections(base + "import torch\n") == list(TENSOR_SECTIONS)
    tensor = base.replace("See:", "Shapes:\n x\nDtype:\n x\nDevice:\n x\nSee:")
    assert missing_sections(tensor + "from torch import nn\n") == []
    assert missing_sections('"""Only a summary."""\n') == list(BASE_SECTIONS)
