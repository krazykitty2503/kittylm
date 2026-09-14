"""CI contract: workflows call real check groups, stay read-only, and never train models."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.conftest import ROOT, load_script

WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))


def load_workflow(path: Path) -> dict[Any, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def run_commands(workflow: dict[Any, Any]) -> list[str]:
    return [
        step["run"] for job in workflow["jobs"].values() for step in job["steps"] if "run" in step
    ]


def test_expected_workflows_exist() -> None:
    assert [p.name for p in WORKFLOWS] == ["quality.yml", "security.yml", "tests.yml"]


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_workflow_calls_known_check_groups(path: Path) -> None:
    check = load_script("check")
    known = set(check.GROUPS) | set(check.STEPS)
    calls = [
        name
        for command in run_commands(load_workflow(path))
        for name in re.findall(r"scripts/check\.py\s+([\w-]+)", command)
    ]
    assert calls, f"{path.name} does not call scripts/check.py"
    assert set(calls) <= known


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_workflow_is_read_only_and_uses_official_actions(path: Path) -> None:
    workflow = load_workflow(path)
    assert workflow["permissions"] == {"contents": "read"}
    for job in workflow["jobs"].values():
        for step in job["steps"]:
            if "uses" in step:
                assert step["uses"].startswith("actions/"), step["uses"]


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_workflow_never_trains_or_runs_gpu_tests(path: Path) -> None:
    commands = "\n".join(run_commands(load_workflow(path)))
    assert "scripts/train.py" not in commands
    assert "-m gpu" not in commands


def test_tests_workflow_matrix() -> None:
    job = load_workflow(ROOT / ".github" / "workflows" / "tests.yml")["jobs"]["tests"]
    matrix = job["strategy"]["matrix"]
    assert matrix["os"] == ["ubuntu-latest", "windows-latest"]
    assert matrix["python-version"] == ["3.12"]


def test_check_groups_exclude_gpu_tests() -> None:
    check = load_script("check")
    assert set(check.GROUPS["all"]) == set(
        check.GROUPS["quality"] + check.GROUPS["tests"] + check.GROUPS["security"]
    )
    assert "hook" not in check.GROUPS["all"]
