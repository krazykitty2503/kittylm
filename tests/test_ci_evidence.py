"""CI evidence (D-018): parsing GitHub results, validation, and formal-record enforcement."""

from __future__ import annotations

from typing import Any

import pytest

from kittylm.ledger import (
    REQUIRED_CI_JOBS,
    CiEvidence,
    CiJob,
    LedgerError,
    ci_evidence_from_github,
    ci_evidence_problems,
    parse_record,
)
from tests.conftest import load_script
from tests.test_ledger import COMMIT, valid_record

RUN = {"id": 777, "name": "Test", "event": "push", "head_sha": COMMIT, "status": "completed"}
JOBS = [
    {"id": 10 + i, "name": name, "conclusion": "success"} for i, name in enumerate(REQUIRED_CI_JOBS)
]


def test_from_github_and_valid_evidence() -> None:
    evidence = ci_evidence_from_github(RUN, JOBS)
    assert evidence.run_id == 777 and evidence.commit == COMMIT
    assert evidence.jobs["Security"] == CiJob(job_id=13, conclusion="success")
    assert ci_evidence_problems(evidence, COMMIT) == []


@pytest.mark.parametrize(
    ("run_changes", "jobs", "problem"),
    [
        ({}, JOBS[:-1], "missing required job 'Security'"),
        ({}, [{**JOBS[0], "conclusion": "failure"}, *JOBS[1:]], "concluded 'failure'"),
        ({"head_sha": "e" * 40}, JOBS, "is for commit"),
        ({"name": "quality"}, JOBS, "workflow must be 'Test'"),
        ({"id": 0}, JOBS, "positive GitHub run id"),
    ],
)
def test_problems(run_changes: dict[str, Any], jobs: list[dict[str, Any]], problem: str) -> None:
    evidence = ci_evidence_from_github({**RUN, **run_changes}, jobs)
    assert any(problem in p for p in ci_evidence_problems(evidence, COMMIT))


def test_missing_evidence() -> None:
    assert "ci_evidence is required" in ci_evidence_problems(None, COMMIT)[0]


def test_formal_records_require_ci_evidence_but_smoke_records_do_not() -> None:
    data = valid_record()
    data["ci_evidence"] = None
    with pytest.raises(LedgerError, match="ci_evidence is required"):
        parse_record(data)
    smoke = valid_record()
    smoke["experiment"].update(id="SMOKE-GPU-001", kind="smoke")
    smoke["ci_evidence"] = None
    from kittylm.ledger import SMOKE_LIMITATION

    smoke["limitations"] = [SMOKE_LIMITATION]
    assert parse_record(smoke).ci_evidence is None


def test_formal_record_evidence_must_match_its_commit() -> None:
    data = valid_record()
    data["ci_evidence"]["commit"] = "e" * 40
    with pytest.raises(LedgerError, match="is for commit"):
        parse_record(data)


def test_verify_ci_selects_the_latest_completed_push_run_for_the_exact_commit() -> None:
    verify = load_script("verify_ci")
    runs = [
        {**RUN, "id": 1, "run_attempt": 1},
        {**RUN, "id": 2, "run_attempt": 2},
        {**RUN, "id": 3, "event": "pull_request"},
        {**RUN, "id": 4, "status": "in_progress"},
        {**RUN, "id": 5, "head_sha": "e" * 40},
        {**RUN, "id": 6, "name": "Dependabot Updates"},
    ]
    assert verify.select_run(runs, COMMIT)["id"] == 2
    assert verify.select_run(runs[2:], COMMIT) is None


def test_evidence_dataclass_is_frozen() -> None:
    evidence = CiEvidence(commit=COMMIT, workflow="Test", run_id=1, event="push", jobs={})
    with pytest.raises(AttributeError):
        evidence.run_id = 2  # type: ignore[misc]
