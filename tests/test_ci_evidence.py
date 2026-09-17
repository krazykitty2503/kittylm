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


# --- review regressions (PR #4) -------------------------------------------------------------------


@pytest.mark.parametrize("event", ["pull_request", "workflow_dispatch", "schedule"])
def test_formal_evidence_must_come_from_a_push_run(event: str) -> None:
    # Regression: the event was recorded but never validated, so evidence from a pull-request
    # (merge-preview) or manually dispatched run satisfied the formal-run gate.
    evidence = ci_evidence_from_github({**RUN, "event": event}, JOBS)
    problems = ci_evidence_problems(evidence, COMMIT)
    assert problems == [f"ci_evidence event must be 'push', not '{event}'"]
    data = valid_record()
    data["ci_evidence"]["event"] = event
    with pytest.raises(LedgerError, match="event must be 'push'"):
        parse_record(data)


class FakeGitHub:
    """Serves list endpoints page by page, recording every URL requested."""

    def __init__(self, items: dict[str, list[dict[str, Any]]], total_count: bool) -> None:
        self.items = items
        self.total_count = total_count
        self.urls: list[str] = []

    def __call__(self, url: str) -> dict[str, Any]:
        self.urls.append(url)
        path, query = url.split("?", 1)
        params = dict(part.split("=", 1) for part in query.split("&"))
        per_page, page = int(params["per_page"]), int(params["page"])
        key = "jobs" if path.endswith("/jobs") else "workflow_runs"
        everything = self.items[key]
        page_items = everything[(page - 1) * per_page : page * per_page]
        body: dict[str, Any] = {key: page_items}
        if self.total_count:
            body["total_count"] = len(everything)
        return body


@pytest.mark.parametrize("total_count", [True, False])
def test_verify_ci_reads_every_page_of_runs_and_jobs(total_count: bool) -> None:
    # Regression: only one 100-item page was read, so a matching run or a required job beyond
    # the first page was treated as absent.
    verify = load_script("verify_ci")
    other = [
        {**RUN, "id": 1000 + i, "name": "Dependabot Updates", "event": "dynamic"}
        for i in range(verify.PER_PAGE)
    ]
    runs = [*other, {**RUN, "id": 4242, "run_attempt": 1}]
    filler = [
        {"id": 5000 + i, "name": f"matrix shard {i}", "conclusion": "success"}
        for i in range(verify.PER_PAGE + 20)
    ]
    jobs = [*filler, *JOBS]  # every required job is on page 2
    github = FakeGitHub({"workflow_runs": runs, "jobs": jobs}, total_count)

    fetched_runs = verify.fetch_all(
        github, f"repos/o/r/actions/runs?head_sha={COMMIT}", "workflow_runs"
    )
    assert len(fetched_runs) == len(runs)
    run = verify.select_run(fetched_runs, COMMIT)
    assert run is not None and run["id"] == 4242
    fetched_jobs = verify.fetch_all(
        github, "repos/o/r/actions/runs/4242/jobs?filter=latest", "jobs"
    )
    assert len(fetched_jobs) == len(jobs)
    evidence = ci_evidence_from_github(run, fetched_jobs)
    assert ci_evidence_problems(evidence, COMMIT) == []
    assert any("page=2" in url for url in github.urls)
    assert all(f"per_page={verify.PER_PAGE}" in url for url in github.urls)


def test_verify_ci_stops_on_an_exactly_full_last_page_with_total_count() -> None:
    verify = load_script("verify_ci")
    items = [{"id": i} for i in range(4)]
    github = FakeGitHub({"workflow_runs": items, "jobs": []}, total_count=True)
    assert verify.fetch_all(github, "repos/o/r/actions/runs", "workflow_runs", per_page=2) == items
    assert len(github.urls) == 2  # total_count reached: no third request


def test_verify_ci_refuses_unbounded_pagination() -> None:
    verify = load_script("verify_ci")

    def always_full(url: str) -> dict[str, Any]:
        return {"workflow_runs": [{"id": 1}] * verify.PER_PAGE}

    with pytest.raises(RuntimeError, match="refusing to guess"):
        verify.fetch_all(always_full, "repos/o/r/actions/runs", "workflow_runs")
