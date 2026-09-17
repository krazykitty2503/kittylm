"""Produce ci_evidence: prove required GitHub CI is green on an exact commit (D-018).

Formal experiments (EXP-*) may only run on a commit whose required jobs (workflow `Test`:
Quality, Tests (ubuntu-latest), Tests (windows-latest), Security) all concluded `success`.
This dev tool queries GitHub through the `gh` CLI, builds `CiEvidence`, validates it, prints it
as JSON and writes it under `runs/ci_evidence/` (never committed). KittyLM itself makes no
network requests; only this script does.

Usage:
    python scripts/verify_ci.py                    # HEAD
    python scripts/verify_ci.py --commit <sha> --out runs/ci_evidence/<sha>.json

Exit code 0 means every required job succeeded on exactly that commit.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from kittylm.config import to_dict
from kittylm.ledger import REQUIRED_CI_WORKFLOW, ci_evidence_from_github, ci_evidence_problems

ROOT = Path(__file__).resolve().parents[1]


def gh_json(*args: str) -> Any:
    result = subprocess.run(
        ["gh", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args[:2])} failed: {result.stderr.strip()[:300]}")
    return json.loads(result.stdout)


def select_run(runs: Sequence[dict[str, Any]], commit: str) -> dict[str, Any] | None:
    """Latest completed push run of the required workflow for exactly ``commit``."""
    candidates = [
        run
        for run in runs
        if run.get("head_sha") == commit
        and run.get("name") == REQUIRED_CI_WORKFLOW
        and run.get("event") == "push"
        and run.get("status") == "completed"
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda run: (run.get("run_attempt", 1), run.get("id", 0)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--commit", help="full commit sha (default: HEAD)")
    parser.add_argument("--repo", help="owner/name (default: the current gh repository)")
    parser.add_argument("--out", type=Path, help="default: runs/ci_evidence/<sha>.json")
    args = parser.parse_args(argv)

    commit = (
        args.commit
        or subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    )
    repo = args.repo or gh_json("repo", "view", "--json", "nameWithOwner")["nameWithOwner"]
    runs = gh_json("api", f"repos/{repo}/actions/runs?head_sha={commit}&per_page=100")
    run = select_run(runs.get("workflow_runs", []), commit)
    if run is None:
        print(f"no completed '{REQUIRED_CI_WORKFLOW}' push run found for {commit}")
        return 1
    jobs = gh_json("api", f"repos/{repo}/actions/runs/{run['id']}/jobs?per_page=100")
    evidence = ci_evidence_from_github(run, jobs.get("jobs", []))
    problems = ci_evidence_problems(evidence, commit)

    out = args.out or ROOT / "runs" / "ci_evidence" / f"{commit}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(to_dict(evidence), indent=2, sort_keys=True)
    out.write_text(text + "\n", encoding="utf-8")
    print(text)
    if problems:
        print("\n".join(f"NOT GREEN: {problem}" for problem in problems))
        return 1
    print(f"required CI green on {commit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
