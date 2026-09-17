# Discrepancy Log

Mirror of the discrepancy log in the approved implementation plan (rev 3.3). A discrepancy is
recorded whenever implementation reveals that a planned decision or assumption is wrong. It is
closed only with evidence.

| ID | Opened | Summary | Status |
|---|---|---|---|
| X-001 | 2026-09-14 | GitHub Actions runs ended in `startup_failure` with zero jobs while the repository was private (run `34884505467`). | **Closed 2026-09-17.** After the repository became public, jobs start; the remaining failures were code failures (X-003). Workflow `Test` run `35180323722` on `f4bf86c` succeeded with all required jobs (Quality, Tests ubuntu-latest, Tests windows-latest, Security). Root cause of the startup failure never confirmed. |
| X-002 | 2026-09-17 | Repository became public (`d665a14`) while the plan said private; contact and pre-publication wording were not reconciled. | **Closed 2026-09-17** by plan rev 3.3 (public visibility and published contacts recorded). Mailbox delivery not independently tested. |
| X-003 | 2026-09-17 | Web edits Markdown-mangled `scripts/check.py` (`from **future** import …`, lost indentation), `.gitignore` (`**pycache**/`) and truncated `CONTRIBUTING.md`; every required CI job failed. | **Closed 2026-09-17.** Repaired in `3c672d9` (all web edits preserved; workflows consolidated into `tests.yml`) and `f4bf86c` (`setuptools` upgraded in CI after `pip-audit` findings). `tests/test_repo_integrity.py` now guards against this damage. |
| X-004 | 2026-09-17 | `docs/public-release.md` claimed private vulnerability reporting, secret scanning and push protection were enabled; the GitHub API reports them (and Dependabot security updates) **disabled**. | **Open.** Requires KrazyKitty to enable the settings. `docs/public-release.md` now shows the verified state. Not blocking. |
| X-005 | 2026-09-17 | SonarCloud (GitHub App, not a required job) quality gate fails: "C Security Rating on New Code (required ≥ A)". | **Open, advisory.** Decide whether to keep it advisory, make it required, or remove it; review issues in the SonarCloud UI. Not blocking. |
