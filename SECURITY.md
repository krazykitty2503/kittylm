# Security Policy

## Reporting

Report security concerns to `security@krazykitty.dev`.

Please do **not** include live secrets, credentials, or private data in a report. Describe where the problem is and how to reproduce it.

## Supported versions

KittyLM is pre-release (0.x). Only the latest commit on `main` receives fixes.

## Report categories

| Category                    | Examples                                                                    | First response                                 |
| --------------------------- | --------------------------------------------------------------------------- | ---------------------------------------------- |
| Repository security issues  | CI misconfiguration, workflow permissions, unsafe scripts                   | Fix, add a regression check                    |
| Dataset contamination       | Unlicensed, private, harmful, or out-of-scope material in a dataset         | Exclude source, bump `recipe_version`, rebuild |
| Accidental secret inclusion | A credential in git, a manifest, a log, a record, a checkpoint, or a sample | Follow the runbook below                       |
| Model-security concerns     | Special-token injection, unsafe generations, memorized sensitive data       | Reproduce, document, mitigate, add a test      |
| Dependency vulnerabilities  | A vulnerable version reported by `pip-audit` or Dependabot                  | Upgrade or document why not affected           |

## Secret-inclusion runbook

Order matters. The secret must be treated as compromised the moment it is found.

1. **Revoke or rotate the credential first**, at its issuing service. Removing it from files does not un-leak it.
2. **Exclude the source**: add an exclusion to the data recipe or delete the file from the repository.
3. **Bump `recipe_version`** so the rebuilt dataset gets a new `dataset_version`.
4. **Mark affected experiment records `compromised`** in their notes and **delete the affected checkpoints** (they may have memorized the value).
5. **Retrain** anything that was trained on the contaminated dataset version.
6. **If the secret was ever committed to git**, purge it from history and force-push, and assume every clone or cache already has it. Step 1 is what actually protects you.

## Built-in safeguards

* Secret scanner (`kittylm/data/secrets.py`) shared by the data pipeline and repository checks. Findings report file, line, and rule only, never the matched value.
* Pre-commit hook (`.githooks/pre-commit`) scans staged files and blocks forbidden artifacts.
* CI (the Security job in `.github/workflows/tests.yml`) re-runs the secret scan, artifact guard, provenance check, and `pip-audit` on every push.
* Workflows run with `contents: read` permissions and only official `actions/*` actions.
* No self-hosted CI runner: it would execute workflow code on a personal machine.

These are defense-in-depth layers, not guarantees. See [docs/safety.md](docs/safety.md) for which properties are implemented today.
