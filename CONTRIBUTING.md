# Contributing to KittyLM

KittyLM is currently a single-maintainer research project. This document defines how
work is done so that every result stays reproducible and every change stays understandable.

**Core rule: no benchmark result enters the repository without an experiment record.**
Results live only in `experiments/<ID>/record.yaml`; `experiments/ablations.md` is generated
from those records and CI fails if it is edited by hand or out of sync.

## Plan authority

The approved plan (currently rev 3.1) is the source of truth. No implementation, repository
creation, push, model training, architecture change, dataset expansion or dependency
expansion happens outside its execution order without an explicit plan revision. If
implementation shows a decision is wrong: stop, record the discrepancy, revise the plan,
review, then continue.

## Development setup

See [README.md](README.md#development-setup-windows-local-rocm-gpu). In short: Python 3.12
venv with `--system-site-packages` (inherits the local ROCm PyTorch), `pip install -e ".[dev]"`,
and `git config core.hooksPath .githooks`.

## Test requirements

* `python scripts/check.py all` must pass locally before pushing. It runs the same groups as
  CI: `quality`, `tests`, `security`.
* **GitHub Actions is authoritative.** All required CI jobs must pass. If branch protection with
  required status checks is unavailable for this repository, that rule is followed by
  convention.
* GPU-only tests are marked `@pytest.mark.gpu`, never run in CI, and are run locally with
  `pytest -m gpu`. Pull requests state whether they were run.
* CPU tests must be deterministic. Tests that construct fake credentials must assemble them at
  runtime (see `tests/fakes.py`) so the repository secret scan stays clean.
* Model training never runs in CI. Experiments are explicit research runs.

## Coding standards

* Python 3.12, typed where practical, `ruff format` + `ruff check` clean, `mypy` clean.
* Clear package boundaries: `kittylm.model` never imports `kittylm.training` or
  `kittylm.data`; `kittylm.tokenizer` never imports torch. Enforced by tests.
* Standard library `logging`; no new frameworks without a plan revision.
* **Documentation rule:** every non-trivial subsystem explains what it does *and why*. Modules
  in `tokenizer/`, `data/`, `model/`, `training/`, `evaluation/` have docstrings with
  `Purpose:`, `Public API:`, `Invariants:`, `Failure modes:`, `See:` sections, plus
  `Shapes:`, `Dtype:`, `Device:` for modules that import torch. Enforced by
  `tests/test_docs_contract.py`.
  Design decisions are logged in [docs/decisions.md](docs/decisions.md) and referenced as `D-###`.

## Commits and pull requests

* One logical change per commit; messages explain *why*.
* Pull requests use the template checklist: tests, local GPU tests (y/n/NA), records for any
  result, regenerated ablation table, provenance for new data, no forbidden artifacts.
* Never commit secrets, `.env` files, credentials, raw/cleaned/tokenized data, checkpoints,
  weights, generated samples, logs or run directories. The pre-commit hook and CI enforce this.

## Experiment requirements

* Every meaningful experiment has an ID (`EXP-###`, variants `EXP-###-suffix`).
* Every run writes a validated record with: git commit (+dirty flag), environment, dataset and
  tokenizer versions, model shape, **full parameter accounting**, training setup, measured
  results, reproducibility status, limitations and notes.
* Values that were not measured are `null`; resume status is `not_run` unless the resume
  harness actually ran. Nothing is estimated into a record.
* Every architectural addition needs an ablation against a named control record (normally
  EXP-001), with the parameter budget reported so a variant cannot win by size alone.

## Dataset provenance requirements

* Every source declares a license; the recipe's `allowed_licenses` gates ingestion.
* Documents pass license validation and the secret scan before anything is written to disk.
* Manifests record hashes and relative paths only; `DATA_SOURCES.md` is generated from them.
* Feedback or user data is never trained on automatically; it must go through human review
  into a new, versioned dataset (see `kittylm/integration/README.md`).

## Benchmark reproducibility

A result is reproducible when its record pins the commit, dataset version, tokenizer version,
resolved config and hardware, and the committed ablation table is regenerated from it.

## Proposing an architecture change

1. Open an *Experiment proposal* issue: hypothesis, the single component under test, control
   record, what is held constant, metrics, expected parameter budget, scope classification.
2. Revise the plan if the change is outside the current execution order.
3. Implement behind the existing mixer seams, with tests and documentation.
4. Run the ablation, write records, regenerate the table, and report what was measured and
   what remains uncertain.
