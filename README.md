# KittyLM

KittyLM is a personal language-model research platform within **KittyOS**, created by
**KrazyKitty**. It is built to understand the complete LLM stack by implementing it:

```text
tokenizer → dataset → model → training → evaluation → post-training → tools/agents → KittyOS
```

**Guiding principle:** build it → measure it → understand it → improve it.
**Scope principle for 0.1:** build the hooks now; implement the heavy systems later.

> KittyLM 0.1 is a small research model under construction. It is not a general-purpose
> assistant, and nothing here should be read as evidence of general language capability.
> See [docs/limitations.md](docs/limitations.md) and [docs/responsible-use.md](docs/responsible-use.md).

## Status

The execution contract is the approved **plan rev 3.1**. Work proceeds one step at a time;
all required CI jobs must pass before the next step starts.

| Step | Scope | Status |
|---:|---|---|
| 1 | Foundation: repo, config system, security scanner, artifact guard, ledger, CI, docs | **in progress** |
| 2 | Byte-level BPE tokenizer | not started |
| 3 | Data pipeline (security-boundary order, deterministic dataset version) | not started |
| 4 | Synthetic generators | not started |
| 5 | Build `local-v1`, tokenizer stats, pack | not started |
| 6 | Model (LLaMA-style baseline), KV cache, parameter accounting | not started |
| 7 | Training engine, timing metrics, checkpoint/resume | not started |
| 8 | Evaluation, generation, inference speed | not started |
| 9 | EXP-000 overfit gate | not started |
| 10 | EXP-001 baseline | not started |
| 11 | Vocabulary ablation | not started |
| 12 | Documentation finalized with measured results | not started |

No model has been trained. No experiment results exist yet; when they do, they appear only
in [experiments/ablations.md](experiments/ablations.md), generated from validated records.

## Repository layout (current)

```text
kittylm/            package
  config.py         strict typed YAML configuration
  ledger.py         experiment-record schema, validator, ablation table
  data/secrets.py   secret scanner (data pipeline + repository)
  utils/            git access, artifact guard
  export/           future export boundary (README only)
  integration/      future KittyOS contract + feedback loop (README only)
scripts/            check.py (local == CI), scan_secrets.py, build_ablation_table.py
tests/              foundation tests
experiments/        experiment records and the generated ablation table
configs/            configuration files (none yet)
docs/               architecture, decisions, safety, limitations, scaling, backlog
.github/            CI workflows, templates, CODEOWNERS, Dependabot
.githooks/          pre-commit hook (secret scan + artifact guard)
```

## Development setup (Windows, local ROCm GPU)

```bash
py -3.12 -m venv --system-site-packages .venv
```

The venv inherits the system ROCm build of PyTorch. Then install KittyLM with dev tools:

```bash
.venv/Scripts/python -m pip install -e ".[dev]"
```

Enable the repository's pre-commit hook:

```bash
git config core.hooksPath .githooks
```

Run every check CI runs (quality, tests, security):

```bash
.venv/Scripts/python scripts/check.py all
```

GPU-only tests are never run in CI; run them locally when relevant:

```bash
.venv/Scripts/python -m pytest -m gpu
```

## Licensing

| What | License |
|---|---|
| Code | Apache-2.0 ([LICENSE](LICENSE)) |
| Model weights (none released) | Apache-2.0 ([MODEL_LICENSE.md](MODEL_LICENSE.md)) |
| Documentation | CC BY 4.0 ([docs/LICENSE.md](docs/LICENSE.md)) |
| Training data | Per-source licenses ([DATA_SOURCES.md](DATA_SOURCES.md)) |
| Third-party software | Original licenses ([THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)) |

Copyright 2026 KrazyKitty.

## Project documents

- [CONTRIBUTING.md](CONTRIBUTING.md) — workflow, standards, experiment and provenance rules
- [SECURITY.md](SECURITY.md) — reporting and the secret-inclusion runbook
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- [docs/architecture.md](docs/architecture.md) · [docs/decisions.md](docs/decisions.md) ·
  [docs/safety.md](docs/safety.md) · [docs/scaling.md](docs/scaling.md) ·
  [docs/backlog.md](docs/backlog.md) · [docs/pre-publication.md](docs/pre-publication.md)
