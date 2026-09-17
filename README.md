# KittyLM

KittyLM is a personal language-model research platform within **KittyOS**, created by
**KrazyKitty**. It is built to understand the complete LLM stack by implementing it:

```text
tokenizer → dataset → model → training → evaluation → post-training → tools/agents → KittyOS
```
What is KittyLM?

KittyLM is KrazyKitty's from-scratch language-model research project. The goal is not to immediately produce a competitive assistant, but to understand and measure each layer of an LLM system by implementing the stack locally and recording reproducible experiments.

KittyLM is designed as a research component of the broader KittyOS ecosystem. The project prioritizes reproducibility, security, explicit provenance, and measured results over model size or benchmark chasing.

**Guiding principle:** build it → measure it → understand it → improve it.
**Scope principle for 0.1:** build the hooks now; implement the heavy systems later.

> KittyLM 0.1 is a small research model under construction. It is not a general-purpose
> assistant, and nothing here should be read as evidence of general language capability.
> See [docs/limitations.md](docs/limitations.md) and [docs/responsible-use.md](docs/responsible-use.md).

## Status

The execution contract is the approved **plan rev 3.3**. Work proceeds one milestone at a time
(model-first, D-015). Every milestone commit needs `scripts/check.py all` locally **and** the
required GitHub CI jobs green on that exact commit (D-018).

| Milestone | Scope | Status |
|:---:|---|---|
| Step 1 | Foundation: repo, config system, secret scanner, artifact guard, ledger, CI, docs | done |
| A | Byte-level BPE tokenizer, smoke tokenizer config, acceptance tests | done |
| B | Model (LLaMA-style baseline), KV cache, parameter accounting, ROCm attention benchmark (BENCH-ATTN-001), D-014 | done |
| C | Training engine, checkpoint/resume validation, timing metrics, `ci_evidence` tooling | done |
| D | Evaluation (loss, ppl, bpb, overfit gate), generation, inference speed, windowing (D-023) | **in review** |
| E | SMOKE-GPU-001 — engineering-only GPU overfit test (never a quality result) | not started |
| F | Data pipeline and `local-v1` corpus (synthetic generators deferred to `local-v2`, D-016) | not started |
| G | Formal experiments: EXP-000 overfit gate, EXP-001 baseline, vocabulary ablation; docs finalized | not started |

No model has been trained. No experiment results exist yet; when they do, they appear only
in [experiments/ablations.md](experiments/ablations.md), generated from validated records.
Open discrepancies are listed in [docs/discrepancies.md](docs/discrepancies.md).

## Repository layout (current)

```text
kittylm/            package
  config.py         strict typed YAML configuration
  ledger.py         experiment-record schema, validator, ablation table
  tokenizer/        byte-level BPE: pre-tokenization, training, encode/decode, artifacts
  model/            decoder-only Transformer, attention paths, KV cache, parameter accounting
  training/         training engine, checkpoints, deterministic resume harness, timing, metrics
  evaluation/       windowing (D-023), loss/perplexity/bits-per-byte, overfit gate, inference speed
  inference/        generation (greedy/temperature/top-k/top-p, KV cache), secret-scanned samples
  data/secrets.py   secret scanner (data pipeline + repository)
  data/loader.py    memory-mapped token windows (seeded training order, fixed validation)
  utils/            git access, artifact guard
  export/           future export boundary (README only)
  integration/      future KittyOS contract + feedback loop (README only)
scripts/            check.py (local == CI), scan_secrets.py, build_ablation_table.py,
                    train_tokenizer.py, bench_attention.py, verify_ci.py,
                    evaluate.py, generate.py
tests/              foundation, tokenizer, model, training, resume, evaluation, generation and
                    offline end-to-end tests
tests/fixtures/     hand-written smoke corpus
experiments/        experiment records, benchmarks (BENCH-ATTN-001) and the generated ablation table
configs/            tokenizer/smoke.yaml, model/nano.yaml, model/tiny.yaml
docs/               architecture, decisions, discrepancies, safety, limitations, scaling, backlog
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

Train the engineering smoke tokenizer (the artifact goes to `runs/`, which is never committed):

```bash
.venv/Scripts/python scripts/train_tokenizer.py --config configs/tokenizer/smoke.yaml
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
  [docs/backlog.md](docs/backlog.md) · [docs/public-release.md](docs/public-release.md)
