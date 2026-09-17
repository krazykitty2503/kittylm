# Experiments

Every meaningful experiment has an ID and a validated record:

```text
experiments/
├── ablations.md          generated — never edit by hand
└── EXP-###[-suffix]/
    └── record.yaml       written by the training/evaluation tooling via kittylm.ledger
```

Run directories with checkpoints, metrics and samples live under `runs/`, which is never
committed.

## Rules

- **No result without a record.** `ablations.md` is rendered from records by
  `scripts/build_ablation_table.py`; CI fails if it is stale or hand-edited.
- Records are validated (`kittylm/ledger.py`): full parameter accounting that sums to the total,
  `final_ppl == exp(final_loss)`, consistent byte/token accounting, no absolute paths, no local
  username or hostname, no secrets, at least one stated limitation.
- Unmeasured values are `null`. `checkpoint_resume_test` is `not_run` unless the resume harness
  actually ran and produced evidence (D-012).
- The directory name must equal `experiment.id`.

## Planned experiments (0.1)

| ID | Kind | Purpose | Status |
|---|---|---|---|
| BENCH-ATTN-001 | benchmark | ROCm attention paths × sequence lengths: throughput, VRAM, failures (Milestone B) | **done** — `BENCH-ATTN-001/benchmark.yaml`; decided D-014 (`reference@bf16`, flag unset) |
| SMOKE-GPU-001 | smoke | Engineering-only GPU overfit test with a 512 smoke tokenizer; never a model-quality result (Milestone E, D-017) | not run |
| EXP-000 | formal | Overfit gate: Nano model memorizes a tiny fixture with the real 16,384 tokenizer | not run |
| EXP-001 | formal | Baseline dense Transformer, vocab 16,384 | not run |
| EXP-001-v08k / v12k / v24k / v32k | formal | Vocabulary ablation at equal non-embedding size and training bytes | not run |

Formal experiments require the required GitHub CI jobs to be green on the exact commit they run
on (D-018).

## Benchmarks

Engineering measurements live in `experiments/BENCH-<AREA>-###/benchmark.yaml` and are validated
by `kittylm.ledger`: the grid must be complete (every variant × flag setting × length), every
non-ok cell must explain itself, and the recorded selection must equal the selection recomputed
from the cells. Benchmarks never appear in the ablation table. A changed software/hardware stack
gets a new benchmark id rather than an edited file.
