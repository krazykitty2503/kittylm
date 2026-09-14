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

| ID | Purpose | Status |
|---|---|---|
| EXP-000 | Overfit gate: Nano model memorizes a tiny fixture | not run |
| EXP-001 | Baseline dense Transformer, vocab 16,384 | not run |
| EXP-001-v08k / v12k / v24k / v32k | Vocabulary ablation at equal non-embedding size and training bytes | not run |
