# KittyLM Decision Log

Numbered design decisions. Module docstrings reference them as `D-###`. A decision changes
only through a plan revision; superseded decisions stay in the log, marked as such.

---

### D-001 — Byte-level BPE tokenizer written from scratch

**Decision.** Implement byte-level BPE ourselves: code-aware regex pre-tokenization,
incremental pair counts with a lazy-invalidation heap, deterministic tie-breaks.
**Why.** The project's purpose is understanding every layer. Byte-level BPE is lossless on
any input (no `<unk>`), which matters for code, shell and configuration text.
**Trade-off.** Slower than a Rust tokenizer; acceptable at `local-v1` scale. A faster backend
is backlogged as NICE_TO_HAVE.

### D-002 — 16,384 primary vocabulary; 8,192 / 12,288 / 24,576 / 32,768 as ablations

**Decision.** Train BPE once to 32,768 and truncate; 16,384 is the primary tokenizer.
**Why.** BPE is greedy, so the first N merges of a larger run *are* the N-merge tokenizer,
which makes the ablation clean and cheap. All sizes are multiples of 64, which suits
embedding and LM-head matrix kernels. At 32,768 the tiny model would be ~54% embeddings on a
small corpus; 16,384 keeps it at ~37%.
**Revisit.** Move to 32,768 when the corpus is substantially broader (0.2).

### D-003 — Bits-per-byte as the cross-tokenizer comparison metric

**Decision.** Report bits-per-byte (overall and per category) next to loss and perplexity.
**Why.** Per-token loss is not comparable across vocabularies: a larger vocabulary predicts
fewer, longer tokens. Bits-per-byte normalizes by the bytes actually modeled.

### D-004 — LLaMA-style dense baseline

**Decision.** Pre-norm RMSNorm, RoPE, causal SDPA attention, SwiGLU, no biases, tied
embeddings.
**Why.** Every future architecture is judged against this model; a weak baseline would make
any change look good.

### D-005 — Split by document via content-hash buckets

**Decision.** Assign each document to train/validation/test by its content hash (98/1/1).
**Why.** Splitting chunks or windows leaks near-identical text across splits; hash buckets
are stable when new sources are added.

### D-006 — The whole document is dropped on any secret-scan hit

**Decision.** A secret-scan finding drops the entire document; only `(doc_id, rule)` is
recorded, never the matched value. Documents are scanned before anything is written to disk.
**Why.** Redaction is error-prone and can leave fragments; a dropped document cannot leak.
**Trade-off.** Some benign files (for example base64 test fixtures) are lost.

### D-007 — Subpackage layout under `kittylm/`

**Decision.** `data`, `training`, `evaluation`, `inference` code lives under `kittylm/`;
top-level `data/` holds data only.
**Why.** Generic top-level package names (`training`, `evaluation`) collide once installed.

### D-008 — HF-LLaMA RoPE layout and tied weights for export compatibility

**Decision.** Use the `rotate_half` RoPE convention with tied embedding/LM head.
**Why.** The dense baseline then maps onto the llama GGUF layout without re-deriving weights.
Future hybrids will not, which is a deployment cost to weigh in those experiments.

### D-009 — No automatic online learning

**Decision.** Feedback never trains a model directly. It flows through human review into a
curated, versioned dataset, then an experiment and an evaluation.
**Why.** Protects the model from learning bad or malicious feedback; keeps every model
reproducible from a dataset version.

### D-010 — No self-hosted GPU CI runner in 0.1

**Decision.** GPU tests run locally (`pytest -m gpu`); CI is CPU-only on hosted runners.
**Why.** A self-hosted runner on a personal machine executes workflow code, which is a
security risk without a hardened setup.

### D-011 — Strict configuration loading

**Decision.** Unknown keys, duplicate keys, missing keys and type mismatches are errors. Every
YAML under `configs/` declares a `kind` and is validated in CI.
**Why.** A silently ignored typo produces a run that looks valid but is not the intended
experiment.

### D-012 — Resume status requires harness evidence

**Decision.** `checkpoint_resume_test` may be `passed`/`failed` only with `resume_evidence`
produced by `kittylm.training.resume_harness` (kill step, resumed step, metrics hash);
`not_run` carries no evidence.
**Why.** Makes a fabricated or hand-typed "passed" structurally detectable.
**Limit.** It is a structural check, not cryptographic proof.

### D-013 — Dev tools pinned exactly; runtime dependencies lower-bounded

**Decision.** `pytest`, `ruff`, `mypy`, `pip-audit` are pinned in `pyproject.toml`; `torch`,
`numpy`, `pyyaml`, `regex` have lower bounds only.
**Why.** Formatter and linter output must not drift between local runs and CI; the local ROCm
PyTorch build must satisfy runtime requirements without being replaced. Dependabot proposes
upgrades.
