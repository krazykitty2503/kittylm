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
produced by `kittylm.training.resume_harness` (kill step, resumed step, metrics hash and a
per-category comparison summary); `not_run` carries no evidence. The harness runs three fresh
Python processes (uninterrupted to N; to the kill step with a checkpoint; a new process that
resumes to N) and compares global step, scheduler/LR series, optimizer state, parameters, every
RNG state (Python, NumPy, torch CPU, torch GPU), loader generator state, loss series, the batch
indices drawn after the resume and the next batch indices. On CPU (nano, fp32, deterministic)
every category must be bit-exact; on GPU counters, scheduler, RNG and loader state must be exact
and tensor/loss differences are reported as a measured maximum absolute deviation. A test
parses the source tree and fails if anything other than the harness constructs `ResumeEvidence`.
**Why.** Makes a fabricated or hand-typed "passed" structurally detectable, and proves resume
across a real process boundary rather than within one interpreter.
**Limit.** It is a structural check, not cryptographic proof.

### D-013 — Dev tools pinned exactly; runtime dependencies lower-bounded

**Decision.** `pytest`, `ruff`, `mypy`, `pip-audit` are pinned in `pyproject.toml`; `torch`,
`numpy`, `pyyaml`, `regex` have lower bounds only.
**Why.** Formatter and linter output must not drift between local runs and CI; the local ROCm
PyTorch build must satisfy runtime requirements without being replaced. Dependabot proposes
upgrades.

### D-014 — Attention kernel: the `reference` implementation (bf16, no experimental flag)

**Decision.** `nano` and `tiny` use `attention_backend: reference`, with
`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` unset. The `reference` path stays the correctness oracle;
on this hardware it is also the production path.
**Evidence.** `experiments/BENCH-ATTN-001/benchmark.yaml`, measured on a clean checkout of
`da91fdb` (AMD Radeon RX 9060 XT, ROCm/HIP 7.15, PyTorch 2.13.0+rocm10.0.0), 72 cells:
- flash, memory-efficient and cuDNN SDPA kernels are *unsupported* with the flag unset, and
  flash/memory-efficient fail with `CUDA error: invalid argument` at every length with the flag
  set; cuDNN is unsupported in both settings;
- `sdpa_math` works and matches the bf16 reference within the 1e-2 relative tolerance, but is
  slower and uses more memory: at T=1024 (tiny's context) 375k vs 555k tokens/s and 1,820 vs
  993 MiB peak; at T=256 810k vs 1,169k tokens/s.

The selection is recomputed from the cells by the ledger validator (rule: fastest bf16 variant
that is ok and equivalent at every selection length, 256 and 1024, for one flag setting; exact
ties prefer the flag unset).
**Trade-off.** Both working paths materialize the full T×T attention matrix, so memory grows
quadratically (6.4 GB peak at T=8192, batch 2). That is acceptable for 0.1 contexts (≤ 1024) but
is a limit for long-context work. Revisit when the PyTorch/ROCm stack or GPU changes: rerun the
benchmark (as a new BENCH id) before switching kernels.
**Known limit of the rule.** Throughput is a single measurement window per cell; the flag-
independent reference path varied by up to ~10% between flag runs (and one T=2048 window
stalled). A near-tie between flag settings could therefore be decided by noise; this run's choice
does not depend on the flag.

### D-015 — Model-first milestone order

**Decision.** Remaining work runs A tokenizer → B model + BENCH-ATTN-001 → C training engine →
D evaluation/generation → E SMOKE-GPU-001 → F data pipeline + `local-v1` → G formal experiments.
**Why.** The highest technical risk is training correctly on an AMD GPU under ROCm on Windows
(kernels, bf16, throughput, checkpoint/resume), not the data pipeline. Proving the model stack on
fixtures first surfaces that risk early. Nothing is skipped; only the order changed (plan rev 3.2).

### D-016 — Synthetic generators deferred to `local-v2`

**Decision.** `local-v1` contains no synthetic data; the seeded generators move to `local-v2`,
after EXP-001.
**Why.** They are at most 5% of bytes and do not change what the baseline measures, while being
the largest block of code before the first training run.

### D-017 — Smoke records are separate from formal experiments

**Decision.** Engineering smoke runs (`SMOKE-*`, `kind: smoke`) are recorded but can never
support architecture or model-quality conclusions: they carry a fixed limitation and are
excluded from the ablation table. Formal experiments are `EXP-*`, `kind: formal`.
**Why.** An overfit smoke test proves the pipeline works; it says nothing about the model, and
mixing the two would invite exactly that misreading.

### D-018 — CI gate policy

**Decision.** Required CI is workflow `Test` with jobs Quality, Tests (ubuntu-latest), Tests
(windows-latest) and Security. Every milestone commit needs `scripts/check.py all` locally and
required CI green on that exact commit; every formal experiment additionally records
`ci_evidence` for its exact commit. If CI fails for infrastructure (not code) reasons, milestones
may proceed on local validation until the new discrepancy is closed, but formal experiments stay
blocked (plan rev 3.3).
**Why.** Results must never depend on code that CI has not verified, while an external CI outage
should not freeze engineering work.

### D-019 — Special-token ids are fixed per tokenizer, not across vocabulary sizes

**Decision.** Special tokens occupy the top 32 ids of each vocabulary, so their ids are fixed and
deterministic within a tokenizer artifact/configuration but differ across vocabulary sizes. Every
artifact records the exact name → id mapping, and loading rejects any mismatch; tokens are never
silently renumbered. Ordinary text, including literal special-token strings, never encodes to a
special id unless the caller allows that name explicitly.
**Why.** Keeping merges contiguous from id 256 makes smaller vocabularies exact truncations of
larger ones; recording and validating the mapping prevents silent drift; refusing literal
special tokens by default prevents data or tool output from injecting control tokens.

### D-020 — Tokenizer training fails instead of shrinking the vocabulary

**Decision.** If the corpus cannot supply enough merges with pair count ≥ `min_pair_count`,
training raises an error naming how many merges were possible and how many were required.
**Why.** A silently smaller vocabulary would change parameter counts, special-token ids and
every downstream comparison. Research tokenizers use `min_pair_count: 2`; the engineering smoke
tokenizer uses `1` because its fixture supports only 158 merges at 2 (measured).

### D-021 — Checkpoint format: checksummed header, safe loading, atomic replace

**Decision.** A checkpoint file is one JSON header line (`magic`, `format_version`,
`payload_sha256`, `payload_bytes`) followed by a `torch.save` payload. Writes go to a temporary
file in the same directory, are fsynced and then renamed with `os.replace`; `latest.json` is
written the same way and records the file name and payload checksum. Loading verifies the magic,
version, length and checksum before calling `torch.load(weights_only=True)`, so arbitrary pickled
objects are refused. The top-level keys and the metadata keys (config hash, tokenizer sha256,
dataset version, git commit/dirty, precision, device type, determinism) are fixed whitelists;
resume refuses a checkpoint whose config hash, tokenizer or dataset version differs from the run.
**Why.** A crash mid-save must leave the previous checkpoint loadable, corruption must be
detected rather than silently loaded, a checkpoint must not be able to execute code, and
environment variables, paths or hostnames cannot leak into a checkpoint through metadata.
**Limit.** The checksum detects accidental damage, not deliberate tampering by someone who can
rewrite the header.

### D-022 — `ci_evidence` is read from GitHub, never typed

**Decision.** `scripts/verify_ci.py` asks the GitHub API (via `gh`) for the latest completed
`push` run of workflow `Test` on the exact commit and records the run id and each required job's
id and conclusion. The ledger rejects formal records without evidence whose commit matches the
record and whose four required jobs all concluded `success`; the training engine refuses to start
a formal run without such evidence or from a dirty working tree.
**Why.** Formal results must be traceable to code that required CI verified (D-018).
**Limit.** Like D-012 this is structural: evidence could be hand-written, but it names a run id
anyone can check on GitHub.
