# KittyLM Limitations

**Status: no model has been trained.** This document is completed with measured EXP-001
numbers at plan step 12. Until then it states the limitations that are already known by
design.

## Known by design (0.1)

- **Small research model.** The 0.1 baseline is on the order of 17M parameters. It is a
  laboratory for understanding and measuring the LLM stack, not a capable assistant.
- **Narrow dataset.** `local-v1` is intentionally not representative of general language. It is
  a technical/code corpus built to validate KittyLM's training stack. EXP-001 must not be read
  as evidence of general language capability.
- **Not evidence of general intelligence.** Loss, perplexity and bits-per-byte on `local-v1`
  measure next-token prediction on that corpus only.
- **Outputs can be wrong.** Generated code, explanations and security content can be
  incorrect, insecure or fabricated.
- **No task benchmarks in 0.1.** Coding, computer-science, security, reasoning and agent
  benchmark suites are deferred; a model this small would score near zero on them.
- **Exact deduplication only.** Near-duplicate documents may remain (MinHash dedup is
  backlogged).
- **No document-boundary attention masking.** Packed training windows can attend across
  document boundaries.
- **Hardware reproducibility.** CPU runs are bit-exact; ROCm GPU runs are semi-deterministic
  (same initialization and data order, kernels may differ).

## Measured limitations

To be filled from experiment records at step 12.
