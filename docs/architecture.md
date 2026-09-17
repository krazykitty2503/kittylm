# KittyLM Architecture

This document describes the system KittyLM 0.1 is building. Each component lists its
implementation status; nothing here is claimed as working until its step is complete and
tested.

## System overview

```text
                              KITTYLM
                                 │
           ┌─────────────────────┼─────────────────────┐
           │                     │                     │
         DATA                  MODEL                TRAINING
           │                     │                     │
      provenance            Transformer          checkpointing
      licensing             tokenizer            determinism
      security              attention            metrics
      filtering             SwiGLU               scheduling
      dedup                 KV cache             precision
           │                     │                     │
           └─────────────────────┼─────────────────────┘
                                 │
                            EVALUATION
                                 │
                 ┌───────────────┼───────────────┐
                 │               │               │
              loss/PPL          bpb          throughput
                 │               │               │
                 └───────────────┼───────────────┘
                                 │
                         EXPERIMENT LEDGER
                                 │
           ┌─────────────────────┼─────────────────────┐
           │                     │                     │
        dataset              tokenizer               model
        version               version               version
           │                     │                     │
           └─────────────────────┼─────────────────────┘
                                 │
                              EXPORT
                                 │
                 ┌───────────────┼───────────────┐
                 │               │               │
             native PT      Hugging Face     GGUF/Ollama
                 │               │               │
                 └───────────────┼───────────────┘
                                 │
                              KITTYOS
                                 │
                 ┌───────────────┼───────────────┐
                 │               │               │
               coding         research         agents
                 │               │               │
                 └───────────────┼───────────────┘
                                 │
                          FEEDBACK REVIEW
                                 │
                          curated datasets
                                 │
                             EXPERIMENT
```

Export, KittyOS integration and the feedback loop are architectural boundaries only in 0.1
(README files, no code).

## Components and status

| Component | Location | Milestone | Status |
|---|---|:---:|---|
| Strict configuration | `kittylm/config.py` | Step 1 | implemented, tested |
| Experiment ledger + generated ablation table | `kittylm/ledger.py` | Step 1 | implemented, tested (no records yet) |
| Secret scanner | `kittylm/data/secrets.py` | Step 1 | implemented, tested |
| Artifact guard + pre-commit hook | `kittylm/utils/artifacts.py`, `.githooks/` | Step 1 | implemented, tested |
| CI (`Test`: Quality, Tests ×2, Security) | `.github/workflows/tests.yml`, `scripts/check.py` | Step 1 | implemented, green |
| Byte-level BPE tokenizer | `kittylm/tokenizer/` | A | implemented, tested |
| Dense Transformer baseline + BENCH-ATTN-001 | `kittylm/model/` | B | not started |
| Training engine + resume validation | `kittylm/training/` | C | not started |
| Evaluation + generation | `kittylm/evaluation/`, `kittylm/inference/` | D | not started |
| SMOKE-GPU-001 (engineering-only) | `experiments/SMOKE-GPU-001/` | E | not started |
| Data pipeline + `local-v1` | `kittylm/data/` | F | not started |
| Synthetic generators | `kittylm/synthetic/` | `local-v2` | deferred (D-016) |

## Concepts (expanded as each step lands)

- **Byte-level BPE** (D-001): start from 256 byte tokens and repeatedly merge the most frequent
  adjacent pair. Lossless on any input, because every byte is already a token.
  - *Pre-tokenization* splits text into chunks (words with one leading space or symbol, digit
    groups, punctuation runs, newline + indentation runs) so merges never cross them and the
    trainer can count unique chunks instead of every position.
  - *Training* keeps pair counts incrementally: merging a pair only rewrites the chunks that
    contain it and adjusts the counts of the pairs that appear or disappear. A slow reference
    trainer that recounts everything after each merge is the test oracle for this bookkeeping.
  - *Encoding* repeatedly merges the adjacent pair with the lowest learned rank, which
    reproduces training exactly.
  - *Vocabulary layout* (D-019): bytes 0–255, merges from 256 in learned order, 32 special
    tokens at the top. Smaller vocabularies are exact truncations (D-002); special ids are
    fixed within each tokenizer and recorded in its artifact.
  - *Special tokens are never produced from ordinary text*: a literal `<|endoftext|>` in data
    or tool output is just bytes unless the caller allows that name explicitly.
  - *Insufficient corpus* fails loudly instead of shrinking the vocabulary (D-020).
- **Bits-per-byte** (D-003): total negative log-likelihood in bits divided by the number of
  bytes modeled; comparable across tokenizers, unlike per-token loss.
- **Parameter accounting**: every parameter is attributed to embedding, attention, MLP,
  normalization, positional or output, and the parts must sum to the total. This separates
  "the model improved" from "the model got more parameters".
- **Pre-norm, RoPE, SwiGLU, KV cache**, and **asynchronous GPU timing** are documented here
  when the model and training engine are implemented (steps 6–7).
