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

| Component | Location | Plan step | Status |
|---|---|---:|---|
| Strict configuration | `kittylm/config.py` | 1 | implemented, tested |
| Experiment ledger + generated ablation table | `kittylm/ledger.py` | 1 | implemented, tested (no records yet) |
| Secret scanner | `kittylm/data/secrets.py` | 1 | implemented, tested |
| Artifact guard + pre-commit hook | `kittylm/utils/artifacts.py`, `.githooks/` | 1 | implemented, tested |
| CI (quality, tests, security) | `.github/workflows/`, `scripts/check.py` | 1 | implemented |
| Byte-level BPE tokenizer | `kittylm/tokenizer/` | 2 | not started |
| Data pipeline | `kittylm/data/` | 3 | not started |
| Synthetic generators | `kittylm/synthetic/` | 4 | not started |
| Dense Transformer baseline | `kittylm/model/` | 6 | not started |
| Training engine | `kittylm/training/` | 7 | not started |
| Evaluation + generation | `kittylm/evaluation/`, `kittylm/inference/` | 8 | not started |

## Concepts (expanded as each step lands)

- **Byte-level BPE** (D-001): start from 256 byte tokens and repeatedly merge the most frequent
  adjacent pair. Lossless on any input.
- **Bits-per-byte** (D-003): total negative log-likelihood in bits divided by the number of
  bytes modeled; comparable across tokenizers, unlike per-token loss.
- **Parameter accounting**: every parameter is attributed to embedding, attention, MLP,
  normalization, positional or output, and the parts must sum to the total. This separates
  "the model improved" from "the model got more parameters".
- **Pre-norm, RoPE, SwiGLU, KV cache**, and **asynchronous GPU timing** are documented here
  when the model and training engine are implemented (steps 6–7).
