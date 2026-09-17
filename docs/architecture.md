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
| Dense Transformer baseline, KV cache, accounting | `kittylm/model/` | B | implemented, tested (CPU + GPU) |
| BENCH-ATTN-001 → D-014 (`reference@bf16`) | `experiments/BENCH-ATTN-001/`, `scripts/bench_attention.py` | B | measured |
| Ledger schema v2 (record kinds, benchmark schema) | `kittylm/ledger.py` | B | implemented, tested |
| Training engine, checkpoints, resume harness | `kittylm/training/`, `kittylm/data/loader.py` | C | implemented, tested (CPU bit-exact; GPU measured) |
| CI evidence (`verify_ci.py`) | `scripts/verify_ci.py`, `kittylm/ledger.py` | C | implemented, tested |
| Evaluation (windows, ppl/bpb, overfit gate, speed) + generation | `kittylm/evaluation/`, `kittylm/inference/`, `scripts/evaluate.py`, `scripts/generate.py` | D | implemented, tested (CPU + GPU; offline end-to-end) |
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
- **The dense baseline** (D-004, `kittylm/model/`): token embedding → N pre-norm blocks → final
  RMSNorm → LM head sharing the embedding matrix (tied weights).
  - *Pre-norm residual blocks*: each sublayer sees `RMSNorm(x)` and adds its output back to `x`,
    so the residual stream carries information through depth unchanged by normalization.
  - *RMSNorm*: divides by the root mean square of the features and applies a learned gain; no
    mean subtraction, no bias.
  - *RoPE* (D-008): queries and keys are rotated by position-dependent angles, so their dot
    product depends on relative distance. No parameters (positional count is 0); the HF-LLaMA
    `rotate_half` layout keeps export straightforward.
  - *Causal attention*: `softmax(QKᵀ/√Dh + mask)·V`. One `attend()` function serves every kernel;
    the plain-PyTorch `reference` path is the oracle every SDPA kernel is compared against, and an
    unavailable kernel is an explicit error, never a silent fallback. The kernel used by
    configs is chosen from measurements (BENCH-ATTN-001, D-014).
  - *SwiGLU*: `down(SiLU(gate(x)) · up(x))`, a gated feed-forward block applied per position.
  - *Mixer seam*: a block is `sequence mixer` + `channel mixer`, looked up by name, which is where
    future GDN or MoE experiments plug in after EXP-001.
  - *Initialization*: weights `~N(0, 0.02)`; the two residual output projections per block use
    `0.02/√(2·layers)` so the residual stream does not grow with depth.
- **KV cache**: during generation each layer's keys and values are stored in preallocated
  buffers, so a new token only computes its own projections and attends over the stored history
  instead of recomputing the whole prefix. Tested to reproduce the full forward pass exactly.
- **Parameter accounting in practice**: `tiny` (d 384, 6 layers, FFN 1024, vocab 16,384) has
  16,913,280 parameters: 6,291,456 embedding (37%, shared with the LM head), 3,538,944 attention,
  7,077,888 MLP, 4,992 normalization, 0 positional, 0 extra output.
- **The training step** (`kittylm/training/engine.py`): set the learning rate for this step
  (linear warmup, then cosine decay to a floor) → for each micro-batch run forward under the
  precision policy, divide the loss by the accumulation count and backpropagate → clip the global
  gradient norm to 1.0 → AdamW step (weight decay only on matrices, not gains) → advance counters.
  A non-finite loss or gradient norm saves a `crash` checkpoint and stops before the bad update;
  that checkpoint holds the loader and RNG state from *before* the step, so a retry replays it.
- **fp16 loss scaling**: fp16 gradients can overflow. The loss scaler detects inf/NaN gradients,
  skips that optimizer update and lowers the scale. A skipped attempt does not count as a step:
  the schedule, `global_step` and `tokens_seen` stay put, the skip is counted and logged, and the
  next attempt uses the next batches. Thirty skips in a row stop the run.
- **Validation**: fixed, non-overlapping windows; the loss is averaged per *token*, so a smaller
  final batch is not over-weighted. With `eval_every > 0` the engine validates on schedule, and
  `best_val.pt` is replaced only when the loss beats the checkpointed best.
- **Precision**: bf16 autocast keeps fp32 master weights and runs matmuls in bf16; fp16 would
  additionally need a gradient scaler; the loss is always computed in fp32.
- **Asynchronous GPU timing**: GPU kernels are queued and return immediately, so a wall-clock
  timer around `model(x)` measures only the enqueueing. The step timer calls
  `torch.cuda.synchronize()` at phase boundaries, but only every `timing_sync_every` steps,
  because synchronizing every step would itself slow training.
- **Checkpoints** (D-021): header with checksum + `weights_only` payload, written to a temporary
  file and atomically renamed; `latest.json` is replaced the same way, so a crash at any moment
  leaves a loadable previous checkpoint.
- **Deterministic resume** (D-012): a checkpoint holds everything that decides the future (model,
  optimizer moments and step counts, scheduler step, global step and tokens seen, every RNG state
  and the loader's generator). The harness proves it by resuming in a fresh process and comparing
  all of these with an uninterrupted run.
- **Windowing** (D-023): a stream longer than the context is split into windows starting at
  multiples of half the context. Each window scores only targets no earlier window scored, and
  generation resets its KV cache at the same starts, so every prediction sees the same history in
  evaluation, in the overfit gate and during generation.
- **Loss, perplexity, bits-per-byte** (D-024): loss is mean nats per predicted token; perplexity
  is `exp(loss)`, "how many equally likely choices" the model is effectively choosing between;
  bits-per-byte converts total NLL to bits and divides by the text bytes, which removes the
  tokenizer from the comparison. Totals are summed before dividing, so every token counts once.
- **Sampling**: temperature divides logits (below 1 sharpens, above 1 flattens); top-k keeps the
  k most likely tokens; top-p keeps the smallest most-likely set whose probability reaches p.
  Temperature 0 is greedy argmax. Generation stops at `<|endoftext|>` or the token budget.
- **KV cache in generation**: prefill runs the prompt once and stores keys/values; each decode
  step then processes one token. Prefill is parallel and fast; batch-1 decode is dominated by the
  fixed cost of launching kernels per token, which is why decode tokens/s is far below prefill.
- **Overfit gate**: memorizing a tiny fixture (loss < 0.05, greedy continuation >= 95% correct)
  proves the stack can fit data before any real experiment is trusted; SMOKE-GPU-001 and EXP-000
  share one implementation.
