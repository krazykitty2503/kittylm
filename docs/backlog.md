# KittyLM Backlog

Ideas are classified so that nothing non-critical derails the current milestone:
**BLOCKING** · **IMPORTANT** · **NICE_TO_HAVE** · **FUTURE** · **EXPERIMENTAL**.
Moving an item into active work requires a plan revision.

## BLOCKING

_None._

## IMPORTANT

- Broader licensed corpus and a 32,768-vocabulary tokenizer (0.2).
- Near-duplicate (MinHash) deduplication.
- Document-boundary attention masking for packed windows.
- Coding and computer-science evaluation suites.
- Per-source mixture weights.

## NICE_TO_HAVE

- `torch.compile` once Triton works on ROCm/Windows.
- Hugging Face `tokenizers` backend for faster encoding.
- TensorBoard view of `metrics.jsonl`.

## EXPERIMENTAL

- Synthetic-only training.
- Staged domain curriculum versus mixed training (needs EXP-001 as the control).
- Code → documentation → explanation → test → debugging pair corpus.

## FUTURE (intentionally deferred from 0.1)

| Item | Hook that exists in 0.1 | Earliest milestone |
|---|---|---|
| Distributed training (DDP/FSDP/TP/PP) | `docs/scaling.md` stages and rule | after single-device correctness at larger scale |
| Cloud/HPC training | `docs/scaling.md` stage 5 | later |
| DVC / data platform | deterministic `dataset_version` hash | when the corpus reaches hundreds of GB |
| Production telemetry/monitoring | per-run metrics in the training logger; no telemetry by design | KittyOS integration |
| Automatic online learning | never automatic; curated feedback loop (`kittylm/integration/`) | post-training milestones |
| Hugging Face training integration / export code | `kittylm/export/huggingface/README.md` | release milestone |
| Ollama / GGUF export | `kittylm/export/ollama/README.md`, llama-compatible baseline | release milestone |
| Large-scale benchmark infrastructure | ledger + generated ablation table | 0.2+ |
| Self-hosted GPU CI runner | `gpu` pytest marker, PR checklist | only with a hardened runner setup |
| GDN, hybrid blocks, MoE (8 experts / 2 active first), recurrence → adaptive halting, long context, LoRA adapters, KittyOS protocol post-training | mixer seam in the block (step 6); reserved special tokens (step 2) | experiment series after EXP-001 |
