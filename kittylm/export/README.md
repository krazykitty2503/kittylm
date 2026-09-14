# Export boundary (architectural hook, no code in 0.1)

KittyLM's research implementation is native PyTorch and does not depend on any inference
runtime. Export is a separate, later boundary so the research code never has to bend around
one deployment format.

```text
PyTorch checkpoint
      │
      ├── native KittyLM inference (kittylm/inference, step 8)
      │
      ├── Hugging Face (config + safetensors)      → huggingface/README.md
      │
      └── deployment formats
             └── GGUF → Ollama → KittyOS            → ollama/README.md
```

## Rules

- No export code exists in 0.1, and export libraries are not dependencies.
- Export consumes checkpoints plus their experiment record; it never modifies training code.
- Every exported artifact must carry the source record ID, dataset version and tokenizer
  version so provenance survives conversion.
- Exported weights are never committed to git (artifact guard).

Status: **deferred** (release milestone). See `docs/backlog.md`.
