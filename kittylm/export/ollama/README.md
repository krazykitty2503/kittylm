# GGUF / Ollama export (future boundary)

Status: **deferred** — no code in 0.1.

## Intended path

```text
KittyLM checkpoint → safetensors (huggingface/) → GGUF conversion → Ollama model → KittyOS
```

## Notes

- The dense baseline is llama-compatible by construction: RMSNorm, RoPE in the `rotate_half`
  layout, SwiGLU, no biases, tied embeddings (D-004, D-008). That makes GGUF conversion a
  mapping exercise rather than a porting project.
- Future hybrid architectures (Gated DeltaNet, MoE variants, recurrence) would need custom
  llama.cpp support. That deployment cost must be weighed in those experiments.
- The byte-level BPE tokenizer and its reserved special tokens must be carried into the GGUF
  vocabulary exactly; a conversion that changes token ids invalidates evaluation results.
- Quantized exports are new artifacts and need their own evaluation before any claim about
  quality is made.
