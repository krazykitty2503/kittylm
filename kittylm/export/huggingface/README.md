# Hugging Face export (future boundary)

Status: **deferred** — no code, no dependency in 0.1.

## Intent

Expose KittyLM to the Hugging Face ecosystem through thin adapters, without making Hugging Face
part of training:

- `KittyLMConfig` — maps a resolved KittyLM model config to an HF-style config.
- `KittyLMModel` — wraps the native model for HF-style loading and generation.
- `KittyLMTokenizer` — wraps the native byte-level BPE tokenizer and its special tokens.

Weights would be exported as safetensors with the experiment record ID, dataset version and
tokenizer version in the metadata.

## Constraints

- `transformers` is **never** a training dependency. KittyLM implements its fundamentals
  itself on purpose; adapters translate, they do not replace.
- The dense baseline uses the HF-LLaMA RoPE layout and tied embeddings (D-008), so its weights
  map onto a LLaMA-style layout directly. Hybrid architectures (GDN, MoE) would need custom
  modeling code.
