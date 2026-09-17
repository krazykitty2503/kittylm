# Configuration

Every YAML file under `configs/` must declare a top-level `kind:` that maps to a registered
typed schema (`kittylm.config.register_config_kind`). CI loads every file strictly (D-011):
unknown keys, duplicate keys, missing keys and type mismatches are errors.

YAML note: write floats with a dot in exponents (`3.0e-4`, not `3e-4`); YAML 1.1 reads `3e-4`
as a string. Command-line overrides (`key.path=3e-4`) are parsed as numbers.

## Kinds

| Kind | Schema | Files |
|---|---|---|
| `tokenizer` | `kittylm.tokenizer.trainer.TokenizerConfig` | `tokenizer/smoke.yaml` |

`tokenizer/smoke.yaml` is the engineering-only SMOKE-GPU-001 vocabulary (512 ids). It uses
`min_pair_count: 1` because its small fixture supports only 158 merges at the research default of
2 (measured); research tokenizers keep `min_pair_count: 2` (D-020).

Vocabulary floors: the trainer itself accepts the theoretical minimum of 288 ids (256 bytes + 32
special tokens), but `kind: tokenizer` configs must also be a multiple of 64 (D-002), so 320 is
the smallest valid configured `vocab_size`.

Data recipes, model and experiment configs arrive with Milestones B–G.
