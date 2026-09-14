# Configuration

Every YAML file under `configs/` must declare a top-level `kind:` that maps to a registered
typed schema (`kittylm.config.register_config_kind`). CI loads every file strictly (D-011):
unknown keys, duplicate keys, missing keys and type mismatches are errors.

YAML note: write floats with a dot in exponents (`3.0e-4`, not `3e-4`); YAML 1.1 reads `3e-4`
as a string. Command-line overrides (`key.path=3e-4`) are parsed as numbers.

No configuration files exist yet. Data recipes, tokenizer, model and experiment configs arrive
with plan steps 2–10.
