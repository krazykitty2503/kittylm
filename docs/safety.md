# KittyLM Safety Properties

This document lists the safety properties the system is designed to have, with an honest
implementation status. A property marked *planned* is a design commitment, not a current
guarantee.

| Property | How it is achieved | Status |
|---|---|---|
| Secrets never enter git | Pre-commit hook + CI secret scan and artifact guard | **implemented** (step 1) |
| Secrets never enter records | Ledger validation secret-scans the rendered record before writing | **implemented** (step 1) |
| Test failures cannot print environment secrets | Git/hook test sandboxes strip credential-named environment variables and never render their environment | **implemented** (step 1) |
| Secrets never enter datasets or manifests | Documents pass license validation and the secret scan before anything is written; hits drop the whole document (D-006) | planned (step 3) |
| Secrets never enter logs, checkpoints or samples | Checkpoint metadata whitelist; samples scanned and redacted before saving | planned (steps 7–8) |
| Special tokens cannot be injected from text | Tokenizer never produces special tokens from raw text unless explicitly allowed | planned (step 2) |
| No outbound network requests in 0.1 | See below | planned (checked from step 8) |
| No telemetry | KittyLM contains no telemetry code | **true today** (nothing is sent anywhere) |
| Tool access mediated by KittyOS | The model never touches the OS; see `kittylm/integration/README.md` | design boundary (future) |
| No automatic learning from feedback | Feedback → human review → curated, versioned dataset (D-009) | design boundary (future) |

## Network behaviour

**KittyLM 0.1 performs no outbound network requests and has no telemetry.** This is checked at
runtime/integration level where applicable, as defense-in-depth: the end-to-end test (step 8)
runs the pipeline with outbound socket connections blocked. That check catches accidental
network use on the tested paths; it is **not** proof that no code path can ever make a request.

There is deliberately **no** ban on importing network modules, so future KittyOS integration is
not constrained. Future networked functionality must live behind the explicit integration
boundary (`kittylm/integration/`) and is out of scope for 0.1.

Development tooling is separate from KittyLM itself: `pip-audit` queries vulnerability
databases when the security checks run.
