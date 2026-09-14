# KittyOS integration boundary (architectural hook, no code in 0.1)

KittyLM is intended to become an intelligence component of KittyOS. This file defines the
contract so research code does not drift into shapes that would be unsafe to integrate.
Status: **deferred** — nothing here is implemented in 0.1.

## Request / response contract

```text
KittyOS
   │
   ├── prompt
   ├── context
   ├── tool_result          (output of a tool KittyOS executed)
   └── validation_result    (output of a KittyOS validator)
          │
          ▼
      KittyLM  ──────────►  response  (plan, tool request, validation, final answer)
                                │
                                ▼
                            validator (KittyOS)
                                │
                  ┌─────────────┼─────────────┐
                  ▼             ▼             ▼
              accepted      corrected      rejected
```

- **The model never touches the operating system directly.** It can only *request* tools;
  KittyOS decides whether and how to execute them, and returns results as data.
- Tool results and context are untrusted data. The tokenizer never turns raw text into control
  tokens (planned, step 2), so a tool result cannot impersonate a protocol segment.
- Structured protocol segments (`system`, `user`, `context`, `plan`, `tool_request`,
  `tool_result`, `validation`, `final`) have reserved special tokens from the first tokenizer.
- Any networked functionality lives behind this boundary, not inside the model package
  (see `docs/safety.md`).

## Feedback loop (never automatic)

```text
feedback → human review → curated dataset → versioned dataset (new recipe_version)
         → experiment → evaluation → new model
```

- Feedback **never** trains a model directly (D-009).
- Feedback data passes the same boundary as every other source: license validation, secret
  scan, quality filtering, classification, deduplication, split.
- A model trained on feedback is a new experiment with its own record, evaluated against the
  previous model before it replaces anything.
- This protects the model from learning bad, wrong or malicious feedback, and keeps every
  model reproducible from a dataset version.
