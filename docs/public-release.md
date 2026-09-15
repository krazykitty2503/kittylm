# Pre-publication Checklist

The repository is **public**. Every item below must be resolved before any visibility change.
None of them blocks KittyLM 0.1 development. `scripts/check.py quality` lists every
`PRE-PUBLICATION` marker in the repository as a warning.

- [✔️ ] Activate the planned `security@krazykitty.dev` mailbox and remove the PRE-PUBLICATION
      note from `SECURITY.md`.
- [✔️ ] Activate the planned `contact@krazykitty.dev` mailbox and remove the PRE-PUBLICATION
      note from `CODE_OF_CONDUCT.md`.
- [✔️ ] Enable GitHub private vulnerability reporting.
- [✔️ ] Enable secret scanning and push protection.
- [✔️ ] Review committed dataset manifests for private project names or paths.
- [✔️ ] Confirm `DATA_SOURCES.md` is complete and regenerated from the manifests.
- [✔️ ] Re-read `docs/limitations.md`, `docs/responsible-use.md` and `docs/safety.md` against the
      release being published.
- [✔️ ] Decide on visibility explicitly (the default is to stay private).
