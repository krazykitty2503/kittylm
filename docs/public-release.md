# Public Repository Checklist

The repository is **public** (KrazyKitty's decision, 2026-09-15). This checklist separates what
is verified complete, what is attested but not independently verified, what is currently
disabled, and what does not apply yet. Discrepancies are tracked in
[discrepancies.md](discrepancies.md).

Last verified: 2026-09-17 (Milestone A), via the GitHub API and repository checks.

## 1. Verified complete

- [x] Visibility decided explicitly: **public** (GitHub API reports `PUBLIC`).
- [x] Required CI (`Test`: Quality, Tests ubuntu-latest, Tests windows-latest, Security) passes on
      `main`.
- [x] Repository secret scan and artifact guard clean on all tracked files (local and CI).
- [x] `docs/limitations.md`, `docs/responsible-use.md` and `docs/safety.md` reviewed for a public
      repository; no model, weights or results are published yet.

## 2. Attested by KrazyKitty, not independently verified

- [x] `security@krazykitty.dev` published in `SECURITY.md` as active. The domain has an MX record
      (`smtp.google.com`); delivery to this mailbox has not been tested.
- [x] `contact@krazykitty.dev` published in `CODE_OF_CONDUCT.md` as active; same note.

## 3. Currently disabled — action required (X-004)

Verified disabled through the GitHub API on 2026-09-17. Enable in Settings → Code security.

- [ ] Private vulnerability reporting — **disabled**
- [ ] Secret scanning — **disabled**
- [ ] Secret scanning push protection — **disabled**
- [ ] Dependabot security updates — **disabled**

The repository's own secret scan, artifact guard and pre-commit hook still run, but they do not
replace GitHub's server-side push protection.

## 4. Advisory, currently failing (X-005)

- [ ] SonarCloud quality gate (not a required CI job): **failing** — "C Security Rating on New
      Code (required ≥ A)". Review in the SonarCloud UI.

## 5. Not yet applicable (Milestone F)

- [ ] Review each dataset manifest for private project names or paths **before** committing it.
      No dataset exists yet; manifests become public as soon as they are pushed.
- [ ] Regenerate `DATA_SOURCES.md` from the committed manifests. No manifests exist yet.
