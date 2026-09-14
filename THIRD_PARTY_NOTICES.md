# Third-Party Notices

KittyLM's own code is Apache-2.0. It depends on third-party software that keeps its original
licenses. KittyLM does not vendor or redistribute these packages; they are installed from
their publishers.

License identifiers below are the SPDX expressions declared in each package's installed
metadata (read from the development environment for the versions shown). Consult each project
for its authoritative license text.

## Runtime dependencies

| Package | Version checked | Declared license |
|---|---|---|
| PyTorch (`torch`) | 2.13.0+rocm10.0.0 (local ROCm build; CI uses the CPU wheel) | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT |
| NumPy (`numpy`) | 2.5.3 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| PyYAML (`pyyaml`) | 6.0.3 | MIT |
| regex (`regex`) | 2026.9.10 | Apache-2.0 AND CNRI-Python |

## Development tools (not required at runtime)

| Package | Version | Declared license |
|---|---|---|
| pytest | 9.1.1 | MIT |
| Ruff | 0.16.7 | MIT |
| mypy | 2.3.1 | MIT |
| pip-audit | 2.10.1 | Apache Software License (classifier) |

## Documents adapted from third parties

| Document | Source | License |
|---|---|---|
| `LICENSE` | Apache License 2.0, canonical text via the GitHub licenses API | Apache-2.0 |
| `CODE_OF_CONDUCT.md` | Contributor Covenant 2.0, via the GitHub codes-of-conduct API | CC BY 4.0 (Contributor Covenant) |

## Training data

No dataset has been built yet. Per-source training data licenses are recorded in
[DATA_SOURCES.md](DATA_SOURCES.md), generated from dataset manifests.
