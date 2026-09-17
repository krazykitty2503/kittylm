"""Run KittyLM checks. GitHub Actions calls these groups, so local == CI.

Usage:
python scripts/check.py all          # quality + tests + security
python scripts/check.py quality      # format, lint, types, configs, imports, structure, ledger,
                                     # package
python scripts/check.py tests        # pytest -m "not gpu"
python scripts/check.py security     # secrets, artifacts, provenance, dependency audit
python scripts/check.py hook         # pre-commit: staged secrets + artifacts
python scripts/check.py package      # build wheel + sdist
python scripts/check.py lint types   # any individual steps

GPU tests are never part of these groups; run them locally with `pytest -m gpu`.
"""

from __future__ import annotations

import importlib
import pkgutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def _run(*args: str) -> int:
    print(f"  $ {' '.join(args)}", flush=True)
    return subprocess.call([PY, *args], cwd=ROOT)


def check_configs() -> int:
    from kittylm.config import validate_config_tree

    count, errors = validate_config_tree(ROOT / "configs")
    for error in errors:
        print(f"  {error}")
    print(f"  config validation: {count} file(s), {len(errors)} error(s)")
    return 1 if errors else 0


def check_imports() -> int:
    import kittylm

    failures = 0
    modules = sorted(m.name for m in pkgutil.walk_packages(kittylm.__path__, "kittylm."))

    for name in ["kittylm", *modules]:
        try:
            importlib.import_module(name)
        except Exception as exc:  # report every broken module, not just the first
            failures += 1
            print(f"  import failed: {name}: {type(exc).__name__}: {exc}")

    print(f"  import check: {len(modules) + 1} module(s), {failures} failure(s)")
    return 1 if failures else 0


def check_ledger() -> int:
    return _run("scripts/build_ablation_table.py", "--check")


def check_package() -> int:
    """Build the source distribution and wheel to verify packaging."""
    build_dir = ROOT / "dist"

    if build_dir.exists():
        for path in build_dir.iterdir():
            if path.is_file():
                path.unlink()

    code = _run("-m", "build", "--outdir", str(build_dir))
    if code != 0:
        return code

    artifacts = sorted(path.name for path in build_dir.iterdir() if path.is_file())
    wheels = [name for name in artifacts if name.endswith(".whl")]
    sdists = [name for name in artifacts if name.endswith(".tar.gz")]

    print(f"  package build: {len(wheels)} wheel(s), {len(sdists)} source distribution(s)")

    if not wheels or not sdists:
        print("  package build failed: expected both wheel and source distribution")
        return 1

    for artifact in artifacts:
        print(f"    {artifact}")

    return 0


def _artifacts(staged: bool) -> int:
    from kittylm.utils.artifacts import check_entries
    from kittylm.utils.git import index_entries, read_blobs

    entries = index_entries(ROOT, staged_only=staged)
    violations = check_entries(entries, lambda blobs: read_blobs(ROOT, blobs))

    for violation in violations:
        print(f"  {violation.path}: {violation.reason}")

    scope = "staged" if staged else "tracked"
    print(f"  artifact guard: {len(entries)} {scope} file(s), {len(violations)} violation(s)")
    return 1 if violations else 0


def check_provenance() -> int:
    recipes = sorted((ROOT / "configs" / "data").glob("*.yaml"))
    manifests = sorted((ROOT / "data" / "manifests").glob("**/manifest.json"))

    if recipes or manifests:
        # Recipe/manifest validation arrives with the dataset tooling (plan step 3).
        # Until it exists, refuse rather than silently pass unvalidated data.
        print("  data recipes/manifests exist but provenance validation is not implemented yet")
        return 1

    data_sources = (ROOT / "DATA_SOURCES.md").read_text(encoding="utf-8")

    if "No dataset has been built yet." not in data_sources:
        print("  DATA_SOURCES.md must state that no dataset has been built (no manifests exist)")
        return 1

    print("  provenance: 0 recipes, 0 manifests; DATA_SOURCES.md sync check inactive until step 3")
    return 0


def check_audit() -> int:
    # --local keeps local development focused on the active environment.
    # CI runs without a project venv, so the installed CI environment is audited.
    return _run(
        "-m",
        "pip_audit",
        "--local",
        "--skip-editable",
        "--progress-spinner",
        "off",
    )


STEPS: dict[str, Callable[[], int]] = {
    "format": lambda: _run("-m", "ruff", "format", "--check", "."),
    "lint": lambda: _run("-m", "ruff", "check", "."),
    "types": lambda: _run("-m", "mypy"),
    "configs": check_configs,
    "imports": check_imports,
    "structure": lambda: _run(
        "-m",
        "pytest",
        "-q",
        "tests/test_docs_contract.py",
        "tests/test_import_boundaries.py",
        "tests/test_ci_contract.py",
    ),
    "ledger": check_ledger,
    "package": check_package,
    "pytest": lambda: _run("-m", "pytest", "-m", "not gpu"),
    "secrets": lambda: _run("scripts/scan_secrets.py"),
    "artifacts": lambda: _artifacts(staged=False),
    "provenance": check_provenance,
    "audit": check_audit,
    "hook-secrets": lambda: _run("scripts/scan_secrets.py", "--staged"),
    "hook-artifacts": lambda: _artifacts(staged=True),
}

GROUPS: dict[str, list[str]] = {
    "quality": [
        "format",
        "lint",
        "types",
        "configs",
        "imports",
        "structure",
        "ledger",
        "package",
    ],
    "tests": ["pytest"],
    "security": ["secrets", "artifacts", "provenance", "audit"],
    "hook": ["hook-secrets", "hook-artifacts"],
    "package": ["package"],
}

GROUPS["all"] = GROUPS["quality"] + GROUPS["tests"] + GROUPS["security"]


def main(argv: list[str]) -> int:
    if not argv or argv[0] in {"-h", "--help"}:
        print(__doc__)
        print("groups:", ", ".join(sorted(GROUPS)))
        print("steps: ", ", ".join(STEPS))
        return 0 if argv else 2

    selected: list[str] = []

    for name in argv:
        if name in GROUPS:
            selected.extend(GROUPS[name])
        elif name in STEPS:
            selected.append(name)
        else:
            print(f"unknown group or step: {name!r}")
            return 2

    results: list[tuple[str, int, float]] = []

    for step in selected:
        print(f"\n== {step}", flush=True)
        start = time.perf_counter()
        code = STEPS[step]()
        results.append((step, code, time.perf_counter() - start))

    print("\n== summary")

    for step, code, seconds in results:
        print(f"  {'PASS' if code == 0 else 'FAIL'}  {step:<15} {seconds:6.1f}s")

    failed = [step for step, code, _ in results if code != 0]

    print(f"\n{'FAILED: ' + ', '.join(failed) if failed else 'all checks passed'}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
