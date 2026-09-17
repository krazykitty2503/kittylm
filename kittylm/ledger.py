"""Experiment ledger: schemas, validators and writers for records and benchmarks.

Purpose:
    Make every reported number traceable to one validated file that pins the code (git
    commit), data, tokenizer, model shape and parameter budget, and setup. The ablation table is
    *generated* from formal records, so a result that is not in a record cannot appear in the
    repository (CONTRIBUTING.md). Engineering measurements (benchmarks) and smoke runs are
    recorded too, but kept structurally separate from formal experiments (D-017).

Public API:
    ExperimentRecord (and its section dataclasses), schema version 2
        ``experiment.kind`` is ``formal`` (``EXP-###[-suffix]``) or ``smoke``
        (``SMOKE-<AREA>-###``, must carry SMOKE_LIMITATION).
    parse_record / validate_record / render_record / write_record / load_record /
    load_all_records / render_ablation_table (formal records only)
    BenchmarkRecord (and its section dataclasses)
        ``experiments/BENCH-<AREA>-###/benchmark.yaml``.
    parse_benchmark / validate_benchmark / render_benchmark / write_benchmark /
    load_benchmark / load_all_benchmarks / select_attention_path
    LedgerError
        Raised with every problem found, not just the first.

Invariants:
    - Parameter accounting parts sum exactly to ``total``; ``non_embedding = total - embedding``.
    - ``final_ppl == exp(final_loss)`` within a relative tolerance of 1e-3 (natural-log loss).
    - ``checkpoint_resume_test`` is ``passed``/``failed`` only with evidence produced by the
      resume harness (RESUME_HARNESS); ``not_run`` must carry no evidence.
    - Smoke records always state they are engineering-only and never enter the ablation table.
    - A benchmark covers its whole grid (every variant x flag setting x length exactly once),
      every non-ok cell explains itself, and its recorded selection equals the selection
      recomputed from its cells.
    - Files contain no absolute paths, no local username/hostname, and no secrets.

Failure modes:
    - Any schema or semantic violation raises LedgerError listing all problems.
    - Identifier checks against the local username/hostname only apply at write time; CI
      re-validation cannot know the author's machine and checks paths and secrets instead.
    - Formal records without ``ci_evidence`` proving every required job (REQUIRED_CI_JOBS of
      workflow ``Test``) succeeded on ``repository.git_commit`` are rejected (D-018). The check is
      structural (the evidence is produced by scripts/verify_ci.py), not cryptographic.
    - ``resume_evidence`` is produced only by ``kittylm.training.resume_harness``; a test scans
      the code base so nothing else constructs it.

See:
    Plan rev 3.3 section 6, D-012 (resume evidence), D-014 (attention selection), D-017.
"""

from __future__ import annotations

import getpass
import math
import os
import platform
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from kittylm.config import ConfigError, from_dict, load_yaml, to_dict
from kittylm.data.secrets import scan_text

__all__ = [
    "BENCHMARK_STATUSES",
    "RESUME_HARNESS",
    "SCHEMA_VERSION",
    "SMOKE_LIMITATION",
    "BenchmarkCell",
    "BenchmarkRecord",
    "ExperimentRecord",
    "LedgerError",
    "load_all_benchmarks",
    "load_all_records",
    "load_benchmark",
    "load_record",
    "local_identifiers",
    "parse_benchmark",
    "parse_record",
    "render_ablation_table",
    "render_benchmark",
    "render_record",
    "select_attention_path",
    "validate_benchmark",
    "validate_record",
    "write_benchmark",
    "write_record",
    "REQUIRED_CI_JOBS",
    "REQUIRED_CI_WORKFLOW",
    "CiEvidence",
    "CiJob",
    "ResumeEvidence",
    "ci_evidence_from_github",
    "ci_evidence_problems",
]

SCHEMA_VERSION = 2
RESUME_HARNESS = "kittylm.training.resume_harness"
PPL_REL_TOL = 1e-3
RATIO_REL_TOL = 1e-3
SMOKE_LIMITATION = (
    "SMOKE: engineering-only test. Must not be used for architecture or model-quality conclusions."
)

_EXPERIMENT_ID = re.compile(r"^EXP-\d{3}(?:-[a-z0-9]+)*$")
_SMOKE_ID = re.compile(r"^SMOKE-[A-Z0-9]+-\d{3}$")
_BENCHMARK_ID = re.compile(r"^BENCH-[A-Z0-9]+-\d{3}$")
_SHA1_HEX = re.compile(r"^[0-9a-f]{40}$")
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_ABSOLUTE_PATH = re.compile(
    r"""(?:^|[\s"'(=,:])(?:[A-Za-z]:[\\/]|\\\\[^\\\s]+\\|~[\\/]|/(?:home|Users|root|tmp|var|opt|mnt|srv|etc|private)/)"""
)


class LedgerError(ValueError):
    """An experiment record is invalid."""

    def __init__(self, problems: Sequence[str]):
        self.problems = list(problems)
        super().__init__("invalid experiment record:\n  - " + "\n  - ".join(self.problems))


@dataclass(frozen=True)
class ExperimentInfo:
    """Identity of the experiment."""

    id: str
    kind: Literal["formal", "smoke"]
    name: str
    description: str


@dataclass(frozen=True)
class RepositoryInfo:
    """Code state the run used."""

    git_commit: str
    git_dirty: bool


@dataclass(frozen=True)
class EnvironmentInfo:
    """Software and hardware environment (no hostnames, usernames or paths)."""

    python: str
    pytorch: str
    backend: Literal["rocm", "cuda", "cpu"]
    backend_version: str | None
    device: str
    dtype: Literal["bf16", "fp16", "fp32"]
    os: str


@dataclass(frozen=True)
class DatasetInfo:
    """Dataset identity and size, from the manifest and packed metadata."""

    name: str
    version: str  # dataset_version: sha256 of the canonical manifest content
    manifest_sha256: str
    recipe_version: int
    sources: int
    documents: int
    training_bytes: int
    validation_bytes: int | None
    training_tokens: int
    validation_tokens: int | None


@dataclass(frozen=True)
class TokenizerInfo:
    """Tokenizer identity and compression on the training split."""

    version: str
    vocab_size: int
    tokenizer_sha256: str
    bytes_per_token: float
    tokens_per_byte: float


@dataclass(frozen=True)
class ParameterAccounting:
    """Mandatory parameter budget. Parts must sum exactly to ``total``."""

    total: int
    trainable: int
    embedding: int
    attention: int
    mlp: int
    normalization: int
    positional: int
    output: int
    tied_parameters: int
    non_embedding: int


@dataclass(frozen=True)
class ModelInfo:
    """Model shape and its parameter budget."""

    architecture: str
    d_model: int
    layers: int
    heads: int
    ffn_dim: int
    context_length: int
    parameters: ParameterAccounting


@dataclass(frozen=True)
class TrainingInfo:
    """Optimization setup and compute actually consumed."""

    optimizer: str
    learning_rate: float
    schedule: str
    batch_size: int
    gradient_accumulation: int
    max_steps: int
    training_steps: int
    tokens_seen: int
    bytes_seen: int
    precision: Literal["bf16", "fp16", "fp32"]
    attention_kernel: str


@dataclass(frozen=True)
class PhaseTiming:
    """Summary of one training phase's wall-clock time."""

    mean_ms: float
    p50_ms: float
    p95_ms: float


@dataclass(frozen=True)
class InferenceSpeed:
    """Batch-1 inference throughput."""

    prefill_tok_s: float
    decode_tok_s: float


@dataclass(frozen=True)
class ResultsInfo:
    """Measured results. ``None`` means not measured, never estimated."""

    final_loss: float
    final_ppl: float
    final_bpb: float | None
    val_bpb_by_category: dict[str, float]
    test_ppl: float | None
    tokens_per_second: float
    samples_per_second: float | None
    timing_breakdown: dict[str, PhaseTiming]
    peak_vram_gb: float | None
    allocated_vram_gb: float | None
    duration_seconds: float
    inference: InferenceSpeed | None


@dataclass(frozen=True)
class ResumeEvidence:
    """Produced only by the resume harness when it actually kills and resumes a run."""

    harness: str
    run_id: str
    kill_step: int
    resumed_to_step: int
    metrics_sha256: str
    comparison: dict[str, str] = field(default_factory=dict)


REQUIRED_CI_WORKFLOW = "Test"
REQUIRED_CI_JOBS: tuple[str, ...] = (
    "Quality",
    "Tests (ubuntu-latest)",
    "Tests (windows-latest)",
    "Security",
)


@dataclass(frozen=True)
class CiJob:
    """One GitHub Actions job result."""

    job_id: int
    conclusion: str


@dataclass(frozen=True)
class CiEvidence:
    """Required CI results for the exact commit a formal experiment runs on (D-018)."""

    commit: str
    workflow: str
    run_id: int
    event: str
    jobs: dict[str, CiJob]


def ci_evidence_problems(evidence: CiEvidence | None, commit: str) -> list[str]:
    """Why ``evidence`` does not prove required CI is green on ``commit`` (empty if it does)."""
    if evidence is None:
        return ["ci_evidence is required: run scripts/verify_ci.py on the exact commit"]
    problems: list[str] = []
    if evidence.commit != commit:
        problems.append(f"ci_evidence is for commit {evidence.commit[:10]}, not {commit[:10]}")
    if evidence.workflow != REQUIRED_CI_WORKFLOW:
        problems.append(f"ci_evidence workflow must be {REQUIRED_CI_WORKFLOW!r}")
    if evidence.run_id <= 0:
        problems.append("ci_evidence.run_id must be a positive GitHub run id")
    for job in REQUIRED_CI_JOBS:
        result = evidence.jobs.get(job)
        if result is None:
            problems.append(f"ci_evidence is missing required job {job!r}")
        elif result.conclusion != "success":
            problems.append(f"required job {job!r} concluded {result.conclusion!r}")
    return problems


def ci_evidence_from_github(
    run: Mapping[str, Any], jobs: Sequence[Mapping[str, Any]]
) -> CiEvidence:
    """Build CiEvidence from GitHub API run and jobs JSON (no network access here)."""
    return CiEvidence(
        commit=str(run["head_sha"]),
        workflow=str(run["name"]),
        run_id=int(run["id"]),
        event=str(run["event"]),
        jobs={
            str(job["name"]): CiJob(job_id=int(job["id"]), conclusion=str(job["conclusion"]))
            for job in jobs
        },
    )


@dataclass(frozen=True)
class ReproducibilityInfo:
    """How reproducible the run is, and whether resume was actually exercised."""

    determinism_mode: Literal["deterministic", "semi_deterministic", "nondeterministic"]
    checkpoint_resume_test: Literal["passed", "failed", "not_run"]
    resume_evidence: ResumeEvidence | None = None


@dataclass(frozen=True)
class ExperimentRecord:
    """One experiment's complete, validated record."""

    schema_version: int
    experiment: ExperimentInfo
    repository: RepositoryInfo
    environment: EnvironmentInfo
    dataset: DatasetInfo
    tokenizer: TokenizerInfo
    model: ModelInfo
    training: TrainingInfo
    results: ResultsInfo
    reproducibility: ReproducibilityInfo
    ci_evidence: CiEvidence | None = None
    limitations: list[str] = field(default_factory=list)
    notes: str = ""


def local_identifiers() -> tuple[str, ...]:
    """Return the local username and hostname, which must never appear in a record."""
    candidates = []
    try:
        candidates.append(getpass.getuser())
    except (OSError, KeyError):
        pass
    candidates.append(platform.node())
    return tuple(sorted({c for c in candidates if c and len(c) >= 3}))


def _iter_strings(value: Any, path: str) -> Iterable[tuple[str, str]]:
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, Mapping):
        for k, v in value.items():
            yield from _iter_strings(k, f"{path}.<key>")
            yield from _iter_strings(v, f"{path}.{k}")
    elif isinstance(value, list | tuple):
        for i, v in enumerate(value):
            yield from _iter_strings(v, f"{path}[{i}]")


def _close(a: float, b: float, rel_tol: float) -> bool:
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=1e-12)


def validate_record(record: ExperimentRecord, forbidden_identifiers: Sequence[str] = ()) -> None:
    """Check semantic invariants of a structurally valid record.

    Raises:
        LedgerError: Listing every violated invariant.
    """
    problems: list[str] = []
    r = record

    if r.schema_version != SCHEMA_VERSION:
        problems.append(f"schema_version must be {SCHEMA_VERSION}, got {r.schema_version}")
    if r.experiment.kind == "formal" and not _EXPERIMENT_ID.match(r.experiment.id):
        problems.append(f"formal experiment.id {r.experiment.id!r} must match EXP-###[-suffix]")
    if r.experiment.kind == "smoke":
        if not _SMOKE_ID.match(r.experiment.id):
            problems.append(f"smoke experiment.id {r.experiment.id!r} must match SMOKE-<AREA>-###")
        if SMOKE_LIMITATION not in r.limitations:
            problems.append("smoke records must list the SMOKE engineering-only limitation")
    if not _SHA1_HEX.match(r.repository.git_commit):
        problems.append("repository.git_commit must be a full 40-character lowercase hash")
    for field_name, digest in (
        ("dataset.version", r.dataset.version),
        ("dataset.manifest_sha256", r.dataset.manifest_sha256),
        ("tokenizer.tokenizer_sha256", r.tokenizer.tokenizer_sha256),
    ):
        if not _SHA256_HEX.match(digest):
            problems.append(f"{field_name} must be a 64-character lowercase sha256 hex digest")

    # Non-negative / positive numbers.
    plain = to_dict(r)
    for section in ("dataset", "tokenizer", "model", "training", "results"):
        for number_path, number in _iter_numbers(plain[section], section):
            if number < 0:
                problems.append(f"{number_path} must be >= 0, got {number}")
    for field_name, count in (
        ("tokenizer.vocab_size", r.tokenizer.vocab_size),
        ("model.parameters.total", r.model.parameters.total),
        ("training.batch_size", r.training.batch_size),
        ("training.gradient_accumulation", r.training.gradient_accumulation),
        ("dataset.training_tokens", r.dataset.training_tokens),
    ):
        if count <= 0:
            problems.append(f"{field_name} must be > 0, got {count}")

    # Mandatory parameter accounting.
    p = r.model.parameters
    parts = p.embedding + p.attention + p.mlp + p.normalization + p.positional + p.output
    if parts != p.total:
        problems.append(
            "model.parameters: embedding+attention+mlp+normalization+positional+output "
            f"= {parts} != total {p.total}"
        )
    if p.non_embedding != p.total - p.embedding:
        problems.append(
            f"model.parameters.non_embedding {p.non_embedding} != total - embedding "
            f"({p.total - p.embedding})"
        )
    if p.trainable > p.total or p.tied_parameters > p.total:
        problems.append("model.parameters: trainable and tied_parameters cannot exceed total")

    # Perplexity must be exp(natural-log loss).
    try:
        expected_ppl = math.exp(r.results.final_loss)
        if not _close(r.results.final_ppl, expected_ppl, PPL_REL_TOL):
            problems.append(
                f"results.final_ppl {r.results.final_ppl} != exp(final_loss) {expected_ppl:.6g}"
            )
    except OverflowError:
        problems.append("results.final_loss is too large for exp()")

    # Byte/token accounting must be self-consistent.
    t, d = r.tokenizer, r.dataset
    if d.training_tokens > 0 and not _close(
        t.bytes_per_token, d.training_bytes / d.training_tokens, RATIO_REL_TOL
    ):
        problems.append("tokenizer.bytes_per_token != dataset.training_bytes / training_tokens")
    if not _close(t.bytes_per_token * t.tokens_per_byte, 1.0, RATIO_REL_TOL):
        problems.append("tokenizer.bytes_per_token * tokens_per_byte must equal 1")

    # Resume evidence.
    rep = r.reproducibility
    if rep.checkpoint_resume_test == "not_run":
        if rep.resume_evidence is not None:
            problems.append("resume_evidence must be null when checkpoint_resume_test is not_run")
    else:
        ev = rep.resume_evidence
        if ev is None:
            problems.append(
                f"checkpoint_resume_test={rep.checkpoint_resume_test} requires resume_evidence"
            )
        else:
            if ev.harness != RESUME_HARNESS:
                problems.append(f"resume_evidence.harness must be {RESUME_HARNESS!r}")
            if not 0 < ev.kill_step < ev.resumed_to_step:
                problems.append("resume_evidence requires 0 < kill_step < resumed_to_step")
            if not _SHA256_HEX.match(ev.metrics_sha256):
                problems.append("resume_evidence.metrics_sha256 must be a sha256 hex digest")
            if not ev.comparison:
                problems.append("resume_evidence.comparison must summarize every compared category")

    # Required CI evidence (D-018): mandatory for formal experiments.
    if r.experiment.kind == "formal":
        problems.extend(ci_evidence_problems(r.ci_evidence, r.repository.git_commit))
    elif r.ci_evidence is not None:
        problems.extend(ci_evidence_problems(r.ci_evidence, r.repository.git_commit))

    if not r.limitations:
        problems.append("limitations must list at least one known limitation")

    # No absolute paths or local identifiers anywhere.
    for path, text in _iter_strings(plain, "record"):
        if _ABSOLUTE_PATH.search(text):
            problems.append(f"{path} contains an absolute path")
        for ident in forbidden_identifiers:
            pattern = rf"(?<![A-Za-z0-9]){re.escape(ident)}(?![A-Za-z0-9])"
            if re.search(pattern, text, flags=re.IGNORECASE):
                problems.append(f"{path} contains a local username/hostname")

    # No secrets in the rendered record.
    findings = scan_text(render_record(r))
    if findings:
        rules = sorted({f.rule for f in findings})
        problems.append(f"rendered record trips the secret scanner (rules: {rules})")

    if problems:
        raise LedgerError(problems)


def _iter_numbers(value: Any, path: str) -> Iterable[tuple[str, float]]:
    if isinstance(value, bool):
        return
    if isinstance(value, int | float):
        yield path, value
    elif isinstance(value, Mapping):
        for k, v in value.items():
            yield from _iter_numbers(v, f"{path}.{k}")


def parse_record(
    data: Mapping[str, Any], forbidden_identifiers: Sequence[str] = ()
) -> ExperimentRecord:
    """Parse plain data into a validated ExperimentRecord.

    Raises:
        LedgerError: On structural or semantic problems.
    """
    try:
        record = from_dict(ExperimentRecord, data)
    except ConfigError as exc:
        raise LedgerError([str(exc)]) from exc
    validate_record(record, forbidden_identifiers)
    return record


def render_record(record: ExperimentRecord) -> str:
    """Render a record as deterministic YAML in schema field order."""
    return yaml.safe_dump(to_dict(record), sort_keys=False, allow_unicode=True, width=100)


def write_record(record: ExperimentRecord, repo_root: Path) -> Path:
    """Validate and atomically write ``experiments/<ID>/record.yaml``.

    Local username and hostname are forbidden in the record at write time.
    """
    validate_record(record, forbidden_identifiers=local_identifiers())
    target = repo_root / "experiments" / record.experiment.id / "record.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=".record-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(render_record(record))
        os.replace(tmp_name, target)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return target


def load_record(path: Path) -> ExperimentRecord:
    """Load and validate a record; its directory name must equal the experiment id."""
    try:
        data = load_yaml(path)
    except ConfigError as exc:
        raise LedgerError([f"{path.parent.name}/{path.name}: {exc}"]) from exc
    record = parse_record(data)
    if path.parent.name != record.experiment.id:
        raise LedgerError(
            [f"record directory {path.parent.name!r} != experiment.id {record.experiment.id!r}"]
        )
    return record


def load_all_records(experiments_dir: Path) -> list[ExperimentRecord]:
    """Load every ``<experiments_dir>/*/record.yaml`` sorted by experiment id."""
    if not experiments_dir.is_dir():
        return []
    records = [load_record(p) for p in sorted(experiments_dir.glob("*/record.yaml"))]
    return sorted(records, key=lambda rec: rec.experiment.id)


def _fmt_int(value: int | None) -> str:
    return "—" if value is None else f"{value:,}"


def _fmt_float(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


_TABLE_HEADER = """# KittyLM ablation table

<!-- GENERATED FILE: do not edit. Regenerate: python scripts/build_ablation_table.py -->

Every row is rendered from a validated `experiments/<ID>/record.yaml`. A result that is not in
a record cannot appear here (see CONTRIBUTING.md). Loss is natural-log cross-entropy; bpb is
bits per byte, the tokenizer-independent comparison metric (D-003).
"""

_COLUMNS = (
    "ID",
    "Name",
    "Commit",
    "Vocab",
    "Params",
    "Non-emb",
    "Emb %",
    "Train bytes",
    "Train tokens",
    "Steps",
    "Bytes/token",
    "Final loss",
    "PPL",
    "bpb",
    "Tok/s",
    "Peak VRAM GB",
    "Duration s",
    "Resume test",
)


def render_ablation_table(records: Sequence[ExperimentRecord]) -> str:
    """Render the ablation table markdown from *formal* records only (deterministic)."""
    lines = [_TABLE_HEADER]
    formal = [rec for rec in records if rec.experiment.kind == "formal"]
    if not formal:
        lines.append("_No formal experiment records yet._\n")
        return "\n".join(lines)
    lines.append("| " + " | ".join(_COLUMNS) + " |")
    lines.append("|" + "|".join("---" for _ in _COLUMNS) + "|")
    for rec in sorted(formal, key=lambda x: x.experiment.id):
        p = rec.model.parameters
        commit = rec.repository.git_commit[:10] + (" (dirty)" if rec.repository.git_dirty else "")
        row = (
            rec.experiment.id,
            rec.experiment.name,
            f"`{commit}`",
            _fmt_int(rec.tokenizer.vocab_size),
            _fmt_int(p.total),
            _fmt_int(p.non_embedding),
            f"{100.0 * p.embedding / p.total:.1f}",
            _fmt_int(rec.dataset.training_bytes),
            _fmt_int(rec.dataset.training_tokens),
            _fmt_int(rec.training.training_steps),
            _fmt_float(rec.tokenizer.bytes_per_token, 3),
            _fmt_float(rec.results.final_loss),
            _fmt_float(rec.results.final_ppl),
            _fmt_float(rec.results.final_bpb),
            _fmt_float(rec.results.tokens_per_second, 1),
            _fmt_float(rec.results.peak_vram_gb, 2),
            _fmt_float(rec.results.duration_seconds, 1),
            rec.reproducibility.checkpoint_resume_test,
        )
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


# ================================================================================================
# Benchmarks (engineering measurements, e.g. BENCH-ATTN-001)
# ================================================================================================

BENCHMARK_STATUSES = ("ok", "unsupported", "oom", "error", "mismatch", "nonfinite")
ORACLE_VARIANT_PREFIX = "reference@"
SELECTION_DTYPE = "bf16"


@dataclass(frozen=True)
class BenchmarkInfo:
    """Identity of a benchmark."""

    id: str
    name: str
    description: str


@dataclass(frozen=True)
class BenchmarkEnvironment:
    """Software and hardware environment (no hostnames, usernames or paths)."""

    python: str
    pytorch: str
    backend: Literal["rocm", "cuda", "cpu"]
    backend_version: str | None
    device: str
    os: str


@dataclass(frozen=True)
class BenchmarkGrid:
    """The full measurement grid; every combination must appear exactly once in ``cells``."""

    variants: list[str]  # "<attention path>@<dtype>", e.g. "sdpa_math@bf16"
    flag_settings: list[str]  # values of TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL: "unset", "1"
    sequence_lengths: list[int]
    tokens_per_batch: int
    n_heads: int
    head_dim: int
    warmup_iterations: int
    min_iterations: int
    max_iterations: int
    min_measure_seconds: float
    relative_tolerance: float
    selection_lengths: list[int]


@dataclass(frozen=True)
class BenchmarkCell:
    """One measured (variant, flag setting, sequence length) combination."""

    variant: str
    flag_setting: str
    seq_len: int
    batch_size: int
    status: Literal["ok", "unsupported", "oom", "error", "mismatch", "nonfinite"]
    iterations: int | None
    tokens_per_second: float | None
    first_call_seconds: float | None
    peak_vram_mib: float | None
    allocated_vram_mib: float | None
    max_abs_diff_output: float | None
    max_abs_diff_grad: float | None
    relative_diff_output: float | None
    relative_diff_grad: float | None
    detail: str


@dataclass(frozen=True)
class BenchmarkSelection:
    """Decision derived from the cells (for BENCH-ATTN-001: D-014)."""

    decision: str
    variant: str
    flag_setting: str
    rule: str
    tokens_per_second_by_length: dict[str, float]


@dataclass(frozen=True)
class BenchmarkRecord:
    """A complete, validated benchmark."""

    schema_version: int
    benchmark: BenchmarkInfo
    repository: RepositoryInfo
    environment: BenchmarkEnvironment
    grid: BenchmarkGrid
    cells: list[BenchmarkCell]
    selection: BenchmarkSelection
    limitations: list[str] = field(default_factory=list)
    notes: str = ""


SELECTION_RULE = (
    "Fastest bf16 variant (by tokens/s at the largest selection length) whose cells are ok at "
    "every selection length for one flag setting and whose outputs and gradients match the "
    "bf16 reference within the relative tolerance; ties prefer flag 'unset', then grid order."
)


def _cell_equivalent(cell: BenchmarkCell, tolerance: float) -> bool:
    if cell.variant.startswith(ORACLE_VARIANT_PREFIX):
        return True  # the reference implementation is the oracle itself
    if cell.relative_diff_output is None or cell.relative_diff_grad is None:
        return False
    return cell.relative_diff_output <= tolerance and cell.relative_diff_grad <= tolerance


def select_attention_path(
    grid: BenchmarkGrid, cells: Sequence[BenchmarkCell]
) -> BenchmarkSelection | None:
    """Recompute the D-014 selection from cells (``None`` if nothing qualifies)."""
    index = {(c.variant, c.flag_setting, c.seq_len): c for c in cells}
    largest = max(grid.selection_lengths)
    candidates: list[tuple[float, int, int, str, str]] = []
    for flag_rank, flag in enumerate(grid.flag_settings):
        for variant_rank, variant in enumerate(grid.variants):
            if not variant.endswith("@" + SELECTION_DTYPE):
                continue
            chosen = [index.get((variant, flag, length)) for length in grid.selection_lengths]
            if any(
                c is None
                or c.status != "ok"
                or c.tokens_per_second is None
                or not _cell_equivalent(c, grid.relative_tolerance)
                for c in chosen
            ):
                continue
            speed = index[(variant, flag, largest)].tokens_per_second
            assert speed is not None
            candidates.append((-speed, flag_rank, variant_rank, variant, flag))
    if not candidates:
        return None
    _, _, _, variant, flag = min(candidates)
    by_length = {
        str(length): float(index[(variant, flag, length)].tokens_per_second or 0.0)
        for length in grid.selection_lengths
    }
    return BenchmarkSelection(
        decision="D-014",
        variant=variant,
        flag_setting=flag,
        rule=SELECTION_RULE,
        tokens_per_second_by_length=by_length,
    )


def validate_benchmark(record: BenchmarkRecord, forbidden_identifiers: Sequence[str] = ()) -> None:
    """Check completeness, per-cell consistency and the recomputed selection.

    Raises:
        LedgerError: Listing every violated invariant.
    """
    problems: list[str] = []
    r = record
    g = r.grid
    if r.schema_version != SCHEMA_VERSION:
        problems.append(f"schema_version must be {SCHEMA_VERSION}, got {r.schema_version}")
    if not _BENCHMARK_ID.match(r.benchmark.id):
        problems.append(f"benchmark.id {r.benchmark.id!r} must match BENCH-<AREA>-###")
    if not _SHA1_HEX.match(r.repository.git_commit):
        problems.append("repository.git_commit must be a full 40-character lowercase hash")
    if not set(g.selection_lengths) <= set(g.sequence_lengths) or not g.selection_lengths:
        problems.append("grid.selection_lengths must be a non-empty subset of sequence_lengths")
    if not 0 < g.relative_tolerance < 1:
        problems.append("grid.relative_tolerance must be in (0, 1)")
    if not 1 <= g.min_iterations <= g.max_iterations:
        problems.append("grid requires 1 <= min_iterations <= max_iterations")
    for variant in g.variants:
        if "@" not in variant:
            problems.append(f"grid variant {variant!r} must look like '<path>@<dtype>'")

    expected = {(v, f, n) for v in g.variants for f in g.flag_settings for n in g.sequence_lengths}
    seen: dict[tuple[str, str, int], int] = {}
    for cell in r.cells:
        key = (cell.variant, cell.flag_setting, cell.seq_len)
        seen[key] = seen.get(key, 0) + 1
        where = f"cell {cell.variant} flag={cell.flag_setting} T={cell.seq_len}"
        measured = (cell.tokens_per_second, cell.iterations, cell.first_call_seconds)
        if cell.status in ("ok", "mismatch"):
            if any(value is None for value in measured) or (cell.tokens_per_second or 0) <= 0:
                problems.append(f"{where}: {cell.status} cells need throughput and iterations")
            elif cell.iterations is not None and cell.iterations < g.min_iterations:
                problems.append(f"{where}: fewer than min_iterations measured")
            oracle = cell.variant.startswith(ORACLE_VARIANT_PREFIX)
            diffs = (cell.relative_diff_output, cell.relative_diff_grad)
            if not oracle and cell.status == "ok" and any(d is None for d in diffs):
                if "reference unavailable" not in cell.detail:
                    problems.append(f"{where}: missing diffs must explain 'reference unavailable'")
            if not oracle and all(d is not None for d in diffs):
                within = _cell_equivalent(cell, g.relative_tolerance)
                if cell.status == "ok" and not within:
                    problems.append(f"{where}: diffs exceed tolerance but status is ok")
                if cell.status == "mismatch" and within:
                    problems.append(f"{where}: status mismatch but diffs are within tolerance")
        else:
            if not cell.detail.strip():
                problems.append(f"{where}: {cell.status} cells must explain themselves in detail")
            if cell.tokens_per_second is not None:
                problems.append(f"{where}: {cell.status} cells cannot report throughput")
    missing = sorted(expected - set(seen))
    extra = sorted(set(seen) - expected)
    duplicated = sorted(key for key, count in seen.items() if count > 1)
    if missing:
        problems.append(f"grid cells missing: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    if extra:
        problems.append(f"cells outside the grid: {extra[:5]}")
    if duplicated:
        problems.append(f"duplicated cells: {duplicated[:5]}")

    if not missing and not extra and not duplicated:
        recomputed = select_attention_path(g, r.cells)
        if recomputed is None:
            problems.append("no variant qualifies for selection; the benchmark cannot decide D-014")
        elif recomputed != r.selection:
            problems.append(
                f"recorded selection {r.selection.variant}/{r.selection.flag_setting} does not "
                f"match the recomputed selection {recomputed.variant}/{recomputed.flag_setting}"
            )

    if not r.limitations:
        problems.append("limitations must list at least one known limitation")
    plain = to_dict(r)
    for path, text in _iter_strings(plain, "benchmark"):
        if _ABSOLUTE_PATH.search(text):
            problems.append(f"{path} contains an absolute path")
        for ident in forbidden_identifiers:
            pattern = rf"(?<![A-Za-z0-9]){re.escape(ident)}(?![A-Za-z0-9])"
            if re.search(pattern, text, flags=re.IGNORECASE):
                problems.append(f"{path} contains a local username/hostname")
    findings = scan_text(render_benchmark(r))
    if findings:
        problems.append(
            f"rendered benchmark trips the secret scanner ({sorted({f.rule for f in findings})})"
        )
    if problems:
        raise LedgerError(problems)


def parse_benchmark(
    data: Mapping[str, Any], forbidden_identifiers: Sequence[str] = ()
) -> BenchmarkRecord:
    """Parse plain data into a validated BenchmarkRecord."""
    try:
        record = from_dict(BenchmarkRecord, data)
    except ConfigError as exc:
        raise LedgerError([str(exc)]) from exc
    validate_benchmark(record, forbidden_identifiers)
    return record


def render_benchmark(record: BenchmarkRecord) -> str:
    """Render a benchmark as deterministic YAML in schema field order."""
    return yaml.safe_dump(to_dict(record), sort_keys=False, allow_unicode=True, width=100)


def write_benchmark(record: BenchmarkRecord, repo_root: Path) -> Path:
    """Validate and atomically write ``experiments/<BENCH-ID>/benchmark.yaml``."""
    validate_benchmark(record, forbidden_identifiers=local_identifiers())
    target = repo_root / "experiments" / record.benchmark.id / "benchmark.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=".benchmark-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(render_benchmark(record))
        os.replace(tmp_name, target)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return target


def load_benchmark(path: Path) -> BenchmarkRecord:
    """Load and validate a benchmark; its directory name must equal the benchmark id."""
    try:
        data = load_yaml(path)
    except ConfigError as exc:
        raise LedgerError([f"{path.parent.name}/{path.name}: {exc}"]) from exc
    record = parse_benchmark(data)
    if path.parent.name != record.benchmark.id:
        raise LedgerError(
            [f"benchmark directory {path.parent.name!r} != benchmark.id {record.benchmark.id!r}"]
        )
    return record


def load_all_benchmarks(experiments_dir: Path) -> list[BenchmarkRecord]:
    """Load every ``<experiments_dir>/*/benchmark.yaml`` sorted by benchmark id."""
    if not experiments_dir.is_dir():
        return []
    records = [load_benchmark(p) for p in sorted(experiments_dir.glob("*/benchmark.yaml"))]
    return sorted(records, key=lambda rec: rec.benchmark.id)
