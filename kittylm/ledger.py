"""Experiment ledger: the schema, validator and writer for ``experiments/<ID>/record.yaml``.

Purpose:
    Make every reported number traceable to one validated record that pins the code
    (git commit), data (dataset version), tokenizer, model shape and parameter budget, and
    training setup. The ablation table is *generated* from records, so a result that is not
    in a record cannot appear in the repository (CONTRIBUTING.md).

Public API:
    ExperimentRecord (and its section dataclasses)
        Typed schema; ``schema_version`` identifies the format.
    LedgerError
        Raised with every problem found, not just the first.
    parse_record(data, forbidden_identifiers=()) -> ExperimentRecord
        Strict structural parse followed by semantic validation.
    validate_record(record, forbidden_identifiers=()) -> None
    render_record(record) -> str
        Deterministic YAML (schema field order, no timestamps).
    write_record(record, repo_root) -> Path
        Validate with local user/host identifiers forbidden, secret-scan, write atomically.
    load_record(path) / load_all_records(experiments_dir)
    render_ablation_table(records) -> str

Invariants:
    - Parameter accounting parts sum exactly to ``total``; ``non_embedding = total - embedding``.
    - ``final_ppl == exp(final_loss)`` within a relative tolerance of 1e-3 (natural-log loss).
    - ``checkpoint_resume_test`` is ``passed``/``failed`` only with evidence produced by the
      resume harness (RESUME_HARNESS); ``not_run`` must carry no evidence.
    - Records contain no absolute paths, no local username/hostname, and no secrets.
    - ``render_ablation_table`` output depends only on record contents.

Failure modes:
    - Any schema or semantic violation raises LedgerError listing all problems.
    - Identifier checks against the local username/hostname only apply at write time; CI
      re-validation cannot know the author's machine and checks paths and secrets instead.

See:
    Plan rev 3.1 section 6, D-012 (resume evidence), SECURITY.md.
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
    "RESUME_HARNESS",
    "SCHEMA_VERSION",
    "ExperimentRecord",
    "LedgerError",
    "load_all_records",
    "load_record",
    "local_identifiers",
    "parse_record",
    "render_ablation_table",
    "render_record",
    "validate_record",
    "write_record",
]

SCHEMA_VERSION = 1
RESUME_HARNESS = "kittylm.training.resume_harness"
PPL_REL_TOL = 1e-3
RATIO_REL_TOL = 1e-3

_EXPERIMENT_ID = re.compile(r"^EXP-\d{3}(?:-[a-z0-9]+)*$")
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
    if not _EXPERIMENT_ID.match(r.experiment.id):
        problems.append(f"experiment.id {r.experiment.id!r} must match EXP-###[-suffix]")
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
    """Render the ablation table markdown from records (deterministic)."""
    lines = [_TABLE_HEADER]
    if not records:
        lines.append("_No experiment records yet._\n")
        return "\n".join(lines)
    lines.append("| " + " | ".join(_COLUMNS) + " |")
    lines.append("|" + "|".join("---" for _ in _COLUMNS) + "|")
    for rec in sorted(records, key=lambda x: x.experiment.id):
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
