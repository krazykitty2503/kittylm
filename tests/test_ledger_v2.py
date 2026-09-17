"""Ledger schema v2: record kinds (formal/smoke) and the benchmark schema.

All numbers are synthetic fixtures, not measurements.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from kittylm.config import to_dict
from kittylm.ledger import (
    SMOKE_LIMITATION,
    BenchmarkCell,
    BenchmarkRecord,
    LedgerError,
    load_all_benchmarks,
    load_benchmark,
    parse_benchmark,
    parse_record,
    render_ablation_table,
    select_attention_path,
    write_benchmark,
)
from tests.test_ledger import COMMIT, mutate, valid_record

# --- record kinds ---------------------------------------------------------------------------------


def smoke_record() -> dict[str, Any]:
    data = mutate("experiment.id", "SMOKE-GPU-001")
    data["experiment"]["kind"] = "smoke"
    data["limitations"] = [SMOKE_LIMITATION, "tiny fixture"]
    return data


def test_smoke_record_is_valid_and_excluded_from_ablation_table() -> None:
    smoke = parse_record(smoke_record())
    formal = parse_record(valid_record())
    table = render_ablation_table([smoke, formal])
    assert "EXP-900" in table and "SMOKE-GPU-001" not in table
    assert "No formal experiment records yet" in render_ablation_table([smoke])


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        (lambda d: d.update(limitations=["tiny fixture"]), "SMOKE engineering-only limitation"),
        (lambda d: d["experiment"].update(id="EXP-005"), "SMOKE-<AREA>-###"),
        (lambda d: d["experiment"].update(id="SMOKE-gpu-1"), "SMOKE-<AREA>-###"),
        (lambda d: d["experiment"].update(kind="formal"), "EXP-###"),
        (lambda d: d["experiment"].update(kind="benchmark"), "expected one of"),
    ],
)
def test_smoke_rules(change: Any, problem: str) -> None:
    data = smoke_record()
    change(data)
    with pytest.raises(LedgerError, match=problem.replace("(", r"\(").replace("#", "#")):
        parse_record(data)


def test_formal_records_require_formal_ids() -> None:
    data = valid_record()
    data["experiment"]["id"] = "SMOKE-GPU-001"
    with pytest.raises(LedgerError, match="EXP-###"):
        parse_record(data)


# --- benchmark schema -----------------------------------------------------------------------------

VARIANTS = ["reference@fp32", "reference@bf16", "sdpa_math@bf16", "sdpa_flash@bf16"]
FLAGS = ["unset", "1"]
LENGTHS = [256, 1024]


def cell(variant: str, flag: str, length: int, **overrides: Any) -> dict[str, Any]:
    speed = {"reference@fp32": 100.0, "reference@bf16": 150.0, "sdpa_math@bf16": 200.0}
    base: dict[str, Any] = {
        "variant": variant,
        "flag_setting": flag,
        "seq_len": length,
        "batch_size": 16384 // length,
        "status": "ok",
        "iterations": 5,
        "tokens_per_second": speed.get(variant, 1.0)
        * (2 if flag == "1" and variant == "sdpa_math@bf16" else 1),
        "first_call_seconds": 0.5,
        "peak_vram_mib": 100.0,
        "allocated_vram_mib": 50.0,
        "max_abs_diff_output": None,
        "max_abs_diff_grad": None,
        "relative_diff_output": None,
        "relative_diff_grad": None,
        "detail": "",
    }
    if not variant.startswith("reference@"):
        base.update(
            max_abs_diff_output=1e-4,
            max_abs_diff_grad=1e-4,
            relative_diff_output=1e-5,
            relative_diff_grad=1e-5,
        )
    if variant == "sdpa_flash@bf16":
        base.update(
            status="unsupported",
            iterations=None,
            tokens_per_second=None,
            first_call_seconds=None,
            peak_vram_mib=None,
            allocated_vram_mib=None,
            max_abs_diff_output=None,
            max_abs_diff_grad=None,
            relative_diff_output=None,
            relative_diff_grad=None,
            detail="AttentionBackendUnavailable: No available kernel.",
        )
    base.update(overrides)
    return base


def benchmark_data() -> dict[str, Any]:
    cells = [cell(v, f, n) for f in FLAGS for v in VARIANTS for n in LENGTHS]
    data: dict[str, Any] = {
        "schema_version": 2,
        "benchmark": {"id": "BENCH-ATTN-900", "name": "fixture", "description": "fixture"},
        "repository": {"git_commit": COMMIT, "git_dirty": False},
        "environment": {
            "python": "3.12.10",
            "pytorch": "2.13.0",
            "backend": "rocm",
            "backend_version": "7.15",
            "device": "test-gpu",
            "os": "test-os",
        },
        "grid": {
            "variants": VARIANTS,
            "flag_settings": FLAGS,
            "sequence_lengths": LENGTHS,
            "tokens_per_batch": 16384,
            "n_heads": 6,
            "head_dim": 64,
            "warmup_iterations": 2,
            "min_iterations": 3,
            "max_iterations": 20,
            "min_measure_seconds": 1.0,
            "relative_tolerance": 0.01,
            "selection_lengths": LENGTHS,
        },
        "cells": cells,
        "selection": {},
        "limitations": ["fixture"],
        "notes": "",
    }
    return with_recomputed_selection(data)


def with_recomputed_selection(data: dict[str, Any]) -> dict[str, Any]:
    from kittylm.config import from_dict
    from kittylm.ledger import BenchmarkGrid

    grid = from_dict(BenchmarkGrid, data["grid"])
    cells = [from_dict(BenchmarkCell, c) for c in data["cells"]]
    selection = select_attention_path(grid, cells)
    data["selection"] = (
        to_dict(selection)
        if selection is not None
        else {
            "decision": "D-014",
            "variant": "none",
            "flag_setting": "none",
            "rule": "none",
            "tokens_per_second_by_length": {},
        }
    )
    return data


def test_valid_benchmark_and_selection_rule() -> None:
    record = parse_benchmark(benchmark_data())
    # sdpa_math@bf16 with flag=1 is fastest (400 tok/s) and equivalent at every selection length.
    assert (record.selection.variant, record.selection.flag_setting) == ("sdpa_math@bf16", "1")
    assert record.selection.tokens_per_second_by_length == {"256": 400.0, "1024": 400.0}


def test_selection_skips_mismatching_and_non_bf16_variants() -> None:
    data = benchmark_data()
    for c in data["cells"]:
        if c["variant"] == "sdpa_math@bf16" and c["flag_setting"] == "1" and c["seq_len"] == 1024:
            c.update(status="mismatch", relative_diff_output=0.5)
    data = with_recomputed_selection(data)
    record = parse_benchmark(data)
    assert (record.selection.variant, record.selection.flag_setting) == ("sdpa_math@bf16", "unset")

    for c in data["cells"]:
        if c["variant"] == "sdpa_math@bf16":
            c.update(status="mismatch", relative_diff_output=0.5)
    data = with_recomputed_selection(data)
    # reference@bf16 (the oracle) remains; reference@fp32 is never selected (not bf16).
    assert parse_benchmark(data).selection.variant == "reference@bf16"


def test_ties_prefer_unset_flag() -> None:
    data = benchmark_data()
    for c in data["cells"]:
        if c["variant"] == "sdpa_math@bf16":
            c["tokens_per_second"] = 300.0
    record = parse_benchmark(with_recomputed_selection(data))
    assert record.selection.flag_setting == "unset"


def edit_cell(data: dict[str, Any], variant: str, flag: str, length: int, **changes: Any) -> None:
    for c in data["cells"]:
        if (c["variant"], c["flag_setting"], c["seq_len"]) == (variant, flag, length):
            c.update(changes)


@pytest.mark.parametrize(
    ("mutation", "problem"),
    [
        (lambda d: d["cells"].pop(), "grid cells missing"),
        (lambda d: d["cells"].append(copy.deepcopy(d["cells"][0])), "duplicated cells"),
        (lambda d: d["cells"].append(cell("sdpa_cudnn@bf16", "unset", 256)), "outside the grid"),
        (lambda d: d["selection"].update(flag_setting="unset"), "does not match the recomputed"),
        (lambda d: d["benchmark"].update(id="BENCH-1"), "BENCH-<AREA>-###"),
        (lambda d: d.update(limitations=[]), "at least one"),
        (lambda d: d.update(notes="from C:\\Users\\someone"), "absolute path"),
        (lambda d: d["grid"].update(selection_lengths=[4096]), "subset of sequence_lengths"),
        (
            lambda d: edit_cell(d, "sdpa_flash@bf16", "unset", 256, detail=""),
            "must explain themselves",
        ),
        (
            lambda d: edit_cell(d, "sdpa_flash@bf16", "1", 256, tokens_per_second=5.0),
            "cannot report throughput",
        ),
        (
            lambda d: edit_cell(d, "sdpa_math@bf16", "unset", 256, relative_diff_grad=0.9),
            "exceed tolerance but status is ok",
        ),
        (
            lambda d: edit_cell(d, "sdpa_math@bf16", "unset", 256, status="mismatch"),
            "within tolerance",
        ),
        (
            lambda d: edit_cell(
                d,
                "sdpa_math@bf16",
                "unset",
                256,
                relative_diff_output=None,
                relative_diff_grad=None,
            ),
            "reference unavailable",
        ),
        (
            lambda d: edit_cell(d, "reference@bf16", "unset", 256, iterations=1),
            "fewer than min_iterations",
        ),
        (lambda d: edit_cell(d, "reference@bf16", "unset", 256, status="done"), "expected one of"),
    ],
)
def test_invalid_benchmarks_are_rejected(mutation: Any, problem: str) -> None:
    data = benchmark_data()
    mutation(data)
    with pytest.raises(LedgerError) as exc:
        parse_benchmark(data)
    assert problem in str(exc.value)


def test_no_qualifying_variant_is_an_error() -> None:
    data = benchmark_data()
    for c in data["cells"]:
        if c["variant"] != "sdpa_flash@bf16" and c["variant"] != "reference@fp32":
            c.update(
                status="oom",
                iterations=None,
                tokens_per_second=None,
                peak_vram_mib=None,
                allocated_vram_mib=None,
                relative_diff_output=None,
                relative_diff_grad=None,
                max_abs_diff_output=None,
                max_abs_diff_grad=None,
                detail="OutOfMemoryError: fixture",
            )
    with pytest.raises(LedgerError, match="no variant qualifies"):
        parse_benchmark(with_recomputed_selection(data))


def test_write_and_load_benchmark(tmp_path: Path) -> None:
    record = parse_benchmark(benchmark_data())
    path = write_benchmark(record, tmp_path)
    assert path == tmp_path / "experiments" / "BENCH-ATTN-900" / "benchmark.yaml"
    assert load_benchmark(path) == record
    assert load_all_benchmarks(tmp_path / "experiments") == [record]
    moved = tmp_path / "BENCH-ATTN-901" / "benchmark.yaml"
    moved.parent.mkdir()
    moved.write_bytes(path.read_bytes())
    with pytest.raises(LedgerError, match="!= benchmark.id"):
        load_benchmark(moved)


def test_benchmark_records_are_frozen_dataclasses() -> None:
    record = parse_benchmark(benchmark_data())
    assert isinstance(record, BenchmarkRecord)
    with pytest.raises(AttributeError):
        record.notes = "x"  # type: ignore[misc]
    assert replace(record, notes="changed").notes == "changed"
