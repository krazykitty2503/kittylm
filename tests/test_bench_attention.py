"""BENCH-ATTN-001 script mechanics on CPU (the real benchmark runs on the ROCm GPU)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from kittylm.ledger import load_benchmark
from tests.conftest import ROOT, load_script


def test_short_detail_removes_paths() -> None:
    bench = load_script("bench_attention")
    detail = bench.short_detail(RuntimeError("failed in C:\\Users\\someone\\x.py and /home/a/b.py"))
    assert detail.startswith("RuntimeError: failed in <path>")
    assert "someone" not in detail and "/home" not in detail


def test_cpu_benchmark_mechanics(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/bench_attention.py",
            "--device",
            "cpu",
            "--lengths",
            "16",
            "32",
            "--tokens-per-batch",
            "64",
            "--benchmark-id",
            "BENCH-ATTN-999",
            "--output-root",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=900,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    record = load_benchmark(tmp_path / "experiments" / "BENCH-ATTN-999" / "benchmark.yaml")
    assert len(record.cells) == 6 * 2 * 2  # every variant x flag setting x length
    status = {(c.variant, c.flag_setting, c.seq_len): c.status for c in record.cells}
    for flag in ("unset", "1"):
        for length in (16, 32):
            assert status[("reference@bf16", flag, length)] == "ok"
            assert status[("sdpa_math@bf16", flag, length)] == "ok"
            # CPU builds have no memory-efficient or cuDNN kernel: recorded, not hidden.
            assert status[("sdpa_efficient@bf16", flag, length)] == "unsupported"
            assert status[("sdpa_cudnn@bf16", flag, length)] == "unsupported"
    assert record.environment.backend == "cpu"
    assert record.selection.decision == "D-014"
    assert str(tmp_path) not in result.stdout
