"""BENCH-ATTN-001: benchmark every attention path at several sequence lengths.

For each variant ("<path>@<dtype>") and each setting of TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL
("unset" / "1"), a fresh worker process measures forward+backward throughput, first-call time,
peak/allocated VRAM and numerical agreement with the bf16 reference implementation, and records
every unsupported, out-of-memory, erroring, mismatching or non-finite cell. The flag is read at
process start, so each setting needs its own process. Device-level errors can poison a process,
so after one the parent restarts a fresh worker for the remaining lengths.

The attention computation is `kittylm.model.attention.attend`, the same function the model uses.

Usage:
    python scripts/bench_attention.py                      # ROCm/CUDA, writes BENCH-ATTN-001
    python scripts/bench_attention.py --device cpu --lengths 16 32 --tokens-per-batch 64 \
        --output-root <dir>                                # mechanics check (used by tests)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FLAG = "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"
VARIANTS = (
    "reference@fp32",
    "reference@bf16",
    "sdpa_math@bf16",
    "sdpa_efficient@bf16",
    "sdpa_flash@bf16",
    "sdpa_cudnn@bf16",
)
FLAG_SETTINGS = ("unset", "1")
DEFAULT_LENGTHS = (256, 512, 1024, 2048, 4096, 8192)
SELECTION_LENGTHS = (256, 1024)  # nano and tiny context lengths (plan rev 3.3 section 3)
N_HEADS = 6
HEAD_DIM = 64
WARMUP = 2
MIN_ITERS = 3
MAX_ITERS = 20
MIN_SECONDS = 1.0
TOLERANCE = 1e-2
DEVICE_ERROR_EXIT = 3
_PATHLIKE = re.compile(r"([A-Za-z]:[\\/]|/)[^\s'\"]*")


def short_detail(exc: BaseException) -> str:
    """First line of an exception without file paths (records must be path-free)."""
    first = (str(exc).strip().splitlines() or [""])[0]
    return f"{type(exc).__name__}: {_PATHLIKE.sub('<path>', first)}"[:240]


# ------------------------------------------------------------------------------------ worker


def worker(args: argparse.Namespace) -> int:
    import torch

    from kittylm.model.attention import AttentionBackendUnavailable, attend

    device = torch.device(args.device)
    path, dtype_name = args.variant.split("@")
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype_name]
    on_gpu = device.type == "cuda"

    def sync() -> None:
        if on_gpu:
            torch.cuda.synchronize()

    def fwd_bwd(
        name: str, q0: torch.Tensor, k0: torch.Tensor, v0: torch.Tensor, g0: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        leaves = [t.to(device=device, dtype=dtype).requires_grad_() for t in (q0, k0, v0)]
        out = attend(leaves[0], leaves[1], leaves[2], name)
        out.backward(g0.to(device=device, dtype=dtype))
        sync()
        grads = [leaf.grad for leaf in leaves]
        assert all(grad is not None for grad in grads)
        return out.detach(), [grad for grad in grads if grad is not None]

    def to_cpu(t: torch.Tensor) -> torch.Tensor:
        return t.detach().to("cpu", torch.float32)

    out_file = Path(args.json_out)
    for seq_len in args.lengths:
        batch = max(1, args.tokens_per_batch // seq_len)
        cell: dict[str, Any] = {
            "variant": args.variant,
            "flag_setting": args.flag_setting,
            "seq_len": seq_len,
            "batch_size": batch,
            "status": "ok",
            "iterations": None,
            "tokens_per_second": None,
            "first_call_seconds": None,
            "peak_vram_mib": None,
            "allocated_vram_mib": None,
            "max_abs_diff_output": None,
            "max_abs_diff_grad": None,
            "relative_diff_output": None,
            "relative_diff_grad": None,
            "detail": "",
        }
        device_error = False
        try:
            gen = torch.Generator().manual_seed(1000 + seq_len)
            shape = (batch, N_HEADS, seq_len, HEAD_DIM)
            q0, k0, v0, g0 = (torch.randn(shape, generator=gen) for _ in range(4))

            reference: tuple[torch.Tensor, list[torch.Tensor]] | None = None
            if path != "reference":
                try:
                    ref_out, ref_grads = fwd_bwd("reference", q0, k0, v0, g0)
                    reference = (to_cpu(ref_out), [to_cpu(g) for g in ref_grads])
                    del ref_out, ref_grads
                except torch.OutOfMemoryError:
                    cell["detail"] = "reference unavailable (out of memory); diffs not measured"
                if on_gpu:
                    torch.cuda.empty_cache()

            start = time.perf_counter()
            out, grads = fwd_bwd(path, q0, k0, v0, g0)
            cell["first_call_seconds"] = round(time.perf_counter() - start, 4)
            finite = bool(torch.isfinite(out).all()) and all(
                bool(torch.isfinite(g).all()) for g in grads
            )
            if reference is not None:
                ref_out, ref_grads = reference
                out_c = to_cpu(out)
                grads_c = [to_cpu(g) for g in grads]
                diff_out = float((out_c - ref_out).abs().max())
                diff_grad = max(
                    float((a - b).abs().max()) for a, b in zip(grads_c, ref_grads, strict=True)
                )
                scale_out = float(ref_out.abs().max()) + 1e-12
                scale_grad = max(float(g.abs().max()) for g in ref_grads) + 1e-12
                cell["max_abs_diff_output"] = diff_out
                cell["max_abs_diff_grad"] = diff_grad
                cell["relative_diff_output"] = diff_out / scale_out
                cell["relative_diff_grad"] = diff_grad / scale_grad
            del out, grads

            for _ in range(WARMUP):
                fwd_bwd(path, q0, k0, v0, g0)
            if on_gpu:
                torch.cuda.reset_peak_memory_stats(device)
            iterations = 0
            sync()
            start = time.perf_counter()
            while iterations < MAX_ITERS and (
                iterations < MIN_ITERS or time.perf_counter() - start < MIN_SECONDS
            ):
                fwd_bwd(path, q0, k0, v0, g0)
                iterations += 1
            elapsed = time.perf_counter() - start
            cell["iterations"] = iterations
            cell["tokens_per_second"] = round(batch * seq_len * iterations / elapsed, 2)
            if on_gpu:
                cell["peak_vram_mib"] = round(torch.cuda.max_memory_allocated(device) / 2**20, 1)
                cell["allocated_vram_mib"] = round(torch.cuda.memory_allocated(device) / 2**20, 1)

            if not finite:
                cell.update(status="nonfinite", tokens_per_second=None, detail="non-finite values")
            elif (
                cell["relative_diff_output"] is not None
                and max(cell["relative_diff_output"], cell["relative_diff_grad"]) > TOLERANCE
            ):
                cell["status"] = "mismatch"
        except AttentionBackendUnavailable as exc:
            cell.update(status="unsupported", detail=short_detail(exc))
        except torch.OutOfMemoryError as exc:
            cell.update(status="oom", detail=short_detail(exc))
        except Exception as exc:  # recorded, never hidden
            cell.update(status="error", detail=short_detail(exc))
            device_error = "CUDA error" in str(exc) or "HIP error" in str(exc)
        if cell["status"] in ("unsupported", "oom", "error", "nonfinite"):
            for key in ("iterations", "tokens_per_second", "peak_vram_mib", "allocated_vram_mib"):
                cell[key] = None
        with out_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(cell) + "\n")
        if on_gpu:
            torch.cuda.empty_cache()
        if device_error:
            return DEVICE_ERROR_EXIT  # the process may be poisoned; the parent restarts a worker
    return 0


# ------------------------------------------------------------------------------------ parent


def run_variant(
    variant: str, flag_setting: str, args: argparse.Namespace, scratch: Path
) -> list[dict[str, Any]]:
    """Measure one variant under one flag setting, restarting workers after device errors."""
    remaining = list(args.lengths)
    cells: list[dict[str, Any]] = []
    attempt = 0
    while remaining:
        attempt += 1
        json_out = scratch / f"{variant.replace('@', '_')}_{flag_setting}_{attempt}.jsonl"
        env = dict(os.environ)
        env.pop(FLAG, None)
        if flag_setting != "unset":
            env[FLAG] = flag_setting
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--device",
            args.device,
            "--variant",
            variant,
            "--flag-setting",
            flag_setting,
            "--tokens-per-batch",
            str(args.tokens_per_batch),
            "--json-out",
            str(json_out),
            "--lengths",
            *map(str, remaining),
        ]
        result = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, check=False)
        done = []
        if json_out.exists():
            done = [json.loads(line) for line in json_out.read_text(encoding="utf-8").splitlines()]
        cells.extend(done)
        finished = {cell["seq_len"] for cell in done}
        remaining = [n for n in remaining if n not in finished]
        if remaining and result.returncode not in (0, DEVICE_ERROR_EXIT):
            # The worker crashed without recording the current cell: record it as an error.
            lines = [ln for ln in (result.stderr or "").strip().splitlines() if ln.strip()]
            message = _PATHLIKE.sub("<path>", lines[-1] if lines else "no output")[:200]
            seq_len = remaining.pop(0)
            cells.append(
                {
                    "variant": variant,
                    "flag_setting": flag_setting,
                    "seq_len": seq_len,
                    "batch_size": max(1, args.tokens_per_batch // seq_len),
                    "status": "error",
                    "iterations": None,
                    "tokens_per_second": None,
                    "first_call_seconds": None,
                    "peak_vram_mib": None,
                    "allocated_vram_mib": None,
                    "max_abs_diff_output": None,
                    "max_abs_diff_grad": None,
                    "relative_diff_output": None,
                    "relative_diff_grad": None,
                    "detail": f"worker exited with code {result.returncode}: {message}",
                }
            )
        status = ", ".join(f"T={c['seq_len']}:{c['status']}" for c in done)
        print(f"  {variant:<22} flag={flag_setting:<5} attempt {attempt}: {status}", flush=True)
    return cells


def parent(args: argparse.Namespace) -> int:
    import torch

    from kittylm.ledger import (
        BenchmarkCell,
        BenchmarkEnvironment,
        BenchmarkGrid,
        BenchmarkInfo,
        BenchmarkRecord,
        BenchmarkSelection,
        LedgerError,
        RepositoryInfo,
        write_benchmark,
    )
    from kittylm.ledger import select_attention_path as select
    from kittylm.utils.git import head_commit, is_dirty

    on_gpu = args.device == "cuda"
    if on_gpu and not torch.cuda.is_available():
        print("no CUDA/ROCm device available")
        return 2
    backend = "cpu"
    backend_version = None
    device_name = platform.processor() or "cpu"
    if on_gpu:
        backend = "rocm" if torch.version.hip else "cuda"
        backend_version = str(torch.version.hip or torch.version.cuda)
        device_name = str(torch.cuda.get_device_name(0))

    selection_lengths = [n for n in SELECTION_LENGTHS if n in args.lengths] or [max(args.lengths)]
    grid = BenchmarkGrid(
        variants=list(VARIANTS),
        flag_settings=list(FLAG_SETTINGS),
        sequence_lengths=list(args.lengths),
        tokens_per_batch=args.tokens_per_batch,
        n_heads=N_HEADS,
        head_dim=HEAD_DIM,
        warmup_iterations=WARMUP,
        min_iterations=MIN_ITERS,
        max_iterations=MAX_ITERS,
        min_measure_seconds=MIN_SECONDS,
        relative_tolerance=TOLERANCE,
        selection_lengths=selection_lengths,
    )
    raw: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="bench-attn-") as scratch:
        for flag_setting in FLAG_SETTINGS:
            for variant in VARIANTS:
                raw.extend(run_variant(variant, flag_setting, args, Path(scratch)))
    cells = [BenchmarkCell(**cell) for cell in raw]
    selection = select(grid, cells)
    if selection is None:
        print("no variant qualifies; D-014 cannot be decided from this run")
        selection = BenchmarkSelection("D-014", "none", "none", "no qualifying variant", {})

    record = BenchmarkRecord(
        schema_version=2,
        benchmark=BenchmarkInfo(
            id=args.benchmark_id,
            name="attention-paths",
            description=(
                "Forward+backward throughput, VRAM and agreement with the bf16 reference for "
                "every attention path, sequence length and AOTriton experimental-flag setting."
            ),
        ),
        repository=RepositoryInfo(git_commit=head_commit(ROOT), git_dirty=is_dirty(ROOT)),
        environment=BenchmarkEnvironment(
            python=platform.python_version(),
            pytorch=str(torch.__version__),
            backend=backend,  # type: ignore[arg-type]
            backend_version=backend_version,
            device=device_name,
            os=platform.system() + " " + platform.release(),
        ),
        grid=grid,
        cells=cells,
        selection=selection,
        limitations=[
            "Attention kernel only (random Q/K/V, 6 heads, head_dim 64); not a full-model or "
            "end-to-end training measurement.",
            "Single machine and driver/software stack; results do not transfer to other GPUs "
            "or PyTorch/ROCm versions.",
            "Throughput is wall-clock over a short synchronized window after warmup; "
            "run-to-run variance is not measured.",
        ],
        notes=args.notes,
    )
    try:
        target = write_benchmark(record, args.output_root)
    except LedgerError as exc:
        print(exc)
        return 1
    shown = target.relative_to(args.output_root).as_posix()
    print(f"selection: {selection.variant} flag={selection.flag_setting}")
    print(f"wrote {shown}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--lengths", type=int, nargs="+", default=list(DEFAULT_LENGTHS))
    parser.add_argument("--tokens-per-batch", type=int, default=16384)
    parser.add_argument("--variant")
    parser.add_argument("--flag-setting", default="unset")
    parser.add_argument("--json-out")
    parser.add_argument("--benchmark-id", default="BENCH-ATTN-001")
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--notes", default="")
    args = parser.parse_args(argv)
    if args.worker:
        return worker(args)
    return parent(args)


if __name__ == "__main__":
    sys.exit(main())
