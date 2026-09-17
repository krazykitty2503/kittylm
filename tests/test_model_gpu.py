"""GPU checks for Milestone B (run locally with `pytest -m gpu`; never in CI).

The attention path under test is read from the committed BENCH-ATTN-001 selection (D-014), so
these tests always check the path the model configurations actually use.
"""

from __future__ import annotations

import os
from dataclasses import replace

import pytest
import torch

from kittylm.config import from_dict, load_yaml
from kittylm.ledger import load_benchmark
from kittylm.model.accounting import count_parameters
from kittylm.model.attention import attend
from kittylm.model.config import ModelConfig
from kittylm.model.kv_cache import KVCache
from kittylm.model.transformer import KittyLM
from tests.conftest import ROOT

pytestmark = pytest.mark.gpu

FLAG = "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"
BENCHMARK = ROOT / "experiments" / "BENCH-ATTN-001" / "benchmark.yaml"


def require_gpu() -> torch.device:
    if not torch.cuda.is_available():
        pytest.fail("GPU tests were requested (-m gpu) but no CUDA/ROCm device is available")
    return torch.device("cuda")


def selected() -> tuple[str, float, list[int]]:
    record = load_benchmark(BENCHMARK)
    selection = record.selection
    wanted_flag = selection.flag_setting
    actual_flag = os.environ.get(FLAG, "unset")
    if wanted_flag != actual_flag:
        pytest.fail(
            f"D-014 selected {selection.variant} with {FLAG}={wanted_flag}, but this process has "
            f"{FLAG}={actual_flag}; rerun pytest with the selected setting"
        )
    path, dtype = selection.variant.split("@")
    assert dtype == "bf16"
    return path, record.grid.relative_tolerance, record.grid.selection_lengths


def model_config(name: str) -> ModelConfig:
    data = load_yaml(ROOT / "configs" / "model" / f"{name}.yaml")
    return from_dict(ModelConfig, {k: v for k, v in data.items() if k != "kind"})


def relative_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    a32, b32 = a.float(), b.float()
    return float((a32 - b32).abs().max()) / (float(b32.abs().max()) + 1e-12)


def test_configs_use_the_d014_selection() -> None:
    path, _, _ = selected()
    for name in ("nano", "tiny"):
        assert model_config(name).attention_backend == path


@pytest.mark.parametrize("which", ["selected", "sdpa_math"])
def test_working_paths_match_reference_forward_and_backward(which: str) -> None:
    # D-014 selected the reference itself, so the selected-path check is trivially an identity;
    # sdpa_math is the only other working kernel on this stack and must also stay equivalent.
    device = require_gpu()
    selected_path, tolerance, lengths = selected()
    path = selected_path if which == "selected" else "sdpa_math"
    for length in lengths:
        gen = torch.Generator().manual_seed(length)
        base = [torch.randn(2, 6, length, 64, generator=gen) for _ in range(4)]
        results = []
        for name in ("reference", path):
            q, k, v = (t.to(device, torch.bfloat16).requires_grad_() for t in base[:3])
            out = attend(q, k, v, name)
            out.backward(base[3].to(device, torch.bfloat16))
            grads = [t.grad for t in (q, k, v)]
            assert all(g is not None for g in grads)
            results.append((out.detach(), [g for g in grads if g is not None]))
        (ref_out, ref_grads), (out, grads) = results
        assert relative_diff(out, ref_out) <= tolerance
        for grad, ref_grad in zip(grads, ref_grads, strict=True):
            assert relative_diff(grad, ref_grad) <= tolerance


def test_kv_cache_decoding_matches_full_forward_on_gpu() -> None:
    device = require_gpu()
    path, tolerance, _ = selected()
    config = replace(model_config("nano"), vocab_size=512)
    torch.manual_seed(0)
    model = KittyLM(config).to(device).eval()
    ids = torch.randint(0, 512, (2, 64), device=device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        full = model(ids)
        cache = KVCache.for_config(config, 2, torch.bfloat16, device)
        steps = [model(ids[:, :32], cache=cache)]
        for t in range(32, 64):
            steps.append(model(ids[:, t : t + 1], cache=cache))
    assert config.attention_backend == path
    assert relative_diff(torch.cat(steps, dim=1), full) <= tolerance


def test_tiny_forward_backward_on_gpu_is_finite() -> None:
    device = require_gpu()
    config = model_config("tiny")
    torch.manual_seed(0)
    model = KittyLM(config).to(device)
    assert count_parameters(model).total == 16_913_280
    ids = torch.randint(0, config.vocab_size, (2, config.context_length), device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(ids)
        loss = torch.nn.functional.cross_entropy(
            logits.float().flatten(0, 1), ids.roll(-1, dims=1).flatten()
        )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
