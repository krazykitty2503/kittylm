"""GPU checks for Milestone D (run locally with `pytest -m gpu`; never in CI).

Generation with the KV cache against full-forward decoding on ROCm, the overfit gate on a GPU
model, CPU/GPU evaluation agreement, and batch-1 inference speed for Nano in the D-014 precision.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
import torch
from torch import nn

from kittylm.evaluation.gates import evaluate_overfit_gate
from kittylm.evaluation.inference_speed import measure_inference_speed
from kittylm.evaluation.perplexity import stream_nll
from kittylm.evaluation.windows import context_start
from kittylm.inference.generate import SamplingConfig, generate
from kittylm.model.transformer import KittyLM
from tests.test_model_gpu import model_config, require_gpu, selected

pytestmark = pytest.mark.gpu


class Recording(nn.Module):
    def __init__(self, model: KittyLM) -> None:
        super().__init__()
        self.model = model
        self.config = model.config
        self.last_logits: list[torch.Tensor] = []

    def forward(self, input_ids: torch.Tensor, cache: Any = None) -> torch.Tensor:
        logits: torch.Tensor = self.model(input_ids, cache=cache)
        self.last_logits.append(logits[0, -1].detach().float().cpu())
        return logits


def nano_small_vocab() -> Any:
    return replace(model_config("nano"), vocab_size=512)


def cached_vs_full(
    dtype: torch.dtype | None, sampling: SamplingConfig
) -> tuple[Any, Any, Any, Any]:
    device = require_gpu()
    torch.manual_seed(7)
    model = KittyLM(nano_small_vocab()).to(device)
    prompt = torch.randint(0, 512, (40,)).tolist()
    out = []
    for use_cache in (True, False):
        recorder = Recording(model)
        result = generate(
            recorder,
            prompt,
            sampling,
            device=device,
            autocast_dtype=dtype,
            generator=torch.Generator().manual_seed(5),
            use_cache=use_cache,
        )
        out += [result, torch.stack(recorder.last_logits)]
    return out[0], out[1], out[2], out[3]


def test_fp32_kv_cache_generation_matches_full_forward_on_gpu() -> None:
    sampling = SamplingConfig(max_new_tokens=400)  # 40 + 400 > 256: crosses context resets
    cached, cached_logits, full, full_logits = cached_vs_full(None, sampling)
    assert cached.context_resets == full.context_resets > 0
    assert cached.new_ids == full.new_ids
    torch.testing.assert_close(cached_logits, full_logits, rtol=1e-4, atol=1e-4)


def test_bf16_greedy_kv_cache_generation_matches_full_forward_on_gpu() -> None:
    selected()  # the D-014 attention path and flag
    cached, cached_logits, full, full_logits = cached_vs_full(
        torch.bfloat16, SamplingConfig(max_new_tokens=400)
    )
    deviation = float((cached_logits - full_logits).abs().max())
    print(f"bf16 greedy cached vs full-forward max |logit| deviation: {deviation:.3e}")
    assert cached.new_ids == full.new_ids and cached.context_resets > 0
    assert deviation < 5e-2


def test_bf16_sampled_kv_cache_logits_match_full_forward_on_the_same_tokens() -> None:
    # Independently sampled bf16 runs can diverge: bf16 rounding (~1/128 in logits at this
    # scale) can flip a random draw near a probability boundary, after which the contexts
    # differ. Cache equivalence is therefore checked teacher-forced: the full-forward path is
    # fed exactly the contexts the cached run used, and the logit deviation is measured.
    device = require_gpu()
    selected()
    torch.manual_seed(7)
    model = KittyLM(nano_small_vocab()).to(device)
    prompt = torch.randint(0, 512, (40,)).tolist()
    recorder = Recording(model)
    sampling = SamplingConfig(max_new_tokens=400, temperature=0.8, top_k=50, top_p=0.95)
    result = generate(
        recorder,
        prompt,
        sampling,
        device=device,
        autocast_dtype=torch.bfloat16,
        generator=torch.Generator().manual_seed(5),
    )
    assert result.context_resets > 0 and len(result.new_ids) == 400
    ids = list(result.ids)
    ctx = model.config.context_length
    deviations = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for step, cached in enumerate(recorder.last_logits):
            length = len(prompt) + step
            window = ids[context_start(length, ctx) : length]
            full = model(torch.tensor([window], device=device))[0, -1].float().cpu()
            deviations.append(float((cached - full).abs().max()))
    print(f"bf16 teacher-forced cached vs full max |logit| deviation: {max(deviations):.3e}")
    assert max(deviations) < 5e-2


def test_gpu_evaluation_matches_cpu() -> None:
    device = require_gpu()
    torch.manual_seed(3)
    cpu_model = KittyLM(nano_small_vocab())
    gpu_model = KittyLM(nano_small_vocab()).to(device)
    gpu_model.load_state_dict(cpu_model.state_dict())
    ids = torch.randint(0, 512, (477,)).tolist()
    ones = [1] * len(ids)
    cpu = stream_nll(cpu_model, ids, ones, device=torch.device("cpu"))
    gpu = stream_nll(gpu_model, ids, ones, device=device)
    assert (cpu.tokens, cpu.bytes) == (gpu.tokens, gpu.bytes) == (476, 476)
    assert gpu.nats == pytest.approx(cpu.nats, rel=1e-4)


def test_overfit_gate_runs_on_gpu() -> None:
    device = require_gpu()
    torch.manual_seed(0)
    model = KittyLM(nano_small_vocab()).to(device)
    ids = torch.randint(0, 512, (477,)).tolist()
    report = evaluate_overfit_gate(
        model, ids, [1] * len(ids), device=device, autocast_dtype=torch.bfloat16
    )
    assert report.scored_tokens == 476 and report.generated_tokens == 461
    assert not report.passed and report.checks["all_values_finite"]


def test_nano_inference_speed_bf16() -> None:
    device = require_gpu()
    selected()
    model = KittyLM(nano_small_vocab()).to(device)
    result = measure_inference_speed(
        model,
        prompt_tokens=128,
        new_tokens=128,
        device=device,
        autocast_dtype=torch.bfloat16,
        warmup=2,
        repeats=5,
    )
    print(
        f"nano bf16 batch-1: prefill {result.prefill_tok_s:,.0f} tok/s, "
        f"decode {result.decode_tok_s:,.0f} tok/s"
    )
    assert result.prefill_tok_s > 0 and result.decode_tok_s > 0
