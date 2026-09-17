"""Batch-1 prefill/decode throughput measurement."""

from __future__ import annotations

import pytest
import torch

from kittylm.evaluation.inference_speed import measure_inference_speed
from kittylm.ledger import InferenceSpeed
from kittylm.model.transformer import KittyLM
from tests.training_helpers import TINY_MODEL

CPU = torch.device("cpu")


def test_measures_positive_throughput_and_converts_to_ledger() -> None:
    model = KittyLM(TINY_MODEL).train()
    result = measure_inference_speed(
        model, prompt_tokens=8, new_tokens=6, device=CPU, warmup=1, repeats=3
    )
    assert (result.prompt_tokens, result.new_tokens, result.repeats) == (8, 6, 3)
    assert len(result.prefill_seconds) == len(result.decode_seconds) == 3  # warmup excluded
    assert result.prefill_tok_s > 0 and result.decode_tok_s > 0
    assert result.to_ledger() == InferenceSpeed(result.prefill_tok_s, result.decode_tok_s)
    assert model.training  # mode restored


def test_bf16_autocast_measurement_on_cpu() -> None:
    result = measure_inference_speed(
        KittyLM(TINY_MODEL),
        prompt_tokens=4,
        new_tokens=4,
        device=CPU,
        autocast_dtype=torch.bfloat16,
        warmup=0,
        repeats=1,
    )
    assert result.decode_tok_s > 0


def test_decode_uses_the_cache_one_token_per_step(monkeypatch: pytest.MonkeyPatch) -> None:
    model = KittyLM(TINY_MODEL)
    lengths: list[int] = []
    original = model.forward

    def spy(input_ids: torch.Tensor, cache: object = None) -> torch.Tensor:
        lengths.append(input_ids.shape[1])
        assert cache is not None
        return original(input_ids, cache=cache)  # type: ignore[arg-type]

    monkeypatch.setattr(model, "forward", spy)
    measure_inference_speed(model, prompt_tokens=5, new_tokens=3, device=CPU, warmup=0, repeats=2)
    assert lengths == [5, 1, 1, 1] * 2


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"prompt_tokens": 0, "new_tokens": 4}, "must be >= 1"),
        ({"prompt_tokens": 4, "new_tokens": 0}, "must be >= 1"),
        ({"prompt_tokens": 10, "new_tokens": 7}, "exceeds context_length"),
        ({"prompt_tokens": 4, "new_tokens": 4, "repeats": 0}, "repeats"),
        ({"prompt_tokens": 4, "new_tokens": 4, "warmup": -1}, "warmup"),
    ],
)
def test_invalid_arguments(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        measure_inference_speed(KittyLM(TINY_MODEL), device=CPU, **kwargs)  # type: ignore[arg-type]
