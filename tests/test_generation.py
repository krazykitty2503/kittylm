"""Generation: sampling filters, greedy/temperature/top-k/top-p, EOT stop, KV-cache equivalence."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import replace
from typing import Any

import pytest
import torch
from torch import nn

from kittylm.evaluation.windows import context_start
from kittylm.inference.generate import (
    SamplingConfig,
    choose_next_token,
    generate,
    top_k_filter,
    top_p_filter,
)
from kittylm.model.transformer import KittyLM
from tests.training_helpers import TINY_MODEL

CPU = torch.device("cpu")
NEG_INF = float("-inf")


# --- filters --------------------------------------------------------------------------------------


def test_top_k_keeps_the_k_largest_and_ties() -> None:
    logits = torch.tensor([1.0, 3.0, 2.0, 0.5])
    assert top_k_filter(logits, 2).tolist() == [NEG_INF, 3.0, 2.0, NEG_INF]
    assert top_k_filter(logits, 0).tolist() == logits.tolist()  # disabled
    assert top_k_filter(logits, 10).tolist() == logits.tolist()  # k >= vocab: no-op
    ties = torch.tensor([3.0, 1.0, 3.0, 3.0])
    assert top_k_filter(ties, 2).tolist() == [3.0, NEG_INF, 3.0, 3.0]  # ties with the k-th kept


def logits_for(probs: list[float]) -> torch.Tensor:
    return torch.log(torch.tensor(probs))


@pytest.mark.parametrize(
    ("p", "kept"),
    [
        (0.5, [0]),  # the top token alone reaches 0.5
        (0.8, [0, 1]),  # 0.5 + 0.3 reaches 0.8
        (0.81, [0, 1, 2]),
        (0.96, [0, 1, 2, 3]),
        (1.0, [0, 1, 2, 3]),  # disabled
        (0.01, [0]),  # the most likely token is always kept
    ],
)
def test_top_p_keeps_the_smallest_prefix_reaching_p(p: float, kept: list[int]) -> None:
    shuffled = [0.15, 0.5, 0.05, 0.3]  # token order must not matter
    order = [1, 3, 0, 2]  # tokens by descending probability
    filtered = top_p_filter(logits_for(shuffled), p)
    kept_ids = [token for token in order if filtered[token] != NEG_INF]
    assert kept_ids == [order[i] for i in kept]


def test_greedy_is_argmax_with_lowest_id_on_ties() -> None:
    config = SamplingConfig(max_new_tokens=1)
    assert choose_next_token(torch.tensor([0.1, 2.0, 2.0, -1.0]), config) == 1


def sample_counts(logits: torch.Tensor, config: SamplingConfig, draws: int = 4000) -> Counter[int]:
    generator = torch.Generator().manual_seed(1234)
    return Counter(choose_next_token(logits, config, generator) for _ in range(draws))


def test_sampling_respects_filters_and_temperature() -> None:
    probs = [0.4, 0.3, 0.2, 0.1]
    logits = logits_for(probs)
    plain = sample_counts(logits, SamplingConfig(max_new_tokens=1, temperature=1.0))
    for token, p in enumerate(probs):
        assert plain[token] / 4000 == pytest.approx(p, abs=0.03)
    top_k = sample_counts(logits, SamplingConfig(max_new_tokens=1, temperature=1.0, top_k=2))
    assert set(top_k) == {0, 1}
    assert top_k[0] / 4000 == pytest.approx(0.4 / 0.7, abs=0.03)  # renormalized
    top_p = sample_counts(logits, SamplingConfig(max_new_tokens=1, temperature=1.0, top_p=0.6))
    assert set(top_p) == {0, 1}
    cold = sample_counts(logits, SamplingConfig(max_new_tokens=1, temperature=0.05))
    assert cold[0] / 4000 > 0.99
    hot = sample_counts(logits, SamplingConfig(max_new_tokens=1, temperature=50.0))
    assert all(hot[t] / 4000 == pytest.approx(0.25, abs=0.04) for t in range(4))


def test_seeded_sampling_is_reproducible() -> None:
    logits = logits_for([0.25, 0.25, 0.25, 0.25])
    config = SamplingConfig(max_new_tokens=1, temperature=1.0)

    def draws(seed: int) -> list[int]:
        generator = torch.Generator().manual_seed(seed)
        return [choose_next_token(logits, config, generator) for _ in range(40)]

    assert draws(7) == draws(7)
    assert draws(7) != draws(8)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"max_new_tokens": -1}, "max_new_tokens"),
        ({"temperature": -0.1}, "temperature"),
        ({"temperature": math.nan}, "temperature"),
        ({"temperature": math.inf}, "temperature"),
        ({"top_k": -1}, "top_k"),
        ({"top_p": 0.0}, "top_p"),
        ({"top_p": 1.5}, "top_p"),
        ({"top_p": math.nan}, "top_p"),
    ],
)
def test_invalid_sampling_config(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SamplingConfig(**{"max_new_tokens": 4, **changes})


# --- generation loop ------------------------------------------------------------------------------


class Counting(nn.Module):
    """Always predicts ``(last input token + 1) % vocab``; records what it was fed."""

    def __init__(self, context_length: int = 8) -> None:
        super().__init__()
        self.config = replace(TINY_MODEL, vocab_size=32, context_length=context_length)
        self.anchor = nn.Parameter(torch.zeros(1))
        self.inputs: list[list[int]] = []

    def forward(self, input_ids: torch.Tensor, cache: Any = None) -> torch.Tensor:
        self.inputs.append(input_ids[0].tolist())
        logits = torch.zeros(1, input_ids.shape[1], self.config.vocab_size)
        for position, token in enumerate(input_ids[0].tolist()):
            logits[0, position, (token + 1) % self.config.vocab_size] = 10.0
        if cache is not None:
            cache.advance(input_ids.shape[1])
        return logits


def test_eot_stops_generation_and_is_not_emitted() -> None:
    model = Counting()
    result = generate(model, [1], SamplingConfig(max_new_tokens=20, eot_id=5), device=CPU)
    assert result.new_ids == (2, 3, 4)
    assert result.stop_reason == "eot"
    assert result.ids == (1, 2, 3, 4)


def test_max_new_tokens_stop_and_zero_budget() -> None:
    model = Counting()
    result = generate(model, [1], SamplingConfig(max_new_tokens=3), device=CPU)
    assert result.new_ids == (2, 3, 4) and result.stop_reason == "max_new_tokens"
    assert generate(model, [1], SamplingConfig(max_new_tokens=0), device=CPU).new_ids == ()
    with pytest.raises(ValueError, match="at least one token"):
        generate(model, [], SamplingConfig(max_new_tokens=3), device=CPU)


def test_eot_not_stopping_when_disabled() -> None:
    result = generate(Counting(), [3], SamplingConfig(max_new_tokens=4), device=CPU)
    assert result.new_ids == (4, 5, 6, 7)


@pytest.mark.parametrize("use_cache", [True, False])
def test_contexts_follow_the_windowing_policy(use_cache: bool) -> None:
    model = Counting(context_length=8)
    result = generate(
        model, [0, 1, 2], SamplingConfig(max_new_tokens=14), device=CPU, use_cache=use_cache
    )
    assert result.new_ids == tuple(range(3, 17)) and len(result.ids) == 17
    ids = list(range(17))
    if use_cache:
        # Prefills at the start and at every context reset, single tokens otherwise.
        prefills = [inputs for inputs in model.inputs if len(inputs) > 1]
        # Context 8, stride 4: resets when predicting positions 9 and 13 (the last token is
        # position 16, still inside the context starting at 8).
        assert prefills == [ids[0:3], ids[4:9], ids[8:13]]
        assert result.context_resets == 2
    else:
        # Every step feeds exactly ids[context_start(L):L].
        for step, inputs in enumerate(model.inputs):
            length = 3 + step
            assert inputs == ids[context_start(length, 8) : length]
        assert result.context_resets == 2
    for inputs in model.inputs:
        assert len(inputs) <= 8


class Recording(nn.Module):
    """Wraps a model and records the last-position logits of every forward pass."""

    def __init__(self, model: KittyLM) -> None:
        super().__init__()
        self.model = model
        self.config = model.config
        self.last_logits: list[torch.Tensor] = []

    def forward(self, input_ids: torch.Tensor, cache: Any = None) -> torch.Tensor:
        logits: torch.Tensor = self.model(input_ids, cache=cache)
        self.last_logits.append(logits[0, -1].detach().clone())
        return logits


@pytest.mark.parametrize("stride", [None, 3, 16])
@pytest.mark.parametrize(
    "sampling",
    [
        SamplingConfig(max_new_tokens=50),
        SamplingConfig(max_new_tokens=50, temperature=0.9, top_k=12, top_p=0.9),
    ],
    ids=["greedy", "sampled"],
)
def test_kv_cache_generation_equals_full_forward_decoding(
    stride: int | None, sampling: SamplingConfig
) -> None:
    torch.manual_seed(3)
    model = KittyLM(TINY_MODEL)
    prompt = torch.randint(0, TINY_MODEL.vocab_size, (5,)).tolist()
    runs = {}
    for use_cache in (True, False):
        recorder = Recording(model)
        result = generate(
            recorder,
            prompt,
            sampling,
            device=CPU,
            stride=stride,
            generator=torch.Generator().manual_seed(99),
            use_cache=use_cache,
        )
        runs[use_cache] = (result, torch.stack(recorder.last_logits))
    (cached, cached_logits), (full, full_logits) = runs[True], runs[False]
    assert cached.new_ids == full.new_ids
    assert len(cached.new_ids) == 50 and cached.context_resets == full.context_resets > 0
    assert cached_logits.shape == full_logits.shape
    torch.testing.assert_close(cached_logits, full_logits, rtol=1e-4, atol=1e-5)


def test_generation_restores_training_mode_and_records_no_grad() -> None:
    model = KittyLM(TINY_MODEL).train()
    result = generate(model, [1, 2], SamplingConfig(max_new_tokens=3), device=CPU)
    assert model.training and len(result.new_ids) == 3
    assert all(p.grad is None for p in model.parameters())
