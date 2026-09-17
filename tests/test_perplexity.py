"""Loss, perplexity and bits-per-byte, including hand-computed reference cases."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from kittylm.evaluation.perplexity import (
    Document,
    EvaluationReport,
    NLLTotals,
    bits_per_byte,
    evaluate_documents,
    perplexity,
    ppl_is_consistent,
    stream_nll,
    token_byte_lengths,
)
from kittylm.evaluation.windows import context_start
from kittylm.model.transformer import KittyLM
from tests.training_helpers import TINY_MODEL

CPU = torch.device("cpu")
# Cross-entropy runs on float32 logits, so hand-computed values agree to float32 precision.
REL = 1e-6


class FixedDistribution(nn.Module):
    """Predicts the same next-token distribution at every position."""

    def __init__(self, probs: list[float], context_length: int = 8) -> None:
        super().__init__()
        self.config = replace(TINY_MODEL, vocab_size=len(probs), context_length=context_length)
        self.register_buffer("log_probs", torch.log(torch.tensor(probs, dtype=torch.float64)))
        self.anchor = nn.Parameter(torch.zeros(1))  # generation/eval helpers need a parameter
        self.calls: list[list[int]] = []

    def forward(self, input_ids: torch.Tensor, cache: object = None) -> torch.Tensor:
        self.calls.append(input_ids[0].tolist())
        batch, length = input_ids.shape
        return self.log_probs.float().expand(batch, length, -1).clone()


def test_hand_computed_bits_per_byte_and_perplexity() -> None:
    # p = (1/2, 1/4, 1/8, 1/8). Targets 0, 1, 2 cost 1, 2 and 3 bits: 6 bits in total.
    # Their byte lengths are 2, 1 and 3: 6 bytes, so bpb = 6 / 6 = 1.0.
    # Mean loss = 6 ln2 / 3 = 2 ln2 nats, so perplexity = exp(2 ln2) = 4.
    model = FixedDistribution([0.5, 0.25, 0.125, 0.125])
    ids = [3, 0, 1, 2]  # token 0 is context only
    byte_lengths = [5, 2, 1, 3]
    totals = stream_nll(model, ids, byte_lengths, device=CPU)
    assert totals.tokens == 3 and totals.bytes == 6
    assert totals.nats == pytest.approx(6 * math.log(2), rel=REL)
    assert totals.bits_per_byte == pytest.approx(1.0, rel=REL)
    assert totals.mean_loss == pytest.approx(2 * math.log(2), rel=REL)
    assert totals.perplexity == pytest.approx(4.0, rel=REL)


def test_hand_computed_uniform_case_per_category() -> None:
    # Uniform over 4 tokens: every target costs exactly 2 bits.
    # prose: 3 targets of 1 byte each -> 6 bits / 3 bytes = 2.0 bpb.
    # code:  2 targets of 4 bytes each -> 4 bits / 8 bytes = 0.5 bpb.
    # overall: 10 bits / 11 bytes; loss = ln 4, ppl = 4 regardless of category.
    model = FixedDistribution([0.25] * 4)
    report = evaluate_documents(
        model,
        [
            Document("prose", [0, 1, 2, 3], [9, 1, 1, 1]),
            Document("code", [2, 2, 2], [9, 4, 4]),
        ],
        device=CPU,
    )
    metrics = report.metrics()
    assert metrics["bpb_by_category"] == pytest.approx({"code": 0.5, "prose": 2.0}, rel=REL)
    assert metrics["bpb"] == pytest.approx(10 / 11, rel=REL)
    assert metrics["loss"] == pytest.approx(math.log(4), rel=REL)
    assert metrics["ppl"] == pytest.approx(4.0, rel=REL)
    assert (metrics["tokens"], metrics["bytes"]) == (5, 11)


def test_categories_combine_by_summing_not_averaging() -> None:
    # Two prose documents of different lengths: the category bpb weights every byte equally.
    model = FixedDistribution([0.25] * 4)
    report = evaluate_documents(
        model,
        [Document("prose", [0, 1], [0, 1]), Document("prose", [0, 1, 1, 1, 1], [0, 4, 4, 4, 4])],
        device=CPU,
    )
    prose = report.by_category["prose"]
    assert (prose.tokens, prose.bytes) == (5, 17)
    assert prose.bits_per_byte == pytest.approx(10 / 17, rel=REL)  # not mean(2.0, 0.5)


def test_perplexity_is_exactly_exp_of_loss() -> None:
    for loss in (0.0, 0.01, 1.2345, 5.0):
        assert perplexity(loss) == math.exp(loss)  # exact, not approximately
        totals = NLLTotals(nats=loss * 7, tokens=7, bytes=3)
        assert totals.perplexity == math.exp(totals.mean_loss)
        assert ppl_is_consistent(loss, math.exp(loss))
    assert not ppl_is_consistent(1.0, math.exp(1.0) * 1.01)
    assert not ppl_is_consistent(math.nan, 1.0)
    assert not ppl_is_consistent(1.0, math.inf)
    with pytest.raises(ValueError, match="non-finite"):
        perplexity(math.nan)


def test_bits_per_byte_guards_and_units() -> None:
    assert bits_per_byte(8 * math.log(2), 4) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="at least one predicted byte"):
        bits_per_byte(1.0, 0)
    with pytest.raises(ValueError, match="no predicted tokens"):
        _ = NLLTotals(0.0, 0, 0).mean_loss


def test_windowed_loss_equals_explicit_per_target_contexts() -> None:
    # Score each target on its own with inputs ids[context_start(t):t]; the windowed evaluation
    # must produce the same total, so every target is scored once with the documented context.
    torch.manual_seed(0)
    model = KittyLM(TINY_MODEL).eval()
    ids = torch.randint(0, TINY_MODEL.vocab_size, (45,)).tolist()
    ones = [1] * len(ids)
    windowed = stream_nll(model, ids, ones, device=CPU)
    explicit = 0.0
    with torch.no_grad():
        for target in range(1, len(ids)):
            start = context_start(target, TINY_MODEL.context_length)
            logits = model(torch.tensor([ids[start:target]]))[0, -1]
            explicit += float(F.cross_entropy(logits[None].float(), torch.tensor([ids[target]])))
    assert windowed.tokens == len(ids) - 1 == windowed.bytes
    assert windowed.nats == pytest.approx(explicit, rel=1e-5)


def test_stream_fitting_the_context_matches_a_plain_forward() -> None:
    torch.manual_seed(1)
    model = KittyLM(TINY_MODEL).eval()
    ids = torch.randint(0, TINY_MODEL.vocab_size, (TINY_MODEL.context_length + 1,)).tolist()
    totals = stream_nll(model, ids, [1] * len(ids), device=CPU)
    with torch.no_grad():
        logits = model(torch.tensor([ids[:-1]]))[0]
    expected = F.cross_entropy(logits.float(), torch.tensor(ids[1:]), reduction="mean")
    assert totals.mean_loss == pytest.approx(float(expected), rel=1e-6)


class FakeTokenizer:
    first_special_id = 10

    def token_bytes(self, token_id: int) -> bytes:
        return b"x" * (token_id + 1)


def test_special_tokens_count_zero_bytes() -> None:
    assert token_byte_lengths(FakeTokenizer(), [0, 3, 10, 11]) == [1, 4, 0, 0]


def test_argument_validation_and_mode_restored() -> None:
    model = FixedDistribution([0.5, 0.5])
    model.train()
    with pytest.raises(ValueError, match="one entry per token"):
        stream_nll(model, [0, 1, 0], [1, 1], device=CPU)
    with pytest.raises(ValueError, match="at least one document"):
        evaluate_documents(model, [], device=CPU)
    stream_nll(model, [0, 1], [1, 1], device=CPU)
    assert model.training


def test_perplexity_beyond_float_range_is_infinite_not_an_error() -> None:
    # Regression (PR #5 review): exp(loss) overflowed for finite losses above ~709.78 and the
    # evaluation report raised OverflowError instead of reporting infinite perplexity.
    assert perplexity(709.0) == math.exp(709.0)
    assert perplexity(710.0) == math.inf
    totals = NLLTotals(nats=800.0 * 4, tokens=4, bytes=4)
    metrics = EvaluationReport(totals, {"prose": totals}).metrics()
    assert metrics["ppl"] == math.inf and metrics["loss"] == 800.0
    assert ppl_is_consistent(800.0, math.inf)
    assert not ppl_is_consistent(800.0, 1e300)
    assert not ppl_is_consistent(1.0, math.inf)
    assert not ppl_is_consistent(1.0, math.nan)


@pytest.mark.parametrize("context_length", [0, -1, TINY_MODEL.context_length + 1, 10**9])
def test_context_override_is_bounded_by_the_model(context_length: int) -> None:
    # Regression (PR #5 review): an oversized override reached window construction unchecked.
    model = KittyLM(TINY_MODEL)
    with pytest.raises(ValueError, match="context_length must be in"):
        stream_nll(model, [1, 2, 3], [1, 1, 1], device=CPU, context_length=context_length)
