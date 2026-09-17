"""Overfit gate shared by SMOKE-GPU-001 and EXP-000."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from kittylm.evaluation.gates import (
    OverfitGateThresholds,
    compare_loss_series,
    evaluate_overfit_gate,
    nonfinite_metrics,
    token_match,
)
from kittylm.evaluation.windows import evaluation_windows
from kittylm.model.transformer import KittyLM
from tests.training_helpers import TINY_MODEL

CPU = torch.device("cpu")
FIXTURE_LENGTH = 40  # longer than the tiny context (16): the gate must use windows


def fixture_ids() -> list[int]:
    generator = torch.Generator().manual_seed(5)
    return torch.randint(0, TINY_MODEL.vocab_size, (FIXTURE_LENGTH,), generator=generator).tolist()


def memorize(ids: list[int], steps: int = 400) -> KittyLM:
    """Overfit the tiny model on the D-023 windows of ``ids`` (the contexts the gate uses)."""
    torch.manual_seed(0)
    model = KittyLM(TINY_MODEL)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    windows = evaluation_windows(len(ids), TINY_MODEL.context_length)
    for _ in range(steps):
        optimizer.zero_grad()
        for window in windows:
            chunk = torch.tensor([ids[window.start : window.end]])
            logits = model(chunk[:, :-1])
            loss = F.cross_entropy(logits[0], chunk[0, 1:])
            (loss / len(windows)).backward()
        optimizer.step()
    return model


@pytest.fixture(scope="module")
def memorized() -> KittyLM:
    return memorize(fixture_ids())


def test_memorized_fixture_passes_the_gate(memorized: KittyLM) -> None:
    ids = fixture_ids()
    report = evaluate_overfit_gate(memorized, ids, [1] * len(ids), device=CPU)
    assert report.passed, (report.failures, report.final_loss, report.token_match_rate)
    assert report.final_loss < 0.05 and report.final_ppl == math.exp(report.final_loss)
    assert report.scored_tokens == FIXTURE_LENGTH - 1
    assert report.reference_tokens == report.generated_tokens == FIXTURE_LENGTH - 16
    assert report.token_match_rate >= 0.95 and report.exact_match
    assert report.longest_exact_prefix == FIXTURE_LENGTH - 16
    assert report.bits_per_byte is not None and report.nonfinite == ()


def test_untrained_model_fails_with_named_checks() -> None:
    torch.manual_seed(0)
    ids = fixture_ids()
    report = evaluate_overfit_gate(KittyLM(TINY_MODEL), ids, [1] * len(ids), device=CPU)
    assert not report.passed
    assert {"final_loss_below_max", "ppl_at_most_max", "token_match_at_least_min"} <= set(
        report.failures
    )
    assert report.checks["ppl_consistent"] and report.checks["all_values_finite"]


def test_nonfinite_training_metrics_fail_the_gate(memorized: KittyLM) -> None:
    ids = fixture_ids()
    records = [{"step": 1, "loss": 1.0}, {"step": 2, "loss": "nan", "grad_norm": 0.5}]
    report = evaluate_overfit_gate(
        memorized,
        ids,
        [1] * len(ids),
        device=CPU,
        metrics_records=records,
        grad_norms=[1.0, math.inf],
    )
    assert not report.passed and report.failures == ["all_values_finite"]
    assert report.nonfinite == ("step 2: loss", "grad_norm[1]")


def test_nonfinite_model_outputs_fail_without_raising(memorized: KittyLM) -> None:
    broken = memorize(fixture_ids(), steps=1)
    with torch.no_grad():
        broken.norm.weight.fill_(math.nan)
    ids = fixture_ids()
    report = evaluate_overfit_gate(broken, ids, [1] * len(ids), device=CPU)
    assert not report.passed
    assert "final_loss" in report.nonfinite and not report.checks["ppl_consistent"]
    assert report.bits_per_byte is None


def test_thresholds_are_applied_as_configured(memorized: KittyLM) -> None:
    ids = fixture_ids()
    strict = OverfitGateThresholds(max_loss=0.0, prompt_tokens=8)
    report = evaluate_overfit_gate(memorized, ids, [1] * len(ids), device=CPU, thresholds=strict)
    assert report.failures == ["final_loss_below_max"]
    assert report.reference_tokens == FIXTURE_LENGTH - 8
    with pytest.raises(ValueError, match="prompt_tokens"):
        evaluate_overfit_gate(
            memorized, ids[:16], [1] * 16, device=CPU, thresholds=OverfitGateThresholds()
        )
    with pytest.raises(ValueError, match="finite"):
        OverfitGateThresholds(max_loss=math.nan)
    with pytest.raises(ValueError, match="min_token_match"):
        replace(OverfitGateThresholds(), min_token_match=1.5)


@pytest.mark.parametrize(
    ("reference", "generated", "expected"),
    [
        ([1, 2, 3, 4], [1, 2, 3, 4], (1.0, 4, True)),
        ([1, 2, 3, 4], [1, 9, 3, 4], (0.75, 1, False)),
        ([1, 2, 3, 4], [1, 2], (0.5, 2, False)),  # missing tokens are mismatches
        ([1, 2], [1, 2, 3], (1.0, 2, False)),  # extra tokens break exact match only
        ([1, 2, 3, 4], [], (0.0, 0, False)),
    ],
)
def test_token_match(
    reference: list[int], generated: list[int], expected: tuple[float, int, bool]
) -> None:
    assert token_match(reference, generated) == expected


def test_token_match_needs_a_reference() -> None:
    with pytest.raises(ValueError, match="reference"):
        token_match([], [1])


def test_nonfinite_metric_scan() -> None:
    records = [
        {"step": 1, "loss": 2.0, "ok": True, "nested": {"x": 1.0}},
        {"step": 2, "loss": math.inf, "nested": {"x": "-inf"}, "note": "infinite"},
        {"step": 3, "lr": "NaN"},
    ]
    assert nonfinite_metrics(records) == ["step 2: loss", "step 2: nested.x", "step 3: lr"]


def test_compare_loss_series() -> None:
    assert compare_loss_series([1.0, 0.5], [1.0, 0.5]) == "identical"
    assert compare_loss_series([1.0, 0.5], [1.0, 0.5 + 1e-6]) == "max_abs_dev=1.000e-06"
    assert compare_loss_series([1.0], [1.0, 2.0]) == "length differs (1 vs 2)"
    assert compare_loss_series([0.0], [-0.0]) == "max_abs_dev=0.000e+00"  # bit-level, not ==


def test_nonfinite_values_inside_nested_sequences_are_found() -> None:
    # Regression (PR #5 review): lists and tuples were not visited, so a NaN inside a logged
    # sequence left all_values_finite True.
    records = [
        {
            "step": 4,
            "per_layer": [1.0, math.nan, {"inner": ("ok", "inf")}],
            "pair": (0.5, -math.inf),
        },
        {"step": 5, "per_layer": [1.0, 2.0], "names": ["nan-tolerant", "fine"]},
    ]
    assert nonfinite_metrics(records) == [
        "step 4: per_layer[1]",
        "step 4: per_layer[2].inner[1]",
        "step 4: pair[1]",
    ]


def test_nested_nonfinite_metric_fails_the_gate(memorized: KittyLM) -> None:
    ids = fixture_ids()
    report = evaluate_overfit_gate(
        memorized,
        ids,
        [1] * len(ids),
        device=CPU,
        metrics_records=[{"step": 1, "grad_norm_per_layer": [0.1, math.nan]}],
    )
    assert report.failures == ["all_values_finite"]
    assert report.nonfinite == ("step 1: grad_norm_per_layer[1]",)
