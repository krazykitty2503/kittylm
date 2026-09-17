"""Overfit gate: one computation shared by SMOKE-GPU-001 and EXP-000.

Purpose:
    Decide whether a model has *memorized* a small fixture, which proves the training stack can
    fit data at all before any real experiment is trusted. The gate measures, on the whole
    fixture in eval mode:

    - final loss (mean natural-log cross-entropy over every target, windowed per D-023);
    - perplexity ``exp(loss)`` and its consistency with the loss;
    - greedy generation from the first ``prompt_tokens`` fixture tokens, compared token by
      token with the rest of the fixture (match rate, longest exact prefix, exact match);
    - a non-finite scan of the loss, the recorded training metrics and gradient norms.

    SMOKE-GPU-001 (engineering) and EXP-000 (formal) call the same function with the same
    thresholds, so a smoke pass means exactly what the formal gate will check.

Public API:
    OverfitGateThresholds(max_loss=0.05, max_ppl=1.05, min_token_match=0.95, prompt_tokens=16)
    OverfitGateReport (measured values, ``checks``, ``passed``, ``failures``)
    evaluate_overfit_gate(model, ids, byte_lengths, *, device, thresholds=..., context_length=None,
                          stride=None, autocast_dtype=None, metrics_records=(), grad_norms=())
    token_match(reference, generated) -> (rate, longest_exact_prefix, exact_match)
    nonfinite_metrics(records) -> list[str]
    compare_loss_series(a, b) -> str

Shapes:
    Token ids are 1-D sequences; the model sees ``[1, T]`` windows.

Dtype:
    As kittylm/evaluation/perplexity.py (float32 logits, float64 totals).

Device:
    The model runs on ``device``; all gate arithmetic happens on the host.

Invariants:
    - ``final_ppl == math.exp(final_loss)`` exactly (``inf`` beyond float range), and
      ``checks["ppl_consistent"]`` re-checks it with the ledger tolerance so a record written
      from this report validates.
    - ``passed`` is True only if every entry of ``checks`` is True; ``failures`` names the rest.
    - Generation is greedy with no EOT stop, so it always produces ``len(ids) - prompt_tokens``
      tokens to compare; missing or extra tokens count as mismatches.
    - Logger values written as the strings ``"nan"``, ``"inf"`` or ``"-inf"`` count as
      non-finite, at any depth of nested mappings, lists and tuples.

Failure modes:
    - A fixture no longer than ``prompt_tokens`` raises ValueError (nothing to compare).
    - Non-finite model outputs make the report fail (never raise), with the reason in
      ``failures``.

See:
    Plan rev 3.3 sections 5, 7 and 9 (EXP-000); D-023; kittylm/evaluation/perplexity.py.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from kittylm.evaluation.perplexity import perplexity, ppl_is_consistent, stream_nll
from kittylm.inference.generate import SamplingConfig, generate

__all__ = [
    "OverfitGateReport",
    "OverfitGateThresholds",
    "compare_loss_series",
    "evaluate_overfit_gate",
    "nonfinite_metrics",
    "token_match",
]

_NONFINITE_STRINGS = frozenset({"nan", "inf", "-inf"})


@dataclass(frozen=True)
class OverfitGateThresholds:
    """Pass criteria (plan rev 3.3 section 7)."""

    max_loss: float = 0.05
    max_ppl: float = 1.05
    min_token_match: float = 0.95
    prompt_tokens: int = 16

    def __post_init__(self) -> None:
        """Validate thresholds."""
        if not all(math.isfinite(v) for v in (self.max_loss, self.max_ppl, self.min_token_match)):
            raise ValueError("gate thresholds must be finite")
        if not 0 <= self.min_token_match <= 1 or self.prompt_tokens < 1:
            raise ValueError("min_token_match must be in [0, 1] and prompt_tokens >= 1")


@dataclass(frozen=True)
class OverfitGateReport:
    """Everything the gate measured and which checks passed."""

    final_loss: float
    final_ppl: float
    bits_per_byte: float | None
    scored_tokens: int
    token_match_rate: float
    longest_exact_prefix: int
    exact_match: bool
    reference_tokens: int
    generated_tokens: int
    nonfinite: tuple[str, ...]
    thresholds: OverfitGateThresholds
    checks: dict[str, bool] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """True only if every check passed."""
        return bool(self.checks) and all(self.checks.values())

    @property
    def failures(self) -> list[str]:
        """Names of the checks that failed."""
        return [name for name, ok in self.checks.items() if not ok]


def token_match(reference: Sequence[int], generated: Sequence[int]) -> tuple[float, int, bool]:
    """Position-wise match rate over ``reference``, longest exact prefix, and exact equality."""
    if not reference:
        raise ValueError("reference must contain at least one token")
    matches = sum(
        1 for i, token in enumerate(reference) if i < len(generated) and generated[i] == token
    )
    prefix = 0
    for expected, actual in zip(reference, generated, strict=False):
        if expected != actual:
            break
        prefix += 1
    return matches / len(reference), prefix, list(reference) == list(generated)


def nonfinite_metrics(records: Iterable[Mapping[str, Any]]) -> list[str]:
    """``"step N: key"`` for every non-finite number (or logger string) in metric records."""
    found: list[str] = []

    def visit(value: Any, key: str, step: Any) -> None:
        if isinstance(value, Mapping):
            for sub_key, sub_value in value.items():
                visit(sub_value, f"{key}.{sub_key}", step)
        elif isinstance(value, list | tuple):
            for index, item in enumerate(value):
                visit(item, f"{key}[{index}]", step)
        elif isinstance(value, bool):
            return
        elif isinstance(value, int | float) and not math.isfinite(value):
            found.append(f"step {step}: {key}")
        elif isinstance(value, str) and value.lower() in _NONFINITE_STRINGS:
            found.append(f"step {step}: {key}")

    for record in records:
        step = record.get("step")
        for key, value in record.items():
            if key != "step":
                visit(value, key, step)
    return found


def compare_loss_series(a: Sequence[float], b: Sequence[float]) -> str:
    """``identical`` (bit-for-bit), ``max_abs_dev=<x>``, or ``length differs (m vs n)``."""
    if len(a) != len(b):
        return f"length differs ({len(a)} vs {len(b)})"
    if all(float(x).hex() == float(y).hex() for x, y in zip(a, b, strict=True)):
        return "identical"
    deviation = max(abs(float(x) - float(y)) for x, y in zip(a, b, strict=True))
    return f"max_abs_dev={deviation:.3e}"


def evaluate_overfit_gate(
    model: nn.Module,
    ids: Sequence[int],
    byte_lengths: Sequence[int],
    *,
    device: torch.device,
    thresholds: OverfitGateThresholds | None = None,
    context_length: int | None = None,
    stride: int | None = None,
    autocast_dtype: torch.dtype | None = None,
    metrics_records: Iterable[Mapping[str, Any]] = (),
    grad_norms: Iterable[float] = (),
) -> OverfitGateReport:
    """Measure the fixture and apply the overfit-gate thresholds."""
    limits = thresholds or OverfitGateThresholds()
    if len(ids) <= limits.prompt_tokens:
        raise ValueError(
            f"fixture has {len(ids)} tokens; the gate needs more than prompt_tokens="
            f"{limits.prompt_tokens}"
        )
    totals = stream_nll(
        model,
        ids,
        byte_lengths,
        device=device,
        context_length=context_length,
        stride=stride,
        autocast_dtype=autocast_dtype,
    )
    loss = totals.nats / totals.tokens
    loss_finite = math.isfinite(loss)
    ppl = perplexity(loss) if loss_finite else math.inf
    bpb = totals.bits_per_byte if totals.bytes > 0 and loss_finite else None

    prompt, reference = list(ids[: limits.prompt_tokens]), list(ids[limits.prompt_tokens :])
    result = generate(
        model,
        prompt,
        SamplingConfig(max_new_tokens=len(reference)),  # greedy, no EOT stop
        device=device,
        context_length=context_length,
        stride=stride,
        autocast_dtype=autocast_dtype,
    )
    rate, prefix, exact = token_match(reference, result.new_ids)

    nonfinite = [] if loss_finite else ["final_loss"]
    nonfinite += nonfinite_metrics(metrics_records)
    nonfinite += [f"grad_norm[{i}]" for i, g in enumerate(grad_norms) if not math.isfinite(g)]
    checks = {
        "final_loss_below_max": loss_finite and loss < limits.max_loss,
        "ppl_at_most_max": ppl <= limits.max_ppl,
        "ppl_consistent": loss_finite and ppl_is_consistent(loss, ppl),
        "token_match_at_least_min": rate >= limits.min_token_match,
        "all_values_finite": not nonfinite,
    }
    return OverfitGateReport(
        final_loss=loss,
        final_ppl=ppl,
        bits_per_byte=bpb,
        scored_tokens=totals.tokens,
        token_match_rate=rate,
        longest_exact_prefix=prefix,
        exact_match=exact,
        reference_tokens=len(reference),
        generated_tokens=len(result.new_ids),
        nonfinite=tuple(nonfinite),
        thresholds=limits,
        checks=checks,
    )
