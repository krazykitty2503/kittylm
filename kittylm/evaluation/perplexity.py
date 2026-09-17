"""Loss, perplexity and bits-per-byte over windowed token streams.

Purpose:
    Measure how well a model predicts text in three related units:

    - **loss**: mean natural-log cross-entropy per predicted token (nats/token);
    - **perplexity**: ``exp(loss)``, the effective number of equally likely choices per token;
    - **bits-per-byte (bpb)**: total negative log-likelihood in bits divided by the number of
      bytes the predicted tokens decode to. Unlike loss, bpb does not depend on the tokenizer,
      so it is the comparison metric across vocabulary sizes (D-003).

    Streams longer than the context are scored with the shared windowing policy (D-023), so
    every target token is scored exactly once. Totals are kept as sums (nats, tokens, bytes) and
    only divided at the end, so combining documents or categories weights every token equally.

Public API:
    NLLTotals(nats, tokens, bytes)
        ``mean_loss``, ``perplexity``, ``bits_per_byte``; ``+`` combines totals.
    Document(category, ids, byte_lengths)
    EvaluationReport(overall, by_category) with ``metrics()``
    perplexity(mean_loss) -> float
    bits_per_byte(total_nats, n_bytes) -> float
    ppl_is_consistent(loss, ppl, rel_tol=PPL_REL_TOL) -> bool
    token_byte_lengths(tokenizer, ids) -> list[int]
    stream_nll(model, ids, byte_lengths, *, device, context_length=None, stride=None,
               autocast_dtype=None) -> NLLTotals
    evaluate_documents(model, documents, *, device, ...) -> EvaluationReport

Shapes:
    Model input ``[1, T]`` per window (T <= context length); logits ``[1, T, vocab]``;
    per-token NLL ``[T]``.

Dtype:
    Logits are upcast to float32 before cross-entropy; totals are accumulated as Python floats
    (float64) so long streams do not lose precision.

Device:
    The model and windows run on ``device``; totals are returned on the host.

Math:
    ``loss = nats / tokens``; ``ppl = exp(loss)``; ``bpb = nats / (ln 2 * bytes)``.

Invariants:
    - ``perplexity`` is exactly ``math.exp(mean_loss)`` (no separate estimate that could drift).
    - Byte counts include only ordinary tokens: special tokens (``id >= first_special_id``)
      count 0 bytes, because they do not decode to text; their NLL still counts.
    - The first token of a stream is never predicted and contributes no NLL and no bytes.

Failure modes:
    - ``bits_per_byte`` with zero bytes, ``byte_lengths`` of the wrong length, an empty
      document list, or a non-finite loss passed to ``perplexity`` raises ValueError.
    - A model that returns non-finite logits yields non-finite totals; callers (the overfit
      gate) must check ``math.isfinite``.

See:
    D-003, D-023, plan rev 3.3 section 5, kittylm/ledger.py (PPL_REL_TOL).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torch.nn.functional as F
from torch import nn

from kittylm.evaluation.windows import evaluation_windows
from kittylm.inference.runtime import autocast_context, eval_mode, model_config
from kittylm.ledger import PPL_REL_TOL

__all__ = [
    "PPL_REL_TOL",
    "Document",
    "EvaluationReport",
    "NLLTotals",
    "bits_per_byte",
    "evaluate_documents",
    "perplexity",
    "ppl_is_consistent",
    "stream_nll",
    "token_byte_lengths",
]

LN2 = math.log(2.0)


class TokenBytes(Protocol):
    """The part of a tokenizer bpb needs."""

    @property
    def first_special_id(self) -> int: ...

    def token_bytes(self, token_id: int) -> bytes: ...


def perplexity(mean_loss: float) -> float:
    """``exp(mean_loss)``: the perplexity of a mean natural-log cross-entropy."""
    if not math.isfinite(mean_loss):
        raise ValueError(f"perplexity is undefined for a non-finite loss ({mean_loss})")
    return math.exp(mean_loss)


def bits_per_byte(total_nats: float, n_bytes: int) -> float:
    """Total NLL in nats converted to bits, per byte of predicted text."""
    if n_bytes <= 0:
        raise ValueError("bits_per_byte needs at least one predicted byte")
    return total_nats / (LN2 * n_bytes)


def ppl_is_consistent(loss: float, ppl: float, rel_tol: float = PPL_REL_TOL) -> bool:
    """Whether a recorded ``ppl`` equals ``exp(loss)`` (the ledger's rule)."""
    if not (math.isfinite(loss) and math.isfinite(ppl)):
        return False
    return math.isclose(ppl, math.exp(loss), rel_tol=rel_tol, abs_tol=0.0)


def token_byte_lengths(tokenizer: TokenBytes, ids: Sequence[int]) -> list[int]:
    """Bytes each token decodes to; special tokens count 0 (they are not text)."""
    first_special = tokenizer.first_special_id
    return [0 if t >= first_special else len(tokenizer.token_bytes(t)) for t in ids]


@dataclass(frozen=True)
class NLLTotals:
    """Summed negative log-likelihood (nats) over ``tokens`` targets decoding to ``bytes``."""

    nats: float
    tokens: int
    bytes: int

    def __add__(self, other: NLLTotals) -> NLLTotals:
        """Combine totals (sums, so every token keeps equal weight)."""
        return NLLTotals(
            self.nats + other.nats, self.tokens + other.tokens, self.bytes + other.bytes
        )

    @property
    def mean_loss(self) -> float:
        """Mean natural-log cross-entropy per predicted token."""
        if self.tokens <= 0:
            raise ValueError("no predicted tokens")
        return self.nats / self.tokens

    @property
    def perplexity(self) -> float:
        """``exp(mean_loss)``."""
        return perplexity(self.mean_loss)

    @property
    def bits_per_byte(self) -> float:
        """Bits per byte of predicted text."""
        return bits_per_byte(self.nats, self.bytes)


@dataclass(frozen=True)
class Document:
    """One token stream to evaluate, its category, and each token's byte length."""

    category: str
    ids: Sequence[int]
    byte_lengths: Sequence[int]


@dataclass(frozen=True)
class EvaluationReport:
    """Totals over all documents and per category."""

    overall: NLLTotals
    by_category: dict[str, NLLTotals]

    def metrics(self) -> dict[str, Any]:
        """Loss, perplexity and bpb overall, plus bpb per category."""
        return {
            "loss": self.overall.mean_loss,
            "ppl": self.overall.perplexity,
            "bpb": self.overall.bits_per_byte,
            "bpb_by_category": {
                name: totals.bits_per_byte for name, totals in sorted(self.by_category.items())
            },
            "tokens": self.overall.tokens,
            "bytes": self.overall.bytes,
        }


@torch.no_grad()
def stream_nll(
    model: nn.Module,
    ids: Sequence[int],
    byte_lengths: Sequence[int],
    *,
    device: torch.device,
    context_length: int | None = None,
    stride: int | None = None,
    autocast_dtype: torch.dtype | None = None,
) -> NLLTotals:
    """Score every target of ``ids`` once (windowed) and sum NLL, tokens and bytes."""
    if len(byte_lengths) != len(ids):
        raise ValueError("byte_lengths must have one entry per token")
    ctx = context_length if context_length is not None else model_config(model).context_length
    nats, tokens, n_bytes = 0.0, 0, 0
    with eval_mode(model):
        for window in evaluation_windows(len(ids), ctx, stride):
            chunk = torch.tensor([list(ids[window.start : window.end])], device=device)
            inputs, targets = chunk[:, :-1], chunk[0, 1:]
            with autocast_context(device, autocast_dtype):
                logits = model(inputs)
            nll = F.cross_entropy(logits[0].float(), targets, reduction="none")
            scored = nll[window.score_offset :]
            nats += float(scored.double().sum())
            tokens += scored.numel()
            n_bytes += sum(byte_lengths[window.score_from : window.end])
    return NLLTotals(nats, tokens, n_bytes)


def evaluate_documents(
    model: nn.Module,
    documents: Sequence[Document],
    *,
    device: torch.device,
    context_length: int | None = None,
    stride: int | None = None,
    autocast_dtype: torch.dtype | None = None,
) -> EvaluationReport:
    """Evaluate documents independently and aggregate totals overall and per category."""
    if not documents:
        raise ValueError("evaluate_documents needs at least one document")
    overall = NLLTotals(0.0, 0, 0)
    by_category: dict[str, NLLTotals] = {}
    for doc in documents:
        totals = stream_nll(
            model,
            doc.ids,
            doc.byte_lengths,
            device=device,
            context_length=context_length,
            stride=stride,
            autocast_dtype=autocast_dtype,
        )
        overall = overall + totals
        by_category[doc.category] = by_category.get(doc.category, NLLTotals(0.0, 0, 0)) + totals
    return EvaluationReport(overall, by_category)
