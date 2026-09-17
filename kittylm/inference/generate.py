"""Autoregressive generation: greedy, temperature, top-k and top-p sampling with a KV cache.

Purpose:
    Turn a prompt into new tokens one at a time. Each step takes the logits of the last
    position, turns them into a choice (greedy argmax, or a sample from the temperature-scaled,
    top-k/top-p filtered distribution), appends it, and stops at ``<|endoftext|>`` or after
    ``max_new_tokens``. The KV cache makes each step cost one token of compute instead of
    re-running the whole prefix. When the context is full, the cache is reset and re-filled
    following the shared windowing policy (D-023), so generation conditions every prediction on
    the same context that evaluation and the overfit gate use.

Public API:
    SamplingConfig(max_new_tokens, temperature=0.0, top_k=0, top_p=1.0, eot_id=None)
    GenerationResult(prompt_ids, new_ids, stop_reason, context_resets)
    generate(model, prompt_ids, config, *, device, context_length=None, stride=None,
             autocast_dtype=None, generator=None, use_cache=True) -> GenerationResult
    choose_next_token(logits, config, generator=None) -> int
    top_k_filter(logits, k) / top_p_filter(logits, p)

Shapes:
    Batch size 1. Model input ``[1, T]`` (int64); last-position logits ``[vocab]``; filters
    accept ``[..., vocab]`` and return the same shape.

Dtype:
    Logits are upcast to float32 before temperature, filtering and softmax; the KV cache uses
    the autocast dtype when one is given, else the parameter dtype.

Device:
    The model and cache live on ``device``. Sampling runs on CPU with the caller's CPU
    ``torch.Generator``, so a seeded sample sequence is identical on CPU and GPU logits.

Math:
    ``p = softmax(logits / temperature)``; top-k keeps logits ``>= `` the k-th largest; top-p
    keeps the smallest prefix of tokens (by descending probability) whose mass reaches ``p``
    (the most likely token is always kept).

Invariants:
    - ``temperature == 0`` is greedy and ignores top-k/top-p; ties go to the lowest token id.
    - With or without the cache, generation produces the same tokens (the cache is an
      optimization, verified against full-forward decoding over context resets).
    - ``new_ids`` never contains the EOT token; ``stop_reason`` says why generation ended.
    - The model's train/eval mode is restored afterwards; no gradients are recorded.

Failure modes:
    - Invalid sampling settings (negative or non-finite temperature, ``top_p`` outside
      ``(0, 1]``, negative ``top_k`` or ``max_new_tokens``) raise ValueError.
    - An empty prompt raises ValueError.

See:
    D-023 (windowing), plan rev 3.3 section 5, kittylm/model/kv_cache.py.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from kittylm.evaluation.windows import context_start
from kittylm.inference.runtime import autocast_context, cache_dtype, eval_mode, model_config
from kittylm.model.kv_cache import KVCache

__all__ = [
    "GenerationResult",
    "SamplingConfig",
    "choose_next_token",
    "generate",
    "top_k_filter",
    "top_p_filter",
]


@dataclass(frozen=True)
class SamplingConfig:
    """How to choose each new token and when to stop."""

    max_new_tokens: int
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    eot_id: int | None = None

    def __post_init__(self) -> None:
        """Validate settings."""
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and >= 0 (0 means greedy)")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0 (0 disables top-k)")
        if not (math.isfinite(self.top_p) and 0 < self.top_p <= 1):
            raise ValueError("top_p must be in (0, 1] (1 disables top-p)")

    @property
    def greedy(self) -> bool:
        """True when decoding is deterministic argmax."""
        return self.temperature == 0


@dataclass(frozen=True)
class GenerationResult:
    """Prompt, generated continuation, and why generation stopped."""

    prompt_ids: tuple[int, ...]
    new_ids: tuple[int, ...]
    stop_reason: Literal["eot", "max_new_tokens"]
    context_resets: int

    @property
    def ids(self) -> tuple[int, ...]:
        """Prompt followed by the generated tokens."""
        return self.prompt_ids + self.new_ids


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Set every logit below the k-th largest to -inf (ties with the k-th are kept)."""
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    kth = torch.topk(logits, k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < kth, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Keep the smallest most-likely set of tokens whose probability mass reaches ``p``."""
    if p >= 1.0:
        return logits
    sorted_logits, order = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits.float(), dim=-1)
    mass_before = torch.cumsum(probs, dim=-1) - probs
    remove_sorted = mass_before >= p  # the first token has mass_before == 0 and is always kept
    remove = torch.zeros_like(remove_sorted).scatter(-1, order, remove_sorted)
    return logits.masked_fill(remove, float("-inf"))


def choose_next_token(
    logits: torch.Tensor, config: SamplingConfig, generator: torch.Generator | None = None
) -> int:
    """Pick the next token id from last-position logits ``[vocab]``."""
    logits = logits.detach().float().cpu()
    if config.greedy:
        return int(torch.argmax(logits))  # first maximum: ties resolve to the lowest id
    scaled = top_p_filter(top_k_filter(logits / config.temperature, config.top_k), config.top_p)
    probs = torch.softmax(scaled, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator))


@torch.no_grad()
def generate(
    model: nn.Module,
    prompt_ids: Sequence[int],
    config: SamplingConfig,
    *,
    device: torch.device,
    context_length: int | None = None,
    stride: int | None = None,
    autocast_dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
    use_cache: bool = True,
) -> GenerationResult:
    """Generate up to ``config.max_new_tokens`` tokens after ``prompt_ids`` (batch size 1)."""
    ids = [int(t) for t in prompt_ids]
    if not ids:
        raise ValueError("prompt_ids must contain at least one token")
    prompt = tuple(ids)
    config_of_model = model_config(model)
    ctx = context_length if context_length is not None else config_of_model.context_length
    new: list[int] = []
    resets = 0
    stop: Literal["eot", "max_new_tokens"] = "max_new_tokens"
    if config.max_new_tokens == 0:
        return GenerationResult(prompt, (), stop, 0)

    def forward(tokens: list[int], cache: KVCache | None) -> torch.Tensor:
        batch = torch.tensor([tokens], dtype=torch.long, device=device)
        with autocast_context(device, autocast_dtype):
            logits: torch.Tensor = model(batch, cache=cache)
        return logits[0, -1]

    with eval_mode(model):
        cache = (
            KVCache.for_config(
                config_of_model, 1, cache_dtype(model, autocast_dtype), device, max_length=ctx
            )
            if use_cache
            else None
        )
        start = context_start(len(ids), ctx, stride)
        logits = forward(ids[start:], cache)
        while True:
            token = choose_next_token(logits, config, generator)
            if config.eot_id is not None and token == config.eot_id:
                stop = "eot"
                break
            ids.append(token)
            new.append(token)
            if len(new) == config.max_new_tokens:
                break
            next_start = context_start(len(ids), ctx, stride)
            if next_start != start:
                start = next_start
                resets += 1
                if cache is not None:
                    cache.reset()
                logits = forward(ids[start:], cache)
            elif cache is None:
                logits = forward(ids[start:], None)  # reference path: full forward every step
            else:
                logits = forward(ids[-1:], cache)
    return GenerationResult(prompt, tuple(new), stop, resets)
