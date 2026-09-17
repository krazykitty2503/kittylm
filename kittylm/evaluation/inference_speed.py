"""Batch-1 inference throughput: prefill and decode tokens per second.

Purpose:
    Measure the two phases of generation separately, because they behave very differently:

    - **prefill** processes the whole prompt in one forward pass (parallel over positions);
    - **decode** produces one token per forward pass using the KV cache (sequential).

    GPU kernels run asynchronously, so every timer boundary synchronizes the device; otherwise a
    timer would measure how fast work is *queued*, not how fast it runs. Warmup repetitions are
    excluded, and the median over repeats is reported to damp scheduling noise.

Public API:
    InferenceSpeedResult(prompt_tokens, new_tokens, repeats, prefill_tok_s, decode_tok_s,
                         prefill_seconds, decode_seconds)
        ``to_ledger()`` -> kittylm.ledger.InferenceSpeed
    measure_inference_speed(model, *, prompt_tokens, new_tokens, device, autocast_dtype=None,
                            warmup=1, repeats=3, seed=0) -> InferenceSpeedResult

Shapes:
    Prompt ``[1, prompt_tokens]``; each decode step ``[1, 1]``; logits ``[1, T, vocab]``.

Dtype:
    The model runs under ``autocast_dtype`` when given; the KV cache matches it.

Device:
    Prompts, cache and decoding stay on ``device``; the next token is chosen with an on-device
    argmax, so decoding does not force a host synchronization per token.

Math:
    ``prefill_tok_s = prompt_tokens / median(prefill_seconds)``;
    ``decode_tok_s = new_tokens / median(decode_seconds)``.

Invariants:
    - Measurements always use batch size 1 and a fresh cache per repetition.
    - ``prompt_tokens + new_tokens <= context_length``, so no context reset is timed.
    - The prompt is seeded random token ids: throughput does not depend on token values.

Failure modes:
    - Zero or negative sizes, ``repeats < 1``, ``warmup < 0`` or a prompt plus decode length
      beyond the context raise ValueError.
    - Measured numbers are machine-, driver- and load-dependent; they are measurements, not
      properties of the model (see docs/limitations.md).

See:
    Plan rev 3.3 section 5; kittylm/model/kv_cache.py; kittylm/inference/runtime.py.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import torch
from torch import nn

from kittylm.inference.runtime import (
    autocast_context,
    cache_dtype,
    eval_mode,
    model_config,
    synchronize,
)
from kittylm.ledger import InferenceSpeed
from kittylm.model.kv_cache import KVCache

__all__ = ["InferenceSpeedResult", "measure_inference_speed"]


@dataclass(frozen=True)
class InferenceSpeedResult:
    """Median batch-1 prefill and decode throughput over ``repeats`` timed runs."""

    prompt_tokens: int
    new_tokens: int
    repeats: int
    prefill_tok_s: float
    decode_tok_s: float
    prefill_seconds: tuple[float, ...]
    decode_seconds: tuple[float, ...]

    def to_ledger(self) -> InferenceSpeed:
        """The record section for experiment ledgers."""
        return InferenceSpeed(prefill_tok_s=self.prefill_tok_s, decode_tok_s=self.decode_tok_s)


@torch.no_grad()
def measure_inference_speed(
    model: nn.Module,
    *,
    prompt_tokens: int,
    new_tokens: int,
    device: torch.device,
    autocast_dtype: torch.dtype | None = None,
    warmup: int = 1,
    repeats: int = 3,
    seed: int = 0,
) -> InferenceSpeedResult:
    """Time prefill of a ``prompt_tokens`` prompt and ``new_tokens`` cached decode steps."""
    if prompt_tokens < 1 or new_tokens < 1:
        raise ValueError("prompt_tokens and new_tokens must be >= 1")
    if repeats < 1 or warmup < 0:
        raise ValueError("repeats must be >= 1 and warmup >= 0")
    config = model_config(model)
    if prompt_tokens + new_tokens > config.context_length:
        raise ValueError(
            f"prompt_tokens + new_tokens = {prompt_tokens + new_tokens} exceeds context_length "
            f"{config.context_length}; context resets are not timed"
        )
    generator = torch.Generator().manual_seed(seed)
    prompt = torch.randint(0, config.vocab_size, (1, prompt_tokens), generator=generator).to(device)
    dtype = cache_dtype(model, autocast_dtype)
    prefill: list[float] = []
    decode: list[float] = []
    with eval_mode(model):
        for run in range(warmup + repeats):
            cache = KVCache.for_config(config, 1, dtype, device)
            synchronize(device)
            t0 = time.perf_counter()
            with autocast_context(device, autocast_dtype):
                logits = model(prompt, cache=cache)
            synchronize(device)
            t1 = time.perf_counter()
            with autocast_context(device, autocast_dtype):
                for _ in range(new_tokens):
                    token = logits[:, -1].argmax(dim=-1, keepdim=True)
                    logits = model(token, cache=cache)
            synchronize(device)
            t2 = time.perf_counter()
            if run >= warmup:
                prefill.append(t1 - t0)
                decode.append(t2 - t1)
    return InferenceSpeedResult(
        prompt_tokens=prompt_tokens,
        new_tokens=new_tokens,
        repeats=repeats,
        prefill_tok_s=prompt_tokens / statistics.median(prefill),
        decode_tok_s=new_tokens / statistics.median(decode),
        prefill_seconds=tuple(prefill),
        decode_seconds=tuple(decode),
    )
