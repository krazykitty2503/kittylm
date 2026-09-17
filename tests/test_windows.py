"""Windowing policy (D-023): every target scored once, contexts shared with generation."""

from __future__ import annotations

import itertools

import pytest

from kittylm.config import from_dict, load_yaml
from kittylm.evaluation.windows import (
    Window,
    context_start,
    default_stride,
    evaluation_windows,
)
from kittylm.model.config import ModelConfig
from kittylm.tokenizer.trainer import TokenizerConfig, train_from_config
from tests.conftest import ROOT


def smoke_fixture_ids() -> list[int]:
    data = {
        k: v for k, v in load_yaml(ROOT / "configs/tokenizer/smoke.yaml").items() if k != "kind"
    }
    tokenizer = train_from_config(from_dict(TokenizerConfig, data), ROOT)
    text = (ROOT / "tests/fixtures/smoke_corpus.txt").read_text(encoding="utf-8")
    return tokenizer.encode(text)


def nano_context() -> int:
    data = {k: v for k, v in load_yaml(ROOT / "configs/model/nano.yaml").items() if k != "kind"}
    return from_dict(ModelConfig, data).context_length


def test_smoke_fixture_policy_for_nano_context() -> None:
    # The measured mismatch this policy exists for: 477 fixture tokens vs a 256-token context.
    ids = smoke_fixture_ids()
    context = nano_context()
    assert (len(ids), context) == (477, 256)
    windows = evaluation_windows(len(ids), context)
    assert default_stride(context) == 128
    assert windows == [
        Window(start=0, end=257, score_from=1),
        Window(start=128, end=385, score_from=257),
        Window(start=256, end=477, score_from=385),
    ]
    assert [w.scored_targets for w in windows] == [256, 128, 92]
    assert sum(w.scored_targets for w in windows) == len(ids) - 1
    assert [w.input_length for w in windows] == [256, 256, 220]
    # Generation of the fixture from a 16-token prompt resets its context at exactly these starts.
    starts = sorted({context_start(t, context) for t in range(16, len(ids))})
    assert starts == [0, 128, 256]


CASES = list(itertools.product([2, 3, 17, 33, 100, 257, 477], [1, 4, 16, 256], [None, 1, 3, 8, 16]))


@pytest.mark.parametrize(("n_tokens", "context", "stride"), CASES)
def test_every_target_scored_exactly_once_with_shared_contexts(
    n_tokens: int, context: int, stride: int | None
) -> None:
    if stride is not None and stride > context:
        with pytest.raises(ValueError, match="stride"):
            evaluation_windows(n_tokens, context, stride)
        return
    windows = evaluation_windows(n_tokens, context, stride)
    scored = [t for w in windows for t in range(w.score_from, w.end)]
    assert scored == list(range(1, n_tokens))  # once each, in order
    for window in windows:
        assert 0 <= window.start < window.score_from <= window.end <= n_tokens
        assert window.input_length <= context
        assert window.score_offset == window.score_from - window.start - 1
        for target in range(window.score_from, window.end):
            assert context_start(target, context, stride) == window.start
            assert target - window.start <= context
    assert windows[-1].end == n_tokens
    # Deterministic: the policy depends only on (n_tokens, context, stride).
    assert evaluation_windows(n_tokens, context, stride) == windows


def test_stream_that_fits_uses_one_window() -> None:
    assert evaluation_windows(10, 16) == [Window(0, 10, 1)]
    assert evaluation_windows(17, 16) == [Window(0, 17, 1)]
    assert context_start(16, 16) == 0 and context_start(17, 16) == 8


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: evaluation_windows(1, 16), "at least 2 tokens"),
        (lambda: evaluation_windows(10, 0), "context_length"),
        (lambda: evaluation_windows(10, 16, 0), "stride"),
        (lambda: evaluation_windows(10, 16, 17), "stride"),
        (lambda: context_start(0, 16), "position"),
    ],
)
def test_invalid_arguments(call: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()  # type: ignore[operator]
