"""End-to-end CPU pipeline with outbound network access blocked (plan rev 3.3 section 5).

Real smoke tokenizer -> Nano model (vocab 512) -> training engine -> checkpoint -> evaluation
CLI (loss, ppl, bpb per category, inference speed) -> generation CLI (sampling, secret-scanned
sample) -> overfit-gate computation, all while every socket connection attempt fails loudly.

This is defense-in-depth: it proves the tested paths make no network requests, not that no code
path ever could (docs/safety.md).
"""

from __future__ import annotations

import json
import math
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from kittylm.config import from_dict, load_yaml
from kittylm.data.loader import TOKEN_DTYPE, TrainWindowSampler, open_token_file
from kittylm.evaluation.gates import evaluate_overfit_gate
from kittylm.evaluation.perplexity import stream_nll, token_byte_lengths
from kittylm.inference.loading import LoadError, load_for_inference
from kittylm.model.config import ModelConfig
from kittylm.model.transformer import KittyLM
from kittylm.tokenizer.bpe import BPETokenizer
from kittylm.tokenizer.trainer import TokenizerConfig, train_from_config
from kittylm.training.checkpoint import RunIdentity, read_checkpoint, write_checkpoint
from kittylm.training.config import TrainingConfig
from kittylm.training.determinism import seed_everything
from kittylm.training.engine import RunInfo, TrainingEngine
from tests.conftest import ROOT, load_script
from tests.training_helpers import COMMIT

CPU = torch.device("cpu")
FIXTURE = ROOT / "tests" / "fixtures" / "smoke_corpus.txt"
NANO = ROOT / "configs" / "model" / "nano.yaml"


class NetworkAccessError(AssertionError):
    """Raised by the socket guard: the code under test tried to open a connection."""


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    attempts: list[str] = []

    def refuse(name: str) -> Any:
        def blocked(*args: Any, **kwargs: Any) -> Any:
            attempts.append(name)
            raise NetworkAccessError(f"outbound network access attempted via {name}")

        return blocked

    for name in ("connect", "connect_ex", "sendto"):
        monkeypatch.setattr(socket.socket, name, refuse(f"socket.{name}"))
    monkeypatch.setattr(socket, "create_connection", refuse("create_connection"))
    monkeypatch.setattr(socket, "getaddrinfo", refuse("getaddrinfo"))
    yield attempts


def test_socket_guard_blocks_connections(no_network: list[str]) -> None:
    with pytest.raises(NetworkAccessError):
        socket.create_connection(("127.0.0.1", 9), timeout=0.1)
    with (
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock,
        pytest.raises(NetworkAccessError),
    ):
        sock.connect(("127.0.0.1", 9))
    with pytest.raises(NetworkAccessError):
        socket.getaddrinfo("example.invalid", 443)
    assert no_network == ["create_connection", "socket.connect", "getaddrinfo"]


def smoke_tokenizer() -> BPETokenizer:
    data = {
        k: v for k, v in load_yaml(ROOT / "configs/tokenizer/smoke.yaml").items() if k != "kind"
    }
    return train_from_config(from_dict(TokenizerConfig, data), ROOT)


def train_nano(tmp_path: Path, tokenizer: BPETokenizer, steps: int) -> Path:
    ids = tokenizer.encode(FIXTURE.read_text(encoding="utf-8"))
    token_file = tmp_path / "fixture.bin"
    np.asarray(ids, dtype=TOKEN_DTYPE).tofile(token_file)
    data = {k: v for k, v in load_yaml(NANO).items() if k != "kind"}
    model_config = from_dict(ModelConfig, {**data, "vocab_size": tokenizer.vocab_size})
    training = TrainingConfig(
        seed=11,
        batch_size=2,
        gradient_accumulation=1,
        max_steps=steps,
        learning_rate=3e-3,
        min_learning_rate=3e-4,
        warmup_steps=1,
        precision="fp32",
        determinism="deterministic",
        log_every=1,
        timing_sync_every=0,
        cpu_threads=2,
    )
    seed_everything(training.seed)
    engine = TrainingEngine(
        model=KittyLM(model_config),
        model_config=model_config,
        training_config=training,
        sampler=TrainWindowSampler(
            open_token_file(token_file),
            context_length=model_config.context_length,
            batch_size=training.batch_size,
            seed=training.seed,
        ),
        device=CPU,
        run_dir=tmp_path / "run",
        identity=RunIdentity(
            config_hash="0" * 64,
            tokenizer_sha256=tokenizer.sha256,
            dataset_version="1" * 64,
        ),
        run=RunInfo(kind="engineering", experiment_id="e2e", git_commit=COMMIT, git_dirty=False),
    )
    engine.train(steps)
    path = engine.save_checkpoint()
    engine.close()
    return path


def test_pipeline_runs_offline(
    tmp_path: Path, no_network: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    tokenizer = smoke_tokenizer()
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    checkpoint = train_nano(tmp_path, tokenizer, steps=3)
    code_doc = tmp_path / "snippet.py"
    code_doc.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    common = [
        "--model-config", str(NANO), "--set", f"vocab_size={tokenizer.vocab_size}",
        "--checkpoint", str(checkpoint), "--tokenizer", str(tokenizer_path),
    ]  # fmt: skip

    # Evaluation CLI: windowed loss/ppl/bpb per category and batch-1 speed.
    report_path = tmp_path / "eval.json"
    evaluate = load_script("evaluate")
    code = evaluate.main(
        [*common, "--doc", f"prose={FIXTURE}", "--doc", f"code={code_doc}",
         "--speed", "32,16", "--out", str(report_path)]
    )  # fmt: skip
    assert code == 0, capsys.readouterr().out
    capsys.readouterr()  # discard the evaluation printout
    report = json.loads(report_path.read_text(encoding="utf-8"))
    fixture_ids = tokenizer.encode(FIXTURE.read_text(encoding="utf-8"))
    code_ids = tokenizer.encode(code_doc.read_text(encoding="utf-8"))
    assert len(fixture_ids) == 477
    assert report["tokens"] == (len(fixture_ids) - 1) + (len(code_ids) - 1)
    assert report["ppl"] == math.exp(report["loss"])
    assert set(report["bpb_by_category"]) == {"code", "prose"}
    assert report["global_step"] == 3 and report["tokenizer_sha256"] == tokenizer.sha256
    assert report["inference"]["prefill_tok_s"] > 0 and report["inference"]["decode_tok_s"] > 0

    # The CLI numbers equal an in-process evaluation of the same checkpoint.
    loaded = load_for_inference(
        NANO, checkpoint, tokenizer_path, CPU, overrides=[f"vocab_size={tokenizer.vocab_size}"]
    )
    prose = stream_nll(
        loaded.model, fixture_ids, token_byte_lengths(tokenizer, fixture_ids), device=CPU
    )
    assert prose.bits_per_byte == pytest.approx(report["bpb_by_category"]["prose"], rel=1e-9)
    first_token_bytes = len(tokenizer.token_bytes(fixture_ids[0]))
    assert prose.bytes == len(FIXTURE.read_bytes()) - first_token_bytes  # token 0 is not predicted

    # Generation CLI: sampled continuation, secret-scanned sample written under the out dir.
    generate = load_script("generate")
    samples_dir = tmp_path / "samples"
    code = generate.main(
        [*common, "--prompt", "KittyLM", "--max-new-tokens", "300", "--temperature", "0.8",
         "--top-k", "40", "--top-p", "0.95", "--seed", "3", "--out-dir", str(samples_dir),
         "--name", "e2e"]
    )  # fmt: skip
    output = capsys.readouterr().out
    assert code == 0, output
    summary = json.loads(output[output.index("{") :])
    assert summary["sample"] == "e2e.txt" and summary["stop_reason"] in {"eot", "max_new_tokens"}
    assert summary["context_resets"] >= 0
    assert (samples_dir / "e2e.txt").exists() and (samples_dir / "e2e.json").exists()

    # Overfit-gate computation on the fixture (3 steps: measured, expected to fail).
    gate = evaluate_overfit_gate(
        loaded.model,
        fixture_ids,
        token_byte_lengths(tokenizer, fixture_ids),
        device=CPU,
        metrics_records=[
            json.loads(line)
            for line in (tmp_path / "run" / "metrics.jsonl").read_text("utf-8").splitlines()
        ],
    )
    assert gate.scored_tokens == 476 and gate.reference_tokens == gate.generated_tokens == 461
    assert not gate.passed and "final_loss_below_max" in gate.failures
    assert gate.checks["all_values_finite"]

    assert no_network == []  # nothing even attempted a connection


def test_loader_refuses_mismatched_artifacts(tmp_path: Path, no_network: list[str]) -> None:
    tokenizer = smoke_tokenizer()
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    checkpoint = train_nano(tmp_path, tokenizer, steps=1)
    vocab = [f"vocab_size={tokenizer.vocab_size}"]

    smaller = tokenizer.truncated(384)
    smaller_path = tmp_path / "smaller.json"
    smaller.save(smaller_path)
    with pytest.raises(LoadError, match="vocab_size"):
        load_for_inference(NANO, checkpoint, smaller_path, CPU, overrides=vocab)

    other = replace_corpus_tokenizer(tmp_path)
    with pytest.raises(LoadError, match="different tokenizer"):
        load_for_inference(NANO, checkpoint, other, CPU, overrides=vocab)

    with pytest.raises(LoadError, match="kind: model"):
        load_for_inference(
            ROOT / "configs/tokenizer/smoke.yaml", checkpoint, tokenizer_path, CPU, overrides=vocab
        )
    with pytest.raises(LoadError, match="model_config_sha256 mismatch"):
        load_for_inference(NANO, checkpoint, tokenizer_path, CPU, overrides=[*vocab, "n_layers=2"])
    state = read_checkpoint(checkpoint)
    del state["model"]["norm.weight"]  # same configuration identity, incomplete weights
    incomplete = tmp_path / "incomplete.pt"
    write_checkpoint(incomplete, state)
    with pytest.raises(LoadError, match="do not fit"):
        load_for_inference(NANO, incomplete, tokenizer_path, CPU, overrides=vocab)

    generate = load_script("generate")
    assert generate.output_directory_problem(ROOT / "docs") is not None
    assert generate.output_directory_problem(ROOT / "runs" / "samples") is None
    assert generate.output_directory_problem(tmp_path) is None
    assert no_network == []


def replace_corpus_tokenizer(tmp_path: Path) -> Path:
    """A 512-token tokenizer with the same layout but trained on a different corpus."""
    corpus = tmp_path / "other_corpus.txt"
    corpus.write_text(FIXTURE.read_text(encoding="utf-8")[::-1], encoding="utf-8")
    data = {
        k: v for k, v in load_yaml(ROOT / "configs/tokenizer/smoke.yaml").items() if k != "kind"
    }
    config = from_dict(TokenizerConfig, {**data, "corpus_files": [corpus.name]})
    path = tmp_path / "other.json"
    train_from_config(config, tmp_path).save(path)
    return path
