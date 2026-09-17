"""Evaluation/generation CLIs and loader: every operational failure is reported, never a traceback.

Regression tests for the PR #5 review: invalid or unavailable devices, missing artifacts,
unwritable outputs, perplexity overflow in reports, and model configurations that differ from
the trained one without changing parameter shapes.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
import torch

from kittylm.inference.loading import LoadError, load_for_inference, resolve_device
from kittylm.tokenizer.bpe import BPETokenizer
from kittylm.training.checkpoint import read_checkpoint, write_checkpoint
from tests.conftest import load_script
from tests.test_end_to_end import FIXTURE, NANO, smoke_tokenizer, train_nano

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    tmp = tmp_path_factory.mktemp("cli")
    tokenizer: BPETokenizer = smoke_tokenizer()
    tokenizer_path = tmp / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    checkpoint = train_nano(tmp, tokenizer, steps=1)
    common = [
        "--model-config", str(NANO), "--set", f"vocab_size={tokenizer.vocab_size}",
        "--checkpoint", str(checkpoint), "--tokenizer", str(tokenizer_path),
    ]  # fmt: skip
    return {"tmp": tmp, "tokenizer": tokenizer_path, "checkpoint": checkpoint, "common": common}


def run(script: str, argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    code = load_script(script).main(argv)
    return code, capsys.readouterr().out


# --- devices --------------------------------------------------------------------------------------


def test_resolve_device() -> None:
    assert resolve_device("cpu") == CPU
    with pytest.raises(LoadError, match="invalid device"):
        resolve_device("not-a-device")
    with pytest.raises(LoadError, match="unsupported device type"):
        resolve_device("meta")
    with pytest.raises(LoadError, match="GPU"):
        resolve_device("cuda:99")  # no GPU in CI; index out of range on a GPU machine


@pytest.mark.parametrize("script", ["evaluate", "generate"])
@pytest.mark.parametrize("device", ["not-a-device", "cuda:99"])
def test_invalid_or_unavailable_device_is_a_handled_failure(
    artifacts: dict[str, Any], capsys: pytest.CaptureFixture[str], script: str, device: str
) -> None:
    extra = (
        ["--doc", f"prose={FIXTURE}"]
        if script == "evaluate"
        else ["--prompt", "hi", "--out-dir", str(artifacts["tmp"] / "samples")]
    )
    code, out = run(script, [*artifacts["common"], *extra, "--device", device], capsys)
    assert code == 1 and out.startswith(f"{script.replace('ate', 'ation')} failed:")


# --- files ----------------------------------------------------------------------------------------


def test_generate_missing_tokenizer_is_a_handled_failure(
    artifacts: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    argv = [a if a != str(artifacts["tokenizer"]) else str(artifacts["tmp"] / "missing.json")
            for a in artifacts["common"]]  # fmt: skip
    code, out = run(
        "generate", [*argv, "--prompt", "hi", "--out-dir", str(artifacts["tmp"] / "s")], capsys
    )
    assert code == 1 and out.startswith("generation failed:")


def test_generate_unwritable_output_directory_is_a_handled_failure(
    artifacts: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    blocker = artifacts["tmp"] / "a-file"
    blocker.write_text("not a directory", encoding="utf-8")
    code, out = run(
        "generate",
        [
            *artifacts["common"],
            "--prompt",
            "hi",
            "--max-new-tokens",
            "2",
            "--out-dir",
            str(blocker / "sub"),
        ],
        capsys,
    )
    assert code == 1 and out.startswith("generation failed:")


def test_evaluate_unwritable_report_path_is_a_handled_failure(
    artifacts: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    directory = artifacts["tmp"] / "report-is-a-directory"
    directory.mkdir()
    code, out = run(
        "evaluate",
        [*artifacts["common"], "--doc", f"prose={FIXTURE}", "--out", str(directory)],
        capsys,
    )
    assert code == 1 and out.startswith("evaluation failed:")
    blocker = artifacts["tmp"] / "report-parent-file"
    blocker.write_text("x", encoding="utf-8")
    code, out = run(
        "evaluate",
        [*artifacts["common"], "--doc", f"prose={FIXTURE}", "--out", str(blocker / "r.json")],
        capsys,
    )
    assert code == 1 and out.startswith("evaluation failed:")


def test_evaluate_missing_checkpoint_is_a_handled_failure(
    artifacts: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    argv = [a if a != str(artifacts["checkpoint"]) else str(artifacts["tmp"] / "none.pt")
            for a in artifacts["common"]]  # fmt: skip
    code, out = run("evaluate", [*argv, "--doc", f"prose={FIXTURE}"], capsys)
    assert code == 1 and out.startswith("evaluation failed:")


# --- perplexity overflow in the report ------------------------------------------------------------


def test_evaluate_reports_infinite_perplexity_for_huge_losses(
    artifacts: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    state = read_checkpoint(artifacts["checkpoint"])
    for key in ("tok_embeddings.weight", "lm_head.weight"):  # tied: both copies are loaded
        state["model"][key] = state["model"][key] * 1e5
    blown_up = artifacts["tmp"] / "blown-up.pt"
    write_checkpoint(blown_up, state)
    argv = [a if a != str(artifacts["checkpoint"]) else str(blown_up) for a in artifacts["common"]]
    out_path = artifacts["tmp"] / "blown-up.json"
    code, out = run(
        "evaluate", [*argv, "--doc", f"prose={FIXTURE}", "--out", str(out_path)], capsys
    )
    assert code == 0, out
    report = json.loads(out_path.read_text(encoding="utf-8"))  # strict JSON: no bare Infinity
    assert report["loss"] > 709.78 and report["ppl"] == "inf"


# --- model configuration identity -----------------------------------------------------------------


@pytest.mark.parametrize("override", ["rope_theta=500000.0", "norm_eps=0.001"])
def test_shape_compatible_config_changes_are_refused(
    artifacts: dict[str, Any], override: str
) -> None:
    vocab = next(a for a in artifacts["common"] if a.startswith("vocab_size="))
    with pytest.raises(LoadError, match="model_config_sha256 mismatch"):
        load_for_inference(
            NANO, artifacts["checkpoint"], artifacts["tokenizer"], CPU, overrides=[vocab, override]
        )
    loaded = load_for_inference(
        NANO, artifacts["checkpoint"], artifacts["tokenizer"], CPU, overrides=[vocab]
    )
    assert loaded.global_step == 1
