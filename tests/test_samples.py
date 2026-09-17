"""Generated samples are secret-scanned and redacted before anything is written."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kittylm.data.secrets import Finding, scan_text
from kittylm.inference import samples
from kittylm.inference.samples import SampleSecretError, redact_secrets, save_sample
from tests.fakes import fake_aws_key_id, fake_github_token, fake_private_key_header


def test_clean_text_is_saved_unchanged(tmp_path: Path) -> None:
    text = "def add(a, b):\n    return a + b\n"
    saved = save_sample(tmp_path / "samples", "clean", text)
    assert saved.path.read_text(encoding="utf-8") == text
    assert saved.redacted_lines == () and saved.rules == ()
    sidecar = json.loads((tmp_path / "samples" / "clean.json").read_text(encoding="utf-8"))
    assert sidecar == {"redacted_lines": [], "rules": [], "secret_scan": "clean"}


def test_secret_lines_are_redacted_before_writing(tmp_path: Path) -> None:
    token, key_id, header = fake_github_token(), fake_aws_key_id(), fake_private_key_header()
    text = "\n".join(["intro line", f"token = {token}", "middle", f"aws {key_id}", header, "end"])
    assert scan_text(text)  # the fixture really contains findings
    saved = save_sample(tmp_path, "leaky", text)
    written = saved.path.read_text(encoding="utf-8")
    for secret in (token, key_id, header):
        assert secret not in written
    assert scan_text(written) == []
    lines = written.split("\n")
    assert lines[0] == "intro line" and lines[2] == "middle" and lines[5] == "end"
    assert all(lines[i].startswith("[REDACTED by secret scan: ") for i in (1, 3, 4))
    assert saved.redacted_lines == (2, 4, 5)
    sidecar_text = (tmp_path / "leaky.json").read_text(encoding="utf-8")
    assert token not in sidecar_text and json.loads(sidecar_text)["redacted_lines"] == [2, 4, 5]
    # Nothing but the two final files remains: the raw text never touched disk.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["leaky.json", "leaky.txt"]


def test_redaction_markers_never_contain_values() -> None:
    token = fake_github_token()
    clean, lines, rules = redact_secrets(f"key: {token}")
    assert token not in clean and lines == (1,) and rules
    assert all(rule in clean for rule in rules)


def test_unredactable_text_is_refused_and_nothing_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(samples, "scan_text", lambda text: [Finding(line=1, rule="stubborn")])
    with pytest.raises(SampleSecretError, match="still flagged"):
        save_sample(tmp_path / "out", "stubborn", "anything")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("name", ["", "../escape", "a/b", "a\\b", ".hidden", "x" * 200, "a..b"])
def test_sample_names_cannot_escape_the_directory(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match="invalid sample name"):
        save_sample(tmp_path, name, "text")
    assert list(tmp_path.iterdir()) == []
