from __future__ import annotations

import hashlib
import json

import pytest

from kittylm.data.secrets import RULE_NAMES, Finding, scan_text
from tests.fakes import ALL_FAKES, fake_env_assignment


def test_every_rule_has_a_fake_and_is_known() -> None:
    covered = set(ALL_FAKES)
    assert covered <= set(RULE_NAMES)
    assert covered == set(RULE_NAMES), f"rules without a test fake: {set(RULE_NAMES) - covered}"


@pytest.mark.parametrize(("rule", "secret"), sorted(ALL_FAKES.items()))
def test_each_fake_secret_is_detected_by_its_rule(rule: str, secret: str) -> None:
    text = f"line one\nprefix {secret} suffix\nline three\n"
    findings = scan_text(text)
    assert Finding(line=2, rule=rule) in findings


def test_unquoted_env_style_assignment_is_detected() -> None:
    findings = scan_text("# config\n" + fake_env_assignment() + "\n")
    assert Finding(line=2, rule="credential_assignment") in findings


@pytest.mark.parametrize(("rule", "secret"), sorted(ALL_FAKES.items()))
def test_findings_never_reveal_the_secret_value(rule: str, secret: str) -> None:
    findings = scan_text(secret)
    assert findings
    rendered = repr(findings) + str(findings) + json.dumps([vars_of(f) for f in findings])
    # The core of each fake (without surrounding syntax) must not leak.
    core = secret.split('"')[1] if '"' in secret else secret
    assert core not in rendered


def vars_of(finding: Finding) -> dict[str, object]:
    return {"line": finding.line, "rule": finding.rule}


@pytest.mark.parametrize(
    "benign",
    [
        hashlib.sha256(b"kittylm").hexdigest(),  # hashes are not secrets
        'digest = "' + hashlib.sha256(b"x").hexdigest() + '"',
        'EOT_TOKEN = "<|endoftext|>"',  # special-token literals are not credentials
        'api_key = "your-api-key-here"',  # placeholder
        "password = os.environ['SERVICE_PASSWORD']",  # indirection, no literal
        "pip install scikit-learn  # sk-learn-contrib-extras-and-more",
        "max_tokens = 1024",
        'name = "ThisIsALongDescriptiveIdentifierName"',
    ],
)
def test_benign_text_is_not_flagged(benign: str) -> None:
    assert scan_text(benign) == []


def test_scan_is_deterministic_and_sorted() -> None:
    text = "\n".join(ALL_FAKES.values())
    first = scan_text(text)
    assert first == scan_text(text)
    assert first == sorted(first)
