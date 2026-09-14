from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pytest

from kittylm.config import (
    CONFIG_KINDS,
    ConfigError,
    apply_overrides,
    canonical_json,
    config_hash,
    from_dict,
    load_yaml,
    register_config_kind,
    to_dict,
    validate_config_tree,
)


@dataclass(frozen=True)
class Optim:
    lr: float
    betas: tuple[float, float] = (0.9, 0.95)


@dataclass(frozen=True)
class Toy:
    name: str
    steps: int
    optim: Optim
    precision: Literal["bf16", "fp32"] = "bf16"
    tags: list[str] = field(default_factory=list)
    extra: dict[str, int] = field(default_factory=dict)
    note: str | None = None


GOOD = {"name": "toy", "steps": 10, "optim": {"lr": 0.001}, "tags": ["a"], "extra": {"k": 1}}


def test_round_trip() -> None:
    cfg = from_dict(Toy, GOOD)
    assert cfg.optim.betas == (0.9, 0.95)
    assert from_dict(Toy, to_dict(cfg)) == cfg


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({**GOOD, "stpes": 3}, "unknown key(s) ['stpes']"),
        ({**GOOD, "optim": {"lr": 0.001, "lr_typo": 1.0}}, "optim.lr_typo"),
        ({k: v for k, v in GOOD.items() if k != "steps"}, "missing required key 'steps'"),
        ({**GOOD, "steps": True}, "steps: expected int"),
        ({**GOOD, "steps": 1.5}, "steps: expected int"),
        ({**GOOD, "optim": {"lr": "3e-4"}}, "write 3.0e-4"),
        ({**GOOD, "precision": "fp16"}, "expected one of"),
        ({**GOOD, "tags": "a"}, "tags: expected list"),
        ({**GOOD, "extra": {"k": "1"}}, "extra.k: expected int"),
    ],
)
def test_strictness(data: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigError) as exc:
        from_dict(Toy, data)
    assert message in str(exc.value)


def test_int_is_accepted_for_float() -> None:
    assert from_dict(Toy, {**GOOD, "optim": {"lr": 1}}).optim.lr == 1.0


def test_load_yaml_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "dup.yaml"
    path.write_text("steps: 1\nsteps: 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="duplicate key 'steps' at line 2"):
        load_yaml(path)


def test_load_yaml_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="top level must be a mapping"):
        load_yaml(path)


def test_overrides_parse_numbers_and_nested_keys() -> None:
    out = apply_overrides(GOOD, ["optim.lr=3e-4", "steps=20", "note=hello", "tags=[x, y]"])
    cfg = from_dict(Toy, out)
    assert cfg.optim.lr == pytest.approx(3e-4)
    assert (cfg.steps, cfg.note, cfg.tags) == (20, "hello", ["x", "y"])
    assert GOOD["steps"] == 10  # input not mutated


def test_override_typo_is_caught_by_schema() -> None:
    with pytest.raises(ConfigError, match="optim.lrr"):
        from_dict(Toy, apply_overrides(GOOD, ["optim.lrr=0.1"]))


def test_malformed_override() -> None:
    with pytest.raises(ConfigError, match="a.b=value"):
        apply_overrides(GOOD, ["steps"])


def test_hash_ignores_key_order() -> None:
    a = {"x": 1, "y": {"b": 2, "a": 1}}
    b = {"y": {"a": 1, "b": 2}, "x": 1}
    assert canonical_json(a) == canonical_json(b)
    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash({**a, "x": 2})


def test_validate_config_tree(tmp_path: Path) -> None:
    assert validate_config_tree(tmp_path / "missing") == (0, [])
    (tmp_path / "ok.yaml").write_text(
        "kind: toy-test\nname: t\nsteps: 1\noptim: {lr: 0.1}\n", encoding="utf-8"
    )
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "bad.yml").write_text("kind: nope\n", encoding="utf-8")
    (tmp_path / "nokind.yaml").write_text("name: x\n", encoding="utf-8")
    register_config_kind("toy-test", Toy)
    try:
        count, errors = validate_config_tree(tmp_path)
    finally:
        CONFIG_KINDS.pop("toy-test")
    assert count == 3
    assert len(errors) == 2
    assert any("unknown config kind 'nope'" in e for e in errors)
    assert any("missing top-level 'kind" in e for e in errors)
