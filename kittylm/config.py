"""Strict, typed configuration loading for KittyLM.

Purpose:
    Turn YAML into typed dataclasses and refuse anything ambiguous. In research code a
    silently ignored typo (``learnig_rate: 1e-3``) produces a run that *looks* valid but
    isn't the experiment you meant. Every ambiguity is therefore an error (D-011).

Public API:
    ConfigError
        Raised for any invalid configuration; the message names the dotted key path.
    load_yaml(path) -> dict
        Safe YAML load that rejects duplicate keys and non-mapping documents.
    from_dict(cls, data) -> cls
        Build a dataclass recursively; unknown keys, missing keys and type mismatches fail.
    to_dict(obj) -> dict
        Plain-data snapshot of a dataclass (tuples become lists).
    apply_overrides(data, ["train.lr=0.001", ...]) -> dict
        Dotted ``key=value`` overrides for ablations; values are parsed as numbers/YAML.
    canonical_json(obj) -> str, config_hash(obj) -> str
        Deterministic serialization and its sha256, used to identify resolved configs.
    register_config_kind(kind, cls), validate_config_tree(root)
        Every YAML file under ``configs/`` declares ``kind:``; CI validates each one.
    CONFIG_KIND_MODULES, load_config_kinds()
        Modules that register kinds; imported lazily so ``config`` has no import cycles.

Invariants:
    - ``from_dict(cls, to_dict(x)) == x`` for supported dataclasses.
    - ``config_hash`` depends only on content, never on key order or file formatting.
    - bool is never accepted where int or float is expected; int is accepted for float.

Failure modes:
    - YAML 1.1 parses ``3e-4`` (no dot) as a string; a float field then fails with a hint to
      write ``3.0e-4``. Overrides parse ``3e-4`` as a float.
    - Unsupported annotation types raise ConfigError at load time, not silently.

See:
    D-011 (strict configuration), plan rev 3.1 section 7 (config validation in CI).
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

import yaml

__all__ = [
    "CONFIG_KINDS",
    "CONFIG_KIND_MODULES",
    "load_config_kinds",
    "ConfigError",
    "apply_overrides",
    "canonical_json",
    "config_hash",
    "from_dict",
    "load_yaml",
    "register_config_kind",
    "to_dict",
    "validate_config_tree",
]


class ConfigError(ValueError):
    """Invalid configuration."""


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys."""


def _construct_strict_mapping(
    loader: _StrictLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            line = key_node.start_mark.line + 1
            raise ConfigError(f"duplicate key {key!r} at line {line}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_strict_mapping
)


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping strictly.

    Args:
        path: File to load.

    Raises:
        ConfigError: On syntax errors, duplicate keys, or a non-mapping document.
    """
    try:
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=_StrictLoader)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path.name}: invalid YAML: {exc}") from exc
    except ConfigError as exc:
        raise ConfigError(f"{path.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path.name}: top level must be a mapping")
    return data


def _type_name(tp: Any) -> str:
    return getattr(tp, "__name__", None) or str(tp)


def _convert(value: Any, tp: Any, path: str) -> Any:
    origin = get_origin(tp)
    args = get_args(tp)
    where = path or "<root>"

    if tp is Any:
        return value
    if origin is Union or origin is types.UnionType:
        if value is None and type(None) in args:
            return None
        errors = []
        for arg in args:
            if arg is type(None):
                continue
            try:
                return _convert(value, arg, path)
            except ConfigError as exc:
                errors.append(str(exc))
        raise ConfigError(f"{where}: value {value!r} matches none of {args}: {errors}")
    if origin is Literal:
        if value in args and type(value) in {type(a) for a in args}:
            return value
        raise ConfigError(f"{where}: expected one of {list(args)}, got {value!r}")
    if dataclasses.is_dataclass(tp) and isinstance(tp, type):
        if not isinstance(value, Mapping):
            raise ConfigError(f"{where}: expected mapping for {tp.__name__}")
        return from_dict(tp, value, path=path)
    if origin in (list, Sequence) or tp is list:
        if not isinstance(value, list):
            raise ConfigError(f"{where}: expected list, got {type(value).__name__}")
        (item_tp,) = args or (Any,)
        return [_convert(v, item_tp, f"{path}[{i}]") for i, v in enumerate(value)]
    if origin is tuple:
        if not isinstance(value, list | tuple):
            raise ConfigError(f"{where}: expected list, got {type(value).__name__}")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_convert(v, args[0], f"{path}[{i}]") for i, v in enumerate(value))
        if len(args) != len(value):
            raise ConfigError(f"{where}: expected {len(args)} items, got {len(value)}")
        pairs = enumerate(zip(value, args, strict=True))
        return tuple(_convert(v, a, f"{path}[{i}]") for i, (v, a) in pairs)
    if origin in (dict, Mapping) or tp is dict:
        if not isinstance(value, Mapping):
            raise ConfigError(f"{where}: expected mapping, got {type(value).__name__}")
        key_tp, val_tp = args or (str, Any)
        return {
            _convert(k, key_tp, f"{path}.<key>"): _convert(v, val_tp, f"{path}.{k}")
            for k, v in value.items()
        }
    if tp is bool:
        if isinstance(value, bool):
            return value
        raise ConfigError(f"{where}: expected bool, got {value!r}")
    if tp is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        raise ConfigError(f"{where}: expected int, got {value!r}")
    if tp is float:
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
        hint = (
            " (YAML needs a dot in exponents: write 3.0e-4, not 3e-4)"
            if isinstance(value, str)
            else ""
        )
        raise ConfigError(f"{where}: expected float, got {value!r}{hint}")
    if tp is str:
        if isinstance(value, str):
            return value
        raise ConfigError(f"{where}: expected str, got {value!r}")
    if tp is type(None):
        if value is None:
            return None
        raise ConfigError(f"{where}: expected null, got {value!r}")
    raise ConfigError(f"{where}: unsupported annotation {_type_name(tp)}")


def from_dict[T](cls: type[T], data: Mapping[str, Any], *, path: str = "") -> T:
    """Construct dataclass ``cls`` from plain data, strictly.

    Args:
        cls: Target dataclass type.
        data: Mapping of field names to values.
        path: Dotted path prefix used in error messages.

    Raises:
        ConfigError: On unknown keys, missing required keys, or type mismatches.
    """
    if not (dataclasses.is_dataclass(cls) and isinstance(cls, type)):
        raise ConfigError(f"{cls!r} is not a dataclass type")
    hints = get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls) if f.init}
    prefix = f"{path}." if path else ""

    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise ConfigError(f"unknown key(s) {[prefix + k for k in unknown]} for {cls.__name__}")

    kwargs: dict[str, Any] = {}
    for name, field in fields.items():
        if name in data:
            kwargs[name] = _convert(data[name], hints[name], prefix + name)
        elif field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING:
            raise ConfigError(f"missing required key '{prefix + name}' for {cls.__name__}")
    return cls(**kwargs)


def _plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    return value


def to_dict(obj: Any) -> dict[str, Any]:
    """Return a plain-data snapshot of a dataclass instance."""
    plain = _plain(obj)
    if not isinstance(plain, dict):
        raise ConfigError("to_dict expects a dataclass instance or mapping")
    return plain


def _parse_override_value(raw: str) -> Any:
    for parse in (int, float):
        try:
            return parse(raw)
        except ValueError:
            pass
    return yaml.safe_load(raw)


def apply_overrides(data: Mapping[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Return a copy of ``data`` with dotted ``key=value`` overrides applied.

    Raises:
        ConfigError: If an override is malformed or traverses a non-mapping value.
    """
    result: dict[str, Any] = json.loads(json.dumps(_plain(data)))
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise ConfigError(f"override must look like 'a.b=value', got {item!r}")
        parts = key.split(".")
        node: dict[str, Any] = result
        for depth, part in enumerate(parts[:-1]):
            child = node.setdefault(part, {})
            if not isinstance(child, dict):
                raise ConfigError(
                    f"override {item!r}: '{'.'.join(parts[: depth + 1])}' is not a mapping"
                )
            node = child
        node[parts[-1]] = _parse_override_value(raw)
    return result


def canonical_json(obj: Any) -> str:
    """Serialize deterministically: sorted keys, compact separators, UTF-8, no NaN."""
    return json.dumps(
        _plain(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def config_hash(obj: Any) -> str:
    """Return the sha256 hex digest of ``canonical_json(obj)``."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


CONFIG_KINDS: dict[str, type] = {}

# Modules whose import registers config kinds. Imported lazily by validate_config_tree.
CONFIG_KIND_MODULES: tuple[str, ...] = ("kittylm.tokenizer.trainer",)


def load_config_kinds() -> None:
    """Import every module in CONFIG_KIND_MODULES so its config kinds are registered."""
    for module in CONFIG_KIND_MODULES:
        importlib.import_module(module)


def register_config_kind(kind: str, cls: type) -> None:
    """Register the dataclass that validates YAML files declaring ``kind: <kind>``."""
    if kind in CONFIG_KINDS and CONFIG_KINDS[kind] is not cls:
        raise ConfigError(f"config kind {kind!r} already registered")
    CONFIG_KINDS[kind] = cls


def validate_config_tree(root: Path) -> tuple[int, list[str]]:
    """Validate every YAML file under ``root``.

    Returns:
        (number of files checked, list of error messages). A missing ``root`` is not an
        error: it simply contains zero configs.
    """
    load_config_kinds()
    if not root.is_dir():
        return 0, []
    files = sorted(p for p in root.rglob("*") if p.suffix in {".yaml", ".yml"})
    errors: list[str] = []
    for file in files:
        rel = file.relative_to(root).as_posix()
        try:
            data = load_yaml(file)
            kind = data.get("kind")
            if not isinstance(kind, str):
                raise ConfigError("missing top-level 'kind: <name>'")
            cls = CONFIG_KINDS.get(kind)
            if cls is None:
                raise ConfigError(f"unknown config kind {kind!r}; known: {sorted(CONFIG_KINDS)}")
            from_dict(cls, {k: v for k, v in data.items() if k != "kind"})
        except ConfigError as exc:
            errors.append(f"configs/{rel}: {exc}")
    return len(files), errors
