"""Mandatory parameter accounting: where every parameter of a model lives.

Purpose:
    Comparisons between architectures are only meaningful if we know *where* the parameters
    are. A variant that "wins" because it received millions more embedding parameters has not
    shown a better architecture. This module attributes every parameter to exactly one
    category, counts tied tensors once, and refuses to report a budget whose parts do not add up.

Public API:
    ParameterCounts
        ``total, trainable, embedding, attention, mlp, normalization, positional, output,
        tied_parameters, non_embedding``; ``as_dict()``.
    count_parameters(model) -> ParameterCounts
    render_parameter_tree(model) -> str
    register_category(module_type, category), CATEGORIES
    AccountingError

Shapes:
    Not applicable: counts use ``Tensor.numel()`` only, so models on the ``meta`` device work.

Dtype:
    Any; only element counts are used.

Device:
    Any, including ``meta`` (no memory is allocated).

Math:
    ``total = embedding + attention + mlp + normalization + positional + output`` over unique
    tensors; ``non_embedding = total - embedding``; ``tied_parameters`` is the size of tensors
    reachable under more than one name (counted once in ``total``).

Invariants:
    - Every parameter is classified by its nearest registered ancestor module; a module listed
      in ``model.output_module_names`` is ``output``. Unclassified parameters are an error, so a
      new architecture component cannot be silently left out of the budget.
    - Parts sum exactly to ``total``; RoPE has buffers but no parameters (``positional == 0``).

Failure modes:
    - An unclassified parameter, or parts that do not sum to the total, raises AccountingError.

See:
    Plan rev 3.3 section 3 (mandatory parameter accounting), kittylm/ledger.py.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from torch import nn

from kittylm.model.attention import CausalSelfAttention
from kittylm.model.mlp import SwiGLU
from kittylm.model.normalization import RMSNorm
from kittylm.model.positional import RotaryEmbedding

__all__ = [
    "CATEGORIES",
    "AccountingError",
    "ParameterCounts",
    "count_parameters",
    "register_category",
    "render_parameter_tree",
]

CATEGORIES: tuple[str, ...] = (
    "embedding",
    "attention",
    "mlp",
    "normalization",
    "positional",
    "output",
)
_CATEGORY_BY_TYPE: dict[type[nn.Module], str] = {
    nn.Embedding: "embedding",
    CausalSelfAttention: "attention",
    SwiGLU: "mlp",
    RMSNorm: "normalization",
    RotaryEmbedding: "positional",
}


class AccountingError(RuntimeError):
    """Parameters could not be fully and consistently attributed."""


@dataclass(frozen=True)
class ParameterCounts:
    """Parameter budget of a model (all values are element counts)."""

    total: int
    trainable: int
    embedding: int
    attention: int
    mlp: int
    normalization: int
    positional: int
    output: int
    tied_parameters: int
    non_embedding: int

    def as_dict(self) -> dict[str, int]:
        """Return the counts as a plain dict (ledger ``model.parameters`` shape)."""
        return asdict(self)


def register_category(module_type: type[nn.Module], category: str) -> None:
    """Classify parameters owned by ``module_type`` (and its children) as ``category``."""
    if category not in CATEGORIES:
        raise AccountingError(f"unknown category {category!r}; known: {CATEGORIES}")
    _CATEGORY_BY_TYPE[module_type] = category


def _categorize(model: nn.Module) -> tuple[dict[int, tuple[str, int, bool]], int]:
    modules = dict(model.named_modules())
    output_names = set(getattr(model, "output_module_names", ()))
    unique: dict[int, tuple[str, int, bool]] = {}
    tied_ids: set[int] = set()
    for full_name, param in model.named_parameters(remove_duplicate=False):
        parts = full_name.split(".")[:-1]
        category: str | None = None
        for depth in range(len(parts), -1, -1):
            prefix = ".".join(parts[:depth])
            if prefix in output_names:
                category = "output"
                break
            category = _CATEGORY_BY_TYPE.get(type(modules[prefix]))
            if category is not None:
                break
        if category is None:
            raise AccountingError(f"parameter {full_name!r} has no accounting category")
        key = id(param)
        if key in unique:
            tied_ids.add(key)
            continue
        unique[key] = (category, param.numel(), param.requires_grad)
    tied = sum(unique[key][1] for key in tied_ids)
    return unique, tied


def count_parameters(model: nn.Module) -> ParameterCounts:
    """Attribute every unique parameter tensor of ``model`` to a category."""
    unique, tied = _categorize(model)
    by_category = dict.fromkeys(CATEGORIES, 0)
    total = trainable = 0
    for category, numel, requires_grad in unique.values():
        by_category[category] += numel
        total += numel
        trainable += numel if requires_grad else 0
    if sum(by_category.values()) != total:
        raise AccountingError("parameter categories do not sum to the total")
    return ParameterCounts(
        total=total,
        trainable=trainable,
        tied_parameters=tied,
        non_embedding=total - by_category["embedding"],
        **by_category,
    )


def render_parameter_tree(model: nn.Module) -> str:
    """Human-readable budget, e.g. for logs and milestone reports."""
    counts = count_parameters(model)
    layers = getattr(model, "layers", None)
    n_layers = len(layers) if layers is not None else 0
    final_norm = getattr(model, "norm", None)
    final_norm_params = (
        sum(p.numel() for p in final_norm.parameters()) if final_norm is not None else 0
    )
    block_norm = counts.normalization - final_norm_params
    blocks_total = counts.attention + counts.mlp + block_norm
    per_block = blocks_total // max(n_layers, 1)
    tied_note = "tied to token embeddings, 0 extra" if counts.tied_parameters else "untied"

    def row(label: str, value: int) -> str:
        return f"{label:<42}{value:>14,}"

    lines = [
        row(f"total (trainable {counts.trainable:,})", counts.total),
        row("├── token embeddings", counts.embedding),
        row(f"├── blocks x{n_layers} ({per_block:,} per block)", blocks_total),
        row("│   ├── attention", counts.attention),
        row("│   ├── normalization", block_norm),
        row("│   └── mlp (SwiGLU)", counts.mlp),
        row("├── final norm", final_norm_params),
        row("├── positional (RoPE: buffers only)", counts.positional),
        row(f"└── lm head ({tied_note})", counts.output),
        row("non-embedding", counts.non_embedding),
    ]
    return "\n".join(lines)
