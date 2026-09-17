"""Byte-level BPE tokenizer for KittyLM (plan rev 3.3, milestone A).

``pretokenize`` splits text into chunks, ``trainer`` learns merges, ``bpe`` encodes, decodes
and serializes. This package never imports torch (enforced by tests/test_import_boundaries.py).
"""

from kittylm.tokenizer.bpe import (
    NAMED_SPECIAL_TOKENS,
    SPECIAL_TOKENS,
    BPETokenizer,
    TokenizerError,
)
from kittylm.tokenizer.pretokenize import PRETOKENIZER_ID, pretokenize
from kittylm.tokenizer.trainer import (
    TokenizerConfig,
    TokenizerTrainingError,
    corpus_sha256,
    train_bpe,
    train_from_config,
)

__all__ = [
    "NAMED_SPECIAL_TOKENS",
    "PRETOKENIZER_ID",
    "SPECIAL_TOKENS",
    "BPETokenizer",
    "TokenizerConfig",
    "TokenizerError",
    "TokenizerTrainingError",
    "corpus_sha256",
    "pretokenize",
    "train_bpe",
    "train_from_config",
]
