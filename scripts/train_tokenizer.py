"""Train a byte-level BPE tokenizer from a `kind: tokenizer` config.

Usage:
    python scripts/train_tokenizer.py --config configs/tokenizer/smoke.yaml
    python scripts/train_tokenizer.py --config configs/tokenizer/smoke.yaml --set vocab_size=384

Default output: runs/tokenizers/<name>/tokenizer.json (override with --out <path>).

Tokenizer artifacts are generated outputs and are never committed: inside the repository the
output must be under `runs/` (gitignored and blocked by the artifact guard). Paths outside the
repository (for example a temporary directory) are allowed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kittylm.config import ConfigError, apply_overrides, from_dict, load_yaml
from kittylm.tokenizer.trainer import (
    TokenizerConfig,
    TokenizerTrainingError,
    read_corpus_files,
    train_from_config,
)

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", type=Path, help="default: runs/tokenizers/<name>/tokenizer.json")
    parser.add_argument("--set", action="append", default=[], help="override, e.g. vocab_size=768")
    args = parser.parse_args(argv)

    try:
        data = load_yaml(args.config)
        if data.get("kind") != "tokenizer":
            raise ConfigError("config must declare kind: tokenizer")
        data = apply_overrides({k: v for k, v in data.items() if k != "kind"}, args.set)
        config = from_dict(TokenizerConfig, data)
    except ConfigError as exc:
        print(f"invalid config: {exc}")
        return 2

    out = (args.out or ROOT / "runs" / "tokenizers" / config.name / "tokenizer.json").resolve()
    root = ROOT.resolve()
    if out.is_relative_to(root) and not out.is_relative_to(root / "runs"):
        print("refusing to write a tokenizer artifact inside the repository outside runs/")
        return 2

    try:
        tokenizer = train_from_config(config, ROOT)
    except TokenizerTrainingError as exc:
        print(f"training failed: {exc}")
        return 1
    tokenizer.save(out)

    documents = read_corpus_files([ROOT / p for p in config.corpus_files])
    corpus_bytes = sum(len(doc.encode("utf-8")) for doc in documents)
    corpus_tokens = sum(len(tokenizer.encode(doc)) for doc in documents)
    shown = out.relative_to(root).as_posix() if out.is_relative_to(root) else out.name
    print(f"tokenizer:        {config.name} ({config.purpose})")
    print(f"vocab_size:       {tokenizer.vocab_size} ({tokenizer.n_merges} merges)")
    print(f"eot_id:           {tokenizer.eot_id}")
    print(f"tokenizer_sha256: {tokenizer.sha256}")
    print(f"corpus_sha256:    {tokenizer.corpus_sha256}")
    print(f"corpus:           {corpus_bytes} bytes -> {corpus_tokens} tokens")
    print(f"bytes_per_token:  {corpus_bytes / corpus_tokens:.3f}")
    print(f"artifact:         {shown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
