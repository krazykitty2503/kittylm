"""Evaluate a checkpoint: loss, perplexity, bits-per-byte (overall and per category), speed.

Usage:
    python scripts/evaluate.py --model-config configs/model/nano.yaml --set vocab_size=512 \
        --checkpoint runs/<run>/checkpoints/final.pt \
        --tokenizer runs/tokenizers/smoke-512/tokenizer.json \
        --doc prose=tests/fixtures/smoke_corpus.txt [--doc code=path/to/file.py ...] \
        [--speed 128,64] [--out runs/<run>/eval.json]

Each --doc is one document of the named category, scored with the shared windowing policy
(D-023). --speed PROMPT,NEW adds batch-1 prefill/decode throughput. The report is printed as
JSON and optionally written to --out (under runs/ when inside the repository).
KittyLM makes no network requests; this script only reads local files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from kittylm.config import ConfigError
from kittylm.evaluation.inference_speed import measure_inference_speed
from kittylm.evaluation.perplexity import Document, evaluate_documents, token_byte_lengths
from kittylm.inference.loading import LoadError, load_for_inference, precision_dtype
from kittylm.training.checkpoint import CheckpointError

ROOT = Path(__file__).resolve().parents[1]


def parse_doc(value: str) -> tuple[str, Path]:
    category, sep, path = value.partition("=")
    if not sep or not category or not path:
        raise argparse.ArgumentTypeError("--doc must be CATEGORY=PATH")
    return category, Path(path)


def parse_speed(value: str) -> tuple[int, int]:
    try:
        prompt, new = (int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--speed must be PROMPT_TOKENS,NEW_TOKENS") from exc
    return prompt, new


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[], help="model override, key=value")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--doc", type=parse_doc, action="append", required=True)
    parser.add_argument("--stride", type=int, help="window stride (default: context/2)")
    parser.add_argument("--speed", type=parse_speed, help="PROMPT_TOKENS,NEW_TOKENS")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", default="fp32", choices=["fp32", "bf16"])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    if args.out is not None:
        out, root = args.out.resolve(), ROOT.resolve()
        if out.is_relative_to(root) and not out.is_relative_to(root / "runs"):
            print("refusing to write an evaluation report inside the repository outside runs/")
            return 2
    device = torch.device(args.device)
    try:
        loaded = load_for_inference(
            args.model_config, args.checkpoint, args.tokenizer, device, overrides=args.set
        )
        dtype = precision_dtype(args.precision)
        documents = []
        for category, path in args.doc:
            ids = loaded.tokenizer.encode(path.read_text(encoding="utf-8"))
            documents.append(Document(category, ids, token_byte_lengths(loaded.tokenizer, ids)))
        report = evaluate_documents(
            loaded.model, documents, device=device, stride=args.stride, autocast_dtype=dtype
        )
        result = {
            **report.metrics(),
            "global_step": loaded.global_step,
            "tokenizer_sha256": loaded.tokenizer.sha256,
            "precision": args.precision,
            "device_type": device.type,
        }
        if args.speed is not None:
            speed = measure_inference_speed(
                loaded.model,
                prompt_tokens=args.speed[0],
                new_tokens=args.speed[1],
                device=device,
                autocast_dtype=dtype,
            )
            result["inference"] = {
                "prefill_tok_s": speed.prefill_tok_s,
                "decode_tok_s": speed.decode_tok_s,
                "prompt_tokens": speed.prompt_tokens,
                "new_tokens": speed.new_tokens,
                "repeats": speed.repeats,
            }
    except (ConfigError, LoadError, CheckpointError, ValueError, OSError) as exc:
        print(f"evaluation failed: {exc}")
        return 1
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
