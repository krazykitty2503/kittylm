"""Generate text from a checkpoint; the sample is secret-scanned and redacted before saving.

Usage:
    python scripts/generate.py --model-config configs/model/nano.yaml --set vocab_size=512 \
        --checkpoint runs/<run>/checkpoints/final.pt \
        --tokenizer runs/tokenizers/smoke-512/tokenizer.json \
        --prompt "def add(" --max-new-tokens 64 --temperature 0.8 --top-k 40 --top-p 0.95 --seed 1

Default output: runs/samples/<name>.txt plus <name>.json (redaction summary). Inside the
repository, samples may only be written under runs/ (gitignored, blocked by the artifact guard).
KittyLM makes no network requests; this script only reads local files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from kittylm.config import ConfigError
from kittylm.inference.generate import SamplingConfig, generate
from kittylm.inference.loading import LoadError, load_for_inference, precision_dtype
from kittylm.inference.samples import SampleSecretError, save_sample
from kittylm.training.checkpoint import CheckpointError

ROOT = Path(__file__).resolve().parents[1]


def output_directory_problem(directory: Path) -> str | None:
    """Samples inside the repository must live under runs/."""
    resolved, root = directory.resolve(), ROOT.resolve()
    if resolved.is_relative_to(root) and not resolved.is_relative_to(root / "runs"):
        return "refusing to write samples inside the repository outside runs/"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[], help="model override, key=value")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 means greedy")
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-eot-stop", action="store_true")
    parser.add_argument("--stride", type=int, help="context re-use stride (default: context/2)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", default="fp32", choices=["fp32", "bf16"])
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "samples")
    parser.add_argument("--name", default="sample")
    args = parser.parse_args(argv)

    problem = output_directory_problem(args.out_dir)
    if problem:
        print(problem)
        return 2
    device = torch.device(args.device)
    try:
        loaded = load_for_inference(
            args.model_config, args.checkpoint, args.tokenizer, device, overrides=args.set
        )
        tokenizer = loaded.tokenizer
        config = SamplingConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            eot_id=None if args.no_eot_stop else tokenizer.eot_id,
        )
        prompt_ids = tokenizer.encode(args.prompt)
        result = generate(
            loaded.model,
            prompt_ids,
            config,
            device=device,
            stride=args.stride,
            autocast_dtype=precision_dtype(args.precision),
            generator=torch.Generator().manual_seed(args.seed),
        )
        saved = save_sample(args.out_dir, args.name, tokenizer.decode(result.ids))
    except (ConfigError, LoadError, CheckpointError, ValueError, SampleSecretError) as exc:
        print(f"generation failed: {exc}")
        return 1
    summary = {
        "sample": saved.path.name,
        "prompt_tokens": len(result.prompt_ids),
        "new_tokens": len(result.new_ids),
        "stop_reason": result.stop_reason,
        "context_resets": result.context_resets,
        "redacted_lines": list(saved.redacted_lines),
        "redaction_rules": list(saved.rules),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
