"""Generate experiments/ablations.md from validated experiment records.

Usage:
    python scripts/build_ablation_table.py          # write the table
    python scripts/build_ablation_table.py --check  # exit 1 if the committed table is stale
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kittylm.ledger import LedgerError, load_all_records, render_ablation_table

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "experiments" / "ablations.md"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="verify instead of writing")
    args = parser.parse_args(argv)

    try:
        records = load_all_records(ROOT / "experiments")
    except LedgerError as exc:
        print(exc)
        return 1
    rendered = render_ablation_table(records)

    if args.check:
        current = TABLE.read_text(encoding="utf-8") if TABLE.exists() else None
        if current != rendered:
            print("experiments/ablations.md is out of sync; run scripts/build_ablation_table.py")
            return 1
        print(f"ablation table: in sync ({len(records)} record(s))")
        return 0

    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"wrote experiments/ablations.md ({len(records)} record(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
