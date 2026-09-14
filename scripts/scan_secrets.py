"""Scan tracked (or staged) files for secrets.

Prints ``path:line: rule`` for each finding and never the matched value. Exits 1 if
anything is found. Binary blobs (containing NUL bytes) are skipped here; the artifact guard
decides whether such files may be tracked at all.

Usage:
    python scripts/scan_secrets.py            # every file in the git index
    python scripts/scan_secrets.py --staged   # only files staged for the next commit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kittylm.data.secrets import scan_text
from kittylm.utils.git import index_entries, read_blobs, repo_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--staged", action="store_true", help="scan only staged changes")
    args = parser.parse_args(argv)

    root = repo_root(Path.cwd())
    entries = index_entries(root, staged_only=args.staged)
    blobs = read_blobs(root, [e.blob for e in entries])

    total_findings = 0
    scanned = 0
    for entry in entries:
        data = blobs[entry.blob]
        if b"\0" in data:
            continue
        scanned += 1
        for finding in scan_text(data.decode("utf-8", errors="replace")):
            total_findings += 1
            print(f"{entry.path}:{finding.line}: {finding.rule}")

    scope = "staged" if args.staged else "tracked"
    if total_findings:
        print(f"secret scan: {total_findings} finding(s) in {scanned} {scope} text file(s)")
        return 1
    print(f"secret scan: clean ({scanned} {scope} text file(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
