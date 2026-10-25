#!/usr/bin/env python3
"""Validate structured OCR manifest rows without starting training."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.structured_ocr.manifest import assert_training_ready, load_jsonl, validate_manifest_row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="JSONL manifest path")
    parser.add_argument(
        "--assert-training-ready",
        action="store_true",
        help="Exit 1 unless every row is human-verified and structurally valid",
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args(argv)

    rows = load_jsonl(args.manifest)
    issue_counts: Counter[str] = Counter()
    blocked = 0
    for row in rows:
        issues = validate_manifest_row(row, repo_root=args.repo_root)
        if issues:
            blocked += 1
            for code in issues:
                issue_counts[code] += 1

    print(f"rows={len(rows)} blocked={blocked}")
    for code, count in issue_counts.most_common():
        print(f"  {code}: {count}")

    if args.assert_training_ready:
        try:
            assert_training_ready(rows, repo_root=args.repo_root)
        except Exception as exc:
            print(f"NOT READY: {exc}", file=sys.stderr)
            return 1
        print("READY FOR TRAINING: YES (manifest gate passed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
