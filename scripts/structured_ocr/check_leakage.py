#!/usr/bin/env python3
"""Report split leaks and duplicate image hashes in a manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.structured_ocr.leakage import check_document_split_leaks, check_duplicate_image_hashes
from training.structured_ocr.manifest import load_jsonl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args(argv)

    rows = load_jsonl(args.manifest)
    issues = check_document_split_leaks(rows) + check_duplicate_image_hashes(rows, repo_root=args.repo_root)
    if issues:
        for item in issues:
            print(item)
        return 1
    print(f"ok rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
