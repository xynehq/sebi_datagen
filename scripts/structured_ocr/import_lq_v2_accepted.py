#!/usr/bin/env python3
"""Export IndicPhotoOCR LQ v2 accepted rows to a structured-OCR manifest JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.structured_ocr.leakage import check_document_split_leaks, check_duplicate_image_hashes
from training.structured_ocr.lq_v2 import load_lq_v2_accepted, normalize_lq_v2_row
from training.structured_ocr.manifest import assert_training_ready, load_jsonl, validate_manifest_row

DEFAULT_INDIC = Path.home() / "Desktop" / "IndicPhotoOCR"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indic-root", type=Path, default=DEFAULT_INDIC)
    parser.add_argument(
        "--source",
        choices=("accepted", "manifest"),
        default="accepted",
        help="accepted.jsonl or an existing manifest path via --manifest",
    )
    parser.add_argument("--manifest", type=Path, default=None, help="Optional JSONL (e.g. pilot queue) to normalize")
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "lq_v2_structured_manifest.jsonl")
    parser.add_argument("--assert-training-ready", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_INDIC)
    args = parser.parse_args(argv)

    if args.source == "accepted":
        rows = load_lq_v2_accepted(args.indic_root)
    else:
        if not args.manifest:
            print("--manifest required when --source manifest", file=sys.stderr)
            return 2
        raw = load_jsonl(args.manifest)
        rows = [normalize_lq_v2_row(r, indic_root=args.indic_root) for r in raw]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    blocked = sum(1 for row in rows if validate_manifest_row(row, repo_root=args.repo_root))
    leaks = check_document_split_leaks(rows) + check_duplicate_image_hashes(rows, repo_root=args.repo_root)
    print(f"wrote {len(rows)} rows -> {args.out}")
    print(f"rows_with_issues={blocked} leakage_issues={len(leaks)}")
    for item in leaks[:10]:
        print(f"  {item}")

    if args.assert_training_ready:
        try:
            assert_training_ready(rows, repo_root=args.repo_root)
        except Exception as exc:
            print(f"NOT READY: {exc}", file=sys.stderr)
            return 1
        print("READY FOR TRAINING: YES (accepted manifest gate passed)")
    elif not rows:
        print("No accepted rows yet — run human pilot first.", file=sys.stderr)
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
