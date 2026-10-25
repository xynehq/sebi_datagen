#!/usr/bin/env python3
"""Run Chitrapathak inference for structured-OCR baseline (IndicPhotoOCR eval harness).

Does not train. Requires local model weights and optional LoRA adapter.
Writes predictions JSONL compatible with eval_structure.py and IndicPhotoOCR
score_tabular_reading_order.py.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INDIC_REPO = Path(os.environ.get("INDICPHOTOOCR_ROOT", Path.home() / "Desktop" / "IndicPhotoOCR"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indic-repo", type=Path, default=DEFAULT_INDIC_REPO)
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--model-path", default="data/chitrapathak2")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-pixels", type=int, default=600_000)
    parser.add_argument(
        "--prompt",
        default=None,
        help="Override prompt; default uses IndicPhotoOCR line prompt unless --structured",
    )
    parser.add_argument("--structured", action="store_true", help="Use STRUCTURED_PAGE_PROMPT")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    eval_script = args.indic_repo / "scripts" / "eval_chitrapathak.py"
    if not eval_script.is_file():
        print(f"Missing {eval_script}; set INDICPHOTOOCR_ROOT", file=sys.stderr)
        return 2

    prompt = args.prompt
    if prompt is None and args.structured:
        sys.path.insert(0, str(ROOT))
        from training.structured_ocr.prompts import STRUCTURED_PAGE_PROMPT

        prompt = STRUCTURED_PAGE_PROMPT

    cmd = [
        sys.executable,
        str(eval_script),
        "--test-jsonl",
        str(args.test_jsonl),
        "--test-dir",
        str(args.test_dir),
        "--model-path",
        args.model_path,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--max-pixels",
        str(args.max_pixels),
        "--out",
        str(args.out),
    ]
    if args.adapter:
        cmd.extend(["--adapter", args.adapter])
    if prompt:
        cmd.extend(["--prompt", prompt])
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])

    print("Running:", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(args.indic_repo))


if __name__ == "__main__":
    raise SystemExit(main())
