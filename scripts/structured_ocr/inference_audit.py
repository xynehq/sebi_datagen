#!/usr/bin/env python3
"""Document controlled inference sweeps (delegates to IndicPhotoOCR eval)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDIC = Path.home() / "Desktop" / "IndicPhotoOCR"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indic-repo", type=Path, default=DEFAULT_INDIC)
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--model-path", default="data/chitrapathak2")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "inference_audit")
    parser.add_argument("--temperatures", default="0.0,0.1")
    parser.add_argument("--max-pixels", default="300000,600000")
    parser.add_argument("--structured", action="store_true")
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    baseline = ROOT / "scripts" / "structured_ocr" / "run_baseline_inference.py"
    for temp in args.temperatures.split(","):
        for pixels in args.max_pixels.split(","):
            tag = f"t{temp.strip()}_px{pixels.strip()}"
            out = args.out_dir / f"preds_{tag}.jsonl"
            cmd = [
                sys.executable,
                str(baseline),
                "--indic-repo",
                str(args.indic_repo),
                "--test-jsonl",
                str(args.test_jsonl),
                "--test-dir",
                str(args.test_dir),
                "--model-path",
                args.model_path,
                "--out",
                str(out),
                "--max-pixels",
                pixels.strip(),
            ]
            if args.adapter:
                cmd.extend(["--adapter", args.adapter])
            if args.structured:
                cmd.append("--structured")
            # eval_chitrapathak uses greedy decode; temperature documented for future extension
            manifest.append({"tag": tag, "temperature": temp, "max_pixels": int(pixels), "preds": str(out)})
            print("Would run:", " ".join(cmd), flush=True)
    summary = args.out_dir / "sweep_manifest.json"
    summary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {summary} ({len(manifest)} configurations). Run run_baseline_inference per entry when weights are available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
