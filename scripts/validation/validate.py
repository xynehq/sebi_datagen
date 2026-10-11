#!/usr/bin/env python3
"""Score a checkpoint on the development validation split.

Uses ``data/datasets/val.jsonl``. This is the split that chooses a checkpoint.
Frozen-100 is a different exam and is not read here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "models")]

from evaluation.harness import score_corpus  # noqa: E402
from pipeline.config import load_config  # noqa: E402
from training.dataset import assert_training_rows_safe, load_frozen_ids  # noqa: E402


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _image(row: dict) -> Path:
    raw = row.get("image_filename") or row.get("crop_path") or row.get("image_path")
    path = Path(str(raw))
    return path if path.is_absolute() else ROOT / path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score data/datasets/val.jsonl.")
    parser.add_argument("--checkpoint", required=True, help="A .ckpt file for the val images.")
    parser.add_argument("--labels", type=Path, default=ROOT / "data" / "datasets" / "val.jsonl")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs" / "manual" / "validation" / "metrics.json",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    rows = _rows(args.labels)
    if not rows:
        print(
            f"{args.labels} is empty. Approve human gold before scoring the validation split.",
            file=sys.stderr,
        )
        return 1
    config = load_config(ROOT)
    frozen_dir = ROOT / (config.get("paths") or {}).get("frozen_benchmark", "data/eval_data")
    frozen_ids = load_frozen_ids(frozen_dir)
    assert_training_rows_safe(rows, frozen_ids, frozen_dir)
    sample_ids = {str(row.get("sample_id") or "") for row in rows}
    if sample_ids & frozen_ids:
        print("Validation labels include frozen-100 ids.", file=sys.stderr)
        return 1

    from evaluation.canonical import pipeline_fingerprint
    from evaluation.inference import load_model, recognise_canonical, resolve_device

    device = resolve_device(args.device).device
    model = load_model(args.checkpoint, device)
    expected = [str(row.get("expected_text") or row.get("gold_text") or "") for row in rows]
    predicted = [
        recognise_canonical(model, device, str(_image(row))).prediction for row in rows
    ]
    metrics = score_corpus(expected, predicted)
    metrics["split"] = "val"
    metrics["checkpoint"] = args.checkpoint
    metrics["rows"] = len(rows)
    metrics["used_for_selection"] = True
    metrics["mode"] = "piecewise"
    metrics["pipeline"] = pipeline_fingerprint()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(metrics, ensure_ascii=False, indent=2) + "\n"
    args.output.write_text(payload, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
