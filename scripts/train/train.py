#!/usr/bin/env python3
"""Fine-tune on the human-approved training split.

Reads ``data/datasets/train.jsonl`` and checks progress on
``data/datasets/val.jsonl``. Frozen-100 is not loaded.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "models")]

from pipeline.config import load_config  # noqa: E402
from training.dataset import assert_training_rows_safe, load_frozen_ids  # noqa: E402


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train on data/datasets/train.jsonl.")
    parser.add_argument("--checkpoint", default=None, help="Base .ckpt. Defaults to CHECKPOINT.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs" / "manual" / "checkpoints" / "finetuned.ckpt",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    config = load_config(ROOT)
    training_cfg = config.get("training") or {}
    train_path = ROOT / "data" / "datasets" / "train.jsonl"
    val_path = ROOT / "data" / "datasets" / "val.jsonl"
    train_rows = _rows(train_path)
    val_rows = _rows(val_path)
    if not train_rows or not val_rows:
        print(
            "data/datasets/train.jsonl and data/datasets/val.jsonl must both contain rows.\n"
            "Those files stay empty until a person approves gold. Consensus is not gold.",
            file=sys.stderr,
        )
        return 1
    frozen_dir = ROOT / (config.get("paths") or {}).get("frozen_benchmark", "data/eval_data")
    assert_training_rows_safe(train_rows + val_rows, load_frozen_ids(frozen_dir), frozen_dir)

    from pipeline.weights import WeightError, resolve_weight

    checkpoint = args.checkpoint or os.environ.get("CHECKPOINT") or ""
    if not checkpoint:
        checkpoint = str(training_cfg.get("base_checkpoint") or "")
    try:
        checkpoint = str(resolve_weight(checkpoint, ROOT / "models" / "cache", expect="file"))
    except WeightError as exc:
        print(
            f"{exc}\nPass --checkpoint with a local file or a Hugging Face link.",
            file=sys.stderr,
        )
        return 2

    from IndicPhotoOCR.recognition.finetune import finetune

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = finetune(
        checkpoint=str(checkpoint),
        image_dir=str(ROOT),
        labels=str(train_path),
        output=str(args.output),
        val_labels=str(val_path),
        val_split=0.0,
        epochs=int(args.epochs or training_cfg.get("epochs", 1)),
        lr=float(training_cfg.get("learning_rate", 1e-4)),
        batch_size=int(training_cfg.get("batch_size", 8)),
        device=args.device or training_cfg.get("device"),
        seed=int(config.get("seed", 3407)),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
