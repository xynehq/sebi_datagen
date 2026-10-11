"""Prepare an isolated fixture repository for CI."""

from __future__ import annotations

import json
from pathlib import Path

from pipeline.line_image import write_text_image


def prepare_fixture(root: Path) -> None:
    """Write configs and a two-crop frozen exam for the isolated fixture run."""
    root.mkdir(parents=True, exist_ok=True)
    configs = root / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    (configs / "pipeline.yaml").write_text(
        """
seed: 3407
execution:
  mode: fixture
runtime:
  require_gpu: false
  require_tesseract: false
  require_ocr_models: false
source:
  documents_dir: data/raw/documents
paths:
  data_dir: data
  frozen_benchmark: frozen/marathi100
  train_manifest: data/datasets/train.jsonl
  val_manifest: data/datasets/val.jsonl
ocr:
  backends: [indicocr, chitrapathak, tesseract]
  nonce: 1
calibration:
  seed: 3407
  rates:
    ALL_3_AGREE: 0.05
    TWO_AGREE: 0.20
    ALL_DIFFER: 1.0
    EMPTY: 1.0
    BAD_BOUNDARY: 1.0
training:
  backend: fixture
  primary_metric: cer
  epochs: 1
evaluation:
  device: cpu
""".lstrip(),
        encoding="utf-8",
    )
    (configs / "dataset.yaml").write_text(
        "split:\n  train: 0.8\n  val: 0.1\n  test: 0.1\n  seed: 3407\n",
        encoding="utf-8",
    )
    (configs / "training.yaml").write_text(
        "backend: fixture\nprimary_metric: cer\n",
        encoding="utf-8",
    )
    (configs / "evaluation.yaml").write_text(
        "primary_metric: cer\nbenchmark: frozen100\n",
        encoding="utf-8",
    )
    for name in ("rendering.yaml", "detection.yaml", "cropping.yaml", "ocr.yaml", "review.yaml"):
        (configs / name).write_text("{}\n", encoding="utf-8")
    frozen = root / "frozen" / "marathi100"
    (frozen / "images").mkdir(parents=True, exist_ok=True)
    write_text_image(frozen / "images" / "frozen.png", "परीक्षा")
    row = {
        "sample_id": "FROZEN_1",
        "image_filename": "/images/frozen.png",
        "expected_text": "परीक्षा",
        "issue_type": "fixture",
    }
    (frozen / "validation.jsonl").write_text(
        json.dumps(row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (root / "data" / "raw" / "documents").mkdir(parents=True, exist_ok=True)
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    (root / "runs").mkdir(parents=True, exist_ok=True)
