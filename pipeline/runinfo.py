"""Run directory allocation and progress lines."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def next_run_id(runs_dir: Path) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    number = 1
    while (runs_dir / f"{today}_{number:03d}").exists():
        number += 1
    return f"{today}_{number:03d}"


def read_latest(runs_dir: Path) -> Path | None:
    pointer = runs_dir / "latest_id.txt"
    if pointer.is_file():
        name = pointer.read_text(encoding="utf-8").strip()
        path = runs_dir / name
        if path.is_dir():
            return path
    link = runs_dir / "latest"
    if link.is_symlink() or link.is_dir():
        target = link.resolve()
        if target.is_dir():
            return target
    return None


def publish_latest(runs_dir: Path, run_dir: Path) -> None:
    (runs_dir / "latest_id.txt").write_text(run_dir.name + "\n", encoding="utf-8")
    link = runs_dir / "latest"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(run_dir.name)


GRAPH = """
PRECHECK
  ↓
INGEST
  ↓
RENDER
  ↓
DETECT
  ↓
CROP
  ↓
OCR
  ↓
COMPARE
  ↓
REVIEW/CALIBRATION
  ↓
EXPORT DATASET
  ↓
TRAIN
  ↓
EVALUATE CHECKPOINTS
  ↓
SELECT BEST
  ↓
FROZEN-100
  ↓
REPORT
""".strip()
