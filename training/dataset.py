"""Training manifests and the frozen-benchmark guard.

Training reads only approved gold splits. Frozen-100 ids and paths are refused.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pipeline.gold import GoldError


def load_frozen_ids(frozen_dir: Path) -> set[str]:
    """Collect every identifier that belongs to the frozen exam."""
    path = frozen_dir / "validation.jsonl"
    found: set[str] = set()
    if not path.is_file():
        return found
    found.add(str(path.resolve()))
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for key in ("sample_id", "image_filename", "crop_id", "expected_text"):
            value = row.get(key)
            if key != "expected_text" and isinstance(value, str) and value:
                found.add(value)
                found.add(value.lstrip("/"))
                found.add(Path(value).name)
    return found


def assert_training_rows_safe(
    rows: Sequence[Mapping[str, Any]],
    frozen_ids: set[str],
    frozen_dir: Path,
    *,
    allow_auto: bool = False,
) -> None:
    """Raise when a training or validation row touches the frozen exam.

    Auto-accepted rows stay out unless the caller passed ``allow_auto``.
    They still cannot be marked gold, and they still cannot carry a frozen id.
    """
    frozen_marker = str(frozen_dir)
    for row in rows:
        blob = " ".join(
            str(row.get(key) or "")
            for key in (
                "sample_id",
                "crop_id",
                "image_filename",
                "crop_path",
                "image_path",
                "source",
            )
        )
        method = str(row.get("verification_method") or "human")
        status = str(row.get("data_status") or "")
        auto = method.startswith("auto_") or status.startswith("auto_accepted")
        if auto and row.get("gold") is True:
            name = row.get("sample_id") or row.get("crop_id")
            raise GoldError(f"auto-accepted row {name} cannot have gold true")
        if auto and not allow_auto:
            raise GoldError(
                f"auto-accepted row {row.get('sample_id') or row.get('crop_id')} "
                "is not human gold"
            )
        if "frozen" in blob.lower() and "marathi100" in blob.lower():
            raise GoldError(f"frozen benchmark path in training row {row.get('sample_id')}")
        if frozen_marker and frozen_marker in blob:
            raise GoldError(f"frozen benchmark path in training row {row.get('sample_id')}")
        for key in ("sample_id", "crop_id", "image_filename", "crop_path"):
            value = row.get(key)
            if isinstance(value, str) and value in frozen_ids:
                raise GoldError(f"frozen id {value!r} cannot enter training")
            name = Path(value).name if isinstance(value, str) else ""
            points_at_exam = isinstance(value, str) and "images/" in value.replace("\\", "/")
            if name in frozen_ids and points_at_exam:
                # image basenames such as 0001.png are only frozen when the
                # filename itself was registered AND the row points at the exam.
                if str(frozen_dir) in value or value.lstrip("/").startswith("images/"):
                    raise GoldError(f"frozen image {value!r} cannot enter training")


def assign_document_splits(
    document_ids: Sequence[str],
    *,
    seed: int,
    train: float,
    val: float,
) -> dict[str, str]:
    """Assign each document to train, val, or test. Val is kept non-empty when possible."""
    documents = sorted(set(document_ids))
    assignments: dict[str, str] = {}
    for document_id in documents:
        digest = hashlib.sha256(f"{seed}:{document_id}".encode()).hexdigest()
        bucket = int(digest[:8], 16) / 0xFFFFFFFF
        if bucket < train:
            split = "train"
        elif bucket < train + val:
            split = "val"
        else:
            split = "test"
        assignments[document_id] = split
    if documents and "val" not in assignments.values():
        assignments[documents[-1]] = "val"
    return assignments


def build_splits(
    gold_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    train: float,
    val: float,
    frozen_ids: set[str],
    frozen_dir: Path,
    allow_auto: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Split gold by document. Frozen rows are rejected before the split is written."""
    assert_training_rows_safe(gold_rows, frozen_ids, frozen_dir, allow_auto=allow_auto)
    documents = [str(row["document_id"]) for row in gold_rows]
    assignments = assign_document_splits(documents, seed=seed, train=train, val=val)
    grouped: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for row in gold_rows:
        item = dict(row)
        split = assignments[str(row["document_id"])]
        item["split"] = split
        grouped[split].append(item)
    assert_training_rows_safe(
        grouped["train"] + grouped["val"] + grouped["test"],
        frozen_ids,
        frozen_dir,
        allow_auto=allow_auto,
    )
    return grouped
