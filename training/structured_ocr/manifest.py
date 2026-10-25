"""Training manifest gates for verified structured OCR."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from training.structured_ocr.targets import TargetValidationError, validate_target

VERIFIED_STATUSES = frozenset(
    {
        "human_verified",
        "transcription_verified",
        "gold",
        "accepted",
    }
)

BLOCKED_LABEL_SOURCES = frozenset(
    {
        "tesseract_draft",
        "auto_ocr",
        "pending",
        "candidate",
    }
)


def validate_manifest_row(row: Mapping[str, Any], *, repo_root: Path | None = None) -> list[str]:
    """Return issue codes; empty list means row is eligible for training targets."""
    issues: list[str] = []
    status = str(row.get("verification_status") or row.get("review_status") or "").lower()
    label_source = str(row.get("label_source") or row.get("label_origin") or "").lower()
    method = str(row.get("verification_method") or "").lower()

    if row.get("transcription_verified") is not True and status not in VERIFIED_STATUSES:
        if method.startswith("auto_") or label_source in BLOCKED_LABEL_SOURCES:
            issues.append("label_not_human_verified")
        elif not row.get("gold"):
            issues.append("label_not_verified")

    text = str(row.get("text") or row.get("expected_text") or row.get("target") or "")
    if not text.strip() and not row.get("allow_empty_page"):
        issues.append("missing_target_text")

    task = str(row.get("task") or row.get("sample_type") or "line_crop")
    if task not in {"line_crop", "region_crop", "full_page"}:
        issues.append(f"unknown_task_{task}")

    image = str(row.get("image") or row.get("image_path") or row.get("image_filename") or "")
    if repo_root and image and not image.startswith("http"):
        path = Path(image)
        if not path.is_absolute():
            path = repo_root / path
        if not path.is_file():
            issues.append("missing_image")

    if text.strip():
        report = validate_target(
            text,
            expect_table_rows=row.get("expect_table_rows"),
            expect_table_cols=row.get("expect_table_cols"),
            allow_empty=bool(row.get("allow_empty_page")),
        )
        if not report.ok:
            issues.extend(report.issues)

    for key in ("sample_id", "document_id", "page", "language", "split"):
        if key in row and row[key] in (None, ""):
            issues.append(f"missing_{key}")

    return issues


def assert_training_ready(rows: Sequence[Mapping[str, Any]], *, repo_root: Path) -> None:
    """Raise when any row is not verified or structurally valid."""
    if not rows:
        raise TargetValidationError("training manifest is empty — human verification required")
    blocked = []
    for row in rows:
        issues = validate_manifest_row(row, repo_root=repo_root)
        if issues:
            blocked.append((row.get("sample_id") or row.get("id"), issues))
    if blocked:
        sample = blocked[0]
        raise TargetValidationError(
            f"{len(blocked)} rows blocked; example {sample[0]}: {', '.join(sample[1])}"
        )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows
