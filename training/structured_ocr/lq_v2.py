"""Map IndicPhotoOCR LQ v2 review rows to structured-OCR training manifests."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

LQ_V2_PACK_REL = "data/chitrapathak_shared_low_quality_v2"
DRAFT_LABEL_MARKERS = (
    "tesseract_seed_not_ground_truth",
    "tesseract_draft",
    "auto_ocr",
)


def crop_type_to_task(crop_type: str | None) -> str:
    kind = (crop_type or "").lower()
    if kind in {"full_page", "page"}:
        return "full_page"
    if kind in {"text_block", "block", "region", "table_region"}:
        return "region_crop"
    return "line_crop"


def is_lq_v2_draft_label(row: Mapping[str, Any]) -> bool:
    draft = str(row.get("draft_label_source") or row.get("label_source") or "").lower()
    if any(marker in draft for marker in DRAFT_LABEL_MARKERS):
        return True
    if row.get("transcription_status") == "pending_human" and row.get("transcription_verified") is not True:
        return True
    return False


def normalize_lq_v2_row(row: Mapping[str, Any], *, indic_root: Path) -> dict[str, Any]:
    """Convert one accepted (or queue) row into structured_ocr manifest fields."""
    pack = indic_root / LQ_V2_PACK_REL
    image = str(row.get("image") or "")
    if image and not Path(image).is_absolute():
        image = str((pack / image).resolve()) if (pack / image).exists() else str((indic_root / image).resolve())

    text = str(
        row.get("verified_text")
        or row.get("text")
        or row.get("expected_text")
        or row.get("target")
        or ""
    )
    split = str(row.get("final_split") or row.get("split_pool") or row.get("split") or "")
    page = row.get("page")
    if page in (None, ""):
        page = row.get("page_index")

    out: dict[str, Any] = {
        "sample_id": row.get("sample_id") or row.get("id") or row.get("candidate_id"),
        "document_id": row.get("document_id"),
        "page": page,
        "language": row.get("language"),
        "split": split,
        "task": row.get("task") or crop_type_to_task(str(row.get("crop_type") or "")),
        "image": image,
        "text": text,
        "transcription_verified": bool(row.get("transcription_verified")),
        "verification_status": "human_verified" if row.get("transcription_verified") else row.get("review_status"),
        "label_source": "human" if row.get("transcription_verified") else row.get("draft_label_source"),
        "category": row.get("crop_type") or row.get("category"),
        "image_sha256": row.get("image_sha256"),
        "split_pool": row.get("split_pool"),
        "lq_v2_id": row.get("id"),
    }
    return out


def load_lq_v2_accepted(indic_root: Path) -> list[dict[str, Any]]:
    path = indic_root / LQ_V2_PACK_REL / "verification" / "accepted.jsonl"
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            raw = json.loads(line)
            rows.append(normalize_lq_v2_row(raw, indic_root=indic_root))
    return rows
