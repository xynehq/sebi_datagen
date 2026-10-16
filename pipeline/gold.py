"""Gold rows come only from human decisions.

Historical Phase 10 review files are preserved and marked INVALIDATED_REVIEW.
Consensus statuses AUTO_3_AGREE and AUTO_2_AGREE are not gold.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pipeline.records import records_dir

ARCHIVE_DIR = Path("data/review/archive/invalidated_phase10")
STATUS = "INVALIDATED_REVIEW"
GOLD_DECISIONS = frozenset({"HUMAN_VERIFIED", "HUMAN_CORRECTED"})
BLOCKED_REVIEWERS = frozenset({"cursor", "agent", "assistant", "auto"})

_DECISION_MAP = {
    "human_verified": "HUMAN_VERIFIED",
    "accept_candidate": "HUMAN_VERIFIED",
    "HUMAN_VERIFIED": "HUMAN_VERIFIED",
    "human_corrected": "HUMAN_CORRECTED",
    "correct_text": "HUMAN_CORRECTED",
    "HUMAN_CORRECTED": "HUMAN_CORRECTED",
    "rejected": "HUMAN_REJECTED",
    "reject_crop": "HUMAN_REJECTED",
    "HUMAN_REJECTED": "HUMAN_REJECTED",
    "bad_boundary": "HUMAN_BAD_BOUNDARY",
    "HUMAN_BAD_BOUNDARY": "HUMAN_BAD_BOUNDARY",
}


class GoldError(ValueError):
    """A row cannot enter the gold or training manifest."""


def decision_of(row: Mapping[str, Any]) -> str:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    raw = (
        row.get("decision")
        or meta.get("decision_label")
        or row.get("review_decision")
        or row.get("verification_status")
        or ""
    )
    return _DECISION_MAP.get(str(raw), str(raw).upper())


def ensure_invalidated_archive(root: Path, verified_path: Path | None = None) -> set[str]:
    """Snapshot historical verified rows once. Later human decisions stay eligible.

    The original verified file is copied, not deleted.
    """
    archive = root / ARCHIVE_DIR
    archive.mkdir(parents=True, exist_ok=True)
    id_path = archive / "invalidated_ids.json"
    if id_path.is_file():
        payload = json.loads(id_path.read_text(encoding="utf-8"))
        return {str(item["crop_id"]) for item in payload["rows"]}

    source = verified_path or (records_dir(root) / "verified.jsonl")
    rows: list[dict[str, Any]] = []
    if source.is_file():
        shutil.copy2(source, archive / "verified.jsonl")
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            meta = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
            origin = raw.get("review_origin") or meta.get("review_origin")
            if origin == "human":
                continue
            rows.append(
                {
                    "crop_id": raw.get("crop_id"),
                    "verified_id": raw.get("verified_id"),
                    "document_id": raw.get("document_id"),
                    "reviewer_id": raw.get("reviewer_id"),
                    "decision": decision_of(raw),
                    "verification_status": raw.get("verification_status"),
                    "reviewer_app_version": meta.get("reviewer_app_version"),
                    "status": STATUS,
                }
            )
    marker = {
        "status": STATUS,
        "reason": (
            "Phase 10 review decisions were generated without a human reviewer. "
            "They are preserved for audit and are not gold."
        ),
        "source": str(source) if source.is_file() else None,
        "rows": rows,
    }
    id_path.write_text(json.dumps(marker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readme = archive / "README.md"
    if not readme.exists():
        readme.write_text(
            "\n".join(
                [
                    "# INVALIDATED_REVIEW",
                    "",
                    "These rows are historical review artifacts.",
                    "They are not human gold and must not be merged into training.",
                    "The original `data/output/verified.jsonl` is left in place.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    return {str(item["crop_id"]) for item in rows if item.get("crop_id")}


def load_invalidated_ids(root: Path) -> set[str]:
    path = root / ARCHIVE_DIR / "invalidated_ids.json"
    if not path.is_file():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(item["crop_id"]) for item in payload.get("rows", []) if item.get("crop_id")}


def gold_rejection_reason(row: Mapping[str, Any], invalidated_ids: set[str]) -> str | None:
    """Return why a row is not gold, or None when it is admissible."""
    crop_id = str(row.get("crop_id") or "")
    if crop_id and crop_id in invalidated_ids:
        return STATUS
    if str(row.get("status") or "") == STATUS:
        return STATUS
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    origin = row.get("review_origin") or meta.get("review_origin")
    if origin != "human":
        return "missing human review origin"
    reviewer = str(row.get("reviewer_id") or "")
    if not reviewer or reviewer.strip().lower() in BLOCKED_REVIEWERS:
        return "missing or blocked reviewer_id"
    reviewed_at = row.get("reviewed_at") or row.get("verification_timestamp")
    if not reviewed_at:
        return "missing reviewed_at"
    decision = decision_of(row)
    if decision not in GOLD_DECISIONS:
        return f"decision {decision} is not gold"
    gold_text = row.get("gold_text")
    if gold_text is None:
        gold_text = row.get("final_text")
    if not isinstance(gold_text, str) or not gold_text.strip():
        return "missing gold_text"
    for field in ("document_id", "page_id"):
        if not row.get(field):
            return f"missing {field}"
    line_id = row.get("line_id") or row.get("parent_line_id")
    if not line_id:
        return "missing line_id"
    crop_path = row.get("crop_path") or row.get("image_path")
    if not crop_path:
        return "missing crop_path"
    originals = (
        row.get("original_ocr")
        or row.get("original_candidate_texts")
        or row.get("predictions")
    )
    if not isinstance(originals, Mapping):
        return "missing original OCR outputs"
    return None


def to_gold_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical gold record. Caller has already admitted the row."""
    gold_text = row.get("gold_text")
    if gold_text is None:
        gold_text = row.get("final_text")
    originals = (
        row.get("original_ocr")
        or row.get("original_candidate_texts")
        or row.get("predictions")
        or {}
    )
    crop_id = str(row["crop_id"])
    return {
        "sample_id": row.get("sample_id") or f"{crop_id}_S",
        "document_id": row["document_id"],
        "page_id": row["page_id"],
        "line_id": row.get("line_id") or row.get("parent_line_id"),
        "crop_id": crop_id,
        "crop_path": row.get("crop_path") or row.get("image_path"),
        "image_filename": (
            row.get("image_filename") or row.get("crop_path") or row.get("image_path")
        ),
        "gold_text": gold_text,
        "expected_text": gold_text,
        "reviewer_id": row["reviewer_id"],
        "reviewed_at": row.get("reviewed_at") or row.get("verification_timestamp"),
        "decision": decision_of(row),
        "verification_status": decision_of(row),
        "review_origin": "human",
        "verification_method": "human",
        "human_verified_text": gold_text,
        "data_status": (
            "human_corrected" if decision_of(row) == "HUMAN_CORRECTED" else "human_verified"
        ),
        "original_ocr": dict(originals),
        "source": "human_review",
        "gold": True,
    }


def select_gold_rows(
    rows: Sequence[Mapping[str, Any]],
    invalidated_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split rows into gold records and rejected records with reasons."""
    gold: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in rows:
        reason = gold_rejection_reason(row, invalidated_ids)
        if reason is None:
            gold.append(to_gold_row(row))
        else:
            rejected.append(
                {
                    "crop_id": row.get("crop_id"),
                    "status": STATUS if reason == STATUS else "rejected",
                    "reason": reason,
                }
            )
    return gold, rejected
