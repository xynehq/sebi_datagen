"""Record review decisions and leave gold export to the next pipeline stage."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pipeline.dataset_flow import read_jsonl, write_jsonl
from pipeline.gold import BLOCKED_REVIEWERS
from pipeline.records import records_dir
from pipeline.review_pack import DECISIONS_NAME, PACK_DIR, default_decisions


class AcceptError(ValueError):
    """A review file could not be recorded."""


def accept_reviews(
    root: Path,
    *,
    decisions_path: Path | None = None,
    reviewer_id: str = "operator",
) -> int:
    """Write verified rows from the decisions file. Does not start training."""
    root = root.resolve()
    reviewer = reviewer_id.strip()
    if not reviewer or reviewer.lower() in BLOCKED_REVIEWERS:
        print(
            "reviewer_id must be a person. cursor, agent, assistant, and auto are refused.",
        )
        return 2
    rows = read_jsonl(records_dir(root) / "consensus.jsonl")
    if not rows:
        print("No consensus rows to accept. Run through calibrate first.")
        return 2
    path = decisions_path or (root / PACK_DIR / DECISIONS_NAME)
    if not path.is_file():
        payload = default_decisions(rows, reviewer)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    by_crop = {str(row["crop_id"]): row for row in rows}
    decisions = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(decisions, list) or not decisions:
        print(f"{path} has no rows.")
        return 2
    file_reviewer = payload.get("reviewer_id") if isinstance(payload, dict) else None
    if isinstance(file_reviewer, str) and file_reviewer.strip():
        reviewer = file_reviewer.strip()
        if reviewer.lower() in BLOCKED_REVIEWERS:
            print(
                "reviewer_id must be a person. cursor, agent, assistant, and auto are refused.",
            )
            return 2
    verified_path = records_dir(root) / "verified.jsonl"
    existing = {str(row.get("crop_id")): row for row in read_jsonl(verified_path)}
    written = 0
    skipped = 0
    pending = 0
    now = datetime.now(UTC).isoformat()
    for item in decisions:
        if not isinstance(item, dict) or not item.get("crop_id"):
            raise AcceptError(f"decision row is missing crop_id: {item!r}")
        label = str(item.get("decision") or "none").strip().lower()
        if label in {"none", "pending"}:
            pending += 1
            continue
        crop_id = str(item["crop_id"])
        source = by_crop.get(crop_id)
        if source is None:
            raise AcceptError(f"{crop_id} is not in the consensus manifest")
        record = _verified_row(source, item, reviewer, now)
        previous = existing.get(crop_id)
        if previous is not None:
            if _same_decision(previous, record):
                skipped += 1
                continue
            raise AcceptError(
                f"{crop_id} already has a different verified decision. "
                "The existing row was kept."
            )
        existing[crop_id] = record
        written += 1
    if written or existing:
        write_jsonl(verified_path, list(existing.values()))
        print(f"verified: {verified_path}")
    print(
        f"recorded {written} decisions, skipped {skipped} unchanged, "
        f"{pending} still unreviewed"
    )
    return 0


def _verified_row(
    source: dict[str, Any],
    item: dict[str, Any],
    reviewer_id: str,
    timestamp: str,
) -> dict[str, Any]:
    label = str(item.get("decision") or "accept").strip().lower()
    text = item.get("text")
    stored = text.strip() if isinstance(text, str) else ""
    predictions = source.get("predictions") if isinstance(source.get("predictions"), dict) else {}
    if label in {"accept", "human_verified"}:
        if not stored:
            raise AcceptError(f"{source.get('crop_id')} accept requires text")
        decision = "human_verified"
        gold_text: str | None = stored
    elif label in {"correct", "human_corrected"}:
        if not stored:
            raise AcceptError(f"{source.get('crop_id')} correct requires text")
        decision = "human_corrected"
        gold_text = stored
    elif label in {"reject", "rejected"}:
        decision = "rejected"
        gold_text = None
    else:
        raise AcceptError(f"unknown decision {label!r}")
    crop_id = str(source["crop_id"])
    return {
        "crop_id": crop_id,
        "document_id": source["document_id"],
        "page_id": source["page_id"],
        "parent_line_id": source.get("line_id") or source.get("parent_line_id"),
        "line_id": source.get("line_id") or source.get("parent_line_id"),
        "image_path": source.get("crop_path") or source.get("image_path"),
        "crop_path": source.get("crop_path") or source.get("image_path"),
        "reviewer_id": reviewer_id,
        "verification_timestamp": timestamp,
        "reviewed_at": timestamp,
        "review_origin": "human",
        "verification_status": decision,
        "decision": decision,
        "final_text": gold_text,
        "gold_text": gold_text,
        "metadata": {
            "decision_label": decision,
            "review_origin": "human",
            "example_run": True,
        },
        "original_candidate_texts": {
            "indicocr": predictions.get("indicocr"),
            "chitrapathak": predictions.get("chitrapathak"),
            "tesseract": predictions.get("tesseract"),
        },
    }


def _same_decision(previous: dict[str, Any], record: dict[str, Any]) -> bool:
    return (
        previous.get("crop_id") == record.get("crop_id")
        and previous.get("decision") == record.get("decision")
        and previous.get("final_text") == record.get("final_text")
        and previous.get("reviewer_id") == record.get("reviewer_id")
    )
