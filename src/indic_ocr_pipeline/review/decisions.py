"""Commit immutable human review decisions.

Verified rows are append-only. The same verified_id with the same payload is a
resume skip. A different payload is a conflict and is never overwritten.

UI labels map to schema values:

* human_verified  → verification_status=human_verified, decision=accept_candidate
* human_corrected → verification_status=human_corrected, decision=correct_text
* rejected        → verification_status=rejected, decision=reject_crop
* bad_boundary    → verification_status=rejected, decision=bad_boundary

``final_text`` is the storage-form transcription. It is not gold until a later
export copies human_verified / human_corrected rows.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from indic_ocr_pipeline.errors import DatasetError, ManifestConflictError, SchemaError
from indic_ocr_pipeline.ids import make_review_id, make_verified_id
from indic_ocr_pipeline.manifests import JsonlManifest, open_manifest
from indic_ocr_pipeline.pdf.ingest import utc_now
from indic_ocr_pipeline.schemas import (
    QualityTier,
    ReviewDecision,
    ReviewRecord,
    StageStatus,
    VerificationStatus,
    VerifiedRecord,
)
from indic_ocr_pipeline.text_policy import apply_storage_normalization
from indic_ocr_pipeline.utils.hashing import sha256_file

REVIEWER_APP_VERSION = "phase8-review-1"

UI_DECISIONS = frozenset(
    {"human_verified", "human_corrected", "rejected", "bad_boundary"}
)


class ReviewDecisionError(DatasetError):
    """A human review decision could not be recorded."""


def map_ui_decision(label: str) -> tuple[VerificationStatus, ReviewDecision]:
    """Map a UI decision label onto schema enums."""
    if label == "human_verified":
        return VerificationStatus.HUMAN_VERIFIED, ReviewDecision.ACCEPT_CANDIDATE
    if label == "human_corrected":
        return VerificationStatus.HUMAN_CORRECTED, ReviewDecision.CORRECT_TEXT
    if label == "rejected":
        return VerificationStatus.REJECTED, ReviewDecision.REJECT_CROP
    if label == "bad_boundary":
        return VerificationStatus.REJECTED, ReviewDecision.BAD_BOUNDARY
    raise ReviewDecisionError(f"unknown review decision {label!r}")


def _data_status_for_decision(label: str) -> str:
    if label == "human_verified":
        return "human_verified"
    if label == "human_corrected":
        return "human_corrected"
    if label in {"rejected", "bad_boundary"}:
        return "rejected"
    return "needs_human_review"


def prepare_gold_text(text: str | None, *, decision_label: str) -> str | None:
    """Return storage-form text for accept/correct decisions, else null."""
    if decision_label in {"rejected", "bad_boundary"}:
        if text is not None and str(text).strip() != "":
            raise ReviewDecisionError(f"{decision_label} cannot store gold_text")
        return None
    if text is None or not isinstance(text, str):
        raise ReviewDecisionError(f"{decision_label} requires gold_text")
    stored = apply_storage_normalization(text)
    if not stored:
        raise ReviewDecisionError(f"{decision_label} requires non-empty gold_text")
    return stored


def build_verified_record(
    *,
    crop: Mapping[str, Any],
    review: Mapping[str, Any],
    decision_label: str,
    gold_text: str | None,
    selected_candidate_text: str | None,
    reviewer_id: str,
    notes: str | None,
    error_tags: list[str] | None,
    candidate_texts: Mapping[str, str | None],
    timestamp: str | None = None,
) -> VerifiedRecord:
    """Build one verified decision. Does not write gold."""
    if decision_label not in UI_DECISIONS:
        raise ReviewDecisionError(f"unknown review decision {decision_label!r}")
    status, decision = map_ui_decision(decision_label)
    stored = prepare_gold_text(gold_text, decision_label=decision_label)
    if decision is ReviewDecision.ACCEPT_CANDIDATE:
        selected = prepare_gold_text(
            selected_candidate_text if selected_candidate_text is not None else gold_text,
            decision_label="human_verified",
        )
        if selected != stored:
            raise ReviewDecisionError(
                "human_verified requires gold_text equal to the selected candidate text"
            )
    diagnostics = review.get("diagnostics") if isinstance(review.get("diagnostics"), dict) else {}
    image_path = crop.get("image_path") or diagnostics.get("image_path")
    if not isinstance(image_path, str) or not image_path:
        raise ReviewDecisionError(f"crop {crop.get('crop_id')} has no image_path")
    from pathlib import Path

    path = Path(image_path)
    if not path.is_absolute():
        # Caller may resolve; hash only when the file exists at this path.
        pass
    source_sha = crop.get("sha256")
    if not isinstance(source_sha, str) or len(source_sha) != 64:
        if path.is_file():
            source_sha = sha256_file(path)
        else:
            raise ReviewDecisionError(f"missing crop sha256 for {crop.get('crop_id')}")
    metadata = {
        "review_id": make_review_id(str(crop["crop_id"])),
        "region_id": crop.get("region_id") or diagnostics.get("region_id"),
        "decision_label": decision_label,
        "previous_review_risk_score": review.get("risk_score"),
        "agreement_category": diagnostics.get("agreement_category"),
        "review_priority": diagnostics.get("review_priority"),
        "reviewer_app_version": REVIEWER_APP_VERSION,
        "error_tags": list(error_tags or []),
        "auto_approval_enabled": False,
        "gold_export": False,
        "data_status": _data_status_for_decision(decision_label),
        "gold": decision_label in {"human_verified", "human_corrected"},
        # New decisions are human-origin. Historical rows lack this field and
        # cannot enter the training gold manifest.
        "review_origin": "human",
    }
    return VerifiedRecord(
        verified_id=make_verified_id(str(crop["crop_id"])),
        crop_id=str(crop["crop_id"]),
        document_id=str(crop["document_id"]),
        page_id=str(crop["page_id"]),
        parent_line_id=str(crop["parent_line_id"]),
        verification_status=status,
        final_text=stored,
        review_decision=decision,
        original_candidate_texts={
            "indicocr": candidate_texts.get("indicocr"),
            "chitrapathak": candidate_texts.get("chitrapathak"),
            "tesseract": candidate_texts.get("tesseract"),
        },
        source_sha256=source_sha,
        verification_timestamp=timestamp or utc_now(),
        reviewer_id=reviewer_id,
        notes=notes,
        metadata=metadata,
        tier=QualityTier.TIER_4_HUMAN_REVIEWED,
    )


def commit_decision(
    manifests_dir,
    *,
    crop_id: str,
    decision_label: str,
    gold_text: str | None,
    selected_candidate_text: str | None,
    reviewer_id: str,
    notes: str | None = None,
    error_tags: list[str] | None = None,
    repo_root=None,
) -> dict[str, Any]:
    """Append a verified decision and mark the review row complete.

    Historical verified rows are never rewritten. Changing a decision requires a
    different payload and raises ``ManifestConflictError``.
    """
    from pathlib import Path

    manifests_dir = Path(manifests_dir)
    root = Path(repo_root) if repo_root is not None else manifests_dir.parent
    review_manifest = open_manifest(manifests_dir, "review")
    crop_manifest = open_manifest(manifests_dir, "crops")
    verified_manifest = open_manifest(manifests_dir, "verified")
    review = review_manifest.get(make_review_id(crop_id))
    if review is None:
        raise ReviewDecisionError(f"missing review row for {crop_id}")
    crop = crop_manifest.get(crop_id)
    if crop is None:
        raise ReviewDecisionError(f"missing crop {crop_id}")
    diagnostics = review.get("diagnostics") if isinstance(review.get("diagnostics"), dict) else {}
    image_rel = crop.get("image_path") or diagnostics.get("image_path")
    if isinstance(image_rel, str):
        image_path = Path(image_rel)
        if not image_path.is_absolute():
            image_path = root / image_path
        crop = dict(crop)
        crop["image_path"] = str(image_path)
        if image_path.is_file() and not crop.get("sha256"):
            crop["sha256"] = sha256_file(image_path)
        elif not image_path.is_file() and not crop.get("sha256"):
            raise ReviewDecisionError(f"missing crop image {image_path}")
    texts = {
        "indicocr": diagnostics.get("indicocr_text"),
        "chitrapathak": diagnostics.get("chitrapathak_text"),
        "tesseract": diagnostics.get("tesseract_text"),
    }
    existing = verified_manifest.get(make_verified_id(crop_id))
    record = build_verified_record(
        crop=crop,
        review=review,
        decision_label=decision_label,
        gold_text=gold_text,
        selected_candidate_text=selected_candidate_text,
        reviewer_id=reviewer_id,
        notes=notes,
        error_tags=error_tags,
        candidate_texts=texts,
        timestamp=None if existing is None else str(existing.get("verification_timestamp")),
    )
    payload = record.to_dict()
    if existing is not None:
        if existing == payload:
            _mark_review_complete(
                review_manifest,
                review,
                record,
                selected_candidate_text=(
                    record.final_text
                    if decision_label == "human_verified"
                    else selected_candidate_text
                ),
            )
            return {"outcome": "skipped", "verified": existing}
        raise ManifestConflictError(
            f"verified_id={payload['verified_id']} already exists with different content"
        )
    try:
        outcome = verified_manifest.append(payload)
    except ManifestConflictError:
        raise
    except SchemaError as exc:
        raise ReviewDecisionError(str(exc)) from exc
    _mark_review_complete(
        review_manifest,
        review,
        record,
        selected_candidate_text=(
            record.final_text if decision_label == "human_verified" else selected_candidate_text
        ),
    )
    return {"outcome": outcome, "verified": payload}


def _mark_review_complete(
    review_manifest: JsonlManifest,
    review: Mapping[str, Any],
    verified: VerifiedRecord,
    *,
    selected_candidate_text: str | None,
) -> None:
    review_id = str(review["review_id"])
    current = review_manifest.get(review_id)
    if current is None:
        raise ReviewDecisionError(f"missing review row {review_id}")
    if current.get("status") == StageStatus.COMPLETE.value:
        return
    selected = (
        verified.final_text
        if verified.review_decision is ReviewDecision.ACCEPT_CANDIDATE
        else selected_candidate_text
    )
    if isinstance(selected, str):
        selected = apply_storage_normalization(selected) or None
    changes = {
        "decision": verified.review_decision.value,
        "final_text": verified.final_text,
        "selected_candidate_text": selected,
        "reviewer_id": verified.reviewer_id,
        "reviewed_at": verified.verification_timestamp,
        "notes": verified.notes,
        "audit_sample": True,
    }
    complete_view = {
        **current,
        **changes,
        "status": StageStatus.COMPLETE.value,
    }
    ReviewRecord.from_dict(complete_view)
    if current.get("status") == StageStatus.PENDING.value:
        review_manifest.update(
            review_id,
            {**changes, "status": StageStatus.RUNNING.value},
        )
    elif current.get("status") == StageStatus.RUNNING.value:
        review_manifest.update(review_id, changes)
    review_manifest.update(review_id, {"status": StageStatus.COMPLETE.value})
