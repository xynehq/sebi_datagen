"""Gold eligibility validation without export.

Only human_verified and human_corrected rows may become gold. This module
checks proposed gold rows and never writes data/output/gold.jsonl.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from indic_ocr_pipeline.errors import GoldPolicyError
from indic_ocr_pipeline.ids import make_sample_id
from indic_ocr_pipeline.provenance import (
    find_conflicting_gold_labels,
    find_document_leakage,
    find_duplicate_values,
    find_parent_line_leakage,
    missing_provenance_fields,
)
from indic_ocr_pipeline.schemas import (
    VerificationStatus,
    admits_to_gold,
)
from indic_ocr_pipeline.text_policy import (
    apply_storage_normalization,
    has_replacement_character,
    is_storage_form,
)
from indic_ocr_pipeline.utils.hashing import sha256_file

GOLD_ELIGIBLE_STATUSES = frozenset(
    {
        VerificationStatus.HUMAN_VERIFIED.value,
        VerificationStatus.HUMAN_CORRECTED.value,
    }
)

GOLD_INELIGIBLE_STATUSES = frozenset(
    {
        VerificationStatus.CANDIDATE.value,
        VerificationStatus.AUTO_APPROVED_CANDIDATE.value,
        VerificationStatus.REJECTED.value,
        "pending",
        "bad_boundary",
    }
)


def is_gold_eligible_status(status: str | VerificationStatus) -> bool:
    """Return whether a verification status may enter gold."""
    return admits_to_gold(status)


def is_gold_ineligible_status(status: str) -> bool:
    """Return whether a status is explicitly barred from gold."""
    if status in GOLD_INELIGIBLE_STATUSES:
        return True
    if status == "bad_boundary":
        return True
    return not admits_to_gold(status)


def proposed_gold_from_verified(
    verified: Mapping[str, Any],
    *,
    crop: Mapping[str, Any],
    page: Mapping[str, Any],
    region_id: str | None = None,
) -> dict[str, Any]:
    """Build one proposed gold payload. Does not write gold."""
    status = str(verified.get("verification_status"))
    decision_label = None
    meta = verified.get("metadata") if isinstance(verified.get("metadata"), dict) else {}
    decision_label = meta.get("decision_label")
    if decision_label == "bad_boundary" or status not in GOLD_ELIGIBLE_STATUSES:
        raise GoldPolicyError(
            f"verified row {verified.get('crop_id')} is not gold-eligible ({status})"
        )
    gold_text = verified.get("final_text")
    if not isinstance(gold_text, str) or not gold_text:
        raise GoldPolicyError(f"missing gold_text for {verified.get('crop_id')}")
    crop_id = str(verified["crop_id"])
    resolved_region = region_id or crop.get("region_id") or meta.get("region_id")
    if not isinstance(resolved_region, str) or not resolved_region:
        raise GoldPolicyError(f"missing region_id for {crop_id}")
    return {
        "sample_id": make_sample_id(crop_id),
        "image_path": crop.get("image_path"),
        "gold_text": gold_text,
        "document_id": verified["document_id"],
        "page_id": verified["page_id"],
        "page_number": page.get("page_number"),
        "region_id": resolved_region,
        "parent_line_id": verified["parent_line_id"],
        "crop_id": crop_id,
        "crop_type": crop.get("crop_type"),
        "bbox": crop.get("bbox"),
        "verification_status": status,
        "source_sha256": verified.get("source_sha256") or crop.get("sha256"),
        "original_candidate_texts": verified.get("original_candidate_texts") or {},
        "verification_timestamp": verified.get("verification_timestamp"),
        "reviewer_id": verified.get("reviewer_id"),
        "tier": "tier_5_gold",
    }


def validate_proposed_gold_row(
    row: Mapping[str, Any],
    *,
    repo_root: Path | None = None,
    crop: Mapping[str, Any] | None = None,
    page: Mapping[str, Any] | None = None,
    document: Mapping[str, Any] | None = None,
    check_image_bytes: bool = True,
) -> list[str]:
    """Return validation errors for one proposed gold row. Empty means pass."""
    errors: list[str] = []
    status = str(row.get("verification_status") or "")
    if status not in GOLD_ELIGIBLE_STATUSES:
        errors.append(f"ineligible verification_status={status}")
    gold_text = row.get("gold_text")
    if gold_text is None or gold_text == "":
        errors.append("missing gold_text")
    elif not isinstance(gold_text, str):
        errors.append("gold_text is not a string")
    else:
        if not is_storage_form(gold_text):
            errors.append("gold_text is not Unicode NFC storage form")
        if has_replacement_character(gold_text):
            errors.append("gold_text contains U+FFFD")
        if apply_storage_normalization(gold_text) != gold_text:
            errors.append("gold_text failed storage normalization identity")
    image_path = row.get("image_path")
    if not isinstance(image_path, str) or not image_path:
        errors.append("missing image_path")
    root = Path(repo_root) if repo_root is not None else None
    crop_path: Path | None = None
    if isinstance(image_path, str) and image_path:
        crop_path = Path(image_path)
        if root is not None and not crop_path.is_absolute():
            crop_path = root / crop_path
        if check_image_bytes and (crop_path is None or not crop_path.is_file()):
            errors.append("missing crop image")
    expected_sha = row.get("source_sha256")
    if check_image_bytes and crop_path is not None and crop_path.is_file():
        actual = sha256_file(crop_path)
        if isinstance(expected_sha, str) and expected_sha and expected_sha != actual:
            errors.append("crop SHA-256 mismatch")
        if crop is not None and isinstance(crop.get("sha256"), str):
            if crop["sha256"] != actual:
                errors.append("crop manifest SHA-256 mismatch")
    if page is None:
        errors.append("missing page")
    if document is None:
        errors.append("missing source document record")
    if document is not None and not document.get("source_path") and not document.get("sha256"):
        errors.append("document provenance incomplete")
    for field_name in (
        "document_id",
        "page_id",
        "parent_line_id",
        "crop_id",
        "sample_id",
        "region_id",
    ):
        if not row.get(field_name):
            errors.append(f"missing {field_name}")
    provenance_missing = missing_provenance_fields(row)
    for field_name in provenance_missing:
        message = f"missing provenance field {field_name}"
        if message not in errors and f"missing {field_name}" not in errors:
            errors.append(message)
    return errors


def validate_gold_pool(
    proposed: Sequence[Mapping[str, Any]],
    *,
    repo_root: Path | None = None,
    crops: Mapping[str, Mapping[str, Any]] | None = None,
    pages: Mapping[str, Mapping[str, Any]] | None = None,
    documents: Mapping[str, Mapping[str, Any]] | None = None,
    check_image_bytes: bool = True,
) -> dict[str, Any]:
    """Validate a proposed gold pool without writing gold.jsonl."""
    crops = crops or {}
    pages = pages or {}
    documents = documents or {}
    row_errors: dict[str, list[str]] = {}
    for row in proposed:
        crop_id = str(row.get("crop_id") or row.get("sample_id") or "?")
        errors = validate_proposed_gold_row(
            row,
            repo_root=repo_root,
            crop=crops.get(str(row.get("crop_id"))),
            page=pages.get(str(row.get("page_id"))),
            document=documents.get(str(row.get("document_id"))),
            check_image_bytes=check_image_bytes,
        )
        if errors:
            row_errors[crop_id] = errors
    duplicates = find_duplicate_values(proposed, "sample_id")
    image_dupes = find_duplicate_values(proposed, "image_path")
    conflicts = find_conflicting_gold_labels(proposed)
    # Document split readiness uses ids only; no split assignment yet.
    split_ready = document_split_readiness(proposed)
    ineligible = [
        str(row.get("crop_id"))
        for row in proposed
        if str(row.get("verification_status")) not in GOLD_ELIGIBLE_STATUSES
    ]
    return {
        "proposed_count": len(proposed),
        "ok": not row_errors and not duplicates and not conflicts and not ineligible,
        "row_errors": row_errors,
        "duplicate_sample_ids": duplicates,
        "duplicate_image_paths": image_dupes,
        "conflicting_gold_labels": {
            path: sorted(texts) for path, texts in conflicts.items()
        },
        "ineligible_rows": ineligible,
        "document_split_readiness": split_ready,
        "gold_export": False,
        "auto_approval_enabled": False,
    }


def collect_eligible_verified(
    verified_rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Return verified rows that are conceptually gold-eligible."""
    eligible: list[Mapping[str, Any]] = []
    for row in verified_rows:
        status = str(row.get("verification_status"))
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        if meta.get("decision_label") == "bad_boundary":
            continue
        if status in GOLD_ELIGIBLE_STATUSES and row.get("final_text"):
            eligible.append(row)
    return eligible


def document_split_readiness(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Verify future document-level splits have the required identifiers."""
    missing_document = []
    missing_page = []
    missing_line = []
    by_document: dict[str, int] = Counter()
    for row in rows:
        document_id = row.get("document_id")
        page_id = row.get("page_id")
        line_id = row.get("parent_line_id")
        if not isinstance(document_id, str) or not document_id:
            missing_document.append(str(row.get("crop_id")))
        else:
            by_document[document_id] += 1
        if not isinstance(page_id, str) or not page_id:
            missing_page.append(str(row.get("crop_id")))
        if not isinstance(line_id, str) or not line_id:
            missing_line.append(str(row.get("crop_id")))
    # No split field yet — leakage check over empty splits must be clean.
    leakage_docs = find_document_leakage(rows)
    leakage_lines = find_parent_line_leakage(rows)
    return {
        "ready": not missing_document and not missing_page and not missing_line,
        "documents": dict(sorted(by_document.items())),
        "missing_document_id": missing_document,
        "missing_page_id": missing_page,
        "missing_parent_line_id": missing_line,
        "document_leakage_if_split_assigned": {
            key: sorted(value) for key, value in leakage_docs.items()
        },
        "parent_line_leakage_if_split_assigned": {
            key: sorted(value) for key, value in leakage_lines.items()
        },
        "split_policy": "document_level_only",
        "random_line_split_allowed": False,
    }


def assert_no_automatic_gold_from_agreement() -> None:
    """Policy guard: agreement never creates gold by itself."""
    raise GoldPolicyError("agreement is never sufficient to create gold")


def assert_no_automatic_gold_from_low_risk() -> None:
    """Policy guard: low risk never creates gold by itself."""
    raise GoldPolicyError("low risk score is never sufficient to create gold")
