"""Provenance and leakage checks over manifest rows.

These functions report problems. They do not rewrite labels.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping

from indic_ocr_pipeline.errors import GoldPolicyError

REQUIRED_GOLD_FIELDS = (
    "sample_id",
    "image_path",
    "gold_text",
    "document_id",
    "page_id",
    "page_number",
    "region_id",
    "parent_line_id",
    "crop_id",
    "crop_type",
    "bbox",
    "verification_status",
    "source_sha256",
    "verification_timestamp",
    "original_candidate_texts",
)


def missing_provenance_fields(record: Mapping[str, object]) -> list[str]:
    """Return required gold fields that are missing or empty."""
    missing: list[str] = []
    for field_name in REQUIRED_GOLD_FIELDS:
        value = record.get(field_name)
        if value is None or value == "" or value == [] or value == {}:
            missing.append(field_name)
    return missing


def find_grouped_leakage(
    records: Iterable[Mapping[str, object]],
    key_field: str,
    split_field: str = "split",
) -> dict[str, set[str]]:
    """Return keys that appear in more than one split."""
    grouped: dict[str, set[str]] = {}
    for record in records:
        key = record.get(key_field)
        split = record.get(split_field)
        if not isinstance(key, str) or not key:
            continue
        if not isinstance(split, str) or not split:
            continue
        grouped.setdefault(key, set()).add(split)
    return {key: splits for key, splits in grouped.items() if len(splits) > 1}


def find_document_leakage(
    records: Iterable[Mapping[str, object]],
) -> dict[str, set[str]]:
    """Return document IDs that occur in more than one split."""
    return find_grouped_leakage(records, "document_id")


def find_parent_line_leakage(
    records: Iterable[Mapping[str, object]],
) -> dict[str, set[str]]:
    """Return parent line IDs that occur in more than one split."""
    return find_grouped_leakage(records, "parent_line_id")


def assert_no_split_leakage(records: Iterable[Mapping[str, object]]) -> None:
    """Raise when a document or parent line crosses train, validation, and test."""
    materialized = list(records)
    documents = find_document_leakage(materialized)
    lines = find_parent_line_leakage(materialized)
    if not documents and not lines:
        return
    parts: list[str] = []
    if documents:
        parts.append(f"documents={sorted(documents)}")
    if lines:
        parts.append(f"parent_lines={sorted(lines)}")
    raise GoldPolicyError("split leakage detected: " + "; ".join(parts))


def find_duplicate_values(
    records: Iterable[Mapping[str, object]],
    field_name: str,
) -> dict[str, int]:
    """Return values of ``field_name`` that occur more than once."""
    counts: Counter[str] = Counter()
    for record in records:
        value = record.get(field_name)
        if isinstance(value, str) and value:
            counts[value] += 1
    return {value: count for value, count in counts.items() if count > 1}


def find_conflicting_gold_labels(
    records: Iterable[Mapping[str, object]],
) -> dict[str, set[str]]:
    """Return image paths that have more than one gold text."""
    grouped: dict[str, set[str]] = {}
    for record in records:
        image_path = record.get("image_path")
        gold_text = record.get("gold_text")
        if not isinstance(image_path, str) or not image_path:
            continue
        if not isinstance(gold_text, str):
            continue
        grouped.setdefault(image_path, set()).add(gold_text)
    return {image_path: texts for image_path, texts in grouped.items() if len(texts) > 1}


def find_sha256_collisions(
    records: Iterable[Mapping[str, object]],
) -> dict[str, list[str]]:
    """Return SHA-256 values assigned to more than one document ID."""
    grouped: dict[str, list[str]] = {}
    for record in records:
        sha256 = record.get("sha256")
        document_id = record.get("document_id")
        if not isinstance(sha256, str) or not isinstance(document_id, str):
            continue
        documents = grouped.setdefault(sha256, [])
        if document_id not in documents:
            documents.append(document_id)
    return {sha256: documents for sha256, documents in grouped.items() if len(documents) > 1}
