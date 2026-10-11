"""Deterministic document, page, line, crop, and sample IDs."""

from __future__ import annotations

import pytest

from indic_ocr_pipeline.errors import ManifestConflictError
from indic_ocr_pipeline.ids import (
    DocumentIdAllocator,
    document_id_from_identifier,
    format_document_id,
    make_candidate_id,
    make_crop_id,
    make_line_id,
    make_page_id,
    make_prediction_id,
    make_region_id,
    make_review_id,
    make_sample_id,
    parent_line_id_from_crop_id,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def test_document_ids_are_fixed_width() -> None:
    assert format_document_id(1) == "MHGR_000001"
    assert format_document_id(42) == "MHGR_000042"


def test_document_sequence_rejects_zero_and_bool() -> None:
    with pytest.raises(ValueError):
        format_document_id(0)
    with pytest.raises(ValueError):
        format_document_id(True)  # type: ignore[arg-type]


def test_page_region_line_and_crop_ids_are_pure_functions() -> None:
    page_id = make_page_id("MHGR_000001", 3)
    assert page_id == "MHGR_000001_P0003"
    region_id = make_region_id(page_id, 1)
    line_id = make_line_id(page_id, 12)
    crop_id = make_crop_id(line_id, "full", 1)
    assert region_id == "MHGR_000001_P0003_R0001"
    assert line_id == "MHGR_000001_P0003_L0012"
    assert crop_id == "MHGR_000001_P0003_L0012_C_full_01"
    assert make_sample_id(crop_id) == f"SAMPLE_{crop_id}"
    assert make_prediction_id(crop_id, "tesseract") == f"{crop_id}_OCR_tesseract"
    assert make_candidate_id(crop_id) == f"{crop_id}_CAND"
    assert make_review_id(crop_id) == f"{crop_id}_REV"
    assert parent_line_id_from_crop_id(crop_id) == line_id
    assert document_id_from_identifier(make_sample_id(crop_id)) == "MHGR_000001"


def test_page_numbers_are_one_based() -> None:
    with pytest.raises(ValueError):
        make_page_id("MHGR_000001", 0)


def test_same_inputs_repeat_the_same_ids() -> None:
    first = make_crop_id(make_line_id(make_page_id("MHGR_000007", 2), 4), "left", 1)
    second = make_crop_id(make_line_id(make_page_id("MHGR_000007", 2), 4), "left", 1)
    assert first == second


def test_allocator_reuses_sha256_and_never_reuses_sequence_numbers() -> None:
    allocator = DocumentIdAllocator()
    first = allocator.assign(SHA_A)
    second = allocator.assign(SHA_B)
    assert first == "MHGR_000001"
    assert second == "MHGR_000002"
    assert allocator.assign(SHA_A) == first

    restored = DocumentIdAllocator.from_records(
        [
            {"sha256": SHA_A, "document_id": first},
            {"sha256": SHA_B, "document_id": second},
        ]
    )
    assert restored.assign(SHA_A) == first
    assert restored.assign(SHA_C) == "MHGR_000003"

    gapped = DocumentIdAllocator.from_records(
        [{"sha256": SHA_A, "document_id": "MHGR_000005"}]
    )
    assert gapped.assign(SHA_B) == "MHGR_000006"


def test_allocator_rejects_one_hash_with_two_document_ids() -> None:
    with pytest.raises(ManifestConflictError):
        DocumentIdAllocator.from_records(
            [
                {"sha256": SHA_A, "document_id": "MHGR_000001"},
                {"sha256": SHA_A, "document_id": "MHGR_000002"},
            ]
        )


def test_allocator_rejects_uppercase_and_short_hashes() -> None:
    allocator = DocumentIdAllocator()
    with pytest.raises(ValueError):
        allocator.assign(SHA_A.upper())
    with pytest.raises(ValueError):
        allocator.assign("abc")
