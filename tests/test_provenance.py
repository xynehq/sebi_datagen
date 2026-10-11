"""Leakage and duplicate checks over manifest rows."""

from __future__ import annotations

import pytest

from indic_ocr_pipeline.errors import GoldPolicyError
from indic_ocr_pipeline.provenance import (
    assert_no_split_leakage,
    find_conflicting_gold_labels,
    find_document_leakage,
    find_duplicate_values,
    find_parent_line_leakage,
    find_sha256_collisions,
    missing_provenance_fields,
)

LINE = "MHGR_000001_P0001_L0001"


def test_same_document_and_line_cannot_cross_splits() -> None:
    leaked = [
        {"document_id": "MHGR_000001", "parent_line_id": LINE, "split": "train"},
        {
            "document_id": "MHGR_000001",
            "parent_line_id": LINE,
            "split": "val",
            "crop_type": "left",
        },
    ]
    assert find_document_leakage(leaked)["MHGR_000001"] == {"train", "val"}
    assert find_parent_line_leakage(leaked)[LINE] == {"train", "val"}
    with pytest.raises(GoldPolicyError):
        assert_no_split_leakage(leaked)


def test_document_level_split_is_accepted() -> None:
    rows = [
        {"document_id": "MHGR_000001", "parent_line_id": LINE, "split": "train"},
        {
            "document_id": "MHGR_000001",
            "parent_line_id": "MHGR_000001_P0001_L0002",
            "split": "train",
        },
        {
            "document_id": "MHGR_000002",
            "parent_line_id": "MHGR_000002_P0001_L0001",
            "split": "val",
        },
    ]
    assert find_document_leakage(rows) == {}
    assert_no_split_leakage(rows)


def test_conflicting_labels_and_duplicate_ids_are_reported() -> None:
    rows = [
        {"sample_id": "SAMPLE_1", "image_path": "a.png", "gold_text": "एक"},
        {"sample_id": "SAMPLE_1", "image_path": "a.png", "gold_text": "दोन"},
    ]
    assert find_duplicate_values(rows, "sample_id") == {"SAMPLE_1": 2}
    assert find_conflicting_gold_labels(rows)["a.png"] == {"एक", "दोन"}


def test_sha256_collision_and_missing_provenance() -> None:
    sha = "a" * 64
    collisions = find_sha256_collisions(
        [
            {"sha256": sha, "document_id": "MHGR_000001"},
            {"sha256": sha, "document_id": "MHGR_000002"},
        ]
    )
    assert collisions[sha] == ["MHGR_000001", "MHGR_000002"]
    missing = missing_provenance_fields({"sample_id": "SAMPLE_1", "gold_text": ""})
    assert "image_path" in missing
    assert "gold_text" in missing
    assert "document_id" in missing
