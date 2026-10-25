"""Structured OCR targets, metrics, manifest gates, and synthetic rewards."""

from __future__ import annotations

from pathlib import Path

import pytest

from training.structured_ocr.manifest import assert_training_ready, validate_manifest_row
from training.structured_ocr.metrics import evaluate_structure, looks_column_major_dump
from training.structured_ocr.rewards import score_verifiable_output
from training.structured_ocr.synthetic import column_major_bad_prediction, holiday_table_reference
from training.structured_ocr.targets import EMPTY_PAGE_TARGET, normalize_target, validate_target

ROOT = Path(__file__).resolve().parents[1]


def test_normalize_strips_fences() -> None:
    raw = "```markdown\n# Title\n\n<table></table>\n```"
    out = normalize_target(raw)
    assert out.startswith("# Title")
    assert "```" not in out


def test_valid_html_table_reference() -> None:
    ref = holiday_table_reference(3, "guj")
    report = validate_target(ref, expect_table_rows=3)
    assert report.ok
    assert report.table_row_count == 3


def test_empty_page() -> None:
    assert validate_target("", allow_empty=True).ok
    assert validate_target(EMPTY_PAGE_TARGET, allow_empty=True).ok
    assert not validate_target("").ok


def test_column_major_detection() -> None:
    bad = column_major_bad_prediction(8)
    assert looks_column_major_dump(bad)
    good = holiday_table_reference(3)
    assert not looks_column_major_dump(good)


def test_verifiable_rewards() -> None:
    ref = holiday_table_reference(3)
    good = score_verifiable_output(ref, ref, expect_table_rows=3)
    bad = score_verifiable_output(ref, column_major_bad_prediction(3), expect_table_rows=3)
    assert good.total > 0.95
    assert bad.total < 0.5
    assert bad.no_column_dump == 0.0


def test_manifest_blocks_unverified() -> None:
    row = {
        "sample_id": "x1",
        "document_id": "doc1",
        "page": 1,
        "language": "guj",
        "split": "train",
        "task": "full_page",
        "image": "ocr_service/examples/gujarati_holidays_2015.jpg",
        "text": holiday_table_reference(1),
        "verification_status": "pending",
        "label_source": "tesseract_draft",
    }
    issues = validate_manifest_row(row, repo_root=ROOT)
    assert "label_not_human_verified" in issues


def test_manifest_training_ready_requires_verified() -> None:
    row = {
        "sample_id": "x1",
        "document_id": "doc1",
        "page": 1,
        "language": "guj",
        "split": "train",
        "task": "full_page",
        "image": "ocr_service/examples/gujarati_holidays_2015.jpg",
        "text": holiday_table_reference(1),
        "verification_status": "human_verified",
        "transcription_verified": True,
        "label_source": "human",
    }
    image_path = ROOT / row["image"]
    if not image_path.is_file():
        pytest.skip("holiday fixture image not present")
    assert_training_ready([row], repo_root=ROOT)


def test_structure_metrics_on_bad_dump() -> None:
    ref = holiday_table_reference(3)
    bad = column_major_bad_prediction(3)
    m = evaluate_structure(ref, bad)
    assert m.column_major_dump
    assert not m.table_row_count_match
