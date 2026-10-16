"""Schema rules that keep candidates out of gold."""

from __future__ import annotations

import pytest

from indic_ocr_pipeline.errors import SchemaError
from indic_ocr_pipeline.schemas import (
    BBox,
    CandidateRecord,
    CropRecord,
    CropType,
    DocumentRecord,
    GoldRecord,
    PageRecord,
    QualityTier,
    VerificationStatus,
    admits_to_gold,
    assert_bbox_within_page,
    assert_crop_within_parent,
    assert_partial_label_matches_crop,
    decision_creates_gold,
)

SHA = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
LINE = "महाराष्ट्र शासनाने नवीन आदेश जारी केला आहे."
PARTIAL = "नवीन आदेश जारी"


def test_gold_admission_is_limited_to_human_decisions() -> None:
    assert admits_to_gold(VerificationStatus.HUMAN_VERIFIED)
    assert admits_to_gold("human_corrected")
    assert not admits_to_gold(VerificationStatus.CANDIDATE)
    assert not admits_to_gold(VerificationStatus.AUTO_APPROVED_CANDIDATE)
    assert not admits_to_gold(VerificationStatus.REJECTED)
    assert decision_creates_gold("accept_candidate")
    assert decision_creates_gold("correct_text")
    assert not decision_creates_gold("reject_crop")
    assert not decision_creates_gold("uncertain")


def test_bbox_rejects_bools_and_empty_area() -> None:
    with pytest.raises(SchemaError):
        BBox(True, 0, 2, 2)  # type: ignore[arg-type]
    with pytest.raises(SchemaError):
        BBox(0, 0, 0, 10)
    box = BBox.from_list([10, 20, 110, 60])
    assert box.width == 100
    assert box.height == 40
    assert_bbox_within_page(box, 200, 200)
    with pytest.raises(SchemaError):
        assert_bbox_within_page(box, 100, 200)


def test_crop_must_stay_inside_parent_and_has_no_label_field() -> None:
    parent = BBox(10, 20, 200, 60)
    inside = BBox(80, 20, 150, 60)
    assert_crop_within_parent(inside, parent)
    with pytest.raises(SchemaError):
        assert_crop_within_parent(BBox(10, 20, 220, 60), parent)
    crop = CropRecord(
        crop_id="MHGR_000001_P0003_L0001_C_left_01",
        document_id="MHGR_000001",
        page_id="MHGR_000001_P0003",
        page_number=3,
        region_id="MHGR_000001_P0003_R0001",
        parent_line_id="MHGR_000001_P0003_L0001",
        crop_type=CropType.LEFT,
        crop_index=1,
        image_path="data/crops/partial/left.png",
        bbox=inside,
        parent_bbox=parent,
        random_seed=3407,
    )
    payload = crop.to_dict()
    assert "gold_text" not in payload
    assert payload["x_start"] == 80
    assert payload["x_end"] == 150
    restored = CropRecord.from_dict(payload)
    assert restored.crop_id == crop.crop_id


def test_partial_label_cannot_copy_the_full_line() -> None:
    with pytest.raises(SchemaError):
        assert_partial_label_matches_crop(CropType.LEFT, LINE, LINE)
    assert_partial_label_matches_crop(CropType.MIDDLE, PARTIAL, LINE)
    assert_partial_label_matches_crop(CropType.FULL, LINE, LINE)
    assert_partial_label_matches_crop(CropType.RANDOM, PARTIAL, None)


def test_page_text_layer_cannot_be_relabeled_as_truth() -> None:
    page = PageRecord(
        document_id="MHGR_000001",
        page_id="MHGR_000001_P0001",
        page_number=1,
        image_path="data/pages/MHGR_000001/page_0001.png",
        width=1000,
        height=1400,
        dpi=200,
        sha256=SHA,
        pdf_text_layer="broken layer",
    )
    assert page.pdf_text_layer_role == "weak_diagnostic_not_ground_truth"
    page.pdf_text_layer_role = "ground_truth"
    with pytest.raises(SchemaError):
        page.validate()


def test_document_record_round_trip() -> None:
    record = DocumentRecord(
        document_id="MHGR_000001",
        original_filename="order.pdf",
        source_path="data/documents/order.pdf",
        sha256=SHA,
        file_size=32,
        department="revenue",
        document_type="government_resolution",
        document_date="2024-01-15",
    )
    assert DocumentRecord.from_dict(record.to_dict()).sha256 == SHA


def test_candidate_cannot_be_marked_human_verified() -> None:
    with pytest.raises(SchemaError):
        CandidateRecord(
            candidate_id="MHGR_000001_P0003_L0001_C_full_01_CAND",
            crop_id="MHGR_000001_P0003_L0001_C_full_01",
            document_id="MHGR_000001",
            page_id="MHGR_000001_P0003",
            parent_line_id="MHGR_000001_P0003_L0001",
            texts_by_backend={"tesseract": LINE, "indicocr": LINE, "chitrapathak": None},
            exact_agreement_count=2,
            backends_present=3,
            pairwise_agreement={"tesseract|indicocr": True},
            normalized_agreement={"tesseract|indicocr": True},
            consensus_text=LINE,
            disagreement_type=None,
            confidence_score=0.96,
            risk_score=0.04,
            reasons=["all_backends_agree"],
            verification_status=VerificationStatus.HUMAN_VERIFIED,
            tier=QualityTier.TIER_5_GOLD,
        )


def test_gold_record_requires_human_status_and_storage_form() -> None:
    base = dict(
        sample_id="SAMPLE_MHGR_000001_P0003_L0001_C_middle_01",
        image_path="data/gold/middle.png",
        gold_text=PARTIAL,
        document_id="MHGR_000001",
        page_id="MHGR_000001_P0003",
        page_number=3,
        region_id="MHGR_000001_P0003_R0001",
        parent_line_id="MHGR_000001_P0003_L0001",
        crop_id="MHGR_000001_P0003_L0001_C_middle_01",
        crop_type=CropType.MIDDLE,
        bbox=BBox(40, 20, 120, 60),
        verification_status=VerificationStatus.HUMAN_CORRECTED,
        source_sha256=SHA,
        original_candidate_texts={"tesseract": "नवीन आदेश", "indicocr": PARTIAL},
        verification_timestamp="2026-10-07T12:00:00+05:30",
        reviewer_id="reviewer-1",
        consensus_text=LINE,
    )
    gold = GoldRecord(**base)
    assert gold.tier is QualityTier.TIER_5_GOLD
    assert gold.consensus_text != gold.gold_text
    with pytest.raises(SchemaError):
        GoldRecord(**{**base, "verification_status": VerificationStatus.AUTO_APPROVED_CANDIDATE})
    with pytest.raises(SchemaError):
        GoldRecord(**{**base, "gold_text": f" {PARTIAL} "})
    with pytest.raises(SchemaError):
        GoldRecord(**{**base, "gold_text": "\ufffd"})
    with pytest.raises(SchemaError):
        GoldRecord(**{**base, "gold_text": "e\u0301"})
