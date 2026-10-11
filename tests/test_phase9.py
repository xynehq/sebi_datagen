"""Phase 9 calibration analysis and gold eligibility. No OCR. No gold export."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from indic_ocr_pipeline.dataset.eligibility import (
    assert_no_automatic_gold_from_agreement,
    assert_no_automatic_gold_from_low_risk,
    collect_eligible_verified,
    document_split_readiness,
    is_gold_eligible_status,
    is_gold_ineligible_status,
    proposed_gold_from_verified,
    validate_gold_pool,
    validate_proposed_gold_row,
)
from indic_ocr_pipeline.errors import GoldPolicyError
from indic_ocr_pipeline.review.analysis import (
    FIRST_CALIBRATION_BATCH_ID,
    SECOND_CALIBRATION_BATCH_ID,
    annotate_strict_two_agree_rule,
    build_combined_phase9_report,
    build_phase9_report,
    build_reviewed_rows,
    classify_reviewed_row,
    evaluate_strict_two_agree_filter,
    is_boundary_valid,
    is_candidate_exact,
    shared_error_all3,
    two_agree_investigation,
)
from indic_ocr_pipeline.review.sample import (
    PHASE10_BATCH_ID,
    SECOND_BATCH_ID,
    available_population,
    sample_calibration_batch,
    sample_phase10_multidoc_batch,
    sample_twoagree_calibration_batch,
)
from indic_ocr_pipeline.text_policy import apply_storage_normalization
from indic_ocr_pipeline.utils.hashing import sha256_file


def test_candidate_exact_only_for_human_verified() -> None:
    assert is_candidate_exact("human_verified") is True
    assert is_candidate_exact("human_corrected") is False
    assert is_candidate_exact("rejected") is False
    assert is_candidate_exact("bad_boundary") is False


def test_boundary_invalid_separation() -> None:
    assert is_boundary_valid("human_verified") is True
    assert is_boundary_valid("human_corrected") is True
    assert is_boundary_valid("rejected") is True
    assert is_boundary_valid("bad_boundary") is False
    row = classify_reviewed_row(
        {
            "crop_id": "MHGR_000001_P0001_L0001_C_full_01",
            "document_id": "MHGR_000001",
            "page_id": "MHGR_000001_P0001",
            "parent_line_id": "MHGR_000001_P0001_L0001",
            "verification_status": "rejected",
            "review_decision": "bad_boundary",
            "final_text": None,
            "original_candidate_texts": {
                "indicocr": "अ",
                "chitrapathak": "अ",
                "tesseract": "अ",
            },
            "metadata": {
                "decision_label": "bad_boundary",
                "agreement_category": "ALL_3_AGREE",
                "review_priority": "P0",
            },
        }
    )
    assert row["boundary_invalid"] is True
    assert row["ocr_correctness_unknown"] is True
    assert row["candidate_exact"] is False


def test_agreement_category_calibration_separates_outcomes() -> None:
    rows = [
        _verified("human_verified", "TWO_AGREE_ONE_DIFFERS", "अ"),
        _verified("human_corrected", "TWO_AGREE_ONE_DIFFERS", "आ"),
        _verified("bad_boundary", "ALL_3_DIFFER", None),
        _verified("human_verified", "ALL_3_AGREE", "इ"),
    ]
    report = build_phase9_report(rows)
    two = report["by_agreement_category"]["TWO_AGREE_ONE_DIFFERS"]
    assert two["reviewed"] == 2
    assert two["human_verified"] == 1
    assert two["human_corrected"] == 1
    assert two["candidate_exact_rate_among_boundary_valid"] == 50.0


def test_three_way_agreement_shared_error_analysis() -> None:
    rows = [
        _verified("human_verified", "ALL_3_AGREE", "संख्या"),
        _verified(
            "human_corrected",
            "ALL_3_AGREE",
            "संख्या",
            tags=["shared_ocr_error", "missing_matra"],
            candidates=("सख्या", "सख्या", "सख्या"),
        ),
        _verified("rejected", "ALL_3_AGREE", None, tags=["non_text_fragment"]),
    ]
    shared = shared_error_all3(rows)
    assert shared["n_reviewed"] == 3
    assert shared["agreement_but_human_corrected"] == 1
    assert shared["shared_error_rate_corrected"] == pytest.approx(33.33)


def test_two_agree_analysis_and_strict_rule() -> None:
    good = _verified(
        "human_verified",
        "TWO_AGREE_ONE_DIFFERS",
        "अ",
        candidates=("अ", "अ", "आ"),
        flags=[],
    )
    bad_geom = _verified(
        "bad_boundary",
        "TWO_AGREE_ONE_DIFFERS",
        None,
        candidates=("अ", "अ", "आ"),
        flags=["large_length_disagreement"],
    )
    rows = [good, bad_geom]
    for row in rows:
        row["passes_strict_two_agree_rule"] = annotate_strict_two_agree_rule(row)
    report = two_agree_investigation(rows)
    assert report["n_reviewed"] == 2
    assert report["boundary_failures"] == 1
    assert report["strict_rule"]["n_matching_reviewed"] == 1
    assert report["strict_rule"]["auto_approval_enabled"] is False


def test_no_automatic_gold_from_agreement() -> None:
    with pytest.raises(GoldPolicyError):
        assert_no_automatic_gold_from_agreement()


def test_no_automatic_gold_from_low_risk() -> None:
    with pytest.raises(GoldPolicyError):
        assert_no_automatic_gold_from_low_risk()


def test_gold_eligibility_human_statuses() -> None:
    assert is_gold_eligible_status("human_verified")
    assert is_gold_eligible_status("human_corrected")


def test_gold_ineligibility_non_human_statuses() -> None:
    for status in (
        "candidate",
        "auto_approved_candidate",
        "pending",
        "rejected",
        "bad_boundary",
    ):
        assert is_gold_ineligible_status(status)


def test_gold_validation_rejects_missing_gold_text(tmp_path: Path) -> None:
    row = _proposed_gold(tmp_path, gold_text="")
    errors = validate_proposed_gold_row(row, repo_root=tmp_path, check_image_bytes=False)
    assert "missing gold_text" in errors


def test_gold_validation_rejects_non_nfc_text(tmp_path: Path) -> None:
    # Force a non-storage form with surrounding whitespace.
    row = _proposed_gold(tmp_path, gold_text="  अ  ")
    errors = validate_proposed_gold_row(row, repo_root=tmp_path, check_image_bytes=False)
    assert any("NFC" in err or "storage" in err for err in errors)


def test_gold_validation_rejects_missing_crop(tmp_path: Path) -> None:
    row = _proposed_gold(tmp_path, gold_text="अ", create_image=False)
    errors = validate_proposed_gold_row(
        row,
        repo_root=tmp_path,
        page={"page_number": 1},
        document={"source_path": "x.pdf", "sha256": "a" * 64},
        check_image_bytes=True,
    )
    assert "missing crop image" in errors


def test_gold_validation_detects_crop_sha_mismatch(tmp_path: Path) -> None:
    row = _proposed_gold(tmp_path, gold_text="अ", create_image=True)
    row["source_sha256"] = "0" * 64
    errors = validate_proposed_gold_row(
        row,
        repo_root=tmp_path,
        page={"page_number": 1},
        document={"source_path": "x.pdf", "sha256": "a" * 64},
        check_image_bytes=True,
    )
    assert "crop SHA-256 mismatch" in errors


def test_provenance_chain_validation(tmp_path: Path) -> None:
    row = _proposed_gold(tmp_path, gold_text="अ", create_image=True)
    errors = validate_proposed_gold_row(
        row,
        repo_root=tmp_path,
        page=None,
        document=None,
        check_image_bytes=False,
    )
    assert "missing page" in errors
    assert "missing source document record" in errors


def test_duplicate_sample_detection(tmp_path: Path) -> None:
    row = _proposed_gold(tmp_path, gold_text="अ", create_image=False)
    result = validate_gold_pool([row, dict(row)], check_image_bytes=False)
    assert result["duplicate_sample_ids"]


def test_conflicting_gold_label_detection(tmp_path: Path) -> None:
    left = _proposed_gold(tmp_path, gold_text="अ", create_image=False)
    right = dict(left)
    right["sample_id"] = left["sample_id"].replace("L0001", "L0002")
    right["crop_id"] = left["crop_id"].replace("L0001", "L0002")
    right["gold_text"] = "आ"
    result = validate_gold_pool([left, right], check_image_bytes=False)
    assert result["conflicting_gold_labels"]


def test_document_level_split_validation() -> None:
    rows = [
        {
            "document_id": "MHGR_000001",
            "page_id": "MHGR_000001_P0001",
            "parent_line_id": "MHGR_000001_P0001_L0001",
            "crop_id": "MHGR_000001_P0001_L0001_C_full_01",
        },
        {
            "document_id": "MHGR_000002",
            "page_id": "MHGR_000002_P0001",
            "parent_line_id": "MHGR_000002_P0001_L0001",
            "crop_id": "MHGR_000002_P0001_L0001_C_full_01",
        },
    ]
    readiness = document_split_readiness(rows)
    assert readiness["ready"] is True
    assert readiness["random_line_split_allowed"] is False
    assert readiness["documents"]["MHGR_000001"] == 1


def test_second_calibration_sampler_is_deterministic() -> None:
    reviews = _review_pool()
    exclude = {"MHGR_000001_P0001_L0001_C_full_01"}
    left = sample_twoagree_calibration_batch(
        reviews,
        exclude_crop_ids=exclude,
        seed=3407,
        batch_id=SECOND_BATCH_ID,
        boundary_quota=5,
        start_index=601,
    )
    right = sample_twoagree_calibration_batch(
        reviews,
        exclude_crop_ids=exclude,
        seed=3407,
        batch_id=SECOND_BATCH_ID,
        boundary_quota=5,
        start_index=601,
    )
    assert [item["crop_id"] for item in left] == [item["crop_id"] for item in right]
    assert left[0]["sample_index"] == 601


def test_second_batch_does_not_overlap_first_600() -> None:
    reviews = _review_pool()
    first = sample_calibration_batch(
        reviews,
        batch_size=6,
        tier_quota={"P0": None, "P1": 2, "P2": 2, "P3": 2},
        seed=3407,
        batch_id="calibration_3407_600",
    )
    exclude = {item["crop_id"] for item in first}
    second = sample_twoagree_calibration_batch(
        reviews,
        exclude_crop_ids=exclude,
        seed=3407,
        boundary_quota=10,
        start_index=7,
    )
    assert {item["crop_id"] for item in second}.isdisjoint(exclude)


def test_analysis_does_not_mutate_source_manifests(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    payload = {"crop_id": "MHGR_000001_P0001_L0001_C_full_01", "x": 1}
    path = manifests / "crops.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = [
        _verified("human_verified", "TWO_AGREE_ONE_DIFFERS", "अ"),
    ]
    build_phase9_report(rows)
    after = hashlib.sha256(path.read_bytes()).hexdigest()
    assert before == after


def test_phase9_report_never_enables_auto_approval_or_gold() -> None:
    report = build_phase9_report(
        [_verified("human_verified", "ALL_3_AGREE", "अ")]
    )
    assert report["auto_approval_enabled"] is False
    assert report["gold_export"] is False
    assert report["ocr_invoked"] is False
    assert report["network_used"] is False
    assert report["models_loaded"] is False
    assert report["unsafe_policy_checks"]["all_3_agree_auto_approve"]["enabled"] is False
    assert report["unsafe_policy_checks"]["low_risk_p3_auto_approve"]["enabled"] is False


def test_collect_eligible_verified_excludes_bad_boundary() -> None:
    rows = [
        {
            "verification_status": "human_verified",
            "final_text": "अ",
            "metadata": {"decision_label": "human_verified"},
        },
        {
            "verification_status": "rejected",
            "final_text": None,
            "metadata": {"decision_label": "bad_boundary"},
        },
        {
            "verification_status": "rejected",
            "final_text": None,
            "metadata": {"decision_label": "rejected"},
        },
    ]
    eligible = collect_eligible_verified(rows)
    assert len(eligible) == 1


def test_proposed_gold_from_verified_requires_human_status() -> None:
    with pytest.raises(GoldPolicyError):
        proposed_gold_from_verified(
            {
                "crop_id": "MHGR_000001_P0001_L0001_C_full_01",
                "document_id": "MHGR_000001",
                "page_id": "MHGR_000001_P0001",
                "parent_line_id": "MHGR_000001_P0001_L0001",
                "verification_status": "rejected",
                "final_text": None,
                "metadata": {"decision_label": "rejected"},
            },
            crop={
                "image_path": "x.png",
                "region_id": "MHGR_000001_P0001_R0001",
                "crop_type": "full",
                "bbox": [0, 0, 10, 10],
                "sha256": "a" * 64,
            },
            page={"page_number": 1},
        )


def test_build_reviewed_rows_joins_batch(tmp_path: Path) -> None:
    batch = [
        {
            "batch_id": "calibration_3407_600",
            "crop_id": "MHGR_000001_P0001_L0001_C_full_01",
        }
    ]
    verified = [
        {
            "crop_id": "MHGR_000001_P0001_L0001_C_full_01",
            "document_id": "MHGR_000001",
            "page_id": "MHGR_000001_P0001",
            "parent_line_id": "MHGR_000001_P0001_L0001",
            "verification_status": "human_verified",
            "review_decision": "accept_candidate",
            "final_text": "अ",
            "original_candidate_texts": {
                "indicocr": "अ",
                "chitrapathak": "अ",
                "tesseract": "अ",
            },
            "metadata": {
                "decision_label": "human_verified",
                "agreement_category": "ALL_3_AGREE",
                "review_priority": "P3",
            },
        }
    ]
    reviews = [
        {
            "crop_id": "MHGR_000001_P0001_L0001_C_full_01",
            "diagnostics": {
                "agreement_category": "ALL_3_AGREE",
                "review_priority": "P3",
                "flags": [],
                "feature_risks": {"crop_geometry": 0.1},
                "tesseract_confidence": 0.95,
                "region_id": "MHGR_000001_P0001_R0001",
            },
            "risk_score": 0.2,
        }
    ]
    rows = build_reviewed_rows(
        batch_items=batch,
        verified_rows=verified,
        review_rows=reviews,
    )
    assert len(rows) == 1
    assert rows[0]["candidate_exact"] is True


def test_available_population_excludes_completed() -> None:
    reviews = _review_pool()
    population = available_population(
        reviews, exclude_crop_ids={"MHGR_000001_P0001_L0001_C_full_01"}
    )
    assert population["excluded"] == 1
    assert population["remaining_total"] == len(reviews) - 1


def test_storage_form_helper_used_for_gold() -> None:
    assert apply_storage_normalization("  अ  ") == "अ"


def test_second_batch_is_read_independently() -> None:
    first = [_verified("human_verified", "TWO_AGREE_ONE_DIFFERS", "अ")]
    second = [
        _verified(
            "human_corrected",
            "TWO_AGREE_ONE_DIFFERS",
            "संख्या",
            candidates=("संख्या", "सख्या", "सख्या"),
            tags=["shared_ocr_error"],
        )
    ]
    second[0]["document_id"] = "MHGR_000003"
    report = build_combined_phase9_report(first_rows=first, second_rows=second)
    assert report["first_batch"]["batch_id"] == FIRST_CALIBRATION_BATCH_ID
    assert report["second_batch"]["batch_id"] == SECOND_CALIBRATION_BATCH_ID
    assert report["first_batch"]["reviewed_total"] == 1
    assert report["second_batch"]["reviewed_total"] == 1
    assert report["combined"]["reviewed_total"] == 2


def test_combined_analysis_preserves_per_batch_statistics() -> None:
    first = [_verified("bad_boundary", "ALL_3_DIFFER", None)]
    second = [_verified("human_verified", "ALL_3_AGREE", "अ")]
    report = build_combined_phase9_report(first_rows=first, second_rows=second)
    assert report["first_batch"]["overall"]["bad_boundary"] == 1
    assert report["second_batch"]["overall"]["human_verified"] == 1
    assert report["combined"]["overall"]["reviewed"] == 2


def test_batch_overlap_is_detected() -> None:
    row = _verified("human_verified", "TWO_AGREE_ONE_DIFFERS", "अ")
    report = build_combined_phase9_report(first_rows=[row], second_rows=[row])
    assert report["batch_overlap_count"] == 1


def test_two_agree_filter_definition_is_deterministic() -> None:
    row = _verified(
        "human_verified",
        "TWO_AGREE_ONE_DIFFERS",
        "अ",
        candidates=("अ", "अ", "आ"),
        flags=[],
    )
    left = annotate_strict_two_agree_rule(row)
    right = annotate_strict_two_agree_rule(row)
    assert left is True and right is True


def test_filter_does_not_depend_on_final_human_outcome() -> None:
    base = _verified(
        "human_verified",
        "TWO_AGREE_ONE_DIFFERS",
        "अ",
        candidates=("अ", "अ", "आ"),
        flags=[],
    )
    corrected = _verified(
        "human_corrected",
        "TWO_AGREE_ONE_DIFFERS",
        "आ",
        candidates=("अ", "अ", "आ"),
        flags=[],
    )
    assert annotate_strict_two_agree_rule(base) == annotate_strict_two_agree_rule(
        corrected
    )


def test_strict_filter_false_accept_uses_human_gold() -> None:
    rows = [
        _verified(
            "human_corrected",
            "TWO_AGREE_ONE_DIFFERS",
            "संख्या",
            candidates=("संख्या", "सख्या", "सख्या"),
            flags=[],
        )
    ]
    rows[0]["proposed_consensus"] = "सख्या"
    result = evaluate_strict_two_agree_filter(rows)
    assert result["filter_depends_on_human_outcome"] is False
    assert result["false_accept_count"] == 1
    assert result["auto_approval_enabled"] is False


def test_source_distribution_limitation_represented() -> None:
    rows = [
        _verified("human_verified", "TWO_AGREE_ONE_DIFFERS", "अ"),
    ]
    rows[0]["document_id"] = "MHGR_000003"
    report = build_phase9_report(rows, batch_id=SECOND_CALIBRATION_BATCH_ID)
    assert report["source_distribution_limitation"] == "single_document_batch"
    combined = build_combined_phase9_report(first_rows=rows, second_rows=rows)
    assert "MHGR_000003" in combined["source_distribution_caveat"]
    assert combined["policy_classification"]["auto_approval_enabled"] is False
    assert combined["gold_export"] is False


def test_all3_control_analysis_works() -> None:
    rows = [
        _verified("human_verified", "ALL_3_AGREE", "अ"),
        _verified(
            "human_corrected",
            "ALL_3_AGREE",
            "संख्या",
            tags=["shared_ocr_error"],
            candidates=("सख्या", "सख्या", "सख्या"),
        ),
    ]
    shared = shared_error_all3(rows)
    assert shared["n_reviewed"] == 2
    assert shared["agreement_but_human_corrected"] == 1


def test_phase10_sampler_is_deterministic_and_document_scoped() -> None:
    reviews = _review_pool()
    # Expand pool with synthetic new-document rows.
    extra = []
    index = 100
    for document in (4, 5, 6):
        for agreement, tier in (
            ("TWO_AGREE_ONE_DIFFERS", "P2"),
            ("ALL_3_AGREE", "P3"),
            ("ALL_3_DIFFER", "P1"),
        ):
            for _ in range(4):
                crop_id = f"MHGR_{document:06d}_P0001_L{index:04d}_C_full_01"
                extra.append(
                    {
                        "crop_id": crop_id,
                        "document_id": f"MHGR_{document:06d}",
                        "page_id": f"MHGR_{document:06d}_P0001",
                        "parent_line_id": f"MHGR_{document:06d}_P0001_L{index:04d}",
                        "priority_rank": index,
                        "risk_score": 0.4,
                        "diagnostics": {
                            "agreement_category": agreement,
                            "review_priority": tier,
                            "region_id": f"MHGR_{document:06d}_P0001_R0001",
                            "flags": [],
                            "feature_risks": {"crop_geometry": 0.2},
                            "proposed_consensus": "अ" * ((index % 5) + 1),
                        },
                    }
                )
                index += 1
    left = sample_phase10_multidoc_batch(
        reviews + extra,
        document_ids={"MHGR_000004", "MHGR_000005", "MHGR_000006"},
        exclude_crop_ids=set(),
        seed=3407,
        batch_size=20,
        batch_id=PHASE10_BATCH_ID,
        start_index=889,
    )
    right = sample_phase10_multidoc_batch(
        reviews + extra,
        document_ids={"MHGR_000004", "MHGR_000005", "MHGR_000006"},
        exclude_crop_ids=set(),
        seed=3407,
        batch_size=20,
        batch_id=PHASE10_BATCH_ID,
        start_index=889,
    )
    assert [item["crop_id"] for item in left] == [item["crop_id"] for item in right]
    assert all(item["document_id"].startswith("MHGR_00000") for item in left)
    assert {item["document_id"] for item in left} <= {
        "MHGR_000004",
        "MHGR_000005",
        "MHGR_000006",
    }
    assert left[0]["sample_index"] == 889
    assert left[0]["batch_id"] == PHASE10_BATCH_ID


def _verified(
    label: str,
    agreement: str,
    final_text: str | None,
    *,
    tags: list[str] | None = None,
    candidates: tuple[str, str, str] | None = None,
    flags: list[str] | None = None,
) -> dict:
    status = {
        "human_verified": "human_verified",
        "human_corrected": "human_corrected",
        "rejected": "rejected",
        "bad_boundary": "rejected",
    }[label]
    decision = {
        "human_verified": "accept_candidate",
        "human_corrected": "correct_text",
        "rejected": "reject_crop",
        "bad_boundary": "bad_boundary",
    }[label]
    texts = candidates or ("अ", "अ", "अ")
    row = {
        "crop_id": "MHGR_000001_P0001_L0001_C_full_01",
        "document_id": "MHGR_000001",
        "page_id": "MHGR_000001_P0001",
        "parent_line_id": "MHGR_000001_P0001_L0001",
        "verification_status": status,
        "review_decision": decision,
        "final_text": final_text,
        "original_candidate_texts": {
            "indicocr": texts[0],
            "chitrapathak": texts[1],
            "tesseract": texts[2],
        },
        "metadata": {
            "decision_label": label,
            "agreement_category": agreement,
            "review_priority": "P2",
            "error_tags": tags or [],
        },
        "decision_label": label,
        "agreement_category": agreement,
        "review_priority": "P2",
        "flags": flags if flags is not None else [],
        "empty_present": "empty_prediction" in (flags or []),
        "large_length_disagreement": "large_length_disagreement" in (flags or []),
        "unicode_anomaly": "unicode_anomaly" in (flags or []),
        "boundary_valid": label != "bad_boundary",
        "boundary_invalid": label == "bad_boundary",
        "table_hint": "non_table",
        "line_length_bucket": "short_le_8",
        "error_tags": tags or [],
        "proposed_consensus": texts[0],
    }
    return row


def _proposed_gold(
    tmp_path: Path,
    *,
    gold_text: str,
    create_image: bool = True,
) -> dict:
    rel = "crop.png"
    path = tmp_path / rel
    digest = "b" * 64
    if create_image:
        Image.new("L", (16, 8), color=255).save(path)
        digest = sha256_file(path)
    return {
        "sample_id": "SAMPLE_MHGR_000001_P0001_L0001_C_full_01",
        "image_path": rel,
        "gold_text": gold_text,
        "document_id": "MHGR_000001",
        "page_id": "MHGR_000001_P0001",
        "page_number": 1,
        "region_id": "MHGR_000001_P0001_R0001",
        "parent_line_id": "MHGR_000001_P0001_L0001",
        "crop_id": "MHGR_000001_P0001_L0001_C_full_01",
        "crop_type": "full",
        "bbox": [0, 0, 16, 8],
        "verification_status": "human_verified",
        "source_sha256": digest,
        "original_candidate_texts": {"indicocr": "अ", "chitrapathak": "अ", "tesseract": "अ"},
        "verification_timestamp": "2026-10-07T00:00:00+00:00",
        "reviewer_id": "local_reviewer",
    }


def _review_pool() -> list[dict]:
    rows = []
    index = 1
    for document in (1, 2, 3):
        for agreement, tier in (
            ("TWO_AGREE_ONE_DIFFERS", "P2"),
            ("ALL_3_AGREE", "P3"),
            ("ALL_3_DIFFER", "P1"),
            ("ONE_OR_MORE_EMPTY", "P0"),
        ):
            for _ in range(3):
                crop_id = f"MHGR_{document:06d}_P0001_L{index:04d}_C_full_01"
                rows.append(
                    {
                        "crop_id": crop_id,
                        "document_id": f"MHGR_{document:06d}",
                        "page_id": f"MHGR_{document:06d}_P0001",
                        "parent_line_id": f"MHGR_{document:06d}_P0001_L{index:04d}",
                        "priority_rank": index,
                        "risk_score": 0.5,
                        "diagnostics": {
                            "agreement_category": agreement,
                            "review_priority": tier,
                            "region_id": f"MHGR_{document:06d}_P0001_R0001",
                            "flags": ["large_length_disagreement"]
                            if tier == "P1"
                            else [],
                            "feature_risks": {
                                "crop_geometry": 0.7 if tier in {"P0", "P1"} else 0.1
                            },
                            "proposed_consensus": "अ" * (index % 5 + 1),
                        },
                    }
                )
                index += 1
    return rows
