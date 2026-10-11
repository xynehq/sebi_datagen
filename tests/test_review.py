"""Phase 8 human review and calibration. No OCR. No gold export."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest
from PIL import Image

from indic_ocr_pipeline.errors import ManifestConflictError
from indic_ocr_pipeline.ids import make_review_id, make_verified_id
from indic_ocr_pipeline.manifests import open_manifest
from indic_ocr_pipeline.review.calibrate import build_calibration_report
from indic_ocr_pipeline.review.decisions import (
    ReviewDecisionError,
    build_verified_record,
    commit_decision,
    map_ui_decision,
    prepare_gold_text,
)
from indic_ocr_pipeline.review.sample import (
    enqueue_pending_reviews,
    sample_calibration_batch,
    summarize_batch,
)
from indic_ocr_pipeline.schemas import (
    ReviewDecision,
    ReviewRecord,
    StageStatus,
    VerificationStatus,
    VerifiedRecord,
)
from indic_ocr_pipeline.text_policy import apply_storage_normalization
from indic_ocr_pipeline.utils.hashing import sha256_file


def test_review_decision_schema_mapping() -> None:
    assert map_ui_decision("human_verified") == (
        VerificationStatus.HUMAN_VERIFIED,
        ReviewDecision.ACCEPT_CANDIDATE,
    )
    assert map_ui_decision("human_corrected") == (
        VerificationStatus.HUMAN_CORRECTED,
        ReviewDecision.CORRECT_TEXT,
    )
    assert map_ui_decision("rejected")[0] is VerificationStatus.REJECTED
    assert map_ui_decision("bad_boundary") == (
        VerificationStatus.REJECTED,
        ReviewDecision.BAD_BOUNDARY,
    )


def test_human_verified_requires_gold_text() -> None:
    with pytest.raises(ReviewDecisionError):
        prepare_gold_text(None, decision_label="human_verified")


def test_human_corrected_requires_gold_text() -> None:
    with pytest.raises(ReviewDecisionError):
        prepare_gold_text("   ", decision_label="human_corrected")


def test_rejected_cannot_have_gold_text() -> None:
    assert prepare_gold_text(None, decision_label="rejected") is None
    with pytest.raises(ReviewDecisionError):
        prepare_gold_text("अ", decision_label="rejected")


def test_bad_boundary_cannot_have_gold_text() -> None:
    with pytest.raises(ReviewDecisionError):
        prepare_gold_text("अ", decision_label="bad_boundary")


def test_candidate_prediction_cannot_directly_become_gold(tmp_path: Path) -> None:
    _seed(tmp_path)
    commit_decision(
        tmp_path,
        crop_id="MHGR_000001_P0001_L0001_C_full_01",
        decision_label="human_verified",
        gold_text="अ",
        selected_candidate_text="अ",
        reviewer_id="local_reviewer",
        repo_root=tmp_path,
    )
    verified = open_manifest(tmp_path, "verified").load()[0]
    assert verified["tier"] == "tier_4_human_reviewed"
    assert not (tmp_path / "gold.jsonl").exists()


def test_stored_gold_text_uses_label_normalization() -> None:
    assert prepare_gold_text("  अ  ब  ", decision_label="human_corrected") == "अ  ब"


def test_punctuation_is_preserved() -> None:
    assert prepare_gold_text("मुंबई-", decision_label="human_corrected") == "मुंबई-"


def test_marathi_digits_are_preserved() -> None:
    assert prepare_gold_text("४०० ०३२", decision_label="human_corrected") == "४०० ०३२"


def test_human_decision_is_immutable(tmp_path: Path) -> None:
    _seed(tmp_path)
    crop_id = "MHGR_000001_P0001_L0001_C_full_01"
    commit_decision(
        tmp_path,
        crop_id=crop_id,
        decision_label="rejected",
        gold_text=None,
        selected_candidate_text=None,
        reviewer_id="local_reviewer",
        repo_root=tmp_path,
    )
    with pytest.raises(ManifestConflictError):
        commit_decision(
            tmp_path,
            crop_id=crop_id,
            decision_label="bad_boundary",
            gold_text=None,
            selected_candidate_text=None,
            reviewer_id="local_reviewer",
            repo_root=tmp_path,
        )


def test_same_decision_payload_is_idempotent(tmp_path: Path) -> None:
    _seed(tmp_path)
    crop_id = "MHGR_000001_P0001_L0001_C_full_01"
    first = commit_decision(
        tmp_path,
        crop_id=crop_id,
        decision_label="human_corrected",
        gold_text="अ",
        selected_candidate_text=None,
        reviewer_id="local_reviewer",
        repo_root=tmp_path,
    )
    second = commit_decision(
        tmp_path,
        crop_id=crop_id,
        decision_label="human_corrected",
        gold_text="अ",
        selected_candidate_text=None,
        reviewer_id="local_reviewer",
        notes=None,
        repo_root=tmp_path,
    )
    assert first["outcome"] == "created"
    assert second["outcome"] == "skipped"
    assert len(open_manifest(tmp_path, "verified").load()) == 1


def test_conflicting_decision_raises(tmp_path: Path) -> None:
    _seed(tmp_path)
    crop_id = "MHGR_000001_P0001_L0001_C_full_01"
    commit_decision(
        tmp_path,
        crop_id=crop_id,
        decision_label="human_verified",
        gold_text="अ",
        selected_candidate_text="अ",
        reviewer_id="local_reviewer",
        repo_root=tmp_path,
    )
    with pytest.raises(ManifestConflictError):
        commit_decision(
            tmp_path,
            crop_id=crop_id,
            decision_label="human_verified",
            gold_text="ब",
            selected_candidate_text="ब",
            reviewer_id="local_reviewer",
            repo_root=tmp_path,
        )


def test_missing_crop_is_clear(tmp_path: Path) -> None:
    _seed(tmp_path)
    open_manifest(tmp_path, "crops").path.write_text("", encoding="utf-8")
    with pytest.raises(ReviewDecisionError, match="missing crop"):
        commit_decision(
            tmp_path,
            crop_id="MHGR_000001_P0001_L0001_C_full_01",
            decision_label="rejected",
            gold_text=None,
            selected_candidate_text=None,
            reviewer_id="local_reviewer",
            repo_root=tmp_path,
        )


def test_missing_review_row_is_clear(tmp_path: Path) -> None:
    _seed(tmp_path)
    (tmp_path / "review.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ReviewDecisionError, match="missing review"):
        commit_decision(
            tmp_path,
            crop_id="MHGR_000001_P0001_L0001_C_full_01",
            decision_label="rejected",
            gold_text=None,
            selected_candidate_text=None,
            reviewer_id="local_reviewer",
            repo_root=tmp_path,
        )


def test_missing_page_context_is_clear(tmp_path: Path) -> None:
    from indic_ocr_pipeline.review.app import ReviewApp

    _seed(tmp_path, with_page=False)
    _write_batch(tmp_path)
    app = ReviewApp(tmp_path, tmp_path, "local_reviewer")
    with pytest.raises(Exception, match="missing page|no image"):
        app.context_bytes("MHGR_000001_P0001_L0001_C_full_01")


def test_enqueue_pending_leaves_existing_batch_rows(tmp_path: Path) -> None:
    reviews = _fake_reviews()[:3]
    for row in reviews:
        row["review_id"] = row["crop_id"] + "_REV"
        row["status"] = "pending"
    reviews[0]["status"] = "complete"
    _rewrite(tmp_path / "review.jsonl", reviews)
    _rewrite(
        tmp_path / "verified.jsonl",
        [{"crop_id": reviews[2]["crop_id"], "verified_id": "keep"}],
    )
    existing = {
        "batch_item_id": f"historical:{reviews[0]['crop_id']}",
        "batch_id": "historical",
        "sample_index": 7,
        "crop_id": reviews[0]["crop_id"],
        "review_id": reviews[0]["crop_id"] + "_REV",
        "document_id": reviews[0]["document_id"],
        "page_id": reviews[0]["page_id"],
        "parent_line_id": reviews[0]["parent_line_id"],
        "region_id": "MHGR_000001_P0001_R0001",
        "review_priority": "P0",
        "agreement_category": "ALL_3_DIFFER",
        "priority_rank": reviews[0]["priority_rank"],
        "risk_score": reviews[0]["risk_score"],
        "seed": 1,
    }
    _rewrite(tmp_path / "review_batch.jsonl", [existing])
    first = enqueue_pending_reviews(tmp_path, seed=3407)
    assert first["created"] == 1
    assert first["queued"] == 1
    stored = [
        json.loads(line)
        for line in (tmp_path / "review_batch.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert stored[0]["batch_item_id"] == existing["batch_item_id"]
    assert stored[0]["batch_id"] == "historical"
    assert stored[0]["sample_index"] == 7
    assert stored[0]["crop_id"] == reviews[0]["crop_id"]
    assert stored[1]["crop_id"] == reviews[1]["crop_id"]
    assert stored[1]["sample_index"] == 8
    second = enqueue_pending_reviews(tmp_path, seed=3407)
    assert second["created"] == 0
    assert second["queued"] == 0


def test_sampler_is_deterministic() -> None:
    reviews = _fake_reviews()
    first = sample_calibration_batch(
        reviews,
        batch_size=40,
        tier_quota={"P0": None, "P1": 10, "P2": 10, "P3": 10},
        seed=3407,
    )
    second = sample_calibration_batch(
        reviews,
        batch_size=40,
        tier_quota={"P0": None, "P1": 10, "P2": 10, "P3": 10},
        seed=3407,
    )
    assert [item["crop_id"] for item in first] == [item["crop_id"] for item in second]


def test_sampler_includes_all_p0() -> None:
    reviews = _fake_reviews()
    items = sample_calibration_batch(
        reviews,
        batch_size=40,
        tier_quota={"P0": None, "P1": 10, "P2": 10, "P3": 10},
        seed=1,
    )
    p0 = [row for row in reviews if row["diagnostics"]["review_priority"] == "P0"]
    selected = {item["crop_id"] for item in items}
    assert {row["crop_id"] for row in p0} <= selected


def test_sampler_includes_p2_and_p3() -> None:
    reviews = _fake_reviews()
    summary = summarize_batch(
        sample_calibration_batch(
            reviews,
            batch_size=40,
            tier_quota={"P0": None, "P1": 10, "P2": 10, "P3": 10},
            seed=2,
        )
    )
    assert summary["tiers"]["P2"] > 0
    assert summary["tiers"]["P3"] > 0


def test_sampling_does_not_duplicate_crop_ids() -> None:
    items = sample_calibration_batch(
        _fake_reviews(),
        batch_size=40,
        tier_quota={"P0": None, "P1": 10, "P2": 10, "P3": 10},
        seed=3,
    )
    ids = [item["crop_id"] for item in items]
    assert len(ids) == len(set(ids))


def test_sampling_reproducible_with_fixed_seed() -> None:
    reviews = _fake_reviews()
    a = sample_calibration_batch(
        reviews, batch_size=30, tier_quota={"P0": None, "P1": 8, "P2": 8, "P3": 8}, seed=99
    )
    b = sample_calibration_batch(
        reviews, batch_size=30, tier_quota={"P0": None, "P1": 8, "P2": 8, "P3": 8}, seed=99
    )
    c = sample_calibration_batch(
        reviews, batch_size=30, tier_quota={"P0": None, "P1": 8, "P2": 8, "P3": 8}, seed=100
    )
    assert [i["crop_id"] for i in a] == [i["crop_id"] for i in b]
    assert [i["crop_id"] for i in a] != [i["crop_id"] for i in c]


def test_pending_to_human_verified(tmp_path: Path) -> None:
    _seed(tmp_path)
    commit_decision(
        tmp_path,
        crop_id="MHGR_000001_P0001_L0001_C_full_01",
        decision_label="human_verified",
        gold_text="अ",
        selected_candidate_text="अ",
        reviewer_id="local_reviewer",
        repo_root=tmp_path,
    )
    review = open_manifest(tmp_path, "review").load()[0]
    assert review["status"] == StageStatus.COMPLETE.value
    assert review["decision"] == ReviewDecision.ACCEPT_CANDIDATE.value
    verified = VerifiedRecord.from_dict(open_manifest(tmp_path, "verified").load()[0])
    assert verified.verification_status is VerificationStatus.HUMAN_VERIFIED


def test_pending_to_human_corrected(tmp_path: Path) -> None:
    _seed(tmp_path)
    commit_decision(
        tmp_path,
        crop_id="MHGR_000001_P0001_L0001_C_full_01",
        decision_label="human_corrected",
        gold_text="अ.",
        selected_candidate_text=None,
        reviewer_id="local_reviewer",
        repo_root=tmp_path,
    )
    verified = open_manifest(tmp_path, "verified").load()[0]
    assert verified["verification_status"] == "human_corrected"
    assert verified["final_text"] == "अ."


def test_pending_to_rejected(tmp_path: Path) -> None:
    _seed(tmp_path)
    commit_decision(
        tmp_path,
        crop_id="MHGR_000001_P0001_L0001_C_full_01",
        decision_label="rejected",
        gold_text=None,
        selected_candidate_text=None,
        reviewer_id="local_reviewer",
        repo_root=tmp_path,
    )
    assert open_manifest(tmp_path, "verified").load()[0]["final_text"] is None


def test_pending_to_bad_boundary(tmp_path: Path) -> None:
    _seed(tmp_path)
    commit_decision(
        tmp_path,
        crop_id="MHGR_000001_P0001_L0001_C_full_01",
        decision_label="bad_boundary",
        gold_text=None,
        selected_candidate_text=None,
        reviewer_id="local_reviewer",
        repo_root=tmp_path,
    )
    row = open_manifest(tmp_path, "verified").load()[0]
    assert row["review_decision"] == "bad_boundary"
    assert row["verification_status"] == "rejected"


def test_completed_review_cannot_be_silently_rewritten(tmp_path: Path) -> None:
    _seed(tmp_path)
    crop_id = "MHGR_000001_P0001_L0001_C_full_01"
    commit_decision(
        tmp_path,
        crop_id=crop_id,
        decision_label="rejected",
        gold_text=None,
        selected_candidate_text=None,
        reviewer_id="local_reviewer",
        repo_root=tmp_path,
    )
    review = open_manifest(tmp_path, "review").get(make_review_id(crop_id))
    assert review is not None
    from indic_ocr_pipeline.errors import ImmutableFieldError

    with pytest.raises(ImmutableFieldError):
        open_manifest(tmp_path, "review").update(
            make_review_id(crop_id),
            {"decision": "bad_boundary"},
        )


def test_calibration_report_separates_outcomes() -> None:
    report = build_calibration_report(
        [
            {
                "crop_id": "a",
                "verification_status": "human_verified",
                "review_decision": "accept_candidate",
                "metadata": {"review_priority": "P3"},
            },
            {
                "crop_id": "b",
                "verification_status": "human_corrected",
                "review_decision": "correct_text",
                "metadata": {"review_priority": "P3"},
            },
            {
                "crop_id": "c",
                "verification_status": "rejected",
                "review_decision": "reject_crop",
                "metadata": {"review_priority": "P1"},
            },
            {
                "crop_id": "d",
                "verification_status": "rejected",
                "review_decision": "bad_boundary",
                "metadata": {"review_priority": "P0"},
            },
        ]
    )
    assert report["auto_approval_enabled"] is False
    assert report["gold_export"] is False
    assert report["tiers"]["P3"]["human_verified"] == 1
    assert report["tiers"]["P3"]["human_corrected"] == 1
    assert report["tiers"]["P3"]["candidate_exact"] == 1
    assert report["tiers"]["P1"]["rejected"] == 1
    assert report["tiers"]["P0"]["bad_boundary"] == 1


def test_no_auto_approval_by_default(tmp_path: Path) -> None:
    _seed(tmp_path)
    result = commit_decision(
        tmp_path,
        crop_id="MHGR_000001_P0001_L0001_C_full_01",
        decision_label="human_verified",
        gold_text="अ",
        selected_candidate_text="अ",
        reviewer_id="local_reviewer",
        repo_root=tmp_path,
    )
    assert result["verified"]["metadata"]["auto_approval_enabled"] is False


def test_no_gold_export_in_phase_8(tmp_path: Path) -> None:
    _seed(tmp_path)
    commit_decision(
        tmp_path,
        crop_id="MHGR_000001_P0001_L0001_C_full_01",
        decision_label="human_corrected",
        gold_text="अ",
        selected_candidate_text=None,
        reviewer_id="local_reviewer",
        repo_root=tmp_path,
    )
    assert not (tmp_path / "gold.jsonl").exists()
    assert open_manifest(tmp_path, "verified").load()[0]["metadata"]["gold_export"] is False


def test_review_ui_is_local_only() -> None:
    html = (
        Path(__file__).resolve().parents[1]
        / "src/indic_ocr_pipeline/review/static/index.html"
    ).read_text(encoding="utf-8")
    assert "cdn" not in html.lower()
    assert "http://" not in html
    assert "https://" not in html
    assert "googleapis" not in html


def test_no_network_and_no_ocr_modules(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network used")

    monkeypatch.setattr(socket, "socket", fail)
    monkeypatch.setattr(socket, "create_connection", fail)
    before = set(sys.modules)
    _seed(tmp_path)
    commit_decision(
        tmp_path,
        crop_id="MHGR_000001_P0001_L0001_C_full_01",
        decision_label="rejected",
        gold_text=None,
        selected_candidate_text=None,
        reviewer_id="local_reviewer",
        repo_root=tmp_path,
    )
    loaded = set(sys.modules) - before
    assert not any(name.startswith("indic_ocr_pipeline.ocr") for name in loaded)
    assert not any(name.split(".")[0] in {"torch", "transformers", "requests"} for name in loaded)


def test_verified_record_round_trip() -> None:
    record = build_verified_record(
        crop={
            "crop_id": "MHGR_000001_P0001_L0001_C_full_01",
            "document_id": "MHGR_000001",
            "page_id": "MHGR_000001_P0001",
            "parent_line_id": "MHGR_000001_P0001_L0001",
            "region_id": "MHGR_000001_P0001_R0001",
            "image_path": "x.png",
            "sha256": "a" * 64,
        },
        review={
            "risk_score": 0.2,
            "diagnostics": {"agreement_category": "ALL_3_AGREE", "review_priority": "P3"},
        },
        decision_label="human_verified",
        gold_text="  अ  ",
        selected_candidate_text="अ",
        reviewer_id="local_reviewer",
        notes="ok",
        error_tags=[],
        candidate_texts={"indicocr": "अ", "chitrapathak": "अ", "tesseract": "अ"},
        timestamp="2026-10-07T12:00:00+00:00",
    )
    assert record.verified_id == make_verified_id(record.crop_id)
    assert record.final_text == apply_storage_normalization("  अ  ")
    again = VerifiedRecord.from_dict(record.to_dict())
    assert again.final_text == record.final_text


def _seed(tmp_path: Path, *, with_page: bool = True) -> None:
    crop_id = "MHGR_000001_P0001_L0001_C_full_01"
    image = tmp_path / "crop.png"
    Image.new("L", (40, 16), 255).save(image)
    digest = sha256_file(image)
    crop = {
        "crop_id": crop_id,
        "document_id": "MHGR_000001",
        "page_id": "MHGR_000001_P0001",
        "parent_line_id": "MHGR_000001_P0001_L0001",
        "region_id": "MHGR_000001_P0001_R0001",
        "image_path": str(image),
        "sha256": digest,
        "bbox": [10, 10, 50, 26],
    }
    review = ReviewRecord(
        review_id=make_review_id(crop_id),
        crop_id=crop_id,
        candidate_id=f"{crop_id}_CAND",
        document_id="MHGR_000001",
        page_id="MHGR_000001_P0001",
        parent_line_id="MHGR_000001_P0001_L0001",
        priority_rank=1,
        risk_score=0.5,
        audit_sample=False,
        diagnostics={
            "agreement_category": "ALL_3_DIFFER",
            "review_priority": "P1",
            "image_path": str(image),
            "region_id": "MHGR_000001_P0001_R0001",
            "indicocr_text": "अ",
            "chitrapathak_text": "ब",
            "tesseract_text": "क",
            "proposed_consensus": None,
            "flags": [],
        },
        status=StageStatus.PENDING,
    ).to_dict()
    _rewrite(tmp_path / "crops.jsonl", [crop])
    _rewrite(tmp_path / "review.jsonl", [review])
    _rewrite(
        tmp_path / "lines.jsonl",
        [{"line_id": "MHGR_000001_P0001_L0001", "bbox": [10, 10, 50, 26]}],
    )
    if with_page:
        page_png = tmp_path / "page.png"
        Image.new("RGB", (100, 100), (255, 255, 255)).save(page_png)
        _rewrite(
            tmp_path / "pages.jsonl",
            [
                {
                    "page_id": "MHGR_000001_P0001",
                    "image_path": str(page_png),
                    "document_id": "MHGR_000001",
                }
            ],
        )
    _rewrite(tmp_path / "regions.jsonl", [])


def _write_batch(tmp_path: Path) -> None:
    _rewrite(
        tmp_path / "review_batch.jsonl",
        [
            {
                "batch_item_id": "t:MHGR_000001_P0001_L0001_C_full_01",
                "batch_id": "t",
                "sample_index": 1,
                "crop_id": "MHGR_000001_P0001_L0001_C_full_01",
                "review_priority": "P1",
            }
        ],
    )


def _fake_reviews() -> list[dict]:
    rows = []
    rank = 1
    for document in ("MHGR_000001", "MHGR_000002", "MHGR_000003"):
        for tier, count in (("P0", 2), ("P1", 8), ("P2", 6), ("P3", 6)):
            for _index in range(count):
                crop_id = f"{document}_P0001_L{rank:04d}_C_full_01"
                rows.append(
                    {
                        "crop_id": crop_id,
                        "document_id": document,
                        "page_id": f"{document}_P0001",
                        "parent_line_id": f"{document}_P0001_L{rank:04d}",
                        "priority_rank": rank,
                        "risk_score": 0.1 * (5 - int(tier[1])),
                        "diagnostics": {
                            "review_priority": tier,
                            "agreement_category": "ALL_3_AGREE"
                            if tier == "P3"
                            else "ALL_3_DIFFER",
                            "region_id": f"{document}_P0001_R0001",
                        },
                    }
                )
                rank += 1
    return rows


def _rewrite(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
