"""Data-status rules: agreement is never human gold."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pipeline.audit_budget import allocate_audit
from pipeline.dataset_flow import export_datasets
from pipeline.golden_sets import write_auto_accepted, write_golden
from pipeline.quality import quality_report, resolved_data_status
from training.dataset import assert_training_rows_safe

from indic_ocr_pipeline.ocr.consensus import classify_consensus


def _same(names: tuple[str, ...], text: str = "क") -> dict[str, object]:
    return classify_consensus({name: text for name in names}, backends=names)


def _audits(kind: str, text: str, count: int) -> list[dict[str, str]]:
    return [
        {"agreement_class": kind, "human_verified_text": text, "proposed_text": text}
        for _ in range(count)
    ]


def test_three_model_agreement_is_not_gold() -> None:
    row = classify_consensus({"indicocr": "क", "chitrapathak": "क", "tesseract": "क"})
    assert row["gold"] is False
    assert row["acceptance_candidate"] == "auto_accepted_unanimous"
    assert row["data_status"] == "pending_calibration"
    assert row["models_configured"] == 3
    assert row["models_agreed"] == 3
    report = quality_report(
        [row] * 30,
        _audits("ALL_3_AGREE", "क", 30),
        threshold=0.5,
        min_audited=30,
    )
    assert resolved_data_status(row, report) == "auto_accepted_unanimous"


def test_majority_is_a_suggestion_until_the_category_is_calibrated() -> None:
    row = classify_consensus({"indicocr": "ब", "chitrapathak": "ब", "tesseract": "क"})
    assert row["gold"] is False
    assert row["agreement_class"] == "TWO_AGREE"
    assert row["acceptance_candidate"] == "auto_accepted_majority"
    assert row["data_status"] == "needs_human_review"
    assert row["models_agreed"] == 2
    assert row["models_differed"] == 1
    assert "tesseract" in row["differing_similarity"]
    assert resolved_data_status(row, {"classes": {}}) == "needs_human_review"


def test_no_agreement_stays_in_human_review() -> None:
    row = classify_consensus({"indicocr": "ड", "chitrapathak": "ई", "tesseract": "फ"})
    assert row["data_status"] == "needs_human_review"
    assert row["acceptance_candidate"] is None
    assert row["gold"] is False


def test_model_counts_from_two_through_five() -> None:
    two = _same(("one", "two"))
    assert two["consensus_status"] == "CONSENSUS_2_2"
    assert two["status"] != "AUTO_3_AGREE"
    assert two["stricter_reliability"] is True
    assert two["acceptance_candidate"] == "auto_accepted_unanimous"
    assert two["gold"] is False

    three = _same(("indicocr", "chitrapathak", "tesseract"))
    assert three["consensus_status"] == "CONSENSUS_3_3"
    assert three["status"] == "AUTO_3_AGREE"

    four = _same(("a", "b", "c", "d"))
    assert four["consensus_status"] == "CONSENSUS_4_4"
    assert four["models_configured"] == 4

    five = _same(("a", "b", "c", "d", "e"))
    assert five["consensus_status"] == "CONSENSUS_5_5"
    assert five["models_agreed"] == 5

    split = classify_consensus(
        {"a": "क", "b": "क", "c": "क", "d": "ख", "e": "ख"},
        backends=("a", "b", "c", "d", "e"),
    )
    assert split["consensus_status"] == "CONSENSUS_3_5"
    assert split["acceptance_candidate"] == "auto_accepted_majority"
    assert split["gold"] is False


def test_two_model_agreement_needs_more_audits_than_three() -> None:
    population = [
        {"agreement_class": "CONSENSUS_2_2", "models_configured": 2} for _ in range(40)
    ]
    audits = _audits("CONSENSUS_2_2", "क", 40)
    report = quality_report(population, audits, threshold=0.5, min_audited=30)
    stats = report["classes"]["CONSENSUS_2_2"]
    assert stats["min_audited"] == 60
    assert stats["calibration_state"] == "pending_calibration"
    assert stats["auto_accept"] is False


def test_uncalibrated_category_is_not_written_to_auto_accepted(tmp_path: Path) -> None:
    row = classify_consensus({"indicocr": "क", "chitrapathak": "क", "tesseract": "क"})
    report = quality_report([row], [], threshold=0.9, min_audited=30)
    assert report["classes"]["ALL_3_AGREE"]["auto_accept"] is False
    accepted = write_auto_accepted(
        tmp_path,
        [row | {"crop_id": "C1"}],
        report,
        human_ids=set(),
    )
    assert accepted == []


def test_threshold_uses_the_lower_bound_not_the_observed_rate() -> None:
    population = [{"agreement_class": "ALL_3_AGREE", "models_configured": 3} for _ in range(40)]
    audits = _audits("ALL_3_AGREE", "क", 30)
    audits += [
        {"agreement_class": "ALL_3_AGREE", "human_verified_text": "ख", "proposed_text": "क"}
        for _ in range(10)
    ]
    report = quality_report(population, audits, threshold=0.9, min_audited=30)
    stats = report["classes"]["ALL_3_AGREE"]
    assert stats["precision"] == 0.75
    assert stats["precision"] > stats["interval_low"]
    assert stats["auto_accept"] is False


def test_sensitive_unanimous_line_is_not_auto_accepted(tmp_path: Path) -> None:
    row = classify_consensus({"indicocr": "₹100", "chitrapathak": "₹100", "tesseract": "₹100"})
    assert row["sensitive"] is True
    assert row["acceptance_candidate"] is None
    assert row["data_status"] == "needs_human_review"
    report = {"classes": {"ALL_3_AGREE": {"auto_accept": True, "calibration_state": "accepted"}}}
    assert write_auto_accepted(tmp_path, [row | {"crop_id": "S"}], report, human_ids=set()) == []


def test_human_approval_and_correction_are_golden_and_rejection_is_not(tmp_path: Path) -> None:
    approved = {
        "crop_id": "A",
        "gold_text": "क",
        "verification_method": "human",
        "decision": "HUMAN_VERIFIED",
    }
    corrected = {
        "crop_id": "B",
        "gold_text": "ख",
        "verification_method": "human",
        "decision": "HUMAN_CORRECTED",
    }
    rejected = {
        "crop_id": "C",
        "gold_text": "ग",
        "verification_method": "human",
        "decision": "HUMAN_REJECTED",
        "data_status": "rejected",
    }
    auto = {
        "crop_id": "D",
        "gold_text": "घ",
        "verification_method": "auto_auto_accepted_unanimous",
        "data_status": "auto_accepted_unanimous",
        "gold": False,
    }
    count = write_golden(tmp_path, [approved, corrected, rejected, auto])
    saved = [
        json.loads(line)
        for line in (tmp_path / "data" / "golden" / "records.jsonl").read_text().splitlines()
    ]
    assert count == 2
    assert [row["crop_id"] for row in saved] == ["A", "B"]
    assert all(row["gold"] is True for row in saved)
    assert saved[0]["data_status"] == "human_verified"
    assert saved[1]["data_status"] == "human_corrected"
    assert "D" not in {row["crop_id"] for row in saved}


def test_training_includes_auto_rows_only_when_asked(tmp_path: Path) -> None:
    auto = {
        "sample_id": "S",
        "crop_id": "C",
        "data_status": "auto_accepted_unanimous",
        "verification_method": "auto_auto_accepted_unanimous",
        "gold": False,
    }
    with pytest.raises(Exception, match="not human gold"):
        assert_training_rows_safe([auto], set(), tmp_path)
    assert_training_rows_safe([auto], set(), tmp_path, allow_auto=True)
    tainted = dict(auto)
    tainted["gold"] = True
    with pytest.raises(Exception, match="cannot have gold true"):
        assert_training_rows_safe([tainted], set(), tmp_path, allow_auto=True)


def test_review_sample_covers_documents_and_keeps_budget_for_disagreement() -> None:
    rows = []
    for index in range(12):
        rows.append(
            {
                "crop_id": f"S{index}",
                "document_id": "BIG",
                "agreement_class": "ALL_3_AGREE",
                "sensitive": True,
                "proposed_text": "12",
            }
        )
    for index in range(12):
        rows.append(
            {
                "crop_id": f"D{index}",
                "document_id": f"DOC{index % 3}",
                "agreement_class": "ALL_DIFFER",
                "proposed_text": "line",
            }
        )
    chosen = allocate_audit(rows, 10, seed=3407)
    assert len(chosen) == 10
    assert sum(row.get("sensitive") is True for row in chosen) < 10
    assert any(row["agreement_class"] == "ALL_DIFFER" for row in chosen)
    assert len({row["document_id"] for row in chosen}) > 1


def test_export_keeps_existing_human_reviews_and_skips_frozen_ids(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    human = {
        "crop_id": "HUMAN1",
        "document_id": "D",
        "page_id": "P",
        "line_id": "L",
        "crop_path": "crops/human.png",
        "reviewer_id": "local_reviewer",
        "reviewed_at": "2026-10-09T00:00:00+00:00",
        "review_origin": "human",
        "decision": "HUMAN_VERIFIED",
        "gold_text": "क",
        "original_ocr": {"indicocr": "क"},
    }
    frozen = dict(human)
    frozen["crop_id"] = "FROZEN1"
    text = json.dumps(human) + "\n"
    (manifests / "verified.jsonl").write_text(text, encoding="utf-8")
    (manifests / "consensus.jsonl").write_text(
        json.dumps(
            {
                "crop_id": "HUMAN1",
                "agreement_class": "ALL_3_AGREE",
                "consensus_status": "CONSENSUS_3_3",
                "proposed_text": "क",
                "models_configured": 3,
                "sensitive": False,
                "gold": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    frozen_dir = tmp_path / "data" / "eval_data"
    frozen_dir.mkdir(parents=True)
    (frozen_dir / "validation.jsonl").write_text(
        json.dumps({"sample_id": "FROZEN1", "crop_id": "FROZEN1", "image_filename": "a.png"})
        + "\n",
        encoding="utf-8",
    )
    config = {"paths": {"frozen_benchmark": "data/eval_data"}, "seed": 3407}
    export_datasets(tmp_path, config)
    again = export_datasets(tmp_path, config)
    assert (manifests / "verified.jsonl").read_text(encoding="utf-8") == text
    golden = (tmp_path / "data" / "golden" / "records.jsonl").read_text(encoding="utf-8")
    assert "HUMAN1" in golden
    assert "FROZEN1" not in golden
    assert again["golden"] == 1
