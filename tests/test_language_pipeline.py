"""Language-agnostic consensus, audit budget, and golden-set rules."""

from __future__ import annotations

from pathlib import Path

import pytest
from pipeline.audit_budget import allocate_audit
from pipeline.cli import main
from pipeline.golden_sets import write_auto_accepted, write_golden
from pipeline.quality import quality_report
from pipeline.registry import parse_model_flags, resolve_models
from training.dataset import assert_training_rows_safe

from indic_ocr_pipeline.ocr.consensus import classify_consensus


def test_pairwise_similarity_separates_a_near_miss_from_a_different_string() -> None:
    near = classify_consensus(
        {"indicocr": "abcdef", "chitrapathak": "abcdef", "tesseract": "abcdeg"}
    )
    far = classify_consensus(
        {"indicocr": "abcdef", "chitrapathak": "abcdef", "tesseract": "zzzzzz"}
    )
    assert near["agreement_class"] == "TWO_AGREE"
    assert near["consensus_status"] == "CONSENSUS_2_3"
    assert near["gold"] is False
    near_score = near["pairwise_similarity"]["indicocr__tesseract"]
    far_score = far["pairwise_similarity"]["indicocr__tesseract"]
    assert near_score > far_score


def test_consensus_accepts_more_than_three_models() -> None:
    texts = {"one": "क", "two": "क", "three": "क", "four": "क"}
    row = classify_consensus(texts, backends=("one", "two", "three", "four"))
    assert row["consensus_status"] == "CONSENSUS_4_4"
    assert row["gold"] is False
    majority = classify_consensus(
        {"one": "क", "two": "क", "three": "क", "four": "ख"},
        backends=("one", "two", "three", "four"),
    )
    assert majority["consensus_status"] == "CONSENSUS_3_4"
    assert majority["proposed_text"] == "क"


def test_sensitive_lines_stay_out_of_auto_accept(tmp_path: Path) -> None:
    row = classify_consensus({"indicocr": "₹100", "chitrapathak": "₹100", "tesseract": "₹100"})
    assert row["sensitive"] is True
    assert row["review_required"] is True
    report = {"classes": {"ALL_3_AGREE": {"auto_accept": True}}}
    accepted = write_auto_accepted(
        tmp_path,
        [row | {"crop_id": "C1", "proposed_text": "₹100", "agreement_class": "ALL_3_AGREE"}],
        report,
        human_ids=set(),
    )
    assert accepted == []


def test_audit_budget_spends_more_on_disagreement_than_unanimous(tmp_path: Path) -> None:
    del tmp_path
    rows = []
    for index in range(30):
        rows.append(
            {
                "crop_id": f"D{index:03d}",
                "document_id": f"DOC{index % 3}",
                "agreement_class": "ALL_DIFFER",
                "proposed_text": "x" * (index + 1),
            }
        )
        rows.append(
            {
                "crop_id": f"A{index:03d}",
                "document_id": f"DOC{index % 3}",
                "agreement_class": "ALL_3_AGREE",
                "proposed_text": "y" * 20,
            }
        )
    chosen = allocate_audit(rows, 20, seed=3407)
    differ = sum(row["agreement_class"] == "ALL_DIFFER" for row in chosen)
    agree = sum(row["agreement_class"] == "ALL_3_AGREE" for row in chosen)
    assert len(chosen) == 20
    assert differ > agree
    again = allocate_audit(rows, 20, seed=3407)
    assert [row["crop_id"] for row in chosen] == [row["crop_id"] for row in again]


def test_sensitive_lines_are_taken_before_the_rest_of_the_budget() -> None:
    rows = [
        {
            "crop_id": "S1",
            "document_id": "DOC",
            "agreement_class": "ALL_3_AGREE",
            "sensitive": True,
            "proposed_text": "12",
        },
        *[
            {
                "crop_id": f"A{index}",
                "document_id": "DOC",
                "agreement_class": "ALL_DIFFER",
                "proposed_text": "text",
            }
            for index in range(10)
        ],
    ]
    chosen = allocate_audit(rows, 3, seed=3407)
    assert "S1" in {row["crop_id"] for row in chosen}


def test_auto_accept_requires_the_lower_bound() -> None:
    population = [{"agreement_class": "ALL_3_AGREE"} for _ in range(10)]
    audits = [
        {"agreement_class": "ALL_3_AGREE", "human_verified_text": "क", "proposed_text": "क"}
        for _ in range(10)
    ]
    blocked = quality_report(population, audits, threshold=0.99, min_audited=1)
    allowed = quality_report(population, audits, threshold=0.5, min_audited=1)
    unset = quality_report(population, audits, threshold=None, min_audited=1)
    pending = quality_report(population, audits, threshold=0.5)
    stats = allowed["classes"]["ALL_3_AGREE"]
    assert blocked["classes"]["ALL_3_AGREE"]["precision"] == 1
    assert blocked["classes"]["ALL_3_AGREE"]["auto_accept"] is False
    assert stats["auto_accept"] is True
    assert stats["interval_low"] < stats["precision"]
    assert unset["classes"]["ALL_3_AGREE"]["auto_accept"] is False
    assert pending["classes"]["ALL_3_AGREE"]["auto_accept"] is False
    assert pending["classes"]["ALL_3_AGREE"]["calibration_state"] == "pending_calibration"


def test_golden_is_human_and_training_rejects_auto_rows(tmp_path: Path) -> None:
    count = write_golden(
        tmp_path,
        [
            {"crop_id": "C1", "gold_text": "क", "verification_method": "human"},
            {
                "crop_id": "C2",
                "gold_text": "ख",
                "verification_method": "auto_consensus_all_3_agree",
            },
        ],
    )
    assert count == 1
    saved = (tmp_path / "data" / "golden" / "records.jsonl").read_text(encoding="utf-8")
    assert "auto_consensus" not in saved
    assert '"verification_method": "human"' in saved
    with pytest.raises(Exception, match="not human gold"):
        assert_training_rows_safe(
            [{"sample_id": "S", "verification_method": "auto_consensus_all_3_agree"}],
            set(),
            tmp_path,
        )


def test_local_model_path_wins_over_a_huggingface_id(tmp_path: Path) -> None:
    weights = tmp_path / "weights"
    weights.mkdir()
    tess = tmp_path / "tess"
    tess.mkdir()
    slots = parse_model_flags(
        [
            {"name": "indicocr", "path": str(weights), "hf": "org/should-not-download"},
            {"name": "tesseract", "path": str(tess), "hf": "org/other"},
        ]
    )
    resolved = resolve_models(slots, tmp_path / "cache", environ={})
    by_name = {item.name: item for item in resolved}
    assert by_name["indicocr"].path == str(weights.resolve())
    assert by_name["indicocr"].revision == "local"
    assert not (tmp_path / "cache").exists()


def test_one_model_is_rejected() -> None:
    assert main(["--model-1", "alpha", "--model-1-path", "/tmp/missing-model"]) == 2


def test_language_runs_stay_separate(tmp_path: Path) -> None:
    from pipeline.orchestrator import _output_runs

    marathi = _output_runs(tmp_path, {"language": "marathi", "language_profile": {}})
    gujarati = _output_runs(
        tmp_path,
        {"language": "gujarati", "language_profile": {"runs_dir": "runs/gujarati"}},
    )
    bengali = _output_runs(
        tmp_path,
        {"language": "bengali", "language_profile": {"runs_dir": "runs/bengali"}},
    )
    assert marathi == tmp_path / "runs"
    assert gujarati == tmp_path / "runs" / "gujarati"
    assert bengali == tmp_path / "runs" / "bengali"
    assert len({marathi, gujarati, bengali}) == 3
