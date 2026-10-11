"""Phase 7 compares predictions and ranks review. It does not create gold."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from indic_ocr_pipeline.config import load_settings
from indic_ocr_pipeline.errors import RankingError
from indic_ocr_pipeline.ids import make_candidate_id, make_prediction_id, make_review_id
from indic_ocr_pipeline.manifests import open_manifest
from indic_ocr_pipeline.schemas import ReviewRecord, VerificationStatus
from indic_ocr_pipeline.scoring.metrics import (
    comparison_key,
    edit_counts,
    pair_metrics,
    proposed_consensus,
)
from indic_ocr_pipeline.scoring.queue import format_report, rank_review_queue
from indic_ocr_pipeline.scoring.rank import WEIGHT_KEYS, score_predictions
from indic_ocr_pipeline.utils.hashing import sha256_file

WEIGHTS = {
    "agreement": 0.45,
    "tesseract_confidence": 0.15,
    "image_quality": 0.15,
    "crop_geometry": 0.10,
    "unicode_sanity": 0.15,
}


def test_three_identical_predictions_agree() -> None:
    result = _score("महाराष्ट्र", "महाराष्ट्र", "महाराष्ट्र")
    assert result.agreement_category == "ALL_3_AGREE"
    assert result.agreement_category != "correct"


def test_two_identical_and_one_different() -> None:
    result = _score("महाराष्ट्र", "महाराष्ट्र", "महाराष्ट्र शासन")
    assert result.agreement_category == "TWO_AGREE_ONE_DIFFERS"


def test_three_different_predictions() -> None:
    result = _score("अ", "ब", "क")
    assert result.agreement_category == "ALL_3_DIFFER"


def test_empty_prediction_is_detected() -> None:
    result = _score("अ", "", "अ")
    assert result.agreement_category == "ONE_OR_MORE_EMPTY"
    assert "empty_prediction" in result.flags
    assert result.proposed_consensus == "अ"
    blank = _score("", "", "")
    assert blank.proposed_consensus is None
    assert blank.review_risk_score == 1.0
    assert blank.review_priority == "P0"


def test_consensus_only_when_a_majority_exists() -> None:
    assert proposed_consensus(_texts("अ", "अ", "ब")) == "अ"
    assert proposed_consensus(_texts("अ", "अ", "अ")) == "अ"


def test_all_different_predictions_have_no_fabricated_consensus() -> None:
    result = _score("अ", "ब", "क")
    assert result.proposed_consensus is None


def test_nfc_normalization_does_not_replace_the_prediction() -> None:
    combining = "e\u0301"
    composed = "\u00e9"
    assert comparison_key(combining) == comparison_key(composed)
    assert proposed_consensus(_texts(combining, combining, combining)) == combining


def test_whitespace_normalization_collapses_only_the_comparison_key() -> None:
    assert comparison_key("  a   b  ") == "a b"
    assert proposed_consensus(_texts("a  b", "a  b", "a  b")) == "a  b"


def test_punctuation_is_preserved() -> None:
    assert comparison_key("मुंबई-") == "मुंबई-"
    assert comparison_key("अ.") == "अ."


def test_marathi_digits_are_preserved() -> None:
    assert comparison_key("४००") == "४००"
    assert comparison_key("०३") == "०३"


def test_internal_spaces_are_not_deleted_from_the_prediction() -> None:
    text = "मुंबई  ४००"
    assert "  " not in comparison_key(text)
    assert proposed_consensus(_texts(text, text, text)) == text


def test_pairwise_edit_distance() -> None:
    metrics = pair_metrics("kitten", "sitting")
    assert metrics.edit_distance == 3
    assert metrics.prediction_pair_cer == round(3 / 7, 4)


def test_pairwise_insertion_deletion_substitution_counts() -> None:
    assert edit_counts("ab", "acb") == (1, 1, 0, 0)
    assert edit_counts("ab", "a") == (1, 0, 1, 0)
    assert edit_counts("ab", "cb") == (1, 0, 0, 1)


def test_prediction_pair_cer_is_not_named_as_ground_truth() -> None:
    payload = pair_metrics("ab", "a").to_dict()
    assert "prediction_pair_cer" in payload
    assert "ground_truth_cer" not in payload


def test_tesseract_confidence_stays_on_its_stored_scale() -> None:
    result = _score("अ", "ब", "क", confidence=0.7659)
    assert result.tesseract_confidence == 0.7659


def test_null_confidence_remains_null() -> None:
    result = _score("अ", "अ", "अ", confidence=None)
    assert result.tesseract_confidence is None


def test_no_fabricated_confidence_for_models_without_one(tmp_path: Path) -> None:
    rank_review_queue(_write_case(tmp_path, "अ", "अ", "अ"), weights=WEIGHTS)
    review = open_manifest(tmp_path, "review").load()[0]
    assert review["diagnostics"]["indicocr_confidence"] is None
    assert review["diagnostics"]["chitrapathak_confidence"] is None


def test_risk_score_is_deterministic() -> None:
    first = _score("अ", "ब", "क", confidence=0.4)
    second = _score("अ", "ब", "क", confidence=0.4)
    assert first.review_risk_score == second.review_risk_score
    assert first.reasons == second.reasons


def test_risk_weights_sum_to_one() -> None:
    weights = load_settings().scoring.weights
    assert set(weights) == set(WEIGHT_KEYS)
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_large_edit_distance_ranks_ahead_of_a_one_character_difference() -> None:
    short = _score("अ", "ब", "क")
    long = _score("अ" * 30, "ब", "क")
    assert long.max_edit_distance > short.max_edit_distance
    assert long.sort_key("MHGR_000001_P0001_L0002_C_full_01") < short.sort_key(
        "MHGR_000001_P0001_L0001_C_full_01"
    )


def test_risk_score_increases_for_stronger_disagreement() -> None:
    agreed = _score("अ", "अ", "अ", confidence=0.9)
    split = _score("अ", "ब", "क", confidence=0.9)
    assert split.review_risk_score > agreed.review_risk_score


def test_empty_prediction_receives_higher_review_priority() -> None:
    empty = _score("अ", "", "ब")
    agreed = _score("अ", "अ", "अ")
    assert empty.review_priority == "P0"
    assert agreed.review_priority == "P3"
    assert empty.review_risk_score > agreed.review_risk_score


def test_three_way_agreement_is_lower_risk_and_not_labeled_correct() -> None:
    agreed = _score("अ", "अ", "अ", confidence=0.95)
    differed = _score("अ", "ब", "क", confidence=0.95)
    assert agreed.review_risk_score < differed.review_risk_score
    assert agreed.agreement_category == "ALL_3_AGREE"
    assert "correct" not in agreed.reasons


def test_candidate_remains_candidate(tmp_path: Path) -> None:
    rank_review_queue(_write_case(tmp_path, "अ", "ब", "क"), weights=WEIGHTS)
    candidate = open_manifest(tmp_path, "candidates").load()[0]
    assert candidate["verification_status"] == VerificationStatus.CANDIDATE.value
    assert candidate["tier"] == "tier_2_candidate"


def test_no_human_verified_status_is_generated(tmp_path: Path) -> None:
    rank_review_queue(_write_case(tmp_path, "अ", "अ", "ब"), weights=WEIGHTS)
    blob = (tmp_path / "candidates.jsonl").read_text(encoding="utf-8")
    blob += (tmp_path / "review.jsonl").read_text(encoding="utf-8")
    assert "human_verified" not in blob
    assert "human_corrected" not in blob


def test_no_gold_records_are_generated(tmp_path: Path) -> None:
    rank_review_queue(_write_case(tmp_path, "अ", "अ", "अ"), weights=WEIGHTS)
    assert not (tmp_path / "gold.jsonl").exists()
    assert not (tmp_path / "verified.jsonl").exists()


def test_stable_review_ids(tmp_path: Path) -> None:
    rank_review_queue(_write_case(tmp_path, "अ", "ब", "क"), weights=WEIGHTS)
    review = open_manifest(tmp_path, "review").load()[0]
    crop_id = "MHGR_000001_P0001_L0001_C_full_01"
    assert review["review_id"] == make_review_id(crop_id)
    assert review["candidate_id"] == make_candidate_id(crop_id)


def test_same_predictions_rerun_to_the_same_output(tmp_path: Path) -> None:
    first = rank_review_queue(_write_case(tmp_path, "अ", "ब", "क"), weights=WEIGHTS)
    before = (tmp_path / "review.jsonl").read_text(encoding="utf-8")
    second = rank_review_queue(tmp_path, weights=WEIGHTS)
    after = (tmp_path / "review.jsonl").read_text(encoding="utf-8")
    assert first["top_risk_crop_ids"] == second["top_risk_crop_ids"]
    assert before == after


def test_ranking_appends_new_crops_without_touching_complete_reviews(
    tmp_path: Path,
) -> None:
    rank_review_queue(_write_case(tmp_path, "अ", "ब", "क"), weights=WEIGHTS)
    review_path = tmp_path / "review.jsonl"
    first_blob = review_path.read_text(encoding="utf-8")
    first = json.loads(first_blob.splitlines()[0])
    first["status"] = "complete"
    first["decision"] = "accept_candidate"
    first["final_text"] = "अ"
    _rewrite(review_path, [first])
    frozen = review_path.read_text(encoding="utf-8")

    crop_id = "MHGR_000001_P0001_L0002_C_full_01"
    crops = open_manifest(tmp_path, "crops").load()
    crops.append(
        {
            "crop_id": crop_id,
            "document_id": "MHGR_000001",
            "page_id": "MHGR_000001_P0001",
            "parent_line_id": "MHGR_000001_P0001_L0002",
            "region_id": "MHGR_000001_P0001_R0001",
            "image_path": (
                "data/crops/full/MHGR_000001/MHGR_000001_P0001/"
                "MHGR_000001_P0001_L0002.png"
            ),
        }
    )
    lines = open_manifest(tmp_path, "lines").load()
    lines.append({"line_id": "MHGR_000001_P0001_L0002", "bbox": [0, 40, 400, 70]})
    predictions = open_manifest(tmp_path, "predictions").load()
    for name, text in _texts("एक", "दोन", "तीन").items():
        predictions.append(
            {
                "prediction_id": make_prediction_id(crop_id, name),
                "crop_id": crop_id,
                "document_id": "MHGR_000001",
                "page_id": "MHGR_000001_P0001",
                "parent_line_id": "MHGR_000001_P0001_L0002",
                "backend": name,
                "text": text,
                "confidence": 0.5 if name == "tesseract" else None,
                "status": "complete",
                "error": None,
            }
        )
    _rewrite(tmp_path / "crops.jsonl", crops)
    _rewrite(tmp_path / "lines.jsonl", lines)
    _rewrite(tmp_path / "predictions.jsonl", predictions)

    summary = rank_review_queue(tmp_path, weights=WEIGHTS)
    reviews = open_manifest(tmp_path, "review").load()
    assert summary["total_crops"] == 2
    assert len(reviews) == 2
    assert reviews[0] == first
    assert reviews[0]["status"] == "complete"
    assert reviews[0]["decision"] == "accept_candidate"
    assert reviews[0]["final_text"] == "अ"
    assert json.loads(frozen.splitlines()[0]) == reviews[0]
    assert reviews[1]["crop_id"] == crop_id
    assert reviews[1]["status"] == "pending"
    assert reviews[1]["decision"] is None


def test_changed_prediction_payload_conflicts(tmp_path: Path) -> None:
    rank_review_queue(_write_case(tmp_path, "अ", "अ", "अ"), weights=WEIGHTS)
    before = (tmp_path / "candidates.jsonl").read_text(encoding="utf-8")
    _replace_prediction_text(tmp_path, "indicocr", "ब")
    with pytest.raises(RankingError, match="changed"):
        rank_review_queue(tmp_path, weights=WEIGHTS)
    assert (tmp_path / "candidates.jsonl").read_text(encoding="utf-8") == before


def test_missing_prediction_is_a_clear_error(tmp_path: Path) -> None:
    _write_case(tmp_path, "अ", "अ", "अ")
    predictions = open_manifest(tmp_path, "predictions").load()
    kept = [row for row in predictions if row["backend"] != "tesseract"]
    _rewrite(tmp_path / "predictions.jsonl", kept)
    with pytest.raises(RankingError, match="missing predictions"):
        rank_review_queue(tmp_path, weights=WEIGHTS)


def test_duplicate_prediction_for_same_engine_is_a_clear_error(tmp_path: Path) -> None:
    _write_case(tmp_path, "अ", "अ", "अ")
    rows = open_manifest(tmp_path, "predictions").load()
    extra = dict(rows[0])
    extra["prediction_id"] = extra["prediction_id"] + "_DUP"
    rows.append(extra)
    _rewrite(tmp_path / "predictions.jsonl", rows)
    with pytest.raises(RankingError, match="duplicate"):
        rank_review_queue(tmp_path, weights=WEIGHTS)


def test_unmatched_prediction_is_a_clear_error(tmp_path: Path) -> None:
    _write_case(tmp_path, "अ", "अ", "अ")
    rows = open_manifest(tmp_path, "predictions").load()
    rows[0]["crop_id"] = "MHGR_000001_P0001_L0002_C_full_01"
    _rewrite(tmp_path / "predictions.jsonl", rows)
    with pytest.raises(RankingError, match="does not match a crop"):
        rank_review_queue(tmp_path, weights=WEIGHTS)


def test_jsonl_manifest_validity(tmp_path: Path) -> None:
    rank_review_queue(_write_case(tmp_path, "अ", "ब", "क", confidence=0.5), weights=WEIGHTS)
    for name in ("candidates", "review"):
        path = tmp_path / f"{name}.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines
        for line in lines:
            json.loads(line)
    review = ReviewRecord.from_dict(open_manifest(tmp_path, "review").load()[0])
    assert review.decision is None
    assert review.diagnostics["tesseract_confidence"] == 0.5
    assert review.diagnostics["tesseract_confidence_scale"].startswith("stored_0_to_1")


def test_input_manifests_remain_unchanged(tmp_path: Path) -> None:
    _write_case(tmp_path, "अ", "अ", "ब")
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-1.4 unchanged")
    page = tmp_path / "pages.jsonl"
    page.write_text('{"page_id":"keep"}\n', encoding="utf-8")
    before = {
        "pages": sha256_file(page),
        "crops": sha256_file(tmp_path / "crops.jsonl"),
        "predictions": sha256_file(tmp_path / "predictions.jsonl"),
        "source": sha256_file(source),
    }
    rank_review_queue(tmp_path, weights=WEIGHTS, source_paths=(source,))
    assert sha256_file(page) == before["pages"]
    assert sha256_file(tmp_path / "crops.jsonl") == before["crops"]
    assert sha256_file(tmp_path / "predictions.jsonl") == before["predictions"]
    assert sha256_file(source) == before["source"]


def test_no_ocr_inference_is_invoked(tmp_path: Path) -> None:
    import sys

    before = set(sys.modules)
    rank_review_queue(_write_case(tmp_path, "अ", "ब", "क"), weights=WEIGHTS)
    loaded = set(sys.modules) - before
    assert not any(name.startswith("indic_ocr_pipeline.ocr") for name in loaded)
    assert not any(name.split(".")[0] in {"torch", "transformers", "requests"} for name in loaded)


def test_no_network_is_used(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_socket(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network socket opened")

    monkeypatch.setattr(socket, "socket", fail_socket)
    monkeypatch.setattr(socket, "create_connection", fail_socket)
    rank_review_queue(_write_case(tmp_path, "अ", "अ", "अ"), weights=WEIGHTS)


def test_report_does_not_claim_accuracy(tmp_path: Path) -> None:
    summary = rank_review_queue(_write_case(tmp_path, "अ", "", "ब"), weights=WEIGHTS)
    report = format_report(summary)
    assert "accuracy" not in report.lower()
    assert "prediction_pair_cer" in report
    assert "auto_approved_candidates: 0" in report
    assert "gold_records_created: 0" in report


def _score(
    indicocr: str,
    chitrapathak: str,
    tesseract: str,
    *,
    confidence: float | None = 0.8,
) -> object:
    return score_predictions(
        _texts(indicocr, chitrapathak, tesseract),
        tesseract_confidence=confidence,
        line_width=400,
        line_height=30,
        weights=WEIGHTS,
        high_confidence_threshold=0.85,
    )


def _texts(indicocr: str, chitrapathak: str, tesseract: str) -> dict[str, str]:
    return {
        "indicocr": indicocr,
        "chitrapathak": chitrapathak,
        "tesseract": tesseract,
    }


def _write_case(
    directory: Path,
    indicocr: str,
    chitrapathak: str,
    tesseract: str,
    *,
    confidence: float | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    crop_id = "MHGR_000001_P0001_L0001_C_full_01"
    crop = {
        "crop_id": crop_id,
        "document_id": "MHGR_000001",
        "page_id": "MHGR_000001_P0001",
        "parent_line_id": "MHGR_000001_P0001_L0001",
        "region_id": "MHGR_000001_P0001_R0001",
        "image_path": "data/crops/full/MHGR_000001/MHGR_000001_P0001/MHGR_000001_P0001_L0001.png",
    }
    line = {
        "line_id": "MHGR_000001_P0001_L0001",
        "bbox": [0, 0, 400, 30],
    }
    _rewrite(directory / "crops.jsonl", [crop])
    _rewrite(directory / "lines.jsonl", [line])
    texts = _texts(indicocr, chitrapathak, tesseract)
    rows = []
    for name, text in texts.items():
        rows.append(
            {
                "prediction_id": make_prediction_id(crop_id, name),
                "crop_id": crop_id,
                "document_id": "MHGR_000001",
                "page_id": "MHGR_000001_P0001",
                "parent_line_id": "MHGR_000001_P0001_L0001",
                "backend": name,
                "text": text,
                "confidence": confidence if name == "tesseract" else None,
                "status": "complete",
                "error": None,
            }
        )
    _rewrite(directory / "predictions.jsonl", rows)
    return directory


def _replace_prediction_text(directory: Path, backend: str, text: str) -> None:
    rows = open_manifest(directory, "predictions").load()
    for row in rows:
        if row["backend"] == backend:
            row["text"] = text
    _rewrite(directory / "predictions.jsonl", rows)


def _rewrite(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
