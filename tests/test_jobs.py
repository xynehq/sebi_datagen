"""Run options: models, language, datapoint caps, train split, and holdout eval."""

from __future__ import annotations

import pytest

from indic_ocr_pipeline.jobs import (
    JobError,
    assign_document_splits,
    evaluate_holdout,
    parse_language,
    parse_models,
    prepare_train_split,
    require_datapoints,
    select_documents_for_datapoints,
)


def test_models_keep_standard_order_and_reject_unknown() -> None:
    assert parse_models("tesseract,indicocr") == ("indicocr", "tesseract")
    assert parse_models(None) == ("indicocr", "chitrapathak", "tesseract")
    with pytest.raises(JobError, match="unknown model"):
        parse_models("gpt")


def test_only_marathi_is_enabled() -> None:
    assert parse_language("Marathi") == "marathi"
    with pytest.raises(JobError, match="not enabled"):
        parse_language("hindi")
    with pytest.raises(JobError, match="unknown language"):
        parse_language("tamil")


def test_datapoints_keep_whole_documents() -> None:
    counts = {"MHGR_000002": 30, "MHGR_000001": 10, "MHGR_000003": 100}
    assert select_documents_for_datapoints(counts, 25) == ["MHGR_000001"]
    assert select_documents_for_datapoints(counts, 40) == ["MHGR_000001", "MHGR_000002"]
    with pytest.raises(JobError, match="at least 1"):
        require_datapoints(0)


def test_one_document_stays_in_one_split() -> None:
    first = assign_document_splits(["MHGR_000001", "MHGR_000002"], seed=3407, train=0.8, val=0.1)
    second = assign_document_splits(["MHGR_000002", "MHGR_000001"], seed=3407, train=0.8, val=0.1)
    assert first == second
    assert set(first.values()) <= {"train", "val", "test"}


def test_eval_uses_holdout_documents_only() -> None:
    verified = [
        _row("MHGR_000001", "human_verified", "accept_candidate"),
        _row("MHGR_000002", "human_corrected", "correct_text"),
        _row("MHGR_000003", "human_verified", "accept_candidate"),
    ]
    plan = prepare_train_split(verified, seed=1, train=0.0, val=0.0, datapoints=None)
    report = evaluate_holdout(verified, plan)
    assert report["reviewed"] == 3
    assert report["candidate_exact"] == 2
    assert report["human_corrected"] == 1
    assert report["weights_trained"] is False


def _row(document_id: str, status: str, decision: str) -> dict[str, str]:
    return {
        "document_id": document_id,
        "crop_id": f"{document_id}_P0001_L0001_C_full_01",
        "verification_status": status,
        "review_decision": decision,
    }
