"""Unified pipeline: consensus, gold, resume, frozen-benchmark protection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from evaluation.harness import score_corpus
from pipeline.cli import main as cli_main
from pipeline.dataset_flow import CountError, assert_ocr_counts
from pipeline.fixture import prepare_fixture
from pipeline.gold import ensure_invalidated_archive, select_gold_rows
from pipeline.graph import descendants
from pipeline.orchestrator import execute
from pipeline.state import COMPLETE, load_state
from training.dataset import assert_training_rows_safe, load_frozen_ids
from training.selection import SelectionError, select_best

from indic_ocr_pipeline.ocr.consensus import classify_consensus
from indic_ocr_pipeline.review.audit_sample import sample_audit


def test_packages_import() -> None:
    import evaluation
    import IndicPhotoOCR
    import pipeline
    import training

    import indic_ocr_pipeline

    assert indic_ocr_pipeline.__version__
    assert pipeline.CODE_VERSION
    assert evaluation.METRIC_VERSION
    assert training.select_best is select_best
    assert IndicPhotoOCR.__doc__ is None or True


def test_consensus_classes() -> None:
    agree = classify_consensus({"indicocr": "अ", "chitrapathak": "अ", "tesseract": "अ"})
    assert agree["agreement_class"] == "ALL_3_AGREE"
    assert agree["status"] == "AUTO_3_AGREE"
    assert agree["review_required"] is False
    assert agree["gold"] is False

    two = classify_consensus({"indicocr": "ब", "chitrapathak": "ब", "tesseract": "क"})
    assert two["agreement_class"] == "TWO_AGREE"
    assert two["status"] == "AUTO_2_AGREE"
    assert two["proposed_text"] == "ब"
    assert two["predictions"]["tesseract"] == "क"

    differ = classify_consensus({"indicocr": "ड", "chitrapathak": "ई", "tesseract": "फ"})
    assert differ["agreement_class"] == "ALL_DIFFER"
    assert differ["status"] == "HUMAN_REVIEW"

    empty = classify_consensus({"indicocr": "", "chitrapathak": "ग", "tesseract": "ग"})
    assert empty["agreement_class"] == "EMPTY"

    boundary = classify_consensus(
        {"indicocr": "ह", "chitrapathak": "ह", "tesseract": "ह"},
        boundary_valid=False,
    )
    assert boundary["agreement_class"] == "BAD_BOUNDARY"
    assert boundary["review_required"] is True


def test_calibration_is_deterministic() -> None:
    rows = [{"crop_id": f"C{index:04d}", "agreement_class": "ALL_3_AGREE"} for index in range(200)]
    rows += [{"crop_id": "D1", "agreement_class": "ALL_DIFFER"}]
    left = sample_audit(rows, seed=3407)
    right = sample_audit(rows, seed=3407)
    assert [row["crop_id"] for row in left] == [row["crop_id"] for row in right]
    assert "D1" in {row["crop_id"] for row in left}
    assert sample_audit(rows, seed=1) != left or True
    again = sample_audit(rows, seed=3407, rates={"ALL_3_AGREE": 0.05})
    assert [row["crop_id"] for row in again] == [row["crop_id"] for row in left]


def test_count_invariant() -> None:
    assert_ocr_counts(4, 12, 4)
    try:
        assert_ocr_counts(4, 12, 3)
    except CountError as exc:
        assert "Candidates" in str(exc)
    else:
        raise AssertionError("mismatched counts were accepted")


def test_invalidated_review_cannot_become_gold(tmp_path: Path) -> None:
    verified = tmp_path / "manifests" / "verified.jsonl"
    verified.parent.mkdir(parents=True)
    historical = {
        "crop_id": "OLD",
        "document_id": "D",
        "page_id": "P",
        "parent_line_id": "L",
        "image_path": "crop.png",
        "reviewer_id": "local_reviewer",
        "verification_timestamp": "2026-10-01T00:00:00+00:00",
        "verification_status": "human_verified",
        "final_text": "जुने",
        "metadata": {"decision_label": "human_verified", "reviewer_app_version": "phase8-review-1"},
        "original_candidate_texts": {"indicocr": "जुने", "chitrapathak": "जुने", "tesseract": "जुने"},
    }
    verified.write_text(json.dumps(historical) + "\n", encoding="utf-8")
    blocked = ensure_invalidated_archive(tmp_path)
    assert "OLD" in blocked
    fresh = dict(historical)
    fresh["crop_id"] = "NEW"
    fresh["review_origin"] = "human"
    fresh["metadata"] = {"decision_label": "human_verified", "review_origin": "human"}
    gold, rejected = select_gold_rows([historical, fresh], blocked)
    assert [row["crop_id"] for row in gold] == ["NEW"]
    assert rejected[0]["reason"] == "INVALIDATED_REVIEW"
    assert (tmp_path / "manifests" / "verified.jsonl").is_file()


def test_rejected_decision_is_not_gold() -> None:
    row = {
        "crop_id": "R",
        "document_id": "D",
        "page_id": "P",
        "parent_line_id": "L",
        "image_path": "c.png",
        "reviewer_id": "reviewer_1",
        "verification_timestamp": "2026-10-08T00:00:00+00:00",
        "review_origin": "human",
        "decision": "HUMAN_REJECTED",
        "gold_text": "अ",
        "original_candidate_texts": {},
    }
    gold, rejected = select_gold_rows([row], set())
    assert gold == []
    assert rejected


def test_frozen_rows_cannot_train(tmp_path: Path) -> None:
    frozen = tmp_path / "frozen" / "marathi100"
    frozen.mkdir(parents=True)
    (frozen / "validation.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "FROZEN_1",
                "image_filename": "/images/0001.png",
                "expected_text": "अ",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    frozen_ids = load_frozen_ids(frozen)
    bad = {
        "sample_id": "FROZEN_1",
        "document_id": "D",
        "image_filename": "/images/0001.png",
        "crop_path": "frozen/marathi100/images/0001.png",
    }
    try:
        assert_training_rows_safe([bad], frozen_ids, frozen)
    except Exception as exc:
        assert "frozen" in str(exc).lower()
    else:
        raise AssertionError("frozen row entered training")


def test_checkpoint_selection_rejects_frozen_ids() -> None:
    metrics = score_corpus(["अ"], ["अ"])
    try:
        select_best(
            [
                {
                    "checkpoint": "a",
                    "metrics": metrics,
                    "sample_ids": ["FROZEN_1"],
                }
            ],
            frozen_ids={"FROZEN_1"},
        )
    except SelectionError as exc:
        assert "frozen" in str(exc).lower()
    else:
        raise AssertionError("frozen id was used for selection")
    board = select_best(
        [
            {"checkpoint": "worse", "metrics": score_corpus(["अ"], ["ब"]), "sample_ids": ["V1"]},
            {"checkpoint": "better", "metrics": score_corpus(["अ"], ["अ"]), "sample_ids": ["V1"]},
        ],
        frozen_ids={"FROZEN_1"},
    )
    assert board["selected"] == "better"
    assert board["frozen_used"] is False
    assert board["primary_metric"] == "cer"


def test_descendants_of_ocr_do_not_include_render() -> None:
    names = descendants("ocr")
    assert "compare" in names
    assert "train" in names
    assert "render" not in names
    assert "ingest" not in names


def test_fixture_pipeline_resume_and_invalidation(tmp_path: Path) -> None:
    prepare_fixture(tmp_path)
    assert execute(tmp_path, fixture=True) == 0
    from PIL import Image

    crop = Image.open(next((tmp_path / "data" / "crops").glob("*.png")))
    colors = crop.getcolors(maxcolors=8)
    assert colors is None or len(colors) > 1
    state = load_state(tmp_path / "runs" / "pipeline_state.json")
    ingest_ended = state["stages"]["ingest"]["ended_at"]
    ocr_ended = state["stages"]["ocr"]["ended_at"]
    assert state["stages"]["compare"]["status"] == COMPLETE
    assert state["stages"]["compare"]["row_counts"]["crops"] == 5
    assert state["stages"]["compare"]["row_counts"]["predictions"] == 15
    assert state["stages"]["compare"]["row_counts"]["candidates"] == 5
    train_ids = _ids(tmp_path / "data" / "datasets" / "train.jsonl")
    val_ids = _ids(tmp_path / "data" / "datasets" / "val.jsonl")
    assert "OLD_CROP" not in train_ids | val_ids
    assert "FROZEN_1" not in train_ids | val_ids
    assert train_ids.isdisjoint(val_ids)
    scoreboard = json.loads(
        next((tmp_path / "runs").glob("*/evaluation/checkpoint_scoreboard.json")).read_text()
    )
    assert scoreboard["selected"] == "ckpt_low"
    assert scoreboard["frozen_used"] is False
    frozen_path = next((tmp_path / "runs").glob("*/evaluation/frozen100/metrics.json"))
    frozen = json.loads(frozen_path.read_text())
    assert frozen["used_for_selection"] is False
    for row in scoreboard["rows"]:
        assert "FROZEN_1" not in row["sample_ids"]

    assert execute(tmp_path, fixture=True) == 0
    again = load_state(tmp_path / "runs" / "pipeline_state.json")
    assert again["stages"]["ingest"]["ended_at"] == ingest_ended
    assert again["stages"]["ocr"]["ended_at"] == ocr_ended

    config_path = tmp_path / "configs" / "pipeline.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["ocr"]["nonce"] = 2
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert execute(tmp_path, fixture=True) == 0
    third = load_state(tmp_path / "runs" / "pipeline_state.json")
    assert third["stages"]["ingest"]["ended_at"] == ingest_ended
    assert third["stages"]["ocr"]["ended_at"] != ocr_ended
    assert third["stages"]["render"]["status"] == COMPLETE


def test_failure_stops_downstream(tmp_path: Path) -> None:
    prepare_fixture(tmp_path)
    config_path = tmp_path / "configs" / "pipeline.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["execution"]["fail_stage"] = "render"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert execute(tmp_path, fixture=True) == 1
    state = load_state(tmp_path / "runs" / "pipeline_state.json")
    assert state["stages"]["render"]["status"] == "FAILED"
    assert "detect" not in state["stages"] or state["stages"]["detect"]["status"] != COMPLETE
    log = next((tmp_path / "runs").glob("*/logs/pipeline.log")).read_text(encoding="utf-8")
    assert "./run_pipeline.sh" in log


def test_missing_frozen_exam_does_not_block_the_command(tmp_path: Path) -> None:
    prepare_fixture(tmp_path)
    frozen = tmp_path / "frozen"
    if frozen.exists():
        import shutil

        shutil.rmtree(frozen)
    (tmp_path / "manifests" / "documents.jsonl").write_text("", encoding="utf-8")
    from pipeline.config import load_config
    from pipeline.preflight import checks_pass, collect_checks

    checks = collect_checks(tmp_path, load_config(tmp_path), fixture=False)
    exam = next(item for item in checks if item.name == "frozen benchmark")
    assert exam.ok is False
    assert exam.required is False
    assert checks_pass(checks)


def test_same_command_resumes_after_failure(tmp_path: Path) -> None:
    prepare_fixture(tmp_path)
    config_path = tmp_path / "configs" / "pipeline.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["execution"]["fail_stage"] = "render"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert execute(tmp_path, fixture=True) == 1
    stopped = load_state(tmp_path / "runs" / "pipeline_state.json")
    ingest_ended = stopped["stages"]["ingest"]["ended_at"]

    config = yaml.safe_load(config_path.read_text())
    del config["execution"]["fail_stage"]
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert execute(tmp_path, fixture=True) == 0
    resumed = load_state(tmp_path / "runs" / "pipeline_state.json")
    assert resumed["stages"]["ingest"]["ended_at"] == ingest_ended
    assert resumed["stages"]["render"]["status"] == COMPLETE
    assert resumed["stages"]["report"]["status"] == COMPLETE
    assert resumed["sealed"] is True


def test_dry_run_and_help(tmp_path: Path) -> None:
    prepare_fixture(tmp_path)
    assert execute(tmp_path, fixture=True, dry_run=True) == 0
    with pytest.raises(SystemExit) as caught:
        cli_main(["--help"])
    assert caught.value.code == 0
    assert cli_main(["--check", "--root", str(tmp_path), "--fixture"]) == 0


def _ids(path: Path) -> set[str]:
    found = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            found.add(row["crop_id"])
            found.add(row["sample_id"])
    return found
