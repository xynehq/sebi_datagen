"""OCR writes predictions and candidates. It does not write gold."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from indic_ocr_pipeline.config import load_settings
from indic_ocr_pipeline.ids import make_candidate_id, make_prediction_id
from indic_ocr_pipeline.manifests import JsonlManifest
from indic_ocr_pipeline.ocr.candidates import build_candidate
from indic_ocr_pipeline.ocr.run import run_ocr
from indic_ocr_pipeline.ocr.tesseract import OCRError, _parse_tsv
from indic_ocr_pipeline.schemas import OCRPrediction


def _png(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (40, 16), 255).save(path)
    from indic_ocr_pipeline.utils.hashing import sha256_file

    return sha256_file(path)


def _crop(path: Path, digest: str, *, crop_type: str = "full", index: int = 1) -> dict:
    crop_id = f"MHGR_000001_P0001_L000{index}_C_{crop_type}_01"
    return {
        "crop_id": crop_id,
        "document_id": "MHGR_000001",
        "page_id": "MHGR_000001_P0001",
        "page_number": 1,
        "parent_line_id": f"MHGR_000001_P0001_L000{index}",
        "crop_type": crop_type,
        "image_path": str(path),
        "sha256": digest,
        "status": "complete",
    }


class _Fake:
    def __init__(self, name: str, text: str, *, fail: bool = False) -> None:
        self.name = name
        self.version = "fake-1"
        self._text = text
        self._fail = fail
        self.calls = 0
        self.closed = False

    def predict(self, image_path: Path) -> OCRPrediction:
        del image_path
        self.calls += 1
        if self._fail:
            raise OCRError(f"{self.name} failed")
        return OCRPrediction(
            backend=self.name,
            backend_version=self.version,
            text=self._text,
            raw_text=f" {self._text} ",
        )

    def close(self) -> None:
        self.closed = True


def _opener(backends: dict[str, _Fake]):
    def open_backend(name: str, settings: object) -> _Fake:
        del settings
        backend = backends[name]
        if backend.calls < 0:
            raise OCRError("unavailable")
        return backend

    return open_backend


def test_tsv_joins_words_and_scales_confidence() -> None:
    payload = (
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
        "5\t1\t1\t1\t1\t1\t20\t1\t10\t8\t80\tआदेश\n"
        "5\t1\t1\t1\t1\t2\t4\t1\t10\t8\t40\tनवीन\n"
    )
    text, confidence = _parse_tsv(payload)
    assert text == "नवीन आदेश"
    assert confidence == 0.6


def test_whitespace_disagreement_is_not_stored_as_consensus() -> None:
    crop = _crop(Path("crop.png"), "a" * 64)
    rows = [
        {"backend": "tesseract", "text": "नवीन  आदेश", "status": "complete"},
        {"backend": "indicocr", "text": "नवीन आदेश", "status": "complete"},
        {"backend": "chitrapathak", "text": "नवीन आदेश", "status": "complete"},
    ]
    candidate = build_candidate(crop, rows, ("tesseract", "indicocr", "chitrapathak"))
    assert candidate.consensus_text is None
    assert candidate.disagreement_type is not None
    assert candidate.disagreement_type.value == "whitespace"
    assert candidate.confidence_score is None
    assert candidate.verification_status.value == "candidate"


def test_backends_run_in_order_and_rerun_skips(tmp_path: Path, repo_root: Path) -> None:
    image = tmp_path / "line.png"
    digest = _png(image)
    crop = _crop(image, digest)
    predictions = JsonlManifest(tmp_path / "predictions.jsonl", "prediction_id")
    candidates = JsonlManifest(tmp_path / "candidates.jsonl", "candidate_id")
    opened: list[str] = []
    fakes = {
        "indicocr": _Fake("indicocr", "आदेश"),
        "chitrapathak": _Fake("chitrapathak", "आदेश"),
        "tesseract": _Fake("tesseract", "आदेश"),
    }

    def opener(name: str, settings: object) -> _Fake:
        del settings
        opened.append(name)
        return fakes[name]

    settings = load_settings(repo_root, environ={}).ocr
    first = run_ocr(
        [crop, _crop(image, digest, crop_type="left", index=2)],
        predictions,
        candidates,
        repo_root=repo_root,
        settings=settings,
        backends=("indicocr", "chitrapathak", "tesseract"),
        backend_opener=opener,
    )
    assert opened == ["indicocr", "chitrapathak", "tesseract"]
    assert first.completed == 3
    assert first.candidates == 1
    assert all(fake.closed for fake in fakes.values())
    rows = [json.loads(line) for line in predictions.path.read_text().splitlines()]
    assert [row["backend"] for row in rows] == ["indicocr", "chitrapathak", "tesseract"]
    assert rows[0]["text"] == "आदेश"
    assert rows[0]["raw_text"] == " आदेश "
    candidate = json.loads(candidates.path.read_text().splitlines()[0])
    assert candidate["candidate_id"] == make_candidate_id(crop["crop_id"])
    assert candidate["consensus_text"] == "आदेश"
    assert candidate["confidence_score"] is None
    assert candidate["verification_status"] == "candidate"
    opened.clear()
    second = run_ocr(
        [crop],
        predictions,
        candidates,
        repo_root=repo_root,
        settings=settings,
        backends=("indicocr", "chitrapathak", "tesseract"),
        backend_opener=opener,
    )
    assert opened == []
    assert second.skipped == 3
    assert second.candidates_skipped == 1


def test_one_backend_failure_keeps_the_other_predictions(tmp_path: Path, repo_root: Path) -> None:
    image = tmp_path / "line.png"
    digest = _png(image)
    crop = _crop(image, digest)
    predictions = JsonlManifest(tmp_path / "predictions.jsonl", "prediction_id")
    candidates = JsonlManifest(tmp_path / "candidates.jsonl", "candidate_id")
    fakes = {
        "indicocr": _Fake("indicocr", "एक"),
        "chitrapathak": _Fake("chitrapathak", "दोन", fail=True),
        "tesseract": _Fake("tesseract", "एक"),
    }
    settings = load_settings(repo_root, environ={}).ocr
    report = run_ocr(
        [crop],
        predictions,
        candidates,
        repo_root=repo_root,
        settings=settings,
        backends=("indicocr", "chitrapathak", "tesseract"),
        backend_opener=lambda name, _settings: fakes[name],
    )
    assert report.completed == 2
    assert report.failed == 1
    assert report.candidates == 0
    stored = {row["backend"] for row in predictions.load()}
    assert stored == {"indicocr", "tesseract"}
    assert fakes["chitrapathak"].closed


def test_missing_backend_does_not_write_a_null_candidate(tmp_path: Path, repo_root: Path) -> None:
    image = tmp_path / "line.png"
    digest = _png(image)
    crop = _crop(image, digest)
    predictions = JsonlManifest(tmp_path / "predictions.jsonl", "prediction_id")
    candidates = JsonlManifest(tmp_path / "candidates.jsonl", "candidate_id")

    def opener(name: str, settings: object) -> _Fake:
        del settings
        if name == "indicocr":
            raise OCRError("model_path is empty")
        return _Fake(name, "आदेश")

    settings = load_settings(repo_root, environ={}).ocr
    report = run_ocr(
        [crop],
        predictions,
        candidates,
        repo_root=repo_root,
        settings=settings,
        backends=("indicocr", "chitrapathak", "tesseract"),
        backend_opener=opener,
    )
    assert report.backend_errors
    assert report.candidates == 0
    assert not candidates.path.exists() or candidates.path.read_text() == ""
    assert make_prediction_id(crop["crop_id"], "indicocr") not in {
        row["prediction_id"] for row in predictions.load()
    }


def test_changed_crop_hash_does_not_overwrite_a_prediction(tmp_path: Path, repo_root: Path) -> None:
    image = tmp_path / "line.png"
    digest = _png(image)
    crop = _crop(image, digest)
    predictions = JsonlManifest(tmp_path / "predictions.jsonl", "prediction_id")
    candidates = JsonlManifest(tmp_path / "candidates.jsonl", "candidate_id")
    prediction_id = make_prediction_id(crop["crop_id"], "tesseract")
    predictions.append(
        {
            "prediction_id": prediction_id,
            "crop_id": crop["crop_id"],
            "document_id": "MHGR_000001",
            "page_id": "MHGR_000001_P0001",
            "parent_line_id": "MHGR_000001_P0001_L0001",
            "backend": "tesseract",
            "backend_version": "old",
            "text": "जुने",
            "raw_text": "जुने",
            "image_sha256": "b" * 64,
            "confidence": None,
            "token_confidences": None,
            "latency_ms": None,
            "error": None,
            "metadata": {},
            "status": "complete",
        }
    )
    settings = load_settings(repo_root, environ={}).ocr
    report = run_ocr(
        [crop],
        predictions,
        candidates,
        repo_root=repo_root,
        settings=settings,
        backends=("tesseract",),
        backend_opener=lambda name, _settings: _Fake(name, "नवीन"),
    )
    assert report.failed == 1
    assert report.completed == 0
    stored = predictions.load()[0]
    assert stored["text"] == "जुने"
