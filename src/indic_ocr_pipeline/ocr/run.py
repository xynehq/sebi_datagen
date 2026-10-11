"""Run local OCR backends one after another and write candidate rows.

IndicOCR, then Chitrapathak, then Tesseract. Each backend is loaded, used,
and released before the next one starts. A missing backend does not delete
predictions already stored. Nothing written here is a gold label.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from indic_ocr_pipeline.config import OCRSettings, load_settings
from indic_ocr_pipeline.errors import ManifestConflictError
from indic_ocr_pipeline.ids import make_prediction_id
from indic_ocr_pipeline.manifests import JsonlManifest
from indic_ocr_pipeline.ocr.candidates import build_candidate
from indic_ocr_pipeline.ocr.chitrapathak import ChitrapathakBackend
from indic_ocr_pipeline.ocr.indicocr import IndicOCRBackend
from indic_ocr_pipeline.ocr.tesseract import OCRError, TesseractBackend
from indic_ocr_pipeline.schemas import PredictionRecord, StageStatus
from indic_ocr_pipeline.utils.hashing import sha256_file
from indic_ocr_pipeline.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

BACKEND_ORDER = ("indicocr", "chitrapathak", "tesseract")
_PROGRESS_EVERY = 25


@dataclass
class OCRReport:
    """Counters for one OCR pass."""

    crops: int = 0
    completed: int = 0
    skipped: int = 0
    failed: int = 0
    candidates: int = 0
    candidates_skipped: int = 0
    backend_errors: list[str] = field(default_factory=list)
    crop_errors: list[str] = field(default_factory=list)


def open_backend(
    name: str, settings: OCRSettings
) -> TesseractBackend | IndicOCRBackend | ChitrapathakBackend:
    """Open one local backend. The caller must ``close`` it."""
    if name == "tesseract":
        return TesseractBackend(settings.tesseract)
    if name == "indicocr":
        return IndicOCRBackend(settings.indicocr)
    if name == "chitrapathak":
        return ChitrapathakBackend(settings.chitrapathak)
    extra = _resolved_models().get(name)
    if extra and extra.get("kind") == "transformers":
        from models.transformers_ocr.backend import TransformersOCRBackend

        return TransformersOCRBackend(extra)
    raise OCRError(f"unknown OCR backend: {name}")


def run_ocr(
    crop_rows: Sequence[Mapping[str, object]],
    prediction_manifest: JsonlManifest,
    candidate_manifest: JsonlManifest,
    *,
    repo_root: Path,
    settings: OCRSettings,
    backends: Sequence[str],
    backend_opener: object | None = None,
) -> OCRReport:
    """Transcribe full-line crops, then write a candidate for each finished crop."""
    opener = open_backend if backend_opener is None else backend_opener
    crops = _full_crops(crop_rows)
    report = OCRReport(crops=len(crops))
    configured = os.environ.get("OCR_BACKENDS", "").strip()
    if configured:
        selected = [name for name in configured.split(",") if name]
    else:
        selected = [name for name in backends if name in settings.enabled_backends]
    for name in selected:
        _run_backend(
            name,
            crops,
            prediction_manifest,
            repo_root=repo_root,
            settings=settings,
            opener=opener,
            report=report,
        )
    _write_candidates(
        crops,
        prediction_manifest,
        candidate_manifest,
        backends=tuple(settings.enabled_backends),
        report=report,
    )
    return report


def format_report(report: OCRReport) -> str:
    """Return the human-readable OCR summary."""
    lines = [
        "Phase 6: OCR predictions and candidate manifest",
        "",
        f"Full crops: {report.crops}",
        f"Predictions written: {report.completed}",
        f"Predictions skipped: {report.skipped}",
        f"Crop failures: {report.failed}",
        f"Candidates written: {report.candidates}",
        f"Candidates skipped: {report.candidates_skipped}",
    ]
    for message in report.backend_errors:
        lines.append(f"Backend error: {message}")
    for message in report.crop_errors[:20]:
        lines.append(f"Crop error: {message}")
    if len(report.crop_errors) > 20:
        lines.append(f"Crop error: {len(report.crop_errors) - 20} more")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the configured backends in diagram order, then write candidates."""
    parser = argparse.ArgumentParser(
        description=(
            "Run IndicOCR, Chitrapathak, and Tesseract on full-line crops. "
            "Predictions and candidates are not gold labels."
        )
    )
    parser.add_argument("--backend", choices=BACKEND_ORDER)
    parser.add_argument(
        "--document-id",
        action="append",
        default=None,
        help=(
            "Limit OCR to one or more document IDs (repeatable). "
            "Resume still skips completed rows."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    started = time.monotonic()
    settings = load_settings()
    configured = tuple(name for name in os.environ.get("OCR_BACKENDS", "").split(",") if name)
    names = (args.backend,) if args.backend else configured or BACKEND_ORDER
    crops = _load_crops(settings.manifests_dir / "crops.jsonl")
    if args.document_id:
        allowed = set(args.document_id)
        crops = [row for row in crops if str(row.get("document_id")) in allowed]
        if not crops:
            raise SystemExit(f"no crops for document-id filter: {sorted(allowed)}")
    report = run_ocr(
        crops,
        JsonlManifest(settings.manifests_dir / "predictions.jsonl", "prediction_id"),
        JsonlManifest(settings.manifests_dir / "candidates.jsonl", "candidate_id"),
        repo_root=settings.repo_root,
        settings=settings.ocr,
        backends=names,
    )
    print(format_report(report), flush=True)
    logger.info(
        "ocr finished",
        extra={
            "processed": report.crops * len(names),
            "completed": report.completed,
            "skipped": report.skipped,
            "failed": report.failed + len(report.backend_errors),
            "duration_seconds": round(time.monotonic() - started, 3),
        },
    )
    if report.backend_errors or report.failed:
        return 1
    return 0


def _run_backend(
    name: str,
    crops: Sequence[Mapping[str, object]],
    manifest: JsonlManifest,
    *,
    repo_root: Path,
    settings: OCRSettings,
    opener: object,
    report: OCRReport,
) -> None:
    stored = {row["prediction_id"]: row for row in manifest.load()}
    pending = [crop for crop in crops if _needs_prediction(crop, stored, repo_root, name, report)]
    if not pending:
        return
    try:
        backend = opener(name, settings)  # type: ignore[operator]
    except (OCRError, OSError) as exc:
        report.backend_errors.append(f"{name}: {exc}")
        print(f"{name}: {exc}", file=sys.stderr, flush=True)
        return
    try:
        for index, crop in enumerate(pending, start=1):
            crop_id = str(crop["crop_id"])
            path = _crop_path(crop, repo_root)
            try:
                started = time.monotonic()
                prediction = backend.predict(path)
                elapsed_ms = (time.monotonic() - started) * 1000
                spec = _resolved_models().get(name, {})
                record = PredictionRecord(
                    prediction_id=make_prediction_id(crop_id, name),
                    crop_id=crop_id,
                    document_id=str(crop["document_id"]),
                    page_id=str(crop["page_id"]),
                    parent_line_id=str(crop["parent_line_id"]),
                    backend=prediction.backend,
                    backend_version=prediction.backend_version,
                    text=prediction.text,
                    raw_text=prediction.raw_text,
                    image_sha256=sha256_file(path),
                    confidence=prediction.confidence,
                    latency_ms=(
                        prediction.latency_ms if prediction.latency_ms is not None else elapsed_ms
                    ),
                    metadata={
                        "model_name": name,
                        "revision": spec.get("revision") or prediction.backend_version,
                        "prompt": spec.get("prompt") or "",
                        "model_path": spec.get("path") or "",
                    },
                    status=StageStatus.COMPLETE,
                )
                manifest.append(record.to_dict())
            except (OCRError, OSError, ManifestConflictError) as exc:
                report.failed += 1
                report.crop_errors.append(f"{crop_id} {name}: {exc}")
                continue
            report.completed += 1
            if index % _PROGRESS_EVERY == 0:
                print(f"{name}: {index}/{len(pending)}", flush=True)
    finally:
        backend.close()


def _needs_prediction(
    crop: Mapping[str, object],
    stored: Mapping[str, Mapping[str, object]],
    repo_root: Path,
    backend: str,
    report: OCRReport,
) -> bool:
    crop_id = str(crop["crop_id"])
    prediction_id = make_prediction_id(crop_id, backend)
    path = _crop_path(crop, repo_root)
    if not path.is_file():
        report.failed += 1
        report.crop_errors.append(f"{crop_id}: crop image is missing")
        return False
    digest = sha256_file(path)
    recorded = crop.get("sha256")
    if isinstance(recorded, str) and recorded != digest:
        report.failed += 1
        report.crop_errors.append(f"{crop_id}: crop image does not match the recorded sha256")
        return False
    existing = stored.get(prediction_id)
    if existing is None:
        return True
    if existing.get("status") == "complete" and existing.get("image_sha256") == digest:
        report.skipped += 1
        return False
    report.failed += 1
    report.crop_errors.append(
        f"{prediction_id} already exists and does not match this crop. "
        "The existing prediction was kept."
    )
    return False


def _write_candidates(
    crops: Sequence[Mapping[str, object]],
    prediction_manifest: JsonlManifest,
    candidate_manifest: JsonlManifest,
    *,
    backends: Sequence[str],
    report: OCRReport,
) -> None:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in prediction_manifest.load():
        grouped[str(row.get("crop_id"))].append(row)
    stored = {row["candidate_id"]: row for row in candidate_manifest.load()}
    for crop in crops:
        crop_id = str(crop["crop_id"])
        rows = grouped.get(crop_id, [])
        complete = {
            str(row.get("backend"))
            for row in rows
            if row.get("status") == "complete"
        }
        if any(name not in complete for name in backends):
            continue
        record = build_candidate(crop, rows, backends)
        payload = record.to_dict()
        existing = stored.get(record.candidate_id)
        if existing is not None:
            if existing == payload or _same_transcription(existing, payload):
                report.candidates_skipped += 1
                continue
            report.failed += 1
            report.crop_errors.append(
                f"{record.candidate_id} already exists with a different transcription. "
                "The existing candidate was kept."
            )
            continue
        try:
            candidate_manifest.append(payload)
        except ManifestConflictError as exc:
            report.failed += 1
            report.crop_errors.append(str(exc))
            continue
        report.candidates += 1


def _same_transcription(existing: Mapping[str, object], payload: Mapping[str, object]) -> bool:
    """A stored candidate is reusable when the crop and backend texts are unchanged.

    Score and reason fields can move when the scorer changes. Resume keeps the
    stored row instead of treating that as a failed OCR.
    """
    return (
        existing.get("candidate_id") == payload.get("candidate_id")
        and existing.get("crop_id") == payload.get("crop_id")
        and existing.get("texts_by_backend") == payload.get("texts_by_backend")
    )


def _resolved_models() -> dict[str, dict[str, str]]:
    path = os.environ.get("OCR_RESOLVED", "").strip()
    if not path or not Path(path).is_file():
        return {}
    import json

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return {}
    found = {}
    for item in payload:
        if isinstance(item, dict) and item.get("name"):
            found[str(item["name"])] = item
    return found


def _full_crops(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    full = [dict(row) for row in rows if str(row.get("crop_type")) == "full"]
    full.sort(
        key=lambda row: (
            str(row.get("document_id")),
            int(row.get("page_number", 0)),
            str(row.get("parent_line_id")),
            str(row.get("crop_id")),
        )
    )
    return full


def _crop_path(crop: Mapping[str, object], repo_root: Path) -> Path:
    raw = Path(str(crop.get("image_path", "")))
    if raw.is_absolute():
        return raw
    return repo_root / raw


def _load_crops(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    manifest = JsonlManifest(path, "crop_id")
    return manifest.load()
