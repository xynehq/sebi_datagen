"""Build the review queue from existing OCR predictions.

This stage does not run OCR, does not read crop pixels, and does not write
gold or human-verified rows. Every queued row stays ``candidate``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from indic_ocr_pipeline.config import load_settings
from indic_ocr_pipeline.errors import DatasetError, RankingError
from indic_ocr_pipeline.ids import make_candidate_id, make_review_id
from indic_ocr_pipeline.manifests import open_manifest
from indic_ocr_pipeline.schemas import (
    SCORE_IS_NOT_A_PROBABILITY,
    CandidateRecord,
    QualityTier,
    ReviewRecord,
    StageStatus,
    VerificationStatus,
)
from indic_ocr_pipeline.scoring.metrics import (
    BACKENDS,
    TESSERACT_CONFIDENCE_SCALE,
    disagreement_label,
)
from indic_ocr_pipeline.scoring.rank import WEIGHT_KEYS, ReviewScore, score_predictions
from indic_ocr_pipeline.utils.hashing import sha256_file
from indic_ocr_pipeline.utils.jsonutil import dumps_record
from indic_ocr_pipeline.utils.logging import configure_logging, get_logger

logger = get_logger("scoring.queue")

UNCHANGED_MANIFESTS = ("documents", "pages", "regions", "lines", "crops", "predictions")


def rank_review_queue(
    manifests_dir: Path,
    *,
    reports_dir: Path | None = None,
    weights: Mapping[str, float] | None = None,
    high_confidence_threshold: float = 0.85,
    source_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    """Score every full-line crop and write the review queue.

    A second call with the same predictions skips identical rows. A changed
    prediction payload is a conflict and is not overwritten.
    """
    started = time.monotonic()
    manifests_dir = Path(manifests_dir)
    before = _hashes(manifests_dir, source_paths)
    crops = open_manifest(manifests_dir, "crops").load()
    lines = open_manifest(manifests_dir, "lines").load()
    predictions = open_manifest(manifests_dir, "predictions").load()
    stored_candidates = open_manifest(manifests_dir, "candidates").load()
    crops_by_id = _index(crops, "crop_id")
    lines_by_id = _index(lines, "line_id")
    candidates_by_id = _index(stored_candidates, "candidate_id")
    grouped = _group_predictions(predictions, crops_by_id)
    _require_complete(crops_by_id, grouped)
    weight_map = dict(weights) if weights is not None else _default_weights()
    scored: list[tuple[str, ReviewScore, dict[str, Any]]] = []
    for crop_id in sorted(crops_by_id):
        crop = crops_by_id[crop_id]
        rows = grouped[crop_id]
        texts = {name: str(rows[name]["text"]) for name in BACKENDS}
        _reject_text_drift(crop_id, texts, candidates_by_id.get(make_candidate_id(crop_id)))
        line = lines_by_id.get(str(crop["parent_line_id"]))
        if line is None:
            raise RankingError(f"crop {crop_id} has no line record {crop['parent_line_id']}")
        width, height = _line_size(line)
        confidence = rows["tesseract"].get("confidence")
        if confidence is not None and not isinstance(confidence, int | float):
            raise RankingError(f"tesseract confidence for {crop_id} is not a number or null")
        result = score_predictions(
            texts,
            tesseract_confidence=None if confidence is None else float(confidence),
            line_width=width,
            line_height=height,
            weights=weight_map,
            high_confidence_threshold=high_confidence_threshold,
        )
        scored.append((crop_id, result, crop))
    scored.sort(key=lambda item: item[1].sort_key(item[0]))
    candidate_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    for rank, (crop_id, result, crop) in enumerate(scored, start=1):
        texts = {name: str(grouped[crop_id][name]["text"]) for name in BACKENDS}
        candidate_rows.append(_stable(_candidate_payload(crop, texts, result)))
        review_rows.append(
            _stable(
                _review_payload(
                    crop,
                    texts,
                    result,
                    rank,
                    grouped[crop_id],
                    weight_map,
                )
            )
        )
    candidate_counts = open_manifest(manifests_dir, "candidates").commit_batch(
        candidate_rows,
        fill_unscored=True,
    )
    # Human decisions freeze review rows as complete. Existing pending queue
    # rows also keep their historical priority_rank. Only append missing crops.
    existing_review_ids = {
        str(row["review_id"])
        for row in open_manifest(manifests_dir, "review").load()
        if row.get("review_id")
    }
    review_to_write = [
        row for row in review_rows if str(row["review_id"]) not in existing_review_ids
    ]
    review_counts = open_manifest(manifests_dir, "review").commit_batch(review_to_write)
    review_counts = {
        **review_counts,
        "skipped": review_counts["skipped"] + len(existing_review_ids),
    }
    after = _hashes(manifests_dir, source_paths)
    if before != after:
        raise RankingError("ranking changed a source manifest or source file")
    summary = _summary(
        crops_by_id,
        grouped,
        scored,
        weight_map,
        candidate_counts,
        review_counts,
    )
    report = format_report(summary)
    if reports_dir is not None:
        reports_dir.mkdir(parents=True, exist_ok=True)
        (Path(reports_dir) / "ocr_comparison.txt").write_text(report, encoding="utf-8")
    elapsed = time.monotonic() - started
    logger.info(
        "review ranking finished",
        extra={
            "processed": summary["total_crops"],
            "completed": summary["candidate_records"],
            "skipped": candidate_counts["skipped"],
            "failed": 0,
            "duration_seconds": round(elapsed, 3),
        },
    )
    return summary


def format_report(summary: Mapping[str, Any]) -> str:
    """Return the deterministic comparison report. It does not claim accuracy."""
    lines = [
        "Phase 7: OCR comparison and review ranking",
        SCORE_IS_NOT_A_PROBABILITY,
        "review_risk_score ranks human review. It is not a probability of being wrong.",
        "Three-way agreement is evidence, not a correct transcription and not gold.",
        f"tesseract_confidence_scale: {TESSERACT_CONFIDENCE_SCALE}",
        "auto_approved_candidate was not applied. Every row stays candidate.",
        "",
        f"total_crops: {summary['total_crops']}",
    ]
    for name in BACKENDS:
        backend = summary["backends"][name]
        lines.append(
            f"backend.{name}: predictions={backend['predictions']} "
            f"successful={backend['successful']} empty={backend['empty']}"
        )
    lines.extend(
        [
            f"all_3_agree: {summary['all_3_agree']} ({summary['all_3_agree_percent']}%)",
            f"two_agree_one_differs: {summary['two_agree_one_differs']}",
            f"all_3_differ: {summary['all_3_differ']}",
            f"empty_present: {summary['empty_present']}",
            f"prediction_error: {summary['prediction_error']}",
            "",
            "pairwise prediction_pair_cer is OCR-versus-OCR, not ground_truth_cer.",
        ]
    )
    for key, stats in summary["pairwise"].items():
        lines.append(
            f"pair.{key}: exact_agreement_rate={stats['exact_agreement_rate']} "
            f"normalized_agreement_rate={stats['normalized_agreement_rate']} "
            f"mean_edit_distance={stats['mean_edit_distance']} "
            f"median_edit_distance={stats['median_edit_distance']} "
            f"max_edit_distance={stats['max_edit_distance']}"
        )
    lines.extend(
        [
            "",
            "weights sum to 1. Feature direction: higher value means higher review risk.",
            "agreement: more disagreement or a larger prediction_pair_cer raises risk. "
            "All-three agreement stays at 0.15 so a shared mistake is not treated as safe.",
            "tesseract_confidence: lower stored confidence raises risk. Null is 0.5. "
            "This weight is 0.15 and cannot cancel disagreement.",
            "image_quality: an empty prediction raises risk. Pixels are not loaded.",
            "crop_geometry: line height <= 12px or width <= 40px, or height >= 80px, "
            "raises risk. Sizes come from the line manifest.",
            "unicode_sanity: U+FFFD, $$, C0 controls, or private-use characters raise risk.",
            "empty predictions are forced to at least 0.95, and all-empty rows are 1.0.",
            f"weights: {summary['weights']}",
            f"risk_min: {summary['risk_min']}",
            f"risk_median: {summary['risk_median']}",
            f"risk_max: {summary['risk_max']}",
            f"priority.P0: {summary['priority']['P0']}",
            f"priority.P1: {summary['priority']['P1']}",
            f"priority.P2: {summary['priority']['P2']}",
            f"priority.P3: {summary['priority']['P3']}",
            f"high_risk: {summary['high_risk']}",
            f"medium_risk: {summary['medium_risk']}",
            f"low_risk: {summary['low_risk']}",
            f"candidate_records: {summary['candidate_records']}",
            f"auto_approved_candidates: {summary['auto_approved_candidates']}",
            f"gold_records_created: {summary['gold_records_created']}",
            f"human_verified_created: {summary['human_verified_created']}",
            "top_risk_crop_ids:",
        ]
    )
    lines.extend(f"  {crop_id}" for crop_id in summary["top_risk_crop_ids"])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Rank the repository's existing predictions. OCR is not invoked."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare OCR predictions and write the review queue. "
            "Scores rank review. They are not gold and not a probability."
        )
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)
    settings = load_settings()
    sources = _document_sources(settings.repo_root, settings.manifests_dir)
    try:
        summary = rank_review_queue(
            settings.manifests_dir,
            reports_dir=settings.reports_dir,
            weights=settings.scoring.weights,
            high_confidence_threshold=settings.scoring.high_confidence_threshold,
            source_paths=sources,
        )
    except DatasetError as exc:
        logger.error(str(exc))
        print(str(exc), file=sys.stderr)
        return 1
    print(format_report(summary), flush=True)
    return 0


def _stable(record: dict[str, Any]) -> dict[str, Any]:
    """Match the JSON manifest encoding so a rerun compares equal."""
    return json.loads(dumps_record(record))


def _default_weights() -> dict[str, float]:
    return dict(load_settings().scoring.weights)


def _document_sources(repo_root: Path, manifests_dir: Path) -> list[Path]:
    paths: list[Path] = []
    manifest = manifests_dir / "documents.jsonl"
    if not manifest.exists():
        return paths
    for row in open_manifest(manifests_dir, "documents").load():
        source = row.get("source_path")
        if isinstance(source, str) and source:
            path = Path(source)
            paths.append(path if path.is_absolute() else repo_root / path)
    return paths


def _hashes(manifests_dir: Path, source_paths: Sequence[Path]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in UNCHANGED_MANIFESTS:
        path = manifests_dir / f"{name}.jsonl"
        hashes[name] = sha256_file(path) if path.exists() else ""
    for path in source_paths:
        hashes[str(path)] = sha256_file(path) if path.exists() else ""
    return hashes


def _index(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row.get(field)
        if not isinstance(key, str) or not key:
            raise RankingError(f"a {field} row has no string id")
        if key in indexed:
            raise RankingError(f"duplicate {field}={key}")
        indexed[key] = row
    return indexed


def _group_predictions(
    predictions: list[dict[str, Any]],
    crops_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in predictions:
        crop_id = row.get("crop_id")
        backend = row.get("backend")
        if not isinstance(crop_id, str) or not isinstance(backend, str):
            raise RankingError("a prediction is missing crop_id or backend")
        if crop_id not in crops_by_id:
            raise RankingError(
                f"prediction {row.get('prediction_id')} does not match a crop: {crop_id}"
            )
        if backend not in BACKENDS:
            raise RankingError(f"unexpected backend {backend} on crop {crop_id}")
        if backend in grouped[crop_id]:
            raise RankingError(f"duplicate {backend} prediction for crop {crop_id}")
        crop = crops_by_id[crop_id]
        for field in ("document_id", "page_id", "parent_line_id"):
            if row.get(field) != crop.get(field):
                raise RankingError(
                    f"prediction {row.get('prediction_id')} {field} does not match its crop"
                )
        complete = row.get("status") == "complete" and not row.get("error")
        if not complete or not isinstance(row.get("text"), str):
            prediction_id = row.get("prediction_id")
            raise RankingError(f"prediction {prediction_id} is not a complete transcription")
        grouped[crop_id][backend] = row
    return grouped


def _require_complete(
    crops_by_id: Mapping[str, Mapping[str, Any]],
    grouped: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> None:
    for crop_id in sorted(crops_by_id):
        present = grouped.get(crop_id, {})
        missing = [name for name in BACKENDS if name not in present]
        if missing:
            raise RankingError(f"crop {crop_id} is missing predictions: {', '.join(missing)}")


def _reject_text_drift(
    crop_id: str,
    texts: Mapping[str, str],
    candidate: Mapping[str, Any] | None,
) -> None:
    if candidate is None:
        return
    stored = candidate.get("texts_by_backend")
    if not isinstance(stored, dict):
        raise RankingError(f"candidate for {crop_id} has no texts_by_backend")
    for name in BACKENDS:
        if stored.get(name) != texts[name]:
            raise RankingError(
                f"prediction text for {crop_id} {name} changed after the candidate was written; "
                "refusing to overwrite it"
            )


def _line_size(line: Mapping[str, Any]) -> tuple[int, int]:
    bbox = line.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise RankingError(f"line {line.get('line_id')} has no bbox")
    width = int(bbox[2]) - int(bbox[0])
    height = int(bbox[3]) - int(bbox[1])
    return width, height


def _candidate_payload(
    crop: Mapping[str, Any],
    texts: Mapping[str, str],
    result: ReviewScore,
) -> dict[str, Any]:
    ordered = [texts[name] for name in BACKENDS]
    counts: dict[str, int] = {}
    for text in ordered:
        counts[text] = counts.get(text, 0) + 1
    record = CandidateRecord(
        candidate_id=make_candidate_id(str(crop["crop_id"])),
        crop_id=str(crop["crop_id"]),
        document_id=str(crop["document_id"]),
        page_id=str(crop["page_id"]),
        parent_line_id=str(crop["parent_line_id"]),
        texts_by_backend={name: texts[name] for name in BACKENDS},
        exact_agreement_count=max(counts.values()),
        backends_present=len(BACKENDS),
        pairwise_agreement={key: pair.exact for key, pair in result.pairs.items()},
        normalized_agreement={key: pair.normalized for key, pair in result.pairs.items()},
        consensus_text=result.proposed_consensus,
        disagreement_type=disagreement_label(ordered),
        confidence_score=result.confidence_score,
        risk_score=result.review_risk_score,
        reasons=result.reasons,
        verification_status=VerificationStatus.CANDIDATE,
        tier=QualityTier.TIER_2_CANDIDATE,
    )
    if record.verification_status is not VerificationStatus.CANDIDATE:
        raise RankingError("ranking must not leave the candidate status")
    return record.to_dict()


def _review_payload(
    crop: Mapping[str, Any],
    texts: Mapping[str, str],
    result: ReviewScore,
    priority_rank: int,
    predictions: Mapping[str, Mapping[str, Any]],
    weights: Mapping[str, float],
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "score_meaning": "review_risk_score_ranking_not_probability",
        "agreement_category": result.agreement_category,
        "proposed_consensus": result.proposed_consensus,
        "review_priority": result.review_priority,
        "review_risk_score": result.review_risk_score,
        "image_path": crop.get("image_path"),
        "region_id": crop.get("region_id"),
        "line_id": crop.get("parent_line_id"),
        "indicocr_text": texts["indicocr"],
        "chitrapathak_text": texts["chitrapathak"],
        "tesseract_text": texts["tesseract"],
        "indicocr_confidence": predictions["indicocr"].get("confidence"),
        "chitrapathak_confidence": predictions["chitrapathak"].get("confidence"),
        "tesseract_confidence": result.tesseract_confidence,
        "tesseract_confidence_scale": TESSERACT_CONFIDENCE_SCALE,
        "flags": result.flags,
        "pairwise": {key: pair.to_dict() for key, pair in result.pairs.items()},
        "lengths": result.lengths,
        "feature_risks": result.feature_risks,
        "weights": {key: weights[key] for key in WEIGHT_KEYS},
        "verification_status": VerificationStatus.CANDIDATE.value,
    }
    record = ReviewRecord(
        review_id=make_review_id(str(crop["crop_id"])),
        crop_id=str(crop["crop_id"]),
        candidate_id=make_candidate_id(str(crop["crop_id"])),
        document_id=str(crop["document_id"]),
        page_id=str(crop["page_id"]),
        parent_line_id=str(crop["parent_line_id"]),
        priority_rank=priority_rank,
        risk_score=result.review_risk_score,
        audit_sample=False,
        diagnostics=diagnostics,
        status=StageStatus.PENDING,
    )
    if record.decision is not None or record.final_text is not None:
        raise RankingError("ranking must not record a human decision")
    return record.to_dict()


def _summary(
    crops_by_id: Mapping[str, Mapping[str, Any]],
    grouped: Mapping[str, Mapping[str, Mapping[str, Any]]],
    scored: list[tuple[str, ReviewScore, dict[str, Any]]],
    weights: Mapping[str, float],
    candidate_counts: Mapping[str, int],
    review_counts: Mapping[str, int],
) -> dict[str, Any]:
    del crops_by_id, candidate_counts, review_counts
    categories = {name: 0 for name in (
        "ALL_3_AGREE",
        "TWO_AGREE_ONE_DIFFERS",
        "ALL_3_DIFFER",
        "ONE_OR_MORE_EMPTY",
        "PREDICTION_ERROR",
    )}
    priority = {"P0": 0, "P1": 0, "P2": 0, "P3": 0}
    risks: list[float] = []
    distances: dict[str, list[int]] = defaultdict(list)
    exact_hits: dict[str, int] = defaultdict(int)
    normalized_hits: dict[str, int] = defaultdict(int)
    for _crop_id, result, _crop in scored:
        categories[result.agreement_category] += 1
        priority[result.review_priority] += 1
        risks.append(result.review_risk_score)
        for key, pair in result.pairs.items():
            distances[key].append(pair.edit_distance)
            exact_hits[key] += int(pair.exact)
            normalized_hits[key] += int(pair.normalized)
    total = len(scored)
    backends: dict[str, dict[str, int]] = {}
    for name in BACKENDS:
        rows = [grouped[crop_id][name] for crop_id, _result, _crop in scored]
        backends[name] = {
            "predictions": len(rows),
            "successful": sum(1 for row in rows if row.get("status") == "complete"),
            "empty": sum(1 for row in rows if row.get("text") == ""),
        }
    pairwise: dict[str, dict[str, float]] = {}
    for key, values in distances.items():
        pairwise[key] = {
            "exact_agreement_rate": _rate(exact_hits[key], total),
            "normalized_agreement_rate": _rate(normalized_hits[key], total),
            "mean_edit_distance": round(sum(values) / len(values), 4) if values else 0.0,
            "median_edit_distance": _median(values),
            "max_edit_distance": max(values) if values else 0,
        }
    ordered_risks = sorted(risks)
    return {
        "total_crops": total,
        "backends": backends,
        "all_3_agree": categories["ALL_3_AGREE"],
        "all_3_agree_percent": _rate(categories["ALL_3_AGREE"], total),
        "two_agree_one_differs": categories["TWO_AGREE_ONE_DIFFERS"],
        "all_3_differ": categories["ALL_3_DIFFER"],
        "empty_present": categories["ONE_OR_MORE_EMPTY"],
        "prediction_error": categories["PREDICTION_ERROR"],
        "pairwise": pairwise,
        "weights": {key: weights[key] for key in WEIGHT_KEYS},
        "risk_min": ordered_risks[0] if ordered_risks else 0.0,
        "risk_median": _median_float(ordered_risks),
        "risk_max": ordered_risks[-1] if ordered_risks else 0.0,
        "priority": priority,
        "high_risk": priority["P0"] + priority["P1"],
        "medium_risk": priority["P2"],
        "low_risk": priority["P3"],
        "candidate_records": total,
        "auto_approved_candidates": 0,
        "gold_records_created": 0,
        "human_verified_created": 0,
        "top_risk_crop_ids": [crop_id for crop_id, _result, _crop in scored[:50]],
    }


def _rate(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round(100.0 * count / total, 2)


def _median(values: list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return round((ordered[mid - 1] + ordered[mid]) / 2, 4)


def _median_float(values: list[float]) -> float:
    if not values:
        return 0.0
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return round((values[mid - 1] + values[mid]) / 2, 4)
