"""Consensus, calibration, and gold export."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from training.dataset import assert_training_rows_safe, build_splits, load_frozen_ids

from indic_ocr_pipeline.ocr.consensus import BACKENDS, classify_consensus
from indic_ocr_pipeline.review.audit_sample import sample_audit
from pipeline.audit_budget import allocate_audit, allocation_summary
from pipeline.gold import ensure_invalidated_archive, select_gold_rows
from pipeline.golden_sets import (
    write_auto_accepted,
    write_dataset_report,
    write_golden,
    write_quality,
)
from pipeline.quality import quality_report


class CountError(RuntimeError):
    """Crop, prediction, and candidate counts do not match."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(body, encoding="utf-8")
    temporary.replace(path)


def assert_ocr_counts(crops: int, predictions: int, candidates: int, backends: int = 3) -> None:
    if predictions != backends * crops or candidates != crops:
        raise CountError(
            f"Crops : {crops}; OCR predictions : {predictions}; Candidates : {candidates}. "
            f"Expected {crops}, {backends * crops}, {crops}."
        )


def count_summary(crops: int, predictions: int, candidates: int) -> list[str]:
    return [
        f"Crops              : {crops}",
        f"OCR predictions    : {predictions}",
        f"Candidates         : {candidates}",
    ]


def build_consensus(root: Path, crop_ids: set[str] | None = None) -> list[dict[str, Any]]:
    """Build one candidate per crop from the three OCR prediction rows."""
    crops = read_jsonl(root / "manifests" / "crops.jsonl")
    if crop_ids is not None:
        crops = [row for row in crops if str(row.get("crop_id")) in crop_ids]
    predictions = read_jsonl(root / "manifests" / "predictions.jsonl")
    if crop_ids is not None:
        predictions = [row for row in predictions if str(row.get("crop_id")) in crop_ids]
    by_crop: dict[str, dict[str, str | None]] = defaultdict(dict)
    for row in predictions:
        text = row.get("text") if row.get("status") == "complete" else None
        by_crop[str(row["crop_id"])][str(row["backend"])] = text if isinstance(text, str) else None
    names = tuple(dict.fromkeys(str(row["backend"]) for row in predictions if row.get("backend")))
    backends = names or BACKENDS
    records: list[dict[str, Any]] = []
    for crop in crops:
        crop_id = str(crop["crop_id"])
        boundary_valid = crop.get("status") == "complete" and not bool(crop.get("bad_boundary"))
        classified = classify_consensus(
            by_crop.get(crop_id, {}),
            boundary_valid=boundary_valid,
            backends=backends,
        )
        records.append(
            {
                "candidate_id": f"{crop_id}_CAND",
                "sample_id": f"{crop_id}_S",
                "crop_id": crop_id,
                "document_id": crop["document_id"],
                "page_id": crop["page_id"],
                "line_id": crop.get("parent_line_id"),
                "crop_path": crop.get("image_path"),
                **classified,
            }
        )
    assert_ocr_counts(len(crops), len(predictions), len(records), backends=len(backends))
    write_jsonl(root / "manifests" / "consensus.jsonl", records)
    return records


def write_calibration(root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = read_jsonl(root / "manifests" / "consensus.jsonl")
    calibration = config.get("calibration") or {}
    seed = int(calibration.get("seed", config.get("seed", 3407)))
    budget = config.get("human_audit_budget")
    if budget:
        chosen = allocate_audit(rows, int(budget), seed=seed)
    else:
        chosen = sample_audit(rows, seed=seed, rates=calibration.get("rates"))
    write_jsonl(root / "data" / "review" / "calibration" / "sample.jsonl", chosen)
    write_jsonl(root / "data" / "calibration" / "sample.jsonl", chosen)
    summary = allocation_summary(chosen, int(budget or len(chosen)), seed)
    allocation = root / "data" / "calibration" / "sample_allocation.json"
    allocation.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    allocation.write_text(payload, encoding="utf-8")
    return chosen


def export_datasets(root: Path, config: dict[str, Any]) -> dict[str, int]:
    """Write train and validation manifests from human gold only."""
    invalidated = ensure_invalidated_archive(root)
    verified = read_jsonl(root / "manifests" / "verified.jsonl")
    gold, rejected = select_gold_rows(verified, invalidated)
    write_jsonl(root / "data" / "review" / "gold_rejected.jsonl", rejected)
    dataset_cfg = _read_yaml(root / "configs" / "dataset.yaml")
    split = (dataset_cfg.get("split") or {}) if isinstance(dataset_cfg, dict) else {}
    frozen_dir = root / (config.get("paths") or {}).get("frozen_benchmark", "data/eval_data")
    frozen_ids = load_frozen_ids(frozen_dir)
    grouped = {"train": [], "val": [], "test": []}
    if gold:
        grouped = build_splits(
            gold,
            seed=int(split.get("seed", config.get("seed", 3407))),
            train=float(config.get("train_ratio", split.get("train", 0.8))),
            val=float(config.get("val_ratio", split.get("val", 0.1))),
            frozen_ids=frozen_ids,
            frozen_dir=frozen_dir,
        )
    paths = config.get("paths") or {}
    train_path = root / paths.get("train_manifest", "data/datasets/train.jsonl")
    write_jsonl(root / paths.get("val_manifest", "data/datasets/val.jsonl"), grouped["val"])
    write_jsonl(root / "data" / "datasets" / "test.jsonl", grouped["test"])
    golden = write_golden(root, gold)
    consensus = read_jsonl(root / "manifests" / "consensus.jsonl")
    by_crop = {str(row.get("crop_id")): row for row in consensus}
    audits = []
    for row in verified:
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        origin = row.get("review_origin") or meta.get("review_origin")
        if origin != "human":
            continue
        if _touches_frozen(row, frozen_ids):
            continue
        match = by_crop.get(str(row.get("crop_id")))
        if match is None or _touches_frozen(match, frozen_ids):
            continue
        audits.append(
            {
                **match,
                "human_verified_text": row.get("gold_text") or row.get("final_text"),
            }
        )
    threshold = config.get("auto_accept_threshold")
    min_audited = int(config.get("auto_accept_min_audited") or 30)
    report = quality_report(
        consensus,
        audits,
        threshold=None if threshold is None else float(threshold),
        min_audited=min_audited,
    )
    write_quality(root, report)
    auto = write_auto_accepted(
        root,
        consensus,
        report,
        human_ids={str(row.get("crop_id")) for row in gold},
        frozen_ids=frozen_ids,
    )
    train_rows = list(grouped["train"])
    train_auto = 0
    if config.get("include_auto_accepted"):
        for row in auto:
            if _touches_frozen(row, frozen_ids):
                continue
            train_rows.append(row)
            train_auto += 1
    assert_training_rows_safe(
        train_rows,
        frozen_ids,
        frozen_dir,
        allow_auto=bool(config.get("include_auto_accepted")),
    )
    assert_training_rows_safe(grouped["val"] + grouped["test"], frozen_ids, frozen_dir)
    write_jsonl(train_path, train_rows)
    sample = read_jsonl(root / "data" / "calibration" / "sample.jsonl")
    summary = _dataset_summary(
        consensus,
        sample,
        golden=golden,
        rejected=len(rejected),
        auto_accepted=len(auto),
        train_rows=train_rows,
        train_auto=train_auto,
    )
    write_dataset_report(root, summary)
    return {
        "gold": len(gold),
        "rejected": len(rejected),
        "train": len(train_rows),
        "val": len(grouped["val"]),
        "invalidated": len(invalidated),
        "golden": golden,
        "auto_accepted": len(auto),
        "train_auto": train_auto,
    }


def _touches_frozen(row: dict[str, Any], frozen_ids: set[str]) -> bool:
    if not frozen_ids:
        return False
    for key in ("crop_id", "sample_id", "image_filename", "crop_path"):
        value = row.get(key)
        if not isinstance(value, str) or not value:
            continue
        if value in frozen_ids or value.lstrip("/") in frozen_ids:
            return True
    return False


def _dataset_summary(
    consensus: list[dict[str, Any]],
    sample: list[dict[str, Any]],
    *,
    golden: int,
    rejected: int,
    auto_accepted: int,
    train_rows: list[dict[str, Any]],
    train_auto: int,
) -> dict[str, int]:
    unanimous = partial = none = sensitive = 0
    for row in consensus:
        kind = str(row.get("agreement_class") or "")
        status = str(row.get("consensus_status") or "")
        parts = status.split("_")
        full = kind == "ALL_3_AGREE" or (
            len(parts) == 3 and parts[0] == "CONSENSUS" and parts[1] == parts[2]
        )
        if full:
            unanimous += 1
        elif kind == "TWO_AGREE" or status.startswith("CONSENSUS_"):
            partial += 1
        else:
            none += 1
        if row.get("sensitive"):
            sensitive += 1
    return {
        "processed": len(consensus),
        "unanimous": unanimous,
        "partial": partial,
        "no_agreement": none,
        "sensitive": sensitive,
        "audit_selected": len(sample),
        "auto_accepted": auto_accepted,
        "golden": golden,
        "rejected": rejected,
        "train": len(train_rows),
        "train_human": len(train_rows) - train_auto,
        "train_auto": train_auto,
    }


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    import yaml

    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return loaded if isinstance(loaded, dict) else {}
