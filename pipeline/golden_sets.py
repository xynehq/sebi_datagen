"""Write the human golden set and, only when the audit allows it, an auto-accept set."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pipeline.quality import format_dataset_report, format_quality_report, resolved_data_status


def write_golden(root: Path, gold_rows: Sequence[Mapping[str, Any]]) -> int:
    """Write ``data/golden`` with human verification only."""
    records = []
    for row in gold_rows:
        method = str(row.get("verification_method") or "human")
        status = str(row.get("data_status") or "")
        decision = str(row.get("decision") or row.get("verification_status") or "")
        if method != "human" or status.startswith("auto_") or method.startswith("auto_"):
            continue
        if decision in {"HUMAN_REJECTED", "rejected", "HUMAN_BAD_BOUNDARY", "bad_boundary"}:
            continue
        if status == "rejected":
            continue
        item = dict(row)
        item["verification_method"] = "human"
        text = row.get("human_verified_text") or row.get("gold_text") or row.get("expected_text")
        item["human_verified_text"] = text
        if status in {"human_verified", "human_corrected"}:
            item["data_status"] = status
        elif decision == "HUMAN_CORRECTED":
            item["data_status"] = "human_corrected"
        else:
            item["data_status"] = "human_verified"
        item["gold"] = True
        records.append(item)
    path = root / "data" / "golden" / "records.jsonl"
    _write(path, records)
    return len(records)


def write_auto_accepted(
    root: Path,
    consensus_rows: Sequence[Mapping[str, Any]],
    report: dict[str, Any],
    *,
    human_ids: set[str],
    frozen_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Write rows whose category cleared the lower bound. Sensitive lines stay out."""
    blocked = frozen_ids or set()
    accepted: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in consensus_rows:
        crop_id = str(row.get("crop_id") or "")
        if not crop_id or crop_id in human_ids or crop_id in seen or crop_id in blocked:
            continue
        if row.get("sensitive") or row.get("gold") is True:
            continue
        status = resolved_data_status(row, report)
        if not status.startswith("auto_accepted"):
            continue
        text = row.get("proposed_text")
        if not isinstance(text, str) or not text.strip():
            continue
        stats = (report.get("classes") or {}).get(str(row.get("agreement_class") or ""), {})
        seen.add(crop_id)
        accepted.append(
            {
                "crop_id": crop_id,
                "document_id": row.get("document_id"),
                "page_id": row.get("page_id"),
                "line_id": row.get("line_id"),
                "crop_path": row.get("crop_path"),
                "text": text,
                "proposed_text": text,
                "predictions": row.get("predictions") or {},
                "agreement_class": row.get("agreement_class"),
                "consensus_status": row.get("consensus_status"),
                "models_configured": row.get("models_configured"),
                "models_agreed": row.get("models_agreed"),
                "models_differed": row.get("models_differed"),
                "differing_similarity": row.get("differing_similarity") or {},
                "pairwise_similarity": row.get("pairwise_similarity") or {},
                "data_status": status,
                "decision_reason": row.get("decision_reason"),
                "audited": stats.get("audited"),
                "interval_low": stats.get("interval_low"),
                "threshold": report.get("threshold"),
                "verification_method": f"auto_{status}",
                "review_origin": "auto_consensus",
                "training_weight": 1.0,
                "gold": False,
            }
        )
    _write(root / "data" / "auto_accepted" / "records.jsonl", accepted)
    return accepted


def write_dataset_report(root: Path, summary: Mapping[str, Any]) -> None:
    """Write the plain-language dataset counts next to the calibration report."""
    directory = root / "data" / "calibration"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "dataset_quality.json").write_text(
        json.dumps(dict(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (directory / "dataset_quality.txt").write_text(
        format_dataset_report(summary),
        encoding="utf-8",
    )


def write_quality(root: Path, report: dict[str, Any]) -> None:
    """Write the precision report next to the calibration sample."""
    directory = root / "data" / "calibration"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "consensus_quality.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    text = format_quality_report(report)
    (directory / "consensus_quality.txt").write_text(text, encoding="utf-8")


def _write(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")
