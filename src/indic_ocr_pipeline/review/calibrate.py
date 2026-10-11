"""Calibration report from human-reviewed samples only.

This report does not enable auto-approval and does not export gold.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from indic_ocr_pipeline.config import load_settings
from indic_ocr_pipeline.manifests import open_manifest
from indic_ocr_pipeline.schemas import ReviewDecision, VerificationStatus
from indic_ocr_pipeline.utils.logging import configure_logging, get_logger

logger = get_logger("review.calibrate")

TIERS = ("P0", "P1", "P2", "P3")


def build_calibration_report(
    verified: Sequence[Mapping[str, Any]],
    reviews: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Summarize human decisions. Unreviewed rows are ignored for rates."""
    review_by_crop = {}
    if reviews:
        review_by_crop = {str(row["crop_id"]): row for row in reviews}
    tiers: dict[str, dict[str, int]] = {
        tier: {
            "reviewed": 0,
            "human_verified": 0,
            "human_corrected": 0,
            "rejected": 0,
            "bad_boundary": 0,
            "candidate_exact": 0,
            "transcription_valid": 0,
        }
        for tier in TIERS
    }
    tiers["unknown"] = {
        "reviewed": 0,
        "human_verified": 0,
        "human_corrected": 0,
        "rejected": 0,
        "bad_boundary": 0,
        "candidate_exact": 0,
        "transcription_valid": 0,
    }
    for row in verified:
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        crop_id = str(row["crop_id"])
        review = review_by_crop.get(crop_id, {})
        if isinstance(review.get("diagnostics"), dict):
            diagnostics = review["diagnostics"]
        else:
            diagnostics = {}
        tier = str(
            meta.get("review_priority") or diagnostics.get("review_priority") or "unknown"
        )
        bucket = tiers.setdefault(
            tier,
            {
                "reviewed": 0,
                "human_verified": 0,
                "human_corrected": 0,
                "rejected": 0,
                "bad_boundary": 0,
                "candidate_exact": 0,
                "transcription_valid": 0,
            },
        )
        bucket["reviewed"] += 1
        status = row.get("verification_status")
        decision = row.get("review_decision")
        if status == VerificationStatus.HUMAN_VERIFIED.value:
            bucket["human_verified"] += 1
            bucket["transcription_valid"] += 1
            bucket["candidate_exact"] += 1
        elif status == VerificationStatus.HUMAN_CORRECTED.value:
            bucket["human_corrected"] += 1
            bucket["transcription_valid"] += 1
        elif decision == ReviewDecision.BAD_BOUNDARY.value:
            bucket["bad_boundary"] += 1
        else:
            bucket["rejected"] += 1
    report: dict[str, Any] = {
        "reviewed_total": len(verified),
        "auto_approval_enabled": False,
        "gold_export": False,
        "score_meaning": "review_risk_score is a ranking heuristic, not correctness probability",
        "tiers": {},
    }
    for tier, bucket in tiers.items():
        if bucket["reviewed"] == 0 and tier == "unknown":
            continue
        reviewed = bucket["reviewed"]
        report["tiers"][tier] = {
            **bucket,
            "exact_accept_rate": _rate(bucket["candidate_exact"], reviewed),
            "correction_rate": _rate(bucket["human_corrected"], reviewed),
            "rejection_rate": _rate(bucket["rejected"], reviewed),
            "bad_boundary_rate": _rate(bucket["bad_boundary"], reviewed),
            "transcription_valid_rate": _rate(bucket["transcription_valid"], reviewed),
        }
    return report


def format_calibration_report(report: Mapping[str, Any]) -> str:
    """Return a text report. It does not claim model accuracy."""
    lines = [
        "Phase 8: human review calibration",
        "Rates use only human-reviewed samples. Unreviewed rows are excluded.",
        "human_verified means an existing candidate text was accepted exactly.",
        "human_corrected means the reviewer typed or edited the transcription.",
        "Do not call human_verified + human_corrected model accuracy.",
        f"auto_approval_enabled: {report['auto_approval_enabled']}",
        f"gold_export: {report['gold_export']}",
        f"reviewed_total: {report['reviewed_total']}",
        "",
    ]
    for tier in [*TIERS, "unknown"]:
        bucket = report["tiers"].get(tier)
        if not bucket:
            continue
        lines.extend(
            [
                f"tier.{tier}:",
                f"  reviewed: {bucket['reviewed']}",
                f"  human_verified (candidate_exact): {bucket['human_verified']}",
                f"  human_corrected: {bucket['human_corrected']}",
                f"  rejected: {bucket['rejected']}",
                f"  bad_boundary: {bucket['bad_boundary']}",
                f"  exact_accept_rate: {bucket['exact_accept_rate']}%",
                f"  correction_rate: {bucket['correction_rate']}%",
                f"  rejection_rate: {bucket['rejection_rate']}%",
                f"  bad_boundary_rate: {bucket['bad_boundary_rate']}%",
                f"  transcription_valid_rate: {bucket['transcription_valid_rate']}%",
                "",
            ]
        )
    lines.append(
        "No auto-approval threshold is enabled. Choose one only after a larger "
        "human-reviewed sample shows an acceptable false-accept rate."
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Report human-review calibration by risk tier. No auto-approval."
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)
    settings = load_settings()
    verified = open_manifest(settings.manifests_dir, "verified").load()
    reviews = open_manifest(settings.manifests_dir, "review").load()
    report = build_calibration_report(verified, reviews)
    text = format_calibration_report(report)
    out = Path(settings.reports_dir) / "review_calibration.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text, flush=True)
    logger.info(
        "calibration report written",
        extra={
            "processed": report["reviewed_total"],
            "completed": report["reviewed_total"],
            "skipped": 0,
            "failed": 0,
        },
    )
    return 0


def _rate(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round(100.0 * count / total, 2)


if __name__ == "__main__":
    raise SystemExit(main())
