"""Measure consensus precision from human audits. Agreement is not ground truth."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


def wilson_interval(
    correct: int,
    audited: int,
    *,
    z: float = 1.96,
) -> tuple[float | None, float | None]:
    """Return the Wilson score interval for a precision estimate."""
    if audited <= 0:
        return None, None
    proportion = correct / audited
    denominator = 1 + (z * z) / audited
    centre = proportion + (z * z) / (2 * audited)
    spread = (proportion * (1 - proportion) / audited) + (z * z) / (4 * audited * audited)
    margin = z * math.sqrt(spread)
    low = max(0.0, (centre - margin) / denominator)
    high = min(1.0, (centre + margin) / denominator)
    return low, high


_NOT_ELIGIBLE = {"EMPTY", "BAD_BOUNDARY", "ALL_DIFFER", "NO_CONSENSUS"}
STATUS_GUIDE = {
    "human_verified": "A person looked at the line image and approved the transcription.",
    "human_corrected": "A person looked at the line image and replaced the transcription.",
    "auto_accepted_unanimous": (
        "Every configured model matched after comparison normalization, and the "
        "human audit of that category cleared the lower bound. This is not human gold."
    ),
    "auto_accepted_majority": (
        "A majority of models agreed, and the human audit of that category cleared "
        "the lower bound. The agreement count is not a probability, and this is not human gold."
    ),
    "needs_human_review": (
        "A person still has to check this line. Agreement alone did not accept it."
    ),
    "pending_calibration": (
        "This category does not yet have enough human-checked examples to accept lines."
    ),
    "rejected": "A person rejected the line. It is not training gold.",
}


def quality_report(
    population: Sequence[Mapping[str, Any]],
    audits: Sequence[Mapping[str, Any]],
    *,
    threshold: float | None = None,
    min_audited: int = 30,
) -> dict[str, Any]:
    """Compare human text with the suggested transcription for each consensus class.

    Automatic acceptance uses the Wilson lower bound, not the raw accuracy.
    A category with fewer than ``min_audited`` checks stays pending. Two-model
    agreement needs twice that many checks.
    """
    counts: dict[str, int] = defaultdict(int)
    configured: dict[str, int] = {}
    for row in population:
        kind = str(row.get("agreement_class") or "NO_CONSENSUS")
        counts[kind] += 1
        models = int(row.get("models_configured") or 3)
        configured[kind] = min(configured.get(kind, models), models)
    audited: dict[str, int] = defaultdict(int)
    correct: dict[str, int] = defaultdict(int)
    for row in audits:
        kind = str(row.get("agreement_class") or "NO_CONSENSUS")
        audited[kind] += 1
        human = row.get("human_verified_text") or row.get("gold_text") or row.get("final_text")
        suggested = row.get("proposed_text")
        if isinstance(human, str) and isinstance(suggested, str) and human == suggested:
            correct[kind] += 1
    classes: dict[str, Any] = {}
    for kind in sorted(set(counts) | set(audited)):
        low, high = wilson_interval(correct[kind], audited[kind])
        precision = None if audited[kind] == 0 else correct[kind] / audited[kind]
        incorrect = audited[kind] - correct[kind]
        model_count = configured.get(kind, 3)
        required = min_audited * 2 if model_count == 2 else min_audited
        blocked = kind in _NOT_ELIGIBLE
        enough = audited[kind] >= required and required > 0
        eligible = (
            threshold is not None
            and not blocked
            and enough
            and low is not None
            and low >= threshold
        )
        if blocked:
            state = "not_eligible"
        elif threshold is None:
            state = "threshold_not_set"
        elif not enough:
            state = "pending_calibration"
        elif eligible:
            state = "accepted"
        else:
            state = "below_threshold"
        classes[kind] = {
            "population": counts[kind],
            "audited": audited[kind],
            "correct": correct[kind],
            "incorrect": incorrect,
            "precision": precision,
            "interval_low": low,
            "interval_high": high,
            "min_audited": required,
            "models_configured": model_count,
            "calibration_state": state,
            "auto_accept": eligible,
        }
    return {"threshold": threshold, "min_audited": min_audited, "classes": classes}


def resolved_data_status(row: Mapping[str, Any], report: Mapping[str, Any]) -> str:
    """Apply the calibration result. Human gold is never assigned here."""
    if row.get("sensitive"):
        return "needs_human_review"
    candidate = row.get("acceptance_candidate")
    stats = (report.get("classes") or {}).get(str(row.get("agreement_class") or ""))
    if not isinstance(candidate, str) or not isinstance(stats, dict):
        return str(row.get("data_status") or "needs_human_review")
    if stats.get("auto_accept"):
        return candidate
    if stats.get("calibration_state") == "pending_calibration":
        return "pending_calibration"
    return "needs_human_review"


def format_quality_report(report: dict[str, Any]) -> str:
    """Render the consensus quality report in plain language."""
    lines = [
        "Consensus quality report",
        "",
        "Agreement is not ground truth. A category is auto-accepted only when you set",
        "a threshold and the Wilson lower bound on the human audit is at least that",
        "threshold. A small sample stays pending. Calibration does not prove that any",
        "single unread line is correct.",
        "",
        "Status guide",
    ]
    for name, text in STATUS_GUIDE.items():
        lines.append(f"- {name}: {text}")
    lines.append("")
    for kind, stats in report["classes"].items():
        precision = stats["precision"]
        shown = "n/a" if precision is None else f"{precision:.2%}"
        lines.append(f"{kind}:")
        lines.append(f"  population: {stats['population']}")
        lines.append(f"  audited: {stats['audited']}")
        lines.append(f"  correct: {stats['correct']}")
        lines.append(f"  incorrect: {stats.get('incorrect', 0)}")
        lines.append(f"  observed accuracy: {shown}")
        low = stats["interval_low"]
        high = stats["interval_high"]
        lines.append(f"  conservative interval: {low} .. {high}")
        lines.append(f"  audited required: {stats.get('min_audited')}")
        lines.append(f"  calibration: {stats.get('calibration_state')}")
        lines.append(f"  auto_accept: {stats['auto_accept']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def format_dataset_report(summary: Mapping[str, Any]) -> str:
    """Render dataset counts and explain what each count means."""
    lines = [
        "Dataset status report",
        "",
        "data/golden contains only lines a person approved or corrected.",
        "data/auto_accepted contains machine-accepted lines and is not human gold.",
        "The frozen exam is not part of these counts.",
        "",
        f"Processed lines: {summary.get('processed', 0)}",
        f"Unanimous agreement: {summary.get('unanimous', 0)}",
        f"Partial agreement: {summary.get('partial', 0)}",
        f"No agreement: {summary.get('no_agreement', 0)}",
        f"Sensitive lines: {summary.get('sensitive', 0)}",
        f"Selected for manual checking: {summary.get('audit_selected', 0)}",
        f"Automatically accepted: {summary.get('auto_accepted', 0)}",
        f"Human-verified golden: {summary.get('golden', 0)}",
        f"Rejected: {summary.get('rejected', 0)}",
        f"Training rows: {summary.get('train', 0)}",
        f"Training rows from human gold: {summary.get('train_human', 0)}",
        f"Training rows from auto-accept: {summary.get('train_auto', 0)}",
        "",
        "Status guide",
    ]
    for name, text in STATUS_GUIDE.items():
        lines.append(f"- {name}: {text}")
    lines.append("")
    return "\n".join(lines)
