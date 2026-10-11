"""Matra / Devanagari diacritic metrics via alignment."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Set

from .core_metrics import alignment_pairs

DEFAULT_MATRAS = set("ािीुूृेैोौंःँ्")


@dataclass
class MatraSampleResult:
    has_matra: bool
    matra_accuracy: Optional[float]
    matra_error_count: int
    per_matra: Dict[str, Dict[str, float]] = field(default_factory=dict)
    missing: int = 0
    wrong: int = 0
    extra: int = 0


def evaluate_matras(
    expected: str,
    prediction: str,
    matras: Optional[Set[str]] = None,
) -> MatraSampleResult:
    matras = matras or DEFAULT_MATRAS
    gt_has = any(c in matras for c in expected)
    pred_has = any(c in matras for c in prediction)
    if not gt_has and not pred_has:
        return MatraSampleResult(has_matra=False, matra_accuracy=None, matra_error_count=0)

    totals: Dict[str, int] = {}
    corrects: Dict[str, int] = {}
    missing = wrong = extra = 0
    correct = total = 0

    for op, rc, hc in alignment_pairs(expected, prediction):
        if op == "eq":
            if rc in matras:
                total += 1
                correct += 1
                totals[rc] = totals.get(rc, 0) + 1
                corrects[rc] = corrects.get(rc, 0) + 1
        elif op == "sub":
            if rc in matras:
                total += 1
                totals[rc] = totals.get(rc, 0) + 1
                if hc == rc:
                    correct += 1
                    corrects[rc] = corrects.get(rc, 0) + 1
                else:
                    wrong += 1
            elif hc in matras:
                extra += 1
        elif op == "del":
            if rc in matras:
                total += 1
                totals[rc] = totals.get(rc, 0) + 1
                missing += 1
        elif op == "ins":
            if hc in matras:
                extra += 1

    errors = missing + wrong + extra
    acc = (correct / total) if total else (1.0 if errors == 0 else 0.0)
    per = {
        m: {
            "total": totals[m],
            "correct": corrects.get(m, 0),
            "accuracy": (corrects.get(m, 0) / totals[m]) if totals[m] else 0.0,
        }
        for m in totals
    }
    return MatraSampleResult(
        has_matra=True,
        matra_accuracy=acc,
        matra_error_count=errors,
        per_matra=per,
        missing=missing,
        wrong=wrong,
        extra=extra,
    )


@dataclass
class MatraCorpusMetrics:
    samples_with_matra: int
    matra_accuracy: Optional[float]
    matra_error_count: int
    per_matra: Dict[str, Dict[str, float]]


def aggregate_matras(results: Sequence[MatraSampleResult]) -> MatraCorpusMetrics:
    usable = [r for r in results if r.has_matra]
    if not usable:
        return MatraCorpusMetrics(0, None, 0, {})
    totals: Dict[str, int] = {}
    corrects: Dict[str, int] = {}
    for r in usable:
        for m, info in r.per_matra.items():
            totals[m] = totals.get(m, 0) + int(info["total"])
            corrects[m] = corrects.get(m, 0) + int(info["correct"])
    total = sum(totals.values())
    correct = sum(corrects.values())
    acc = (correct / total) if total else None
    per = {
        m: {
            "total": totals[m],
            "correct": corrects.get(m, 0),
            "accuracy": (corrects.get(m, 0) / totals[m]) if totals[m] else 0.0,
        }
        for m in totals
    }
    return MatraCorpusMetrics(
        samples_with_matra=len(usable),
        matra_accuracy=acc,
        matra_error_count=sum(r.matra_error_count for r in usable),
        per_matra=per,
    )
