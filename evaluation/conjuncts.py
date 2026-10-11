"""Conjunct / Jodakshar occurrence metrics via alignment."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import yaml

from .core_metrics import alignment_pairs

_DEFAULT_PATTERNS = [
    "क्ष", "ज्ञ", "त्र", "श्र", "क्त", "त्त", "द्ध", "ष्ट", "ष्ट्र",
    "न्त्र", "न्द्र", "स्त्र", "प्र", "क्र", "ग्र", "द्र", "ध्य", "त्व", "स्व",
]


def load_conjunct_patterns(config_path: Optional[str | Path] = None) -> List[str]:
    if config_path is None:
        config_path = Path(__file__).parent / "config" / "conjuncts.yaml"
    path = Path(config_path)
    if not path.is_file():
        return list(_DEFAULT_PATTERNS)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    patterns = data.get("patterns") or _DEFAULT_PATTERNS
    # longest first for greedy matching
    return sorted({str(p) for p in patterns}, key=len, reverse=True)


def _find_spans(text: str, patterns: Sequence[str]) -> List[tuple]:
    """Greedy non-overlapping leftmost-longest spans: (start, end, pattern)."""
    spans = []
    i = 0
    n = len(text)
    while i < n:
        matched = None
        for p in patterns:
            if text.startswith(p, i):
                matched = p
                break
        if matched:
            spans.append((i, i + len(matched), matched))
            i += len(matched)
        else:
            i += 1
    return spans


@dataclass
class ConjunctSampleResult:
    has_conjunct: bool
    conjunct_occurrence_accuracy: Optional[float]
    conjunct_exact_match: Optional[bool]
    conjunct_error_count: int
    per_conjunct: Dict[str, Dict[str, float]] = field(default_factory=dict)
    gt_counts: Dict[str, int] = field(default_factory=dict)
    correct_counts: Dict[str, int] = field(default_factory=dict)


def evaluate_conjuncts(
    expected: str,
    prediction: str,
    patterns: Optional[Sequence[str]] = None,
) -> ConjunctSampleResult:
    patterns = list(patterns or load_conjunct_patterns())
    gt_spans = _find_spans(expected, patterns)
    pred_spans = _find_spans(prediction, patterns)

    if not gt_spans and not pred_spans:
        return ConjunctSampleResult(
            has_conjunct=False,
            conjunct_occurrence_accuracy=None,
            conjunct_exact_match=None,
            conjunct_error_count=0,
        )

    # Alignment-based: for each GT conjunct span, check if corresponding
    # aligned hyp characters equal the pattern.
    pairs = alignment_pairs(expected, prediction)
    # Build ref index -> hyp char (or None if deleted)
    ref_to_hyp: List[Optional[str]] = []
    for op, rc, hc in pairs:
        if op in ("eq", "sub"):
            ref_to_hyp.append(hc)
        elif op == "del":
            ref_to_hyp.append(None)
        # insertions don't advance ref

    gt_counts: Dict[str, int] = {}
    correct_counts: Dict[str, int] = {}
    correct = 0
    total = len(gt_spans)
    for start, end, pat in gt_spans:
        gt_counts[pat] = gt_counts.get(pat, 0) + 1
        hyp_chars = ref_to_hyp[start:end]
        if None in hyp_chars:
            ok = False
        else:
            ok = "".join(hyp_chars) == pat  # type: ignore
        if ok:
            correct += 1
            correct_counts[pat] = correct_counts.get(pat, 0) + 1

    # errors = missed GT + extra pred occurrences (approx by count mismatch)
    pred_counts: Dict[str, int] = {}
    for _, _, pat in pred_spans:
        pred_counts[pat] = pred_counts.get(pat, 0) + 1
    error_count = 0
    all_pats = set(gt_counts) | set(pred_counts)
    for p in all_pats:
        g = gt_counts.get(p, 0)
        c = correct_counts.get(p, 0)
        pr = pred_counts.get(p, 0)
        error_count += (g - c) + max(0, pr - g)

    per: Dict[str, Dict[str, float]] = {}
    for p, t in gt_counts.items():
        c = correct_counts.get(p, 0)
        per[p] = {"total": t, "correct": c, "accuracy": (c / t) if t else 0.0}

    acc = (correct / total) if total else (1.0 if not pred_spans else 0.0)
    exact = (gt_counts == pred_counts and correct == total)

    return ConjunctSampleResult(
        has_conjunct=True,
        conjunct_occurrence_accuracy=acc,
        conjunct_exact_match=exact,
        conjunct_error_count=error_count,
        per_conjunct=per,
        gt_counts=gt_counts,
        correct_counts=correct_counts,
    )


@dataclass
class ConjunctCorpusMetrics:
    samples_with_conjunct: int
    conjunct_occurrence_accuracy: Optional[float]
    conjunct_exact_match_rate: Optional[float]
    conjunct_error_count: int
    per_conjunct: Dict[str, Dict[str, float]]


def aggregate_conjuncts(results: Sequence[ConjunctSampleResult]) -> ConjunctCorpusMetrics:
    usable = [r for r in results if r.has_conjunct]
    if not usable:
        return ConjunctCorpusMetrics(0, None, None, 0, {})
    totals: Dict[str, int] = {}
    corrects: Dict[str, int] = {}
    for r in usable:
        for p, t in r.gt_counts.items():
            totals[p] = totals.get(p, 0) + t
            corrects[p] = corrects.get(p, 0) + r.correct_counts.get(p, 0)
    total_occ = sum(totals.values())
    correct_occ = sum(corrects.values())
    acc = (correct_occ / total_occ) if total_occ else None
    exact_rate = sum(1 for r in usable if r.conjunct_exact_match) / len(usable)
    per = {
        p: {
            "total": totals[p],
            "correct": corrects.get(p, 0),
            "accuracy": (corrects.get(p, 0) / totals[p]) if totals[p] else 0.0,
        }
        for p in totals
    }
    return ConjunctCorpusMetrics(
        samples_with_conjunct=len(usable),
        conjunct_occurrence_accuracy=acc,
        conjunct_exact_match_rate=exact_rate,
        conjunct_error_count=sum(r.conjunct_error_count for r in usable),
        per_conjunct=per,
    )
