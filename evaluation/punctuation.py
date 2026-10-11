"""Punctuation precision / recall / F1 from alignment."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Set

import yaml

from .core_metrics import alignment_pairs

_DEFAULT_PUNCT = set(".,/-—:;()[]{}*%₹\"'!?&|")


def load_punctuation_set(config_path: Optional[str | Path] = None) -> Set[str]:
    if config_path is None:
        config_path = Path(__file__).parent / "config" / "punctuation.yaml"
    path = Path(config_path)
    if not path.is_file():
        return set(_DEFAULT_PUNCT)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    chars = data.get("characters") or []
    return set(str(c) for c in chars) or set(_DEFAULT_PUNCT)


@dataclass
class PunctuationSampleResult:
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    exact_match: Optional[bool]
    has_punctuation: bool
    tp: int = 0
    fp: int = 0
    fn: int = 0


def evaluate_punctuation(
    expected: str,
    prediction: str,
    punct: Optional[Set[str]] = None,
) -> PunctuationSampleResult:
    punct = punct or load_punctuation_set()
    gt_punct = [c for c in expected if c in punct]
    pred_punct = [c for c in prediction if c in punct]
    has = bool(gt_punct) or bool(pred_punct)
    if not has:
        return PunctuationSampleResult(
            precision=None, recall=None, f1=None, exact_match=None, has_punctuation=False
        )

    # Alignment-based TP/FP/FN on punctuation positions
    tp = fp = fn = 0
    for op, rc, hc in alignment_pairs(expected, prediction):
        r_p = rc in punct
        h_p = hc in punct
        if op == "eq":
            if r_p:
                tp += 1
        elif op == "sub":
            if r_p and h_p and rc == hc:
                tp += 1
            elif r_p and h_p and rc != hc:
                fn += 1
                fp += 1
            elif r_p and not h_p:
                fn += 1
            elif h_p and not r_p:
                fp += 1
        elif op == "del":
            if r_p:
                fn += 1
        elif op == "ins":
            if h_p:
                fp += 1

    precision = tp / (tp + fp) if (tp + fp) else (1.0 if not pred_punct else 0.0)
    recall = tp / (tp + fn) if (tp + fn) else (1.0 if not gt_punct else 0.0)
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )
    exact = "".join(gt_punct) == "".join(pred_punct) and len(gt_punct) == len(pred_punct)
    # stricter: punctuation subsequence exact via strings of only punct
    exact = "".join(gt_punct) == "".join(pred_punct)

    return PunctuationSampleResult(
        precision=precision,
        recall=recall,
        f1=f1,
        exact_match=exact,
        has_punctuation=True,
        tp=tp,
        fp=fp,
        fn=fn,
    )


@dataclass
class PunctuationCorpusMetrics:
    samples_with_punctuation: int
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    exact_match_rate: Optional[float]
    tp: int = 0
    fp: int = 0
    fn: int = 0


def aggregate_punctuation(results: Sequence[PunctuationSampleResult]) -> PunctuationCorpusMetrics:
    usable = [r for r in results if r.has_punctuation]
    if not usable:
        return PunctuationCorpusMetrics(0, None, None, None, None, 0, 0, 0)
    tp = sum(r.tp for r in usable)
    fp = sum(r.fp for r in usable)
    fn = sum(r.fn for r in usable)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    exact = sum(1 for r in usable if r.exact_match) / len(usable)
    return PunctuationCorpusMetrics(
        samples_with_punctuation=len(usable),
        precision=precision,
        recall=recall,
        f1=f1,
        exact_match_rate=exact,
        tp=tp,
        fp=fp,
        fn=fn,
    )
