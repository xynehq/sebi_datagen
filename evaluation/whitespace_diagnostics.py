"""Whitespace / content diagnostic views (STRICT remains primary)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence, Tuple

from .core_metrics import alignment_pairs, corpus_core_metrics, sample_core_metrics
from .normalization import (
    content_without_whitespace,
    diagnostic_nfc_whitespace,
    strict,
)


@dataclass
class WhitespaceViews:
    strict_exact_match: bool
    whitespace_normalized_exact_match: bool
    content_without_whitespace_exact_match: bool
    strict_cer: float
    whitespace_normalized_cer: float
    content_without_whitespace_cer: float
    space_insertion_count: int
    space_deletion_count: int
    space_substitution_count: int
    whitespace_error_count: int
    word_boundary_merge_count: int
    word_boundary_split_count: int


def _space_edit_counts(gt: str, pred: str) -> Tuple[int, int, int]:
    """Count space-related ops from character alignment."""
    ins = dels = subs = 0
    for op, rc, hc in alignment_pairs(gt, pred):
        if op == "ins" and hc.isspace():
            ins += 1
        elif op == "del" and rc.isspace():
            dels += 1
        elif op == "sub" and (rc.isspace() or hc.isspace()):
            subs += 1
    return ins, dels, subs


def _word_boundary_events(gt: str, pred: str) -> Tuple[int, int]:
    """Heuristic merge/split counts.

    merge: adjacent GT words appear concatenated in prediction
    split: a GT word appears split by inserted space(s) in prediction
    """
    gt_words = gt.split()
    pred_words = pred.split()
    merges = 0
    splits = 0

    pred_nospace = content_without_whitespace(pred)
    i = 0
    while i < len(gt_words) - 1:
        a, b = gt_words[i], gt_words[i + 1]
        joined = a + b
        if joined and joined in pred_nospace:
            if not re.search(re.escape(a) + r"\s+" + re.escape(b), pred):
                merges += 1
                i += 2
                continue
        i += 1

    for w in gt_words:
        if len(w) < 2:
            continue
        pattern = r"\s+".join(re.escape(ch) for ch in w)
        if re.search(pattern, pred) and w not in pred_words:
            splits += 1

    return merges, splits


def evaluate_whitespace_views(expected: str, prediction: str) -> WhitespaceViews:
    s_gt, s_pr = strict(expected), strict(prediction)
    w_gt, w_pr = diagnostic_nfc_whitespace(expected), diagnostic_nfc_whitespace(prediction)
    c_gt, c_pr = content_without_whitespace(expected), content_without_whitespace(prediction)

    s_cer = sample_core_metrics(s_gt, s_pr).cer
    w_cer = sample_core_metrics(w_gt, w_pr).cer
    c_cer = sample_core_metrics(c_gt, c_pr).cer

    si, sd, ss = _space_edit_counts(s_gt, s_pr)
    merges, splits = _word_boundary_events(s_gt, s_pr)

    return WhitespaceViews(
        strict_exact_match=(s_gt == s_pr),
        whitespace_normalized_exact_match=(w_gt == w_pr),
        content_without_whitespace_exact_match=(c_gt == c_pr),
        strict_cer=s_cer,
        whitespace_normalized_cer=w_cer,
        content_without_whitespace_cer=c_cer,
        space_insertion_count=si,
        space_deletion_count=sd,
        space_substitution_count=ss,
        whitespace_error_count=si + sd + ss,
        word_boundary_merge_count=merges,
        word_boundary_split_count=splits,
    )


def corpus_view_metrics(expecteds: Sequence[str], predictions: Sequence[str], view: str) -> dict:
    if view == "strict":
        gts = [strict(t) for t in expecteds]
        preds = [strict(t) for t in predictions]
    elif view == "whitespace_normalized":
        gts = [diagnostic_nfc_whitespace(t) for t in expecteds]
        preds = [diagnostic_nfc_whitespace(t) for t in predictions]
    elif view == "content_without_whitespace":
        gts = [content_without_whitespace(t) for t in expecteds]
        preds = [content_without_whitespace(t) for t in predictions]
    else:
        raise ValueError(view)
    m = corpus_core_metrics(gts, preds)
    return {
        "cer": m.cer,
        "wer": m.wer,
        "exact_match_rate": m.exact_match_rate,
        "exact_match_count": m.exact_match_count,
    }


def corpus_view_cer(expecteds: Sequence[str], predictions: Sequence[str], view: str) -> float:
    return corpus_view_metrics(expecteds, predictions, view)["cer"]


def aggregate_whitespace(
    expecteds: Sequence[str],
    predictions: Sequence[str],
) -> dict:
    views = [evaluate_whitespace_views(g, p) for g, p in zip(expecteds, predictions)]
    n = len(views) or 1
    strict_m = corpus_view_metrics(expecteds, predictions, "strict")
    ws_m = corpus_view_metrics(expecteds, predictions, "whitespace_normalized")
    c_m = corpus_view_metrics(expecteds, predictions, "content_without_whitespace")
    return {
        "strict_exact_match_rate": sum(v.strict_exact_match for v in views) / n,
        "whitespace_normalized_exact_match_rate": sum(
            v.whitespace_normalized_exact_match for v in views
        )
        / n,
        "content_without_whitespace_exact_match_rate": sum(
            v.content_without_whitespace_exact_match for v in views
        )
        / n,
        "strict_cer": strict_m["cer"],
        "strict_wer": strict_m["wer"],
        "whitespace_normalized_cer": ws_m["cer"],
        "whitespace_normalized_wer": ws_m["wer"],
        "whitespace_normalized_exact_match_count": ws_m["exact_match_count"],
        "content_without_whitespace_cer": c_m["cer"],
        "content_without_whitespace_exact_match_count": c_m["exact_match_count"],
        "space_insertion_count": sum(v.space_insertion_count for v in views),
        "space_deletion_count": sum(v.space_deletion_count for v in views),
        "space_substitution_count": sum(v.space_substitution_count for v in views),
        "whitespace_error_count": sum(v.whitespace_error_count for v in views),
        "word_boundary_merge_count": sum(v.word_boundary_merge_count for v in views),
        "word_boundary_split_count": sum(v.word_boundary_split_count for v in views),
    }
