"""Issue-type and optional difficulty stratification."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence

from .core_metrics import corpus_core_metrics
from .dataset import EvalSample


def difficulty_source_label(
    samples: Sequence[EvalSample],
    used_hard_issue_types: bool,
) -> str:
    if used_hard_issue_types:
        return "configured_issue_type_mapping"
    if any(s.difficulty is not None for s in samples):
        return "explicit_field"
    return "unavailable"


def stratify_by_issue_type(
    samples: Sequence[EvalSample],
    predictions: Sequence[str],
) -> Dict[str, dict]:
    buckets: Dict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        buckets[s.issue_type].append(i)

    out: Dict[str, dict] = {}
    for issue_type, idxs in sorted(buckets.items()):
        gts = [samples[i].expected_text for i in idxs]
        preds = [predictions[i] for i in idxs]
        m = corpus_core_metrics(gts, preds)
        out[issue_type] = {
            "sample_count": len(idxs),
            "cer": m.cer,
            "wer": m.wer,
            "exact_match": m.exact_match_rate,
            "character_accuracy": m.character_accuracy,
            "word_accuracy": m.word_accuracy,
            "empty_prediction_rate": m.empty_prediction_rate,
        }
    return out


def stratify_by_difficulty(
    samples: Sequence[EvalSample],
    predictions: Sequence[str],
) -> Optional[Dict[str, dict]]:
    if not any(s.difficulty is not None for s in samples):
        return None
    buckets: Dict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        key = s.difficulty if s.difficulty is not None else "unspecified"
        buckets[key].append(i)
    out: Dict[str, dict] = {}
    for key, idxs in sorted(buckets.items()):
        gts = [samples[i].expected_text for i in idxs]
        preds = [predictions[i] for i in idxs]
        m = corpus_core_metrics(gts, preds)
        out[key] = {
            "sample_count": len(idxs),
            "cer": m.cer,
            "wer": m.wer,
            "exact_match": m.exact_match_rate,
            "character_accuracy": m.character_accuracy,
            "word_accuracy": m.word_accuracy,
            "empty_prediction_rate": m.empty_prediction_rate,
        }
    return out
