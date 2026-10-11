"""Score prediction pairs with the canonical evaluation harness.

This module does not reimplement CER, WER, or the script diagnostics.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from evaluation.conjuncts import aggregate_conjuncts, evaluate_conjuncts
from evaluation.core_metrics import corpus_core_metrics, sample_core_metrics
from evaluation.length_diagnostics import compute_gt_length_buckets
from evaluation.matras import aggregate_matras, evaluate_matras
from evaluation.normalization import content_without_whitespace
from evaluation.numerals import aggregate_numerals, evaluate_numerals
from evaluation.punctuation import aggregate_punctuation, evaluate_punctuation
from evaluation.reference_ids import aggregate_reference_ids, evaluate_reference_id
from evaluation.taxonomy import classify_errors
from evaluation.wer_diagnostics import aggregate_wer_views
from evaluation.whitespace_diagnostics import aggregate_whitespace


def _fields(value: Any) -> dict[str, Any]:
    if hasattr(value, "__dataclass_fields__"):
        return {name: getattr(value, name) for name in value.__dataclass_fields__}
    return {"value": value}


def score_corpus(
    expecteds: Sequence[str],
    predictions: Sequence[str],
) -> dict[str, Any]:
    """Return strict primary metrics plus the canonical diagnostics.

    Primary metric is strict CER (lower is better). Content CER is diagnostic.
    """
    if len(expecteds) != len(predictions):
        raise ValueError("expected and prediction sequences differ in length")
    strict = corpus_core_metrics(list(expecteds), list(predictions))
    content = corpus_core_metrics(
        [content_without_whitespace(text) for text in expecteds],
        [content_without_whitespace(text) for text in predictions],
    )
    pairs = list(zip(expecteds, predictions, strict=True))
    numeral_rows = [evaluate_numerals(exp, pred) for exp, pred in pairs]
    punct_rows = [evaluate_punctuation(exp, pred) for exp, pred in pairs]
    conjunct_rows = [evaluate_conjuncts(exp, pred) for exp, pred in pairs]
    matra_rows = [evaluate_matras(exp, pred) for exp, pred in pairs]
    reference_rows = [evaluate_reference_id(exp, pred) for exp, pred in pairs]
    issues: Counter[str] = Counter()
    per_sample: list[dict[str, Any]] = []
    for index, (exp, pred) in enumerate(zip(expecteds, predictions, strict=True)):
        sample = sample_core_metrics(exp, pred)
        labels = classify_errors(
            exp,
            pred,
            numeral=numeral_rows[index],
            punctuation=punct_rows[index],
            conjunct=conjunct_rows[index],
            matra=matra_rows[index],
            reference=reference_rows[index],
        )
        issues.update(labels)
        per_sample.append(
            {
                "expected_text": exp,
                "prediction": pred,
                "reference_length": sample.reference_length,
                "prediction_length": len(pred),
                "edit_distance": sample.edit_distance,
            }
        )
    return {
        "primary_metric": "cer",
        "goal": "minimize",
        "cer": strict.cer,
        "cer_repo_compat": strict.cer_repo_compat,
        "content_cer": content.cer,
        "wer": strict.wer,
        "wer_strict": strict.wer,
        "exact_match_rate": strict.exact_match_rate,
        "exact_match_count": strict.exact_match_count,
        "character_accuracy": strict.character_accuracy,
        "word_accuracy": strict.word_accuracy,
        "substitutions": strict.total_substitutions,
        "deletions": strict.total_deletions,
        "insertions": strict.total_insertions,
        "sample_count": strict.total_samples,
        "punctuation": _fields(aggregate_punctuation(punct_rows)),
        "matra": _fields(aggregate_matras(matra_rows)),
        "conjunct": _fields(aggregate_conjuncts(conjunct_rows)),
        "numeral": _fields(aggregate_numerals(numeral_rows)),
        "reference_id": _fields(aggregate_reference_ids(reference_rows)),
        "length_buckets": compute_gt_length_buckets(per_sample),
        "issue_types": dict(issues),
        "whitespace_diagnostics": aggregate_whitespace(list(expecteds), list(predictions)),
        "wer_views": aggregate_wer_views(list(expecteds), list(predictions)),
        "metric_source": "evaluation.core_metrics",
    }
