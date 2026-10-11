"""Deterministic diagnostic error taxonomy."""
from __future__ import annotations

import re
from typing import List, Optional

from .core_metrics import align_edit_ops, sample_core_metrics
from .numerals import NumeralSampleResult, extract_numeric_tokens
from .punctuation import PunctuationSampleResult
from .conjuncts import ConjunctSampleResult
from .matras import MatraSampleResult
from .reference_ids import ReferenceSampleResult

_DATE_RE = re.compile(
    r"[०-९0-9]{1,2}[./-][०-९0-9]{1,2}[./-][०-९0-9]{2,4}"
)


def classify_errors(
    expected: str,
    prediction: str,
    *,
    numeral: Optional[NumeralSampleResult] = None,
    punctuation: Optional[PunctuationSampleResult] = None,
    conjunct: Optional[ConjunctSampleResult] = None,
    matra: Optional[MatraSampleResult] = None,
    reference: Optional[ReferenceSampleResult] = None,
) -> List[str]:
    """Return ordered taxonomy labels for an incorrect (or empty) prediction.

    Exact matches return an empty list.
    """
    if expected == prediction:
        return []

    labels: List[str] = []

    if prediction == "":
        labels.append("empty_prediction")
        return labels

    if numeral and numeral.numeral_script_error:
        labels.append("numeral_script_error")

    if reference and reference.has_reference_id and not reference.reference_id_exact_match:
        labels.append("reference_id_error")

    if punctuation and punctuation.has_punctuation and punctuation.exact_match is False:
        labels.append("punctuation_error")

    if conjunct and conjunct.has_conjunct and conjunct.conjunct_error_count > 0:
        labels.append("conjunct_error")

    if matra and matra.has_matra and matra.matra_error_count > 0:
        labels.append("matra_error")

    if numeral and numeral.has_numeric and numeral.numeric_exact_match is False:
        if "numeral_script_error" not in labels:
            labels.append("numeric_error")

    gt_dates = _DATE_RE.findall(expected)
    pred_dates = _DATE_RE.findall(prediction)
    if gt_dates and gt_dates != pred_dates:
        labels.append("date_error")

    # whitespace-only difference under NFC collapse would be whitespace_error;
    # detect raw whitespace difference with same non-ws content
    if expected.replace(" ", "") == prediction.replace(" ", "") and expected != prediction:
        labels.append("whitespace_error")

    ops = align_edit_ops(expected, prediction)
    kinds = []
    if ops.insertions:
        kinds.append("insertion")
    if ops.deletions:
        kinds.append("deletion")
    if ops.substitutions:
        kinds.append("substitution")

    if not labels:
        if len(kinds) == 1:
            labels.append(kinds[0])
        elif len(kinds) > 1:
            labels.append("mixed")
            labels.extend(kinds)
        else:
            labels.append("mixed")
    else:
        # still attach dominant edit op when useful
        if len(kinds) > 1:
            labels.append("mixed")
        elif len(kinds) == 1 and kinds[0] not in labels:
            labels.append(kinds[0])

    # de-dupe preserving order
    seen = set()
    out = []
    for x in labels:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
