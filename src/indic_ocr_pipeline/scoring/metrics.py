"""Compare OCR predictions. Agreement is evidence for review ranking, not truth.

The comparison key is NFC plus surrounding-whitespace trim plus collapsed
internal whitespace. It is never stored as a transcription.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from indic_ocr_pipeline.schemas import DisagreementType
from indic_ocr_pipeline.text_policy import normalize_for_comparison

BACKENDS: tuple[str, ...] = ("indicocr", "chitrapathak", "tesseract")
PAIR_ORDER: tuple[tuple[str, str], ...] = (
    ("indicocr", "chitrapathak"),
    ("indicocr", "tesseract"),
    ("chitrapathak", "tesseract"),
)

ALL_3_AGREE = "ALL_3_AGREE"
TWO_AGREE_ONE_DIFFERS = "TWO_AGREE_ONE_DIFFERS"
ALL_3_DIFFER = "ALL_3_DIFFER"
ONE_OR_MORE_EMPTY = "ONE_OR_MORE_EMPTY"
PREDICTION_ERROR = "PREDICTION_ERROR"

TESSERACT_CONFIDENCE_SCALE = (
    "stored_0_to_1_mean_of_tesseract_word_confidence_0_to_100_divided_by_100"
)


@dataclass(frozen=True)
class PairMetrics:
    """Edit statistics from one engine's text to the other. Not ground truth."""

    exact: bool
    normalized: bool
    edit_distance: int
    insertions: int
    deletions: int
    substitutions: int
    prediction_pair_cer: float

    def to_dict(self) -> dict[str, object]:
        return {
            "exact": self.exact,
            "normalized": self.normalized,
            "edit_distance": self.edit_distance,
            "insertions": self.insertions,
            "deletions": self.deletions,
            "substitutions": self.substitutions,
            "prediction_pair_cer": self.prediction_pair_cer,
        }


def comparison_key(text: str) -> str:
    """Return the comparison key. Callers must not store this as a label."""
    return normalize_for_comparison(text)


def edit_counts(left: str, right: str) -> tuple[int, int, int, int]:
    """Return distance, insertions, deletions, and substitutions from left to right.

    Equal-cost paths prefer a substitution over a deletion over an insertion
    so the counts are deterministic. Characters are Python code points.
    """
    if not isinstance(left, str) or not isinstance(right, str):
        raise TypeError("edit_counts requires strings")
    rows = len(left) + 1
    cols = len(right) + 1
    distance = [[0] * cols for _ in range(rows)]
    for row in range(1, rows):
        distance[row][0] = row
    for col in range(1, cols):
        distance[0][col] = col
    for row in range(1, rows):
        for col in range(1, cols):
            substitution_cost = 0 if left[row - 1] == right[col - 1] else 1
            distance[row][col] = min(
                distance[row - 1][col] + 1,
                distance[row][col - 1] + 1,
                distance[row - 1][col - 1] + substitution_cost,
            )
    insertions = 0
    deletions = 0
    substitutions = 0
    row = len(left)
    col = len(right)
    while row > 0 or col > 0:
        if row > 0 and col > 0:
            cost = 0 if left[row - 1] == right[col - 1] else 1
            if distance[row][col] == distance[row - 1][col - 1] + cost:
                if cost:
                    substitutions += 1
                row -= 1
                col -= 1
                continue
        if row > 0 and distance[row][col] == distance[row - 1][col] + 1:
            deletions += 1
            row -= 1
            continue
        insertions += 1
        col -= 1
    return distance[len(left)][len(right)], insertions, deletions, substitutions


def pair_metrics(left: str, right: str) -> PairMetrics:
    """Compare two predictions. ``prediction_pair_cer`` is not a ground-truth CER."""
    distance, insertions, deletions, substitutions = edit_counts(left, right)
    denominator = max(len(left), len(right), 1)
    return PairMetrics(
        exact=left == right,
        normalized=comparison_key(left) == comparison_key(right),
        edit_distance=distance,
        insertions=insertions,
        deletions=deletions,
        substitutions=substitutions,
        prediction_pair_cer=round(distance / denominator, 4),
    )


def agreement_category(texts: Mapping[str, str | None]) -> str:
    """Classify three predictions. No category means the text is correct."""
    values = [texts.get(name) for name in BACKENDS]
    if any(value is None for value in values):
        return PREDICTION_ERROR
    present = [value for value in values if isinstance(value, str)]
    if any(value == "" for value in present):
        return ONE_OR_MORE_EMPTY
    counts = Counter(present)
    if len(counts) == 1:
        return ALL_3_AGREE
    if counts.most_common(1)[0][1] == 2:
        return TWO_AGREE_ONE_DIFFERS
    return ALL_3_DIFFER


def proposed_consensus(texts: Mapping[str, str | None]) -> str | None:
    """Return a non-empty majority string, or null when no majority exists.

    The result is one of the original prediction strings. It is not gold and
    it is not the comparison key. An empty majority is not a consensus.
    """
    values = [texts.get(name) for name in BACKENDS]
    if any(not isinstance(value, str) for value in values):
        return None
    counts = Counter(value for value in values if isinstance(value, str))
    text, votes = counts.most_common(1)[0]
    if votes >= 2 and text != "":
        return text
    return None


def disagreement_label(texts: Sequence[str]) -> DisagreementType:
    """Label how the stored strings differ. This label is not a correction."""
    if any(text == "" for text in texts):
        return DisagreementType.EMPTY
    if len(set(texts)) == 1:
        return DisagreementType.NONE
    keys = [comparison_key(text) for text in texts]
    if len(set(keys)) == 1:
        stripped = [unicodedata.normalize("NFC", text).strip() for text in texts]
        if len(set(stripped)) > 1:
            return DisagreementType.WHITESPACE
        return DisagreementType.UNICODE_NORMALIZATION
    if _only_digit_difference(keys):
        return DisagreementType.NUMERAL
    if _only_punctuation_difference(keys):
        return DisagreementType.PUNCTUATION
    return DisagreementType.CHARACTER_SUBSTITUTION


def diagnostic_flags(texts: Mapping[str, str]) -> list[str]:
    """Return sorted review diagnostics. These flags are not truth labels."""
    flags: set[str] = set()
    values = [texts[name] for name in BACKENDS]
    if any(text == "" for text in values):
        flags.add("empty_prediction")
    if len(set(values)) != 1:
        flags.add("multi_engine_disagreement")
    lengths = [len(text) for text in values]
    length_min = min(lengths)
    length_max = max(lengths)
    if length_max - length_min >= 8 or (length_min > 0 and length_max >= 2 * length_min):
        flags.add("large_length_disagreement")
    if any(_unicode_anomaly(text) for text in values):
        flags.add("unicode_anomaly")
    if _punctuation_sequences(values):
        flags.add("punctuation_disagreement")
    if len({_digits(text) for text in values}) > 1:
        flags.add("numeral_disagreement")
    keys = [comparison_key(text) for text in values if text]
    if _prefix_gap(keys):
        flags.add("likely_truncation")
        flags.add("likely_extra_text")
    return sorted(flags)


def length_stats(texts: Mapping[str, str]) -> dict[str, int]:
    """Compare prediction lengths with each other. There is no gold length."""
    lengths = [len(texts[name]) for name in BACKENDS]
    return {
        "length_min": min(lengths),
        "length_max": max(lengths),
        "length_range": max(lengths) - min(lengths),
    }


def _digits(text: str) -> str:
    return "".join(character for character in text if character.isdecimal())


def _without_digits(text: str) -> str:
    return "".join(character for character in text if not character.isdecimal())


def _without_punctuation(text: str) -> str:
    return "".join(
        character for character in text if not unicodedata.category(character).startswith("P")
    )


def _only_digit_difference(keys: Sequence[str]) -> bool:
    if len({_digits(key) for key in keys}) <= 1:
        return False
    return len({_without_digits(key) for key in keys}) == 1


def _only_punctuation_difference(keys: Sequence[str]) -> bool:
    stripped = [_without_punctuation(key) for key in keys]
    if len(set(stripped)) != 1:
        return False
    return len({_punctuation(key) for key in keys}) > 1


def _punctuation(text: str) -> str:
    return "".join(
        character for character in text if unicodedata.category(character).startswith("P")
    )


def _punctuation_sequences(texts: Sequence[str]) -> bool:
    sequences = {_punctuation(comparison_key(text)) for text in texts}
    return len(sequences) > 1


def _unicode_anomaly(text: str) -> bool:
    if "\ufffd" in text or "$$" in text:
        return True
    for character in text:
        code = ord(character)
        if character in "\n\t":
            continue
        if code < 32 or 0xE000 <= code <= 0xF8FF:
            return True
    return False


def _prefix_gap(keys: Sequence[str]) -> bool:
    ordered = sorted(set(keys), key=len)
    if len(ordered) < 2:
        return False
    short = ordered[0]
    long = ordered[-1]
    if not short or not long.startswith(short):
        return False
    return len(long) - len(short) >= 4 and len(short) / len(long) <= 0.6
