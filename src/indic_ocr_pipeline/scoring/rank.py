"""Review-risk ranking. The score orders human review. It is not a probability."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from indic_ocr_pipeline.schemas import ScoreReason
from indic_ocr_pipeline.scoring.metrics import (
    ALL_3_AGREE,
    ALL_3_DIFFER,
    BACKENDS,
    ONE_OR_MORE_EMPTY,
    PAIR_ORDER,
    PREDICTION_ERROR,
    TWO_AGREE_ONE_DIFFERS,
    PairMetrics,
    agreement_category,
    diagnostic_flags,
    length_stats,
    pair_metrics,
    proposed_consensus,
)

WEIGHT_KEYS: tuple[str, ...] = (
    "agreement",
    "tesseract_confidence",
    "image_quality",
    "crop_geometry",
    "unicode_sanity",
)

# Line-box signals already stored by physical-line detection. Pixels are not read.
SHORT_LINE_HEIGHT_PX = 12
TALL_LINE_HEIGHT_PX = 80
NARROW_LINE_WIDTH_PX = 40

PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
CATEGORY_ORDER = {
    ONE_OR_MORE_EMPTY: 0,
    PREDICTION_ERROR: 1,
    ALL_3_DIFFER: 2,
    TWO_AGREE_ONE_DIFFERS: 3,
    ALL_3_AGREE: 4,
}


@dataclass(frozen=True)
class ReviewScore:
    """One crop's review ranking. ``review_risk_score`` is not correctness."""

    agreement_category: str
    proposed_consensus: str | None
    review_risk_score: float
    confidence_score: float
    review_priority: str
    reasons: list[str]
    flags: list[str]
    pairs: dict[str, PairMetrics]
    lengths: dict[str, int]
    feature_risks: dict[str, float]
    tesseract_confidence: float | None

    @property
    def max_prediction_pair_cer(self) -> float:
        if not self.pairs:
            return 1.0
        return max(pair.prediction_pair_cer for pair in self.pairs.values())

    @property
    def max_edit_distance(self) -> int:
        if not self.pairs:
            return 0
        return max(pair.edit_distance for pair in self.pairs.values())

    def sort_key(self, crop_id: str) -> tuple[object, ...]:
        """Higher bands and larger edit distances are inspected first."""
        return (
            PRIORITY_ORDER[self.review_priority],
            CATEGORY_ORDER[self.agreement_category],
            -self.max_edit_distance,
            -self.max_prediction_pair_cer,
            -self.review_risk_score,
            crop_id,
        )


def score_predictions(
    texts: Mapping[str, str | None],
    *,
    tesseract_confidence: float | None,
    line_width: int | None,
    line_height: int | None,
    weights: Mapping[str, float],
    high_confidence_threshold: float,
) -> ReviewScore:
    """Rank one crop from the three stored prediction strings.

    Higher ``review_risk_score`` means inspect sooner. Three-way agreement
    lowers the score and does not mark the text correct. Tesseract confidence
    is only the ``tesseract_confidence`` weight, so it cannot cancel disagreement.
    """
    _require_weights(weights)
    category = agreement_category(texts)
    consensus = proposed_consensus(texts)
    present = {name: texts[name] for name in BACKENDS if isinstance(texts.get(name), str)}
    pairs: dict[str, PairMetrics] = {}
    if len(present) == len(BACKENDS):
        for left, right in PAIR_ORDER:
            pairs[f"{left}|{right}"] = pair_metrics(present[left], present[right])
    if len(present) == len(BACKENDS):
        flags = diagnostic_flags(present)
    else:
        flags = ["prediction_error"]
    lengths = length_stats(present) if len(present) == len(BACKENDS) else {
        "length_min": 0,
        "length_max": 0,
        "length_range": 0,
    }
    max_cer = max((pair.prediction_pair_cer for pair in pairs.values()), default=1.0)
    features = {
        "agreement": _agreement_risk(category, max_cer),
        "tesseract_confidence": _tesseract_risk(tesseract_confidence),
        "image_quality": 1.0 if category == ONE_OR_MORE_EMPTY else 0.0,
        "crop_geometry": _geometry_risk(line_width, line_height),
        "unicode_sanity": 1.0 if "unicode_anomaly" in flags else 0.0,
    }
    risk = round(sum(weights[key] * features[key] for key in WEIGHT_KEYS), 4)
    if category == ONE_OR_MORE_EMPTY:
        nonempty = [text for text in present.values() if text]
        risk = 1.0 if not nonempty else max(risk, 0.95)
    if category == PREDICTION_ERROR:
        risk = 1.0
    risk = round(min(1.0, max(0.0, risk)), 4)
    confidence = round(1.0 - risk, 4)
    stored_texts = {name: present[name] for name in BACKENDS} if len(present) == 3 else {}
    reasons = _reasons(
        category,
        flags,
        tesseract_confidence,
        features,
        high_confidence_threshold,
        stored_texts,
    )
    return ReviewScore(
        agreement_category=category,
        proposed_consensus=consensus,
        review_risk_score=risk,
        confidence_score=confidence,
        review_priority=_priority(category, max_cer, flags),
        reasons=reasons,
        flags=flags,
        pairs=pairs,
        lengths=lengths,
        feature_risks={key: round(features[key], 4) for key in WEIGHT_KEYS},
        tesseract_confidence=tesseract_confidence,
    )


def _require_weights(weights: Mapping[str, float]) -> None:
    if tuple(weights) != WEIGHT_KEYS and set(weights) != set(WEIGHT_KEYS):
        raise ValueError("review weights must be the scoring.yaml keys")
    total = sum(weights[key] for key in WEIGHT_KEYS)
    if abs(total - 1.0) > 1e-9:
        raise ValueError("review weights must sum to 1")


def _agreement_risk(category: str, max_cer: float) -> float:
    """Map disagreement to risk in ``[0, 1]``. Exact agreement stays above zero."""
    if category == ALL_3_AGREE:
        return 0.15
    if category == TWO_AGREE_ONE_DIFFERS:
        return 0.40 + 0.35 * max_cer
    if category == ALL_3_DIFFER:
        return 0.80 + 0.20 * max_cer
    return 1.0


def _tesseract_risk(confidence: float | None) -> float:
    """Use the stored 0-1 score. Null stays a neutral 0.5 and is not invented."""
    if confidence is None:
        return 0.5
    if isinstance(confidence, bool) or not isinstance(confidence, int | float):
        raise TypeError("tesseract confidence must be a number or null")
    if not 0.0 <= float(confidence) <= 1.0:
        raise ValueError("stored tesseract confidence must be in [0, 1]")
    return 1.0 - float(confidence)


def _geometry_risk(width: int | None, height: int | None) -> float:
    if width is None or height is None:
        return 0.5
    if height <= SHORT_LINE_HEIGHT_PX or width <= NARROW_LINE_WIDTH_PX:
        return 1.0
    if height >= TALL_LINE_HEIGHT_PX:
        return 0.6
    return 0.0


def _priority(category: str, max_cer: float, flags: list[str]) -> str:
    if category in {ONE_OR_MORE_EMPTY, PREDICTION_ERROR}:
        return "P0"
    severe = {
        "large_length_disagreement",
        "unicode_anomaly",
        "likely_truncation",
        "likely_extra_text",
    }
    if category == ALL_3_DIFFER or max_cer >= 0.5 or any(flag in severe for flag in flags):
        return "P1"
    if category == TWO_AGREE_ONE_DIFFERS:
        return "P2"
    return "P3"


def _reasons(
    category: str,
    flags: list[str],
    tesseract_confidence: float | None,
    features: Mapping[str, float],
    high_confidence_threshold: float,
    texts: Mapping[str, str],
) -> list[str]:
    reasons: list[str] = []
    if category == ALL_3_AGREE:
        reasons.append(ScoreReason.ALL_BACKENDS_AGREE.value)
    elif category == PREDICTION_ERROR:
        reasons.append(ScoreReason.BACKENDS_DISAGREE.value)
    elif category == TWO_AGREE_ONE_DIFFERS:
        reasons.append(ScoreReason.BACKENDS_DISAGREE.value)
    else:
        reasons.append(ScoreReason.BACKENDS_DISAGREE.value)
    if "empty_prediction" in flags:
        reasons.append(ScoreReason.EMPTY_OUTPUT.value)
    reasons.append(ScoreReason.MODEL_CONFIDENCE_UNAVAILABLE.value)
    if tesseract_confidence is None:
        reasons.append(ScoreReason.TESSERACT_CONFIDENCE_UNAVAILABLE.value)
    elif tesseract_confidence >= high_confidence_threshold:
        reasons.append(ScoreReason.TESSERACT_HIGH_CONFIDENCE.value)
    elif tesseract_confidence < 0.5:
        reasons.append(ScoreReason.TESSERACT_LOW_CONFIDENCE.value)
    if features["crop_geometry"] > 0:
        reasons.append(ScoreReason.SUSPICIOUS_CROP_BOUNDARY.value)
    else:
        reasons.append(ScoreReason.CLEAN_CROP.value)
    if features["image_quality"] > 0:
        reasons.append(ScoreReason.LOW_IMAGE_QUALITY.value)
    if "unicode_anomaly" in flags:
        reasons.append(ScoreReason.SUSPICIOUS_UNICODE.value)
    elif texts:
        reasons.append(ScoreReason.VALID_UNICODE.value)
    if texts and max(len(text) for text in texts.values()) <= 2:
        reasons.append(ScoreReason.VERY_SHORT_OUTPUT.value)
    if "numeral_disagreement" in flags:
        reasons.append(ScoreReason.NUMERIC_MISMATCH.value)
    elif texts:
        reasons.append(ScoreReason.NO_NUMERIC_CONFLICT.value)
    if "punctuation_disagreement" in flags:
        reasons.append(ScoreReason.PUNCTUATION_DISAGREEMENT.value)
    return reasons
