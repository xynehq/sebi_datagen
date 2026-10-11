"""Build one candidate row from the predictions for a crop.

``consensus_text`` is a proposal for a later reviewer. It is not gold.
``confidence_score`` and ``risk_score`` stay empty here; ranking is a later stage.
The comparison key is never stored as the transcription.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence

from indic_ocr_pipeline.ids import make_candidate_id
from indic_ocr_pipeline.schemas import (
    CandidateRecord,
    DisagreementType,
    QualityTier,
    ScoreReason,
    VerificationStatus,
)
from indic_ocr_pipeline.text_policy import normalize_for_comparison


def build_candidate(
    crop: Mapping[str, object],
    predictions: Sequence[Mapping[str, object]],
    backends: Sequence[str],
) -> CandidateRecord:
    """Return the candidate for one crop. Every named backend must have a row."""
    by_backend = {str(row["backend"]): row for row in predictions}
    texts: dict[str, str | None] = {}
    for name in backends:
        row = by_backend.get(name)
        if row is None or row.get("status") != "complete":
            texts[name] = None
            continue
        value = row.get("text")
        texts[name] = value if isinstance(value, str) else None
    present = [(name, text) for name, text in texts.items() if text is not None]
    exact_counts: dict[str, int] = {}
    for _name, text in present:
        exact_counts[text] = exact_counts.get(text, 0) + 1
    exact_agreement = max(exact_counts.values(), default=0)
    text_of = dict(present)
    names = [name for name, _text in present]
    pairwise: dict[str, bool] = {}
    normalized: dict[str, bool] = {}
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            key = f"{left_name}|{right_name}"
            pairwise[key] = text_of[left_name] == text_of[right_name]
            normalized[key] = normalize_for_comparison(
                text_of[left_name]
            ) == normalize_for_comparison(text_of[right_name])
    agreement = _agreement(present)
    reasons = _reasons(texts, present, agreement)
    return CandidateRecord(
        candidate_id=make_candidate_id(str(crop["crop_id"])),
        crop_id=str(crop["crop_id"]),
        document_id=str(crop["document_id"]),
        page_id=str(crop["page_id"]),
        parent_line_id=str(crop["parent_line_id"]),
        texts_by_backend=texts,
        exact_agreement_count=exact_agreement,
        backends_present=len(present),
        pairwise_agreement=pairwise,
        normalized_agreement=normalized,
        consensus_text=agreement[0],
        disagreement_type=agreement[1],
        confidence_score=None,
        risk_score=None,
        reasons=reasons,
        verification_status=VerificationStatus.CANDIDATE,
        tier=QualityTier.TIER_2_CANDIDATE,
    )


def _agreement(
    present: Sequence[tuple[str, str]],
) -> tuple[str | None, DisagreementType | None]:
    if not present:
        return None, DisagreementType.EMPTY
    texts = [text for _name, text in present]
    if len(set(texts)) == 1:
        return texts[0], DisagreementType.NONE
    keys = [normalize_for_comparison(text) for text in texts]
    if len(set(keys)) == 1:
        normalized = [unicodedata.normalize("NFC", text).strip() for text in texts]
        kind = (
            DisagreementType.WHITESPACE
            if len(set(normalized)) > 1
            else DisagreementType.UNICODE_NORMALIZATION
        )
        return None, kind
    if len({_digits(text) for text in texts}) > 1:
        return None, DisagreementType.NUMERAL
    if len({normalize_for_comparison(_without_punctuation(text)) for text in texts}) == 1:
        return None, DisagreementType.PUNCTUATION
    return None, DisagreementType.CHARACTER_SUBSTITUTION


def _reasons(
    texts: Mapping[str, str | None],
    present: Sequence[tuple[str, str]],
    agreement: tuple[str | None, DisagreementType | None],
) -> list[str]:
    reasons: list[str] = []
    if any(text == "" for text in texts.values() if text is not None):
        reasons.append(ScoreReason.EMPTY_OUTPUT.value)
    if not present:
        return reasons
    if agreement[1] is DisagreementType.NONE and all(text is not None for text in texts.values()):
        reasons.append(ScoreReason.ALL_BACKENDS_AGREE.value)
    elif agreement[1] is DisagreementType.NONE:
        reasons.append(ScoreReason.PARTIAL_AGREEMENT.value)
    else:
        reasons.append(ScoreReason.BACKENDS_DISAGREE.value)
    return reasons


def _digits(text: str) -> str:
    return "".join(character for character in text if character.isdecimal())


def _without_punctuation(text: str) -> str:
    return "".join(
        character
        for character in text
        if not unicodedata.category(character).startswith("P")
    )
