"""One candidate per crop from IndicOCR, Chitrapathak, and Tesseract.

Agreement is a consensus label. It is not gold. Only a later human decision
can produce HUMAN_VERIFIED or HUMAN_CORRECTED.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

from indic_ocr_pipeline.ocr.sensitive import line_is_sensitive
from indic_ocr_pipeline.ocr.similarity import pairwise_similarity, similarity
from indic_ocr_pipeline.text_policy import normalize_for_comparison

BACKENDS = ("indicocr", "chitrapathak", "tesseract")

AUTO_3 = "AUTO_3_AGREE"
AUTO_2 = "AUTO_2_AGREE"
HUMAN = "HUMAN_REVIEW"

ALL_3 = "ALL_3_AGREE"
TWO = "TWO_AGREE"
DIFFER = "ALL_DIFFER"
EMPTY = "EMPTY"
BAD_BOUNDARY = "BAD_BOUNDARY"


def classify_consensus(
    texts: Mapping[str, str | None],
    *,
    boundary_valid: bool = True,
    backends: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Classify one crop. Every backend is stored, including dissent and empties."""
    names = backends if backends is not None else BACKENDS
    stored = {name: _as_text(texts.get(name)) for name in names}
    if not boundary_valid:
        return _row(BAD_BOUNDARY, HUMAN, True, None, stored, "BAD_BOUNDARY")
    if any(stored[name] is None or stored[name] == "" for name in names):
        return _row(EMPTY, HUMAN, True, None, stored, "EMPTY")
    raw = [str(stored[name]) for name in names]
    keys = [normalize_for_comparison(text) for text in raw]
    counts = Counter(keys)
    size = len(names)
    if len(counts) == 1:
        label, status, consensus = _unanimous(size)
        return _row(label, status, False, raw[0], stored, consensus)
    majority = max(counts.values())
    if majority > size / 2 and majority < size:
        majority_key = next(key for key, count in counts.items() if count == majority)
        proposed = next(text for text, key in zip(raw, keys, strict=True) if key == majority_key)
        label, status, consensus = _majority(size, majority)
        return _row(label, status, False, proposed, stored, consensus)
    label = DIFFER if size == 3 else "NO_CONSENSUS"
    return _row(label, HUMAN, True, None, stored, "NO_CONSENSUS")


def _as_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return None


def _unanimous(size: int) -> tuple[str, str, str]:
    if size == 3:
        return ALL_3, AUTO_3, "CONSENSUS_3_3"
    # Two-model agreement is not a three-model unanimous result.
    return f"CONSENSUS_{size}_{size}", f"AUTO_{size}_{size}", f"CONSENSUS_{size}_{size}"


def _majority(size: int, count: int) -> tuple[str, str, str]:
    if size == 3 and count == 2:
        return TWO, AUTO_2, "CONSENSUS_2_3"
    return f"CONSENSUS_{count}_{size}", f"AUTO_{count}_{size}", f"CONSENSUS_{count}_{size}"


def _row(
    agreement_class: str,
    status: str,
    review_required: bool,
    proposed_text: str | None,
    texts: Mapping[str, str | None],
    consensus_status: str,
) -> dict[str, object]:
    comparable = {name: text or "" for name, text in texts.items()}
    normalized = {name: normalize_for_comparison(text) for name, text in comparable.items()}
    sensitive = line_is_sensitive(texts)
    agreed, differed, differing = _agreement_detail(normalized, proposed_text)
    candidate, data_status, reason = _label_status(
        agreement_class,
        consensus_status,
        configured=len(texts),
        agreed=agreed,
        differed=differed,
        sensitive=sensitive,
    )
    return {
        "agreement_class": agreement_class,
        "consensus_status": consensus_status,
        "status": status,
        "data_status": data_status,
        "acceptance_candidate": candidate,
        "review_required": review_required or sensitive,
        "proposed_text": proposed_text,
        "predictions": dict(texts),
        "pairwise_similarity": pairwise_similarity(normalized),
        "models_configured": len(texts),
        "models_agreed": agreed,
        "models_differed": differed,
        "differing_similarity": differing,
        "decision_reason": reason,
        "sensitive": sensitive,
        "stricter_reliability": len(texts) == 2 and candidate == "auto_accepted_unanimous",
        "gold": False,
    }


def _agreement_detail(
    normalized: Mapping[str, str],
    proposed_text: str | None,
) -> tuple[int, int, dict[str, float]]:
    size = len(normalized)
    if proposed_text is None:
        return 0, size, {}
    proposed_key = normalize_for_comparison(proposed_text)
    agreed = sum(1 for key in normalized.values() if key and key == proposed_key)
    differing = {
        name: round(similarity(key, proposed_key), 4)
        for name, key in normalized.items()
        if key != proposed_key
    }
    return agreed, size - agreed, differing


def _is_unanimous(agreement_class: str, consensus_status: str) -> bool:
    if agreement_class == ALL_3:
        return True
    parts = consensus_status.split("_")
    return len(parts) == 3 and parts[0] == "CONSENSUS" and parts[1] == parts[2]


def _label_status(
    agreement_class: str,
    consensus_status: str,
    *,
    configured: int,
    agreed: int,
    differed: int,
    sensitive: bool,
) -> tuple[str | None, str, str]:
    """Return the acceptance candidate, the current data_status, and the reason.

    Exact agreement is only a candidate. It stays out of the auto-accepted set
    until a human audit clears the category. ``gold`` is never set here.
    """
    if sensitive:
        return (
            None,
            "needs_human_review",
            "This line contains a number, amount, date, identifier, email, or URL. "
            "It is not auto-accepted, even when every model agrees.",
        )
    if agreement_class in {EMPTY, BAD_BOUNDARY} or consensus_status in {"EMPTY", "BAD_BOUNDARY"}:
        return (
            None,
            "needs_human_review",
            "The line is empty or the boundary is bad, so nobody accepts it automatically.",
        )
    if _is_unanimous(agreement_class, consensus_status):
        scope = f"all {configured} configured models"
        extra = ""
        if configured == 2:
            extra = (
                " Agreement between 2 models is not a 3-model result, so calibration is stricter."
            )
        return (
            "auto_accepted_unanimous",
            "pending_calibration",
            f"{scope.capitalize()} produced the same comparison key. "
            "This can enter the auto-accepted set only after that category's "
            "human audit clears the configured lower bound. It is not human-verified."
            + extra,
        )
    if agreed >= 2 and differed > 0:
        return (
            "auto_accepted_majority",
            "needs_human_review",
            f"{agreed} of {configured} models share the proposed text and {differed} differ. "
            "That count is not a probability. The line stays in review until this "
            "category's measured lower bound clears the threshold.",
        )
    return (
        None,
        "needs_human_review",
        "The models do not agree. Every output stays visible for a person to review. "
        "Nothing here is accepted because one string looks more plausible.",
    )
