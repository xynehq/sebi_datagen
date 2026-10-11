"""Phase 9 calibration analysis.

Reads completed human decisions and Phase 7 diagnostics. Does not mutate
manifests, enable auto-approval, export gold, run OCR, or load models.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from indic_ocr_pipeline.errors import DatasetError
from indic_ocr_pipeline.schemas import ReviewDecision, VerificationStatus
from indic_ocr_pipeline.scoring.metrics import proposed_consensus
from indic_ocr_pipeline.text_policy import (
    apply_storage_normalization,
    normalize_for_comparison,
)

FIRST_CALIBRATION_BATCH_ID = "calibration_3407_600"
SECOND_CALIBRATION_BATCH_ID = "calibration_3407_twoagree_v1"

STRICT_TWO_AGREE_FILTER_DEFINITION = (
    "exactly two engines agree (TWO_AGREE_ONE_DIFFERS) "
    "AND no prediction is empty "
    "AND boundary is valid "
    "AND majority agreement is exact after comparison normalization "
    "AND no unicode_anomaly flag "
    "AND no large_length_disagreement flag"
)

OUTCOME_LABELS = (
    "human_verified",
    "human_corrected",
    "rejected",
    "bad_boundary",
)

AGREEMENT_CATEGORIES = (
    "ALL_3_AGREE",
    "TWO_AGREE_ONE_DIFFERS",
    "ALL_3_DIFFER",
    "ONE_OR_MORE_EMPTY",
    "PREDICTION_ERROR",
)

GEOMETRY_FLAGS = frozenset(
    {
        "empty_prediction",
        "large_length_disagreement",
        "unicode_anomaly",
        "likely_truncation",
        "likely_extra_text",
    }
)


class CalibrationAnalysisError(DatasetError):
    """Calibration analysis could not be completed."""


def decision_label_of(verified: Mapping[str, Any]) -> str:
    """Return the UI decision label for one verified row."""
    meta = verified.get("metadata") if isinstance(verified.get("metadata"), dict) else {}
    label = meta.get("decision_label")
    if isinstance(label, str) and label in OUTCOME_LABELS:
        return label
    status = verified.get("verification_status")
    decision = verified.get("review_decision")
    if decision == ReviewDecision.BAD_BOUNDARY.value:
        return "bad_boundary"
    if status == VerificationStatus.HUMAN_VERIFIED.value:
        return "human_verified"
    if status == VerificationStatus.HUMAN_CORRECTED.value:
        return "human_corrected"
    if status == VerificationStatus.REJECTED.value:
        return "rejected"
    raise CalibrationAnalysisError(
        f"cannot classify verified row {verified.get('crop_id')!r}"
    )


def is_boundary_valid(decision_label: str) -> bool:
    """Geometry outcome. Bad-boundary rows are invalid; others are valid crops."""
    return decision_label != "bad_boundary"


def is_candidate_exact(decision_label: str) -> bool:
    """True only when a human accepted an existing candidate string exactly."""
    return decision_label == "human_verified"


def classify_reviewed_row(
    verified: Mapping[str, Any],
    *,
    review: Mapping[str, Any] | None = None,
    crop: Mapping[str, Any] | None = None,
    region: Mapping[str, Any] | None = None,
    candidate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize one reviewed sample for stratified analysis."""
    label = decision_label_of(verified)
    diagnostics = {}
    if review is not None and isinstance(review.get("diagnostics"), dict):
        diagnostics = review["diagnostics"]
    meta = verified.get("metadata") if isinstance(verified.get("metadata"), dict) else {}
    agreement = (
        meta.get("agreement_category")
        or diagnostics.get("agreement_category")
        or "unknown"
    )
    tier = meta.get("review_priority") or diagnostics.get("review_priority") or "unknown"
    flags = list(diagnostics.get("flags") or [])
    feature_risks = diagnostics.get("feature_risks") or {}
    tess_conf = diagnostics.get("tesseract_confidence")
    texts = verified.get("original_candidate_texts") or {}
    if not isinstance(texts, dict):
        texts = {}
    bbox = None if crop is None else crop.get("bbox")
    width, height = _bbox_size(bbox)
    consensus = diagnostics.get("proposed_consensus")
    if consensus is None:
        consensus = proposed_consensus(
            {
                "indicocr": texts.get("indicocr"),
                "chitrapathak": texts.get("chitrapathak"),
                "tesseract": texts.get("tesseract"),
            }
        )
    boundary_valid = is_boundary_valid(label)
    candidate_exact = is_candidate_exact(label)
    return {
        "crop_id": verified["crop_id"],
        "document_id": verified["document_id"],
        "page_id": verified["page_id"],
        "parent_line_id": verified["parent_line_id"],
        "decision_label": label,
        "boundary_valid": boundary_valid,
        "boundary_invalid": not boundary_valid,
        "candidate_exact": candidate_exact,
        "candidate_corrected": label == "human_corrected",
        "candidate_rejected": label == "rejected",
        "ocr_correctness_unknown": label == "bad_boundary",
        "agreement_category": str(agreement),
        "review_priority": str(tier),
        "flags": flags,
        "geometry_flag_present": bool(GEOMETRY_FLAGS.intersection(flags)),
        "empty_present": "empty_prediction" in flags
        or str(agreement) == "ONE_OR_MORE_EMPTY",
        "large_length_disagreement": "large_length_disagreement" in flags,
        "unicode_anomaly": "unicode_anomaly" in flags,
        "tesseract_confidence": tess_conf if isinstance(tess_conf, int | float) else None,
        "tesseract_confidence_bucket": _confidence_bucket(tess_conf),
        "crop_geometry_risk": _risk_value(feature_risks, "crop_geometry"),
        "geometry_risk_bucket": _geometry_risk_bucket(
            _risk_value(feature_risks, "crop_geometry")
        ),
        "region_type": None if region is None else region.get("region_type"),
        "table_hint": _table_hint(region),
        "line_length": _line_length(texts, consensus, verified.get("final_text")),
        "line_length_bucket": _length_bucket(
            _line_length(texts, consensus, verified.get("final_text"))
        ),
        "crop_width": width,
        "crop_height": height,
        "crop_height_bucket": _height_bucket(height),
        "candidate_status": None if candidate is None else candidate.get("verification_status"),
        "final_text": verified.get("final_text"),
        "original_candidate_texts": {
            "indicocr": texts.get("indicocr"),
            "chitrapathak": texts.get("chitrapathak"),
            "tesseract": texts.get("tesseract"),
        },
        "proposed_consensus": consensus,
        "error_tags": list(meta.get("error_tags") or []),
        "notes": verified.get("notes"),
        "risk_score": review.get("risk_score") if review is not None else None,
        "passes_strict_two_agree_rule": False,
    }


def annotate_strict_two_agree_rule(row: Mapping[str, Any]) -> bool:
    """Analytical candidate rule. Does not approve anything."""
    if row.get("agreement_category") != "TWO_AGREE_ONE_DIFFERS":
        return False
    if row.get("empty_present"):
        return False
    if row.get("unicode_anomaly"):
        return False
    if row.get("large_length_disagreement"):
        return False
    if row.get("boundary_invalid"):
        return False
    texts = row.get("original_candidate_texts") or {}
    present = [
        text
        for text in (
            texts.get("indicocr"),
            texts.get("chitrapathak"),
            texts.get("tesseract"),
        )
        if isinstance(text, str) and text != ""
    ]
    if len(present) < 2:
        return False
    keys = [normalize_for_comparison(text) for text in present]
    counts = Counter(keys)
    if counts.most_common(1)[0][1] < 2:
        return False
    return True


def build_reviewed_rows(
    *,
    batch_items: Sequence[Mapping[str, Any]],
    verified_rows: Sequence[Mapping[str, Any]],
    review_rows: Sequence[Mapping[str, Any]],
    crop_rows: Sequence[Mapping[str, Any]] | None = None,
    region_rows: Sequence[Mapping[str, Any]] | None = None,
    candidate_rows: Sequence[Mapping[str, Any]] | None = None,
    batch_id: str = FIRST_CALIBRATION_BATCH_ID,
) -> list[dict[str, Any]]:
    """Join one calibration batch with verified decisions and diagnostics."""
    batch_crop_ids = {
        str(item["crop_id"])
        for item in batch_items
        if str(item.get("batch_id")) == batch_id
    }
    if not batch_crop_ids:
        raise CalibrationAnalysisError(f"no batch items for {batch_id!r}")
    verified_by_crop = {str(row["crop_id"]): row for row in verified_rows}
    review_by_crop = {str(row["crop_id"]): row for row in review_rows}
    crops = {str(row["crop_id"]): row for row in (crop_rows or [])}
    regions = {str(row["region_id"]): row for row in (region_rows or [])}
    candidates = {str(row["crop_id"]): row for row in (candidate_rows or [])}
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for crop_id in sorted(batch_crop_ids):
        verified = verified_by_crop.get(crop_id)
        if verified is None:
            missing.append(crop_id)
            continue
        review = review_by_crop.get(crop_id)
        crop = crops.get(crop_id)
        region_id = None
        if review is not None and isinstance(review.get("diagnostics"), dict):
            region_id = review["diagnostics"].get("region_id")
        if region_id is None and crop is not None:
            region_id = crop.get("region_id")
        region = regions.get(str(region_id)) if region_id else None
        row = classify_reviewed_row(
            verified,
            review=review,
            crop=crop,
            region=region,
            candidate=candidates.get(crop_id),
        )
        row["passes_strict_two_agree_rule"] = annotate_strict_two_agree_rule(row)
        rows.append(row)
    if missing:
        raise CalibrationAnalysisError(
            f"{len(missing)} batch rows lack verified decisions; first={missing[0]}"
        )
    return rows


def outcome_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count the four human outcomes. Do not collapse them."""
    counts = {label: 0 for label in OUTCOME_LABELS}
    for row in rows:
        counts[str(row["decision_label"])] += 1
    return counts


def rates_from_counts(counts: Mapping[str, int]) -> dict[str, Any]:
    """Attach rates to an outcome count dict."""
    reviewed = sum(int(counts.get(label, 0)) for label in OUTCOME_LABELS)
    exact = int(counts.get("human_verified", 0))
    corrected = int(counts.get("human_corrected", 0))
    rejected = int(counts.get("rejected", 0))
    bad = int(counts.get("bad_boundary", 0))
    boundary_valid = reviewed - bad
    return {
        "reviewed": reviewed,
        "human_verified": exact,
        "human_corrected": corrected,
        "rejected": rejected,
        "bad_boundary": bad,
        "boundary_valid": boundary_valid,
        "boundary_invalid": bad,
        "candidate_exact": exact,
        "candidate_exact_rate": _rate(exact, reviewed),
        "correction_rate": _rate(corrected, reviewed),
        "rejection_rate": _rate(rejected, reviewed),
        "boundary_failure_rate": _rate(bad, reviewed),
        "candidate_exact_among_boundary_valid": exact,
        "candidate_corrected_among_boundary_valid": corrected,
        "candidate_rejected_among_boundary_valid": rejected,
        "candidate_exact_rate_among_boundary_valid": _rate(exact, boundary_valid),
        "correction_rate_among_boundary_valid": _rate(corrected, boundary_valid),
        "rejection_rate_among_boundary_valid": _rate(rejected, boundary_valid),
    }


def stratify(
    rows: Sequence[Mapping[str, Any]],
    key: str,
) -> dict[str, dict[str, Any]]:
    """Outcome breakdown for one categorical key."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, "unknown"))].append(row)
    report: dict[str, dict[str, Any]] = {}
    for name in sorted(grouped):
        counts = outcome_counts(grouped[name])
        report[name] = rates_from_counts(counts)
    return report


def agreement_boundary_report(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Agreement categories with boundary-valid transcription metrics."""
    return stratify(rows, "agreement_category")


def shared_error_all3(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """ALL_3_AGREE rows where humans still corrected or rejected."""
    agree = [row for row in rows if row.get("agreement_category") == "ALL_3_AGREE"]
    counts = outcome_counts(agree)
    corrected = [row for row in agree if row["decision_label"] == "human_corrected"]
    rejected = [row for row in agree if row["decision_label"] == "rejected"]
    tag_counts: Counter[str] = Counter()
    char_buckets: Counter[str] = Counter()
    for row in corrected + rejected:
        tags = row.get("error_tags") or []
        if tags:
            tag_counts.update(str(tag) for tag in tags)
        else:
            char_buckets["untagged"] += 1
            continue
        for tag in tags:
            char_buckets[_tag_to_char_bucket(str(tag))] += 1
    n = len(agree)
    return {
        "n_reviewed": n,
        "exact_agreement": counts["human_verified"],
        "agreement_but_human_corrected": counts["human_corrected"],
        "agreement_but_rejected": counts["rejected"],
        "agreement_but_bad_boundary": counts["bad_boundary"],
        "shared_error_rate_corrected": _rate(counts["human_corrected"], n),
        "shared_error_rate_corrected_or_rejected": _rate(
            counts["human_corrected"] + counts["rejected"], n
        ),
        "error_tags": dict(tag_counts.most_common()),
        "character_error_buckets": dict(char_buckets.most_common()),
        "examples": [
            {
                "crop_id": row["crop_id"],
                "consensus": row.get("proposed_consensus"),
                "gold_text": row.get("final_text"),
                "tags": row.get("error_tags") or [],
                "notes": row.get("notes"),
            }
            for row in corrected[:20]
        ],
    }


def two_agree_investigation(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Inspect whether a narrow TWO_AGREE acceptance rule looks promising."""
    two = [row for row in rows if row.get("agreement_category") == "TWO_AGREE_ONE_DIFFERS"]
    counts = outcome_counts(two)
    rates = rates_from_counts(counts)
    strict = [row for row in two if row.get("passes_strict_two_agree_rule")]
    strict_counts = outcome_counts(strict)
    strict_rates = rates_from_counts(strict_counts)
    error_clusters = _cluster_errors(
        [row for row in two if row["decision_label"] in {"human_corrected", "rejected"}]
    )
    boundary_failures = [row for row in two if row["decision_label"] == "bad_boundary"]
    return {
        "n_reviewed": len(two),
        "outcomes": counts,
        "rates": rates,
        "boundary_failures": len(boundary_failures),
        "candidate_corrections": counts["human_corrected"],
        "candidate_rejections": counts["rejected"],
        "error_clusters": error_clusters,
        "strict_rule": {
            "definition": (
                "TWO_AGREE_ONE_DIFFERS + no empty + no unicode_anomaly + "
                "no large_length_disagreement + boundary_valid + "
                "exact agreement after comparison normalization"
            ),
            "n_matching_reviewed": len(strict),
            "outcomes": strict_counts,
            "rates": strict_rates,
            "auto_approval_enabled": False,
            "recommendation": _two_agree_recommendation(rates, strict_rates, len(two)),
        },
    }


def evaluate_unsafe_policies(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Show why broad auto-approval policies are unsafe. Does not enable them."""
    all3 = [row for row in rows if row.get("agreement_category") == "ALL_3_AGREE"]
    p3 = [row for row in rows if row.get("review_priority") == "P3"]
    return {
        "all_3_agree_auto_approve": {
            "enabled": False,
            "n": len(all3),
            "rates": rates_from_counts(outcome_counts(all3)),
            "safe": False,
            "reason": "calibration shows substantial human_corrected rate under 3/3 agreement",
        },
        "low_risk_p3_auto_approve": {
            "enabled": False,
            "n": len(p3),
            "rates": rates_from_counts(outcome_counts(p3)),
            "safe": False,
            "reason": "P3 still contains shared OCR character errors",
        },
    }


def data_quality_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Audit duplicates and coverage. Does not rebalance data."""
    texts = [row.get("final_text") for row in rows if isinstance(row.get("final_text"), str)]
    exact_dupes = _duplicate_values(texts)
    normalized = [
        normalize_for_comparison(text)
        for text in texts
        if isinstance(text, str) and text
    ]
    norm_dupes = _duplicate_values(normalized)
    skeletons = [_lexical_skeleton(text) for text in texts if isinstance(text, str)]
    skeleton_dupes = _duplicate_values([item for item in skeletons if item])
    number_norm = [_number_normalized(text) for text in texts if isinstance(text, str)]
    number_dupes = _duplicate_values([item for item in number_norm if item])
    return {
        "eligible_text_rows": len(texts),
        "exact_duplicate_text_values": len(exact_dupes),
        "normalized_duplicate_text_values": len(norm_dupes),
        "number_normalized_duplicate_values": len(number_dupes),
        "lexical_skeleton_duplicate_values": len(skeleton_dupes),
        "documents": dict(Counter(str(row["document_id"]) for row in rows)),
        "region_types": dict(Counter(str(row.get("table_hint")) for row in rows)),
        "line_length_buckets": dict(Counter(str(row.get("line_length_bucket")) for row in rows)),
        "crop_height_buckets": dict(Counter(str(row.get("crop_height_bucket")) for row in rows)),
    }


def risk_ordering_usefulness(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Quantify whether P0–P3 still separate geometry vs transcription risk."""
    by_tier = stratify(rows, "review_priority")
    geometry_signal = {
        tier: by_tier.get(tier, {}).get("boundary_failure_rate", 0.0)
        for tier in ("P0", "P1", "P2", "P3")
        if tier in by_tier
    }
    transcription_signal = {
        tier: by_tier.get(tier, {}).get("correction_rate_among_boundary_valid", 0.0)
        for tier in ("P0", "P1", "P2", "P3")
        if tier in by_tier
    }
    return {
        "boundary_failure_rate_by_tier": geometry_signal,
        "correction_rate_among_boundary_valid_by_tier": transcription_signal,
        "weights_unchanged": True,
        "weights": {
            "agreement": 0.45,
            "tesseract_confidence": 0.15,
            "image_quality": 0.15,
            "crop_geometry": 0.10,
            "unicode_sanity": 0.15,
        },
        "interpretation": (
            "Risk tiers remain directionally useful for geometry failures "
            "(higher in P0/P1). They are not sufficient for transcription "
            "correctness under unanimous OCR agreement."
        ),
        "rewrite_weights": False,
    }


def evaluate_strict_two_agree_filter(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Score the fixed strict TWO_AGREE filter against human outcomes.

    The filter definition does not use the human decision. False accepts are
    rows the filter would accept whose consensus text is not the human gold.
    """
    population = [
        row for row in rows if row.get("agreement_category") == "TWO_AGREE_ONE_DIFFERS"
    ]
    matching = []
    true_accept = 0
    false_accept = 0
    examples: list[dict[str, Any]] = []
    for row in population:
        item = dict(row)
        passes = annotate_strict_two_agree_rule(item)
        item["passes_strict_two_agree_rule"] = passes
        if not passes:
            continue
        matching.append(item)
        consensus = item.get("proposed_consensus")
        gold = item.get("final_text")
        label = item.get("decision_label")
        if (
            label == "human_verified"
            and isinstance(consensus, str)
            and isinstance(gold, str)
            and apply_storage_normalization(consensus) == apply_storage_normalization(gold)
        ):
            true_accept += 1
            continue
        false_accept += 1
        if len(examples) < 20:
            examples.append(
                {
                    "crop_id": item.get("crop_id"),
                    "decision_label": label,
                    "consensus": consensus,
                    "gold_text": gold,
                    "tags": item.get("error_tags") or [],
                }
            )
    n = len(matching)
    return {
        "filter_definition": STRICT_TWO_AGREE_FILTER_DEFINITION,
        "filter_depends_on_human_outcome": False,
        "two_agree_population": len(population),
        "n_matching_reviewed": n,
        "candidate_exact": true_accept,
        "false_accept_count": false_accept,
        "empirical_exact_rate": _rate(true_accept, n),
        "outcomes_among_matching": outcome_counts(matching),
        "false_accept_examples": examples,
        "auto_approval_enabled": False,
    }


def classify_policy_strength(
    *,
    first_strict: Mapping[str, Any],
    second_strict: Mapping[str, Any],
    combined_strict: Mapping[str, Any],
    source_documents: Sequence[str],
) -> dict[str, Any]:
    """Classify the strict TWO_AGREE rule without enabling it."""
    false_accepts = int(combined_strict.get("false_accept_count") or 0)
    n = int(combined_strict.get("n_matching_reviewed") or 0)
    docs = sorted({str(doc) for doc in source_documents if doc})
    single_doc = len(docs) <= 1
    if n < 50:
        category = "NOT_SUPPORTED"
        reason = "filtered sample too small for a dataset-quality acceptance rule"
    elif false_accepts > 0 and single_doc:
        category = "PROMISING_BUT_NEEDS_MORE_DATA"
        reason = (
            f"combined filtered n={n} with false_accept_count={false_accepts}; "
            "shared OCR errors still occur under TWO_AGREE, and the second batch "
            "is not document-independent"
        )
    elif false_accepts == 0 and single_doc:
        category = "PROMISING_BUT_NEEDS_MORE_DATA"
        reason = (
            f"combined filtered n={n} shows zero false accepts in-sample, but "
            "evidence is concentrated in one source document and cannot support "
            "a universal auto-approval claim"
        )
    elif false_accepts == 0 and n >= 200:
        category = "STRONG_CANDIDATE_FOR_LIMITED_AUTO_APPROVAL"
        reason = (
            "large multi-document filtered sample with zero observed false accepts; "
            "still requires random human audit before any enablement"
        )
    else:
        category = "PROMISING_BUT_NEEDS_MORE_DATA"
        reason = (
            f"combined filtered n={n}, false_accept_count={false_accepts}; "
            "more diverse calibration is required"
        )
    return {
        "category": category,
        "reason": reason,
        "source_documents": docs,
        "document_independent": not single_doc,
        "auto_approval_enabled": False,
        "gold_export": False,
        "required_safeguards_if_ever_enabled": [
            "random human audit of auto-accepted rows",
            "exclude table/geometry anomaly flags",
            "exclude empty predictions and length anomalies",
            "document-diverse confirmation sample",
            "never write gold without human_verified/human_corrected status",
        ],
        "first_batch_strict": {
            "n": first_strict.get("n_matching_reviewed"),
            "false_accept_count": first_strict.get("false_accept_count"),
            "empirical_exact_rate": first_strict.get("empirical_exact_rate"),
        },
        "second_batch_strict": {
            "n": second_strict.get("n_matching_reviewed"),
            "false_accept_count": second_strict.get("false_accept_count"),
            "empirical_exact_rate": second_strict.get("empirical_exact_rate"),
        },
        "combined_strict": {
            "n": combined_strict.get("n_matching_reviewed"),
            "false_accept_count": combined_strict.get("false_accept_count"),
            "empirical_exact_rate": combined_strict.get("empirical_exact_rate"),
        },
    }


def build_phase9_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    batch_id: str = FIRST_CALIBRATION_BATCH_ID,
    source_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Full Phase 9 analytical report. No policy activation."""
    annotated = []
    for row in rows:
        item = dict(row)
        item["passes_strict_two_agree_rule"] = annotate_strict_two_agree_rule(item)
        annotated.append(item)
    overall = rates_from_counts(outcome_counts(annotated))
    two = two_agree_investigation(annotated)
    shared = shared_error_all3(annotated)
    strict = evaluate_strict_two_agree_filter(annotated)
    documents = sorted({str(row.get("document_id")) for row in annotated})
    return {
        "phase": 9,
        "batch_id": batch_id,
        "auto_approval_enabled": False,
        "gold_export": False,
        "ocr_invoked": False,
        "network_used": False,
        "models_loaded": False,
        "source_manifests_mutated": False,
        "reviewed_total": len(annotated),
        "source_documents": documents,
        "source_distribution_limitation": (
            "single_document_batch"
            if len(documents) <= 1
            else "multi_document_batch"
        ),
        "overall": overall,
        "by_risk_tier": stratify(annotated, "review_priority"),
        "by_agreement_category": agreement_boundary_report(annotated),
        "by_tesseract_confidence_bucket": stratify(annotated, "tesseract_confidence_bucket"),
        "by_geometry_risk_bucket": stratify(annotated, "geometry_risk_bucket"),
        "by_geometry_flag_present": stratify(
            annotated, "geometry_flag_present"
        ),
        "by_empty_present": stratify(annotated, "empty_present"),
        "by_large_length_disagreement": stratify(annotated, "large_length_disagreement"),
        "by_table_hint": stratify(annotated, "table_hint"),
        "by_document": stratify(annotated, "document_id"),
        "by_line_length_bucket": stratify(annotated, "line_length_bucket"),
        "by_candidate_status": stratify(annotated, "candidate_status"),
        "shared_error_all3": shared,
        "two_agree": two,
        "strict_two_agree_filter": strict,
        "unsafe_policy_checks": evaluate_unsafe_policies(annotated),
        "risk_ordering": risk_ordering_usefulness(annotated),
        "data_quality_audit": data_quality_audit(annotated),
        "source_hashes": dict(source_hashes or {}),
        "policy_conclusion": {
            "two_agree_promising": two["strict_rule"]["recommendation"]["promising"],
            "two_agree_safe_to_enable": False,
            "reason": two["strict_rule"]["recommendation"]["reason"],
            "scoring_weights_unchanged": True,
            "next_step": "collect second focused calibration batch before any acceptance rule",
        },
    }


def build_combined_phase9_report(
    *,
    first_rows: Sequence[Mapping[str, Any]],
    second_rows: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Per-batch plus combined report. Sampling designs stay separate."""
    first = build_phase9_report(
        first_rows,
        batch_id=FIRST_CALIBRATION_BATCH_ID,
        source_hashes=source_hashes,
    )
    second = build_phase9_report(
        second_rows,
        batch_id=SECOND_CALIBRATION_BATCH_ID,
        source_hashes=source_hashes,
    )
    combined_rows = [dict(row) for row in first_rows] + [dict(row) for row in second_rows]
    combined = build_phase9_report(
        combined_rows,
        batch_id="combined_888",
        source_hashes=source_hashes,
    )
    policy = classify_policy_strength(
        first_strict=first["strict_two_agree_filter"],
        second_strict=second["strict_two_agree_filter"],
        combined_strict=combined["strict_two_agree_filter"],
        source_documents=list(second.get("source_documents") or []),
    )
    overlap = {str(row["crop_id"]) for row in first_rows} & {
        str(row["crop_id"]) for row in second_rows
    }
    return {
        "phase": 9,
        "auto_approval_enabled": False,
        "gold_export": False,
        "ocr_invoked": False,
        "network_used": False,
        "models_loaded": False,
        "batch_overlap_count": len(overlap),
        "first_batch": first,
        "second_batch": second,
        "combined": combined,
        "policy_classification": policy,
        "source_distribution_caveat": (
            "Second batch calibration_3407_twoagree_v1 is entirely from MHGR_000003. "
            "It is sample-independent from the first 600 but not document-source "
            "independent. Do not conclude TWO_AGREE is universally safe."
        ),
    }


def format_combined_phase9_report(report: Mapping[str, Any]) -> str:
    """Human-readable combined Phase 9 report."""
    first = report["first_batch"]
    second = report["second_batch"]
    combined = report["combined"]
    policy = report["policy_classification"]
    lines = [
        "Phase 9: combined calibration analysis (600 + 288)",
        "Per-batch statistics are preserved; designs are not pooled blindly.",
        f"auto_approval_enabled: {report.get('auto_approval_enabled')}",
        f"gold_export: {report.get('gold_export')}",
        f"batch_overlap_count: {report.get('batch_overlap_count')}",
        f"source_distribution_caveat: {report.get('source_distribution_caveat')}",
        "",
        "=== first_batch calibration_3407_600 ===",
        format_phase9_report(first).rstrip(),
        "",
        "=== second_batch calibration_3407_twoagree_v1 ===",
        format_phase9_report(second).rstrip(),
        "",
        "=== combined_888 ===",
        format_phase9_report(combined).rstrip(),
        "",
        "=== strict_two_agree_filter comparison ===",
        f"filter_definition: {STRICT_TWO_AGREE_FILTER_DEFINITION}",
        f"first_n: {policy['first_batch_strict']['n']} "
        f"false_accepts: {policy['first_batch_strict']['false_accept_count']} "
        f"exact_rate: {policy['first_batch_strict']['empirical_exact_rate']}%",
        f"second_n: {policy['second_batch_strict']['n']} "
        f"false_accepts: {policy['second_batch_strict']['false_accept_count']} "
        f"exact_rate: {policy['second_batch_strict']['empirical_exact_rate']}%",
        f"combined_n: {policy['combined_strict']['n']} "
        f"false_accepts: {policy['combined_strict']['false_accept_count']} "
        f"exact_rate: {policy['combined_strict']['empirical_exact_rate']}%",
        f"policy_category: {policy['category']}",
        f"policy_reason: {policy['reason']}",
        f"document_independent: {policy['document_independent']}",
        "auto_approval_enabled: False",
        "gold_export: False",
    ]
    return "\n".join(lines) + "\n"


def format_phase9_report(report: Mapping[str, Any]) -> str:
    """Human-readable Phase 9 report."""
    lines = [
        "Phase 9: calibration analysis and acceptance-policy evidence",
        "This report does not enable auto-approval and does not export gold.",
        "Geometry failures are separated from transcription failures.",
        f"batch_id: {report.get('batch_id')}",
        f"reviewed_total: {report.get('reviewed_total')}",
        f"auto_approval_enabled: {report.get('auto_approval_enabled')}",
        f"gold_export: {report.get('gold_export')}",
        "",
        "overall:",
    ]
    lines.extend(_format_rate_block(report["overall"], indent="  "))
    for section, title in (
        ("by_risk_tier", "by_risk_tier"),
        ("by_agreement_category", "by_agreement_category"),
        ("by_tesseract_confidence_bucket", "by_tesseract_confidence_bucket"),
        ("by_geometry_risk_bucket", "by_geometry_risk_bucket"),
        ("by_empty_present", "by_empty_present"),
        ("by_large_length_disagreement", "by_large_length_disagreement"),
        ("by_table_hint", "by_table_hint"),
        ("by_document", "by_document"),
        ("by_line_length_bucket", "by_line_length_bucket"),
        ("by_candidate_status", "by_candidate_status"),
    ):
        lines.append("")
        lines.append(f"{title}:")
        for key, bucket in (report.get(section) or {}).items():
            lines.append(f"  {key}:")
            lines.extend(_format_rate_block(bucket, indent="    "))
    shared = report.get("shared_error_all3") or {}
    lines.extend(
        [
            "",
            "shared_error_all3:",
            f"  n_reviewed: {shared.get('n_reviewed')}",
            f"  exact_agreement: {shared.get('exact_agreement')}",
            f"  agreement_but_human_corrected: {shared.get('agreement_but_human_corrected')}",
            f"  agreement_but_rejected: {shared.get('agreement_but_rejected')}",
            f"  agreement_but_bad_boundary: {shared.get('agreement_but_bad_boundary')}",
            f"  shared_error_rate_corrected: {shared.get('shared_error_rate_corrected')}%",
            f"  shared_error_rate_corrected_or_rejected: "
            f"{shared.get('shared_error_rate_corrected_or_rejected')}%",
            f"  character_error_buckets: {shared.get('character_error_buckets')}",
        ]
    )
    two = report.get("two_agree") or {}
    strict = two.get("strict_rule") or {}
    lines.extend(
        [
            "",
            "two_agree_investigation:",
            f"  n_reviewed: {two.get('n_reviewed')}",
            f"  boundary_failures: {two.get('boundary_failures')}",
            f"  candidate_corrections: {two.get('candidate_corrections')}",
            f"  candidate_rejections: {two.get('candidate_rejections')}",
            f"  exact_rate: {(two.get('rates') or {}).get('candidate_exact_rate')}%",
            f"  boundary_failure_rate: {(two.get('rates') or {}).get('boundary_failure_rate')}%",
            f"  exact_rate_among_boundary_valid: "
            f"{(two.get('rates') or {}).get('candidate_exact_rate_among_boundary_valid')}%",
            f"  error_clusters: {two.get('error_clusters')}",
            f"  strict_rule_n: {strict.get('n_matching_reviewed')}",
            f"  strict_rule_exact_rate: {(strict.get('rates') or {}).get('candidate_exact_rate')}%",
            f"  strict_rule_recommendation: {(strict.get('recommendation') or {}).get('reason')}",
            "  auto_approval_enabled: False",
        ]
    )
    risk = report.get("risk_ordering") or {}
    lines.extend(
        [
            "",
            "risk_ordering:",
            f"  boundary_failure_rate_by_tier: {risk.get('boundary_failure_rate_by_tier')}",
            f"  correction_rate_among_boundary_valid_by_tier: "
            f"{risk.get('correction_rate_among_boundary_valid_by_tier')}",
            f"  rewrite_weights: {risk.get('rewrite_weights')}",
            f"  interpretation: {risk.get('interpretation')}",
        ]
    )
    strict = report.get("strict_two_agree_filter") or {}
    lines.extend(
        [
            "",
            "strict_two_agree_filter:",
            f"  definition: {strict.get('filter_definition')}",
            f"  filter_depends_on_human_outcome: {strict.get('filter_depends_on_human_outcome')}",
            f"  two_agree_population: {strict.get('two_agree_population')}",
            f"  n_matching_reviewed: {strict.get('n_matching_reviewed')}",
            f"  candidate_exact: {strict.get('candidate_exact')}",
            f"  false_accept_count: {strict.get('false_accept_count')}",
            f"  empirical_exact_rate: {strict.get('empirical_exact_rate')}%",
        ]
    )
    conclusion = report.get("policy_conclusion") or {}
    lines.extend(
        [
            "",
            "policy_conclusion:",
            f"  two_agree_promising: {conclusion.get('two_agree_promising')}",
            f"  two_agree_safe_to_enable: {conclusion.get('two_agree_safe_to_enable')}",
            f"  scoring_weights_unchanged: {conclusion.get('scoring_weights_unchanged')}",
            f"  reason: {conclusion.get('reason')}",
            f"  next_step: {conclusion.get('next_step')}",
            f"source_documents: {report.get('source_documents')}",
            f"source_distribution_limitation: {report.get('source_distribution_limitation')}",
            "",
            "No auto-approval threshold is enabled.",
            "No gold export was performed.",
        ]
    )
    return "\n".join(lines) + "\n"


def sha256_file_prefix(path: Path, *, length: int = 16) -> str:
    """Return a short content hash for immutability checks."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest[:length]


def _format_rate_block(bucket: Mapping[str, Any], *, indent: str) -> list[str]:
    return [
        f"{indent}reviewed: {bucket.get('reviewed')}",
        f"{indent}human_verified (candidate_exact): {bucket.get('human_verified')}",
        f"{indent}human_corrected: {bucket.get('human_corrected')}",
        f"{indent}rejected: {bucket.get('rejected')}",
        f"{indent}bad_boundary: {bucket.get('bad_boundary')}",
        f"{indent}boundary_valid: {bucket.get('boundary_valid')}",
        f"{indent}candidate_exact_rate: {bucket.get('candidate_exact_rate')}%",
        f"{indent}correction_rate: {bucket.get('correction_rate')}%",
        f"{indent}boundary_failure_rate: {bucket.get('boundary_failure_rate')}%",
        f"{indent}candidate_exact_rate_among_boundary_valid: "
        f"{bucket.get('candidate_exact_rate_among_boundary_valid')}%",
    ]


def _rate(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(100.0 * count / total, 2)


def _bbox_size(bbox: object) -> tuple[int | None, int | None]:
    if not isinstance(bbox, list | tuple) or len(bbox) != 4:
        return None, None
    try:
        x0, y0, x1, y1 = (int(part) for part in bbox)
    except (TypeError, ValueError):
        return None, None
    return max(0, x1 - x0), max(0, y1 - y0)


def _confidence_bucket(value: object) -> str:
    if not isinstance(value, int | float):
        return "unknown"
    score = float(value)
    if score < 0.5:
        return "lt_0.50"
    if score < 0.75:
        return "0.50_0.75"
    if score < 0.90:
        return "0.75_0.90"
    return "ge_0.90"


def _risk_value(feature_risks: object, key: str) -> float | None:
    if not isinstance(feature_risks, Mapping):
        return None
    value = feature_risks.get(key)
    if isinstance(value, int | float):
        return float(value)
    return None


def _geometry_risk_bucket(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value < 0.25:
        return "low"
    if value < 0.60:
        return "medium"
    return "high"


def _table_hint(region: Mapping[str, Any] | None) -> str:
    if region is None:
        return "unknown"
    region_type = str(region.get("region_type") or "unknown")
    if region_type == "table":
        return "table"
    return "non_table"


def _line_length(
    texts: Mapping[str, Any],
    consensus: object,
    final_text: object,
) -> int:
    for value in (final_text, consensus, texts.get("indicocr"), texts.get("chitrapathak")):
        if isinstance(value, str):
            return len(value)
    return 0


def _length_bucket(length: int) -> str:
    if length <= 8:
        return "short_le_8"
    if length <= 40:
        return "medium_9_40"
    if length <= 80:
        return "long_41_80"
    return "very_long_gt_80"


def _height_bucket(height: int | None) -> str:
    if height is None:
        return "unknown"
    if height <= 20:
        return "h_le_20"
    if height <= 40:
        return "h_21_40"
    if height <= 80:
        return "h_41_80"
    return "h_gt_80"


def _tag_to_char_bucket(tag: str) -> str:
    mapping = {
        "missing_matra": "matra",
        "matra_error": "matra",
        "shared_ocr_error": "character_substitution",
        "conjunct_error": "conjunct",
        "numeral_error": "numeral",
        "punctuation_error": "punctuation",
        "hallucinated_suffix": "extra_character",
        "hallucinated_prefix": "extra_character",
        "hallucination": "extra_character",
        "truncation": "missing_character",
    }
    return mapping.get(tag, tag)


def _cluster_errors(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    clusters: Counter[str] = Counter()
    for row in rows:
        tags = [str(tag) for tag in (row.get("error_tags") or [])]
        if tags:
            clusters.update(tags)
        if row.get("table_hint") == "table":
            clusters["tables"] += 1
        bucket = row.get("line_length_bucket")
        if bucket:
            clusters[f"length:{bucket}"] += 1
        text = row.get("final_text") or row.get("proposed_consensus") or ""
        if isinstance(text, str):
            if re.search(r"[\u0966-\u096F0-9]", text):
                clusters["numerals_present"] += 1
            if "ं" in text or "ँ" in text:
                clusters["anusvara_or_candrabindu_in_gold"] += 1
    return dict(clusters.most_common())


def _two_agree_recommendation(
    rates: Mapping[str, Any],
    strict_rates: Mapping[str, Any],
    n: int,
) -> dict[str, Any]:
    exact = float(rates.get("candidate_exact_rate") or 0.0)
    boundary = float(rates.get("boundary_failure_rate") or 0.0)
    strict_exact = float(strict_rates.get("candidate_exact_rate") or 0.0)
    strict_n = int(strict_rates.get("reviewed") or 0)
    promising = exact >= 90.0 and n >= 100
    safe = False
    reason = (
        f"TWO_AGREE shows candidate_exact_rate={exact}% with boundary_failure_rate="
        f"{boundary}% on n={n}. Strict filter exact_rate={strict_exact}% on n={strict_n}. "
        "Promising for further calibration, but not safe to enable: sample is still "
        "one pilot document set, shared-error risk remains, and no second-batch "
        "confirmation exists."
    )
    return {
        "promising": promising,
        "safe_to_enable": safe,
        "reason": reason,
    }


def _duplicate_values(values: Iterable[str]) -> dict[str, int]:
    counts = Counter(values)
    return {value: count for value, count in counts.items() if count > 1}


def _lexical_skeleton(text: str) -> str:
    stored = apply_storage_normalization(text)
    chars = [ch for ch in stored if not ch.isdecimal() and not ch.isspace()]
    return "".join(chars)


def _number_normalized(text: str) -> str:
    stored = normalize_for_comparison(text)
    return re.sub(r"[\u0966-\u096F0-9]+", "#", stored)
