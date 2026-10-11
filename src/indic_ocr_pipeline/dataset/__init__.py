"""Gold dataset helpers.

Phase 9 prepares eligibility checks. Export remains a later phase.
"""

from indic_ocr_pipeline.dataset.eligibility import (
    GOLD_ELIGIBLE_STATUSES,
    GOLD_INELIGIBLE_STATUSES,
    collect_eligible_verified,
    document_split_readiness,
    is_gold_eligible_status,
    is_gold_ineligible_status,
    proposed_gold_from_verified,
    validate_gold_pool,
    validate_proposed_gold_row,
)

__all__ = [
    "GOLD_ELIGIBLE_STATUSES",
    "GOLD_INELIGIBLE_STATUSES",
    "collect_eligible_verified",
    "document_split_readiness",
    "is_gold_eligible_status",
    "is_gold_ineligible_status",
    "proposed_gold_from_verified",
    "validate_gold_pool",
    "validate_proposed_gold_row",
]
