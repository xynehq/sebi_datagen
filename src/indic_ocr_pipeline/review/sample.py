"""Deterministic stratified calibration sample for human review."""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from indic_ocr_pipeline.config import load_settings
from indic_ocr_pipeline.errors import DatasetError
from indic_ocr_pipeline.ids import make_review_id
from indic_ocr_pipeline.manifests import open_manifest
from indic_ocr_pipeline.utils.logging import configure_logging, get_logger

logger = get_logger("review.sample")

TIERS = ("P0", "P1", "P2", "P3")


class ReviewSampleError(DatasetError):
    """The calibration sampler could not build a batch."""


def sample_calibration_batch(
    reviews: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    tier_quota: Mapping[str, int | None],
    seed: int,
    batch_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return a deterministic stratified review batch.

    Every P0 row is included when ``tier_quota['P0']`` is null. Within each
    tier, rows are taken by document round-robin so one page cannot dominate.
    Crop IDs are unique.
    """
    if batch_size < 1:
        raise ReviewSampleError("batch_size must be at least 1")
    by_tier: dict[str, list[Mapping[str, Any]]] = {tier: [] for tier in TIERS}
    seen: set[str] = set()
    for row in reviews:
        crop_id = row.get("crop_id")
        if not isinstance(crop_id, str) or not crop_id:
            raise ReviewSampleError("review row missing crop_id")
        if crop_id in seen:
            raise ReviewSampleError(f"duplicate crop_id in review manifest: {crop_id}")
        seen.add(crop_id)
        diagnostics = row.get("diagnostics") if isinstance(row.get("diagnostics"), dict) else {}
        priority = diagnostics.get("review_priority")
        if priority not in by_tier:
            raise ReviewSampleError(f"unknown review_priority {priority!r} on {crop_id}")
        by_tier[str(priority)].append(row)
    for tier in TIERS:
        by_tier[tier].sort(key=lambda row: (int(row["priority_rank"]), str(row["crop_id"])))
    selected_rows: list[Mapping[str, Any]] = []
    for tier in TIERS:
        quota = tier_quota.get(tier)
        pool = by_tier[tier]
        if quota is None:
            take = list(pool)
        else:
            take = _diversify(pool, quota, seed=seed + TIERS.index(tier))
        selected_rows.extend(take)
    if len(selected_rows) > batch_size:
        # Keep all P0, then fill remaining slots in tier order P1,P2,P3.
        kept: list[Mapping[str, Any]] = []
        for tier in TIERS:
            tier_rows = [row for row in selected_rows if _priority(row) == tier]
            if tier == "P0":
                kept.extend(tier_rows)
                continue
            room = batch_size - len(kept)
            if room <= 0:
                break
            kept.extend(tier_rows[:room])
        selected_rows = kept
    batch_name = batch_id or f"calibration_{seed}_{len(selected_rows)}"
    items: list[dict[str, Any]] = []
    used: set[str] = set()
    for index, row in enumerate(selected_rows, start=1):
        crop_id = str(row["crop_id"])
        if crop_id in used:
            raise ReviewSampleError(f"sampler produced duplicate crop_id {crop_id}")
        used.add(crop_id)
        diagnostics = row.get("diagnostics") if isinstance(row.get("diagnostics"), dict) else {}
        items.append(
            {
                "batch_item_id": f"{batch_name}:{crop_id}",
                "batch_id": batch_name,
                "sample_index": index,
                "crop_id": crop_id,
                "review_id": make_review_id(crop_id),
                "document_id": row["document_id"],
                "page_id": row["page_id"],
                "parent_line_id": row["parent_line_id"],
                "region_id": diagnostics.get("region_id"),
                "review_priority": diagnostics.get("review_priority"),
                "agreement_category": diagnostics.get("agreement_category"),
                "priority_rank": row["priority_rank"],
                "risk_score": row.get("risk_score"),
                "seed": seed,
            }
        )
    return items


def enqueue_pending_reviews(
    manifests_dir: Path,
    *,
    batch_id: str | None = None,
    seed: int = 0,
    limit: int | None = None,
) -> dict[str, int]:
    """Append unreviewed crops that are not already in the review batch.

    Completed human decisions and existing batch rows are left unchanged.
    This does not accept a candidate and does not write gold.
    """
    chosen_batch = INBOX_BATCH_ID if batch_id is None else batch_id
    manifests_dir = Path(manifests_dir)
    reviews = open_manifest(manifests_dir, "review").load()
    verified_ids = {
        str(row["crop_id"])
        for row in open_manifest(manifests_dir, "verified").load()
        if row.get("crop_id")
    }
    existing = open_manifest(manifests_dir, "review_batch").load()
    batched = {str(row["crop_id"]) for row in existing if row.get("crop_id")}
    start_index = 0
    if existing:
        start_index = max(int(row["sample_index"]) for row in existing)
    pending = [
        row
        for row in reviews
        if str(row.get("status")) != "complete"
        and str(row.get("crop_id")) not in verified_ids
        and str(row.get("crop_id")) not in batched
    ]
    pending.sort(key=lambda row: (int(row["priority_rank"]), str(row["crop_id"])))
    if limit is not None:
        if limit < 1:
            raise ReviewSampleError("datapoint limit must be at least 1")
        pending = pending[:limit]
    items: list[dict[str, Any]] = []
    for offset, row in enumerate(pending, start=1):
        crop_id = str(row["crop_id"])
        diagnostics = row.get("diagnostics") if isinstance(row.get("diagnostics"), dict) else {}
        items.append(
            {
                "batch_item_id": f"{chosen_batch}:{crop_id}",
                "batch_id": chosen_batch,
                "sample_index": start_index + offset,
                "crop_id": crop_id,
                "review_id": str(row.get("review_id") or make_review_id(crop_id)),
                "document_id": row["document_id"],
                "page_id": row["page_id"],
                "parent_line_id": row["parent_line_id"],
                "region_id": diagnostics.get("region_id"),
                "review_priority": diagnostics.get("review_priority"),
                "agreement_category": diagnostics.get("agreement_category"),
                "priority_rank": row["priority_rank"],
                "risk_score": row.get("risk_score"),
                "seed": seed,
            }
        )
    counts = write_review_batch(manifests_dir, items)
    return {
        "created": counts["created"],
        "skipped": counts["skipped"],
        "queued": len(items),
        "already_batched": len(batched),
    }


def write_review_batch(manifests_dir: Path, items: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Write the batch manifest. Identical content resumes; changes conflict."""
    manifest = open_manifest(Path(manifests_dir), "review_batch")
    created = 0
    skipped = 0
    for item in items:
        outcome = manifest.append(dict(item))
        if outcome == "created":
            created += 1
        else:
            skipped += 1
    return {"created": created, "skipped": skipped, "total": len(items)}


def summarize_batch(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return tier and document counts for a batch."""
    tiers: dict[str, int] = {tier: 0 for tier in TIERS}
    documents: dict[str, int] = defaultdict(int)
    categories: dict[str, int] = defaultdict(int)
    for item in items:
        tiers[str(item.get("review_priority"))] = tiers.get(str(item.get("review_priority")), 0) + 1
        documents[str(item.get("document_id"))] += 1
        categories[str(item.get("agreement_category"))] += 1
    return {
        "batch_size": len(items),
        "tiers": tiers,
        "documents": dict(sorted(documents.items())),
        "agreement_categories": dict(sorted(categories.items())),
        "unique_crop_ids": len({item["crop_id"] for item in items}),
    }


SECOND_BATCH_ID = "calibration_3407_twoagree_v1"
PHASE10_BATCH_ID = "calibration_3407_phase10_v1"
INBOX_BATCH_ID = "inbox"
BOUNDARY_SENSITIVE_FLAGS = frozenset(
    {
        "empty_prediction",
        "large_length_disagreement",
        "unicode_anomaly",
        "likely_truncation",
        "likely_extra_text",
    }
)


def available_population(
    reviews: Sequence[Mapping[str, Any]],
    *,
    exclude_crop_ids: set[str],
) -> dict[str, Any]:
    """Report unreviewed pools before second-batch sampling."""
    remaining = [
        row for row in reviews if str(row.get("crop_id")) not in exclude_crop_ids
    ]
    by_agreement: dict[str, int] = defaultdict(int)
    by_tier: dict[str, int] = defaultdict(int)
    boundary_sensitive = 0
    for row in remaining:
        diagnostics = row.get("diagnostics") if isinstance(row.get("diagnostics"), dict) else {}
        by_agreement[str(diagnostics.get("agreement_category"))] += 1
        by_tier[str(diagnostics.get("review_priority"))] += 1
        flags = set(diagnostics.get("flags") or [])
        geometry = diagnostics.get("feature_risks") or {}
        crop_geometry = geometry.get("crop_geometry") if isinstance(geometry, dict) else None
        if flags & BOUNDARY_SENSITIVE_FLAGS or (
            isinstance(crop_geometry, int | float) and float(crop_geometry) >= 0.6
        ):
            boundary_sensitive += 1
    return {
        "remaining_total": len(remaining),
        "excluded": len(exclude_crop_ids),
        "by_agreement_category": dict(sorted(by_agreement.items())),
        "by_review_priority": dict(sorted(by_tier.items())),
        "two_agree": by_agreement.get("TWO_AGREE_ONE_DIFFERS", 0),
        "all_3_agree": by_agreement.get("ALL_3_AGREE", 0),
        "p2": by_tier.get("P2", 0),
        "p3": by_tier.get("P3", 0),
        "boundary_sensitive": boundary_sensitive,
    }


def sample_twoagree_calibration_batch(
    reviews: Sequence[Mapping[str, Any]],
    *,
    exclude_crop_ids: set[str],
    seed: int,
    batch_id: str = SECOND_BATCH_ID,
    two_agree_quota: int | None = None,
    all3_quota: int | None = None,
    p2_extra_quota: int = 0,
    p3_extra_quota: int = 0,
    boundary_quota: int = 100,
    start_index: int = 1,
    allow_overlap: bool = False,
) -> list[dict[str, Any]]:
    """Deterministic second calibration batch focused on TWO_AGREE evidence.

    Excludes the completed first-batch crop IDs unless ``allow_overlap`` is true.
    Sample indexes continue from ``start_index`` so the review UI queues new
    rows after the finished batch.
    """
    if start_index < 1:
        raise ReviewSampleError("start_index must be at least 1")
    excluded = set() if allow_overlap else set(exclude_crop_ids)
    remaining = [row for row in reviews if str(row.get("crop_id")) not in excluded]
    if allow_overlap is False:
        overlap = {
            str(row.get("crop_id")) for row in remaining
        } & set(exclude_crop_ids)
        if overlap:
            raise ReviewSampleError("exclude filter failed; overlap remains")

    def _diag(row: Mapping[str, Any]) -> dict[str, Any]:
        diagnostics = row.get("diagnostics")
        return diagnostics if isinstance(diagnostics, dict) else {}

    two_pool = [
        row
        for row in remaining
        if _diag(row).get("agreement_category") == "TWO_AGREE_ONE_DIFFERS"
    ]
    all3_pool = [
        row
        for row in remaining
        if _diag(row).get("agreement_category") == "ALL_3_AGREE"
    ]
    p2_pool = [row for row in remaining if _diag(row).get("review_priority") == "P2"]
    p3_pool = [row for row in remaining if _diag(row).get("review_priority") == "P3"]
    def _boundary_sensitive(row: Mapping[str, Any]) -> bool:
        diagnostics = _diag(row)
        flags = set(diagnostics.get("flags") or [])
        if flags & BOUNDARY_SENSITIVE_FLAGS:
            return True
        risks = diagnostics.get("feature_risks") or {}
        geometry = risks.get("crop_geometry") if isinstance(risks, dict) else None
        return isinstance(geometry, int | float) and float(geometry) >= 0.6

    boundary_pool = [row for row in remaining if _boundary_sensitive(row)]

    selected: list[Mapping[str, Any]] = []
    used: set[str] = set()

    def _take(pool: Sequence[Mapping[str, Any]], quota: int | None, salt: int) -> None:
        if quota is not None and quota <= 0:
            return
        limit = len(pool) if quota is None else min(quota, len(pool))
        chosen = _diversify_by_document_and_length(pool, limit, seed=seed + salt)
        for row in chosen:
            crop_id = str(row["crop_id"])
            if crop_id in used:
                continue
            used.add(crop_id)
            selected.append(row)

    _take(two_pool, two_agree_quota, 11)
    _take(all3_pool, all3_quota, 22)
    _take(p2_pool, p2_extra_quota, 33)
    _take(p3_pool, p3_extra_quota, 44)
    _take(boundary_pool, boundary_quota, 55)

    if not allow_overlap:
        overlap = {str(row["crop_id"]) for row in selected} & set(exclude_crop_ids)
        if overlap:
            raise ReviewSampleError(
                f"second batch overlaps first batch: {sorted(overlap)[:5]}"
            )

    items: list[dict[str, Any]] = []
    for offset, row in enumerate(selected):
        diagnostics = _diag(row)
        crop_id = str(row["crop_id"])
        items.append(
            {
                "batch_item_id": f"{batch_id}:{crop_id}",
                "batch_id": batch_id,
                "sample_index": start_index + offset,
                "crop_id": crop_id,
                "review_id": make_review_id(crop_id),
                "document_id": row["document_id"],
                "page_id": row["page_id"],
                "parent_line_id": row["parent_line_id"],
                "region_id": diagnostics.get("region_id"),
                "review_priority": diagnostics.get("review_priority"),
                "agreement_category": diagnostics.get("agreement_category"),
                "priority_rank": row["priority_rank"],
                "risk_score": row.get("risk_score"),
                "seed": seed,
                "focus": "twoagree_controls_boundary",
            }
        )
    return items


def sample_phase10_multidoc_batch(
    reviews: Sequence[Mapping[str, Any]],
    *,
    document_ids: set[str],
    exclude_crop_ids: set[str],
    seed: int,
    batch_size: int = 400,
    batch_id: str = PHASE10_BATCH_ID,
    start_index: int = 1,
) -> list[dict[str, Any]]:
    """Deterministic multi-document calibration sample for Phase 10.

    Only crops from ``document_ids`` are eligible. Historical pilot crops are
    excluded. Every eligible document contributes when possible.
    """
    if batch_size < 1:
        raise ReviewSampleError("batch_size must be at least 1")
    if start_index < 1:
        raise ReviewSampleError("start_index must be at least 1")
    if not document_ids:
        raise ReviewSampleError("document_ids must be non-empty")
    pool = [
        row
        for row in reviews
        if str(row.get("document_id")) in document_ids
        and str(row.get("crop_id")) not in exclude_crop_ids
    ]
    if not pool:
        raise ReviewSampleError("no eligible review rows for Phase 10 sampling")

    def _diag(row: Mapping[str, Any]) -> dict[str, Any]:
        diagnostics = row.get("diagnostics")
        return diagnostics if isinstance(diagnostics, dict) else {}

    # Strata: agreement × risk tier, with geometry-sensitive and length buckets.
    strata: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in pool:
        diagnostics = _diag(row)
        agreement = str(diagnostics.get("agreement_category") or "unknown")
        tier = str(diagnostics.get("review_priority") or "unknown")
        consensus = str(diagnostics.get("proposed_consensus") or "")
        length = len(consensus)
        if length <= 8:
            length_bucket = "short"
        elif length <= 40:
            length_bucket = "medium"
        elif length <= 80:
            length_bucket = "long"
        else:
            length_bucket = "very_long"
        flags = set(diagnostics.get("flags") or [])
        geometry = "geom" if flags & BOUNDARY_SENSITIVE_FLAGS else "clean"
        numeral = (
            "num"
            if any(ch.isdecimal() for ch in consensus)
            or "numeral_disagreement" in flags
            else "nonum"
        )
        punct = (
            "punct"
            if "punctuation_disagreement" in flags
            or any(ch in consensus for ch in ",.;:()[]/-")
            else "nopunct"
        )
        key = f"{tier}|{agreement}|{geometry}|{length_bucket}|{numeral}|{punct}"
        strata[key].append(row)

    # Document round-robin fill across strata so one document cannot dominate.
    selected: list[Mapping[str, Any]] = []
    used: set[str] = set()
    # First ensure at least one row per document when possible.
    by_document: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in sorted(pool, key=lambda item: (int(item["priority_rank"]), str(item["crop_id"]))):
        by_document[str(row["document_id"])].append(row)
    for document in sorted(by_document):
        for row in by_document[document]:
            crop_id = str(row["crop_id"])
            if crop_id in used:
                continue
            used.add(crop_id)
            selected.append(row)
            break
        if len(selected) >= batch_size:
            break

    # Fill remaining slots from diversified strata.
    stratum_names = sorted(strata)
    rng = random.Random(seed)
    rng.shuffle(stratum_names)
    salt = 0
    while len(selected) < batch_size:
        progress = False
        for name in stratum_names:
            remaining = [row for row in strata[name] if str(row["crop_id"]) not in used]
            if not remaining:
                continue
            take = _diversify_by_document_and_length(remaining, 1, seed=seed + salt)
            salt += 1
            if not take:
                continue
            row = take[0]
            crop_id = str(row["crop_id"])
            if crop_id in used:
                continue
            used.add(crop_id)
            selected.append(row)
            progress = True
            if len(selected) >= batch_size:
                break
        if not progress:
            break

    if {str(row["crop_id"]) for row in selected} & set(exclude_crop_ids):
        raise ReviewSampleError("phase10 sampler overlapped excluded crop ids")

    items: list[dict[str, Any]] = []
    for offset, row in enumerate(selected):
        diagnostics = _diag(row)
        crop_id = str(row["crop_id"])
        items.append(
            {
                "batch_item_id": f"{batch_id}:{crop_id}",
                "batch_id": batch_id,
                "sample_index": start_index + offset,
                "crop_id": crop_id,
                "review_id": make_review_id(crop_id),
                "document_id": row["document_id"],
                "page_id": row["page_id"],
                "parent_line_id": row["parent_line_id"],
                "region_id": diagnostics.get("region_id"),
                "review_priority": diagnostics.get("review_priority"),
                "agreement_category": diagnostics.get("agreement_category"),
                "priority_rank": row["priority_rank"],
                "risk_score": row.get("risk_score"),
                "seed": seed,
                "focus": "phase10_multidoc_generalization",
            }
        )
    return items


def _diversify_by_document_and_length(
    pool: Sequence[Mapping[str, Any]],
    quota: int,
    *,
    seed: int,
) -> list[Mapping[str, Any]]:
    """Document round-robin with secondary length diversity."""
    if quota <= 0 or not pool:
        return []
    if quota >= len(pool):
        return sorted(
            pool,
            key=lambda row: (int(row["priority_rank"]), str(row["crop_id"])),
        )
    by_document: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in sorted(pool, key=lambda item: (int(item["priority_rank"]), str(item["crop_id"]))):
        by_document[str(row["document_id"])].append(row)
    rng = random.Random(seed)
    documents = sorted(by_document)
    rng.shuffle(documents)
    # Rotate buckets so short/long lines mix within each document.
    for document in documents:
        bucket = by_document[document]

        def _length_key(item: Mapping[str, Any]) -> tuple[int, str]:
            raw = item.get("diagnostics")
            diagnostics = raw if isinstance(raw, dict) else {}
            consensus = diagnostics.get("proposed_consensus") or ""
            return (len(str(consensus)), str(item["crop_id"]))

        ordered = sorted(bucket, key=_length_key)
        # Interleave short and long.
        left = ordered[: len(ordered) // 2]
        right = list(reversed(ordered[len(ordered) // 2 :]))
        interleaved: list[Mapping[str, Any]] = []
        for index in range(max(len(left), len(right))):
            if index < len(left):
                interleaved.append(left[index])
            if index < len(right):
                interleaved.append(right[index])
        by_document[document] = interleaved
    selected: list[Mapping[str, Any]] = []
    index = 0
    while len(selected) < quota:
        progress = False
        for _ in range(len(documents)):
            document = documents[index % len(documents)]
            index += 1
            bucket = by_document[document]
            if not bucket:
                continue
            selected.append(bucket.pop(0))
            progress = True
            if len(selected) >= quota:
                break
        if not progress:
            break
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic stratified calibration review batch. "
            "This does not create gold and does not run OCR."
        )
    )
    parser.add_argument("--n", type=int, default=None, help="Override calibration_batch_size")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)
    settings = load_settings()
    reviews = open_manifest(settings.manifests_dir, "review").load()
    if not reviews:
        raise SystemExit("manifests/review.jsonl is empty; run Phase 7 first")
    batch_size = args.n if args.n is not None else settings.review.calibration_batch_size
    items = sample_calibration_batch(
        reviews,
        batch_size=batch_size,
        tier_quota=settings.review.calibration_tier_quota,
        seed=settings.review.random_seed,
    )
    counts = write_review_batch(settings.manifests_dir, items)
    summary = summarize_batch(items)
    print(f"batch_id: {items[0]['batch_id'] if items else None}")
    print(f"batch_size: {summary['batch_size']}")
    print(f"created: {counts['created']}")
    print(f"skipped: {counts['skipped']}")
    for tier in TIERS:
        print(f"tier.{tier}: {summary['tiers'].get(tier, 0)}")
    print(f"unique_crop_ids: {summary['unique_crop_ids']}")
    print(f"documents: {summary['documents']}")
    print("auto_approval_enabled: false")
    print("gold_export: false")
    logger.info(
        "calibration batch written",
        extra={
            "processed": summary["batch_size"],
            "completed": counts["created"],
            "skipped": counts["skipped"],
            "failed": 0,
        },
    )
    return 0


def _priority(row: Mapping[str, Any]) -> str:
    diagnostics = row.get("diagnostics") if isinstance(row.get("diagnostics"), dict) else {}
    return str(diagnostics.get("review_priority"))


def _diversify(
    pool: Sequence[Mapping[str, Any]],
    quota: int,
    *,
    seed: int,
) -> list[Mapping[str, Any]]:
    if quota <= 0 or not pool:
        return []
    if quota >= len(pool):
        return list(pool)
    by_document: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in pool:
        by_document[str(row["document_id"])].append(row)
    rng = random.Random(seed)
    documents = sorted(by_document)
    rng.shuffle(documents)
    selected: list[Mapping[str, Any]] = []
    index = 0
    while len(selected) < quota:
        progress = False
        for _ in range(len(documents)):
            document = documents[index % len(documents)]
            index += 1
            bucket = by_document[document]
            if not bucket:
                continue
            selected.append(bucket.pop(0))
            progress = True
            if len(selected) >= quota:
                break
        if not progress:
            break
    return selected


if __name__ == "__main__":
    raise SystemExit(main())
