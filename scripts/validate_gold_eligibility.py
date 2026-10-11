#!/usr/bin/env python3
"""Validate gold eligibility without exporting gold.jsonl."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from indic_ocr_pipeline.config import load_settings  # noqa: E402
from indic_ocr_pipeline.dataset.eligibility import (  # noqa: E402
    collect_eligible_verified,
    proposed_gold_from_verified,
    validate_gold_pool,
)
from indic_ocr_pipeline.manifests import open_manifest  # noqa: E402
from indic_ocr_pipeline.utils.logging import configure_logging, get_logger  # noqa: E402

logger = get_logger("dataset.gold_eligibility")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether human-reviewed rows are ready for a future gold export. "
            "Does not write gold.jsonl."
        )
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--skip-image-bytes",
        action="store_true",
        help="Skip SHA-256 image byte checks (still validates fields)",
    )
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)
    settings = load_settings()
    manifests = settings.manifests_dir
    root = settings.repo_root
    verified = open_manifest(manifests, "verified").load()
    crops = {row["crop_id"]: row for row in open_manifest(manifests, "crops").load()}
    pages = {row["page_id"]: row for row in open_manifest(manifests, "pages").load()}
    documents = {
        row["document_id"]: row for row in open_manifest(manifests, "documents").load()
    }
    eligible = collect_eligible_verified(verified)
    proposed = []
    build_errors = []
    for row in eligible:
        crop = crops.get(str(row["crop_id"]))
        page = pages.get(str(row["page_id"]))
        if crop is None or page is None:
            build_errors.append(str(row["crop_id"]))
            continue
        try:
            proposed.append(
                proposed_gold_from_verified(
                    row,
                    crop=crop,
                    page=page,
                )
            )
        except Exception as exc:  # noqa: BLE001 - collect and report
            build_errors.append(f"{row.get('crop_id')}: {exc}")
    result = validate_gold_pool(
        proposed,
        repo_root=root,
        crops=crops,
        pages=pages,
        documents=documents,
        check_image_bytes=not args.skip_image_bytes,
    )
    result["eligible_verified_count"] = len(eligible)
    result["build_errors"] = build_errors
    result["gold_jsonl_exists"] = (manifests / "gold.jsonl").exists()
    out = Path(settings.reports_dir) / "gold_eligibility.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Phase 9: gold eligibility validation",
        "gold_export: False",
        "auto_approval_enabled: False",
        f"eligible_verified_count: {result['eligible_verified_count']}",
        f"proposed_count: {result['proposed_count']}",
        f"ok: {result['ok']}",
        f"row_error_count: {len(result['row_errors'])}",
        f"duplicate_sample_ids: {len(result['duplicate_sample_ids'])}",
        f"conflicting_gold_labels: {len(result['conflicting_gold_labels'])}",
        f"gold_jsonl_exists: {result['gold_jsonl_exists']}",
        f"document_split_ready: {result['document_split_readiness']['ready']}",
        f"documents: {result['document_split_readiness']['documents']}",
        "",
        "No gold export was performed.",
    ]
    text = "\n".join(lines) + "\n"
    out.write_text(text, encoding="utf-8")
    json_path = Path(settings.reports_dir) / "gold_eligibility.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(text, flush=True)
    print(f"wrote: {out}", flush=True)
    print(f"wrote: {json_path}", flush=True)
    logger.info(
        "gold eligibility validated",
        extra={
            "processed": result["proposed_count"],
            "completed": result["proposed_count"],
            "skipped": 0,
            "failed": len(result["row_errors"]) + len(build_errors),
        },
    )
    return 0 if result["ok"] and not build_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
