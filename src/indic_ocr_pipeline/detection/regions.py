"""Detect visual text regions on rendered page images.

A region is a rectangular block: a heading, a paragraph, a table, or another
visually separate ink group. This stage does not find physical lines, does
not crop, and does not read text.

Boxes are ``[x0, y0, x1, y1]`` in rendered-page pixels. The origin is the
top-left corner. ``x1`` and ``y1`` are exclusive.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, UnidentifiedImageError

from indic_ocr_pipeline.config import DetectionSettings
from indic_ocr_pipeline.errors import DatasetError, ManifestConflictError, SchemaError
from indic_ocr_pipeline.ids import make_region_id
from indic_ocr_pipeline.manifests import JsonlManifest
from indic_ocr_pipeline.rendering.pages import resolve_existing_path
from indic_ocr_pipeline.schemas import (
    BBox,
    RegionRecord,
    RegionType,
    StageStatus,
    assert_bbox_within_page,
)
from indic_ocr_pipeline.utils.hashing import sha256_file
from indic_ocr_pipeline.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

DETECTOR_VERSION = "projection_blocks_v1"
HIGH_REGION_COUNT = 25
RESUME_NOTE = "Existing region rows were kept. Detection can resume."


class DetectionError(DatasetError):
    """A page could not be segmented into regions."""


class DetectionConflictError(DetectionError):
    """A region id already exists with a different payload, or the page image changed."""


@dataclass(frozen=True)
class _HRule:
    """A horizontal ruling stroke. ``y`` is thickness and ``x`` is length."""

    y0: int
    y1: int
    x0: int
    x1: int

    @property
    def y_center(self) -> int:
        return (self.y0 + self.y1) // 2


@dataclass(frozen=True)
class _VRule:
    """A vertical ruling stroke. ``x`` is thickness and ``y`` is length."""

    x0: int
    x1: int
    y0: int
    y1: int

    @property
    def x_center(self) -> int:
        return (self.x0 + self.x1) // 2


@dataclass(frozen=True)
class DetectedBox:
    """One region before an id is assigned."""

    x0: int
    y0: int
    x1: int
    y1: int
    region_type: RegionType


@dataclass
class DocumentDetectStats:
    """Detection counters for one source document."""

    document_id: str
    pages: int = 0
    regions: int = 0
    skipped: int = 0
    zero_pages: int = 0
    high_pages: int = 0
    errors: int = 0
    message: str = ""


@dataclass
class DetectionReport:
    """Counters for one detection run."""

    pages_discovered: int = 0
    regions: int = 0
    regions_skipped: int = 0
    zero_pages: int = 0
    high_pages: int = 0
    errors: int = 0
    details: list[DocumentDetectStats] = field(default_factory=list)

    def add(self, stats: DocumentDetectStats) -> None:
        self.details.append(stats)
        self.pages_discovered += stats.pages
        self.regions += stats.regions
        self.regions_skipped += stats.skipped
        self.zero_pages += stats.zero_pages
        self.high_pages += stats.high_pages
        self.errors += stats.errors


def detection_method_label(settings: DetectionSettings) -> str:
    """Return a stable label for this detector version and configuration."""
    return (
        f"{DETECTOR_VERSION}"
        f";method={settings.method}"
        f";gap_y={settings.block_gap_y_px}"
        f";gap_max={settings.max_block_gap_px}"
        f";gap_x={settings.column_gap_px}"
        f";bin={settings.binarization}"
        f";block={settings.adaptive_block_size}"
        f";c={settings.adaptive_c}"
        f";pad={settings.padding_px}"
        f";min_w={settings.min_region_width_px}"
        f";min_h={settings.min_region_height_px}"
        f";min_ink={settings.min_region_ink_px}"
        f";rule={settings.rule_min_length_px}"
    )


def detect_regions_on_array(gray: np.ndarray, settings: DetectionSettings) -> list[DetectedBox]:
    """Return regions for one grayscale page, sorted top-to-bottom then left-to-right."""
    if gray.ndim != 2 or gray.dtype != np.uint8 or gray.size == 0:
        raise DetectionError("page array must be a non-empty uint8 grayscale image")
    height, width = gray.shape
    ink = _ink_mask(gray, settings)
    horizontal, vertical = _ruling_lines(ink, settings)
    tables = _table_boxes(horizontal, vertical, width, height, settings)
    text_ink = _ink_for_text(ink, vertical, tables, height)
    regions = list(tables)
    for box in _segment_blocks(text_ink, settings):
        if _overlaps_table(box, tables):
            continue
        regions.append(box)
    padded = [_pad_box(box, settings.padding_px, width, height) for box in regions]
    valid = [box for box in padded if _keep_box(box, ink, settings)]
    for box in valid:
        _require_inside(box, width, height)
    valid.sort(key=lambda box: (box.y0, box.x0, box.y1, box.x1, box.region_type.value))
    return valid


def detect_pages(
    page_rows: Sequence[Mapping[str, object]],
    manifest: JsonlManifest,
    *,
    repo_root: Path,
    settings: DetectionSettings,
    debug_dir: Path | None = None,
) -> DetectionReport:
    """Detect regions for every page row, one image at a time."""
    pages = sorted(
        (dict(row) for row in page_rows),
        key=lambda row: (str(row.get("document_id")), int(row.get("page_number", 0))),
    )
    existing = _regions_by_page(manifest.load())
    report = DetectionReport()
    current_id = ""
    stats: DocumentDetectStats | None = None
    for page in pages:
        document_id = str(page.get("document_id", ""))
        if document_id != current_id:
            if stats is not None:
                report.add(stats)
            current_id = document_id
            stats = DocumentDetectStats(document_id=document_id)
        assert stats is not None
        stats.pages += 1
        try:
            _created, skipped, count = _detect_one_page(
                page,
                manifest,
                existing,
                repo_root=repo_root,
                settings=settings,
                debug_dir=debug_dir,
            )
        except (DetectionError, SchemaError, OSError, ManifestConflictError) as exc:
            stats.errors += 1
            if not stats.message:
                stats.message = str(exc)
            logger.error("%s", exc)
            continue
        stats.regions += count
        stats.skipped += skipped
        if count == 0:
            stats.zero_pages += 1
        if count > HIGH_REGION_COUNT:
            stats.high_pages += 1
    if stats is not None:
        report.add(stats)
    return report


def format_report(report: DetectionReport) -> str:
    """Return the operator-facing detection summary."""
    lines = [
        "Phase 4: Region detection",
        "",
        f"Pages discovered: {report.pages_discovered}",
        "",
    ]
    for stats in report.details:
        lines.append(stats.document_id)
        lines.append(f"  pages: {stats.pages}")
        lines.append(f"  regions: {stats.regions}")
        lines.append(f"  skipped: {stats.skipped}")
        if stats.message:
            lines.append(f"  error: {stats.message}")
        lines.append("")
    lines.extend(
        [
            "Detection complete.",
            f"  pages: {report.pages_discovered}",
            f"  regions: {report.regions}",
            f"  skipped: {report.regions_skipped}",
            f"  pages with zero regions: {report.zero_pages}",
            f"  pages with more than {HIGH_REGION_COUNT} regions: {report.high_pages}",
            f"  errors: {report.errors}",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Detect regions for every rendered page in the page manifest."""
    from indic_ocr_pipeline.config import load_settings
    from indic_ocr_pipeline.manifests import open_manifest

    parser = argparse.ArgumentParser(
        description="Detect text regions on rendered page images. No OCR is run."
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Write annotated copies under data/debug/regions.",
    )
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)
    started = datetime.now(UTC)
    try:
        settings = load_settings()
        pages = open_manifest(settings.manifests_dir, "pages")
        regions = open_manifest(settings.manifests_dir, "regions")
        debug_dir = None
        if args.debug or settings.detection.debug_images:
            debug_dir = settings.repo_root / "data" / "debug" / "regions"
        report = detect_pages(
            pages.load(),
            regions,
            repo_root=settings.repo_root,
            settings=settings.detection,
            debug_dir=debug_dir,
        )
    except (DetectionError, ManifestConflictError, SchemaError) as exc:
        print(f"ERROR  {exc}", file=sys.stderr)
        logger.error("%s", exc)
        return 1
    duration = (datetime.now(UTC) - started).total_seconds()
    print(format_report(report), flush=True)
    logger.info(
        "region detection finished",
        extra={
            "processed": report.pages_discovered,
            "completed": report.regions - report.regions_skipped,
            "skipped": report.regions_skipped,
            "failed": report.errors,
            "duration_seconds": round(duration, 3),
        },
    )
    return 1 if report.errors else 0


def _detect_one_page(
    page: Mapping[str, object],
    manifest: JsonlManifest,
    existing: dict[str, list[dict[str, object]]],
    *,
    repo_root: Path,
    settings: DetectionSettings,
    debug_dir: Path | None,
) -> tuple[int, int, int]:
    page_id = str(page.get("page_id", ""))
    document_id = str(page.get("document_id", ""))
    page_number = page.get("page_number")
    if not isinstance(page_number, int) or isinstance(page_number, bool):
        raise DetectionError(f"{page_id} has no integer page_number")
    image_path = resolve_existing_path(str(page.get("image_path", "")), repo_root)
    if not image_path.is_file():
        raise DetectionError(f"{page_id} rendered page image is missing: {image_path}")
    recorded_sha = str(page.get("sha256", ""))
    actual_sha = sha256_file(image_path)
    if actual_sha != recorded_sha:
        raise DetectionConflictError(
            f"{page_id} page image SHA-256 does not match the page manifest. {RESUME_NOTE}"
        )
    recorded_width = page.get("width")
    recorded_height = page.get("height")
    gray = _load_gray(image_path, page_id)
    height, width = gray.shape
    if recorded_width != width or recorded_height != height:
        raise DetectionConflictError(
            f"{page_id} page image size {width}x{height} does not match the page manifest. "
            f"{RESUME_NOTE}"
        )
    boxes = detect_regions_on_array(gray, settings)
    records = _records_for_page(
        boxes,
        document_id=document_id,
        page_id=page_id,
        page_number=page_number,
        settings=settings,
    )
    created, skipped = _commit_page_regions(
        records,
        existing.get(page_id, []),
        manifest,
        existing,
        page_id=page_id,
    )
    if debug_dir is not None:
        _write_debug(image_path, debug_dir, document_id, page_number, records)
    del gray
    return created, skipped, len(records)


def _records_for_page(
    boxes: Sequence[DetectedBox],
    *,
    document_id: str,
    page_id: str,
    page_number: int,
    settings: DetectionSettings,
) -> list[RegionRecord]:
    method = detection_method_label(settings)
    records: list[RegionRecord] = []
    for index, box in enumerate(boxes, start=1):
        records.append(
            RegionRecord(
                region_id=make_region_id(page_id, index),
                document_id=document_id,
                page_id=page_id,
                page_number=page_number,
                region_index=index,
                region_type=box.region_type,
                bbox=BBox(box.x0, box.y0, box.x1, box.y1),
                detection_method=method,
                confidence=None,
                reading_order=index,
                status=StageStatus.COMPLETE,
            )
        )
    return records


def _commit_page_regions(
    records: Sequence[RegionRecord],
    stored: Sequence[Mapping[str, object]],
    manifest: JsonlManifest,
    existing: dict[str, list[dict[str, object]]],
    *,
    page_id: str,
) -> tuple[int, int]:
    payloads = [record.to_dict() for record in records]
    stored_by_id = {str(row.get("region_id")): dict(row) for row in stored}
    new_ids = {str(payload["region_id"]) for payload in payloads}
    if set(stored_by_id) - new_ids:
        raise DetectionConflictError(
            f"{page_id} region set changed and an existing region would be left behind. "
            f"{RESUME_NOTE}"
        )
    pending: list[dict[str, object]] = []
    skipped = 0
    for payload in payloads:
        region_id = str(payload["region_id"])
        current = stored_by_id.get(region_id)
        if current is None:
            pending.append(payload)
            continue
        if current != payload:
            raise DetectionConflictError(
                f"{region_id} already exists with a different payload. {RESUME_NOTE}"
            )
        skipped += 1
    created = 0
    page_rows = [dict(row) for row in stored]
    for payload in pending:
        try:
            outcome = manifest.append(payload)
        except ManifestConflictError as exc:
            raise DetectionConflictError(str(exc)) from exc
        if outcome == "skipped":
            skipped += 1
        else:
            created += 1
        page_rows.append(payload)
    existing[page_id] = page_rows
    return created, skipped


def _regions_by_page(rows: Sequence[Mapping[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        page_id = str(row.get("page_id", ""))
        grouped.setdefault(page_id, []).append(dict(row))
    return grouped


def _load_gray(path: Path, page_id: str) -> np.ndarray:
    try:
        with Image.open(path) as image:
            gray = np.array(image.convert("L"), dtype=np.uint8, copy=True)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise DetectionError(f"{page_id} rendered page image is not a readable PNG") from exc
    if gray.ndim != 2 or gray.size == 0:
        raise DetectionError(f"{page_id} rendered page image is not a readable PNG")
    return gray


def _ink_mask(gray: np.ndarray, settings: DetectionSettings) -> np.ndarray:
    if settings.binarization == "otsu":
        level = _otsu_threshold(gray)
        return gray <= level
    local = _local_mean(gray, settings.adaptive_block_size)
    adaptive = (gray.astype(np.int16) < (local - settings.adaptive_c)) & (
        gray <= settings.ink_max_gray
    )
    absolute = gray <= settings.absolute_ink_gray
    return adaptive | absolute


def _local_mean(gray: np.ndarray, block: int) -> np.ndarray:
    radius = block // 2
    padded = np.pad(gray.astype(np.float64), radius, mode="edge")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    height, width = gray.shape
    y0 = np.arange(height)
    x0 = np.arange(width)
    y1 = y0 + block
    x1 = x0 + block
    total = (
        integral[np.ix_(y1, x1)]
        - integral[np.ix_(y0, x1)]
        - integral[np.ix_(y1, x0)]
        + integral[np.ix_(y0, x0)]
    )
    return total / float(block * block)


def _otsu_threshold(gray: np.ndarray) -> int:
    histogram = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = float(histogram.sum())
    if total == 0:
        return 0
    sum_total = float(np.dot(np.arange(256), histogram))
    weight_background = 0.0
    sum_background = 0.0
    best_variance = -1.0
    threshold = 0
    for level, count in enumerate(histogram):
        weight_background += count
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break
        sum_background += level * count
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        if variance > best_variance:
            best_variance = variance
            threshold = level
    return threshold


def _ruling_lines(
    ink: np.ndarray, settings: DetectionSettings
) -> tuple[list[_HRule], list[_VRule]]:
    height, width = ink.shape
    min_horizontal = max(
        settings.rule_min_length_px, int(width * settings.rule_min_length_fraction)
    )
    min_vertical = max(
        settings.rule_min_length_px, int(height * settings.rule_min_length_fraction)
    )
    horizontal = [
        _HRule(y0=start, y1=end, x0=cross0, x1=cross1)
        for start, end, cross0, cross1 in _merge_runs(_long_runs(ink, min_horizontal, axis=1))
    ]
    vertical = [
        _VRule(x0=start, x1=end, y0=cross0, y1=cross1)
        for start, end, cross0, cross1 in _merge_runs(_long_runs(ink, min_vertical, axis=0))
    ]
    return horizontal, vertical


def _long_runs(ink: np.ndarray, min_length: int, *, axis: int) -> list[tuple[int, int, int]]:
    """Return ``(index, cross0, cross1)`` runs along ``axis`` (1 = rows)."""
    found: list[tuple[int, int, int]] = []
    lines = ink if axis == 1 else ink.T
    for index, line in enumerate(lines):
        padded = np.concatenate((np.zeros(1, dtype=bool), line, np.zeros(1, dtype=bool)))
        changes = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        for start, end in zip(starts, ends, strict=True):
            if int(end - start) >= min_length:
                found.append((index, int(start), int(end)))
    return found


def _merge_runs(runs: Sequence[tuple[int, int, int]]) -> list[tuple[int, int, int, int]]:
    """Merge runs that continue on the next pixel line and overlap."""
    rules: list[tuple[int, int, int, int]] = []
    open_rules: list[list[int]] = []
    for index, cross0, cross1 in runs:
        matched: list[int] | None = None
        for rule in open_rules:
            if rule[1] == index and _overlap_ratio(rule[2], rule[3], cross0, cross1) >= 0.6:
                matched = rule
                break
        if matched is None:
            open_rules.append([index, index + 1, cross0, cross1])
        else:
            matched[1] = index + 1
            matched[2] = min(matched[2], cross0)
            matched[3] = max(matched[3], cross1)
        still_open: list[list[int]] = []
        for rule in open_rules:
            if rule[1] < index:
                rules.append((rule[0], rule[1], rule[2], rule[3]))
            else:
                still_open.append(rule)
        open_rules = still_open
    for rule in open_rules:
        rules.append((rule[0], rule[1], rule[2], rule[3]))
    return rules


def _table_boxes(
    horizontal: Sequence[_HRule],
    vertical: Sequence[_VRule],
    width: int,
    height: int,
    settings: DetectionSettings,
) -> list[DetectedBox]:
    margin = settings.border_margin_px
    interior_h = [
        rule for rule in horizontal if margin < rule.y_center < height - margin
    ]
    interior_v = [
        rule for rule in vertical if margin < rule.x_center < width - margin
    ]
    if len(interior_h) < settings.table_min_h_rules or len(interior_v) < settings.table_min_v_rules:
        return []
    parent = list(range(len(interior_h)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(len(interior_h)):
        for right in range(left + 1, len(interior_h)):
            top = interior_h[left]
            bottom = interior_h[right]
            if top.y0 > bottom.y0:
                top, bottom = bottom, top
            gap = bottom.y0 - top.y1
            if gap < 0 or gap > settings.table_max_rule_gap_px:
                continue
            if _overlap_ratio(top.x0, top.x1, bottom.x0, bottom.x1) < 0.5:
                continue
            if _vertical_rule_connects(top, bottom, interior_v):
                union(left, right)
    grouped: dict[int, list[_HRule]] = {}
    for index, rule in enumerate(interior_h):
        grouped.setdefault(find(index), []).append(rule)
    tables: list[DetectedBox] = []
    for group in grouped.values():
        if len(group) < settings.table_min_h_rules:
            continue
        x0 = min(rule.x0 for rule in group)
        x1 = max(rule.x1 for rule in group)
        y0 = min(rule.y0 for rule in group)
        y1 = max(rule.y1 for rule in group)
        crossing = [
            rule
            for rule in interior_v
            if rule.y0 < y1 and rule.y1 > y0 and rule.x0 < x1 and rule.x1 > x0
        ]
        if len(crossing) < settings.table_min_v_rules:
            continue
        x0 = min(x0, min(rule.x0 for rule in crossing))
        x1 = max(x1, max(rule.x1 for rule in crossing))
        y0 = min(y0, min(rule.y0 for rule in crossing))
        y1 = max(y1, max(rule.y1 for rule in crossing))
        box = DetectedBox(int(x0), int(y0), int(x1), int(y1), RegionType.TABLE)
        if _is_page_frame(box, width, height, settings):
            continue
        tables.append(box)
    return _merge_touching_tables(tables)


def _vertical_rule_connects(top: _HRule, bottom: _HRule, vertical: Sequence[_VRule]) -> bool:
    for rule in vertical:
        if rule.x_center < top.x0 or rule.x_center >= top.x1:
            continue
        if rule.x_center < bottom.x0 or rule.x_center >= bottom.x1:
            continue
        if rule.y0 <= top.y1 + 2 and rule.y1 >= bottom.y0 - 2:
            return True
    return False


def _is_page_frame(box: DetectedBox, width: int, height: int, settings: DetectionSettings) -> bool:
    area = (box.x1 - box.x0) * (box.y1 - box.y0)
    if area < settings.frame_page_fraction * width * height:
        return False
    margin = settings.border_margin_px
    touches_border = (
        box.x0 <= margin
        and box.y0 <= margin
        and box.x1 >= width - margin
        and box.y1 >= height - margin
    )
    return touches_border


def _merge_touching_tables(tables: Sequence[DetectedBox]) -> list[DetectedBox]:
    pending = list(tables)
    merged: list[DetectedBox] = []
    while pending:
        current = pending.pop()
        rest: list[DetectedBox] = []
        for other in pending:
            if _boxes_touch(current, other):
                current = DetectedBox(
                    min(current.x0, other.x0),
                    min(current.y0, other.y0),
                    max(current.x1, other.x1),
                    max(current.y1, other.y1),
                    RegionType.TABLE,
                )
            else:
                rest.append(other)
        if len(rest) == len(pending):
            merged.append(current)
            pending = rest
        else:
            pending = [current, *rest]
    return merged


def _boxes_touch(left: DetectedBox, right: DetectedBox) -> bool:
    return not (
        left.x1 < right.x0 or right.x1 < left.x0 or left.y1 < right.y0 or right.y1 < left.y0
    )


def _ink_for_text(
    ink: np.ndarray,
    vertical: Sequence[_VRule],
    tables: Sequence[DetectedBox],
    height: int,
) -> np.ndarray:
    text_ink = ink.copy()
    for box in tables:
        text_ink[box.y0 : box.y1, box.x0 : box.x1] = False
    minimum_length = max(80, height // 8)
    for rule in vertical:
        if rule.y1 - rule.y0 < minimum_length:
            continue
        if any(_rule_inside_table(rule, box) for box in tables):
            continue
        text_ink[rule.y0 : rule.y1, rule.x0 : rule.x1] = False
    return text_ink


def _rule_inside_table(rule: _VRule, box: DetectedBox) -> bool:
    return (
        rule.x0 >= box.x0
        and rule.x1 <= box.x1
        and rule.y0 >= box.y0
        and rule.y1 <= box.y1
    )


def _segment_blocks(ink: np.ndarray, settings: DetectionSettings) -> list[DetectedBox]:
    height, width = ink.shape
    return _segment(ink, 0, 0, height, width, settings, depth=0)


def _segment(
    ink: np.ndarray,
    y0: int,
    x0: int,
    y1: int,
    x1: int,
    settings: DetectionSettings,
    *,
    depth: int,
) -> list[DetectedBox]:
    if y1 <= y0 or x1 <= x0 or depth > 16:
        return []
    sub = ink[y0:y1, x0:x1]
    row_counts = sub.sum(axis=1)
    line_bands = _content_spans(
        row_counts, settings.intra_line_gap_px, settings.row_ink_tolerance
    )
    blocks = _merge_line_bands(line_bands, settings)
    if len(blocks) >= 2:
        boxes: list[DetectedBox] = []
        for band_y0, band_y1 in blocks:
            boxes.extend(
                _segment(ink, y0 + band_y0, x0, y0 + band_y1, x1, settings, depth=depth + 1)
            )
        return boxes
    col_counts = sub.sum(axis=0)
    columns = _content_spans(col_counts, settings.column_gap_px, settings.row_ink_tolerance)
    if len(columns) >= 2:
        boxes = []
        for band_x0, band_x1 in columns:
            boxes.extend(
                _segment(ink, y0, x0 + band_x0, y1, x0 + band_x1, settings, depth=depth + 1)
            )
        return boxes
    rows = np.flatnonzero(row_counts > settings.row_ink_tolerance)
    cols = np.flatnonzero(col_counts > settings.row_ink_tolerance)
    if rows.size == 0 or cols.size == 0:
        return []
    return [
        DetectedBox(
            int(x0 + cols[0]),
            int(y0 + rows[0]),
            int(x0 + cols[-1] + 1),
            int(y0 + rows[-1] + 1),
            RegionType.TEXT_BLOCK,
        )
    ]


def _merge_line_bands(
    bands: Sequence[tuple[int, int]], settings: DetectionSettings
) -> list[tuple[int, int]]:
    """Join text lines into a block. A wider-than-usual gap starts a new block."""
    if len(bands) <= 1:
        return list(bands)
    gaps = [right[0] - left[1] for left, right in zip(bands, bands[1:], strict=False)]
    typical = float(np.median(np.asarray(gaps, dtype=np.float64)))
    scaled = int(round(typical * settings.spacing_split_ratio))
    split_at = min(settings.max_block_gap_px, max(settings.block_gap_y_px, scaled))
    merged: list[tuple[int, int]] = []
    start, end = bands[0]
    for (band_start, band_end), gap in zip(bands[1:], gaps, strict=True):
        if gap >= split_at:
            merged.append((start, end))
            start, end = band_start, band_end
        else:
            end = band_end
    merged.append((start, end))
    return merged


def _content_spans(counts: np.ndarray, min_gap: int, tolerance: int) -> list[tuple[int, int]]:
    empty = counts <= tolerance
    spans: list[tuple[int, int]] = []
    start: int | None = None
    gap_run = 0
    for index, is_empty in enumerate(empty.tolist()):
        if not is_empty:
            if start is None:
                start = index
            gap_run = 0
            continue
        if start is None:
            continue
        gap_run += 1
        if gap_run >= min_gap:
            end = index - gap_run + 1
            if end > start:
                spans.append((start, end))
            start = None
            gap_run = 0
    if start is not None:
        spans.append((start, len(counts)))
    return spans


def _overlaps_table(box: DetectedBox, tables: Sequence[DetectedBox]) -> bool:
    area = (box.x1 - box.x0) * (box.y1 - box.y0)
    if area <= 0:
        return False
    for table in tables:
        ix0 = max(box.x0, table.x0)
        iy0 = max(box.y0, table.y0)
        ix1 = min(box.x1, table.x1)
        iy1 = min(box.y1, table.y1)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        if (ix1 - ix0) * (iy1 - iy0) / area >= 0.5:
            return True
    return False


def _pad_box(box: DetectedBox, padding: int, width: int, height: int) -> DetectedBox:
    return DetectedBox(
        max(0, box.x0 - padding),
        max(0, box.y0 - padding),
        min(width, box.x1 + padding),
        min(height, box.y1 + padding),
        box.region_type,
    )


def _keep_box(box: DetectedBox, ink: np.ndarray, settings: DetectionSettings) -> bool:
    width = box.x1 - box.x0
    height = box.y1 - box.y0
    if width < settings.min_region_width_px or height < settings.min_region_height_px:
        return False
    ink_count = int(ink[box.y0 : box.y1, box.x0 : box.x1].sum())
    return ink_count >= settings.min_region_ink_px


def _require_inside(box: DetectedBox, width: int, height: int) -> None:
    try:
        assert_bbox_within_page(BBox(box.x0, box.y0, box.x1, box.y1), width, height)
    except SchemaError as exc:
        raise DetectionError(
            f"detector box [{box.x0}, {box.y0}, {box.x1}, {box.y1}] is outside the page"
        ) from exc


def _overlap_ratio(left0: int, left1: int, right0: int, right1: int) -> float:
    intersection = min(left1, right1) - max(left0, right0)
    if intersection <= 0:
        return 0.0
    shorter = min(left1 - left0, right1 - right0)
    if shorter <= 0:
        return 0.0
    return intersection / shorter


def _write_debug(
    source: Path,
    debug_dir: Path,
    document_id: str,
    page_number: int,
    records: Sequence[RegionRecord],
) -> None:
    destination = debug_dir / document_id / f"page_{page_number:04d}.png"
    if destination.resolve() == source.resolve():
        raise DetectionError("debug image would overwrite the rendered page")
    with Image.open(source) as image:
        canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for record in records:
        box = record.bbox
        color = (180, 40, 40) if record.region_type is RegionType.TABLE else (20, 90, 180)
        draw.rectangle((box.x0, box.y0, box.x1 - 1, box.y1 - 1), outline=color, width=3)
        draw.text((box.x0 + 4, box.y0 + 4), record.region_id.rsplit("_", 1)[-1], fill=color)
    buffer = BytesIO()
    canvas.save(buffer, format="PNG")
    canvas.close()
    _write_bytes_atomic(destination, buffer.getvalue())


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".partial"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
        raise


if __name__ == "__main__":
    sys.exit(main())
