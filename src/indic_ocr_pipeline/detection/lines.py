"""Detect physical text lines inside region boxes and write one crop each.

A physical line is one printed horizontal band of ink. It is not a sentence.
This stage does not read text, call OCR, or modify the rendered page.

Line IDs follow the existing page-scoped contract ``{page_id}_L####``.
``parent_region_id`` is the phase-4 block. ``reading_order`` is the 1-based
index of the line inside that block. Boxes are ``[x0, y0, x1, y1]`` in
rendered-page pixels, origin top-left, ``x1`` and ``y1`` exclusive.

The line geometry is a ``RegionRecord`` with ``region_type`` ``line`` in
``manifests/lines.jsonl``. The PNG path and crop hash are the matching
``CropRecord`` in ``manifests/crops.jsonl``. Neither record stores text.
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
from indic_ocr_pipeline.detection.regions import _content_spans, _ink_mask
from indic_ocr_pipeline.errors import DatasetError, ManifestConflictError, SchemaError
from indic_ocr_pipeline.ids import make_crop_id, make_line_id, make_region_id
from indic_ocr_pipeline.manifests import JsonlManifest
from indic_ocr_pipeline.pdf.ingest import source_path_for
from indic_ocr_pipeline.rendering.pages import resolve_existing_path
from indic_ocr_pipeline.schemas import (
    BBox,
    CropRecord,
    CropType,
    RegionRecord,
    RegionType,
    StageStatus,
    assert_bbox_within_page,
)
from indic_ocr_pipeline.utils.hashing import sha256_bytes, sha256_file
from indic_ocr_pipeline.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

LINE_DETECTOR_VERSION = "physical_lines_v1"
HIGH_LINE_COUNT = 40
RESUME_NOTE = "Existing line rows were kept. Line detection can resume."
_PARENT_TYPES = frozenset(
    {
        RegionType.TEXT_BLOCK.value,
        RegionType.TABLE.value,
        RegionType.HEADER.value,
        RegionType.FOOTER.value,
        RegionType.UNKNOWN.value,
    }
)


class LineDetectionError(DatasetError):
    """A page's physical lines could not be written."""


class LineConflictError(LineDetectionError):
    """A line id already exists with a different payload, or a crop hash disagrees."""


@dataclass(frozen=True)
class PhysicalLine:
    """One line box in page coordinates, before an id is assigned."""

    x0: int
    y0: int
    x1: int
    y1: int
    parent_region_id: str
    reading_order: int


@dataclass
class _PreparedLine:
    line: RegionRecord
    crop: CropRecord
    png: bytes
    crop_file: Path


@dataclass
class DocumentLineStats:
    """Line counters for one source document."""

    document_id: str
    pages: int = 0
    regions: int = 0
    lines: int = 0
    crops: int = 0
    skipped: int = 0
    zero_regions: int = 0
    one_line_regions: int = 0
    high_regions: int = 0
    errors: int = 0
    message: str = ""


@dataclass
class LineReport:
    """Counters and boundary diagnostics for one line-detection run."""

    pages: int = 0
    regions: int = 0
    lines: int = 0
    crops: int = 0
    skipped: int = 0
    zero_regions: int = 0
    one_line_regions: int = 0
    high_regions: int = 0
    errors: int = 0
    heights: list[int] = field(default_factory=list)
    widths: list[int] = field(default_factory=list)
    details: list[DocumentLineStats] = field(default_factory=list)

    def add(self, stats: DocumentLineStats) -> None:
        self.details.append(stats)
        self.pages += stats.pages
        self.regions += stats.regions
        self.lines += stats.lines
        self.crops += stats.crops
        self.skipped += stats.skipped
        self.zero_regions += stats.zero_regions
        self.one_line_regions += stats.one_line_regions
        self.high_regions += stats.high_regions
        self.errors += stats.errors


def line_method_label(settings: DetectionSettings) -> str:
    """Return a stable label for this line detector and its configuration."""
    return (
        f"{LINE_DETECTOR_VERSION}"
        f";split={settings.line_split_gap_px}"
        f";pad={settings.line_padding_px}"
        f";min={settings.min_physical_line_width_px}x{settings.min_physical_line_height_px}"
        f";ink={settings.min_physical_line_ink_px}"
        f";rule_h={settings.rule_band_max_height_px}"
        f";rule_fill={settings.rule_band_min_row_fill}"
        f";busy={settings.rule_band_max_busy_rows}@{settings.rule_band_busy_fill}"
        f";suppress={settings.rule_suppress_max_height_px}@{settings.rule_suppress_row_fill}"
        f";empty={settings.line_row_empty_fraction}"
        f";matra={settings.matra_band_max_height_px}/{settings.matra_merge_gap_px}"
        f";tall={settings.tall_line_px}"
        f";bin={settings.binarization}"
    )


def detect_lines_in_region(
    gray: np.ndarray,
    bbox: BBox,
    parent_region_id: str,
    settings: DetectionSettings,
) -> list[PhysicalLine]:
    """Return physical lines inside one region, top-to-bottom then left-to-right."""
    if gray.ndim != 2:
        raise LineDetectionError("line detection expects a grayscale page")
    page_height, page_width = gray.shape
    assert_bbox_within_page(bbox, page_width, page_height)
    crop = gray[bbox.y0 : bbox.y1, bbox.x0 : bbox.x1]
    if crop.size == 0:
        return []
    ink = _ink_mask(crop, settings)
    ink = _clear_vertical_rules(ink, settings)
    bands = _line_bands(ink, settings)
    raw_boxes: list[list[int]] = []
    for band_y0, band_y1 in bands:
        band = ink[band_y0:band_y1]
        if band.size == 0 or _is_horizontal_rule(band, crop.shape[1], settings):
            continue
        columns = np.flatnonzero(band.any(axis=0))
        if columns.size == 0:
            continue
        raw_boxes.append([int(columns[0]), band_y0, int(columns[-1]) + 1, band_y1])
    lines: list[PhysicalLine] = []
    for x0, band_y0, x1, band_y1 in _merge_attached_fragments(raw_boxes, settings):
        raw = _pad_line(
            x0,
            band_y0,
            x1,
            band_y1,
            crop.shape[1],
            crop.shape[0],
            settings.line_padding_px,
        )
        if not _keep_line(raw, ink, settings):
            continue
        page_box = BBox(
            bbox.x0 + raw[0],
            bbox.y0 + raw[1],
            bbox.x0 + raw[2],
            bbox.y0 + raw[3],
        )
        assert_bbox_within_page(page_box, page_width, page_height)
        if not bbox.contains(page_box):
            raise LineDetectionError(
                f"{parent_region_id} line box {page_box.as_list()} leaves its region"
            )
        lines.append(
            PhysicalLine(
                page_box.x0,
                page_box.y0,
                page_box.x1,
                page_box.y1,
                parent_region_id,
                0,
            )
        )
    lines.sort(key=lambda item: (item.y0, item.x0, item.y1, item.x1))
    return [
        PhysicalLine(item.x0, item.y0, item.x1, item.y1, parent_region_id, index)
        for index, item in enumerate(lines, start=1)
    ]


def detect_physical_lines(
    page_rows: Sequence[Mapping[str, object]],
    region_rows: Sequence[Mapping[str, object]],
    line_manifest: JsonlManifest,
    crop_manifest: JsonlManifest,
    *,
    repo_root: Path,
    crops_dir: Path,
    settings: DetectionSettings,
    debug_dir: Path | None = None,
) -> LineReport:
    """Detect lines for every parent region, one page image at a time."""
    pages = sorted(
        (dict(row) for row in page_rows),
        key=lambda row: (str(row.get("document_id")), int(row.get("page_number", 0))),
    )
    regions_by_page: dict[str, list[dict[str, object]]] = {}
    for row in region_rows:
        if str(row.get("region_type")) not in _PARENT_TYPES:
            continue
        regions_by_page.setdefault(str(row.get("page_id")), []).append(dict(row))
    for rows in regions_by_page.values():
        rows.sort(key=lambda row: (int(row.get("region_index", 0)), str(row.get("region_id"))))
    existing_lines = _group(line_manifest.load(), "page_id")
    existing_crops = _group(crop_manifest.load(), "page_id")
    report = LineReport()
    current_id = ""
    stats: DocumentLineStats | None = None
    for page in pages:
        document_id = str(page.get("document_id", ""))
        if document_id != current_id:
            if stats is not None:
                report.add(stats)
            current_id = document_id
            stats = DocumentLineStats(document_id=document_id)
        assert stats is not None
        stats.pages += 1
        page_id = str(page.get("page_id", ""))
        try:
            page_lines, skipped, heights, widths = _detect_page(
                page,
                regions_by_page.get(page_id, []),
                existing_lines.get(page_id, []),
                existing_crops.get(page_id, []),
                line_manifest,
                crop_manifest,
                repo_root=repo_root,
                crops_dir=crops_dir,
                settings=settings,
                debug_dir=debug_dir,
            )
        except (LineDetectionError, SchemaError, OSError, ManifestConflictError) as exc:
            stats.errors += 1
            if not stats.message:
                stats.message = str(exc)
            logger.error("%s", exc)
            continue
        counts: dict[str, int] = {}
        for item in page_lines:
            counts[item.parent_region_id] = counts.get(item.parent_region_id, 0) + 1
        parent_count = len(regions_by_page.get(page_id, []))
        stats.regions += parent_count
        stats.lines += len(page_lines)
        stats.crops += len(page_lines)
        stats.skipped += skipped
        found = set(counts)
        stats.zero_regions += parent_count - len(found)
        stats.one_line_regions += sum(1 for count in counts.values() if count == 1)
        stats.high_regions += sum(1 for count in counts.values() if count > HIGH_LINE_COUNT)
        report.heights.extend(heights)
        report.widths.extend(widths)
    if stats is not None:
        report.add(stats)
    return report


def format_report(report: LineReport) -> str:
    """Return the operator summary and the boundary audit."""
    lines = [
        "Phase 5: Physical line detection",
        "",
        f"Pages discovered: {report.pages}",
        f"Regions discovered: {report.regions}",
        "",
    ]
    for stats in report.details:
        lines.append(stats.document_id)
        lines.append(f"  pages: {stats.pages}")
        lines.append(f"  regions: {stats.regions}")
        lines.append(f"  lines: {stats.lines}")
        lines.append(f"  crops: {stats.crops}")
        lines.append(f"  skipped: {stats.skipped}")
        if stats.message:
            lines.append(f"  error: {stats.message}")
        lines.append("")
    heights = np.asarray(report.heights, dtype=np.int64)
    widths = np.asarray(report.widths, dtype=np.int64)
    tall, short = _suspicious_counts(heights)
    lines.extend(
        [
            "Detection complete.",
            f"  pages: {report.pages}",
            f"  regions: {report.regions}",
            f"  lines: {report.lines}",
            f"  crops: {report.crops}",
            f"  skipped: {report.skipped}",
            f"  errors: {report.errors}",
            "",
            "Boundary audit",
            f"  regions with zero lines: {report.zero_regions}",
            f"  regions with exactly one line: {report.one_line_regions}",
            f"  regions with more than {HIGH_LINE_COUNT} lines: {report.high_regions}",
            f"  line height px: min={_stat(heights, 'min')} "
            f"median={_stat(heights, 'median')} max={_stat(heights, 'max')}",
            f"  line width px: min={_stat(widths, 'min')} "
            f"median={_stat(widths, 'median')} max={_stat(widths, 'max')}",
            f"  suspiciously tall crops: {tall}",
            f"  suspiciously short crops: {short}",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Detect physical lines for every region in the region manifest."""
    from indic_ocr_pipeline.config import load_settings
    from indic_ocr_pipeline.manifests import open_manifest

    parser = argparse.ArgumentParser(
        description="Detect physical text lines and write full-line crops. No OCR is run."
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Write line overlays under data/debug/lines.",
    )
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)
    started = datetime.now(UTC)
    try:
        settings = load_settings()
        pages = open_manifest(settings.manifests_dir, "pages")
        regions = open_manifest(settings.manifests_dir, "regions")
        line_manifest = open_manifest(settings.manifests_dir, "lines")
        crop_manifest = open_manifest(settings.manifests_dir, "crops")
        debug_dir = None
        if args.debug or settings.detection.debug_images:
            debug_dir = settings.repo_root / "data" / "debug" / "lines"
        report = detect_physical_lines(
            pages.load(),
            regions.load(),
            line_manifest,
            crop_manifest,
            repo_root=settings.repo_root,
            crops_dir=settings.crops_full_dir,
            settings=settings.detection,
            debug_dir=debug_dir,
        )
    except (LineDetectionError, ManifestConflictError, SchemaError) as exc:
        print(f"ERROR  {exc}", file=sys.stderr)
        logger.error("%s", exc)
        return 1
    duration = (datetime.now(UTC) - started).total_seconds()
    print(format_report(report), flush=True)
    logger.info(
        "line detection finished",
        extra={
            "processed": report.regions,
            "completed": report.lines - report.skipped,
            "skipped": report.skipped,
            "failed": report.errors,
            "duration_seconds": round(duration, 3),
        },
    )
    return 1 if report.errors else 0


def _detect_page(
    page: Mapping[str, object],
    regions: Sequence[Mapping[str, object]],
    stored_lines: Sequence[Mapping[str, object]],
    stored_crops: Sequence[Mapping[str, object]],
    line_manifest: JsonlManifest,
    crop_manifest: JsonlManifest,
    *,
    repo_root: Path,
    crops_dir: Path,
    settings: DetectionSettings,
    debug_dir: Path | None,
) -> tuple[list[PhysicalLine], int, list[int], list[int]]:
    page_id = str(page.get("page_id", ""))
    document_id = str(page.get("document_id", ""))
    page_number = page.get("page_number")
    if not isinstance(page_number, int) or isinstance(page_number, bool):
        raise LineDetectionError(f"{page_id} has no integer page_number")
    image_path = resolve_existing_path(str(page.get("image_path", "")), repo_root)
    if not image_path.is_file():
        raise LineDetectionError(f"{page_id} rendered page image is missing: {image_path}")
    recorded_sha = str(page.get("sha256", ""))
    if sha256_file(image_path) != recorded_sha:
        raise LineConflictError(
            f"{page_id} page image SHA-256 does not match the page manifest. {RESUME_NOTE}"
        )
    gray, pixels = _load_page(image_path, page_id)
    recorded_width = page.get("width")
    recorded_height = page.get("height")
    if recorded_width != gray.shape[1] or recorded_height != gray.shape[0]:
        raise LineConflictError(
            f"{page_id} page image size does not match the page manifest. {RESUME_NOTE}"
        )
    _require_index_room(regions, settings.line_region_index_base, page_id)
    detected: list[PhysicalLine] = []
    for region in regions:
        parent_id = str(region.get("region_id", ""))
        box = BBox.from_list(region.get("bbox"))
        try:
            detected.extend(detect_lines_in_region(gray, box, parent_id, settings))
        except SchemaError as exc:
            raise LineDetectionError(f"{parent_id} {exc}") from exc
    prepared = _prepare_lines(
        detected,
        pixels,
        document_id=document_id,
        page_id=page_id,
        page_number=page_number,
        crops_dir=crops_dir,
        repo_root=repo_root,
        settings=settings,
    )
    skipped = _commit_lines(
        prepared,
        stored_lines,
        stored_crops,
        line_manifest,
        crop_manifest,
        page_id=page_id,
    )
    if debug_dir is not None:
        _write_debug(image_path, debug_dir, document_id, page_number, regions, prepared)
    del gray, pixels
    heights = [item.line.bbox.height for item in prepared]
    widths = [item.line.bbox.width for item in prepared]
    return detected, skipped, heights, widths


def _prepare_lines(
    detected: Sequence[PhysicalLine],
    pixels: np.ndarray,
    *,
    document_id: str,
    page_id: str,
    page_number: int,
    crops_dir: Path,
    repo_root: Path,
    settings: DetectionSettings,
) -> list[_PreparedLine]:
    method = line_method_label(settings)
    prepared: list[_PreparedLine] = []
    for ordinal, item in enumerate(detected, start=1):
        line_index = ordinal
        region_index = settings.line_region_index_base + line_index - 1
        line_id = make_line_id(page_id, line_index)
        bbox = BBox(item.x0, item.y0, item.x1, item.y1)
        png = _crop_png(pixels, bbox)
        digest = sha256_bytes(png)
        crop_file = crops_dir / document_id / page_id / f"{line_id}.png"
        line = RegionRecord(
            region_id=make_region_id(page_id, region_index),
            document_id=document_id,
            page_id=page_id,
            page_number=page_number,
            region_index=region_index,
            region_type=RegionType.LINE,
            bbox=bbox,
            detection_method=method,
            confidence=None,
            parent_region_id=item.parent_region_id,
            line_id=line_id,
            line_index=line_index,
            reading_order=item.reading_order,
            status=StageStatus.COMPLETE,
        )
        crop = CropRecord(
            crop_id=make_crop_id(line_id, CropType.FULL.value, 1),
            document_id=document_id,
            page_id=page_id,
            page_number=page_number,
            region_id=item.parent_region_id,
            parent_line_id=line_id,
            crop_type=CropType.FULL,
            crop_index=1,
            image_path=source_path_for(crop_file, repo_root),
            bbox=bbox,
            parent_bbox=bbox,
            sha256=digest,
            status=StageStatus.COMPLETE,
        )
        prepared.append(_PreparedLine(line, crop, png, crop_file))
    return prepared


def _commit_lines(
    prepared: Sequence[_PreparedLine],
    stored_lines: Sequence[Mapping[str, object]],
    stored_crops: Sequence[Mapping[str, object]],
    line_manifest: JsonlManifest,
    crop_manifest: JsonlManifest,
    *,
    page_id: str,
) -> int:
    line_payloads = [item.line.to_dict() for item in prepared]
    crop_payloads = [item.crop.to_dict() for item in prepared]
    stored_line_map = {str(row.get("line_id")): dict(row) for row in stored_lines}
    stored_crop_map = {str(row.get("crop_id")): dict(row) for row in stored_crops}
    new_line_ids = {str(payload["line_id"]) for payload in line_payloads}
    new_crop_ids = {str(payload["crop_id"]) for payload in crop_payloads}
    if set(stored_line_map) - new_line_ids or set(stored_crop_map) - new_crop_ids:
        raise LineConflictError(
            f"{page_id} line set changed and an existing line would be left behind. {RESUME_NOTE}"
        )
    pending: list[_PreparedLine] = []
    skipped = 0
    for item, line_payload, crop_payload in zip(
        prepared, line_payloads, crop_payloads, strict=True
    ):
        line_id = str(line_payload["line_id"])
        crop_id = str(crop_payload["crop_id"])
        current_line = stored_line_map.get(line_id)
        current_crop = stored_crop_map.get(crop_id)
        if current_line is not None and current_line != line_payload:
            raise LineConflictError(
                f"{line_id} already exists with a different payload. {RESUME_NOTE}"
            )
        if current_crop is not None and current_crop != crop_payload:
            raise LineConflictError(
                f"{line_id} already exists with a different payload. {RESUME_NOTE}"
            )
        if current_line is None or current_crop is None:
            pending.append(item)
            continue
        _install_matching_crop(item, current_crop)
        skipped += 1
    for item in pending:
        try:
            line_manifest.append(item.line.to_dict())
            crop_manifest.append(item.crop.to_dict())
        except ManifestConflictError as exc:
            raise LineConflictError(str(exc)) from exc
        _install_new_crop(item)
    return skipped


def _install_matching_crop(item: _PreparedLine, stored_crop: Mapping[str, object] | None) -> None:
    recorded = "" if stored_crop is None else str(stored_crop.get("sha256") or "")
    fresh = sha256_bytes(item.png)
    if recorded and fresh != recorded:
        raise LineConflictError(
            f"{item.line.line_id} crop bytes do not match the recorded SHA-256. {RESUME_NOTE}"
        )
    path = item.crop_file
    if not path.is_file():
        _write_bytes_atomic(path, item.png)
        return
    if sha256_file(path) != fresh:
        raise LineConflictError(
            f"{item.line.line_id} existing crop SHA-256 does not match the manifest. {RESUME_NOTE}"
        )


def _install_new_crop(item: _PreparedLine) -> None:
    path = item.crop_file
    if path.is_file() and sha256_file(path) != sha256_bytes(item.png):
        raise LineConflictError(
            f"{item.line.line_id} crop file already exists with a different SHA-256. {RESUME_NOTE}"
        )
    if not path.is_file():
        _write_bytes_atomic(path, item.png)
    written = _read_crop_size(path)
    if written != (item.line.bbox.width, item.line.bbox.height):
        raise LineDetectionError(f"{item.line.line_id} crop size does not match its bbox")
    if sha256_file(path) != item.crop.sha256:
        raise LineDetectionError(f"{item.line.line_id} crop hash changed while it was written")


def _crop_png(pixels: np.ndarray, bbox: BBox) -> bytes:
    piece = pixels[bbox.y0 : bbox.y1, bbox.x0 : bbox.x1]
    if piece.size == 0:
        raise LineDetectionError("line crop is empty")
    image = Image.fromarray(piece)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image.close()
    payload = buffer.getvalue()
    if len(payload) < 1:
        raise LineDetectionError("line crop PNG is empty")
    return payload


def _read_crop_size(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise LineDetectionError(f"{path} is not a valid PNG") from exc
    if width < 1 or height < 1 or path.stat().st_size < 1:
        raise LineDetectionError(f"{path} is an empty crop")
    return width, height


def _load_page(path: Path, page_id: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        with Image.open(path) as image:
            if image.mode == "L":
                pixels = np.array(image, dtype=np.uint8, copy=True)
                gray = pixels
            elif image.mode == "RGB":
                pixels = np.array(image, dtype=np.uint8, copy=True)
                gray = np.array(image.convert("L"), dtype=np.uint8, copy=True)
            else:
                rgb = image.convert("RGB")
                pixels = np.array(rgb, dtype=np.uint8, copy=True)
                gray = np.array(rgb.convert("L"), dtype=np.uint8, copy=True)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise LineDetectionError(f"{page_id} rendered page image is not a readable PNG") from exc
    if gray.ndim != 2 or gray.size == 0:
        raise LineDetectionError(f"{page_id} rendered page image is not a readable PNG")
    return gray, pixels


def _line_bands(ink: np.ndarray, settings: DetectionSettings) -> list[tuple[int, int]]:
    counts = _line_counts(ink, settings)
    tolerance = _line_tolerance(ink.shape[1], settings)
    refined: list[tuple[int, int]] = []
    finer_gap = max(2, settings.line_split_gap_px // 2)
    for start, end in _closed_spans(counts, settings.line_split_gap_px, tolerance):
        if end - start <= settings.tall_line_px:
            refined.append((start, end))
            continue
        inner = _closed_spans(counts[start:end], finer_gap, tolerance)
        if len(inner) <= 1:
            refined.append((start, end))
            continue
        refined.extend((start + inner_start, start + inner_end) for inner_start, inner_end in inner)
    return refined


def _closed_spans(counts: np.ndarray, min_gap: int, tolerance: int) -> list[tuple[int, int]]:
    """Content spans with leading and trailing empty rows removed."""
    spans: list[tuple[int, int]] = []
    for start, end in _content_spans(counts, min_gap, tolerance):
        while start < end and int(counts[start]) <= tolerance:
            start += 1
        while end > start and int(counts[end - 1]) <= tolerance:
            end -= 1
        if end > start:
            spans.append((start, end))
    return spans


def _clear_vertical_rules(ink: np.ndarray, settings: DetectionSettings) -> np.ndarray:
    height, _width = ink.shape
    if height < settings.vertical_rule_min_region_px:
        return ink
    minimum = max(
        settings.vertical_rule_min_run_px,
        int(height * settings.vertical_rule_min_fraction),
    )
    cleared = ink.copy()
    for column in range(cleared.shape[1]):
        line = cleared[:, column]
        padded = np.concatenate((np.zeros(1, dtype=bool), line, np.zeros(1, dtype=bool)))
        changes = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        for start, end in zip(starts, ends, strict=True):
            if int(end - start) >= minimum:
                cleared[int(start) : int(end), column] = False
    return cleared


def _line_tolerance(width: int, settings: DetectionSettings) -> int:
    return max(settings.row_ink_tolerance, int(width * settings.line_row_empty_fraction))


def _line_counts(ink: np.ndarray, settings: DetectionSettings) -> np.ndarray:
    """Row ink counts after thin ruling rows are blanked."""
    projection = ink.copy()
    _blank_rule_rows(projection, settings)
    return projection.sum(axis=1)


def _blank_rule_rows(ink: np.ndarray, settings: DetectionSettings) -> None:
    """Clear thin, dense rows that sit in whitespace so they cannot join text lines."""
    height, width = ink.shape
    if width < 1 or height < 1:
        return
    fill = ink.mean(axis=1)
    high = fill >= settings.rule_suppress_row_fill
    neighbor_limit = settings.line_row_empty_fraction
    index = 0
    while index < height:
        if not bool(high[index]):
            index += 1
            continue
        end = index + 1
        while end < height and bool(high[end]):
            end += 1
        above = float(fill[index - 1]) if index > 0 else 0.0
        below = float(fill[end]) if end < height else 0.0
        isolated = above < neighbor_limit and below < neighbor_limit
        if end - index <= settings.rule_suppress_max_height_px and isolated:
            ink[index:end] = False
        index = end


def _is_horizontal_rule(band: np.ndarray, region_width: int, settings: DetectionSettings) -> bool:
    height = band.shape[0]
    if height < 1 or height > settings.rule_band_max_height_px or region_width < 1:
        return False
    covered = int(band.any(axis=0).sum())
    if covered < settings.rule_band_min_width_fraction * region_width:
        return False
    row_fill = band.mean(axis=1) if band.size else np.zeros(0)
    if row_fill.size == 0 or float(row_fill.max()) < settings.rule_band_min_row_fill:
        return False
    busy = int((row_fill >= settings.rule_band_busy_fill).sum())
    return busy <= settings.rule_band_max_busy_rows


def _merge_attached_fragments(
    boxes: list[list[int]], settings: DetectionSettings
) -> list[tuple[int, int, int, int]]:
    """Join a short fragment to the single taller line it touches.

    A fragment between two lines is left alone so two printed lines stay apart.
    """
    pending = [list(box) for box in boxes]
    changed = True
    while changed:
        changed = False
        pending.sort(key=lambda box: (box[1], box[0], box[3], box[2]))
        consumed = [False] * len(pending)
        for index, box in enumerate(pending):
            if consumed[index]:
                continue
            if box[3] - box[1] > settings.matra_band_max_height_px:
                continue
            targets: list[int] = []
            for other_index, other in enumerate(pending):
                if other_index == index or consumed[other_index]:
                    continue
                if other[3] - other[1] <= settings.matra_band_max_height_px:
                    continue
                if box[1] >= other[3]:
                    gap = box[1] - other[3]
                else:
                    gap = other[1] - box[3]
                if gap < 0 or gap > settings.matra_merge_gap_px:
                    continue
                overlap = min(box[2], other[2]) - max(box[0], other[0])
                width = box[2] - box[0]
                if width <= 0 or overlap < 0.5 * width:
                    continue
                targets.append(other_index)
            if len(targets) != 1:
                continue
            other = pending[targets[0]]
            other[0] = min(other[0], box[0])
            other[1] = min(other[1], box[1])
            other[2] = max(other[2], box[2])
            other[3] = max(other[3], box[3])
            consumed[index] = True
            changed = True
        if changed:
            pending = [box for index, box in enumerate(pending) if not consumed[index]]
    pending.sort(key=lambda box: (box[1], box[0], box[3], box[2]))
    return [(box[0], box[1], box[2], box[3]) for box in pending]


def _pad_line(
    x0: int, y0: int, x1: int, y1: int, width: int, height: int, padding: int
) -> tuple[int, int, int, int]:
    return (
        max(0, x0 - padding),
        max(0, y0 - padding),
        min(width, x1 + padding),
        min(height, y1 + padding),
    )


def _keep_line(
    raw: tuple[int, int, int, int], ink: np.ndarray, settings: DetectionSettings
) -> bool:
    x0, y0, x1, y1 = raw
    if x1 - x0 < settings.min_physical_line_width_px:
        return False
    if y1 - y0 < settings.min_physical_line_height_px:
        return False
    return int(ink[y0:y1, x0:x1].sum()) >= settings.min_physical_line_ink_px


def _require_index_room(
    regions: Sequence[Mapping[str, object]], base: int, page_id: str
) -> None:
    for region in regions:
        index = region.get("region_index")
        if isinstance(index, int) and not isinstance(index, bool) and index >= base:
            raise LineDetectionError(
                f"{page_id} region index {index} collides with line_region_index_base {base}"
            )


def _group(
    rows: Sequence[Mapping[str, object]], field_name: str
) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(field_name, "")), []).append(dict(row))
    return grouped


def _suspicious_counts(heights: np.ndarray) -> tuple[int, int]:
    if heights.size == 0:
        return 0, 0
    median = float(np.median(heights))
    tall_limit = max(80, int(round(median * 2.2)))
    short_limit = max(8, int(round(median * 0.45)))
    return int(np.sum(heights > tall_limit)), int(np.sum(heights < short_limit))


def _stat(values: np.ndarray, kind: str) -> str:
    if values.size == 0:
        return "n/a"
    if kind == "min":
        return str(int(values.min()))
    if kind == "max":
        return str(int(values.max()))
    return str(int(np.median(values)))


def _write_debug(
    source: Path,
    debug_dir: Path,
    document_id: str,
    page_number: int,
    regions: Sequence[Mapping[str, object]],
    prepared: Sequence[_PreparedLine],
) -> None:
    destination = debug_dir / document_id / f"page_{page_number:04d}.png"
    if destination.resolve() == source.resolve():
        raise LineDetectionError("debug image would overwrite the rendered page")
    with Image.open(source) as image:
        canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for region in regions:
        box = BBox.from_list(region.get("bbox"))
        draw.rectangle((box.x0, box.y0, box.x1 - 1, box.y1 - 1), outline=(20, 90, 180), width=2)
    for item in prepared:
        box = item.line.bbox
        draw.rectangle((box.x0, box.y0, box.x1 - 1, box.y1 - 1), outline=(180, 40, 40), width=2)
        label = item.line.line_id.rsplit("_", 1)[-1]
        draw.text((box.x0 + 2, box.y0 + 1), label, fill=(180, 40, 40))
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
