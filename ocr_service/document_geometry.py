"""Geometry-first page reconstruction (Tesseract word boxes).

Uses spatial coordinates for reading order and row-major tables. The vision
model supplies narrative text when geometry is weak; when a data grid is
detected on the image, the table body is rebuilt from coordinates rather than
from column-major model output.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from PIL import Image

from html_tables import convert_markdown_tables_to_html, has_html_table
from layout_repair import _HAS_TABLE
from page_layout import _has_well_formed_table, _looks_like_column_dump
from table_layout import (
    _tesseract_data_rows,
    _tesseract_table_looks_like_grid,
    _intro_from_image,
    _intro_lines,
    tesseract_lang_for_model,
    tesseract_table_markdown,
)

_VERSION = "geometry-v2"
_WEEKDAY = re.compile(
    r"(સોમવાર|મંગળવાર|બુધવાર|ગુરૂવાર|શુક્રવાર|શનિવાર|રવિવાર|"
    r"सोमवार|मंगळवार|बुधवार|गुरुवार|शुक्रवार|शनिवार|रविवार|"
    r"Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)",
    re.IGNORECASE,
)


@dataclass
class GeometryReport:
    version: str = _VERSION
    geometry_table_rows: int = 0
    fallback_used: bool = True
    table_format: str = "markdown"
    detected_regions: list[str] = field(default_factory=list)


def geometry_version() -> str:
    return _VERSION


def rebuild_page_from_geometry(
    image: Image.Image,
    vlm_text: str,
    *,
    model_id: str,
) -> tuple[str, GeometryReport]:
    """Return (canonical_text, report). Leaves vlm_text unchanged when no grid."""
    report = GeometryReport()
    if os.environ.get("OCR_TABLE_LAYOUT", "1").strip() in {"0", "false", "no"}:
        return vlm_text, report

    lang = tesseract_lang_for_model(model_id)
    table_md = tesseract_table_markdown(image, lang=lang)
    if not table_md or not _tesseract_table_looks_like_grid(table_md):
        return vlm_text, report

    rows = _tesseract_data_rows(table_md)
    report.geometry_table_rows = rows
    if rows < 8:
        return vlm_text, report

    if _has_well_formed_table(vlm_text) and not _vlm_looks_like_broken_table_dump(vlm_text):
        report.fallback_used = False
        report.detected_regions = ["vlm_table_kept"]
        return vlm_text, report

    intro = _intro_from_image(image, lang=lang) or _intro_lines(vlm_text)
    intro = _format_intro_markdown(intro)
    table_body = table_md
    if os.environ.get("OCR_TABLE_FORMAT", "html").strip().lower() == "html":
        if not has_html_table(table_body):
            table_body = convert_markdown_tables_to_html(table_md)
        report.table_format = "html"
    else:
        report.table_format = "markdown"

    parts = [part.strip() for part in (intro, table_body) if part and part.strip()]
    report.fallback_used = False
    report.detected_regions = ["header", "data_table"]
    return "\n\n".join(parts), report


def _vlm_looks_like_broken_table_dump(text: str) -> bool:
    if _looks_like_column_dump(text):
        return True
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 6:
        return False
    weekday_lines = sum(1 for line in lines if _WEEKDAY.search(line) and len(line) < 40)
    headerish = sum(1 for line in lines if "તારીખ" in line or "शक" in line or "वार" in line)
    if weekday_lines >= 4 and headerish >= 1 and not _has_well_formed_table(text):
        return True
    if _HAS_TABLE.search(text) and weekday_lines >= 3:
        return True
    return False


def _format_intro_markdown(intro: str) -> str:
    """Light structure for notice headers without changing OCR characters."""
    lines = [
        line.strip()
        for line in intro.splitlines()
        if line.strip() and not re.fullmatch(r"[_#=*|:.\-]+|ક\s*=?", line.strip())
    ]
    if not lines:
        return intro
    out: list[str] = []
    for index, line in enumerate(lines):
        if index == 0 and len(line) < 80 and not line.startswith("#"):
            out.append(f"# {line}")
        elif index == 1 and len(line) < 80 and not line.startswith("#"):
            out.append(f"## {line}")
        elif index == 2 and (
            "गांधीनगर" in line or "ગાંધીનગર" in line or "सचिवालय" in line or "સચિવાલય" in line
        ):
            out.append(line)
        elif index == 3 and not line.startswith("#"):
            out.append(f"### {line}")
        else:
            out.append(line)
    return "\n\n".join(out)
