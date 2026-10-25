"""Post-process a full-page reading: lists, tables, charts, and prose.

The vision model is primary. Geometry (Tesseract word boxes) only repairs
obvious borderless *data tables* when the model dumps columns separately.
Charts, bullet lists, and well-formed markdown from the model are left intact.
"""

from __future__ import annotations

import re

from html_tables import has_html_table, html_table_row_count
from layout_repair import _HAS_TABLE, repair_column_major_tables
from table_layout import apply_geometry_table_fallback

_LIST_ONLY = re.compile(r"^[\s•\-*◦▪]\S|^[०-९0-9૦-૯]{1,3}[.)]\s+\S")
_WELL_FORMED_DIVIDER = re.compile(r"(?m)^\|?\s*:?-{3,}")


def apply_page_layout(image, text: str, *, model_id: str) -> str:
    original = (text or "").strip()
    # Keep LightOn-style HTML tables in the canonical text. Convert only when
    # the caller explicitly needs markdown (html_tables.convert_html_tables_in_text).
    body = original if has_html_table(original) else repair_column_major_tables(original)
    body = _repair_list_column_dump(body)
    if _should_skip_geometry(original, body):
        return body
    return apply_geometry_table_fallback(image, body, original=original, model_id=model_id)


def _should_skip_geometry(original: str, cleaned: str) -> bool:
    """Only skip geometry when the vision model already returned a real table."""
    return _has_well_formed_table(cleaned) or _has_well_formed_table(original)


def _has_well_formed_table(text: str) -> bool:
    if has_html_table(text) and html_table_row_count(text) >= 3:
        return True
    if not _HAS_TABLE.search(text):
        return False
    rows = [line for line in text.splitlines() if line.strip().startswith("|")]
    if len(rows) < 3:
        return False
    if not _WELL_FORMED_DIVIDER.search(text):
        return False
    data = [row for row in rows if "---" not in row and row.count("|") >= 3]
    return len(data) >= 2


def _reads_as_list_or_prose(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        return True
    list_like = sum(1 for line in lines if _LIST_ONLY.match(line))
    if list_like >= max(3, len(lines) // 3):
        return True
    if not _HAS_TABLE.search(text) and len(lines) >= 2:
        return True
    return False


def _looks_like_column_dump(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 4:
        return False
    serial = re.compile(r"^[૦-૯0-9०-९]{1,3}[.)]?\s*$")
    serial_lines = sum(1 for line in lines if serial.match(line))
    if serial_lines >= 3 and serial_lines >= len(lines) // 3:
        return True
    if len(lines) >= 8 and serial_lines >= max(4, len(lines) // 4):
        return True
    if _HAS_TABLE.search(text) and re.search(
        r"अनु\. क्र\.|विवरण|दिनांक", text
    ):
        return True
    return False


def _repair_list_column_dump(text: str) -> str:
    """Zip a serial block + text block into numbered lines when lengths match."""
    if not text.strip():
        return text
    lines = text.replace("\r\n", "\n").split("\n")
    prefix: list[str] = []
    merged_any = False
    index = 0
    while index < len(lines) and not lines[index].strip():
        prefix.append(lines[index])
        index += 1
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        serial = re.compile(r"^[૦-૯0-9०-९]{1,3}[.)]?\s*$")
        if not serial.match(line):
            prefix.append(line)
            index += 1
            continue
        serial_block: list[str] = []
        while index < len(lines) and lines[index].strip() and serial.match(lines[index].strip()):
            serial_block.append(lines[index].strip())
            index += 1
        text_block: list[str] = []
        while index < len(lines) and lines[index].strip() and not serial.match(lines[index].strip()):
            if lines[index].strip().startswith("|"):
                break
            text_block.append(lines[index].strip())
            index += 1
        if (
            len(serial_block) >= 2
            and len(text_block) == len(serial_block)
            and not _HAS_TABLE.search(text)
        ):
            merged = [f"{num} {body}" for num, body in zip(serial_block, text_block, strict=False)]
            prefix.extend(merged)
            merged_any = True
            continue
        prefix.extend(serial_block)
        prefix.extend(text_block)
    if not merged_any:
        return text
    return "\n".join(prefix).strip()
