"""Assemble borderless tables from Tesseract word boxes (row-major).

Full-page vision models often read column-by-column on Gujarati government
tables. Tesseract supplies coordinates only; recognized cell text stays with
the original layout reconstruction. Output is an HTML table by default.
"""

from __future__ import annotations

TABLE_LAYOUT_VERSION = "table-layout-v3"


def table_layout_version() -> str:
    return TABLE_LAYOUT_VERSION

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from PIL import Image

from html_tables import (
    convert_markdown_tables_to_html,
    has_html_table,
    html_table_row_count,
)
from image_preprocess import normalize_document_image
from layout_repair import _HAS_TABLE, repair_column_major_tables

_SERIAL = re.compile(r"^[૦-૯0-9०-९]{1,3}[.)]?\s*$")
_LEADING_SERIAL = re.compile(r"^([૦-૯0-9०-९]{1,3}[.)]?)\s+(.+)$")
_TWO_SERIALS = re.compile(
    r"^([૦-૯0-9०-९]{1,3}[.)]?)\s+([૦-૯0-9०-९]{1,3}[.)]?)\s*$"
)
_TABLE_HEADER = re.compile(r"ક્રમાંક|तारीख|તારીખ|जाहेर|જાહેર|क्रमांक")
_WEEKDAY = re.compile(
    r"(સોમવાર|મંગળવાર|બુધવાર|ગુરૂવાર|શુક્રવાર|શનિવાર|રવિવાર|રવીવાર|"
    r"सोमवार|मंगळवार|बुधवार|गुरुवार|शुक्रवार|शनिवार|रविवार)"
)
_DATEISH = re.compile(
    r"[૦-૯0-9०-९]{1,2}\s*[\u0A80-\u0AFF\u0900-\u097F]+|[૦-૯0-9०-९]{1,2}[/.\-]"
)
_GUJ = re.compile(r"[\u0A80-\u0AFF]")
_DEVA = re.compile(r"[\u0900-\u097F]")
_NOISE_TOKEN = re.compile(r"^[_#=*|:.\-]+$|^ક$|^ee+$", re.IGNORECASE)
_NOISE_LINE = re.compile(r"^[\s_#=*|:.\-]+$|^ક\s*=?\s*$|^ee+$", re.IGNORECASE)
_HEADER_MARKERS = (
    "ક્રમાંક",
    "क्रमांक",
    "તારીખ",
    "तारीख",
    "જાહેર",
    "जाहेर",
    "અઠવાડિયા",
    "સંવત",
)


@dataclass(frozen=True)
class _Word:
    left: int
    top: int
    width: int
    text: str
    line_id: tuple[int, int, int, int]
    height: int = 18
    conf: float = 0.0

    @property
    def center_x(self) -> float:
        return self.left + self.width / 2

    @property
    def bottom(self) -> int:
        return self.top + self.height


class TableRebuild(NamedTuple):
    rows: list[list[str]]
    bounds: list[int]
    table_top: int
    lines: list[list[_Word]]
    footer: str


def tesseract_lang_for_model(model_id: str) -> str:
    lowered = model_id.casefold()
    if "marguj" in lowered or "guj" in lowered:
        return os.environ.get("OCR_TESSERACT_LANG", "").strip() or "guj+eng"
    if "indicocr" in lowered:
        return os.environ.get("OCR_TESSERACT_LANG", "").strip() or "guj+eng+hin+eng"
    return os.environ.get("OCR_TESSERACT_LANG", "").strip() or "mar+eng"


def apply_table_layout(
    image: Image.Image,
    text: str,
    *,
    model_id: str,
) -> str:
    """Backward-compatible entry point (see page_layout.apply_page_layout)."""
    from page_layout import apply_page_layout

    return apply_page_layout(image, text, model_id=model_id)


def force_tabular_layout(
    image: Image.Image,
    text: str,
    *,
    model_id: str,
) -> str:
    """Last resort: replace body with a Tesseract grid when the VLM missed the table."""
    if os.environ.get("OCR_TABLE_LAYOUT", "1").strip() in {"0", "false", "no"}:
        return text
    if ("|" in text and "---" in text) or "<table" in text.lower():
        return text
    lang = tesseract_lang_for_model(model_id)
    table = tesseract_table_markdown(image, lang=lang)
    if not table or _tesseract_data_rows(table) < 5:
        return text
    intro = _intro_lines(text) or _intro_from_image(image, lang=lang)
    if intro:
        return "\n\n".join([intro, table])
    return table


def apply_geometry_table_fallback(
    image: Image.Image,
    cleaned: str,
    *,
    original: str,
    model_id: str,
) -> str:
    """Rebuild row-major markdown from word boxes when the VLM mangled a grid."""
    if os.environ.get("OCR_TABLE_LAYOUT", "1").strip() in {"0", "false", "no"}:
        return cleaned
    lang = tesseract_lang_for_model(model_id)
    table = tesseract_table_markdown(image, lang=lang)
    if table and _should_use_tesseract_table(original, cleaned, table):
        intro = _intro_lines(original) or _intro_from_image(image, lang=lang)
        if intro:
            return "\n\n".join([intro, table])
        return table
    return cleaned


def tesseract_table_markdown(image: Image.Image, *, lang: str) -> str | None:
    variants = [image, normalize_document_image(image)]
    seen: set[tuple[int, int]] = set()
    best: tuple[int, str] | None = None
    for variant in variants:
        key = variant.size
        if key in seen:
            continue
        seen.add(key)
        for candidate in _tesseract_candidates(variant):
            table = _tesseract_table_from_image(candidate, lang=lang)
            if not table:
                continue
            score = _tesseract_data_rows(table)
            if best is None or score > best[0]:
                best = (score, table)
    return best[1] if best else None


def _tesseract_candidates(image: Image.Image) -> list[Image.Image]:
    width, height = image.size
    candidates = [image]
    if max(width, height) < 1200:
        scale = 2.0
        candidates.append(
            image.resize(
                (max(1, int(width * scale)), max(1, int(height * scale))),
                Image.Resampling.LANCZOS,
            )
        )
    return candidates


def _tesseract_table_from_image(image: Image.Image, *, lang: str) -> str | None:
    payload = _run_tesseract(image, lang=lang)
    if not payload:
        return None
    built = reconstruct_table_from_words(_parse_words(payload), image.width)
    if built is None:
        return None
    if len(built.rows) < 3:
        return None
    table = _to_table_text(built.rows)
    if built.footer:
        return f"{table}\n\n{built.footer}"
    return table


def reconstruct_table_from_words(
    words: list[_Word],
    page_width: int,
) -> TableRebuild | None:
    """Row/column reconstruction used by production and tests."""
    if len(words) < 8:
        return None
    table_top = _table_top(words)
    if table_top is None:
        return None
    body_words = [word for word in words if word.top >= table_top - 5]
    lines = _merge_orphan_visual_lines(
        _tesseract_scan_lines(body_words),
        page_width,
    )
    last_bottom = table_top
    for group in lines:
        if any(_SERIAL.match(word.text) for word in group):
            last_bottom = max(last_bottom, max(word.bottom for word in group))
    pad = 16
    table_lines = [group for group in lines if group and min(word.top for word in group) <= last_bottom + pad]
    footer_lines = [group for group in lines if group and min(word.top for word in group) > last_bottom + pad]
    bounds = _column_bounds(table_lines, page_width)
    if bounds is None:
        return None
    rows = _rows_from_lines(table_lines, bounds)
    if len(rows) < 3:
        return None
    footer = "\n".join(
        _line_text(group).strip()
        for group in footer_lines
        if _keep_intro_line(_line_text(group))
    ).strip()
    return TableRebuild(rows, bounds, table_top, table_lines, footer)


def _run_tesseract(image: Image.Image, *, lang: str) -> str | None:
    cmd = os.environ.get("TESSERACT_CMD", "").strip() or "tesseract"
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        path = Path(handle.name)
    try:
        image.convert("RGB").save(path)
        completed = subprocess.run(
            [cmd, str(path), "stdout", "-l", lang, "--psm", "6", "--oem", "1", "tsv"],
            check=False,
            capture_output=True,
            text=True,
        )
    finally:
        path.unlink(missing_ok=True)
    if completed.returncode != 0:
        return None
    return completed.stdout


def _parse_words(payload: str) -> list[_Word]:
    words: list[_Word] = []
    for line in payload.splitlines()[1:]:
        columns = line.split("\t")
        if len(columns) < 12 or columns[0] != "5":
            continue
        text = columns[11].strip()
        if not text:
            continue
        try:
            left = int(columns[6])
            top = int(columns[7])
            width = int(columns[8])
            height = int(columns[9]) if columns[9] else 18
            block = int(columns[2])
            par = int(columns[3])
            line_num = int(columns[4])
            word_num = int(columns[5])
            conf = float(columns[10])
        except ValueError:
            continue
        if conf < 0:
            continue
        if _NOISE_TOKEN.match(text):
            continue
        words.append(
            _Word(
                left=left,
                top=top,
                width=width,
                text=text,
                line_id=(block, par, line_num, word_num),
                height=max(1, height),
                conf=conf,
            )
        )
    return words


def _tesseract_scan_lines(words: list[_Word]) -> list[list[_Word]]:
    """Group words by Tesseract block/paragraph/line (stable on wide tables)."""
    buckets: dict[tuple[int, int, int], list[_Word]] = {}
    for word in words:
        key = (word.line_id[0], word.line_id[1], word.line_id[2])
        buckets.setdefault(key, []).append(word)
    return [sorted(group, key=lambda item: item.left) for key in sorted(buckets) for group in [buckets[key]]]


def _visual_rows(words: list[_Word], *, tolerance: int | None = None) -> list[list[_Word]]:
    ordered = sorted(words, key=lambda word: (word.top, word.left))
    if not ordered:
        return []
    heights = sorted(word.height for word in ordered)
    median_h = heights[len(heights) // 2]
    gap = tolerance if tolerance is not None else max(10, int(median_h * 0.55))
    rows: list[list[_Word]] = []
    current: list[_Word] = []
    baseline: int | None = None
    for word in ordered:
        if baseline is None or abs(word.top - baseline) <= gap:
            current.append(word)
            if baseline is None:
                baseline = word.top
            continue
        if current:
            rows.append(sorted(current, key=lambda item: item.left))
        current = [word]
        baseline = word.top
    if current:
        rows.append(sorted(current, key=lambda item: item.left))
    return rows


def _table_top(words: list[_Word]) -> int | None:
    """Y of the table region, including the header row (not the first body row)."""
    header_y = None
    for word in words:
        token = word.text.strip().rstrip(":")
        if token in {"ક્રમાંક", "क्रमांक"}:
            header_y = word.top
            break
    if header_y is None:
        for word in words:
            if _TABLE_HEADER.search(word.text):
                header_y = word.top
                break
    serial_ys = [
        word.top
        for word in words
        if _SERIAL.match(word.text) and word.left < max(90, int(word.left) or 90)
        and word.left < 120
    ]
    if header_y is not None:
        return header_y
    if serial_ys:
        return min(serial_ys) - 5
    return None


def _column_bounds(lines: list[list[_Word]], page_width: int) -> list[int] | None:
    anchored = _bounds_from_serial_and_weekday(lines, page_width)
    if anchored:
        return anchored
    header = None
    for group in lines[:4]:
        joined = " ".join(word.text for word in group)
        if "તારીખ" in joined or "क्रमांक" in joined or "ક્રમાંક" in joined:
            header = group
            break
    if header and len(header) >= 3:
        xs = sorted(word.left for word in header)
        seeded = _bounds_from_seeds(xs, page_width)
        if seeded:
            return seeded
    anchors: list[int] = []
    for group in lines:
        if not group:
            continue
        if _SERIAL.match(group[0].text):
            anchors.append(group[0].left)
            if len(group) > 1:
                anchors.append(group[1].left)
        if len(anchors) >= 8:
            break
    if len(anchors) < 4:
        return _refine_bounds(_default_bounds(page_width), page_width)
    return _bounds_from_seeds(sorted(anchors), page_width)


def _bounds_from_serial_and_weekday(
    lines: list[list[_Word]], page_width: int
) -> list[int] | None:
    """Derive column cuts from serial numbers (left) and weekday words (right)."""
    serial_rights: list[int] = []
    weekday_lefts: list[int] = []
    for group in lines:
        for word in group:
            if _SERIAL.match(word.text):
                serial_rights.append(word.left + word.width)
            if _WEEKDAY.search(word.text) and word.left > page_width * 0.45:
                weekday_lefts.append(word.left)
    if len(serial_rights) < 5 or len(weekday_lefts) < 5:
        return None
    serial_rights.sort()
    weekday_lefts.sort()
    col1 = min(page_width // 3, serial_rights[len(serial_rights) // 2] + 20)
    col3 = max(page_width // 2, weekday_lefts[len(weekday_lefts) // 4] - 25)
    col2 = col1 + max(80, int((col3 - col1) * 0.38))
    if col2 >= col3 - 40:
        col2 = int((col1 + col3) / 2)
    return _refine_bounds([0, col1, col2, col3, page_width], page_width)


def _bounds_from_seeds(anchors: list[int], page_width: int) -> list[int]:
    unique: list[int] = []
    for value in anchors:
        if not unique or value - unique[-1] > 25:
            unique.append(value)
    if len(unique) < 3:
        return _default_bounds(page_width)
    while len(unique) < 4:
        unique.append(min(page_width, unique[-1] + page_width // 5))
    cuts = [0]
    for left, right in zip(unique, unique[1:], strict=False):
        cuts.append((left + right) // 2)
    cuts.append(page_width)
    if len(cuts) < 5:
        return _refine_bounds(_default_bounds(page_width), page_width)
    # Always end at page_width. cuts[:5] dropped the right edge when many header words
    # produced extra midpoints (screenshot pages then got 0 data rows).
    refined = [0, cuts[1], cuts[2], cuts[3], page_width]
    return _refine_bounds(refined, page_width)


def _refine_bounds(bounds: list[int], page_width: int) -> list[int]:
    """Keep the weekday column separate from તારીખ / શક સંવત (often mis-seeded)."""
    min_day = int(page_width * 0.72)
    max_date_end = int(page_width * 0.70)
    if bounds[3] < min_day:
        bounds[3] = min(max_date_end, min_day)
    if bounds[2] >= bounds[3] - 40:
        bounds[2] = int(page_width * 0.46)
    if bounds[1] >= bounds[2] - 30:
        bounds[1] = int(page_width * 0.10)
    return bounds


def _line_text(group: list[_Word]) -> str:
    return " ".join(word.text for word in group)


def _is_header_group(group: list[_Word]) -> bool:
    joined = _line_text(group)
    hits = sum(1 for marker in _HEADER_MARKERS if marker in joined)
    return hits >= 2 or ("ક્રમાંક" in joined and "તારીખ" in joined) or (
        "क्रमांक" in joined and "तारीख" in joined
    )


def _is_header_wrap(group: list[_Word]) -> bool:
    joined = _line_text(group).strip()
    if not group:
        return False
    if any(_SERIAL.match(word.text) for word in group):
        return False
    if any(_WEEKDAY.search(word.text) for word in group):
        return False
    compact = joined.replace(" ", "")
    return compact in {"વાર", "वार", "નામ", "नाम", "સંવત", "संवत"} or joined in {
        "અઠવાડિયાનો વાર",
        "अठवड्याचा वार",
    }


def _is_name_wrap(group: list[_Word], page_width: int) -> bool:
    """Wrapped cell text: no serial, no weekday, stays in the name column."""
    if not group or _is_header_group(group) or _is_header_wrap(group):
        return False
    if any(_SERIAL.match(word.text) for word in group):
        return False
    if any(_WEEKDAY.search(word.text) for word in group):
        return False
    if any(_DATEISH.search(word.text) for word in group):
        return False
    name_right = page_width * 0.52
    return all(word.left < name_right for word in group)


def _merge_orphan_visual_lines(
    lines: list[list[_Word]],
    page_width: int,
) -> list[list[_Word]]:
    """Attach wrapped name/header fragments to the previous record, never the next."""
    if not lines:
        return lines
    merged: list[list[_Word]] = []
    for group in lines:
        if not group:
            continue
        if merged and _is_header_wrap(group) and _is_header_group(merged[-1]):
            merged[-1] = sorted(merged[-1] + group, key=lambda word: (word.top, word.left))
            continue
        if merged and _is_name_wrap(group, page_width):
            prev = merged[-1]
            has_serial = any(_SERIAL.match(word.text) for word in prev)
            prev_bottom = max(word.bottom for word in prev)
            group_top = min(word.top for word in group)
            heights = [word.height for word in prev + group]
            gap_limit = max(22, int((sorted(heights)[len(heights) // 2]) * 1.35))
            close = 0 <= group_top - prev_bottom <= gap_limit or 0 <= group_top - min(word.top for word in prev) <= gap_limit
            if (has_serial or _is_header_group(prev)) and close:
                merged[-1] = sorted(prev + group, key=lambda word: (word.top, word.left))
                continue
        merged.append(group)
    return merged


def _default_bounds(page_width: int) -> list[int]:
    return [0, int(page_width * 0.10), int(page_width * 0.46), int(page_width * 0.78), page_width]


def _column_index(left: int, bounds: list[int]) -> int:
    for index in range(4):
        if bounds[index] <= left < bounds[index + 1]:
            return index
    return 3


def _assign_cells(group: list[_Word], bounds: list[int]) -> list[str]:
    cells = ["", "", "", ""]
    for word in group:
        if _NOISE_TOKEN.match(word.text):
            continue
        column = _column_index(word.left, bounds)
        cells[column] = f"{cells[column]} {word.text}".strip()
    if cells[0]:
        two = _TWO_SERIALS.match(cells[0].strip())
        if two:
            # Keep both tokens visible until geometry split; do not glue names here.
            cells[0] = cells[0].strip()
        else:
            lead = _LEADING_SERIAL.match(cells[0])
            if lead and not _SERIAL.match(cells[0]):
                cells[0] = lead.group(1)
                spill = lead.group(2).strip()
                if spill:
                    cells[1] = f"{spill} {cells[1]}".strip()
    return cells


def _split_dual_serial_group(group: list[_Word], bounds: list[int]) -> list[list[_Word]]:
    serials = [
        word
        for word in group
        if _SERIAL.match(word.text) and _column_index(word.left, bounds) == 0
    ]
    weekdays = [word for word in group if _WEEKDAY.search(word.text)]
    if len(serials) < 2 or len(weekdays) < 2:
        return [group]
    serials = sorted(serials, key=lambda word: word.top)
    cuts = [
        (serials[index].top + serials[index + 1].top) / 2
        for index in range(len(serials) - 1)
    ]
    buckets: list[list[_Word]] = [[] for _ in serials]
    for word in group:
        assigned = len(serials) - 1
        for index, cut in enumerate(cuts):
            if word.top < cut:
                assigned = index
                break
        buckets[assigned].append(word)
    return [sorted(bucket, key=lambda word: (word.top, word.left)) for bucket in buckets if bucket]


def _looks_like_header_cells(cells: list[str]) -> bool:
    joined = " ".join(cells)
    return any(marker in joined for marker in _HEADER_MARKERS) and not _SERIAL.match(
        cells[0].strip()
    )


def _rows_from_lines(lines: list[list[_Word]], bounds: list[int]) -> list[list[str]]:
    rows: list[list[str]] = []
    for group in lines:
        if _is_header_group(group) or _is_header_wrap(group):
            continue
        for piece in _split_dual_serial_group(group, bounds):
            cells = _assign_cells(piece, bounds)
            if not any(cells):
                continue
            if _looks_like_header_cells(cells):
                continue
            if cells[0] and _SERIAL.match(cells[0]):
                rows.append(cells)
                continue
            if rows and not cells[0]:
                for col in range(4):
                    if cells[col]:
                        rows[-1][col] = f"{rows[-1][col]} {cells[col]}".strip()
                continue
            if any(cells):
                rows.append(cells)
    trimmed = [_normalize_row_cells(row) for row in rows if any(cell.strip() for cell in row)]
    return _attach_wrap_rows(trimmed)


def _row_is_complete_record(row: list[str]) -> bool:
    blob = " ".join(row)
    return bool(_WEEKDAY.search(blob) and (_DATEISH.search(blob) or (len(row) > 2 and row[2].strip())))


def _attach_wrap_rows(rows: list[list[str]]) -> list[list[str]]:
    """Join name-only wrap rows with the previous record. Never fold one record into the next."""
    out: list[list[str]] = []
    for row in rows:
        if _looks_like_header_cells(row) or _NOISE_LINE.match(" ".join(row).strip() or ""):
            continue
        serial = _SERIAL.match(row[0].strip()) if row[0].strip() else None
        if serial:
            out.append(row)
            continue
        if _row_is_complete_record(row):
            out.append(row)
            continue
        if out and not row[0].strip():
            for col in range(4):
                if row[col]:
                    out[-1][col] = f"{out[-1][col]} {row[col]}".strip()
            continue
        if out and not _WEEKDAY.search(" ".join(row)):
            for col in range(1, 4):
                if row[col]:
                    out[-1][col] = f"{out[-1][col]} {row[col]}".strip()
            continue
        out.append(row)
    return [_normalize_row_cells(row) for row in out]


def _normalize_row_cells(row: list[str]) -> list[str]:
    cells = (row + ["", "", "", ""])[:4]
    combined = f"{cells[2]} {cells[3]}".strip()
    match = _WEEKDAY.search(combined)
    if match and match.start() > 0:
        cells[3] = match.group(1)
        cells[2] = combined[: match.start()].strip(" /|,")
    elif match and not cells[3]:
        cells[3] = match.group(1)
        cells[2] = combined[: match.start()].strip(" /|,") or cells[2]
    cells[2] = re.sub(r"\s*/\s*/\s*", " / ", cells[2])
    cells[2] = re.sub(r"\s+\|\s+", " ", cells[2]).strip()
    for index in range(4):
        cells[index] = cells[index].strip()
    return cells


def _to_table_text(rows: list[list[str]]) -> str:
    markdown = _to_markdown(rows)
    if os.environ.get("OCR_TABLE_FORMAT", "html").strip().lower() == "html":
        return convert_markdown_tables_to_html(markdown)
    return markdown


def _to_markdown(rows: list[list[str]]) -> str:
    header_cells, data_rows = _split_header_row(rows)
    header_line = "| " + " | ".join(_escape_md(cell) for cell in header_cells) + " |"
    divider = "| " + " | ".join("---" for _ in header_cells) + " |"
    body = []
    for row in data_rows:
        padded = (row + [""] * len(header_cells))[: len(header_cells)]
        body.append("| " + " | ".join(_escape_md(cell) for cell in padded) + " |")
    return "\n".join([header_line, divider, *body])


def _escape_md(value: str) -> str:
    return value.replace("|", "\\|").strip()


_GUJ_HEADERS = ["ક્રમાંક", "જાહેર રજાનું નામ", "તારીખ / શક સંવત", "અઠવાડિયાનો વાર"]
_MAR_HEADERS = ["क्रमांक", "जाहेर रजेचे नाव", "तारीख / शक संवत", "आठवड्याचा वार"]


def _split_header_row(rows: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    if not rows:
        return list(_GUJ_HEADERS), []
    first = [cell.strip() for cell in rows[0]]
    joined = " ".join(first)
    if "ક્રમાંક" in joined and ("તારીખ" in joined or "જાહેર" in joined):
        return list(_GUJ_HEADERS), rows[1:] if len(rows) > 1 else []
    if "क्रमांक" in joined and ("तारीख" in joined or "जाहेर" in joined):
        return list(_MAR_HEADERS), rows[1:] if len(rows) > 1 else []
    serial_rows = sum(1 for row in rows if row and _SERIAL.match(row[0].strip()))
    if serial_rows >= max(3, len(rows) // 3):
        sample = " ".join(" ".join(row) for row in rows[: min(6, len(rows))])
        width = max(4, max(len(row) for row in rows))
        if len(_GUJ.findall(sample)) >= 8:
            return list(_GUJ_HEADERS), rows
        if len(_DEVA.findall(sample)) >= 8:
            return list(_MAR_HEADERS)[:width] if width <= 4 else list(_MAR_HEADERS) + [
                "" for _ in range(width - 4)
            ], rows
        if first and not _SERIAL.match(first[0]) and sum(1 for cell in first if cell) >= 2:
            return (first + [""] * width)[:width], rows[1:]
        return ([""] * width), rows
    if first and not _SERIAL.match(first[0]) and sum(1 for cell in first if cell) >= 2:
        width = max(len(first), max(len(row) for row in rows[1:]) if len(rows) > 1 else len(first))
        header = (first + [""] * width)[:width]
        return header, rows[1:] if len(rows) > 1 else []
    width = max(4, max(len(row) for row in rows))
    padded_rows = [(row + [""] * width)[:width] for row in rows]
    return ([""] * width), padded_rows


def _looks_misaligned(text: str) -> bool:
    if _HAS_TABLE.search(text):
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 8:
        return False
    serial = sum(1 for line in lines if _SERIAL.match(line))
    return serial >= max(4, len(lines) // 4)


def _looks_garbled_script(text: str) -> bool:
    guj = len(_GUJ.findall(text))
    deva = len(_DEVA.findall(text))
    if guj < 20:
        return False
    return deva > guj * 0.35


_ZIP_TABLE = re.compile(r"अनु\. क्र\.|विवरण|दिनांक|वार\s*\|")
_GUJ_HOLIDAY_TABLE = re.compile(r"ક્રમાંક.*જાહેર રજાનું નામ")


def _tesseract_data_rows(table_text: str) -> int:
    if has_html_table(table_text):
        return max(0, html_table_row_count(table_text) - 1)
    past_divider = False
    count = 0
    for line in table_text.splitlines():
        if not line.strip().startswith("|"):
            continue
        if "---" in line:
            past_divider = True
            continue
        if not past_divider:
            continue
        if line.count("|") >= 3:
            count += 1
    return count


def _has_proper_gujarati_table(text: str) -> bool:
    if not _HAS_TABLE.search(text):
        return False
    return bool(_GUJ_HOLIDAY_TABLE.search(text.replace("\n", " ")))


def _vlm_has_usable_table(text: str) -> bool:
    from html_tables import has_html_table, html_table_row_count

    if has_html_table(text) and html_table_row_count(text) >= 3:
        return True
    if _has_proper_gujarati_table(text):
        return True
    if not _HAS_TABLE.search(text):
        return False
    past_divider = False
    data_rows = 0
    for line in text.splitlines():
        if not line.strip().startswith("|"):
            continue
        if "---" in line:
            past_divider = True
            continue
        if past_divider and line.count("|") >= 3:
            data_rows += 1
    return data_rows >= 3


def _should_use_tesseract_table(original: str, cleaned: str, table_text: str) -> bool:
    if not (_HAS_TABLE.search(table_text) or has_html_table(table_text)):
        return False
    tess_rows = _tesseract_data_rows(table_text)
    if tess_rows < 5:
        return False
    if not _tesseract_table_looks_like_grid(table_text):
        return False
    if _vlm_has_usable_table(cleaned) or _vlm_has_usable_table(original):
        return False
    if tess_rows >= 10:
        return True
    if _looks_misaligned(original) or _looks_misaligned(cleaned):
        return True
    if _ZIP_TABLE.search(cleaned):
        return True
    if not _HAS_TABLE.search(cleaned):
        return True
    return False


def _html_body_rows(table_text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for match in re.finditer(r"<tr\b[^>]*>(.*?)</tr>", table_text, flags=re.I | re.S):
        cells = [
            re.sub(r"<[^>]+>", "", cell).strip()
            for cell in re.findall(r"<t[hd]\b[^>]*>(.*?)</t[hd]>", match.group(1), flags=re.I | re.S)
        ]
        if cells and not any(marker in cells[0] for marker in ("ક્રમાંક", "क्रमांक")):
            rows.append(cells)
    return rows


def _tesseract_table_looks_like_grid(table_text: str) -> bool:
    """Avoid replacing charts or prose with a fake grid."""
    data_rows: list[list[str]] = []
    if has_html_table(table_text):
        data_rows = _html_body_rows(table_text)
    else:
        past_divider = False
        for line in table_text.splitlines():
            if not line.strip().startswith("|"):
                continue
            if "---" in line:
                past_divider = True
                continue
            if not past_divider:
                continue
            cells = [part.strip() for part in line.strip().strip("|").split("|")]
            if len(cells) >= 2:
                data_rows.append(cells)
    if len(data_rows) < 5:
        return False
    serial_hits = sum(1 for row in data_rows if row and _SERIAL.match(row[0]))
    if serial_hits >= max(3, len(data_rows) // 4):
        return True
    wide = sum(
        1
        for row in data_rows
        if len(row) >= 3 and sum(1 for cell in row[1:] if cell.strip()) >= 2
    )
    return wide >= max(5, len(data_rows) * 2 // 3)


def _prefer_tesseract_table(vlm_text: str, table_text: str) -> bool:
    """Backward-compatible alias for tests."""
    cleaned = repair_column_major_tables(vlm_text)
    return _should_use_tesseract_table(vlm_text, cleaned, table_text)


def _keep_intro_line(part: str) -> bool:
    stripped = part.strip()
    if not stripped or _NOISE_LINE.match(stripped):
        return False
    if stripped in {"અઠવાડિયાનો", "અઠવાડિયાનો વાર", "વાર", "वार"}:
        return False
    if "ક્રમાંક" in stripped or "क्रमांक" in stripped:
        return False
    hits = sum(1 for marker in _HEADER_MARKERS if marker in stripped)
    if hits >= 2 and len(stripped) < 90:
        return False
    return True


def _intro_from_image(image: Image.Image, *, lang: str) -> str:
    payload = _run_tesseract(image, lang=lang)
    if not payload:
        return ""
    words = _parse_words(payload)
    top = _table_top(words)
    if top is None:
        return ""
    header_words = [word for word in words if word.top < top - 8]
    lines = _visual_rows(header_words)
    parts = [" ".join(word.text for word in group).strip() for group in lines]
    return "\n".join(part for part in parts if _keep_intro_line(part)).strip()


def inspect_tesseract_table_geometry(image: Image.Image, *, lang: str) -> dict[str, object]:
    """Debug export: TSV stats, column bounds, and row/cell assignments (no VLM)."""
    import shutil

    report: dict[str, object] = {
        "tesseract_cmd": os.environ.get("TESSERACT_CMD", "").strip() or "tesseract",
        "tesseract_on_path": bool(shutil.which(os.environ.get("TESSERACT_CMD", "").strip() or "tesseract")),
        "lang": lang,
        "image_size": list(image.size),
        "grid_detected": False,
        "failure_stage": None,
    }
    variants = [image, normalize_document_image(image)]
    best_detail: dict[str, object] | None = None
    best_score = -1
    for variant in variants:
        for candidate in _tesseract_candidates(variant):
            payload = _run_tesseract(candidate, lang=lang)
            if not payload:
                report["failure_stage"] = "tesseract_run_failed"
                continue
            report["tsv_line_count"] = len(payload.splitlines())
            words = _parse_words(payload)
            detail: dict[str, object] = {
                "word_count": len(words),
                "words_sample": [
                    {
                        "text": w.text,
                        "left": w.left,
                        "top": w.top,
                        "width": w.width,
                        "conf": None,
                    }
                    for w in words[:30]
                ],
            }
            built = reconstruct_table_from_words(words, candidate.width)
            if built is None:
                detail["failure_stage"] = "reconstruct_failed"
                detail["table_top_y"] = _table_top(words)
                continue
            rows, bounds, table_top, lines = (
                built.rows,
                built.bounds,
                built.table_top,
                built.lines,
            )
            detail["table_top_y"] = table_top
            detail["visual_line_count"] = len(lines)
            detail["column_bounds"] = bounds
            detail["data_rows"] = rows
            detail["data_row_count"] = len(rows)
            detail["footer"] = built.footer
            table_md = _to_table_text(rows)
            detail["markdown_preview"] = table_md.splitlines()[:20]
            detail["grid_detected"] = _tesseract_table_looks_like_grid(table_md)
            detail["tess_data_rows"] = _tesseract_data_rows(table_md)
            score = _tesseract_data_rows(table_md)
            if score > best_score:
                best_score = score
                best_detail = detail
                report["tsv_payload"] = payload
    if best_detail:
        report.update(best_detail)
        report["grid_detected"] = bool(best_detail.get("grid_detected"))
    elif report.get("failure_stage") is None:
        report["failure_stage"] = "no_candidate"
    return report


def export_tesseract_tsv(image: Image.Image, *, lang: str) -> str:
    """Word-level Tesseract TSV for the image (PSM 6)."""
    payload = _run_tesseract(image, lang=lang)
    return payload or ""


def _intro_lines(text: str) -> str:
    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if kept:
                break
            continue
        if _SERIAL.match(stripped) or _TABLE_HEADER.search(stripped):
            break
        if stripped.startswith("|") or stripped.lower().startswith("<table"):
            break
        if not _keep_intro_line(stripped):
            continue
        kept.append(stripped)
    return "\n".join(kept).strip()
