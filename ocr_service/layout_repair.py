"""Turn column-major table dumps into row-major markdown tables.

Some vision models list every value in column one, then column two, instead of
keeping each row together. When that pattern is obvious, zip the columns back
into markdown table rows without changing already-structured markdown.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HAS_TABLE = re.compile(r"(?m)^\|.+\|\s*$")

_DIGITS = re.compile(r"[0-9०-९૦-૯]")
_SERIAL = re.compile(r"^[०-९0-9૦-૯]{1,4}[.)।]?\s*$")
_DATE = re.compile(
    r"^[०-९0-9]{1,2}[/.\-][०-९0-9]{1,2}[/.\-][०-९0-9]{2,4}\s*$"
    r"|^[०-९0-9]{1,2}\s+[\u0900-\u097F\u0A80-\u0AFFA-Za-z]+\s*$"
)
_WEEKDAY = re.compile(
    r"^(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|"
    r"सोमवार|मंगळवार|बुधवार|गुरुवार|शुक्रवार|शनिवार|रविवार|"
    r"સોમવાર|મંગળવાર|બુધવાર|ગુરુવાર|શુક્રવાર|શનિવાર|રવિવાર)"
    r"\s*$",
    re.IGNORECASE,
)
_PIPE = re.compile(r"\s*\|\s*")

_HEADER = {
    "serial": "अनु. क्र.",
    "text": "विवरण",
    "date": "दिनांक",
    "weekday": "वार",
    "label": "लेबल",
    "value": "मूल्य",
}


@dataclass(frozen=True)
class _Block:
    kind: str
    lines: tuple[str, ...]


def repair_column_major_tables(text: str) -> str:
    """Zip equal-length column blocks into a markdown table when safe."""
    body = text.strip()
    if not body or _HAS_TABLE.search(body) or "<table" in body.lower():
        return text
    lines = body.replace("\r\n", "\n").split("\n")
    prefix, blocks = _column_blocks(lines)
    if not blocks:
        return text
    while blocks and len(blocks[0].lines) == 1 and len(blocks) > 1:
        prefix.extend(blocks[0].lines)
        blocks = blocks[1:]
    lengths = {len(block.lines) for block in blocks}
    if len(lengths) != 1:
        min_len = min(len(block.lines) for block in blocks)
        max_len = max(len(block.lines) for block in blocks)
        if min_len >= 3 and max_len - min_len <= 3 and len(blocks) >= 3:
            blocks = [
                _Block(kind=block.kind, lines=block.lines[:min_len])
                for block in blocks
            ]
        else:
            return text
    row_count = next(iter(lengths))
    if row_count < 2 or len(blocks) < 2 or len(blocks) > 8:
        return text
    kinds = {block.kind for block in blocks}
    if kinds <= {"text"}:
        return text
    if "serial" not in kinds and len(blocks) < 3:
        return text
    table = _markdown_table(blocks)
    if prefix:
        return "\n".join(prefix + ["", table])
    return table


def _column_blocks(lines: list[str]) -> tuple[list[str], list[_Block]]:
    prefix: list[str] = []
    blocks: list[_Block] = []
    index = 0
    while index < len(lines) and not lines[index].strip():
        index += 1
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        if not blocks and line.strip().startswith("#"):
            prefix.append(line.strip())
            index += 1
            continue
        if (
            not blocks
            and _line_kind(line) == "text"
            and _followed_by_serial_block(lines, index + 1)
        ):
            prefix.append(line.strip())
            index += 1
            while index < len(lines) and not lines[index].strip():
                index += 1
            continue
        kind = _line_kind(line)
        start = index
        while index < len(lines) and lines[index].strip() and _line_kind(lines[index]) == kind:
            index += 1
        chunk = tuple(item.strip() for item in lines[start:index] if item.strip())
        if chunk:
            blocks.append(_Block(kind=kind, lines=chunk))
        while index < len(lines) and not lines[index].strip():
            index += 1
    return prefix, blocks


def _followed_by_serial_block(lines: list[str], start: int) -> bool:
    index = start
    while index < len(lines) and not lines[index].strip():
        index += 1
    if index >= len(lines):
        return False
    return _line_kind(lines[index]) == "serial"


def _line_kind(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return "blank"
    if _SERIAL.match(stripped):
        return "serial"
    if _DATE.match(stripped):
        return "date"
    if _WEEKDAY.match(stripped):
        return "weekday"
    if _looks_like_label(stripped):
        return "label"
    if _looks_like_value(stripped) and not _looks_like_label(stripped):
        return "value"
    return "text"


def _looks_like_label(line: str) -> bool:
    if len(line) > 80:
        return False
    if line.endswith(":") or line.endswith("："):
        return True
    lowered = line.casefold()
    return any(
        token in lowered
        for token in (
            "दिशा",
            "दिशे",
            "शुल्क",
            "fee",
            "direction",
            "boundary",
            "पक्ष",
            "वय",
            "नाव",
            "name",
            "age",
            "party",
        )
    )


def _looks_like_value(line: str) -> bool:
    if len(line) > 120:
        return False
    return bool(_DIGITS.search(line))


def _markdown_table(blocks: list[_Block]) -> str:
    headers: list[str] = []
    used: set[str] = set()
    for block in blocks:
        base = _HEADER.get(block.kind, "स्तंभ")
        title = base
        suffix = 1
        while title in used:
            suffix += 1
            title = f"{base} {suffix}"
        used.add(title)
        headers.append(title)
    header_line = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join("---" for _ in headers) + " |"
    row_count = len(blocks[0].lines)
    rows: list[str] = []
    for row_index in range(row_count):
        cells = [_escape_cell(block.lines[row_index]) for block in blocks]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header_line, divider, *rows])


def _escape_cell(value: str) -> str:
    return value.replace("|", "\\|").strip()
