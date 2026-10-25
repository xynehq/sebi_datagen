"""Canonical structured OCR targets (Markdown + HTML tables)."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field

EMPTY_PAGE_TARGET = ""

_FENCE = re.compile(r"^```(?:markdown|md|html)?\s*\n?", re.IGNORECASE | re.MULTILINE)
_FENCE_END = re.compile(r"\n?```\s*$")
_HAS_TABLE = re.compile(r"<table\b", re.IGNORECASE)
_TABLE_BLOCK = re.compile(r"<table\b[^>]*>.*?</table>", re.IGNORECASE | re.DOTALL)
_TR = re.compile(r"<tr\b", re.IGNORECASE)
_TH = re.compile(r"<th\b", re.IGNORECASE)
_TD = re.compile(r"<td\b", re.IGNORECASE)
_THEAD = re.compile(r"<thead\b", re.IGNORECASE)
_TBODY = re.compile(r"<tbody\b", re.IGNORECASE)
_MD_TABLE_ROW = re.compile(r"^\|.+\|\s*$")
_MD_DIVIDER = re.compile(r"^\|?\s*:?-{3,}")
_REPEAT_LINE = re.compile(r"(?m)^(.{8,80})\s*(?:\n\1\s*){6,}")


class TargetValidationError(ValueError):
    """Target text failed structural or policy checks."""


@dataclass(frozen=True)
class TargetReport:
    ok: bool
    normalized: str
    content_hash: str
    table_row_count: int = 0
    table_col_count: int = 0
    issues: tuple[str, ...] = field(default_factory=tuple)


def normalize_target(text: str) -> str:
    """Deterministic cleanup without changing script or digits."""
    body = unicodedata.normalize("NFC", (text or "").strip())
    body = _FENCE.sub("", body, count=1)
    body = _FENCE_END.sub("", body).strip()
    body = re.sub(r"\r\n?", "\n", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


def validate_target(
    text: str,
    *,
    expect_table_rows: int | None = None,
    expect_table_cols: int | None = None,
    allow_empty: bool = False,
) -> TargetReport:
    normalized = normalize_target(text)
    issues: list[str] = []

    if not normalized:
        if allow_empty:
            digest = hashlib.sha256(b"").hexdigest()
            return TargetReport(True, EMPTY_PAGE_TARGET, digest, 0, 0, tuple())
        issues.append("empty_target")
        return _report(normalized, issues)

    if _REPEAT_LINE.search(normalized):
        issues.append("suspicious_repeated_lines")

    table_rows, table_cols = _table_shape(normalized)
    if _HAS_TABLE.search(normalized):
        html_issues = _validate_html_tables(normalized)
        issues.extend(html_issues)
    elif _MD_TABLE_ROW.search(normalized):
        md_issues = _validate_markdown_tables(normalized)
        issues.extend(md_issues)

    if expect_table_rows is not None and table_rows != expect_table_rows:
        issues.append(f"table_rows_expected_{expect_table_rows}_got_{table_rows}")
    if expect_table_cols is not None and table_cols and table_cols != expect_table_cols:
        issues.append(f"table_cols_expected_{expect_table_cols}_got_{table_cols}")

    ok = not issues
    return _report(normalized, issues, table_rows, table_cols)


def target_fingerprint(text: str) -> str:
    return hashlib.sha256(normalize_target(text).encode("utf-8")).hexdigest()


def _report(
    normalized: str,
    issues: list[str],
    table_rows: int = 0,
    table_cols: int = 0,
) -> TargetReport:
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return TargetReport(
        ok=not issues,
        normalized=normalized,
        content_hash=digest,
        table_row_count=table_rows,
        table_col_count=table_cols,
        issues=tuple(issues),
    )


def _table_shape(text: str) -> tuple[int, int]:
    if _HAS_TABLE.search(text):
        rows = 0
        cols = 0
        for block in _TABLE_BLOCK.findall(text):
            trs = _TR.findall(block)
            rows += max(0, len(trs) - 1) if _THEAD.search(block) else len(trs)
            for tr in re.findall(r"<tr\b[^>]*>(.*?)</tr>", block, flags=re.IGNORECASE | re.DOTALL):
                cells = len(_TH.findall(tr)) + len(_TD.findall(tr))
                cols = max(cols, cells)
        return rows, cols
    lines = [line for line in text.splitlines() if _MD_TABLE_ROW.match(line)]
    data = [line for line in lines if not _MD_DIVIDER.match(line)]
    if len(data) < 2:
        return 0, 0
    header_cols = data[0].count("|") - 1
    return len(data) - 1, max(0, header_cols)


def _validate_html_tables(text: str) -> list[str]:
    issues: list[str] = []
    blocks = _TABLE_BLOCK.findall(text)
    if not blocks:
        issues.append("unclosed_or_malformed_html_table")
        return issues
    for block in blocks:
        if not _THEAD.search(block) and not _TD.search(block):
            issues.append("html_table_missing_cells")
        if _TR.findall(block) and len(_TR.findall(block)) < 2:
            issues.append("html_table_too_few_rows")
        if "<script" in block.lower():
            issues.append("html_script_not_allowed")
    return issues


def _validate_markdown_tables(text: str) -> list[str]:
    issues: list[str] = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        if not _MD_TABLE_ROW.match(lines[index]):
            index += 1
            continue
        block: list[str] = []
        while index < len(lines) and _MD_TABLE_ROW.match(lines[index]):
            block.append(lines[index])
            index += 1
        if len(block) < 2:
            issues.append("markdown_table_incomplete")
            continue
        if not any(_MD_DIVIDER.match(line) for line in block):
            issues.append("markdown_table_missing_divider")
    return issues
