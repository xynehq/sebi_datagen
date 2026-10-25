"""Recognition and structure metrics for structured OCR evaluation."""

from __future__ import annotations

import re
from dataclasses import dataclass

from training.structured_ocr.targets import _table_shape, normalize_target

_TR = re.compile(r"<tr\b", re.IGNORECASE)
_TD = re.compile(r"<td\b", re.IGNORECASE)
_TH = re.compile(r"<th\b", re.IGNORECASE)
_WEEKDAY = re.compile(
    r"(સોમવાર|मंगळवार|बुधवार|शुक्रवार|शनिवार|रविवार|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)",
    re.IGNORECASE,
)
_SERIAL = re.compile(r"^[૦-૯0-9०-९]{1,3}[.)]?\s*")
_ASCII_SERIAL = re.compile(r"^\d{1,3}[.)]?\s*$")


@dataclass(frozen=True)
class StructureMetrics:
    cer: float
    exact_match: bool
    table_row_count_pred: int
    table_row_count_ref: int
    table_row_count_match: bool
    column_major_dump: bool
    html_valid: bool
    repetition_score: float


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def character_error_rate(reference: str, prediction: str) -> float:
    ref = reference or ""
    if not ref:
        return 0.0 if not prediction else 1.0
    return levenshtein(ref, prediction) / len(ref)


def looks_column_major_dump(text: str) -> bool:
    lines = [line.strip() for line in normalize_target(text).splitlines() if line.strip()]
    if len(lines) < 8:
        return False
    serial = sum(1 for line in lines if _SERIAL.match(line) or _ASCII_SERIAL.match(line))
    weekdays = sum(1 for line in lines if _WEEKDAY.search(line) and len(line) < 48)
    if serial >= max(4, len(lines) // 5) and weekdays >= 3:
        return True
    return serial >= max(6, len(lines) // 3)


def extract_html_table_rows(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for block in re.findall(r"<table\b[^>]*>.*?</table>", text, flags=re.IGNORECASE | re.DOTALL):
        for tr in re.findall(r"<tr\b[^>]*>(.*?)</tr>", block, flags=re.IGNORECASE | re.DOTALL):
            if _TH.search(tr) and not _TD.search(tr):
                continue
            cells = re.findall(r"<t[hd]\b[^>]*>(.*?)</t[hd]>", tr, flags=re.IGNORECASE | re.DOTALL)
            cleaned = [re.sub(r"<[^>]+>", " ", cell).strip() for cell in cells]
            if any(cleaned):
                rows.append(cleaned)
    return rows


def table_cell_association_score(reference: str, prediction: str) -> float:
    """Fraction of reference body rows whose first cell serial appears in same pred row."""
    ref_rows = extract_html_table_rows(reference)
    pred_rows = extract_html_table_rows(prediction)
    if not ref_rows or not pred_rows:
        ref_rows_n, _ = _table_shape(reference)
        pred_rows_n, _ = _table_shape(prediction)
        if ref_rows_n and pred_rows_n:
            return 1.0 if ref_rows_n == pred_rows_n else 0.0
        return 0.0
    ref_body = [row for row in ref_rows if row and not row[0].lower().startswith("क्रम")]
    if not ref_body:
        ref_body = ref_rows[1:] if len(ref_rows) > 1 else ref_rows
    hits = 0
    for ref_row in ref_body:
        serial = ref_row[0].strip()
        if not serial:
            continue
        for pred_row in pred_rows:
            joined = " | ".join(pred_row)
            if serial in joined and (len(ref_row) < 2 or ref_row[1][:4] in joined):
                hits += 1
                break
    return hits / max(1, len(ref_body))


def evaluate_structure(reference: str, prediction: str) -> StructureMetrics:
    ref = normalize_target(reference)
    pred = normalize_target(prediction)
    ref_rows, _ = _table_shape(ref)
    pred_rows, _ = _table_shape(pred)
    rep = _repetition_score(pred)
    return StructureMetrics(
        cer=character_error_rate(ref, pred),
        exact_match=ref == pred,
        table_row_count_pred=pred_rows,
        table_row_count_ref=ref_rows,
        table_row_count_match=ref_rows == pred_rows and ref_rows > 0,
        column_major_dump=looks_column_major_dump(pred),
        html_valid="<table" in pred.lower() and "</table>" in pred.lower(),
        repetition_score=rep,
    )


def _repetition_score(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 4:
        return 0.0
    counts: dict[str, int] = {}
    for line in lines:
        counts[line] = counts.get(line, 0) + 1
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(lines)
