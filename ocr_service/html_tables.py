"""HTML table helpers. Canonical OCR output keeps HTML; conversion to pipes is optional."""

from __future__ import annotations

import html
import re

_HAS_HTML_TABLE = re.compile(r"<table\b", re.IGNORECASE)
_ROW = re.compile(r"<tr\b", re.IGNORECASE)
_TABLE = re.compile(r"<table\b[^>]*>.*?</table>", re.IGNORECASE | re.DOTALL)
_INNER_ROW = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_CELL = re.compile(r"<t[hd]\b[^>]*>(.*?)</t[hd]>", re.IGNORECASE | re.DOTALL)


def has_html_table(text: str) -> bool:
    return bool(_HAS_HTML_TABLE.search(text))


def html_table_row_count(text: str) -> int:
    return len(_ROW.findall(text))


def convert_html_tables_in_text(text: str) -> str:
    """Replace each <table>...</table> with an equivalent markdown table."""
    if not has_html_table(text):
        return text
    out = text
    for match in _TABLE.finditer(text):
        fragment = match.group(0)
        rows = _parse_table_fragment(fragment)
        markdown = _table_to_markdown(rows)
        if markdown:
            out = out.replace(fragment, "\n\n" + markdown + "\n\n", 1)
    return out.strip()


def _parse_table_fragment(fragment: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for row_match in _INNER_ROW.finditer(fragment):
        cells = [
            _clean_cell(re.sub(r"<[^>]+>", " ", cell.group(1)))
            for cell in _CELL.finditer(row_match.group(1))
        ]
        if cells:
            rows.append(cells)
    return rows


def _clean_cell(value: str) -> str:
    text = html.unescape(value)
    return re.sub(r"\s+", " ", text).strip()


def _table_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    padded = [(row + [""] * width)[:width] for row in rows]
    header = padded[0]
    body = padded[1:] if len(padded) > 1 else []
    lines = [
        "| " + " | ".join(_escape(cell) for cell in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(_escape(cell) for cell in row) + " |")
    return "\n".join(lines)


def _escape(value: str) -> str:
    return value.replace("|", "\\|")


def convert_markdown_tables_to_html(text: str) -> str:
    """Replace markdown pipe tables with HTML tables; leave other markdown intact."""
    lines = text.splitlines()
    out: list[str] = []
    index = 0
    while index < len(lines):
        if not lines[index].strip().startswith("|"):
            out.append(lines[index])
            index += 1
            continue
        block: list[str] = []
        while index < len(lines) and lines[index].strip().startswith("|"):
            block.append(lines[index])
            index += 1
        rows = _markdown_block_to_rows(block)
        if rows:
            out.append(_rows_to_html_table(rows))
        else:
            out.extend(block)
    return "\n".join(out).strip()


def _markdown_block_to_rows(block: list[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in block:
        if "---" in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells:
            rows.append(cells)
    return rows


def _rows_to_html_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    header = rows[0]
    body = rows[1:] if len(rows) > 1 else []
    parts = ["<table>", "  <thead>", "    <tr>"]
    parts.extend(f"      <th>{html.escape(cell)}</th>" for cell in header)
    parts.append("    </tr>")
    parts.append("  </thead>")
    parts.append("  <tbody>")
    for row in body:
        parts.append("    <tr>")
        parts.extend(f"      <td>{html.escape(cell)}</td>" for cell in row)
        parts.append("    </tr>")
    parts.extend(["  </tbody>", "</table>"])
    return "\n".join(parts)


def close_incomplete_html_tables(text: str) -> str:
    """Close open <table>/<thead>/<tbody>/<tr> tags so Gradio can render mid-stream."""
    if not has_html_table(text):
        return text
    lower = text.lower()
    if "</table>" in lower and lower.rfind("<table") < lower.rfind("</table>"):
        # Last table looks closed; still fix a dangling open row if present.
        return _close_dangling_row(text)
    body = _close_dangling_row(text)
    lower = body.lower()
    if "<tr" in lower and "</tr>" not in lower[lower.rfind("<tr") :]:
        body += "</tr>"
        lower = body.lower()
    if "<thead" in lower and "</thead>" not in lower[lower.rfind("<thead") :]:
        body += "\n  </thead>"
        lower = body.lower()
    if "<tbody" in lower and "</tbody>" not in lower[lower.rfind("<tbody") :]:
        body += "\n  </tbody>"
        lower = body.lower()
    if "<table" in lower and "</table>" not in lower[lower.rfind("<table") :]:
        body += "\n</table>"
    return body


def _close_dangling_row(text: str) -> str:
    lower = text.lower()
    last_tr = lower.rfind("<tr")
    if last_tr < 0:
        return text
    if "</tr>" in lower[last_tr:]:
        return text
    # Close any open cell first.
    last_th = max(lower.rfind("<th"), lower.rfind("<td"))
    if last_th > last_tr:
        tag = "th" if lower.rfind("<th") >= lower.rfind("<td") else "td"
        if f"</{tag}>" not in lower[last_th:]:
            return text + f"</{tag}></tr>"
    return text + "</tr>"


def convert_tsv_blocks_to_html(text: str) -> str:
    """Turn runs of tab-separated rows into HTML tables for live preview."""
    lines = text.splitlines()
    out: list[str] = []
    index = 0
    while index < len(lines):
        if "\t" not in lines[index] or lines[index].count("\t") < 2:
            out.append(lines[index])
            index += 1
            continue
        block: list[str] = []
        while index < len(lines) and lines[index].count("\t") >= 2:
            block.append(lines[index])
            index += 1
        rows = [[cell.strip() for cell in line.split("\t")] for line in block]
        rows = [row for row in rows if any(cell for cell in row)]
        if len(rows) >= 2 and max(len(row) for row in rows) >= 3:
            width = max(len(row) for row in rows)
            padded = [(row + [""] * width)[:width] for row in rows]
            out.append(_rows_to_html_table(padded))
        else:
            out.extend(block)
    return "\n".join(out)
