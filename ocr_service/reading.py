"""Shape a page reading into the markdown the downstream stack reads.

Chitrapathak is prompted for this shape so a later fine-tune can be served
without changing the caller: headings, markdown tables with one row per item,
and clause numbers on the same line as the clause.
"""

from __future__ import annotations

import re

CHITRA_PROMPT = (
    "Convert this document image to clean markdown. "
    "Use # or ## for real titles. "
    "Keep each table as a markdown table with one row per item, so cells in a row stay together. "
    "Keep a clause number on the same line as its clause. "
    "Transcribe the document's own digits and spelling exactly. "
    "Do not repeat lines. Stop at the end of the page."
)

_LIST_BREAK = re.compile(r"(?<=\S)[ \t]+(?=[०-९0-9]{1,3}\.\s)")
_FENCE_OPEN = re.compile(r"^```(?:markdown|md|html)?\s*\n?", re.IGNORECASE)
_FENCE_CLOSE = re.compile(r"\n?```\s*$")
_TOKEN_RUN = re.compile(r"(\S+)(?:[ \t]+\1){7,}")
_STRUCTURED = re.compile(
    r"(?m)^#{1,6}\s+\S|^\|.+\|\s*$|<table\b|!\[[^\]]*\]\([^)]*\)",
    re.IGNORECASE,
)
_DIGITS = re.compile(r"[0-9०-९૦-૯]+")
_DIGIT_MAP = str.maketrans("०१२३४५६७८९૦૧૨૩૪૫૬૭૮૯", "01234567890123456789")


def clean_reading(text: str, *echoes: str) -> str:
    """Drop fences, an echoed prompt, and runaway repeats."""
    body = text.strip()
    if body.lower().startswith("assistant"):
        body = body[len("assistant") :].lstrip(" :\n")
    body = _FENCE_OPEN.sub("", body, count=1)
    body = _FENCE_CLOSE.sub("", body).strip()
    for echo in echoes:
        if echo and body.startswith(echo):
            body = body[len(echo) :].lstrip(" :\n")
    body = _cut_token_runs(body)
    body = _cut_repeated_lines(body)
    body = _cut_block_loops(body)
    body = _cut_counting_loops(body)
    return body.strip()


def reading_raw(text: str) -> str:
    """Text to store and export. Structured markdown is kept intact."""
    body = text.strip()
    if _STRUCTURED.search(body):
        return body
    return _plain_lines(body)


def reading_markdown(text: str) -> str:
    """Text for the rendered pane. Plain lines stay separated."""
    body = text.strip()
    if _STRUCTURED.search(body):
        return body
    lines = _plain_lines(body)
    if not lines:
        return ""
    return "\n\n".join(lines.split("\n"))


def _plain_lines(text: str) -> str:
    text = _LIST_BREAK.sub("\n", text.replace("\r\n", "\n"))
    return "\n".join(line.strip() for line in text.split("\n") if line.strip())


def _cut_token_runs(text: str) -> str:
    return _TOKEN_RUN.sub(lambda match: f"{match.group(1)} {match.group(1)}", text)


def _cut_repeated_lines(text: str) -> str:
    lines = text.split("\n")
    kept: list[str] = []
    index = 0
    while index < len(lines):
        key = lines[index].strip()
        if not key:
            kept.append(lines[index])
            index += 1
            continue
        end = index + 1
        while end < len(lines) and lines[end].strip() == key:
            end += 1
        run = end - index
        keep = 2 if run >= 4 else run
        kept.extend(lines[index : index + keep])
        index = end
    return "\n".join(kept)


def _cut_block_loops(text: str, max_block: int = 12) -> str:
    lines = text.split("\n")
    count = len(lines)
    for start in range(count):
        for width in range(1, max_block + 1):
            if start + (3 * width) > count:
                break
            block = lines[start : start + width]
            if not any(part.strip() for part in block):
                continue
            second = lines[start + width : start + (2 * width)]
            third = lines[start + (2 * width) : start + (3 * width)]
            if second != block or third != block:
                continue
            stop = start + width
            while stop + width <= count and lines[stop : stop + width] == block:
                stop += width
            tail = lines[stop:]
            if not any(part.strip() for part in tail):
                return "\n".join(lines[: start + width] + tail).strip()
    return text


def _cut_counting_loops(text: str) -> str:
    """Drop a long run of near-copies whose counter jumps by 5 or more."""
    lines = text.split("\n")
    kept: list[str] = []
    index = 0
    while index < len(lines):
        skeleton = _skeleton(lines[index])
        if (
            not lines[index].strip()
            or "|" in lines[index]
            or lines[index].lstrip().startswith("#")
            or len(skeleton) < 4
        ):
            kept.append(lines[index])
            index += 1
            continue
        end = index + 1
        while end < len(lines) and _same_counter_line(lines[end], lines[index], skeleton):
            end += 1
        run = lines[index:end]
        if len(run) >= 8 and all(len(line) <= 120 for line in run) and _coarse_counter(run):
            kept.extend(run[:2])
            index = end
            continue
        kept.append(lines[index])
        index += 1
    return "\n".join(kept)


def _same_counter_line(line: str, first: str, skeleton: str) -> bool:
    return _skeleton(line) == skeleton and line.strip() != first.strip()


def _skeleton(line: str) -> str:
    return _DIGITS.sub("#", line.strip())


def _coarse_counter(lines: list[str]) -> bool:
    groups = [_DIGITS.findall(line) for line in lines]
    if any(not group for group in groups):
        return False
    width = min(len(group) for group in groups)
    for slot in range(width):
        values: list[int] = []
        for group in groups:
            number = _to_int(group[slot])
            if number is None:
                return False
            values.append(number)
        if len(set(values)) < 2:
            continue
        steps = {right - left for left, right in zip(values, values[1:], strict=False)}
        if len(steps) == 1 and abs(next(iter(steps))) >= 5:
            return True
    return False


def _to_int(token: str) -> int | None:
    try:
        return int(token.translate(_DIGIT_MAP))
    except ValueError:
        return None
