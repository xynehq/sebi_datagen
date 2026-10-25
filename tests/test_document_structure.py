"""Document structure regression (geometry + finalizer, no UI)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ocr_service"))

from document_geometry import rebuild_page_from_geometry  # noqa: E402
from finalize import finalize_document_for_response, finalize_document_markdown  # noqa: E402
from html_tables import convert_markdown_tables_to_html  # noqa: E402
from layout_repair import repair_column_major_tables  # noqa: E402
from page_layout import apply_page_layout  # noqa: E402
from reading import clean_reading, finalize_reading  # noqa: E402

_FIXTURE = Path(__file__).resolve().parents[1] / "ocr_service/examples/gujarati_holidays_2015.jpg"
_COLUMN_DUMP = "\n".join(
    [
        "ક્રમાંક",
        "૧.",
        "૨.",
        "૩.",
        "મકર",
        "પ્રજા",
        "મહા",
        "૧૪ જાન્યુઆરી",
        "૨૬ જાન્યુઆરી",
        "બુધવાર",
        "સોમવાર",
    ]
)


def _row_major_table(text: str) -> bool:
    if "<table" in text.lower():
        return "<tr>" in text.lower() and text.lower().count("<tr>") >= 4
    lines = [line for line in text.splitlines() if line.strip().startswith("|")]
    data = [line for line in lines if "---" not in line]
    joined = "\n".join(data)
    return bool(re.search(r"\|[^|]+\|[^|]+\|[^|]+\|", joined))


@pytest.mark.skipif(not _FIXTURE.is_file(), reason="holiday fixture missing")
@pytest.mark.skipif(
    __import__("shutil").which("tesseract") is None,
    reason="tesseract not installed",
)
def test_gujarati_holiday_geometry_row_major() -> None:
    image = Image.open(_FIXTURE).convert("RGB")
    out, report = rebuild_page_from_geometry(
        image,
        _COLUMN_DUMP,
        model_id="chitrapathak-marguj-mixed",
    )
    assert report.geometry_table_rows >= 8
    assert _row_major_table(out)
    assert "બુધવાર" in out or "સોમવાર" in out
    assert out.index("બુધવાર" if "બુધવાર" in out else "સોમવાર") < len(out)


def test_two_column_prose_not_forced_to_table() -> None:
    prose = "## Section\n\nLeft column line one.\n\nRight column line two."
    fixed = repair_column_major_tables(prose)
    assert fixed == prose
    assert not _row_major_table(fixed)


def test_numbered_list_stays_list() -> None:
    text = "૧. પહેલું\n૨. બીજું\n૩. ત્રીજું"
    fixed = repair_column_major_tables(text)
    assert "|" not in fixed or fixed.count("|") < 4


def test_key_value_not_auto_table() -> None:
    text = "નામ: સુરેશ\nઉમર: ૪૨"
    assert not _row_major_table(finalize_reading(text))


def test_already_correct_markdown_unchanged() -> None:
    table = "| A | B |\n|---|---|\n| 1 | 2 |"
    assert finalize_reading(table) == table


def test_markdown_table_to_html() -> None:
    html = convert_markdown_tables_to_html("| H |\n|---|\n| 1 |")
    assert "<table>" in html and "<td>1</td>" in html


@pytest.mark.skipif(not _FIXTURE.is_file(), reason="fixture missing")
@pytest.mark.skipif(__import__("shutil").which("tesseract") is None, reason="no tess")
def test_finalize_matches_api_helper() -> None:
    image = Image.open(_FIXTURE).convert("RGB")
    a = finalize_document_markdown(image, _COLUMN_DUMP, model_id="chitrapathak-marguj-mixed")
    b = finalize_document_for_response(image, _COLUMN_DUMP, model_id="chitrapathak-marguj-mixed")
    assert _row_major_table(a)
    assert _row_major_table(b) or "<table" in b.lower()


def test_clean_reading_preserves_stamp_loop_trim() -> None:
    loop = "સીમા\n" + "\n".join(["સીમા"] * 10)
    trimmed = clean_reading(loop)
    assert trimmed.count("સીમા") <= 3
