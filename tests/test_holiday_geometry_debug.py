"""Regression: production screenshot must get geometry rebuild (not column-major)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ocr_service"))

from document_geometry import rebuild_page_from_geometry  # noqa: E402
from finalize import finalize_document_markdown  # noqa: E402
from table_layout import inspect_tesseract_table_geometry, tesseract_lang_for_model  # noqa: E402

_FIXTURE = Path(__file__).resolve().parents[1] / "ocr_service/examples/gujarati_holidays_2015.jpg"
_SCREENSHOT = (
    Path(__file__).resolve().parents[1]
    / "data/ocr_captures/pages/2d14fdee43b0409a93c85c9a3ba354a9.png"
)
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
_SERIAL = re.compile(r"^[૦-૯0-9]{1,3}[.)]?\s*$")
_WEEKDAY = re.compile(r"(સોમવાર|મંગળવાર|બુધવાર|ગુરૂવાર|શુક્રવાર|શનિવાર|રવિવાર)")


def _html_rows(text: str) -> list[tuple[str, str, str, str]]:
    import re as _re

    rows: list[tuple[str, str, str, str]] = []
    for tr in _re.findall(r"<tr\b[^>]*>(.*?)</tr>", text, flags=_re.I | _re.S):
        cells = _re.findall(r"<t[hd]\b[^>]*>(.*?)</t[hd]>", tr, flags=_re.I | _re.S)
        if len(cells) < 4:
            continue
        cleaned = tuple(_re.sub(r"<[^>]+>", "", c).strip() for c in cells[:4])
        if cleaned[0].lower().startswith("ક્રમાંક"):
            continue
        rows.append(cleaned)
    return rows


def _assert_row_major_associations(text: str, *, min_body_rows: int = 20) -> None:
    body = _html_rows(text)
    assert len(body) >= min_body_rows, f"expected >={min_body_rows} body rows, got {len(body)}"
    serial_rows = [row for row in body if _SERIAL.match(row[0])]
    assert len(serial_rows) >= 12, f"too few serial-led rows: {len(serial_rows)}"
    matched = 0
    for serial, name, _date, weekday in serial_rows:
        if name.strip() and (weekday.strip() or _date.strip()):
            matched += 1
    assert matched >= 12, "serial/name/date/weekday must share the same <tr>"
    assert "પ્રજાસત્તાક દિન" in text and "સોમવાર" in text
    idx_name = text.find("પ્રજાસત્તાક દિન")
    idx_day = text.find("સોમવાર", idx_name)
    assert idx_day > idx_name and idx_day - idx_name < 400


@pytest.mark.skipif(not _FIXTURE.is_file(), reason="holiday fixture missing")
@pytest.mark.skipif(__import__("shutil").which("tesseract") is None, reason="no tess")
def test_geometry_on_fixture_jpg() -> None:
    image = Image.open(_FIXTURE).convert("RGB")
    geo = inspect_tesseract_table_geometry(image, lang=tesseract_lang_for_model("chitrapathak-marguj-mixed"))
    assert geo.get("grid_detected") is True
    assert int(geo.get("data_row_count") or 0) >= 8
    out, report = rebuild_page_from_geometry(image, _COLUMN_DUMP, model_id="chitrapathak-marguj-mixed")
    assert report.geometry_table_rows >= 8
    assert not report.fallback_used
    _assert_row_major_associations(out, min_body_rows=20)


@pytest.mark.skipif(not _SCREENSHOT.is_file(), reason="capture screenshot missing")
@pytest.mark.skipif(__import__("shutil").which("tesseract") is None, reason="no tess")
def test_geometry_on_user_screenshot_png() -> None:
    """Fails before column-boundary fix: tess grid_detected false → column-major final."""
    image = Image.open(_SCREENSHOT).convert("RGB")
    geo = inspect_tesseract_table_geometry(image, lang=tesseract_lang_for_model("chitrapathak-marguj-mixed"))
    assert geo.get("grid_detected") is True, geo.get("failure_stage")
    assert int(geo.get("data_row_count") or 0) >= 20
    final = finalize_document_markdown(image, _COLUMN_DUMP, model_id="chitrapathak-marguj-mixed")
    _assert_row_major_associations(final, min_body_rows=20)
