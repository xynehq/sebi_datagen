"""Row/column reconstruction: synthetic geometry plus the holiday notice."""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ocr_service"))

from page_layout import apply_page_layout  # noqa: E402
from table_layout import (  # noqa: E402
    _Word,
    reconstruct_table_from_words,
    tesseract_table_markdown,
)

_FIXTURE = Path(__file__).resolve().parents[1] / "ocr_service/examples/gujarati_holidays_2015.jpg"
_SERIAL_PAIR = re.compile(r"[૦-૯0-9०-९]{1,3}[.)].*[૦-૯0-9०-९]{1,3}[.)]")


def _w(
    text: str,
    left: int,
    top: int,
    *,
    width: int = 40,
    height: int = 18,
    line: int = 1,
) -> _Word:
    return _Word(left, top, width, text, (0, 0, line, 1), height, 90.0)


def _html_body(text: str) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    for tr in re.findall(r"<tr\b[^>]*>(.*?)</tr>", text, flags=re.I | re.S):
        cells = re.findall(r"<t[hd]\b[^>]*>(.*?)</t[hd]>", tr, flags=re.I | re.S)
        cleaned = tuple(re.sub(r"<[^>]+>", "", cell).strip() for cell in cells)
        if cleaned and cleaned[0] in {"ક્રમાંક", "क्रमांक"}:
            continue
        if cleaned:
            rows.append(cleaned)
    return rows


def test_two_line_header_stays_in_header() -> None:
    words = [
        _w("ક્રમાંક", 20, 100, line=1),
        _w("જાહેર", 200, 100, line=1),
        _w("તારીખ", 600, 100, line=1),
        _w("અઠવાડિયાનો", 900, 100, line=1),
        _w("વાર", 920, 122, line=2),
        _w("૧.", 20, 160, line=3),
        _w("મકરસંક્રાંતિ", 200, 160, line=3),
        _w("૧૪ જાન્યુઆરી", 600, 160, line=3),
        _w("બુધવાર", 920, 160, line=3),
        _w("૨.", 20, 200, line=4),
        _w("પ્રજાસત્તાક", 200, 200, line=4),
        _w("૨૬ જાન્યુઆરી", 600, 200, line=4),
        _w("સોમવાર", 920, 200, line=4),
        _w("૩.", 20, 240, line=5),
        _w("મહાશિવરાત્રી", 200, 240, line=5),
        _w("૧૭ ફેબ્રુઆરી", 600, 240, line=5),
        _w("મંગળવાર", 920, 240, line=5),
    ]
    built = reconstruct_table_from_words(words, 1200)
    assert built is not None
    serials = [row[0] for row in built.rows if re.match(r"^[૦-૯]{1,3}\.?$", row[0].strip())]
    assert serials[:3] == ["૧.", "૨.", "૩."]
    assert all("વાર" not in row[1] for row in built.rows)
    assert all(not _SERIAL_PAIR.search(row[0]) for row in built.rows)


def test_wrapped_name_stays_in_same_row() -> None:
    words = [
        _w("ક્રમાંક", 20, 80, line=1),
        _w("નામ", 200, 80, line=1),
        _w("તારીખ", 600, 80, line=1),
        _w("વાર", 900, 80, line=1),
        _w("૧.", 20, 140, line=2),
        _w("જન્માષ્ટમી", 200, 140, line=2),
        _w("૫ સપ્ટેમ્બર", 600, 140, line=2),
        _w("શનિવાર", 920, 140, line=2),
        _w("જન્મોત્સવ", 210, 158, line=3),
        _w("૨.", 20, 200, line=4),
        _w("દસેરો", 200, 200, line=4),
        _w("૨૨ સપ્ટેમ્બર", 600, 200, line=4),
        _w("મંગળવાર", 920, 200, line=4),
        _w("૩.", 20, 240, line=5),
        _w("દિવાળી", 200, 240, line=5),
        _w("૧૧ નવેમ્બર", 600, 240, line=5),
        _w("બુધવાર", 920, 240, line=5),
    ]
    built = reconstruct_table_from_words(words, 1200)
    assert built is not None
    first = next(row for row in built.rows if row[0].startswith("૧"))
    assert "જન્માષ્ટમી" in first[1] and "જન્મોત્સવ" in first[1]
    second = next(row for row in built.rows if row[0].startswith("૨"))
    assert "દસેરો" in second[1]
    assert "જન્મોત્સવ" not in second[1]


def test_tight_rows_do_not_merge() -> None:
    words = [
        _w("ક્રમાંક", 20, 40, line=1),
        _w("નામ", 200, 40, line=1),
        _w("તારીખ", 600, 40, line=1),
        _w("વાર", 900, 40, line=1),
    ]
    for index, serial in enumerate(["૧.", "૨.", "૩.", "૪."], start=1):
        top = 80 + index * 22
        words.extend(
            [
                _w(serial, 20, top, line=index + 1),
                _w(f"નામ{index}", 200, top, line=index + 1),
                _w(f"{index} જાન્યુઆરી", 600, top, line=index + 1),
                _w("સોમવાર", 920, top, line=index + 1),
            ]
        )
    built = reconstruct_table_from_words(words, 1200)
    assert built is not None
    serials = [row[0] for row in built.rows]
    assert serials[:4] == ["૧.", "૨.", "૩.", "૪."]
    assert all(not _SERIAL_PAIR.search(cell) for cell in serials)


def test_marathi_borderless_table() -> None:
    words = [
        _w("क्रमांक", 20, 50, line=1),
        _w("जाहेर", 180, 50, line=1),
        _w("तारीख", 560, 50, line=1),
        _w("वार", 900, 50, line=1),
        _w("१.", 20, 100, line=2),
        _w("मकरसंक्रांति", 180, 100, line=2),
        _w("१४ जानेवारी", 560, 100, line=2),
        _w("बुधवार", 900, 100, line=2),
        _w("२.", 20, 140, line=3),
        _w("प्रजासत्ताक दिन", 180, 140, line=3),
        _w("२६ जानेवारी", 560, 140, line=3),
        _w("सोमवार", 900, 140, line=3),
        _w("३.", 20, 180, line=4),
        _w("होळी", 180, 180, line=4),
        _w("६ मार्च", 560, 180, line=4),
        _w("शुक्रवार", 900, 180, line=4),
    ]
    built = reconstruct_table_from_words(words, 1100)
    assert built is not None
    assert any("मकरसंक्रांति" in row[1] and "बुधवार" in row[3] for row in built.rows)
    assert any("प्रजासत्ताक" in row[1] and "सोमवार" in row[3] for row in built.rows)


def test_title_table_footer_regions() -> None:
    words = [
        _w("विभाग", 40, 10, line=1),
        _w("सूचना", 40, 40, line=2),
        _w("ક્રમાંક", 20, 90, line=3),
        _w("નામ", 200, 90, line=3),
        _w("તારીખ", 600, 90, line=3),
        _w("વાર", 900, 90, line=3),
        _w("૧.", 20, 140, line=4),
        _w("હોલીડે", 200, 140, line=4),
        _w("૧ માર્ચ", 600, 140, line=4),
        _w("રવિવાર", 920, 140, line=4),
        _w("૨.", 20, 180, line=5),
        _w("બીજી", 200, 180, line=5),
        _w("૨ માર્ચ", 600, 180, line=5),
        _w("સોમવાર", 920, 180, line=5),
        _w("૩.", 20, 220, line=6),
        _w("ત્રીજી", 200, 220, line=6),
        _w("૩ માર્ચ", 600, 220, line=6),
        _w("મંગળવાર", 920, 220, line=6),
        _w("સહી", 40, 400, line=7),
        _w("ગાંધીનગર", 120, 400, line=7),
    ]
    built = reconstruct_table_from_words(words, 1200)
    assert built is not None
    assert "સહી" in built.footer and "ગાંધીનગર" in built.footer
    assert all("સહી" not in row[1] for row in built.rows)


def test_prose_page_not_forced_to_table() -> None:
    image = Image.new("RGB", (400, 400), "white")
    prose = "## Notice\n\nThis page has no grid.\n\nA second paragraph."
    assert apply_page_layout(image, prose, model_id="chitrapathak-2") == prose


def test_orphan_header_is_not_merged_into_next_serial() -> None:
    words = [
        _w("ક્રમાંક", 20, 80, line=1),
        _w("જાહેર", 200, 80, line=1),
        _w("તારીખ", 600, 80, line=1),
        _w("વાર", 900, 80, line=1),
        _w("પ.", 200, 130, line=2),
        _w("૬.", 20, 150, line=3),
        _w("ગુડ ફ્રાઈડે", 200, 150, line=3),
        _w("૩ એપ્રિલ", 600, 150, line=3),
        _w("શુક્રવાર", 920, 150, line=3),
        _w("૭.", 20, 190, line=4),
        _w("બીજી રજા", 200, 190, line=4),
        _w("૪ એપ્રિલ", 600, 190, line=4),
        _w("શનિવાર", 920, 190, line=4),
        _w("૮.", 20, 230, line=5),
        _w("ત્રીજી રજા", 200, 230, line=5),
        _w("૫ એપ્રિલ", 600, 230, line=5),
        _w("રવિવાર", 920, 230, line=5),
    ]
    built = reconstruct_table_from_words(words, 1200)
    assert built is not None
    sixth = next(row for row in built.rows if row[0].startswith("૬"))
    assert "ગુડ ફ્રાઈડે" in sixth[1]
    assert "બીજી" not in sixth[1]
    assert not _SERIAL_PAIR.search(sixth[0])


@pytest.mark.skipif(not _FIXTURE.is_file(), reason="holiday fixture missing")
@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
def test_holiday_notice_html_rows_do_not_merge() -> None:
    image = Image.open(_FIXTURE).convert("RGB")
    table = tesseract_table_markdown(image, lang="guj+eng")
    assert table is not None
    assert "<table>" in table.lower()
    assert "Column 1" not in table
    body = _html_body(table)
    assert len(body) >= 20
    serials = [row[0] for row in body]
    assert all(not _SERIAL_PAIR.search(cell) for cell in serials)
    joined_names = " ".join(row[1] for row in body if len(row) > 1)
    assert "ક્રમાંક" not in joined_names
    assert "જાહેર રજાનું નામ" not in joined_names
    republic = [row for row in body if "પ્રજાસત્તાક" in " ".join(row)]
    assert republic, body[:5]
    assert any("સોમવાર" in row[3] for row in republic)
    makar = [row for row in body if "મકર" in " ".join(row)]
    if makar:
        assert any("બુધવાર" in row[3] or "બુધ" in row[3] for row in makar)
    weekdays = [row[3] for row in body if len(row) > 3 and row[3]]
    assert len(weekdays) >= 12
