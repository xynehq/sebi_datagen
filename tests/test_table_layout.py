"""Geometry-based table layout for borderless Gujarati tables."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ocr_service"))

from table_layout import apply_table_layout, tesseract_table_markdown  # noqa: E402

_FIXTURE = Path(__file__).resolve().parents[1] / "ocr_service/examples/gujarati_holidays_2015.jpg"


@pytest.mark.skipif(not shutil.which("tesseract"), reason="tesseract not installed")
def test_gujarati_holiday_table_rows_align() -> None:
    assert _FIXTURE.is_file(), "missing ocr_service/examples/gujarati_holidays_2015.jpg"
    image = Image.open(_FIXTURE).convert("RGB")
    garbled = "\n".join(["१.", "२.", "३.", "भउरसंक्रांति", "सोमवार"])
    result = apply_table_layout(image, garbled, model_id="chitrapathak-marguj")
    assert "<table>" in result.lower() or "| ક્રમાંક |" in result
    joined = result
    assert "પ્રજાસત્તાક" in joined or "પ્રજાસત્તાક દિન" in joined
    assert "મકરસંક્રાંતિ" in joined
    assert "Column 1" not in result


@pytest.mark.skipif(not shutil.which("tesseract"), reason="tesseract not installed")
def test_tesseract_table_markdown_on_fixture() -> None:
    image = Image.open(_FIXTURE).convert("RGB")
    table = tesseract_table_markdown(image, lang="guj+eng")
    assert table is not None
    assert "<table>" in table.lower() or table.count("\n| ") >= 20
